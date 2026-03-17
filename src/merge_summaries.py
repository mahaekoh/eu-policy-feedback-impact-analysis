"""Merge document and attachment summaries back into initiative detail JSON files.

Takes the summary output from summarize_documents.py and updates the
corresponding initiative JSON files in-place, adding summary fields on
documents (matched by doc_id, falling back to download_url for docs
where doc_id is None) and feedback attachments (matched by feedback id
+ attachment id).

Usage:
    # Dry run — print changes without modifying files
    python3 src/merge_summaries.py data/analysis/summaries/ \
        data/scrape/initiative_details/ --dry-run

    # Apply changes
    python3 src/merge_summaries.py data/analysis/summaries/ \
        data/scrape/initiative_details/
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Merge document/attachment summaries into initiative detail JSON files."
    )
    parser.add_argument(
        "summary_dir",
        help="Directory of summary JSON files (output of summarize_documents.py)",
    )
    parser.add_argument(
        "details_dir",
        help="Directory of per-initiative JSON files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print proposed changes without modifying files.",
    )
    args = parser.parse_args()

    # Discover summary files
    summary_files = sorted(
        f for f in os.listdir(args.summary_dir)
        if f.endswith(".json") and not f.startswith("_")
    )
    print(f"Found {len(summary_files)} summary files in {args.summary_dir}")

    total_doc_merged = 0
    total_att_merged = 0
    total_doc_skipped_no_summary = 0
    total_att_skipped_no_summary = 0
    total_skipped_not_found = 0
    modified_files = set()

    for summary_file in summary_files:
        init_id = summary_file.replace(".json", "")
        summary_path = os.path.join(args.summary_dir, summary_file)
        with open(summary_path, encoding="utf-8") as f:
            summary_data = json.load(f)

        json_path = os.path.join(args.details_dir, f"{init_id}.json")
        if not os.path.isfile(json_path):
            print(f"  SKIP initiative {init_id}: file not found", file=sys.stderr)
            total_skipped_not_found += 1
            continue

        # Build lookups: doc_id -> summary, and download_url -> summary (fallback)
        doc_summary_by_id = {}
        doc_summary_by_url = {}
        for doc in (summary_data.get("documents_before_feedback", [])
                    + summary_data.get("documents_after_feedback", [])):
            summary = doc.get("summary")
            doc_id = doc.get("doc_id")
            url = doc.get("download_url")
            if summary:
                if doc_id:
                    doc_summary_by_id[doc_id] = summary
                elif url:
                    doc_summary_by_url[url] = summary
            else:
                total_doc_skipped_no_summary += 1

        # Build lookup: (feedback_id, attachment_id) -> summary from middle_feedback
        att_summary_lookup = {}
        for fb in summary_data.get("middle_feedback", []):
            fb_id = fb.get("id")
            for att in fb.get("attachments", []):
                att_id = att.get("id")
                summary = att.get("summary")
                if fb_id is not None and att_id is not None and summary:
                    att_summary_lookup[(int(fb_id), int(att_id))] = summary
                elif fb_id is not None and att_id is not None and not summary:
                    total_att_skipped_no_summary += 1

        if not doc_summary_by_id and not doc_summary_by_url and not att_summary_lookup:
            continue

        with open(json_path, encoding="utf-8") as f:
            initiative = json.load(f)

        changed = False
        file_doc_merged = 0
        file_att_merged = 0

        # Merge document summaries
        for pub in initiative.get("publications", []):
            for doc in pub.get("documents", []):
                doc_id = doc.get("doc_id")
                url = doc.get("download_url")
                summary = None
                match_key = None
                if doc_id and doc_id in doc_summary_by_id:
                    summary = doc_summary_by_id[doc_id]
                    match_key = f"doc {doc_id}"
                elif url and url in doc_summary_by_url:
                    summary = doc_summary_by_url[url]
                    match_key = f"url ...{url[-40:]}"
                if summary:
                    if args.dry_run:
                        existing = doc.get("summary")
                        status = "UPDATE" if existing else "NEW"
                        snippet = summary.replace("\n", " ")[:120]
                        print(f"[{status}] init {init_id}, {match_key}: {snippet}")
                    else:
                        doc["summary"] = summary
                        changed = True
                    file_doc_merged += 1

            # Merge attachment summaries
            for fb in pub.get("feedback", []):
                fb_id = fb.get("id")
                if fb_id is None:
                    continue
                for att in fb.get("attachments", []):
                    att_id = att.get("id")
                    if att_id is None:
                        continue
                    key = (int(fb_id), int(att_id))
                    if key in att_summary_lookup:
                        if args.dry_run:
                            existing = att.get("summary")
                            status = "UPDATE" if existing else "NEW"
                            snippet = att_summary_lookup[key].replace("\n", " ")[:120]
                            print(f"[{status}] init {init_id}, fb {fb_id}, att {att_id}: {snippet}")
                        else:
                            att["summary"] = att_summary_lookup[key]
                            changed = True
                        file_att_merged += 1

        total_doc_merged += file_doc_merged
        total_att_merged += file_att_merged

        if changed and not args.dry_run:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(initiative, f, ensure_ascii=False, indent=2)
            modified_files.add(json_path)

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
    print(f"\nDoc summaries merged: {total_doc_merged}, "
          f"Attachment summaries merged: {total_att_merged}")
    print(f"Docs skipped (no summary): {total_doc_skipped_no_summary}, "
          f"Attachments skipped (no summary): {total_att_skipped_no_summary}")
    print(f"Skipped (file not found): {total_skipped_not_found}, "
          f"Files modified: {len(modified_files)}")


if __name__ == "__main__":
    main()
