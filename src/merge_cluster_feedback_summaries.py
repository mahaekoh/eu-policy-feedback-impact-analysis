"""Merge cluster feedback summaries back into initiative detail JSON files.

Takes the cluster summary output from summarize_clusters.py and updates the
corresponding initiative JSON files in-place, setting cluster_feedback_summary
on each feedback item that has a summary.

The cluster_feedback_summary field survives re-scrapes because the scrape merge
strategy preserves all derived fields on feedback items when feedback_text is
unchanged.

Usage:
    # Dry run — print changes without modifying files
    python3 src/merge_cluster_feedback_summaries.py \\
        data/cluster_summaries/<scheme>/ data/scrape/initiative_details/ --dry-run

    # Apply changes
    python3 src/merge_cluster_feedback_summaries.py \\
        data/cluster_summaries/<scheme>/ data/scrape/initiative_details/
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Merge cluster feedback summaries into initiative detail JSON files."
    )
    parser.add_argument(
        "summary_dir",
        help="Directory of cluster summary JSON files (output of summarize_clusters.py)",
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

    total_merged = 0
    total_skipped_lookup = 0
    total_skipped_no_summary = 0
    modified_files = set()

    for summary_file in summary_files:
        summary_path = os.path.join(args.summary_dir, summary_file)
        with open(summary_path, encoding="utf-8") as f:
            summary_data = json.load(f)

        feedback_summaries = summary_data.get("feedback_summaries", {})
        if not feedback_summaries:
            continue

        init_id = summary_file.replace(".json", "")
        json_path = os.path.join(args.details_dir, f"{init_id}.json")
        if not os.path.isfile(json_path):
            print(f"  SKIP initiative {init_id}: file not found", file=sys.stderr)
            total_skipped_lookup += len(feedback_summaries)
            continue

        with open(json_path, encoding="utf-8") as f:
            initiative = json.load(f)

        # Build feedback lookup: str(fb_id) -> fb dict
        fb_lookup = {}
        for pub in initiative.get("publications", []):
            for fb in pub.get("feedback", []):
                fb_lookup[str(fb["id"])] = fb

        changed = False
        for feedback_id, fs in feedback_summaries.items():
            title = fs.get("title", "")
            summary = fs.get("summary", "")
            if not summary:
                total_skipped_no_summary += 1
                continue

            fb = fb_lookup.get(str(feedback_id))
            if fb is None:
                total_skipped_lookup += 1
                continue

            value = {"title": title, "summary": summary}

            if args.dry_run:
                existing = fb.get("cluster_feedback_summary")
                status = "UPDATE" if existing else "NEW"
                title_snippet = title[:100] if title else "(empty)"
                summary_snippet = summary.replace("\n", " ")[:120] if summary else "(empty)"
                print(f"\n[{status}] initiative {init_id}, fb {feedback_id}")
                print(f"  title:   {title_snippet}")
                print(f"  summary: {summary_snippet}")
            else:
                fb["cluster_feedback_summary"] = value
                changed = True

            total_merged += 1

        if changed and not args.dry_run:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(initiative, f, ensure_ascii=False, indent=2)
            modified_files.add(json_path)

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
    print(f"\nMerged: {total_merged}, Skipped (lookup): {total_skipped_lookup}, "
          f"Skipped (no summary): {total_skipped_no_summary}, "
          f"Files modified: {len(modified_files)}")


if __name__ == "__main__":
    main()
