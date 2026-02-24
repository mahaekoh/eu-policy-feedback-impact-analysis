"""Merge OCR results back into initiative detail JSON files.

Takes the OCR report produced by ocr_short_pdfs.py and updates the
corresponding initiative JSON files in-place, replacing extracted_text
with the OCR result and preserving the original as extracted_text_without_ocr.

Usage:
    # Dry run — print changes without modifying files
    python3 src/merge_ocr_results.py short_pdf_report_ocr.json initiative_details/ --dry-run

    # Apply changes
    python3 src/merge_ocr_results.py short_pdf_report_ocr.json initiative_details/
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge OCR results into initiative detail JSON files."
    )
    parser.add_argument(
        "report", help="Path to OCR report JSON (output of ocr_short_pdfs.py)"
    )
    parser.add_argument(
        "details_dir", help="Directory of per-initiative JSON files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print changes without modifying files.",
    )
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as f:
        records = json.load(f)

    # Group records by initiative ID
    by_initiative = {}
    for rec in records:
        ocr_text = rec.get("ocr_text")
        if not ocr_text:
            continue
        init_id = rec["initiative_id"]
        by_initiative.setdefault(init_id, []).append(rec)

    print(f"Loaded {len(records)} records, {sum(len(v) for v in by_initiative.values())} with OCR text across {len(by_initiative)} initiatives")

    updated = 0
    skipped = 0
    modified_files = set()

    for init_id, recs in sorted(by_initiative.items()):
        json_path = os.path.join(args.details_dir, f"{init_id}.json")
        if not os.path.isfile(json_path):
            print(f"  SKIP initiative {init_id}: file not found", file=sys.stderr)
            skipped += len(recs)
            continue

        with open(json_path, encoding="utf-8") as f:
            initiative = json.load(f)

        # Index publications and feedback for fast lookup
        pubs_by_id = {}
        for pub in initiative.get("publications", []):
            pubs_by_id[pub.get("publication_id")] = pub

        changed = False
        for rec in recs:
            pub_id = rec["publication_id"]
            pub = pubs_by_id.get(pub_id)
            if not pub:
                print(f"  SKIP initiative {init_id}: pub {pub_id} not found", file=sys.stderr)
                skipped += 1
                continue

            target = None
            if rec["type"] == "doc":
                for doc in pub.get("documents", []):
                    if doc.get("download_url") == rec["download_url"]:
                        target = doc
                        break
            elif rec["type"] == "fb_att":
                fb_id = rec["feedback_id"]
                att_id = rec["attachment_id"]
                for fb in pub.get("feedback", []):
                    if fb.get("id") != fb_id:
                        continue
                    for att in fb.get("attachments", []):
                        if att.get("id") == att_id:
                            target = att
                            break
                    if target:
                        break

            if not target:
                print(f"  SKIP initiative {init_id}: could not locate attachment for {rec.get('pdf_file', '?')}", file=sys.stderr)
                skipped += 1
                continue

            old_text = target.get("extracted_text", "")
            ocr_text = rec["ocr_text"]

            if args.dry_run:
                old_snippet = old_text.replace("\n", " ").strip()[:120] if old_text else "(empty)"
                ocr_snippet = ocr_text.replace("\n", " ").strip()[:120] if ocr_text else "(empty)"
                print(f"\n[{rec.get('pdf_file', '?')}]")
                print(f"  initiative: {init_id}, pub: {pub_id}, type: {rec['type']}")
                if rec["type"] == "fb_att":
                    print(f"  feedback: {rec['feedback_id']}, attachment: {rec['attachment_id']}")
                print(f"  filename: {rec['filename']}")
                print(f"  extracted_text ({len(old_text)} chars): {old_snippet}")
                print(f"  ocr_text       ({len(ocr_text)} chars): {ocr_snippet}")
            else:
                target["extracted_text_without_ocr"] = old_text
                target["extracted_text"] = ocr_text
                changed = True

            updated += 1

        if changed and not args.dry_run:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(initiative, f, ensure_ascii=False, indent=2)
            modified_files.add(json_path)

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
    print(f"\nUpdated: {updated}, Skipped: {skipped}, Files modified: {len(modified_files)}")


if __name__ == "__main__":
    main()
