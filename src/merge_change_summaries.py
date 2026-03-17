"""Merge change summaries back into initiative detail JSON files.

Takes the change summary output from summarize_changes.py and updates the
corresponding initiative JSON files in-place, adding change_summary and diff
fields at the top level while preserving all existing attributes.

Usage:
    # Dry run — print changes without modifying files
    python3 src/merge_change_summaries.py data/analysis/change_summaries/ \\
        data/scrape/initiative_details/ --dry-run

    # Apply changes
    python3 src/merge_change_summaries.py data/analysis/change_summaries/ \\
        data/scrape/initiative_details/
"""

import argparse
import json
import os
import sys

# Fields to merge from change summary output into initiative_details
MERGE_FIELDS = ("change_summary", "diff")


def main():
    parser = argparse.ArgumentParser(
        description="Merge change summaries into initiative detail JSON files."
    )
    parser.add_argument(
        "summary_dir",
        help="Directory of change summary JSON files (output of summarize_changes.py)",
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
    total_skipped_no_summary = 0
    total_skipped_not_found = 0
    modified_files = set()
    n_files = len(summary_files)

    print(f"Merging change summaries into {args.details_dir}...")
    for i, summary_file in enumerate(summary_files):
        if i % 500 == 0:
            print(f"  Processing {i}/{n_files}...")
        init_id = summary_file.replace(".json", "")
        summary_path = os.path.join(args.summary_dir, summary_file)
        with open(summary_path, encoding="utf-8") as f:
            summary_data = json.load(f)

        change_summary = summary_data.get("change_summary")
        if not change_summary:
            total_skipped_no_summary += 1
            continue

        json_path = os.path.join(args.details_dir, f"{init_id}.json")
        if not os.path.isfile(json_path):
            print(f"  SKIP initiative {init_id}: file not found", file=sys.stderr)
            total_skipped_not_found += 1
            continue

        with open(json_path, encoding="utf-8") as f:
            initiative = json.load(f)

        if args.dry_run:
            existing = initiative.get("change_summary")
            status = "UPDATE" if existing else "NEW"
            snippet = change_summary.replace("\n", " ")[:150]
            print(f"[{status}] initiative {init_id}: {snippet}")
        else:
            for field in MERGE_FIELDS:
                value = summary_data.get(field)
                if value is not None:
                    initiative[field] = value

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(initiative, f, ensure_ascii=False, indent=2)
            modified_files.add(json_path)

        total_merged += 1

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
    print(f"\nMerged: {total_merged}, Skipped (no summary): {total_skipped_no_summary}, "
          f"Skipped (not found): {total_skipped_not_found}, "
          f"Files modified: {len(modified_files)}")


if __name__ == "__main__":
    main()
