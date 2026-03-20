"""Merge cluster summary rewrites into initiative detail JSON files.

Takes the rewrite output from rewrite_cluster_summaries.py and updates the
corresponding initiative JSON files in-place, setting:
  cluster_summaries[label]["rewrites"][format] = {title, body}

Multiple format merges accumulate additively in the rewrites dict. The rewrites
field survives re-scrapes because cluster_summaries is a top-level derived field
preserved by the scraper merge strategy.

Usage:
    # Dry run — print changes without modifying files
    python3 src/merge_cluster_rewrites.py \\
        data/cluster_rewrites/reddit/<scheme>/ data/scrape/initiative_details/ \\
        --format reddit --dry-run

    # Apply changes
    python3 src/merge_cluster_rewrites.py \\
        data/cluster_rewrites/reddit/<scheme>/ data/scrape/initiative_details/ \\
        --format reddit
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Merge cluster summary rewrites into initiative detail JSON files."
    )
    parser.add_argument(
        "rewrite_dir",
        help="Directory of rewrite JSON files (output of rewrite_cluster_summaries.py)",
    )
    parser.add_argument(
        "details_dir",
        help="Directory of per-initiative JSON files",
    )
    parser.add_argument(
        "--format", required=True,
        help="Rewrite format name (e.g. 'reddit').",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print proposed changes without modifying files.",
    )
    args = parser.parse_args()

    rewrite_files = sorted(
        f for f in os.listdir(args.rewrite_dir)
        if f.endswith(".json") and not f.startswith("_")
    )
    print(f"Found {len(rewrite_files)} rewrite files in {args.rewrite_dir}")

    total_merged = 0
    total_skipped_lookup = 0
    total_skipped_no_cs = 0
    modified_files = set()
    n_files = len(rewrite_files)

    print(f"Merging '{args.format}' rewrites into {args.details_dir}...")
    for i, rewrite_file in enumerate(rewrite_files):
        if i % 500 == 0:
            print(f"  Processing {i}/{n_files}...")

        rewrite_path = os.path.join(args.rewrite_dir, rewrite_file)
        with open(rewrite_path, encoding="utf-8") as f:
            rewrite_data = json.load(f)

        init_id = rewrite_file.replace(".json", "")
        json_path = os.path.join(args.details_dir, f"{init_id}.json")
        if not os.path.isfile(json_path):
            print(f"  SKIP initiative {init_id}: file not found", file=sys.stderr)
            total_skipped_lookup += len(rewrite_data.get("cluster_rewrites", {}))
            continue

        cluster_rewrites = rewrite_data.get("cluster_rewrites", {})
        if not cluster_rewrites:
            continue

        with open(json_path, encoding="utf-8") as f:
            initiative = json.load(f)

        cluster_summaries = initiative.get("cluster_summaries")
        if not cluster_summaries:
            print(f"  SKIP initiative {init_id}: no cluster_summaries", file=sys.stderr)
            total_skipped_no_cs += len(cluster_rewrites)
            continue

        changed = False
        for label, rewrite in cluster_rewrites.items():
            if label not in cluster_summaries:
                total_skipped_lookup += 1
                continue

            entry = cluster_summaries[label]
            if "rewrites" not in entry:
                entry["rewrites"] = {}

            value = {"title": rewrite["title"], "body": rewrite["body"]}

            if args.dry_run:
                existing = entry["rewrites"].get(args.format)
                status = "UPDATE" if existing else "NEW"
                print(f"\n[{status}] initiative {init_id}, label {label}")
                print(f"  title: {value['title'][:100]}")
                print(f"  body:  {value['body'].replace(chr(10), ' ')[:120]}")
            else:
                entry["rewrites"][args.format] = value
                changed = True

            total_merged += 1

        if changed and not args.dry_run:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(initiative, f, ensure_ascii=False, indent=2)
            modified_files.add(json_path)

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
    print(f"\nRewrites merged: {total_merged}, "
          f"Skipped (lookup): {total_skipped_lookup}, "
          f"Skipped (no cluster_summaries): {total_skipped_no_cs}, "
          f"Files modified: {len(modified_files)}")


if __name__ == "__main__":
    main()
