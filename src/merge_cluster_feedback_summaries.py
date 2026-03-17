"""Merge cluster summaries back into initiative detail JSON files.

Takes the cluster summary output from summarize_clusters.py and updates the
corresponding initiative JSON files in-place, setting:
- cluster_feedback_summary on each feedback item that has a summary
- cluster_policy_summary at the top level (from policy_summary)
- cluster_summaries at the top level (per-cluster aggregate summaries)

The cluster_feedback_summary field survives re-scrapes because the scrape merge
strategy preserves all derived fields on feedback items when feedback_text is
unchanged. Top-level derived fields are also preserved.

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

    total_fb_merged = 0
    total_skipped_lookup = 0
    total_skipped_no_summary = 0
    total_policy_merged = 0
    total_cluster_merged = 0
    modified_files = set()
    n_files = len(summary_files)

    print(f"Merging cluster feedback summaries into {args.details_dir}...")
    for i, summary_file in enumerate(summary_files):
        if i % 500 == 0:
            print(f"  Processing {i}/{n_files}...")
        summary_path = os.path.join(args.summary_dir, summary_file)
        with open(summary_path, encoding="utf-8") as f:
            summary_data = json.load(f)

        init_id = summary_file.replace(".json", "")
        json_path = os.path.join(args.details_dir, f"{init_id}.json")
        if not os.path.isfile(json_path):
            print(f"  SKIP initiative {init_id}: file not found", file=sys.stderr)
            feedback_summaries = summary_data.get("feedback_summaries", {})
            total_skipped_lookup += len(feedback_summaries)
            continue

        with open(json_path, encoding="utf-8") as f:
            initiative = json.load(f)

        changed = False

        # Merge policy_summary at top level
        policy_summary = summary_data.get("policy_summary")
        if policy_summary and policy_summary.get("summary"):
            value = {"title": policy_summary.get("title", ""),
                     "summary": policy_summary["summary"]}
            if args.dry_run:
                existing = initiative.get("cluster_policy_summary")
                status = "UPDATE" if existing else "NEW"
                print(f"\n[{status}] initiative {init_id}, policy_summary")
                print(f"  title:   {value['title'][:100]}")
                print(f"  summary: {value['summary'].replace(chr(10), ' ')[:120]}")
            else:
                initiative["cluster_policy_summary"] = value
                changed = True
            total_policy_merged += 1

        # Merge cluster_summaries at top level
        cluster_summaries = summary_data.get("cluster_summaries", {})
        if cluster_summaries:
            if args.dry_run:
                existing = initiative.get("cluster_summaries")
                status = "UPDATE" if existing else "NEW"
                print(f"\n[{status}] initiative {init_id}, cluster_summaries ({len(cluster_summaries)} entries)")
            else:
                initiative["cluster_summaries"] = cluster_summaries
                changed = True
            total_cluster_merged += 1

        # Merge per-feedback summaries
        feedback_summaries = summary_data.get("feedback_summaries", {})
        if feedback_summaries:
            # Build feedback lookup: str(fb_id) -> fb dict
            fb_lookup = {}
            for pub in initiative.get("publications", []):
                for fb in pub.get("feedback", []):
                    fb_lookup[str(fb["id"])] = fb

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

                total_fb_merged += 1

        if changed and not args.dry_run:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(initiative, f, ensure_ascii=False, indent=2)
            modified_files.add(json_path)

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
    print(f"\nFeedback merged: {total_fb_merged}, Policy merged: {total_policy_merged}, "
          f"Cluster summaries merged: {total_cluster_merged}")
    print(f"Skipped (lookup): {total_skipped_lookup}, "
          f"Skipped (no summary): {total_skipped_no_summary}, "
          f"Files modified: {len(modified_files)}")


if __name__ == "__main__":
    main()
