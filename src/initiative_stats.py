"""Print publication and attachment stats for a list of initiatives.

For each initiative, shows:
  1. All publications and documents before any feedback is received
     (includes the first publication with feedback, since its documents
     are published before feedback is added)
  2. The final publication and its documents/attachments
  3. Initiatives without any feedback are listed separately at the end

Optionally writes modified initiative JSONs for all initiatives that have
feedback, with added top-level attributes:
  - documents_before_feedback: docs from pubs up to and including first feedback pub
  - documents_after_feedback: docs from the final publication (empty if no post-feedback docs)
  - middle_feedback: all feedback between the first feedback and the final document
    (excludes feedback on the final publication when post-feedback docs exist;
    includes ALL feedback otherwise)

Usage:
    python3 src/initiative_stats.py data/scrape/initiative_details/
    python3 src/initiative_stats.py data/scrape/initiative_details/ -o data/analysis/before_after/
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_initiative(details_dir, init_id):
    json_path = os.path.join(details_dir, f"{init_id}.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def print_pub_summary(pub, indent="  "):
    """Print a publication's documents and feedback attachment counts."""
    pub_id = pub.get("publication_id", "?")
    pub_type = pub.get("type", "?")
    published = pub.get("published_date", "?")
    docs = pub.get("documents", [])
    feedback = pub.get("feedback", [])
    n_fb = len(feedback)
    n_fb_att = sum(len(fb.get("attachments", [])) for fb in feedback)

    print(f"{indent}Publication {pub_id} ({pub_type}) published={published}")
    print(f"{indent}  Feedback: {n_fb} items, {n_fb_att} attachments")

    if docs:
        print(f"{indent}  Documents ({len(docs)}):")
        for doc in docs:
            filename = doc.get("filename", "?")
            has_text = "yes" if doc.get("extracted_text") else "no"
            pages = doc.get("pages", "?")
            size = doc.get("size_bytes", 0)
            size_str = f"{size / 1024:.0f}KB" if size else "?"
            print(f"{indent}    {filename} (pages={pages}, size={size_str}, text={has_text})")


def process_initiative(initiative, details_dir):
    """Process a single initiative. Returns (has_feedback, output_lines)."""
    init_id = initiative["id"]
    title = initiative.get("short_title", "?")[:80]
    pubs = initiative.get("publications", [])

    if not pubs:
        return False, None

    # Sort publications by published_date
    def pub_sort_key(p):
        return p.get("published_date", "") or ""
    pubs_sorted = sorted(pubs, key=pub_sort_key)

    # Check if any publication has feedback
    has_any_feedback = any(len(p.get("feedback", [])) > 0 for p in pubs_sorted)

    if not has_any_feedback:
        return False, pubs_sorted

    # Find pre-feedback publications:
    # All publications up to and including the first one with feedback
    pre_feedback_pubs = []
    for pub in pubs_sorted:
        pre_feedback_pubs.append(pub)
        if len(pub.get("feedback", [])) > 0:
            break

    # Final publication: last non-OPC_LAUNCHED publication with documents
    final_pub = None
    for pub in reversed(pubs_sorted):
        if pub.get("type") != "OPC_LAUNCHED" and pub.get("documents"):
            final_pub = pub
            break
    if final_pub is None:
        final_pub = pubs_sorted[-1]

    return True, {
        "pre_feedback": pre_feedback_pubs,
        "final": final_pub,
        "all": pubs_sorted,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Print publication and attachment stats for initiatives."
    )
    parser.add_argument(
        "details_dir",
        help="Directory of per-initiative JSON files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for modified initiative JSONs (all initiatives with feedback).",
    )
    args = parser.parse_args()

    init_ids = sorted(
        int(p.stem)
        for p in Path(args.details_dir).glob("*.json")
        if p.stem.isdigit()
    )

    print(f"Loading {len(init_ids)} initiatives from {args.details_dir}/\n")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    written_count = 0

    no_feedback = []
    not_found = []
    final_pub_types = {}   # type -> count
    final_pub_examples = {}  # type -> list of (init_id, title)
    feedback_only_on_final = []  # initiatives where all feedback is on the final pub
    no_docs_after_first_fb = []  # initiatives with no publication documents after the first feedback pub

    for init_id in init_ids:
        initiative = load_initiative(args.details_dir, init_id)
        if not initiative:
            not_found.append(init_id)
            continue

        title = initiative.get("short_title", "?")[:80]
        pubs = initiative.get("publications", [])

        has_feedback, result = process_initiative(initiative, args.details_dir)

        if not has_feedback:
            no_feedback.append((init_id, title, result or []))
            continue

        info = result
        pre_feedback = info["pre_feedback"]
        final_pub = info["final"]
        all_pubs = info["all"]

        final_type = final_pub.get("type", "?")
        final_pub_types[final_type] = final_pub_types.get(final_type, 0) + 1
        final_pub_examples.setdefault(final_type, []).append((init_id, title))

        # Check if all feedback is only on the final publication
        non_final_fb = sum(
            len(p.get("feedback", []))
            for p in all_pubs if p["publication_id"] != final_pub["publication_id"]
        )
        if non_final_fb == 0:
            feedback_only_on_final.append((init_id, title))

        # Check if any publications after the first feedback pub have documents
        first_fb_pub_id = pre_feedback[-1]["publication_id"]
        final_pub_id = final_pub["publication_id"]
        found_first_fb = False
        docs_after_first_fb = 0
        for p in all_pubs:
            if found_first_fb:
                docs_after_first_fb += len(p.get("documents", []))
            if p["publication_id"] == first_fb_pub_id:
                found_first_fb = True
        if docs_after_first_fb == 0:
            no_docs_after_first_fb.append((init_id, title))

        if args.output_dir:
            # Build documents_before_feedback: docs from pre-feedback pubs
            docs_before = []
            for p in pre_feedback:
                docs_before.extend(p.get("documents", []))

            # Build documents_after_feedback: empty when no post-feedback documents exist
            if docs_after_first_fb > 0:
                docs_after = list(final_pub.get("documents", []))
            else:
                docs_after = []

            # Build middle_feedback: when no post-feedback docs, include ALL feedback
            # (otherwise the final_pub_id skip would exclude the only feedback pub)
            middle_fb = []
            for p in all_pubs:
                if docs_after_first_fb > 0 and p["publication_id"] == final_pub_id:
                    continue
                middle_fb.extend(p.get("feedback", []))

            initiative["documents_before_feedback"] = docs_before
            initiative["documents_after_feedback"] = docs_after
            initiative["middle_feedback"] = middle_fb

            out_path = os.path.join(args.output_dir, f"{init_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(initiative, f, ensure_ascii=False, indent=2)
            written_count += 1

        print(f"Initiative {init_id}: {title}")
        print(f"  Total publications: {len(all_pubs)}")

        # Pre-feedback publications
        print(f"\n  --- Pre-feedback publications ({len(pre_feedback)}) ---")
        for pub in pre_feedback:
            print_pub_summary(pub, indent="  ")

        # Final publication (skip if same as last pre-feedback pub)
        if final_pub["publication_id"] != pre_feedback[-1]["publication_id"]:
            print(f"\n  --- Final publication ---")
            print_pub_summary(final_pub, indent="  ")
        else:
            print(f"\n  --- Final publication: same as first feedback publication ---")

        # Summary of all publications
        total_docs = sum(len(p.get("documents", [])) for p in all_pubs)
        total_fb = sum(len(p.get("feedback", [])) for p in all_pubs)
        total_fb_att = sum(
            len(fb.get("attachments", []))
            for p in all_pubs
            for fb in p.get("feedback", [])
        )
        print(f"\n  Totals: {total_docs} documents, {total_fb} feedback items, {total_fb_att} feedback attachments")
        print()

    # Initiatives without feedback
    if no_feedback:
        print(f"\n{'='*80}")
        print(f"Initiatives WITHOUT feedback ({len(no_feedback)}):")
        print(f"{'='*80}\n")
        for init_id, title, pubs in no_feedback:
            print(f"Initiative {init_id}: {title}")
            if pubs:
                print(f"  Publications: {len(pubs)}")
                for pub in pubs:
                    print_pub_summary(pub, indent="  ")
            else:
                print(f"  No publications")
            print()

    # Not found
    if not_found:
        print(f"\nNot found ({len(not_found)}): {', '.join(str(i) for i in not_found)}")

    # Final publication type breakdown
    if final_pub_types:
        print(f"\nFinal publication type breakdown:")
        for pub_type, count in sorted(final_pub_types.items(), key=lambda x: -x[1]):
            print(f"  {pub_type:<30} {count}")
            for eid, etitle in final_pub_examples[pub_type][:5]:
                print(f"    e.g. {eid}: {etitle}")

    # No publication documents after first feedback
    if no_docs_after_first_fb:
        print(f"\nInitiatives with NO publication documents after the first feedback ({len(no_docs_after_first_fb)}):")
        for eid, etitle in no_docs_after_first_fb:
            print(f"  {eid}: {etitle}")

    # Feedback only on final publication
    if feedback_only_on_final:
        print(f"\nInitiatives with feedback ONLY on the final publication ({len(feedback_only_on_final)}):")
        for eid, etitle in feedback_only_on_final:
            print(f"  {eid}: {etitle}")

    # Grand totals
    print(f"\nSummary: {len(init_ids)} initiatives, "
          f"{len(init_ids) - len(no_feedback) - len(not_found)} with feedback, "
          f"{len(no_feedback)} without feedback, "
          f"{len(not_found)} not found")

    if args.output_dir:
        print(f"Written {written_count} initiative files to {args.output_dir}/")


if __name__ == "__main__":
    main()
