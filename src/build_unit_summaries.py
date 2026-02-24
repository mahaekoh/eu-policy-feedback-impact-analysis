"""Build per-initiative unified summaries from summarize_documents.py output.

For each initiative, produces:
  - before_feedback_summary: concatenation of summary fields from documents_before_feedback
  - after_feedback_summary: concatenation of summary fields from documents_after_feedback
  - combined_feedback_summary on each middle_feedback item: feedback_text + summaries
    of all attachments, concatenated

All concatenation joins on '\n\n'.

Usage:
    python3 src/build_unit_summaries.py summaries_output_v3/ -o unit_summaries/
"""

import argparse
import json
import os
import sys


def build_unit_summary(initiative):
    """Add unified summary fields to an initiative dict (mutates in place)."""
    # before_feedback_summary
    before_parts = []
    for doc in initiative.get("documents_before_feedback", []):
        summary = doc.get("summary")
        if summary:
            before_parts.append(summary)
    if before_parts:
        initiative["before_feedback_summary"] = "\n\n".join(before_parts)

    # after_feedback_summary
    after_parts = []
    for doc in initiative.get("documents_after_feedback", []):
        summary = doc.get("summary")
        if summary:
            after_parts.append(summary)
    if after_parts:
        initiative["after_feedback_summary"] = "\n\n".join(after_parts)

    # combined_feedback_summary on each middle_feedback item
    for fb in initiative.get("middle_feedback", []):
        parts = []
        feedback_text = fb.get("feedback_text")
        if feedback_text:
            parts.append(feedback_text)
        for att in fb.get("attachments", []):
            summary = att.get("summary")
            if summary:
                parts.append(summary)
        if parts:
            fb["combined_feedback_summary"] = "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Build per-initiative unified summaries from summarize_documents.py output."
    )
    parser.add_argument(
        "input_dir",
        help="Directory of per-initiative JSON files (output of summarize_documents.py)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for initiative JSONs with unified summaries.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    filenames = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".json"))
    print(f"Found {len(filenames)} initiative files in {args.input_dir}/")

    max_policy_len = 0
    max_policy_file = None
    max_feedback_len = 0
    max_feedback_file = None
    max_feedback_id = None

    for filename in filenames:
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            initiative = json.load(f)

        build_unit_summary(initiative)

        for key in ("before_feedback_summary", "after_feedback_summary"):
            length = len(initiative.get(key, ""))
            if length > max_policy_len:
                max_policy_len = length
                max_policy_file = filename

        for fb in initiative.get("middle_feedback", []):
            length = len(fb.get("combined_feedback_summary", ""))
            if length > max_feedback_len:
                max_feedback_len = length
                max_feedback_file = filename
                max_feedback_id = fb.get("id")

        out_path = os.path.join(args.output, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(initiative, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(filenames)} files to {args.output}/")
    print(f"Longest policy-level summary: {max_policy_len} chars ({max_policy_file})")
    print(f"Longest feedback-level summary: {max_feedback_len} chars ({max_feedback_file} fb={max_feedback_id})")


if __name__ == "__main__":
    main()
