"""Find attachments (feedback and publication docs) that have no extracted text.

Usage:
    python3 src/find_missing_extracted_text.py initiative_details/
    python3 src/find_missing_extracted_text.py initiative_details/ -f initiative-whitelist-145.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path


def scan_dir(dir_path):
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                    yield json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping corrupt file: {fname}", file=sys.stderr)
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Find attachments that have no extracted text."
    )
    parser.add_argument(
        "source",
        help="Path to initiative_details/ directory",
    )
    parser.add_argument(
        "-f", "--filter",
        help="Path to newline-delimited file of initiative IDs to include.",
    )
    args = parser.parse_args()

    whitelist = None
    if args.filter:
        with open(args.filter, encoding="utf-8") as f:
            whitelist = {int(line.strip()) for line in f if line.strip()}
        print(f"Filtering to {len(whitelist)} whitelisted initiatives")

    initiatives = scan_dir(args.source)

    missing_docs = 0
    missing_fb_att = 0
    has_docs = 0
    has_fb_att = 0
    header_printed = False
    for obj in initiatives:
        if "error" in obj:
            continue
        init_id = obj["id"]
        if whitelist is not None and init_id not in whitelist:
            continue
        for pub in obj.get("publications", []):
            pub_id = pub.get("publication_id", "?")
            pub_type = pub.get("type", "?")

            # Check publication documents
            for doc in pub.get("documents", []):
                if doc.get("extracted_text"):
                    has_docs += 1
                    continue
                filename = doc.get("filename", "?")
                ext = Path(filename).suffix.lower() or "n/a"
                error = doc.get("extracted_text_error", "")
                if not header_printed:
                    print(f"{'Type':<8} {'Init ID':<10} {'Pub ID':<10} {'Pub Type':<22} {'Ext':<6} {'Error':<40} Filename")
                    print("-" * 130)
                    header_printed = True
                error_short = error[:40] if error else "(none)"
                print(f"{'doc':<8} {init_id:<10} {pub_id:<10} {pub_type:<22} {ext:<6} {error_short:<40} {filename}")
                missing_docs += 1

            # Check feedback attachments
            for fb in pub.get("feedback", []):
                fb_id = fb.get("id", "?")
                for att in fb.get("attachments", []):
                    if att.get("extracted_text"):
                        has_fb_att += 1
                        continue
                    filename = att.get("filename", "?")
                    ext = Path(filename).suffix.lower() or "n/a"
                    error = att.get("extracted_text_error", "")
                    if not header_printed:
                        print(f"{'Type':<8} {'Init ID':<10} {'Pub ID':<10} {'Pub Type':<22} {'Ext':<6} {'Error':<40} Filename")
                        print("-" * 130)
                        header_printed = True
                    error_short = error[:40] if error else "(none)"
                    print(f"{'fb_att':<8} {init_id:<10} {pub_id:<10} {pub_type:<22} {ext:<6} {error_short:<40} {filename}")
                    missing_fb_att += 1

    total_docs = has_docs + missing_docs
    total_fb_att = has_fb_att + missing_fb_att
    print(f"\nPublication docs:      {has_docs} with text, {missing_docs} missing ({total_docs} total)")
    print(f"Feedback attachments:  {has_fb_att} with text, {missing_fb_att} missing ({total_fb_att} total)")
    print(f"Overall:               {has_docs + has_fb_att} with text, {missing_docs + missing_fb_att} missing ({total_docs + total_fb_att} total)")


if __name__ == "__main__":
    main()
