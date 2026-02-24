"""Find feedback items with attachments where the feedback language is not English."""

import argparse
import json
import os
import sys
from pathlib import Path


def scan_jsonl(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


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
        description="Find non-English feedback attachments in initiative data."
    )
    parser.add_argument(
        "source", nargs="?", default=None,
        help="Path to JSONL file or directory of JSON files. "
             "Defaults to eu_initiative_details.jsonl.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output path for JSON file with attachment records for translation.",
    )
    parser.add_argument(
        "-f", "--filter", type=str, default=None,
        help="File with newline-delimited initiative IDs to include.",
    )
    args = parser.parse_args()

    source = Path(args.source) if args.source else Path(__file__).parent.parent / "eu_initiative_details.jsonl"

    whitelist = None
    if args.filter:
        with open(args.filter, encoding="utf-8") as f:
            whitelist = {int(line.strip()) for line in f if line.strip()}
        print(f"Filtering to {len(whitelist)} initiative IDs from {args.filter}")

    if source.is_dir():
        initiatives = scan_dir(source)
    else:
        initiatives = scan_jsonl(source)

    total = 0
    records = []
    for obj in initiatives:
        if "error" in obj:
            continue
        init_id = obj["id"]
        if whitelist is not None and init_id not in whitelist:
            continue
        for pub in obj.get("publications", []):
            pub_id = pub.get("publication_id")
            pub_type = pub.get("type", "")
            for fb in pub.get("feedback", []):
                lang = fb.get("language", "EN")
                if lang == "EN":
                    continue
                attachments = fb.get("attachments", [])
                if not attachments:
                    continue
                fb_id = fb["id"]
                org = fb.get("organization", "")

                if args.output:
                    for att in attachments:
                        records.append({
                            "initiative_id": init_id,
                            "publication_id": pub_id,
                            "publication_type": pub_type,
                            "feedback_id": fb_id,
                            "attachment_id": att.get("id"),
                            "document_id": att.get("document_id", ""),
                            "filename": att.get("filename", ""),
                            "language": lang,
                            "organization": org,
                            "download_url": att.get("download_url", ""),
                            "extracted_text": att.get("extracted_text", ""),
                            "extracted_text_error": att.get("extracted_text_error"),
                        })
                else:
                    n_att = len(attachments)
                    filenames = ", ".join(a["filename"] for a in attachments)
                    if total == 0:
                        print(f"{'Init ID':<10} {'FB ID':<12} {'Lang':<5} {'Atts':<5} {'Organization':<40} Filenames")
                        print("-" * 120)
                    print(f"{init_id:<10} {fb_id:<12} {lang:<5} {n_att:<5} {org[:40]:<40} {filenames[:50]}")
                    for att in attachments:
                        ext = Path(att["filename"]).suffix.lower() or "n/a"
                        text = att.get("extracted_text", "")
                        if text:
                            snippet = text[:150].replace("\n", " ").strip()
                            print(f"             [{ext}] => {snippet}...")
                        else:
                            print(f"             [{ext}] (no extracted text)")
                total += 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(records)} attachment records to {args.output}")
    else:
        print(f"\nTotal: {total} non-English feedback items with attachments")


if __name__ == "__main__":
    main()
