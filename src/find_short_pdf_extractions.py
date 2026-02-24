"""Find feedback PDF attachments where the extracted text is suspiciously short."""

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DOWNLOAD_WORKERS = 20


def scan_dir(dir_path):
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                    yield json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping corrupt file: {fname}", file=sys.stderr)
                continue


def download_pdf(url, dest_path):
    """Download a PDF from url to dest_path."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(dest_path, "wb") as f:
            f.write(resp.read())


def main():
    parser = argparse.ArgumentParser(
        description="Find PDF attachments with suspiciously short extracted text."
    )
    parser.add_argument(
        "source",
        help="Path to initiative_details/ directory.",
    )
    parser.add_argument(
        "-p", "--pdf-dir", type=str, default=None,
        help="Directory to download short-extraction PDFs into.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output path for JSON file with short-extraction records.",
    )
    parser.add_argument(
        "-f", "--filter", type=str, default=None,
        help="File with newline-delimited initiative IDs to include.",
    )
    parser.add_argument(
        "-r", "--repair-report", type=str, default=None,
        help="Path to repair_report.json. When set, only check feedback "
             "attachments listed in the report (any extension, not just .pdf).",
    )
    args = parser.parse_args()

    whitelist = None
    if args.filter:
        with open(args.filter, encoding="utf-8") as f:
            whitelist = {int(line.strip()) for line in f if line.strip()}
        print(f"Filtering to {len(whitelist)} initiative IDs from {args.filter}")

    # Load repair report as a set of (init_id, pub_id, fb_id, att_id) tuples
    repair_set = None
    if args.repair_report:
        with open(args.repair_report, encoding="utf-8") as f:
            repair_data = json.load(f)
        repair_set = {
            (r["initiative_id"], r["publication_id"], r["feedback_id"], r["attachment_id"])
            for r in repair_data
        }
        print(f"Repair report: {len(repair_set)} attachments from {args.repair_report}")

    if args.pdf_dir:
        Path(args.pdf_dir).mkdir(parents=True, exist_ok=True)

    initiatives = scan_dir(args.source)

    # First pass: collect all short-extraction records
    records = []
    for obj in initiatives:
        if "error" in obj:
            continue
        init_id = obj["id"]
        if whitelist is not None and init_id not in whitelist:
            continue
        for pub in obj.get("publications", []):
            pub_id = pub.get("publication_id", "?")

            for doc in pub.get("documents", []):
                text = doc.get("extracted_text", "")
                if not text or len(text) >= 100:
                    continue
                pdf_name = f"{init_id}_pub{pub_id}.pdf"
                records.append({
                    "type": "doc",
                    "initiative_id": init_id,
                    "publication_id": pub_id,
                    "filename": doc.get("filename", ""),
                    "download_url": doc.get("download_url", ""),
                    "extracted_text": text,
                    "extracted_text_chars": len(text),
                    "size_bytes": doc.get("size_bytes"),
                    "pdf_file": pdf_name,
                })

            for fb in pub.get("feedback", []):
                fb_id = fb.get("id", "?")
                for att in fb.get("attachments", []):
                    att_id = att.get("id", "?")
                    # When filtering by repair report, only check listed attachments
                    if repair_set is not None:
                        if (init_id, pub_id, fb_id, att_id) not in repair_set:
                            continue
                    text = att.get("extracted_text", "")
                    if not text or len(text) >= 100:
                        continue
                    pdf_name = f"{init_id}_pub{pub_id}_fb{fb_id}_att{att_id}.pdf"
                    records.append({
                        "type": "fb_att",
                        "initiative_id": init_id,
                        "publication_id": pub_id,
                        "feedback_id": fb_id,
                        "attachment_id": att_id,
                        "filename": att.get("filename", ""),
                        "download_url": att.get("download_url", ""),
                        "extracted_text": text,
                        "extracted_text_chars": len(text),
                        "size_bytes": att.get("size_bytes"),
                        "pdf_file": pdf_name,
                    })

    # Print results
    for i, rec in enumerate(records):
        if i == 0:
            print(f"{'Type':<8} {'Init ID':<10} {'ID':<12} {'Chars':<8} {'Size':<10} Filename")
            print("-" * 110)
        row_id = rec.get("feedback_id", rec["publication_id"])
        snippet = rec["extracted_text"].replace("\n", " ").strip()
        print(f"{rec['type']:<8} {rec['initiative_id']:<10} {row_id:<12} {rec['extracted_text_chars']:<8} {str(rec['size_bytes'] or '?'):<10} {rec['filename']}")
        print(f"         text: {snippet}")
        print(f"         url:  {rec['download_url']}")

    # Download PDFs in parallel
    if args.pdf_dir and records:
        print(f"\nDownloading {len(records)} PDFs to {args.pdf_dir} ({DOWNLOAD_WORKERS} workers)...")
        done = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            future_to_rec = {}
            for rec in records:
                if not rec["download_url"]:
                    rec["pdf_file"] = None
                    continue
                dest = os.path.join(args.pdf_dir, rec["pdf_file"])
                future_to_rec[pool.submit(download_pdf, rec["download_url"], dest)] = rec

            for future in as_completed(future_to_rec):
                rec = future_to_rec[future]
                try:
                    future.result()
                    done += 1
                    print(f"  [{done + failed}/{len(future_to_rec)}] saved: {rec['pdf_file']}")
                except Exception as exc:
                    failed += 1
                    rec["pdf_file"] = None
                    print(f"  [{done + failed}/{len(future_to_rec)}] FAILED: {rec['pdf_file']} — {exc}", file=sys.stderr)

        print(f"Downloaded {done} PDFs ({failed} failed)")
    elif not args.pdf_dir:
        for rec in records:
            rec["pdf_file"] = None

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(records)} records to {args.output}")

    print(f"\nTotal: {len(records)} PDF attachments with suspiciously short extracted text (<100 chars)")


if __name__ == "__main__":
    main()
