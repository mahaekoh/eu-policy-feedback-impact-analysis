"""
Repair broken feedback attachments by retrying text extraction.

Scans initiative_details/ for feedback attachments that have
extracted_text_error and no extracted_text, downloads them, and retries
extraction.  For .doc/.docx/.odt/.rtf files, tries PDF extraction first
(since many are mislabeled PDFs), then falls back to the format-specific
pipeline from scrape_eu_initiative_details.py.

Writes updated copies of initiative JSONs to a specified output folder.
Only writes files that had at least one attachment repaired.

Usage:
    python src/repair_broken_attachments.py -o repaired_details/
    python src/repair_broken_attachments.py -o repaired_details/ -f initiative-whitelist-145.txt
    python src/repair_broken_attachments.py -o repaired_details/ -w 20
"""

import argparse
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
import pymupdf4llm
import pypandoc
from docx2md import DocxFile, Converter, DocxMedia

WORKERS = 20
OCR_MIN_CHARS = 100
OCR_MIN_FILE_BYTES = 2048

# Extensions where we try PDF extraction first before the native pipeline
TRY_PDF_FIRST_EXTENSIONS = {".doc", ".docx", ".odt", ".rtf"}


# ── Download ─────────────────────────────────────────────────────────

def download_bytes(url: str, label: str = "") -> bytes:
    req = urllib.request.Request(url)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  RETRY {attempt+1} (wait {wait}s): {label} — {exc}")
                time.sleep(wait)
            else:
                raise


# ── Extraction from bytes ────────────────────────────────────────────

def extract_pdf_from_bytes(data: bytes, label: str = "") -> str:
    """Extract text from PDF bytes using pymupdf4llm with OCR fallback."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.write(tmp_fd, data)
        os.close(tmp_fd)
        try:
            text = pymupdf4llm.to_markdown(tmp_path)
        except Exception:
            doc = pymupdf.open(tmp_path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()

        stripped = text.strip() if text else ""
        if len(stripped) < OCR_MIN_CHARS and len(data) > OCR_MIN_FILE_BYTES:
            print(f"  PDF text too short ({len(stripped)} chars), OCR fallback: {label}")
            doc = pymupdf.open(tmp_path)
            ocr_pages = []
            for page in doc:
                tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                ocr_pages.append(page.get_text(textpage=tp))
            doc.close()
            text = "\n\n".join(ocr_pages)
        return text
    finally:
        os.unlink(tmp_path)


def extract_docx_from_bytes(data: bytes, suffix: str, label: str = "") -> str:
    """Extract text from DOCX/DOC bytes."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, data)
        os.close(tmp_fd)
        if suffix == ".docx":
            docx = DocxFile(tmp_path)
            xml_text = docx.document()
            media = DocxMedia(docx)
            converter = Converter(xml_text, media, use_md_table=True)
            text = converter.convert()
            docx.close()
        else:
            # .doc — macOS textutil
            txt_path = tmp_path + ".txt"
            subprocess.run(
                ["textutil", "-convert", "txt", "-output", txt_path, tmp_path],
                check=True, capture_output=True,
            )
            text = Path(txt_path).read_text(encoding="utf-8")
            os.unlink(txt_path)
        return text
    finally:
        os.unlink(tmp_path)


def extract_pandoc_from_bytes(data: bytes, suffix: str, label: str = "") -> str:
    """Extract text from RTF/ODT bytes using pypandoc."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, data)
        os.close(tmp_fd)
        return pypandoc.convert_file(tmp_path, "markdown")
    finally:
        os.unlink(tmp_path)


def extract_native(data: bytes, ext: str, label: str = "") -> str:
    """Run the format-specific extraction pipeline (same as scrape_eu_initiative_details)."""
    if ext == ".pdf":
        return extract_pdf_from_bytes(data, label)
    elif ext in (".docx", ".doc"):
        return extract_docx_from_bytes(data, ext, label)
    elif ext == ".txt":
        return data.decode("utf-8", errors="replace")
    elif ext in (".rtf", ".odt"):
        return extract_pandoc_from_bytes(data, ext, label)
    else:
        raise ValueError(f"Unsupported extension: {ext}")


# ── Repair logic ─────────────────────────────────────────────────────

def repair_attachment(att: dict, init_id: int, fb_id: int) -> bool:
    """Try to extract text for a broken attachment.  Mutates att in place.

    Returns True if the attachment was successfully repaired.
    """
    filename = att.get("filename", "")
    download_url = att.get("download_url", "")
    ext = Path(filename).suffix.lower()
    label = f"init={init_id} fb={fb_id} {filename}"

    if not download_url:
        print(f"  SKIP no download_url: {label}")
        return False

    t0 = time.time()
    try:
        data = download_bytes(download_url, label)
    except Exception as exc:
        print(f"  DOWNLOAD FAILED: {label} — {exc}")
        return False

    text = None
    method = None

    # For non-PDF extensions, try PDF extraction first
    if ext in TRY_PDF_FIRST_EXTENSIONS:
        try:
            text = extract_pdf_from_bytes(data, label)
            stripped = (text or "").strip()
            if len(stripped) >= OCR_MIN_CHARS:
                method = "pdf-reinterpret"
            else:
                text = None  # too short, try native
        except Exception:
            text = None  # not a PDF, try native

    # Fall back to native extraction
    if text is None:
        try:
            text = extract_native(data, ext, label)
            method = "native"
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.1f}s): {label} — {exc}")
            return False

    stripped = (text or "").strip()
    if not stripped:
        elapsed = time.time() - t0
        print(f"  EMPTY ({elapsed:.1f}s): {label}")
        return False

    elapsed = time.time() - t0
    att["extracted_text"] = text
    old_error = att.pop("extracted_text_error", None)
    att["repair_method"] = method
    att["repair_old_error"] = old_error
    print(f"  REPAIRED ({elapsed:.1f}s, {method}, {len(stripped)} chars): {label}")
    return True


# ── Main ─────────────────────────────────────────────────────────────

def find_broken_attachments(initiative: dict) -> list[tuple[dict, int, int, int]]:
    """Return list of (attachment_dict, initiative_id, publication_id, feedback_id) for broken attachments."""
    init_id = initiative.get("id", 0)
    broken = []
    for pub in initiative.get("publications", []):
        pub_id = pub.get("publication_id", 0)
        for fb in pub.get("feedback", []):
            for att in fb.get("attachments", []):
                if "extracted_text_error" in att and "extracted_text" not in att:
                    broken.append((att, init_id, pub_id, fb.get("id", 0)))
    return broken


def main():
    parser = argparse.ArgumentParser(
        description="Repair broken feedback attachments by retrying text extraction."
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, default="data/scrape/initiative_details",
        help="Input directory with per-initiative JSON files (default: data/scrape/initiative_details/).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        help="Output directory for repaired JSON files.",
    )
    parser.add_argument(
        "-f", "--filter", type=str, default=None,
        help="File with initiative IDs to process (one per line).",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=WORKERS,
        help=f"Number of parallel workers (default: {WORKERS}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report broken attachments without repairing.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load filter
    filter_ids = None
    if args.filter:
        with open(args.filter) as f:
            filter_ids = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    filter_ids.add(int(line))
        print(f"Filter: {len(filter_ids)} initiative IDs")

    # Scan for broken attachments
    print(f"Scanning {input_dir}/ ...")
    all_tasks = []  # (json_path, initiative_dict, [(att, init_id, fb_id), ...])
    total_broken = 0

    for json_file in sorted(input_dir.glob("*.json")):
        try:
            init_id = int(json_file.stem)
        except ValueError:
            continue
        if filter_ids is not None and init_id not in filter_ids:
            continue

        with open(json_file, encoding="utf-8") as f:
            initiative = json.load(f)

        broken = find_broken_attachments(initiative)
        if broken:
            all_tasks.append((json_file, initiative, broken))
            total_broken += len(broken)

    print(f"Found {total_broken} broken attachments across {len(all_tasks)} initiatives")

    if args.dry_run or total_broken == 0:
        return

    # Repair in parallel across attachments
    repaired_count = 0
    failed_count = 0
    lock = threading.Lock()
    initiatives_with_repairs = set()
    report_records = []

    # Flatten all attachment tasks with back-reference to initiative index
    flat_tasks = []
    for task_idx, (json_file, initiative, broken) in enumerate(all_tasks):
        for att, init_id, pub_id, fb_id in broken:
            flat_tasks.append((task_idx, att, init_id, pub_id, fb_id))

    print(f"Processing {len(flat_tasks)} attachments with {args.workers} workers...")

    def do_repair(task_idx, att, init_id, pub_id, fb_id):
        nonlocal repaired_count, failed_count
        ok = repair_attachment(att, init_id, fb_id)
        with lock:
            if ok:
                repaired_count += 1
                initiatives_with_repairs.add(task_idx)
                report_records.append({
                    "initiative_id": init_id,
                    "publication_id": pub_id,
                    "feedback_id": fb_id,
                    "attachment_id": att.get("id"),
                    "filename": att.get("filename", ""),
                    "download_url": att.get("download_url", ""),
                    "repair_method": att.get("repair_method"),
                    "extracted_text_chars": len((att.get("extracted_text") or "").strip()),
                })
            else:
                failed_count += 1
            done = repaired_count + failed_count
            if done % 100 == 0:
                print(f"  Progress: {done}/{len(flat_tasks)} ({repaired_count} repaired, {failed_count} failed)")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(do_repair, task_idx, att, init_id, pub_id, fb_id)
            for task_idx, att, init_id, pub_id, fb_id in flat_tasks
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"  UNEXPECTED ERROR: {exc}")

    # Write output files (only for initiatives that had repairs)
    written = 0
    for task_idx in initiatives_with_repairs:
        json_file, initiative, _ = all_tasks[task_idx]
        out_path = output_dir / json_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(initiative, f, ensure_ascii=False, indent=2)
        written += 1

    # Write report
    report_records.sort(key=lambda r: (r["initiative_id"], r["publication_id"], r["feedback_id"], r["attachment_id"] or 0))
    report_path = output_dir / "repair_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_records, f, ensure_ascii=False, indent=2)

    print(
        f"\nDone. {repaired_count} repaired, {failed_count} failed. "
        f"Wrote {written} initiative files + repair_report.json to {output_dir}/"
    )


if __name__ == "__main__":
    main()
