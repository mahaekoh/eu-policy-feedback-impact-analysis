"""Run GPU-accelerated OCR on short-extraction PDFs and write updated report.

Takes the short_pdf_report.json and short_pdfs/ directory produced by
find_short_pdf_extractions.py, runs OCR on each PDF using EasyOCR with CUDA,
and writes a new JSON with ocr_text populated for each record.

Usage:
    python3 src/ocr_short_pdfs.py short_pdf_report.json short_pdfs/ -o short_pdf_report_ocr.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import easyocr
import numpy as np
import pymupdf

OCR_DPI = 300


def render_pdf_pages(pdf_path: str) -> list:
    """Render each page of a PDF to a numpy array (RGB)."""
    doc = pymupdf.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        # Convert RGBA to RGB if needed
        if pix.n == 4:
            img = img[:, :, :3]
        images.append(img)
    doc.close()
    return images


def ocr_pdf(reader: easyocr.Reader, pdf_path: str) -> str:
    """Render a PDF and run OCR on all pages, return concatenated text."""
    images = render_pdf_pages(pdf_path)
    pages_text = []
    for img in images:
        results = reader.readtext(img, detail=0, paragraph=True)
        pages_text.append("\n".join(results))
    return "\n\n".join(pages_text)


def main():
    parser = argparse.ArgumentParser(
        description="Run GPU-accelerated OCR on short-extraction PDFs."
    )
    parser.add_argument(
        "report", help="Path to short_pdf_report.json"
    )
    parser.add_argument(
        "pdf_dir", help="Directory containing downloaded short-extraction PDFs"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output path for updated JSON report with OCR results.",
    )
    parser.add_argument(
        "--languages", type=str, default="en",
        help="Comma-separated EasyOCR language codes (default: en).",
    )
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as f:
        records = json.load(f)

    langs = [l.strip() for l in args.languages.split(",")]
    use_gpu = True
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU", file=sys.stderr)
            use_gpu = False
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not found, falling back to CPU", file=sys.stderr)
        use_gpu = False

    print(f"Initializing EasyOCR (languages={langs}, gpu={use_gpu})...")
    reader = easyocr.Reader(langs, gpu=use_gpu)

    total = len(records)
    skipped = 0
    done = 0
    t_start = time.time()

    for i, rec in enumerate(records):
        pdf_file = rec.get("pdf_file")
        if not pdf_file:
            rec["ocr_text"] = None
            rec["ocr_error"] = "no pdf_file"
            skipped += 1
            continue

        pdf_path = os.path.join(args.pdf_dir, pdf_file)
        if not os.path.isfile(pdf_path):
            rec["ocr_text"] = None
            rec["ocr_error"] = f"file not found: {pdf_path}"
            skipped += 1
            continue

        t0 = time.time()
        try:
            text = ocr_pdf(reader, pdf_path)
            rec["ocr_text"] = text
            rec["ocr_text_chars"] = len(text)
            elapsed = time.time() - t0
            done += 1
            print(f"[{done + skipped}/{total}] {pdf_file}: {len(text)} chars ({elapsed:.1f}s)")
        except Exception as exc:
            rec["ocr_text"] = None
            rec["ocr_error"] = str(exc)
            skipped += 1
            print(f"[{done + skipped}/{total}] {pdf_file}: ERROR {exc}", file=sys.stderr)

    total_elapsed = time.time() - t_start
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nDone in {total_elapsed:.1f}s. OCR'd {done}, skipped {skipped}. Wrote {args.output}")


if __name__ == "__main__":
    main()
