"""Run GPU-accelerated OCR on short-extraction PDFs and write updated report.

Takes the short_pdf_report.json and short_pdfs/ directory produced by
find_short_pdf_extractions.py, runs OCR on each PDF using EasyOCR with CUDA,
and writes a new JSON with ocr_text populated for each record.

Supports multi-GPU parallelism: each GPU gets a fully separate subprocess
with its own CUDA_VISIBLE_DEVICES, avoiding CUDA fork issues.

Usage:
    python3 src/ocr_short_pdfs.py short_pdf_report.json short_pdfs/ -o short_pdf_report_ocr.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import easyocr
import numpy as np
import pymupdf

OCR_DPI = 300


def ocr_pdf(reader: easyocr.Reader, pdf_path: str) -> str:
    """Render and OCR a PDF one page at a time to limit memory usage."""
    doc = pymupdf.open(pdf_path)
    pages_text = []
    for page in doc:
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        results = reader.readtext(img, detail=0, paragraph=True)
        pages_text.append("\n".join(results))
        del img, pix
    doc.close()
    return "\n\n".join(pages_text)


def run_worker(shard_path, output_path, pdf_dir, langs):
    """Entry point for subprocess workers. Processes a shard of work items."""
    with open(shard_path, encoding="utf-8") as f:
        work_items = json.load(f)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[GPU {gpu_id}] Initializing EasyOCR (languages={langs})...")
    reader = easyocr.Reader(langs, gpu=True)
    print(f"[GPU {gpu_id}] Processing {len(work_items)} PDFs...")

    results = []
    for wi, item in enumerate(work_items):
        rec_idx = item["rec_idx"]
        pdf_path = os.path.join(pdf_dir, item["pdf_file"])
        t0 = time.time()
        try:
            text = ocr_pdf(reader, pdf_path)
            elapsed = time.time() - t0
            results.append({
                "rec_idx": rec_idx,
                "ocr_text": text,
                "ocr_text_chars": len(text),
            })
            print(f"[GPU {gpu_id}] [{wi+1}/{len(work_items)}] {item['pdf_file']}: {len(text)} chars ({elapsed:.1f}s)")
        except Exception as exc:
            elapsed = time.time() - t0
            results.append({
                "rec_idx": rec_idx,
                "ocr_error": str(exc),
            })
            print(f"[GPU {gpu_id}] [{wi+1}/{len(work_items)}] {item['pdf_file']}: ERROR {exc}", file=sys.stderr)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[GPU {gpu_id}] Done. Wrote {len(results)} results to {output_path}")


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
    parser.add_argument(
        "--worker", nargs=3, metavar=("SHARD", "OUT", "LANGS"),
        help=argparse.SUPPRESS,  # internal: launched by main as subprocess
    )
    args = parser.parse_args()

    # Subprocess worker mode
    if args.worker:
        shard_path, output_path, langs_str = args.worker
        langs = [l.strip() for l in langs_str.split(",")]
        run_worker(shard_path, output_path, args.pdf_dir, langs)
        return

    with open(args.report, encoding="utf-8") as f:
        records = json.load(f)

    langs = [l.strip() for l in args.languages.split(",")]
    langs_str = ",".join(langs)

    # Detect GPUs
    num_gpus = 0
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA not available, falling back to CPU", file=sys.stderr)
    except ImportError:
        print("PyTorch not found, falling back to CPU", file=sys.stderr)

    # Pre-filter: separate valid work items from skips
    work_items = []  # dicts with rec_idx, pdf_file
    skipped = 0
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
        work_items.append({"rec_idx": i, "pdf_file": pdf_file})

    total_work = len(work_items)
    total = len(records)
    print(f"{total_work} PDFs to OCR, {skipped} skipped, {total} total records")

    if not work_items:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Nothing to do. Wrote {args.output}")
        return

    t_start = time.time()

    if num_gpus > 1:
        # Multi-GPU: launch separate subprocesses per GPU
        print(f"Using {num_gpus} GPUs in parallel via subprocesses")

        # Distribute work round-robin
        gpu_shards = [[] for _ in range(num_gpus)]
        for wi, item in enumerate(work_items):
            gpu_shards[wi % num_gpus].append(item)

        # Write shards and launch subprocesses
        tmp_dir = tempfile.mkdtemp(prefix="ocr_shards_")
        procs = []
        out_paths = []
        for gpu_id in range(num_gpus):
            if not gpu_shards[gpu_id]:
                continue
            shard_path = os.path.join(tmp_dir, f"shard_{gpu_id}.json")
            out_path = os.path.join(tmp_dir, f"results_{gpu_id}.json")
            with open(shard_path, "w", encoding="utf-8") as f:
                json.dump(gpu_shards[gpu_id], f, ensure_ascii=False)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            p = subprocess.Popen(
                [sys.executable, __file__, args.report, args.pdf_dir,
                 "-o", args.output,
                 "--worker", shard_path, out_path, langs_str],
                env=env,
            )
            procs.append((gpu_id, p))
            out_paths.append(out_path)
            print(f"  Launched worker on GPU {gpu_id} ({len(gpu_shards[gpu_id])} PDFs, pid={p.pid})")

        # Wait for all workers
        for gpu_id, p in procs:
            rc = p.wait()
            if rc != 0:
                print(f"  WARNING: GPU {gpu_id} worker exited with code {rc}", file=sys.stderr)

        # Collect results
        done = 0
        errors = 0
        for out_path in out_paths:
            if not os.path.isfile(out_path):
                print(f"  WARNING: missing result file {out_path}", file=sys.stderr)
                continue
            with open(out_path, encoding="utf-8") as f:
                results = json.load(f)
            for r in results:
                idx = r["rec_idx"]
                if "ocr_error" in r:
                    records[idx]["ocr_text"] = None
                    records[idx]["ocr_error"] = r["ocr_error"]
                    errors += 1
                else:
                    records[idx]["ocr_text"] = r["ocr_text"]
                    records[idx]["ocr_text_chars"] = r["ocr_text_chars"]
                    done += 1

        # Clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    else:
        # Single GPU or CPU
        use_gpu = num_gpus >= 1
        print(f"Initializing EasyOCR (languages={langs}, gpu={use_gpu})...")
        reader = easyocr.Reader(langs, gpu=use_gpu)

        done = 0
        errors = 0
        for item in work_items:
            rec_idx = item["rec_idx"]
            pdf_file = item["pdf_file"]
            pdf_path = os.path.join(args.pdf_dir, pdf_file)
            t0 = time.time()
            try:
                text = ocr_pdf(reader, pdf_path)
                records[rec_idx]["ocr_text"] = text
                records[rec_idx]["ocr_text_chars"] = len(text)
                elapsed = time.time() - t0
                done += 1
                print(f"[{done + errors + skipped}/{total}] {pdf_file}: {len(text)} chars ({elapsed:.1f}s)")
            except Exception as exc:
                records[rec_idx]["ocr_text"] = None
                records[rec_idx]["ocr_error"] = str(exc)
                errors += 1
                print(f"[{done + errors + skipped}/{total}] {pdf_file}: ERROR {exc}", file=sys.stderr)

    total_elapsed = time.time() - t_start
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nDone in {total_elapsed:.1f}s. OCR'd {done}, errors {errors}, skipped {skipped}. Wrote {args.output}")


if __name__ == "__main__":
    main()
