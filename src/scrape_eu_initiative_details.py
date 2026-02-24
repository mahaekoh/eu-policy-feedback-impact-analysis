"""
Scrape detailed attributes for each EU "Have Your Say" initiative.

Reads initiative URLs from eu_initiatives.csv (produced by scrape_eu_initiatives.py)
and fetches per-initiative detail from the BRP API.

Outputs: initiative_details/*.json  (one JSON file per initiative)

Uses two thread pools: one for initiatives (20 workers), one for feedback
pages within each initiative to avoid deadlocks.
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
import pymupdf4llm
import pypandoc
from docx2md import DocxFile, Converter, DocxMedia

API_BASE = "https://ec.europa.eu/info/law/better-regulation/brpapi"
FEEDBACK_API = "https://ec.europa.eu/info/law/better-regulation/api/allFeedback"
DOWNLOAD_URL = f"{API_BASE}/download"
FEEDBACK_ATTACHMENT_DL = "https://ec.europa.eu/info/law/better-regulation/api/download"
FEEDBACK_PAGE_SIZE = 500
INITIATIVE_WORKERS = 20
FEEDBACK_WORKERS = 20
PDF_WORKERS = 40

# Map publication type codes to human-readable section labels
PUBLICATION_TYPE_LABELS = {
    "INIT_PLANNED": "Planned initiative",
    "CFE_IMPACT_ASSESS": "Call for evidence",
    "CALL_FOR_EVIDENCE": "Call for evidence",
    "CFE_EVL_FC": "Call for evidence",
    "OPC_LAUNCHED": "Public consultation",
    "PROP_REG": "Commission adoption",
    "PROP_DIR": "Commission adoption",
    "PROP_DEC": "Commission adoption",
    "REG_DRAFT": "Draft act",
    "DIR_DRAFT": "Draft act",
    "DEC_DRAFT": "Draft act",
    "DEL_REG_DRAFT": "Draft act",
    "IMPL_REG_DRAFT": "Draft act",
    "IMPL_DEC_DRAFT": "Draft act",
    "DEL_DIR_DRAFT": "Draft act",
    "REG": "Commission adoption",
    "DIR": "Commission adoption",
    "DEC": "Commission adoption",
    "DEL_REG": "Commission adoption",
    "IMPL_REG": "Commission adoption",
    "IMPL_DEC": "Commission adoption",
    "DEL_DIR": "Commission adoption",
    "REPORT": "Commission adoption",
    "SWD": "Commission adoption",
    "COM_NO_LEG": "Commission adoption",
}

ACT_TYPE_LABELS = {
    "PROP_REG": "Proposal for a regulation",
    "PROP_DIR": "Proposal for a directive",
    "PROP_DEC": "Proposal for a decision",
    "REG": "Regulation",
    "DEL_REG": "Delegated regulation",
    "IMPL_REG": "Implementing regulation",
    "DIR": "Directive",
    "DEL_DIR": "Delegated directive",
    "DEC": "Decision",
    "IMPL_DEC": "Implementing decision",
    "REPORT": "Report",
    "SWD": "Staff working document",
    "COM_NO_LEG": "Non-legislative act",
}


def fetch_json(url: str, label: str = "") -> dict:
    """Fetch JSON from URL with retries."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(3):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            elapsed = time.time() - t0
            if elapsed > 5:
                print(f"  SLOW fetch ({elapsed:.1f}s): {label or url}")
            return data
        except (urllib.error.URLError, TimeoutError) as exc:
            elapsed = time.time() - t0
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  RETRY {attempt+1} ({elapsed:.1f}s, wait {wait}s): {label or url} — {exc}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {attempt+1} attempts ({elapsed:.1f}s): {label or url} — {exc}")
                raise


def build_document_label(pub_type: str, attachment: dict) -> str:
    type_label = PUBLICATION_TYPE_LABELS.get(pub_type, pub_type)
    ref = attachment.get("reference", "")
    if attachment.get("type") == "ANNEX":
        type_label = f"{type_label} (Annex)"
    return f"{type_label} - {ref}" if ref else type_label


def extract_english_documents(attachments: list, pub_type: str) -> list[dict]:
    docs = []
    for att in attachments:
        if att.get("language") != "EN":
            continue
        doc_id = att.get("documentId", "")
        filename = att.get("ersFileName") or att.get("filename", "")
        ext = Path(filename).suffix.lower()
        if ext not in (".pdf", ".docx", ".doc", ".rtf", ".odt", ".txt", ".zip"):
            continue
        docs.append({
            "label": build_document_label(pub_type, att),
            "download_url": f"{DOWNLOAD_URL}/{doc_id}" if doc_id else "",
            "filename": filename,
            "title": att.get("title", ""),
            "reference": att.get("reference", ""),
            "doc_type": att.get("type", ""),
            "category": att.get("category", ""),
            "pages": att.get("pages"),
            "size_bytes": att.get("size"),
        })
    return docs


OCR_MIN_CHARS = 100
OCR_MIN_FILE_BYTES = 2048


def download_and_extract_pdf(download_url: str, label: str = "") -> str:
    """Download a PDF and extract its text content using pymupdf4llm.

    Falls back to plain text if markdown extraction fails, then to OCR
    (tesseract via pymupdf) if the extracted text is suspiciously short
    relative to the file size.
    """
    t0 = time.time()
    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        pdf_bytes = resp.read()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.write(tmp_fd, pdf_bytes)
        os.close(tmp_fd)
        try:
            text = pymupdf4llm.to_markdown(tmp_path)
        except (ValueError, Exception) as exc:
            # Fallback: extract plain text when markdown extraction fails
            # (e.g. "not a textpage of this page" on PDFs with complex graphics)
            print(f"  PDF markdown failed ({exc}), falling back to plain text: {label}")
            doc = pymupdf.open(tmp_path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()

        # OCR fallback: if text is suspiciously short for a non-trivial PDF,
        # the fonts likely lack proper Unicode mappings.
        stripped = text.strip() if text else ""
        if len(stripped) < OCR_MIN_CHARS and len(pdf_bytes) > OCR_MIN_FILE_BYTES:
            print(f"  PDF text too short ({len(stripped)} chars, {len(pdf_bytes)} bytes), falling back to OCR: {label}")
            doc = pymupdf.open(tmp_path)
            ocr_pages = []
            for page in doc:
                tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                ocr_pages.append(page.get_text(textpage=tp))
            doc.close()
            text = "\n\n".join(ocr_pages)
    finally:
        os.unlink(tmp_path)
    elapsed = time.time() - t0
    print(f"  PDF extracted ({elapsed:.1f}s, {len(pdf_bytes)} bytes): {label}")
    return text


def download_and_extract_docx(download_url: str, filename: str, label: str = "") -> str:
    """Download a DOCX/DOC file and extract its text.

    Uses docx2md for .docx files and textutil (macOS) for .doc files.
    """
    t0 = time.time()
    suffix = Path(filename).suffix.lower() or ".docx"
    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        doc_bytes = resp.read()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, doc_bytes)
        os.close(tmp_fd)
        if suffix == ".docx":
            docx = DocxFile(tmp_path)
            xml_text = docx.document()
            media = DocxMedia(docx)
            converter = Converter(xml_text, media, use_md_table=True)
            text = converter.convert()
            docx.close()
        else:
            # .doc (old binary format) — use macOS textutil
            txt_path = tmp_path + ".txt"
            subprocess.run(
                ["textutil", "-convert", "txt", "-output", txt_path, tmp_path],
                check=True, capture_output=True,
            )
            text = Path(txt_path).read_text(encoding="utf-8")
            os.unlink(txt_path)
    finally:
        os.unlink(tmp_path)
    elapsed = time.time() - t0
    print(f"  DOCX extracted ({elapsed:.1f}s, {len(doc_bytes)} bytes): {label}")
    return text


def download_and_extract_pandoc(download_url: str, filename: str, label: str = "") -> str:
    """Download an RTF/ODT file and extract its text as markdown using pypandoc."""
    t0 = time.time()
    suffix = Path(filename).suffix.lower()
    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        doc_bytes = resp.read()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, doc_bytes)
        os.close(tmp_fd)
        text = pypandoc.convert_file(tmp_path, "markdown")
    finally:
        os.unlink(tmp_path)
    elapsed = time.time() - t0
    fmt = suffix.lstrip(".")
    print(f"  {fmt.upper()} extracted ({elapsed:.1f}s, {len(doc_bytes)} bytes): {label}")
    return text


def download_and_read_text(download_url: str, label: str = "") -> str:
    """Download a plain text file and return its contents."""
    t0 = time.time()
    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    elapsed = time.time() - t0
    print(f"  TXT read ({elapsed:.1f}s, {len(raw)} bytes): {label}")
    return text


EXTRACTABLE_EXTENSIONS = {".pdf", ".docx", ".doc", ".rtf", ".odt", ".txt"}


def _parse_feedback_items(content: list, initiative_url: str) -> list[dict]:
    results = []
    for item in content:
        feedback_id = item.get("id")
        attachments = []
        for att in item.get("attachments", []):
            doc_id = att.get("documentId", "")
            filename = att.get("fileName", "")
            download_url = f"{FEEDBACK_ATTACHMENT_DL}/{doc_id}" if doc_id else ""
            att_record = {
                "id": att.get("id"),
                "filename": filename,
                "document_id": doc_id,
                "download_url": download_url,
                "pages": att.get("pages"),
                "size_bytes": att.get("size"),
            }
            ext = Path(filename).suffix.lower()
            if ext in EXTRACTABLE_EXTENSIONS and download_url:
                fb_label = f"feedback {feedback_id} {filename}"
                try:
                    if ext == ".pdf":
                        att_record["extracted_text"] = download_and_extract_pdf(
                            download_url, label=fb_label,
                        )
                    elif ext in (".docx", ".doc"):
                        att_record["extracted_text"] = download_and_extract_docx(
                            download_url, filename, label=fb_label,
                        )
                    elif ext == ".txt":
                        att_record["extracted_text"] = download_and_read_text(
                            download_url, label=fb_label,
                        )
                    else:
                        att_record["extracted_text"] = download_and_extract_pandoc(
                            download_url, filename, label=fb_label,
                        )
                except Exception as exc:
                    att_record["extracted_text_error"] = str(exc)
                    print(f"  EXTRACT ERROR feedback {feedback_id} {filename}: {exc}")
            attachments.append(att_record)
        results.append({
            "id": feedback_id,
            "url": f"{initiative_url}/F{feedback_id}_en",
            "date": item.get("dateFeedback", ""),
            "feedback_text": item.get("feedbackTextUserLanguage", "") or item.get("feedback", ""),
            "feedback_text_original": item.get("feedback", ""),
            "language": item.get("language", ""),
            "user_type": item.get("userType", ""),
            "country": item.get("country", ""),
            "company_size": item.get("companySize", ""),
            "organization": item.get("organization", ""),
            "first_name": item.get("firstName", ""),
            "surname": item.get("surname", ""),
            "status": item.get("status", ""),
            "publication": item.get("publication", ""),
            "tr_number": item.get("trNumber", ""),
            "attachments": attachments,
        })
    return results


def _fetch_feedback_page(publication_id: int, page: int) -> dict:
    url = (
        f"{FEEDBACK_API}?publicationId={publication_id}"
        f"&language=EN&page={page}&size={FEEDBACK_PAGE_SIZE}"
        f"&sort=dateFeedback,DESC"
    )
    return fetch_json(url, label=f"feedback pub={publication_id} page={page}")


def fetch_all_feedback(
    publication_id: int,
    initiative_url: str,
) -> tuple:
    """Fetch all feedback for a publication, paginating sequentially.

    Pages are fetched sequentially to avoid deadlocking the thread pool
    (this function itself runs inside a pool). With page size 500, most
    publications need only 1-2 pages.

    Returns (feedback_list, error_string_or_None).
    """
    t0 = time.time()
    all_feedback = []
    page = 0
    error = None

    try:
        while True:
            data = _fetch_feedback_page(publication_id, page)
            all_feedback.extend(
                _parse_feedback_items(data.get("content", []), initiative_url)
            )
            if data.get("last", True):
                break
            page += 1
    except Exception as exc:
        error = str(exc)
        print(
            f"  FEEDBACK ERROR pub={publication_id}: {error} "
            f"(got {len(all_feedback)} items from {page} pages before failure)"
        )

    elapsed = time.time() - t0
    if elapsed > 10 or page > 1:
        print(
            f"  FEEDBACK pub={publication_id}: {len(all_feedback)} items, "
            f"{page+1} pages, {elapsed:.1f}s"
        )
    return all_feedback, error


def _extract_text_for_doc(doc: dict, pub_id: int):
    """Download and extract text for a single document. Mutates doc in place."""
    ext = Path(doc["filename"]).suffix.lower()
    label = f"pub={pub_id} {doc['filename']}"
    try:
        if ext == ".pdf":
            doc["extracted_text"] = download_and_extract_pdf(
                doc["download_url"], label=label,
            )
        elif ext in (".docx", ".doc"):
            doc["extracted_text"] = download_and_extract_docx(
                doc["download_url"], doc["filename"], label=label,
            )
        elif ext == ".txt":
            doc["extracted_text"] = download_and_read_text(
                doc["download_url"], label=label,
            )
        else:
            doc["extracted_text"] = download_and_extract_pandoc(
                doc["download_url"], doc["filename"], label=label,
            )
    except Exception as exc:
        doc["extracted_text_error"] = str(exc)
        print(f"  EXTRACT ERROR pub={pub_id} {doc['filename']}: {exc}")


def extract_publications(
    pubs: list, initiative_url: str, fb_executor: ThreadPoolExecutor,
    pdf_executor: ThreadPoolExecutor = None,
) -> list[dict]:
    sections = []

    # Submit all feedback and PDF fetches to the pool in parallel
    fb_futures: dict[int, object] = {}
    pdf_futures = []
    for i, pub in enumerate(pubs):
        pub_type = pub.get("type", "")
        pub_id = pub.get("id")
        total_feedback = pub.get("totalFeedback", 0)

        documents = extract_english_documents(
            pub.get("attachments", []), pub_type
        )

        section = {
            "publication_id": pub_id,
            "type": pub_type,
            "section_label": PUBLICATION_TYPE_LABELS.get(pub_type, pub_type),
            "reference": pub.get("reference", ""),
            "published_date": pub.get("publishedDate", ""),
            "adoption_date": pub.get("adoptionDate", ""),
            "planned_period": pub.get("plannedPeriod", ""),
            "feedback_end_date": pub.get("endDate", ""),
            "feedback_period_weeks": pub.get("feedbackPeriod", 0),
            "feedback_status": pub.get("receivingFeedbackStatus", ""),
            "total_feedback": total_feedback,
            "documents": documents,
            "feedback": [],
        }
        sections.append(section)

        # Submit text extractions (PDF, DOCX, DOC) in parallel
        _pdf_pool = pdf_executor or fb_executor
        for doc in documents:
            ext = Path(doc["filename"]).suffix.lower()
            if ext in EXTRACTABLE_EXTENSIONS and doc["download_url"]:
                pdf_futures.append(
                    _pdf_pool.submit(_extract_text_for_doc, doc, pub_id)
                )

        if total_feedback > 0 and pub_id:
            fb_futures[i] = fb_executor.submit(
                fetch_all_feedback, pub_id, initiative_url
            )

    # Wait for all PDF extractions
    for future in pdf_futures:
        future.result()

    # Wait for all feedback fetches
    for i, future in fb_futures.items():
        feedback, fb_error = future.result()
        sections[i]["feedback"] = feedback
        if fb_error:
            sections[i]["feedback_error"] = fb_error

    return sections


def extract_initiative(
    data: dict, url: str, fb_executor: ThreadPoolExecutor,
    pdf_executor: ThreadPoolExecutor = None,
) -> dict:
    init_id = data.get("id")
    topics = [t.get("label", "") for t in data.get("topics", [])]
    policy_areas = [p.get("label", "") for p in data.get("policyAreas", [])]
    act_type_code = data.get("foreseenActType", "")

    return {
        "id": init_id,
        "url": url,
        "short_title": data.get("shortTitle", ""),
        "summary": data.get("dossierSummary", ""),
        "reference": data.get("reference", ""),
        "type_of_act": ACT_TYPE_LABELS.get(act_type_code, act_type_code),
        "type_of_act_code": act_type_code,
        "department": data.get("dg", ""),
        "status": data.get("initiativeStatus", ""),
        "stage": data.get("stage", ""),
        "topics": topics,
        "policy_areas": policy_areas,
        "published_date": data.get("publishedDate", ""),
        "publications": extract_publications(
            data.get("publications", []), url, fb_executor, pdf_executor
        ),
    }


def main(out_dir: str = None):
    csv_path = Path(__file__).parent.parent / "eu_initiatives.csv"

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} initiatives from {csv_path.name}")

    out_path = Path(out_dir) if out_dir else Path(__file__).parent.parent / "initiative_details"
    out_path.mkdir(parents=True, exist_ok=True)
    # Resume: check which IDs already have files
    done_ids = set()
    for p in out_path.glob("*.json"):
        try:
            done_ids.add(int(p.stem))
        except ValueError:
            pass
    print(f"Output dir: {out_path}")

    if done_ids:
        print(f"Resuming — {len(done_ids)} already scraped")

    remaining = [(r["url"], r["id"]) for r in rows if int(r["id"]) not in done_ids]
    total = len(rows)
    print(f"Remaining: {len(remaining)}")

    write_lock = threading.Lock()
    done_count = len(done_ids)
    error_count = 0

    # Separate pools to avoid deadlock with initiative pool
    fb_executor = ThreadPoolExecutor(max_workers=FEEDBACK_WORKERS)
    pdf_executor = ThreadPoolExecutor(max_workers=PDF_WORKERS)

    def handle_initiative(url: str, init_id: str):
        nonlocal done_count, error_count
        t_start = time.time()
        try:
            data = fetch_json(
                f"{API_BASE}/groupInitiatives/{init_id}",
                label=f"initiative {init_id}",
            )
            t_api = time.time()
            record = extract_initiative(data, url, fb_executor, pdf_executor)
            t_end = time.time()
            n_docs = sum(len(p["documents"]) for p in record["publications"])
            n_fb = sum(len(p["feedback"]) for p in record["publications"])
            total_elapsed = t_end - t_start
            with write_lock:
                file_path = out_path / f"{init_id}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                done_count += 1
                dc = done_count
            print(
                f"[{dc}/{total}] ID {init_id}: "
                f"{len(record['publications'])} sections, "
                f"{n_docs} docs, {n_fb} feedback "
                f"(api={t_api - t_start:.1f}s, feedback={t_end - t_api:.1f}s, "
                f"total={total_elapsed:.1f}s)"
            )
        except Exception as exc:
            t_end = time.time()
            error_record = {"id": int(init_id), "url": url, "error": str(exc)}
            with write_lock:
                file_path = out_path / f"{init_id}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(error_record, f, ensure_ascii=False, indent=2)
                done_count += 1
                error_count += 1
                dc = done_count
            print(
                f"[{dc}/{total}] ID {init_id}: ERROR {exc} "
                f"(total={t_end - t_start:.1f}s)"
            )

    try:
        with ThreadPoolExecutor(max_workers=INITIATIVE_WORKERS) as init_executor:
            futures = [
                init_executor.submit(handle_initiative, url, init_id)
                for url, init_id in remaining
            ]
            for f in futures:
                f.result()
    finally:
        fb_executor.shutdown(wait=False)
        pdf_executor.shutdown(wait=False)

    # Summary
    total_written = len(list(out_path.glob("*.json")))
    errors = 0
    for p in out_path.glob("*.json"):
        with open(p, encoding="utf-8") as f:
            if "error" in json.load(f):
                errors += 1
    print(f"\nDone. {total_written} initiatives saved to {out_path} ({errors} errors)")


def scrape_one(init_id: int):
    """Scrape a single initiative by ID and print the result as JSON."""
    fb_executor = ThreadPoolExecutor(max_workers=FEEDBACK_WORKERS)
    pdf_executor = ThreadPoolExecutor(max_workers=PDF_WORKERS)
    try:
        data = fetch_json(
            f"{API_BASE}/groupInitiatives/{init_id}",
            label=f"initiative {init_id}",
        )
        slug = data.get("shortTitle", "")
        url = f"https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/{init_id}-{slug}_en"
        record = extract_initiative(data, url, fb_executor, pdf_executor)
        n_docs = sum(len(p["documents"]) for p in record["publications"])
        n_fb = sum(len(p["feedback"]) for p in record["publications"])
        print(
            f"ID {init_id}: {len(record['publications'])} sections, "
            f"{n_docs} docs, {n_fb} feedback"
        )
        print(json.dumps(record, indent=2, ensure_ascii=False))
    finally:
        fb_executor.shutdown(wait=False)
        pdf_executor.shutdown(wait=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape EU 'Have Your Say' initiative details."
    )
    parser.add_argument(
        "initiative_id", nargs="?", type=int, default=None,
        help="Scrape a single initiative by ID and print JSON to stdout.",
    )
    parser.add_argument(
        "-o", "--out-dir", type=str, default=None,
        help="Output directory for per-initiative JSON files. "
             "Defaults to initiative_details/.",
    )
    args = parser.parse_args()

    if args.initiative_id is not None:
        scrape_one(args.initiative_id)
    else:
        main(out_dir=args.out_dir)
