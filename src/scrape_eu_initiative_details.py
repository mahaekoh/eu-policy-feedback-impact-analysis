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
from typing import Optional

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

# Publication document cache directory (set via --cache-dir)
_doc_cache_dir: Optional[Path] = None

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
            "doc_id": doc_id,
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
EXTRACTABLE_EXTENSIONS = {".pdf", ".docx", ".doc", ".rtf", ".odt", ".txt"}

# Extensions where we try PDF extraction first before the native pipeline
# (many uploads are mislabeled PDFs)
TRY_PDF_FIRST_EXTENSIONS = {".doc", ".docx", ".odt", ".rtf"}


def _sanitize_filename(name: str) -> str:
    """Replace non-alphanumeric/dot/dash/underscore chars with underscore."""
    return re.sub(r"[^\w.\-]", "_", name)


def _cache_path(init_id, pub_id, doc_id, filename: str) -> Optional[Path]:
    """Return the cache file path, or None if caching is disabled."""
    if _doc_cache_dir is None:
        return None
    safe_name = _sanitize_filename(filename)
    return _doc_cache_dir / str(init_id) / f"pub{pub_id}_doc{doc_id}_{safe_name}"


def _download_or_cache(download_url: str, init_id, pub_id, doc_id, filename: str, label: str = "") -> bytes:
    """Download file bytes, using cache if available."""
    path = _cache_path(init_id, pub_id, doc_id, filename)
    if path is not None and path.exists():
        print(f"  CACHE HIT: {label}")
        return path.read_bytes()

    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    return data


def _extract_pdf_from_bytes(data: bytes, label: str = "") -> str:
    """Extract text from PDF bytes using pymupdf4llm with OCR fallback."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.write(tmp_fd, data)
        os.close(tmp_fd)
        try:
            text = pymupdf4llm.to_markdown(tmp_path)
        except (ValueError, Exception) as exc:
            print(f"  PDF markdown failed ({exc}), falling back to plain text: {label}")
            doc = pymupdf.open(tmp_path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()

        stripped = text.strip() if text else ""
        if len(stripped) < OCR_MIN_CHARS and len(data) > OCR_MIN_FILE_BYTES:
            print(f"  PDF text too short ({len(stripped)} chars, {len(data)} bytes), falling back to OCR: {label}")
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


def _extract_docx_from_bytes(data: bytes, suffix: str, label: str = "") -> str:
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
            # .doc (old binary format) — use macOS textutil
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


def _extract_pandoc_from_bytes(data: bytes, suffix: str, label: str = "") -> str:
    """Extract text from RTF/ODT bytes using pypandoc."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, data)
        os.close(tmp_fd)
        return pypandoc.convert_file(tmp_path, "markdown")
    finally:
        os.unlink(tmp_path)


def _extract_native_from_bytes(data: bytes, ext: str, label: str = "") -> str:
    """Run the format-specific extraction pipeline on already-downloaded bytes."""
    if ext == ".pdf":
        return _extract_pdf_from_bytes(data, label)
    elif ext in (".docx", ".doc"):
        return _extract_docx_from_bytes(data, ext, label)
    elif ext == ".txt":
        return data.decode("utf-8", errors="replace")
    elif ext in (".rtf", ".odt"):
        return _extract_pandoc_from_bytes(data, ext, label)
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def extract_from_bytes(data: bytes, filename: str, label: str = "") -> str:
    """Extract text from already-downloaded file bytes.

    For .doc/.docx/.odt/.rtf files, tries PDF extraction first (many uploads
    are mislabeled PDFs), then falls back to the format-specific pipeline.
    """
    t0 = time.time()
    ext = Path(filename).suffix.lower()

    # If the bytes are actually a PDF, always extract as PDF regardless of extension
    if ext != ".pdf" and data[:5] == b"%PDF-":
        try:
            text = _extract_pdf_from_bytes(data, label)
            elapsed = time.time() - t0
            print(f"  PDF-reinterpret ({elapsed:.1f}s, {len(data)} bytes): {label}")
            return text
        except Exception:
            pass  # PDF parsing failed, fall through to native

    # For non-PDF extensions that are often mislabeled PDFs, try PDF first
    # (catches cases where the file is a PDF but doesn't start with %PDF- header,
    # e.g. leading whitespace or BOM)
    if ext in TRY_PDF_FIRST_EXTENSIONS and data[:5] != b"%PDF-":
        try:
            text = _extract_pdf_from_bytes(data, label)
            if len((text or "").strip()) >= OCR_MIN_CHARS:
                elapsed = time.time() - t0
                print(f"  PDF-reinterpret ({elapsed:.1f}s, {len(data)} bytes): {label}")
                return text
        except Exception:
            pass  # not a PDF, fall through to native

    # Native extraction for the declared format
    text = _extract_native_from_bytes(data, ext, label)
    elapsed = time.time() - t0
    fmt = ext.lstrip(".").upper() or "FILE"
    print(f"  {fmt} extracted ({elapsed:.1f}s, {len(data)} bytes): {label}")
    return text


def download_and_extract(download_url: str, filename: str, label: str = "") -> str:
    """Download a file and extract text.

    For .doc/.docx/.odt/.rtf files, tries PDF extraction first (many uploads
    are mislabeled PDFs), then falls back to the format-specific pipeline.
    Downloads the file once and reuses the bytes for both attempts.
    """
    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    return extract_from_bytes(data, filename, label)


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
                    att_record["extracted_text"] = download_and_extract(
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


def _extract_text_for_doc(doc: dict, pub_id: int, init_id=None):
    """Download and extract text for a single publication document. Mutates doc in place."""
    label = f"pub={pub_id} {doc['filename']}"
    try:
        data = _download_or_cache(
            doc["download_url"], init_id, pub_id,
            doc.get("doc_id", ""), doc["filename"], label=label,
        )
        doc["extracted_text"] = extract_from_bytes(data, doc["filename"], label=label)
    except Exception as exc:
        doc["extracted_text_error"] = str(exc)
        print(f"  EXTRACT ERROR pub={pub_id} {doc['filename']}: {exc}")


def extract_publications(
    pubs: list, initiative_url: str, fb_executor: ThreadPoolExecutor,
    pdf_executor: ThreadPoolExecutor = None, init_id=None,
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
                    _pdf_pool.submit(_extract_text_for_doc, doc, pub_id, init_id)
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
            data.get("publications", []), url, fb_executor, pdf_executor,
            init_id=init_id,
        ),
    }


def _retry_extraction_errors(out_path: Path):
    """Scan existing initiative JSONs for extraction errors and retry them.

    Looks for documents and feedback attachments that have extracted_text_error
    but no extracted_text. Retries download_and_extract on each. On success,
    sets extracted_text and removes extracted_text_error. Saves updated files.
    """
    # Collect all (file_path, initiative_data) with extraction errors
    retry_items = []  # (file_path, initiative_data, error_locations)
    for p in sorted(out_path.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if "error" in data:
            continue  # top-level error, skip entirely

        locations = []  # list of (description, obj_with_error, pub_id_or_None)
        for pub in data.get("publications", []):
            pub_id = pub.get("publication_id", "?")
            for doc in pub.get("documents", []):
                if "extracted_text_error" in doc and "extracted_text" not in doc:
                    locations.append((
                        f"pub={pub_id} doc={doc.get('filename', '?')}",
                        doc,
                        pub_id,
                    ))
            for fb in pub.get("feedback", []):
                fb_id = fb.get("id", "?")
                for att in fb.get("attachments", []):
                    if "extracted_text_error" in att and "extracted_text" not in att:
                        locations.append((
                            f"fb={fb_id} att={att.get('filename', '?')}",
                            att,
                            None,
                        ))
        if locations:
            retry_items.append((p, data, locations))

    if not retry_items:
        print("\nNo extraction errors to retry.")
        return

    total_errors = sum(len(locs) for _, _, locs in retry_items)
    print(f"\nRetrying {total_errors} extraction errors across {len(retry_items)} files...")

    fixed_total = 0
    failed_total = 0

    def _retry_one(desc, obj, init_id, pub_id):
        """Retry extraction for a single document/attachment. Mutates obj."""
        filename = obj.get("filename", "")
        download_url = obj.get("download_url", "")
        if not download_url or not filename:
            return False
        ext = Path(filename).suffix.lower()
        if ext not in EXTRACTABLE_EXTENSIONS:
            return False
        label = f"retry init={init_id} {desc}"
        try:
            if pub_id is not None and "doc_id" in obj:
                data = _download_or_cache(
                    download_url, init_id, pub_id,
                    obj.get("doc_id", ""), filename, label=label,
                )
                text = extract_from_bytes(data, filename, label=label)
            else:
                text = download_and_extract(download_url, filename, label=label)
            obj["extracted_text"] = text
            old_error = obj.pop("extracted_text_error", None)
            obj["retry_old_error"] = old_error
            return True
        except Exception as exc:
            obj["extracted_text_error"] = str(exc)
            print(f"  RETRY FAILED {label}: {exc}")
            return False

    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as pool:
        for file_path, data, locations in retry_items:
            init_id = data.get("id", file_path.stem)
            futures = []
            for desc, obj, pub_id in locations:
                futures.append(pool.submit(_retry_one, desc, obj, init_id, pub_id))

            fixed = sum(1 for fut in futures if fut.result())
            failed = len(futures) - fixed
            fixed_total += fixed
            failed_total += failed

            if fixed > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  init={init_id}: {fixed} fixed, {failed} still failing — saved")
            elif failed > 0:
                print(f"  init={init_id}: {failed} still failing")

    print(f"Retry complete: {fixed_total} fixed, {failed_total} still failing")


def _fix_pdf_as_text(out_path: Path):
    """Fix feedback attachments where extracted_text contains raw PDF data.

    Some attachments (e.g. .txt extension) are actually PDFs. If the original
    extraction decoded raw PDF bytes as UTF-8 text, the extracted_text starts
    with '%PDF'. Re-downloads and extracts as PDF.
    """
    fix_items = []  # (file_path, initiative_data, locations)
    for p in sorted(out_path.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if "error" in data:
            continue

        locations = []  # (description, att_dict)
        for pub in data.get("publications", []):
            for fb in pub.get("feedback", []):
                fb_id = fb.get("id", "?")
                for att in fb.get("attachments", []):
                    text = att.get("extracted_text", "")
                    if text.startswith("%PDF"):
                        locations.append((
                            f"fb={fb_id} att={att.get('filename', '?')}",
                            att,
                        ))
        if locations:
            fix_items.append((p, data, locations))

    if not fix_items:
        print("\nNo PDF-as-text feedback attachments to fix.")
        return

    total = sum(len(locs) for _, _, locs in fix_items)
    print(f"\nFixing {total} feedback attachments with PDF-as-text across {len(fix_items)} files...")

    fixed_total = 0
    failed_total = 0

    def _fix_one(desc, att, init_id):
        download_url = att.get("download_url", "")
        if not download_url:
            return False
        label = f"pdf-as-text init={init_id} {desc}"
        try:
            req = urllib.request.Request(download_url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            text = _extract_pdf_from_bytes(raw, label)
            att["extracted_text_before_pdf_fix"] = att["extracted_text"]
            att["extracted_text"] = text
            print(f"  FIXED: {label}")
            return True
        except Exception as exc:
            print(f"  FIX FAILED {label}: {exc}")
            return False

    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as pool:
        for file_path, data, locations in fix_items:
            init_id = data.get("id", file_path.stem)
            futures = []
            for desc, att in locations:
                futures.append(pool.submit(_fix_one, desc, att, init_id))

            fixed = sum(1 for fut in futures if fut.result())
            failed = len(futures) - fixed
            fixed_total += fixed
            failed_total += failed

            if fixed > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  init={init_id}: {fixed} fixed, {failed} failed — saved")
            elif failed > 0:
                print(f"  init={init_id}: {failed} could not fix")

    print(f"PDF-as-text fix complete: {fixed_total} fixed, {failed_total} failed")


def main(out_dir: str = None, cache_dir: str = None):
    global _doc_cache_dir
    if cache_dir:
        _doc_cache_dir = Path(cache_dir)
        _doc_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Document cache: {_doc_cache_dir}")

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

    # --- Retry extraction errors in existing files ---
    _retry_extraction_errors(out_path)

    # --- Fix feedback attachments where extracted_text is raw PDF bytes ---
    _fix_pdf_as_text(out_path)

    # Summary
    total_written = len(list(out_path.glob("*.json")))
    errors = 0
    for p in out_path.glob("*.json"):
        with open(p, encoding="utf-8") as f:
            if "error" in json.load(f):
                errors += 1
    print(f"\nDone. {total_written} initiatives saved to {out_path} ({errors} errors)")


def scrape_one(init_id: int, cache_dir: str = None):
    """Scrape a single initiative by ID and print the result as JSON."""
    global _doc_cache_dir
    if cache_dir:
        _doc_cache_dir = Path(cache_dir)
        _doc_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Document cache: {_doc_cache_dir}")

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
    parser.add_argument(
        "-c", "--cache-dir", type=str, default=None,
        help="Directory to cache downloaded publication document files. "
             "When set, raw files are saved and reused on subsequent runs.",
    )
    args = parser.parse_args()

    if args.initiative_id is not None:
        scrape_one(args.initiative_id, cache_dir=args.cache_dir)
    else:
        main(out_dir=args.out_dir, cache_dir=args.cache_dir)
