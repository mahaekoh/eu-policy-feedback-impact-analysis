# Technical Documentation

This document covers the architecture, dependencies, data schemas, and usage of every component in the pipeline. For a research-oriented overview, see [README.md](README.md).

## Architecture

```
                            SCRAPING
                            --------
  EU Better Regulation API
          |
          v
  scrape_eu_initiatives.py ---------> eu_initiatives.csv
          |
          v
  scrape_eu_initiative_details.py --> initiative_details/*.json
          |                            (1,785 files, ~6.8 GB)
          |  text extraction:
          |  PDF (pymupdf4llm + tesseract fallback)
          |  DOCX (docx2md), DOC (textutil)
          |  RTF/ODT (pypandoc), TXT (direct)
          |
          |
                       DATA ENRICHMENT
                       ---------------
          |
          v
  find_short_pdf_extractions.py ----> short_pdf_report.json + short_pdfs/
          |
          v
  ocr_short_pdfs.py ----------------> short_pdf_report_ocr.json
          |                            (EasyOCR + CUDA, 300 DPI)
          v
  merge_ocr_results.py -------------> initiative_details/*.json (updated in-place)
          |
          |
                        TRANSLATION
                        -----------
          |
          v
  find_non_english_feedback_attachments.py -> non_english_attachments.json
          |
          v
  translate_attachments.py ----------> non_english_attachments_translated.json
          |                            (vLLM + unsloth/gpt-oss-120b)
          v
  merge_translations.py ------------> initiative_details/*.json (updated in-place)
          |
          |
                         ANALYSIS
                         --------
          |
          v
  initiative_stats.py -o -----------> before_after_analysis_v2/*.json
          |                            (128 initiative files with before/after structure)
          v
  summarize_documents.py -----------> summaries_output/*.json
                                       (summary fields added to documents & attachments)
```

## Prerequisites

### Python packages

| Package | Purpose |
|---------|---------|
| `pymupdf` / `pymupdf4llm` | PDF text extraction (markdown output, OCR fallback via tesseract at 300 DPI) |
| `docx2md` | DOCX text extraction |
| `pypandoc` / `pypandoc_binary` | RTF and ODT text extraction |
| `easyocr` | GPU-accelerated OCR (used with CUDA) |
| `vllm` | LLM batch inference engine |
| `openai_harmony` | Structured prompt encoding for gpt-oss models |
| `torch` | GPU acceleration (CUDA) |
| `numpy` | Array handling for OCR image processing |

### System requirements

- **Scraping + text extraction**: No GPU needed. Runs on any machine with Python 3.10+. macOS required for `.doc` extraction (uses `textutil`).
- **OCR stage**: NVIDIA GPU with CUDA (designed for H100, works on any CUDA GPU).
- **Translation + summarization**: NVIDIA GPU with enough VRAM to run `unsloth/gpt-oss-120b` via vLLM with tensor parallelism.

## Pipeline Stages

### Stage 1: Scraping

#### `scrape_eu_initiatives.py`

Enumerates all initiatives from the Better Regulation API.

```bash
python3 src/scrape_eu_initiatives.py
```

- **API**: `GET /brpapi/searchInitiatives` with date filters (Dec 2019 - Nov 2024)
- **Pagination**: page size 10, 0.3s delay between pages, 3 retries on failure
- **Output**: `eu_initiatives.csv` with columns: `id`, `reference`, `short_title`, `initiative_status`, `act_type`, `feedback_status`, `feedback_start`, `feedback_end`, `topics`, `url`

#### `scrape_eu_initiative_details.py`

Fetches complete data for each initiative: publications, documents, feedback, and attachments with full text extraction.

```bash
# Scrape all initiatives into per-file JSONs (supports resume)
python3 src/scrape_eu_initiative_details.py -o initiative_details/

# Scrape a single initiative to stdout
python3 src/scrape_eu_initiative_details.py 12970
```

- **APIs used**:
  - `GET /brpapi/groupInitiatives/{id}` — initiative detail with publications
  - `GET /api/allFeedback?publicationId={id}` — feedback (paginated, 500 per page)
  - `GET /brpapi/download/{id}` — publication document download
  - `GET /api/download/{id}` — feedback attachment download
- **Thread pools**: 20 initiative workers, 20 feedback workers, 40 PDF extraction workers
- **Text extraction chain**:
  1. `.pdf` → `pymupdf4llm.to_markdown()` → fallback to `pymupdf` plain text → if < 100 chars and file > 2 KB, OCR via tesseract at 300 DPI
  2. `.docx` → `docx2md`
  3. `.doc` → macOS `textutil -convert txt`
  4. `.rtf`, `.odt` → `pypandoc.convert_file()` to markdown
  5. `.txt` → direct read (UTF-8 with error replacement)
  6. `.zip` → skipped
- **Resume**: skips initiative IDs that already have a JSON file in the output directory
- **Output**: one JSON file per initiative in `initiative_details/`

### Stage 2: Data Enrichment (OCR)

#### `find_short_pdf_extractions.py`

Identifies PDFs where text extraction produced suspiciously little text (likely scanned/image PDFs).

```bash
python3 src/find_short_pdf_extractions.py initiative_details/ \
    -f initiative-whitelist-145.txt \
    -p short_pdfs/ \
    -o short_pdf_report.json
```

- **Threshold**: extracted text < 100 characters AND file size > 2 KB
- **Scans**: both publication documents and feedback attachments
- **Downloads**: PDFs in parallel (20 workers) to `short_pdfs/`
- **Output**: `short_pdf_report.json` with metadata and download paths

#### `ocr_short_pdfs.py`

Runs GPU-accelerated OCR on the problematic PDFs.

```bash
python3 src/ocr_short_pdfs.py short_pdf_report.json short_pdfs/ -o short_pdf_report_ocr.json
```

- **Process**: PDF → render pages at 300 DPI via pymupdf → numpy arrays → EasyOCR with paragraph grouping
- **GPU**: uses CUDA if available, falls back to CPU
- **Languages**: English by default, configurable via `--languages`
- **Output**: same JSON structure with `ocr_text` and `ocr_text_chars` added

#### `merge_ocr_results.py`

Merges OCR text back into the initiative detail files.

```bash
# Preview changes
python3 src/merge_ocr_results.py short_pdf_report_ocr.json initiative_details/ --dry-run

# Apply
python3 src/merge_ocr_results.py short_pdf_report_ocr.json initiative_details/
```

- **Behavior**: replaces `extracted_text` with OCR result, saves original as `extracted_text_without_ocr`
- **Lookup**: matches by initiative ID, publication ID, download URL (for documents) or feedback/attachment ID (for feedback attachments)

### Stage 3: Translation

#### `find_non_english_feedback_attachments.py`

Finds feedback attachments where the feedback language is not English.

```bash
# Print summary to console
python3 src/find_non_english_feedback_attachments.py initiative_details/ \
    -f initiative-whitelist-145.txt

# Write JSON for translation pipeline
python3 src/find_non_english_feedback_attachments.py initiative_details/ \
    -f initiative-whitelist-145.txt \
    -o non_english_attachments.json
```

- **Filter**: `feedback.language != "EN"` and attachment has extractable text
- **Output**: flat list of attachment records with full metadata and extracted text

#### `translate_attachments.py`

Translates non-English text to English using LLM batch inference.

```bash
python3 src/translate_attachments.py non_english_attachments.json \
    -o non_english_attachments_translated.json \
    --batch-size 32
```

- **Model**: `unsloth/gpt-oss-120b` via vLLM
- **Prompt encoding**: `openai_harmony` with `ReasoningEffort.MEDIUM`
- **Chunking**: long texts split at sentence boundaries (default 5,000 chars per chunk via `text_utils.split_into_chunks`)
- **"NO TRANSLATION NEEDED"**: if the model returns this for a chunk, the original text is preserved for that chunk during reassembly
- **Deduplication**: identical text chunks across records are translated only once (cross-batch cache)
- **Resume**: completed batch files are loaded from disk instead of re-running inference
- **Batch output**: per-batch JSON files in `{output_base}_batches/` for incremental progress
- **Output**: copy of input JSON with `extracted_text_translated` added to each record

#### `merge_translations.py`

Merges translations back into initiative detail files.

```bash
# From combined output
python3 src/merge_translations.py non_english_attachments_translated.json initiative_details/

# From batch directory
python3 src/merge_translations.py translated_batches/ initiative_details/

# Dry run
python3 src/merge_translations.py translated_batches/ initiative_details/ --dry-run
```

- **Behavior**: replaces `extracted_text` with translated text, saves original as `extracted_text_before_translation`
- **Skip logic**: records containing "NO TRANSLATION NEEDED" are not merged
- **Input modes**: accepts either the combined JSON file or the batch directory directly
- **Publication ID resolution**: for older batch formats missing `publication_id`, looks it up from `--input-records` or by searching the initiative files

### Stage 4: Analysis

#### `initiative_stats.py`

Analyses the publication/feedback timeline for selected initiatives and creates the before/after data structure.

```bash
# Console stats only
python3 src/initiative_stats.py initiative-whitelist-145.txt initiative_details/

# Write enriched JSONs
python3 src/initiative_stats.py initiative-whitelist-145.txt initiative_details/ \
    -o before_after_analysis_v2/
```

- **Input**: whitelist of initiative IDs + initiative detail directory
- **Timeline logic**:
  - **Pre-feedback publications**: all publications up to and including the first one that received feedback
  - **Final publication**: last non-`OPC_LAUNCHED` publication that has documents (falls back to last publication if none)
  - Only outputs initiatives where documents exist after the first feedback publication
- **Output fields added to initiative JSON**:
  - `documents_before_feedback` — documents from pre-feedback publications
  - `documents_after_feedback` — documents from the final publication
  - `middle_feedback` — all feedback from publications between the first feedback pub and the final pub (excludes feedback on the final publication itself)
- **Console reports**: publication type breakdown, initiatives with no documents after feedback, initiatives with feedback only on the final publication

#### `summarize_documents.py`

Summarizes publication documents and feedback attachments using LLM batch inference.

```bash
python3 src/summarize_documents.py before_after_analysis_v2/ \
    -o summaries_output/ \
    --batch-size 16 \
    --initiative-batch-size 10
```

- **Model**: `unsloth/gpt-oss-120b` via vLLM
- **Recursive summarization**:
  - **Pass 1**: each text chunk is summarized independently
  - **Combine**: for multi-chunk items, chunk summaries are recursively combined in groups of up to `--max-combine-chunks` (default 4) until a single summary remains
- **Prompt structure**:
  - System identity: "You are a policy analyst who summarizes EU regulatory documents clearly and concisely"
  - Pass 1 (chunk summarization):
    - Publication documents: "Summarize it into a text up to 10 paragraphs."
    - Feedback attachments: "Summarize it into a text up to 10 paragraphs."
  - Combine (recursive merging):
    - Publication documents: "Combine them into a single summary up to 10 paragraphs."
    - Feedback attachments: "Combine them into a single summary up to 10 paragraphs."
  - All prompts include: "Be as specific and detailed as possible. If any, preserve all points about nuclear energy, nuclear plants, or small modular reactors. Do not generate any mete commentary (for example stating that there are no nuclear-related points)."
- **Max output tokens**: 131,072 (`32768 * 4`) per inference call
- **Chunking**: sentence-boundary splits at 5,000 chars (configurable via `--chunk-size`)
- **Initiative batching**: processes files in groups (default 10) to manage memory
- **Deduplication + resume**: same as translation pipeline (cross-batch cache, batch file resume)
- **Output**: initiative JSONs with `summary` field added to each document and attachment

### Utilities

#### `text_utils.py`

Shared library with `split_into_chunks(text, max_chars)`. Splits at sentence boundaries (`.!?` followed by whitespace). Falls back to newline splits for sentences exceeding `max_chars`.

#### `find_missing_initiatives.py`

Reports initiative IDs present in the CSV but missing from the detail files, and initiatives with `feedback_error` (API 400 responses).

```bash
python3 src/find_missing_initiatives.py
```

#### `find_missing_extracted_text.py`

Scans initiative data for documents and attachments that have no `extracted_text`. Supports `-f` for whitelist filtering.

```bash
python3 src/find_missing_extracted_text.py initiative_details/ -f initiative-whitelist-145.txt
```

#### `find_initiative_by_pub.py`

Lookup utility to find which initiative contains a given publication ID.

```bash
python3 src/find_initiative_by_pub.py 15688 initiative_details/
```

#### `print_chunk.py`

Debug utility to print a specific text chunk from a feedback attachment.

```bash
python3 src/print_chunk.py "init=12096 fb=503089 att=6276475 chunk=5/15" initiative_details/
```

## Data Schema

### Initiative JSON (`initiative_details/{id}.json`)

```json
{
  "id": 12970,
  "url": "https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/12970-...",
  "short_title": "EU school fruit, vegetables and milk scheme - review",
  "summary": "...",
  "reference": "Ares(2021)...",
  "type_of_act": "Regulation",
  "type_of_act_code": "PROP_REG",
  "department": "AGRI",
  "status": "ADOPTED",
  "stage": "...",
  "topics": ["Agriculture"],
  "policy_areas": ["..."],
  "published_date": "2021/06/29 ...",
  "publications": [
    {
      "publication_id": 15000,
      "type": "CFE_IMPACT_ASSESS",
      "section_label": "Call for evidence",
      "reference": "...",
      "published_date": "2021/06/29 ...",
      "feedback_end_date": "2021/07/27 ...",
      "feedback_period_weeks": 4,
      "feedback_status": "CLOSED",
      "total_feedback": 74,
      "documents": [
        {
          "label": "Call for evidence - Ares(2021)...",
          "download_url": "https://ec.europa.eu/.../download/...",
          "filename": "PART-2021-....pdf",
          "title": "...",
          "reference": "...",
          "doc_type": "MAIN",
          "category": "...",
          "pages": 5,
          "size_bytes": 250000,
          "extracted_text": "# Full markdown text...",
          "extracted_text_without_ocr": "...",
          "summary": "..."
        }
      ],
      "feedback": [
        {
          "id": 503089,
          "url": "https://ec.europa.eu/.../F503089_en",
          "date": "2021/07/27 ...",
          "feedback_text": "Free-text comment...",
          "feedback_text_original": "...",
          "language": "EN",
          "user_type": "COMPANY",
          "country": "SWE",
          "company_size": "LARGE",
          "organization": "Oatly AB",
          "first_name": "...",
          "surname": "...",
          "status": "PUBLISHED",
          "publication": "...",
          "tr_number": "...",
          "attachments": [
            {
              "id": 6276475,
              "filename": "Oatly_position.pdf",
              "document_id": "...",
              "download_url": "https://ec.europa.eu/.../download/...",
              "pages": 3,
              "size_bytes": 120000,
              "extracted_text": "Full text (translated if needed)...",
              "extracted_text_before_translation": "Original non-English text...",
              "extracted_text_without_ocr": "Original pre-OCR text...",
              "summary": "..."
            }
          ]
        }
      ]
    }
  ]
}
```

### Before/after analysis JSON (`before_after_analysis_v2/{id}.json`)

Same structure as the initiative JSON, with three additional top-level fields:

| Field | Type | Description |
|-------|------|-------------|
| `documents_before_feedback` | `list[document]` | Documents from publications up to and including the first one with feedback |
| `documents_after_feedback` | `list[document]` | Documents from the final publication |
| `middle_feedback` | `list[feedback]` | All feedback between the first feedback publication and the final document publication |

## Configuration Files

| File | Contents |
|------|----------|
| `initiative-whitelist-145.txt` | One initiative ID per line. Used by `initiative_stats.py` and the `-f` filter on other scripts. |
| `initiatives-with-no-euc-response-after-feedback.txt` | Similar tracking list for non-responsive initiatives. |

## Running the Full Pipeline

```bash
# 1. Scrape initiative list
python3 src/scrape_eu_initiatives.py

# 2. Scrape all initiative details (resumable, ~6.8 GB output)
python3 src/scrape_eu_initiative_details.py -o initiative_details/

# 3. Find and download PDFs with failed text extraction
python3 src/find_short_pdf_extractions.py initiative_details/ \
    -f initiative-whitelist-145.txt \
    -p short_pdfs/ \
    -o short_pdf_report.json

# 4. Run GPU OCR on those PDFs
python3 src/ocr_short_pdfs.py short_pdf_report.json short_pdfs/ \
    -o short_pdf_report_ocr.json

# 5. Merge OCR results back
python3 src/merge_ocr_results.py short_pdf_report_ocr.json initiative_details/

# 6. Find non-English feedback attachments
python3 src/find_non_english_feedback_attachments.py initiative_details/ \
    -f initiative-whitelist-145.txt \
    -o non_english_attachments.json

# 7. Translate non-English attachments (GPU required)
python3 src/translate_attachments.py non_english_attachments.json \
    -o non_english_attachments_translated.json

# 8. Merge translations back
python3 src/merge_translations.py non_english_attachments_translated.json initiative_details/

# 9. Build before/after analysis structure
python3 src/initiative_stats.py initiative-whitelist-145.txt initiative_details/ \
    -o before_after_analysis_v2/

# 10. Summarize documents and feedback (GPU required)
python3 src/summarize_documents.py before_after_analysis_v2/ \
    -o summaries_output/
```

Steps 1-3 and 5-6 run on CPU. Steps 4, 7, and 10 require a GPU.

## Key Design Decisions

### Parallelism strategy

The detail scraper uses three separate thread pools (initiative, feedback, PDF extraction) to avoid deadlocks. The initiative pool spawns work into the feedback and PDF pools, so sharing a single pool would cause workers waiting on sub-tasks that can never be scheduled.

### Text chunking for LLM inference

Both translation and summarization split long texts at sentence boundaries (default 5,000 characters). The `text_utils.split_into_chunks` function tries `.!?`-followed-by-whitespace splits first, falling back to newline splits if a single sentence exceeds the limit. This keeps semantic units intact while staying within context windows.

### Recursive summarization

For documents spanning multiple chunks, pass 1 summarizes each chunk independently, then chunk summaries are recursively combined in groups of up to 4 (configurable via `--max-combine-chunks`) until a single summary remains. This avoids context window overflow while maintaining global coherence, even for very long documents with many chunks.

### In-place merge pattern

OCR results and translations are generated as separate files, then merged back into `initiative_details/*.json`. The merge scripts preserve original text as `extracted_text_without_ocr` or `extracted_text_before_translation`, making it possible to audit or revert enrichments. All merge scripts support `--dry-run`.

### Deduplication and resume

Translation and summarization pipelines write per-batch result files. On restart, completed batches are loaded from disk. Identical text chunks across different records are cached and only processed once (cross-batch dedup).

### Publication type mapping

The scraper maps EU API publication type codes (e.g. `CFE_IMPACT_ASSESS`, `PROP_REG`, `DEL_REG_DRAFT`) to human-readable labels (e.g. "Call for evidence", "Commission adoption", "Draft act"). The full mapping is in `scrape_eu_initiative_details.py`.
