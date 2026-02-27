# Technical Documentation

This document covers the architecture, dependencies, data schemas, and usage of every component in the pipeline. For a research-oriented overview, see [README.md](README.md).

## Architecture

All data lives under `data/`. The pipeline is orchestrated by `pipeline.sh` (see `./pipeline.sh list`).

```
                            SCRAPING
                            --------
  EU Better Regulation API
          |
          v
  scrape_eu_initiatives.py ---------> data/scrape/eu_initiatives.csv
          |                            (parallel page fetching, 10 workers)
          v
  scrape_eu_initiative_details.py --> data/scrape/initiative_details/*.json
          |                            (1,785 files, ~6.8 GB)
          |  text extraction:
          |  PDF (pymupdf4llm + tesseract fallback)
          |  DOCX (docx2md), DOC (textutil)
          |  RTF/ODT (pypandoc), TXT (direct)
          |  For .doc/.docx/.odt/.rtf: tries PDF first (many are mislabeled)
          |
          |
                          OCR ENRICHMENT
                          --------------
          |
          v
  find_short_pdf_extractions.py ----> data/ocr/short_pdf_report.json + data/ocr/pdfs/
          |
          v
  ocr_short_pdfs.py ----------------> data/ocr/short_pdf_report_ocr.json
          |                            (EasyOCR + CUDA, 300 DPI)
          v
  merge_ocr_results.py -------------> data/scrape/initiative_details/*.json (updated in-place)
          |
          |
                        TRANSLATION
                        -----------
          |
          v
  find_non_english_feedback_attachments.py -> data/translation/non_english_attachments.json
          |                                    (runs after OCR merge for complete text)
          v
  translate_attachments.py ----------> data/translation/non_english_attachments_translated.json
          |                            (vLLM + unsloth/gpt-oss-120b)
          v
  merge_translations.py ------------> data/scrape/initiative_details/*.json (updated in-place)
          |
          |
                         ANALYSIS
                         --------
          |
          v
  initiative_stats.py -o -----------> data/analysis/before_after/*.json
          |                            (all initiatives with feedback, before/after structure)
          v
  summarize_documents.py -----------> data/analysis/summaries/*.json
          |                            (summary fields added to documents & attachments)
          v
  build_unit_summaries.py ----------> data/analysis/unit_summaries/*.json
          |                            (unified per-initiative summary fields)
          v
  summarize_changes.py ------------> data/analysis/change_summaries/*.json
          |                            (change_summary field: before vs after diff)
          |
          |
                        CLUSTERING
                        ----------
          |
          v
  cluster_all_initiatives.py -------> data/clustering/<scheme>/*.json
          |
          v
  summarize_clusters.py ------------> data/cluster_summaries/<scheme>/*.json
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

- **API**: `GET /brpapi/searchInitiatives` (no date filter — fetches everything available)
- **Pagination**: page size 10, 10 parallel workers, 3 retries on failure
- **Output**: `data/scrape/eu_initiatives.csv` with columns: `id`, `reference`, `short_title`, `initiative_status`, `act_type`, `feedback_status`, `feedback_start`, `feedback_end`, `topics`, `url`

#### `scrape_eu_initiative_details.py`

Fetches complete data for each initiative: publications, documents, feedback, and attachments with full text extraction.

```bash
# Scrape all initiatives into per-file JSONs (supports resume)
python3 src/scrape_eu_initiative_details.py

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
  1. `.doc`, `.docx`, `.odt`, `.rtf` → tries PDF extraction first (many uploads are mislabeled PDFs); if the result is < 100 chars, falls back to the native pipeline below
  2. `.pdf` → `pymupdf4llm.to_markdown()` → fallback to `pymupdf` plain text → if < 100 chars and file > 2 KB, OCR via tesseract at 300 DPI
  3. `.docx` → `docx2md`
  4. `.doc` → macOS `textutil -convert txt`
  5. `.rtf`, `.odt` → `pypandoc.convert_file()` to markdown
  6. `.txt` → direct read (UTF-8 with error replacement)
  7. `.zip` → skipped
- **Resume**: skips initiative IDs that already have a JSON file in the output directory
- **Output**: one JSON file per initiative in `data/scrape/initiative_details/`

### Stage 2: Repair (optional, manual recovery only)

#### `repair_broken_attachments.py`

Scans initiative detail files for feedback attachments that have `extracted_text_error` and no `extracted_text`, downloads them, and retries extraction. This is largely redundant with the retry passes built into `scrape_eu_initiative_details.py` (`_retry_extraction_errors()` and `_fix_pdf_as_text()`), but can be useful for recovering from transient network failures after a scrape completes.

```bash
# Scan and report (dry run)
python3 src/repair_broken_attachments.py -o data/repair/ --dry-run

# Repair all
python3 src/repair_broken_attachments.py -o data/repair/
```

- **Strategy**: for `.doc/.docx/.odt/.rtf` files, tries PDF extraction first (since many are mislabeled PDFs), then falls back to the native format-specific pipeline
- **Output**: copies of initiative JSONs (only files with at least one repair) to the output directory, plus `repair_report.json`
- **Not in the main pipeline**: `pipeline.sh full` does not include this stage. Run it manually with `./pipeline.sh repair` if needed.

### Stage 3: Data Enrichment (OCR)

#### `find_short_pdf_extractions.py`

Identifies PDFs where text extraction produced suspiciously little text (likely scanned/image PDFs).

```bash
python3 src/find_short_pdf_extractions.py \
    -i data/scrape/initiative_details \
    -o data/ocr/
```

- **Threshold**: extracted text < 100 characters AND file size > 2 KB
- **Scans**: both publication documents and feedback attachments
- **Downloads**: PDFs in parallel (20 workers) to `data/ocr/pdfs/`
- **Output**: `data/ocr/short_pdf_report.json` with metadata and download paths

#### `ocr_short_pdfs.py`

Runs GPU-accelerated OCR on the problematic PDFs.

```bash
python3 src/ocr_short_pdfs.py data/ocr/
```

- **Process**: PDF → render pages at 300 DPI via pymupdf → numpy arrays → EasyOCR with paragraph grouping
- **GPU**: uses CUDA if available, falls back to CPU
- **Languages**: English by default, configurable via `--languages`
- **Output**: same JSON structure with `ocr_text` and `ocr_text_chars` added

#### `merge_ocr_results.py`

Merges OCR text back into the initiative detail files.

```bash
# Preview changes
python3 src/merge_ocr_results.py data/ocr/short_pdf_report_ocr.json data/scrape/initiative_details/ --dry-run

# Apply
python3 src/merge_ocr_results.py data/ocr/short_pdf_report_ocr.json data/scrape/initiative_details/
```

- **Behavior**: replaces `extracted_text` with OCR result, saves original as `extracted_text_without_ocr`
- **Lookup**: matches by initiative ID, publication ID, download URL (for documents) or feedback/attachment ID (for feedback attachments)

### Stage 4: Translation

#### `find_non_english_feedback_attachments.py`

Finds feedback attachments where the feedback language is not English.

```bash
# Print summary to console
python3 src/find_non_english_feedback_attachments.py data/scrape/initiative_details/

# Write JSON for translation pipeline
python3 src/find_non_english_feedback_attachments.py data/scrape/initiative_details/ \
    -o data/translation/non_english_attachments.json
```

- **Filter**: `feedback.language != "EN"` and attachment has extractable text
- **Output**: flat list of attachment records with full metadata and extracted text

#### `translate_attachments.py`

Translates non-English text to English using LLM batch inference.

```bash
python3 src/translate_attachments.py data/translation/non_english_attachments.json \
    -o data/translation/non_english_attachments_translated.json \
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
python3 src/merge_translations.py data/translation/non_english_attachments_translated.json data/scrape/initiative_details/

# From batch directory
python3 src/merge_translations.py data/translation/translation_batches/ data/scrape/initiative_details/

# Dry run
python3 src/merge_translations.py data/translation/translation_batches/ data/scrape/initiative_details/ --dry-run
```

- **Behavior**: replaces `extracted_text` with translated text, saves original as `extracted_text_before_translation`
- **Skip logic**: records containing "NO TRANSLATION NEEDED" are not merged
- **Input modes**: accepts either the combined JSON file or the batch directory directly
- **Publication ID resolution**: for older batch formats missing `publication_id`, looks it up from `--input-records` or by searching the initiative files

### Stage 5: Analysis

#### `initiative_stats.py`

Analyses the publication/feedback timeline for all initiatives in the details directory and creates the before/after data structure.

```bash
# Console stats only
python3 src/initiative_stats.py data/scrape/initiative_details/

# Write enriched JSONs
python3 src/initiative_stats.py data/scrape/initiative_details/ \
    -o data/analysis/before_after/
```

- **Input**: initiative detail directory (processes all `*.json` files)
- **Timeline logic**:
  - **Pre-feedback publications**: all publications up to and including the first one that received feedback
  - **Final publication**: last non-`OPC_LAUNCHED` publication that has documents (falls back to last publication if none)
  - Outputs all initiatives with feedback (including those with no post-feedback documents)
- **Output fields added to initiative JSON**:
  - `documents_before_feedback` — documents from pre-feedback publications
  - `documents_after_feedback` — documents from the final publication (empty list when no post-feedback documents exist)
  - `middle_feedback` — all feedback from publications between the first feedback pub and the final pub (excludes feedback on the final publication when post-feedback docs exist; includes all feedback otherwise)
- **Console reports**: publication type breakdown, initiatives with no documents after feedback, initiatives with feedback only on the final publication

#### `summarize_documents.py`

Summarizes publication documents and feedback attachments using LLM batch inference.

```bash
python3 src/summarize_documents.py data/analysis/before_after/ \
    -o data/analysis/summaries/ \
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

#### `build_unit_summaries.py`

Consolidates individual document and attachment summaries into per-initiative unified summary fields for downstream analysis.

```bash
python3 src/build_unit_summaries.py data/analysis/summaries/ -o data/analysis/unit_summaries/
```

- **Input**: output directory from `summarize_documents.py`
- **Output fields added to initiative JSON**:
  - `before_feedback_summary` — concatenation (joined by `\n\n`) of all `summary` fields from `documents_before_feedback`
  - `after_feedback_summary` — concatenation of all `summary` fields from `documents_after_feedback`
  - `combined_feedback_summary` — on each `middle_feedback` item: the `feedback_text` plus all attachment `summary` fields, concatenated
- **Stats**: reports the longest policy-level and feedback-level summaries across all initiatives

#### `summarize_changes.py`

Summarizes the substantive changes between before- and after-feedback documents using LLM batch inference.

```bash
python3 src/summarize_changes.py data/analysis/unit_summaries/ \
    -o data/analysis/change_summaries/ \
    --batch-size 16
```

- **Model**: `unsloth/gpt-oss-120b` via vLLM
- **Input**: output directory from `build_unit_summaries.py`
- **Filtering**: only processes initiatives that have both `before_feedback_summary` and `after_feedback_summary`; others are copied through unchanged
- **Diff**: computes `difflib.unified_diff` between the two summaries and includes it in the prompt alongside both full texts
- **Prompt structure**:
  - System identity: "You are a policy analyst who compares EU regulatory documents before and after public consultation feedback"
  - User prompt: before summary, after summary, and unified diff, followed by instructions to summarize substantive changes in up to 10 paragraphs
  - Includes nuclear energy preservation clause
- **Output**: initiative JSONs with `change_summary` field added at top level
- **Deduplication + resume**: same as other inference scripts (cross-batch cache, batch file resume)

### Webapp (`webapp/`)

A Next.js web application for browsing initiatives and feedback interactively.

#### Tech stack

| Package | Version | Purpose |
|---------|---------|---------|
| Next.js | 16.1.6 | App framework (App Router, server components) |
| React | 19.2.3 | UI rendering |
| Tailwind CSS | 4 | Styling (via `@tailwindcss/postcss`) |
| shadcn/ui | — | Component library (Radix UI + Lucide icons + CVA) |
| next-auth | 5.0.0-beta.30 | Google OAuth (JWT sessions, no database) |
| react-markdown | 10.1.0 | Markdown rendering for summaries |

#### Pages

| Route | File | Description |
|-------|------|-------------|
| `/` | `src/app/page.tsx` | Initiative index with search, sort, advanced filters, pagination (50/page) |
| `/initiative/[id]` | `src/app/initiative/[id]/page.tsx` | Initiative detail with publications/clusters views, summaries, timeline sparkline |

#### API routes

| Route | File | Description |
|-------|------|-------------|
| `/api/auth/[...nextauth]` | `src/app/api/auth/[...nextauth]/route.ts` | NextAuth OAuth handlers (Google sign-in/out) |
| `/api/clusters/[id]` | `src/app/api/clusters/[id]/route.ts` | Fetch clustering data for an initiative (requires `scheme` query param) |

#### Data loading (`src/lib/data.ts`)

All data is loaded server-side from `../data/` (relative to `webapp/`):

- **`getInitiativeIndex()`** — Reads all initiative JSON files and extracts metadata using regex (not full JSON parsing) for speed. Extracts feedback IDs to deduplicate initiatives that share identical feedback sets (different policy steps referencing the same consultation). Results are cached for 5 minutes.
- **`getInitiativeDetail(id)`** — Full JSON parse of a single initiative file.
- **`getClusteringSchemesForInitiative(id)`** — Scans `data/clustering/` subdirectories for files matching the initiative ID.
- **`getClusterData(id, scheme)`** — Loads cluster assignments for a specific scheme.

#### Components

**Layout:**
- `header.tsx` — Sticky nav bar with app title and user menu
- `user-menu.tsx` — Client component: Google sign-in button when unauthenticated, avatar + name + sign-out when authenticated

**Index page:**
- `initiative-list.tsx` — Client component with search, sort (most discussed / recently discussed / newest), advanced filters (stage, department, topic, policy area, open feedback only), and paginated card grid
- `initiative-card.tsx` — Card showing title, department badge, stage, feedback count, timeline sparkline, country/user-type distribution bars

**Detail page:**
- `initiative-detail.tsx` — Main detail view with metadata panel, summaries (document, change, before/after), and tabbed publications/clusters views. Filters out empty disabled publications into a compact summary row.
- `publication-section.tsx` — Expandable publication with document and feedback tabs
- `feedback-list.tsx` — Filterable feedback list with user-type chips, text search, hide-empty toggle, infinite scroll (50/chunk)
- `feedback-card.tsx` — Single feedback item with submitter info, text, attachments, extracted/OCR/translated text
- `document-card.tsx` — Document with metadata, download link, summary, extracted text

**Clustering:**
- `cluster-view.tsx` — Cluster explorer with scheme selector, sort options, timeline date filter, nested tree
- `cluster-node.tsx` — Expandable cluster node with item count, country/user-type stats bar, sparkline, feedback items
- `cluster-stats-bar.tsx` — Horizontal stacked bars for country and user-type distributions with tooltips

**Shared:**
- `expandable-text.tsx` — Collapsible text block with optional markdown rendering and configurable preview length

#### Authentication

Optional Google sign-in via Auth.js v5. JWT-based sessions — no database required. The app is fully accessible without signing in. See `webapp/AUTH.md` for Google Cloud Console setup and environment variables.

| File | Purpose |
|------|---------|
| `src/auth.ts` | NextAuth config with Google provider |
| `src/proxy.ts` | Re-exports `auth` as `proxy` for Next.js 16 middleware (session cookie refresh) |
| `.env.local` | `AUTH_SECRET`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET` (git-ignored) |

#### Running

```bash
cd webapp
npm install
npm run dev    # development server at http://localhost:3000
npm run build  # production build
```

The app reads data directly from `../data/scrape/initiative_details/` and `../data/clustering/` — no database or import step needed.

### Standalone viewers

#### `viewers/viewer.html`

Standalone HTML file (no dependencies) for interactively browsing per-initiative JSON files in the browser.

- **File loading**: browser file picker to load any initiative JSON
- **Tabbed navigation**: Before Feedback, After Feedback, Feedback, Publications
- **Document features**: download links, feedback portal links, attachment download links
- **Text display**: expandable blocks for summaries, extracted text, pre-translation originals, pre-OCR originals
- **Feedback features**: user type color coding, filtering by type/search/empty attachments, chunked infinite scroll for large feedback lists

#### `viewers/feedback-viewer.html`

Standalone HTML file (no dependencies) for browsing clustered feedback results. Loads per-initiative clustering JSON files (from `data/clustering/<scheme>/`) via browser file picker.

- **Cluster metadata**: algorithm, model, parameters, cluster count, noise count, silhouette score
- **Nested cluster tree**: expandable top-level clusters with sub-clusters, sorted by size or alphabetically
- **Per-cluster statistics**: country and user-type horizontal stacked bars with tooltips
- **Feedback items**: submitter info (name, organization, country flag), feedback text, attachments with extracted text
- **Search**: text filter across feedback text, organization, and country
- **Noise section**: unclustered feedback items shown separately

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
python3 src/find_missing_extracted_text.py data/scrape/initiative_details/
```

#### `find_initiative_by_pub.py`

Lookup utility to find which initiative contains a given publication ID.

```bash
python3 src/find_initiative_by_pub.py 15688 data/scrape/initiative_details/
```

#### `print_chunk.py`

Debug utility to print a specific text chunk from a feedback attachment.

```bash
python3 src/print_chunk.py "init=12096 fb=503089 att=6276475 chunk=5/15" data/scrape/initiative_details/
```

## Data Schema

### Initiative JSON (`data/scrape/initiative_details/{id}.json`)

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
              "extracted_text_error": "Error message if extraction failed...",
              "repair_method": "pdf-reinterpret or native (if repaired)",
              "repair_old_error": "Original error before repair...",
              "summary": "..."
            }
          ]
        }
      ]
    }
  ]
}
```

### Before/after analysis JSON (`data/analysis/before_after/{id}.json`)

Same structure as the initiative JSON, with three additional top-level fields:

| Field | Type | Description |
|-------|------|-------------|
| `documents_before_feedback` | `list[document]` | Documents from publications up to and including the first one with feedback |
| `documents_after_feedback` | `list[document]` | Documents from the final publication |
| `middle_feedback` | `list[feedback]` | All feedback between the first feedback publication and the final document publication |

### Unit summaries JSON (output of `build_unit_summaries.py`)

Same structure as the summaries output, with additional top-level and per-feedback fields:

| Field | Type | Description |
|-------|------|-------------|
| `before_feedback_summary` | `string` | Concatenation of all document summaries from before feedback |
| `after_feedback_summary` | `string` | Concatenation of all document summaries from after feedback |
| `middle_feedback[].combined_feedback_summary` | `string` | Feedback text + all attachment summaries, concatenated |

## Configuration Files

| File | Contents |
|------|----------|
| `pipeline.conf` | Pipeline orchestration config (remote host, SSH key, clustering schemes). Copy from `pipeline.conf.example`. |

## Running the Full Pipeline

The pipeline is orchestrated by `pipeline.sh`. First, copy `pipeline.conf.example` to `pipeline.conf` and fill in your remote host details.

```bash
# Run the full end-to-end pipeline
./pipeline.sh full

# Or run individual stages
./pipeline.sh scrape                   # scrape initiatives + details
./pipeline.sh find-short-pdfs          # find PDFs needing OCR
./pipeline.sh deploy                   # rsync code to remote GPU host
./pipeline.sh remote ocr               # run OCR on remote
./pipeline.sh pull ocr                 # pull OCR results back
./pipeline.sh merge-ocr                # merge OCR into initiative_details

# Monitor remote jobs
./pipeline.sh logs                     # list recent remote logs
./pipeline.sh logs tail                # tail most recent log
./pipeline.sh logs tail summarize      # tail most recent summarize log
./pipeline.sh logs ocr                 # shorthand for logs tail ocr

# See all available stages
./pipeline.sh list
```

### SSH resilience

Remote commands (`./pipeline.sh remote <step>`) run via `nohup` on the remote host with stdout/stderr captured to timestamped log files under `logs/`. This ensures long-running GPU jobs (OCR, translation, summarization) survive SSH disconnects. The local terminal tails the remote log in real-time and polls for a status file to detect completion. If the SSH session drops, the remote process continues — reconnect and use `./pipeline.sh logs` to monitor progress.

The `full` pipeline runs in this order:

1. **Scrape** — fetch initiative list + details (CPU)
2. **OCR pipeline** — find short extractions → deploy → remote OCR → pull → merge (GPU for OCR)
3. **Translation pipeline** — find non-English (after OCR merge) → deploy → remote translate → pull → merge (GPU for translation)
4. **Analysis** — before/after structure → deploy → remote summarization → pull (GPU for summarization)
5. **Summaries** — build unit summaries → remote change summarization → pull (GPU for summarization)
6. **Clustering** — cluster → deploy → remote cluster summarization → pull (GPU for summarization)

Extra args are passed through to the underlying Python scripts, e.g. `./pipeline.sh remote summarize --batch-size 16`.

## Key Design Decisions

### Parallelism strategy

The detail scraper uses three separate thread pools (initiative, feedback, PDF extraction) to avoid deadlocks. The initiative pool spawns work into the feedback and PDF pools, so sharing a single pool would cause workers waiting on sub-tasks that can never be scheduled.

### Text chunking for LLM inference

Both translation and summarization split long texts at sentence boundaries (default 5,000 characters). The `text_utils.split_into_chunks` function tries `.!?`-followed-by-whitespace splits first, falling back to newline splits if a single sentence exceeds the limit. This keeps semantic units intact while staying within context windows.

### Recursive summarization

For documents spanning multiple chunks, pass 1 summarizes each chunk independently, then chunk summaries are recursively combined in groups of up to 4 (configurable via `--max-combine-chunks`) until a single summary remains. This avoids context window overflow while maintaining global coherence, even for very long documents with many chunks.

### In-place merge pattern

OCR results and translations are generated as separate files, then merged back into `data/scrape/initiative_details/*.json`. The merge scripts preserve original text as `extracted_text_without_ocr` or `extracted_text_before_translation`, making it possible to audit or revert enrichments. All merge scripts support `--dry-run`.

### Deduplication and resume

Translation and summarization pipelines write per-batch result files. On restart, completed batches are loaded from disk. Identical text chunks across different records are cached and only processed once (cross-batch dedup).

### PDF-first extraction strategy

Many file uploads on the EU portal have incorrect extensions (e.g. a `.doc` file that is actually a PDF). Both the scraper and repair script handle this by trying PDF extraction first for `.doc/.docx/.odt/.rtf` files. If the PDF extraction produces fewer than 100 characters, it falls back to the native format-specific pipeline. This recovers text from a significant number of attachments that would otherwise fail extraction.

### Publication type mapping

The scraper maps EU API publication type codes (e.g. `CFE_IMPACT_ASSESS`, `PROP_REG`, `DEL_REG_DRAFT`) to human-readable labels (e.g. "Call for evidence", "Commission adoption", "Draft act"). The full mapping is in `scrape_eu_initiative_details.py`.
