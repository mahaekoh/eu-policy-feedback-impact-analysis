# EU "Have Your Say" Initiative Data Pipeline

This project scrapes, processes, and enriches EU Better Regulation "Have Your Say" initiative data.

## Data directory structure

All data lives under `data/`:

```
data/
  scrape/                          # Scraped raw data (source of truth)
    eu_initiatives.csv
    eu_initiatives_raw.json
    initiative_details/            # Per-initiative JSONs (mutated by merges)
    doc_cache/
  repair/                          # Repair pipeline output
    repaired_details/
    repair_report.json
  ocr/                             # OCR pipeline I/O
    short_pdf_report.json
    pdfs/
    short_pdf_report_ocr.json
  translation/                     # Translation pipeline I/O
    non_english_attachments.json
    non_english_attachments_translated.json
    translation_batches/
  analysis/                        # Analysis & summarization output
    before_after/                  # initiative_stats output
    summaries/                     # summarize_documents output
    unit_summaries/                # build_unit_summaries output
    change_summaries/              # summarize_changes output
  clustering/                      # Clustering output (per-scheme subdirs)
  classification/                  # Classification output
  cluster_summaries/               # Cluster summary output (per-scheme subdirs)
config/
  init-no-response-blacklist-19.txt
```

## Data flow

```
scrape_eu_initiatives.py          → data/scrape/eu_initiatives.csv + eu_initiatives_raw.json
scrape_eu_initiative_details.py   → data/scrape/initiative_details/*.json
find_short_pdf_extractions.py     → data/ocr/short_pdf_report.json + data/ocr/pdfs/
ocr_short_pdfs.py                 → data/ocr/short_pdf_report_ocr.json
merge_ocr_results.py              → updates data/scrape/initiative_details/*.json in-place
find_non_english_feedback_attachments.py → data/translation/non_english_attachments.json
translate_attachments.py          → data/translation/non_english_attachments_translated.json
merge_translations.py             → updates data/scrape/initiative_details/*.json in-place
initiative_stats.py               → data/analysis/before_after/*.json
summarize_documents.py            → data/analysis/summaries/*.json
build_unit_summaries.py           → data/analysis/unit_summaries/*.json
summarize_changes.py              → data/analysis/change_summaries/*.json
cluster_all_initiatives.py        → data/clustering/<scheme>/*.json
classify_initiative_and_feedback.py → data/classification/*.json
summarize_clusters.py             → data/cluster_summaries/<scheme>/*.json
```

### Repair pipeline (re-extracts broken attachments, then re-runs OCR + translation)

```
repair_broken_attachments.py      → data/repair/repaired_details/*.json + data/repair/repair_report.json
find_short_pdf_extractions.py -r  → data/ocr/  (scoped to repaired attachments)
ocr_short_pdfs.py                 → data/ocr/short_pdf_report_ocr.json
merge_ocr_results.py              → updates data/repair/repaired_details/*.json in-place
find_non_english_feedback_attachments.py -r → data/translation/non_english_attachments.json  (scoped to repaired)
translate_attachments.py          → data/translation/non_english_attachments_translated.json
merge_translations.py             → updates data/repair/repaired_details/*.json in-place
```

**Important:** When using `-r repair_report.json` with `find_non_english_feedback_attachments.py` or `find_short_pdf_extractions.py`, point the source at the *repaired* output directory (not the original `data/scrape/initiative_details/`), since the repaired JSONs are the ones with the newly extracted text.

## Pipeline orchestration

`pipeline.sh` orchestrates the full pipeline. Copy `pipeline.conf.example` to `pipeline.conf` and fill in remote host details.

```
./pipeline.sh list                     # show all stages
./pipeline.sh <stage> [extra-args...]  # run a single stage
./pipeline.sh full                     # full pipeline
./pipeline.sh deploy                   # rsync src/ to remote
./pipeline.sh remote summarize         # run summarize on remote GPU
./pipeline.sh pull summaries           # rsync results back
./pipeline.sh logs                     # list recent remote logs
./pipeline.sh logs tail summarize      # tail a specific step's log
```

Remote commands run via `nohup` with stdout/stderr piped to log files under `logs/` on the remote host. This ensures long-running GPU jobs survive SSH disconnects. The local terminal tails the log in real-time and reads the exit code from a status file when the job completes.

## Scripts

### Scraping

**`src/scrape_eu_initiatives.py`** — Scrapes all EU "Have Your Say" initiatives from the Better Regulation API (no date filter — fetches everything available). Fetches all pages in parallel (10 workers). Outputs `data/scrape/eu_initiatives.csv` (flat extracted fields) and `data/scrape/eu_initiatives_raw.json` (full API data for each initiative). Supports `-o` for custom output path.

**`src/scrape_eu_initiative_details.py`** — Fetches detailed data for each initiative (publications, feedback, attachments) and extracts text from attached files. Uses 20-thread parallelism. For `.doc/.docx/.odt/.rtf` files, tries PDF extraction first (many uploads are mislabeled PDFs), then falls back to the format-specific pipeline. Supports PDF (pymupdf with OCR fallback), DOCX (docx2md), DOC (macOS textutil), RTF/ODT (pypandoc), and TXT. Outputs per-initiative JSON files to `data/scrape/initiative_details/`. Each output JSON includes a `last_cached_at` ISO 8601 timestamp. Supports `--cache-dir` / `-c` to cache downloaded publication document files to disk (as `{cache_dir}/{init_id}/pub{pub_id}_doc{doc_id}_{filename}`), so re-runs and retry passes reuse cached files instead of re-downloading. Only publication-level documents are cached, not feedback attachments. Supports `--max-age HOURS` (default 48) for incremental updates: initiatives cached more recently than `max_age` hours are skipped; stale initiatives are re-fetched from the API with a merge strategy that preserves derived fields (`extracted_text`, `extracted_text_without_ocr`, `extracted_text_before_translation`, `summary`, `repair_method`, etc.) on documents and attachments whose source material (pages, size_bytes, document_id, feedback_text) hasn't changed. Terminal stages (SUSPENDED, ABANDONED) and ADOPTION_WORKFLOW initiatives with all-closed feedback are never re-checked regardless of age.

### Analysis / reporting

**`src/find_missing_initiatives.py`** — Reports initiative IDs present in the CSV but missing from `data/scrape/initiative_details/`, or with incomplete feedback data.

**`src/find_initiative_by_pub.py`** — Lookup utility: finds which initiative contains a given publication ID.

**`src/find_missing_extracted_text.py`** — Scans initiative data for publication documents and feedback attachments that have no `extracted_text`.

**`src/find_non_english_feedback_attachments.py`** — Finds feedback attachments where the feedback language is not English. Supports `-o` for JSON output with full metadata, `-f` for initiative ID whitelist filter, `-r` for repair report whitelist (only check attachments listed in `repair_report.json`).

**`src/find_short_pdf_extractions.py`** — Finds attachments where `extracted_text` is suspiciously short (<100 chars). Checks all attachment types regardless of file extension (since many non-PDF extensions are actually PDFs). Downloads files in parallel (20 workers). Supports `-o` for output directory (writes `{out_dir}/short_pdf_report.json` and downloads PDFs to `{out_dir}/pdfs/`), `-f` for whitelist filter, `-r` for repair report whitelist (only check attachments listed in `repair_report.json`).

### OCR pipeline

**`src/ocr_short_pdfs.py`** — GPU-accelerated OCR using EasyOCR with CUDA. Takes the output directory from `find_short_pdf_extractions.py` (reads `short_pdf_report.json` and `pdfs/` from it). Renders PDF pages to 300 DPI images via pymupdf, runs OCR, writes results to `{input_dir}/short_pdf_report_ocr.json` (override with `-o`). Designed for H100.

**`src/merge_ocr_results.py`** — Merges OCR results back into initiative JSON files. Replaces `extracted_text`, preserves original as `extracted_text_without_ocr`. Supports `--dry-run`.

### Translation pipeline

**`src/translate_attachments.py`** — Translates non-English feedback attachment texts to English using vLLM batch inference with `unsloth/gpt-oss-120b`. Uses `openai_harmony` for structured prompts with `ReasoningEffort.MEDIUM`. Long documents are chunked at sentence boundaries (default 5000 chars). Chunks returning "NO TRANSLATION NEEDED" are replaced with the original text during reassembly.

**`src/merge_translations.py`** — Merges translations back into initiative JSON files. Replaces `extracted_text`, preserves original as `extracted_text_before_translation`. Skips records containing "NO TRANSLATION NEEDED". Supports `--dry-run`.

### Summarization pipeline

**`src/initiative_stats.py`** — Analyzes initiative publication/feedback structure for all initiatives in the details directory. With `-o`, writes per-initiative JSONs with `documents_before_feedback`, `documents_after_feedback` (empty when no post-feedback docs exist), and `middle_feedback` attributes for all initiatives with feedback.

**`src/summarize_documents.py`** — Summarizes publication documents and feedback attachments using vLLM batch inference with `unsloth/gpt-oss-120b`. Takes the output of `initiative_stats.py -o`. Long texts are split into chunks at sentence boundaries (default 5000 chars), each chunk is summarized (pass 1), then multi-chunk summaries are recursively combined in groups of up to `--max-combine-chunks` (default 4) until a single summary remains. Both documents and feedback attachments are summarized into up to 10 paragraphs per chunk, and combined summaries are also up to 10 paragraphs. Adds `summary` field to each document and attachment object. Supports resume: skips initiative files whose output already exists (model is not loaded if there is no work). Within a run, per-batch result files in `_batches_pass1/group_NNNN/` and `_batches_pass2/group_NNNN/` provide crash recovery for incomplete groups.

**`src/build_unit_summaries.py`** — Consolidates individual document and attachment summaries into per-initiative unified summary fields. Takes the output of `summarize_documents.py`. Adds `before_feedback_summary` (concatenation of document summaries from before feedback), `after_feedback_summary` (from after feedback), and `combined_feedback_summary` (feedback text + attachment summaries) on each middle feedback item. All concatenation joins on `\n\n`.

**`src/summarize_changes.py`** — Summarizes substantive changes between before- and after-feedback documents using vLLM batch inference with `unsloth/gpt-oss-120b`. Takes the output of `build_unit_summaries.py`. For each initiative with both `before_feedback_summary` and `after_feedback_summary`, computes a unified diff and asks the LLM to describe what changed in up to 10 paragraphs. Adds `change_summary` field at top level. Initiatives missing either summary are copied through unchanged. Supports resume: skips initiative files whose output already exists (model is not loaded if there is no work). Per-batch result files in `_batches/` provide crash recovery.

### Clustering & classification

**`src/cluster_all_initiatives.py`** — Clusters feedback across initiatives using sentence embeddings. Supports agglomerative and HDBSCAN algorithms with configurable parameters. Reads from `data/analysis/unit_summaries/`, writes per-scheme output to `data/clustering/<scheme>/`. Scheme names encode the algorithm, model, and parameters (e.g. `agglomerative_google_embeddinggemma-300m_distance_threshold=0.75_...`).

**`src/classify_initiative_and_feedback.py`** — Classifies initiatives and their feedback using vLLM batch inference with `unsloth/gpt-oss-120b`. Takes unit summaries as input, writes per-initiative classification JSONs to `data/classification/`.

**`src/summarize_clusters.py`** — Summarizes feedback clusters using vLLM batch inference with `unsloth/gpt-oss-120b`. Takes clustering output from `data/clustering/<scheme>/`, writes cluster summaries to `data/cluster_summaries/<scheme>/`.

### Repair pipeline

**`src/repair_broken_attachments.py`** — Scans `data/scrape/initiative_details/` for feedback attachments that have `extracted_text_error` and no `extracted_text`, downloads them, and retries extraction. For `.doc/.docx/.odt/.rtf` files, tries PDF extraction first (since many are mislabeled PDFs), then falls back to the format-specific pipeline. Writes updated initiative JSON copies to a specified output directory (only files with at least one successful repair). Also writes `repair_report.json` — a machine-readable list of all repaired attachments keyed by `(initiative_id, publication_id, feedback_id, attachment_id)`, which can be passed as `-r` to downstream scripts to scope them to just the repaired attachments. Supports `-f` for initiative ID whitelist filter, `-w` for worker count (default 20), `--dry-run` for scanning without repairing. Adds `repair_method` (`"pdf-reinterpret"` or `"native"`) and `repair_old_error` fields to repaired attachments for traceability.

### Webapp

A Next.js 16 web application (`webapp/`) for browsing initiatives and feedback interactively. See `webapp/AUTH.md` for Google sign-in setup.

**Tech stack:** Next.js 16.1.6, React 19, Tailwind CSS 4, shadcn/ui (Radix + Lucide), next-auth v5 (Google OAuth, JWT sessions).

**Pages:**
- `/` — Initiative index with search, sort, filters, pagination (50/page). Deduplicates initiatives sharing identical feedback IDs.
- `/initiative/[id]` — Initiative detail with publications view (documents + feedback) and cluster view. Empty/disabled publications shown as compact links.

**API routes:**
- `/api/auth/[...nextauth]` — NextAuth OAuth handlers
- `/api/clusters/[id]` — Fetch clustering data for an initiative by scheme

**Key lib files:**
- `src/lib/data.ts` — Server-side data loading from `../data/`. Uses regex extraction (not JSON parsing) for fast index builds. 5-minute cache TTL.
- `src/lib/types.ts` — TypeScript interfaces for Initiative, Publication, Feedback, Attachment, ClusterData, etc.
- `src/auth.ts` — Auth.js config with Google provider
- `src/proxy.ts` — Session cookie refresh (Next.js 16 middleware proxy)

**Running:** `cd webapp && npm run dev` (reads data from `../data/scrape/initiative_details/` and `../data/clustering/`).

### Viewer

**`viewers/viewer.html`** — Standalone HTML file (no dependencies) for interactively browsing per-initiative JSON files in the browser. Supports file loading via browser file picker. Shows initiative metadata, tabbed navigation (Before Feedback, After Feedback, Feedback, Publications), document download links, feedback portal links, attachment download links, expandable text blocks (summaries, extracted text, pre-translation/pre-OCR originals), user type color coding, feedback filtering by type/search/empty, and chunked infinite scroll for large feedback lists.

### Utilities

**`src/inference_utils.py`** — Shared vLLM batch inference helpers. Contains `build_prefill(encoding, text, prompt_prefix, reasoning_effort, identity_prompt)` for building openai_harmony prefill dicts, `extract_final_texts(outputs, encoding)` for parsing the 'final' channel from vLLM outputs, and `run_batch_inference(...)` for batched inference with dedup, resume, and per-batch file output. Used by `summarize_documents.py`, `summarize_clusters.py`, and `summarize_changes.py`.

**`src/text_utils.py`** — Shared library. Contains `split_into_chunks(text, max_chars)` which splits text at sentence boundaries with a fallback to newline splits.

**`src/print_chunk.py`** — Debug utility to print a specific chunk of a feedback attachment. Takes a spec like `"init=12096 fb=503089 att=6276475 chunk=5/15"` and the `data/scrape/initiative_details/` directory.

## Key dependencies

- **pymupdf** / **pymupdf4llm** — PDF text extraction with OCR fallback (tesseract at 300 DPI)
- **docx2md** — DOCX text extraction
- **pypandoc** / **pypandoc_binary** — RTF and ODT text extraction
- **easyocr** — GPU-accelerated OCR (CUDA)
- **vllm** — LLM batch inference engine
- **openai_harmony** — Structured prompt encoding for gpt-oss models (reasoning effort, stop tokens, output parsing)
