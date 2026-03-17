# EU Policy Feedback Transparency Platform

A data pipeline and web platform for exploring public consultation feedback in EU policy-making. It processes the full documentary record of the European Commission's ["Have Your Say"](https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives_en) portal — every initiative, every published document, and every piece of public feedback — so that citizens, researchers, and journalists can see what the public is telling the Commission and how the Commission's documents evolve after receiving that input.

## Table of Contents

- [Purpose](#purpose)
- [What It Does](#what-it-does)
- [Data Source](#data-source)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [What the Pipeline Produces](#what-the-pipeline-produces)
- [Data Structure](#data-structure)
- [Key Numbers](#key-numbers)
- [Scope and Limitations](#scope-and-limitations)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Pipeline Orchestration](#pipeline-orchestration)
- [Working with the Data](#working-with-the-data)

## Purpose

The European Commission's "Have Your Say" portal receives thousands of public feedback submissions on proposed regulations, directives, and impact assessments. This feedback — from citizens, companies, NGOs, trade unions, academics, and public authorities — is publicly available, but is scattered across thousands of initiative pages, buried in PDF attachments, written in dozens of languages, and impractical to explore at scale.

This project addresses two questions:

1. **What is the public telling the Commission?** By extracting, translating, summarizing, and clustering feedback across all initiatives, the platform makes it possible to see the full scope of public input — who is participating, what they are saying, and how feedback is distributed across countries, sectors, and topics.

2. **How do Commission documents evolve after public consultation?** By tracking which documents were published before and after feedback periods, summarizing both, and computing structured diffs, the platform makes it possible to examine how policy texts change in the wake of public input.

## What It Does

The pipeline processes the full "Have Your Say" archive:

1. **Collect** all EU "Have Your Say" initiatives (3,949 initiatives)
2. **Extract** the full text of every published document and every feedback attachment (PDFs, Word documents, RTF, ODT, plain text), with automatic retries for mislabeled file formats
3. **Recover** text from scanned or image-based PDFs using optical character recognition (OCR)
4. **Translate** non-English feedback attachments to English using a large language model
5. **Identify** the temporal structure: which documents were published before, during, and after public consultation periods
6. **Summarize** long documents and feedback attachments using AI, making the substance of hundreds of pages accessible at a glance
7. **Compare** before- and after-feedback documents using AI-generated change summaries, highlighting how policy texts evolved after public consultation
8. **Cluster** feedback by topic using sentence embeddings, revealing the thematic structure of public input on each initiative
9. **Visualize** everything through an interactive web application with search, filtering, and drill-down exploration

All 2,970 initiatives with feedback are processed, covering the full range of EU policy areas from 2016 to the present.

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.12 | See `.python-version` |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager (lockfile: `uv.lock`) |
| Node.js | >= 18 | For the webapp only |
| NVIDIA GPU | H100 recommended | For OCR, translation, summarization, clustering, classification. Not needed for scraping, merging, or the webapp. |
| macOS `textutil` | (system) | DOC text extraction (macOS only; optional) |
| Hugging Face account | — | Required for model downloads. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |

### Python dependencies (`pyproject.toml`)

Dependencies are split into two groups in `pyproject.toml`:

- **Base** (`uv sync`): pymupdf, pymupdf4llm, pypandoc, pypandoc_binary, docx2md, huggingface-hub — for local scraping, merging, text extraction, and index building
- **GPU optional** (`uv sync --extra gpu` or `pip install`): vllm, openai-harmony, easyocr, sentence-transformers, scikit-learn, hdbscan, torch, numpy — for the remote GPU host

### Key Python dependencies

| Package | Purpose |
|---|---|
| pymupdf / pymupdf4llm | PDF text extraction with tesseract OCR fallback (300 DPI) |
| docx2md | DOCX text extraction |
| pypandoc / pypandoc_binary | RTF and ODT text extraction |
| easyocr | GPU-accelerated OCR (CUDA) |
| vllm | LLM batch inference engine |
| openai_harmony | Structured prompt encoding for `unsloth/gpt-oss-120b` |
| sentence_transformers | Sentence embeddings (`google/embeddinggemma-300m`) |
| scikit-learn / hdbscan | Feedback clustering algorithms |
| cuML (optional) | GPU-accelerated clustering via RAPIDS |

## Quick Start

### 1. Install dependencies and log in to Hugging Face

```bash
# Install local Python dependencies (scraping, merging, text extraction)
./pipeline.sh setup

# Or manually: uv sync && huggingface-cli login
```

For the remote GPU host (OCR, translation, summarization, clustering):

```bash
# Deploy code, install GPU deps (pip), log in to Hugging Face on remote
./pipeline.sh setup-remote
```

### 2. Scrape initiatives (local, no GPU needed)

```bash
# Scrape all initiative metadata from the Better Regulation API
python src/scrape_eu_initiatives.py

# Scrape detailed data for each initiative (publications, feedback, attachments)
# with text extraction from all attached files
python src/scrape_eu_initiative_details.py -c data/scrape/doc_cache
```

### 3. Run GPU-accelerated stages (requires remote GPU host)

```bash
# Copy pipeline config and fill in remote host details
cp pipeline.conf.example pipeline.conf

# Deploy code to remote, run summarization, pull results
./pipeline.sh deploy
./pipeline.sh remote summarize
./pipeline.sh pull summaries

# Or run the entire 28-stage pipeline end-to-end
./pipeline.sh full
```

### 4. Browse results in the web app

```bash
# Pre-compute webapp data (aggregated statistics, stripped initiative details)
python src/build_webapp_index.py data/scrape/initiative_details

# Start the webapp
cd webapp && npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to browse initiatives, feedback, summaries, and clusters. No sign-in required.

## Project Structure

```
.
├── src/                           # Python pipeline scripts (24 scripts)
│   ├── scrape_*.py                #   Scraping (2 scripts)
│   ├── find_*.py                  #   Analysis / data quality (4 scripts)
│   ├── ocr_*.py, merge_ocr_*.py  #   OCR pipeline (2 scripts)
│   ├── translate_*.py, merge_translations.py  #   Translation (2 scripts)
│   ├── initiative_stats.py        #   Before/after analysis
│   ├── summarize_*.py             #   LLM summarization (3 scripts)
│   ├── build_unit_summaries.py    #   Summary consolidation
│   ├── merge_*_summaries.py       #   Merge results back (2 scripts)
│   ├── cluster_all_initiatives.py #   Sentence-embedding clustering
│   ├── classify_*.py              #   LLM classification
│   ├── build_webapp_index.py      #   Pre-compute webapp data
│   ├── text_utils.py              #   Shared text chunking/filtering
│   ├── inference_utils.py         #   Shared vLLM batch inference helpers
│   └── print_chunk.py             #   Debug utility
├── webapp/                        # Next.js 16 web application
│   ├── src/app/                   #   Pages: /, /initiative/[id], /charts
│   ├── src/components/            #   14 React components + shadcn/ui
│   ├── src/lib/                   #   Data loading, types, utilities
│   └── AUTH.md                    #   Google OAuth setup guide
├── viewers/                       # Standalone HTML viewers (no dependencies)
│   ├── viewer.html                #   Initiative JSON browser
│   └── feedback-viewer.html       #   Clustered feedback browser
├── pipeline.sh                    # Pipeline orchestration (28 stages)
├── pipeline.conf.example          # Pipeline config template
├── pyproject.toml                 # Python project metadata
├── CLAUDE.md                      # Full technical reference
└── data/                          # All pipeline data (gitignored)
```

## Data Source

All data comes from the European Commission's [Better Regulation](https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives_en) portal. The portal publishes:

- **Initiative metadata**: title, type of act, department, policy area, status
- **Publications**: documents at each stage of the policy process (planned initiative, call for evidence, public consultation, draft act, adoption)
- **Feedback**: comments submitted by citizens, companies, business associations, NGOs, trade unions, academic institutions, and public authorities, along with any attached documents

## What the Pipeline Produces

### Pipeline step reference

Each pipeline step reads from the previous step's output and writes its own. The table below shows every step, its inputs, outputs, resume behavior, and whether output files are overwritten on re-runs.

| Step | Script | Input | Output | Resume behavior | Overwritten on re-run? |
|---|---|---|---|---|---|
| **Scrape list** | `scrape_eu_initiatives.py` | EU Better Regulation API | `data/scrape/eu_initiatives.csv` | None — always re-fetches all pages | Yes, regenerated every run |
|  |  |  | `data/scrape/eu_initiatives_raw.json` |  | Yes |
| **Scrape details** | `scrape_eu_initiative_details.py` | API + CSV list | `data/scrape/initiative_details/{id}.json` | Skips initiatives cached within `--max-age` hours (default 48). Terminal stages (SUSPENDED, ABANDONED) and closed ADOPTION_WORKFLOW initiatives are never re-checked. Corrupt JSON files are detected and re-fetched from scratch. | Stale files are re-fetched with a **merge strategy** that preserves derived fields (`summary`, `extracted_text_without_ocr`, `extracted_text_before_translation`, `cluster_feedback_summary`, `change_summary`, `diff`, `cluster_policy_summary`, `cluster_summaries`) on documents/attachments whose source material (pages, size_bytes, document_id, feedback_text) hasn't changed. |
| **Find short PDFs** | `find_short_pdf_extractions.py` | `data/scrape/initiative_details/` | `data/ocr/short_pdf_report.json` | None | Yes |
|  |  |  | `data/ocr/pdfs/{filename}` |  | Yes |
| **OCR** | `ocr_short_pdfs.py` | `data/ocr/short_pdf_report.json` + `data/ocr/pdfs/` | `data/ocr/short_pdf_report_ocr.json` | None | Yes, single file regenerated |
| **Merge OCR** | `merge_ocr_results.py` | OCR report + `initiative_details/` | Updates `data/scrape/initiative_details/{id}.json` **in place** | None — applies all records every run | In-place mutation: sets `extracted_text` and preserves original as `extracted_text_without_ocr` |
| **Find non-English** | `find_non_english_feedback_attachments.py` | `data/scrape/initiative_details/` | `data/translation/non_english_attachments.json` | None | Yes |
| **Translate** | `translate_attachments.py` | `data/translation/non_english_attachments.json` | `data/translation/non_english_attachments_translated.json` | Per-batch file resume: existing batch files in `_batches/` are loaded instead of re-running inference. | Yes, combined output file regenerated. Batch files are append-only. |
| **Merge translations** | `merge_translations.py` | Translation output + `initiative_details/` | Updates `data/scrape/initiative_details/{id}.json` **in place** | None | In-place mutation: sets `extracted_text` and preserves original as `extracted_text_before_translation`. Skips "NO TRANSLATION NEEDED" records. |
| **Analyze** | `initiative_stats.py` | `data/scrape/initiative_details/` | `data/analysis/before_after/{id}.json` | None — always regenerates all files | Yes |
| **Summarize docs** | `summarize_documents.py` | `data/analysis/before_after/` | `data/analysis/summaries/{id}.json` | **File-level**: skips initiatives whose output file already exists. **Item-level**: with `--prev-output`, reuses summaries from a previous output directory for items with unchanged text. **Batch-level**: per-batch files in `_batches_pass1/` and `_batches_pass2/` provide crash recovery within a run. Model is not loaded if there is no work. | No — files are immutable once written. To regenerate, delete the output file. |
| **Build unit summaries** | `build_unit_summaries.py` | `data/analysis/summaries/` | `data/analysis/unit_summaries/{id}.json` | None — always regenerates all files | Yes |
| **Summarize changes** | `summarize_changes.py` | `data/analysis/unit_summaries/` | `data/analysis/change_summaries/{id}.json` | **File-level**: skips initiatives whose output file already exists. **Batch-level**: per-batch files in `_batches/` and `_batches_combine/` provide crash recovery. Model is not loaded if there is no work. | No — files are immutable once written. To regenerate, delete the output file. |
| **Merge change summaries** | `merge_change_summaries.py` | Change summaries + `initiative_details/` | Updates `data/scrape/initiative_details/{id}.json` **in place** | None | In-place mutation: sets `change_summary` and `diff` at initiative top level. |
| **Cluster feedback** | `cluster_all_initiatives.py` | `data/analysis/unit_summaries/` | `data/clustering/{scheme}/{id}_{algo}_{model}_{params}.json` | Optional `--skip-existing` flag skips initiatives with existing output. **Not passed by pipeline.sh** — all files are regenerated every run. | **Yes — files are overwritten every run** (unless `--skip-existing` is used). |
|  |  |  | `data/embeddings/{model}/{id}.npz` | Hash-validated: cached embeddings are reused only if text hashes match. Stale caches are ignored (cache miss, re-encoded). | Yes — overwritten when initiative data changes. |
| **Classify** | `classify_initiative_and_feedback.py` | `data/analysis/unit_summaries/` | `data/classification/{id}.json` | **File-level**: skips initiatives whose output file already exists. **Batch-level**: per-batch files provide crash recovery. Model is not loaded if there is no work. | No — files are immutable once written. To regenerate, delete the output file. |
| **Summarize clusters** | `summarize_clusters.py` | `data/clustering/{scheme}/` | `data/cluster_summaries/{scheme}/{id}.json` | **File-level**: skips initiatives whose output file already exists. **Item-level**: reuses `cluster_feedback_summary` from `initiative_details` when available (Phase 1). **Cache-level**: content-addressed cache (`_cluster_cache.json`) keyed by SHA-256 of sorted feedback IDs skips clusters with unchanged membership (Phase 3). **Batch-level**: per-batch files in `_batches_p1/`, `_batches_p2/`, `_batches_p3/` provide crash recovery. | No — files are immutable once written. `_cluster_cache.json` is updated incrementally. |
| **Merge cluster summaries** | `merge_cluster_feedback_summaries.py` | Cluster summaries + `initiative_details/` | Updates `data/scrape/initiative_details/{id}.json` **in place** | None | In-place mutation: sets `cluster_feedback_summary` on each feedback item, `cluster_policy_summary` and `cluster_summaries` at initiative top level. |
| **Build webapp index** | `build_webapp_index.py` | `data/scrape/initiative_details/` | `data/webapp/initiative_index.json` | None — always regenerates | Yes |
|  |  |  | `data/webapp/global_stats.json` |  | Yes |
|  |  |  | `data/webapp/country_stats.json` |  | Yes |
|  |  |  | `data/webapp/initiative_details/{id}.json` |  | Yes — stripped copies (no `extracted_text`, `extracted_text_without_ocr`, `extracted_text_before_translation` on feedback attachments). |

### Resume and recovery patterns

The pipeline uses three levels of resume to avoid redundant work:

1. **File-level resume** (summarize_documents, summarize_changes, classify, summarize_clusters): if the output file for an initiative already exists in the output directory, the initiative is skipped entirely. The LLM model is not loaded if all work is already done. To force regeneration, delete the specific output file(s).

2. **Batch-level crash recovery** (all vLLM-based scripts): within a single run, inference results are written to per-batch JSON files (`_batches*/batch_NNNN.json`). If the process crashes mid-run, restarting loads completed batches from disk and resumes from where it left off. Batch directories are auto-cleaned by `pipeline.sh` after successful remote runs.

3. **Content-level caching** (summarize_clusters only): a content-addressed cache (`_cluster_cache.json`) maps SHA-256 hashes of sorted feedback ID sets to their cluster summaries. When cluster membership hasn't changed between runs, the cached summary is reused without LLM inference.

### Derived field preservation across re-scrapes

The scraper (`scrape_eu_initiative_details.py`) uses a merge strategy when re-fetching stale initiatives. For each document and feedback attachment, if the source material (page count, file size, document ID, or feedback text) hasn't changed, all derived fields are preserved from the previous version:

| Derived field | Set by | Preserved on |
|---|---|---|
| `extracted_text_without_ocr` | `merge_ocr_results.py` | Attachments (when source unchanged) |
| `extracted_text_before_translation` | `merge_translations.py` | Attachments (when source unchanged) |
| `summary` | `summarize_documents.py` | Documents and attachments (when source unchanged) |
| `cluster_feedback_summary` | `merge_cluster_feedback_summaries.py` | Feedback items (when `feedback_text` unchanged) |
| `change_summary`, `diff` | `merge_change_summaries.py` | Initiative top level |
| `cluster_policy_summary`, `cluster_summaries` | `merge_cluster_feedback_summaries.py` | Initiative top level |

This means running the full pipeline, re-scraping, then running it again does not require re-doing all LLM inference — only initiatives with genuinely changed source data need reprocessing.

### Pull behavior (pipeline.sh)

When pulling results from remote, `pipeline.sh` uses different strategies depending on whether output files are immutable:

| Pull target | rsync strategy | Rationale |
|---|---|---|
| `ocr` (single file) | Overwrite | OCR report is regenerated on each run |
| `translation` (single file) | Overwrite | Combined translation output is regenerated on each run |
| `translation` (batch dir) | Skip existing | Batch files are append-only, never modified |
| `summaries` | Skip existing | File-level resume makes output files immutable |
| `classification` | Skip existing | File-level resume makes output files immutable |
| `clustering` | Overwrite | Files are overwritten every run (no `--skip-existing` in pipeline) |
| `embeddings` | Overwrite | Files are overwritten when initiative data changes |
| `cluster-summaries` | Skip existing | File-level resume makes output files immutable |
| `change-summaries` | Skip existing | File-level resume makes output files immutable |

### Output files overview

```
data/
├── scrape/
│   ├── eu_initiatives.csv                # Overwritten every scrape run
│   ├── eu_initiatives_raw.json           # Overwritten every scrape run
│   ├── initiative_details/{id}.json      # Mutated in-place by merge scripts;
│   │                                     #   re-scraped with field preservation
│   └── doc_cache/{id}/pub{pub}_doc{doc}_{name}  # Cached downloaded files, never deleted
├── ocr/
│   ├── short_pdf_report.json             # Overwritten by find_short_pdf_extractions
│   ├── pdfs/{filename}                   # Downloaded PDFs for OCR
│   └── short_pdf_report_ocr.json         # Overwritten by ocr_short_pdfs
├── translation/
│   ├── non_english_attachments.json      # Overwritten by find_non_english
│   ├── non_english_attachments_translated.json      # Overwritten by translate
│   └── non_english_attachments_translated_batches/  # Append-only batch files
├── analysis/
│   ├── before_after/{id}.json            # Overwritten by initiative_stats
│   ├── summaries/{id}.json               # Immutable (file-level resume)
│   │   ├── _batches_pass1/               # Crash recovery (auto-cleaned)
│   │   └── _batches_pass2/               # Crash recovery (auto-cleaned)
│   ├── unit_summaries/{id}.json          # Overwritten by build_unit_summaries
│   └── change_summaries/{id}.json        # Immutable (file-level resume)
│       └── _batches/                     # Crash recovery (auto-cleaned)
├── clustering/{scheme}/
│   └── {id}_{algo}_{model}_{params}.json # Overwritten every clustering run
├── embeddings/{model}/{id}.npz           # Overwritten when data changes (hash-validated)
├── classification/{id}.json              # Immutable (file-level resume)
├── cluster_summaries/{scheme}/
│   ├── {id}.json                         # Immutable (file-level resume)
│   ├── _cluster_cache.json               # Content-addressed cache (updated incrementally)
│   ├── _batches_p1/                      # Crash recovery (auto-cleaned)
│   ├── _batches_p2/                      # Crash recovery (auto-cleaned)
│   └── _batches_p3/                      # Crash recovery (auto-cleaned)
└── webapp/
    ├── initiative_index.json             # Overwritten by build_webapp_index
    ├── global_stats.json                 # Overwritten by build_webapp_index
    ├── country_stats.json                # Overwritten by build_webapp_index
    └── initiative_details/{id}.json      # Overwritten (stripped copies, no extracted_text)
```

### Web application (`webapp/`)

A Next.js web application for exploring EU public consultation feedback and how the Commission's documents evolve after receiving it. See [`webapp/README.md`](webapp/README.md) for detailed documentation. Features include:

- **Initiative index** with full-text search, sorting (most discussed, recently discussed, newest), filtering by stage/department/topic/policy area, and pagination
- **Initiative detail** with expandable publication sections showing documents and feedback, AI-generated summaries (document, change, before/after), and clustered feedback visualization
- **Feedback exploration** with user type color coding, country flags, filtering by type/search/empty attachments, and infinite scroll
- **Aggregate statistics** with country drill-downs, topic breakdowns, user type distributions, and time series
- **Cluster view** with multiple clustering scheme support, nested cluster trees, and per-cluster statistics
- **Optional Google sign-in** via Auth.js (the app is fully accessible without signing in)

```bash
cd webapp && npm install && npm run dev
```

The app reads pre-computed data from the `data/webapp/` directory and clustering data from `data/clustering/`. Requires `build_webapp_index.py` to have been run first. Data is loaded server-side with a 5-minute in-memory cache.

### Standalone viewers (`viewers/`)

- **`viewer.html`** — Browse per-initiative JSON files in the browser. Tabbed navigation (Before Feedback, After Feedback, Feedback, Publications), expandable text blocks, user type color coding, feedback filtering, and chunked infinite scroll.
- **`feedback-viewer.html`** — Browse clustered feedback results. Loads clustering JSON files, shows nested cluster trees with per-cluster country/user-type statistics, feedback search, and individual feedback items with attachments.

Both are standalone HTML files with no dependencies. Open in any browser, then use the file picker to load JSON files.

## Data Structure

Each initiative JSON file follows this hierarchy:

```
Initiative
  |-- id, title, type of act, department, topics
  |-- Publications (ordered by date)
       |-- publication type (e.g. "Call for evidence", "Draft act")
       |-- published date, feedback end date
       |-- Documents
       |    |-- filename, label, extracted text, page count
       |-- Feedback items
            |-- date, language, respondent type, country, organization
            |-- feedback text (free-text comment)
            |-- Attachments
                 |-- filename, extracted text (translated if needed)
```

### Respondent types in the data

Feedback comes from a range of respondent types as classified by the EU portal:

- `EU_CITIZEN` — individual EU citizens
- `COMPANY` — individual companies
- `BUSINESS_ASSOCIATION` — industry and trade associations
- `NGO` — non-governmental organizations
- `TRADE_UNION` — labour unions
- `ACADEMIC_RESEARCH` — academic and research institutions
- `PUBLIC_AUTHORITY` — national or regional government bodies
- `CONSUMER_ORGANISATION` — consumer advocacy groups
- `ENVIRONMENTAL_ORGANISATION` — environmental advocacy groups
- `OTHER` — other organisations

## Key Numbers

| Metric | Count |
|--------|-------|
| Total initiatives scraped | 3,949 |
| Initiatives with full detail data | 3,898 |
| Initiatives with feedback (included in analysis) | 2,970 |
| Initiatives with no Commission response after feedback | 901 |

## Scope and Limitations

- **Coverage**: 2,970 of 3,949 initiatives have feedback (~75%). The remaining initiatives have no public feedback on the portal.
- **Text extraction quality**: Most PDFs extract cleanly, but some scanned documents required OCR, which can introduce errors. Original text is preserved alongside OCR results for verification.
- **Translation quality**: Non-English feedback was translated by a large language model. Translations are generally accurate but may miss nuance. Original text is preserved alongside translations.
- **Summarization quality**: AI-generated summaries aim to capture the substance of documents and feedback but may omit detail or emphasis. The full extracted text is always available alongside summaries.
- **Feedback text vs. attachments**: Some respondents submit detailed positions as attached documents rather than in the free-text comment field. The pipeline captures both, but analysis should account for this variation.
- **Time period**: Initiatives published from June 2016 to the present. The pipeline supports incremental updates as new initiatives and feedback are published.

## Working with the Data

The output files are standard JSON. You can explore them with:

- **`viewers/viewer.html`**: open in any browser, load an initiative JSON — tabbed navigation, expandable text, filtering, and color-coded respondent types
- **`viewers/feedback-viewer.html`**: open in any browser, load a clustering JSON — nested cluster tree with statistics and search
- **Python**: `json.load()` to read, then navigate the nested structure
- **jq** (command line): e.g. `jq '.publications[0].feedback | length' data/scrape/initiative_details/12970.json` to count feedback items
- **Any JSON viewer**: browser extensions, VS Code, or online tools like [jsoncrack.com](https://jsoncrack.com)

## Running the Full Pipeline

The pipeline has 28 stages that alternate between local processing and remote GPU computation. Here is the complete stage order:

| # | Stage | Location | Description |
|---|---|---|---|
| 1 | `scrape` | local | Scrape initiative list and per-initiative details |
| 2 | `find-short-pdfs` | local | Find PDFs with suspiciously short extracted text |
| 3 | `deploy` | local→remote | Sync source code to remote GPU host |
| 4–6 | `push ocr` → `remote ocr` → `pull ocr` | remote GPU | OCR scanned PDFs (EasyOCR, multi-GPU) |
| 7 | `merge-ocr` | local | Merge OCR results back into initiative JSONs |
| 8 | `find-nonenglish` | local | Find non-English feedback attachments |
| 9–11 | `push translation` → `remote translate` → `pull translation` | remote GPU | Translate to English (120B LLM) |
| 12 | `merge-translations` | local | Merge translations back |
| 13 | `analyze` | local | Compute before/after feedback structure |
| 14–16 | `push analysis` → `remote summarize` → `pull summaries` | remote GPU | Summarize documents and feedback (120B LLM) |
| 17 | `build-summaries` | local | Consolidate per-document summaries |
| 18–20 | `push unit-summaries` → `remote cluster` → `pull clustering` | remote GPU | Cluster feedback (sentence embeddings, multi-GPU) |
| 21 | `build-index` | local | Pre-compute webapp index and statistics |
| 22–24 | `push clustering` → `remote summarize-clusters` → `pull cluster-summaries` | remote GPU | Summarize clusters (120B LLM) |
| 25 | `merge-cluster-feedback-summaries` | local | Merge cluster summaries back (per scheme) |
| 26–27 | `remote summarize-changes` → `pull change-summaries` | remote GPU | Detect before/after changes (120B LLM) |
| 28 | `merge-change-summaries` | local | Merge change summaries back |

All LLM stages use `unsloth/gpt-oss-120b` via vLLM batch inference. LLM stages (summarize, classify, summarize-clusters, summarize-changes) use file-level resume — they skip initiatives whose output already exists and don't load the model if there's no work. All LLM stages also write per-batch result files for crash recovery within a run. See [Resume and recovery patterns](#resume-and-recovery-patterns) for details.

## Pipeline Orchestration

`pipeline.sh` orchestrates the full pipeline. Copy `pipeline.conf.example` to `pipeline.conf` and fill in your remote GPU host details.

### Configuration (`pipeline.conf`)

| Variable | Default | Description |
|---|---|---|
| `REMOTE_HOST` | — | SSH host (e.g. `user@gpu-host`) |
| `REMOTE_DIR` | — | Remote working directory |
| `SSH_KEY` | — | Path to SSH private key |
| `PYTHON` | `python3` | Python executable on remote |
| `CLUSTER_SCHEMES` | — | Space-separated clustering scheme names. Each name encodes algorithm and parameters; `pipeline.sh` parses them into CLI flags. |

### Commands

```bash
./pipeline.sh setup                    # Install local Python deps (uv sync) + Hugging Face login
./pipeline.sh setup-remote             # Deploy code + install remote GPU deps (pip) + HF login
./pipeline.sh list                     # Show all 28 stages
./pipeline.sh full                     # Run entire pipeline end-to-end
./pipeline.sh <stage>                  # Run a single stage
./pipeline.sh deploy                   # Sync src/ to remote
./pipeline.sh remote <step>            # Run GPU step on remote
./pipeline.sh push <target>            # Upload data to remote
./pipeline.sh pull <target>            # Download results from remote
./pipeline.sh logs                     # List recent remote logs
./pipeline.sh logs tail <step>         # Tail a specific step's log
./pipeline.sh clean-batches <target>   # Delete batch recovery files on remote
```

**Push targets:** `ocr`, `translation`, `analysis`, `unit-summaries`, `clustering`, `all`

**Pull targets:** `ocr`, `translation`, `summaries`, `classification`, `clustering`, `embeddings`, `cluster-summaries`, `change-summaries`, `logs`, `all`

**Remote GPU steps:** `ocr`, `translate`, `summarize`, `classify`, `cluster`, `summarize-clusters`, `summarize-changes`

### Remote execution model

- GPU jobs run via `nohup` with stdout/stderr piped to log files under `logs/` on the remote host
- Long-running jobs survive SSH disconnects; the local terminal tails the log in real-time
- Exit code is read from a `.exit` status file when the job completes
- Batch recovery directories (`_batches*`) are auto-cleaned after successful runs
- Push/pull operations use parallel rsync (4 streams) with `--files-from` chunking for efficient large-directory transfers
- Pull behavior varies by target: immutable LLM outputs use `--ignore-existing` (skip already-downloaded files), while targets overwritten every run (clustering, embeddings, single files) use plain rsync to keep local copies current

## Technical Reference

For a complete technical breakdown of every pipeline script (all CLI arguments, defaults, input/output paths), utility libraries, webapp components, TypeScript interfaces, and data loading details, see [CLAUDE.md](CLAUDE.md).
