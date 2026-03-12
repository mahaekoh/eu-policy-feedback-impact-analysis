# EU Policy Feedback Impact Analysis

A data pipeline and web platform for studying whether public consultation feedback influences EU policy documents published through the European Commission's ["Have Your Say"](https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives_en) portal.

## Table of Contents

- [Research Question](#research-question)
- [Methodology](#methodology)
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

## Research Question

When the European Commission publishes a draft regulation, directive, or impact assessment and opens it for public feedback, **does that feedback measurably influence the documents the Commission publishes afterward?**

This project collects the full documentary record needed to investigate that question: every document the Commission published before and after receiving feedback, every piece of feedback submitted (including attached files in any language), and AI-generated summaries to support analysis at scale.

## Methodology

The analysis follows a documentary comparison approach:

1. **Collect** all EU "Have Your Say" initiatives (3,949 initiatives)
2. **Extract** the full text of every published document and every feedback attachment (PDFs, Word documents, RTF, ODT, plain text), with automatic retries for mislabeled file formats
3. **Recover** text from scanned or image-based PDFs using optical character recognition (OCR)
4. **Translate** non-English feedback attachments to English using a large language model
5. **Identify** the temporal boundary: which documents were published *before* feedback was received, and which came *after*
6. **Summarize** long documents and feedback attachments using AI to enable qualitative comparison at scale
7. **Unify** per-initiative summaries into consolidated before/after/feedback summary fields for downstream analysis

All 2,970 initiatives with feedback are included in the before/after analysis, even when no documents were published after the feedback period.

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.12 | See `.python-version` |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager (lockfile: `uv.lock`) |
| Node.js | >= 18 | For the webapp only |
| NVIDIA GPU | H100 recommended | For OCR, translation, summarization, clustering, classification. Not needed for scraping, merging, or the webapp. |
| macOS `textutil` | (system) | DOC text extraction (macOS only; optional) |

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

### 1. Install Python dependencies

```bash
uv sync
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

### Master initiative list (`data/scrape/eu_initiatives.csv`)

A CSV file with all 3,949 initiatives, including their ID, title, type of act, feedback dates, policy topics, and URL.

### Per-initiative detail files (`data/scrape/initiative_details/*.json`)

One JSON file per initiative (3,898 files) containing the complete record: all publications with their documents (full extracted text), all feedback items with their metadata (date, language, respondent type, country, organization) and attached files (full extracted text, translated to English where needed).

### Before/after analysis files (`data/analysis/before_after/*.json`)

One JSON file per analysed initiative (all with feedback) with the data restructured for comparison:

- **`documents_before_feedback`** — full text of all documents published up to and including the first publication that received feedback
- **`documents_after_feedback`** — full text of documents from the final publication (empty when no post-feedback documents exist)
- **`middle_feedback`** — all feedback submitted between the first feedback publication and the final document publication

### AI-generated summaries (`data/analysis/summaries/*.json`)

Each document and feedback attachment in the before/after analysis is enriched with a `summary` field: a detailed summary (up to 10 paragraphs) generated by a 120-billion-parameter language model.

### Unified summaries (`data/analysis/unit_summaries/*.json`)

The `build_unit_summaries.py` script consolidates individual document and attachment summaries into per-initiative summary fields:

- **`before_feedback_summary`** — concatenation of all document summaries from before feedback
- **`after_feedback_summary`** — concatenation of all document summaries from after feedback
- **`combined_feedback_summary`** — on each feedback item: the free-text comment plus all attachment summaries, concatenated

### Change summaries (`data/analysis/change_summaries/*.json`)

For initiatives that have both before- and after-feedback documents, a `change_summary` field describes the substantive policy changes between the two sets of documents. Generated by a 120-billion-parameter language model that receives both summaries and a unified diff.

### Feedback clustering (`data/clustering/<scheme>/*.json`)

Feedback is clustered across initiatives using sentence embeddings. Each clustering scheme (algorithm + model + parameters) produces its own subdirectory. Multiple schemes can be configured in `pipeline.conf`.

### Cluster summaries (`data/cluster_summaries/<scheme>/*.json`)

AI-generated summaries of each feedback cluster, produced by a 120-billion-parameter language model.

### Web application (`webapp/`)

A Next.js web application for browsing all initiatives and their feedback interactively. Features include:

- **Initiative index** with full-text search, sorting (most discussed, recently discussed, newest), filtering by stage/department/topic/policy area, and pagination
- **Initiative detail** with expandable publication sections showing documents and feedback, AI-generated summaries (document, change, before/after), and clustered feedback visualization
- **Feedback exploration** with user type color coding, country flags, filtering by type/search/empty attachments, and infinite scroll
- **Cluster view** with multiple clustering scheme support, nested cluster trees, and per-cluster statistics
- **Optional Google sign-in** via Auth.js (the app is fully accessible without signing in)

```bash
cd webapp && npm install && npm run dev
```

The app reads data directly from the `data/` directory — no separate database or import step needed.

### Standalone viewers (`viewers/`)

- **`viewer.html`** — Browse per-initiative JSON files in the browser. Tabbed navigation (Before Feedback, After Feedback, Feedback, Publications), expandable text blocks, user type color coding, feedback filtering, and chunked infinite scroll.
- **`feedback-viewer.html`** — Browse clustered feedback results. Loads clustering JSON files, shows nested cluster trees with per-cluster country/user-type statistics, feedback search, and individual feedback items with attachments.

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

- **Coverage**: 2,970 of 3,949 initiatives have feedback and are included in the before/after analysis (~75%). The remaining initiatives have no public feedback on the portal.
- **Correlation, not causation**: Finding that a document changed after feedback does not prove the feedback caused the change. The Commission may have planned revisions independently.
- **Text extraction quality**: Most PDFs extract cleanly, but some scanned documents required OCR, which can introduce errors. Original text is preserved alongside OCR results for verification.
- **Translation quality**: Non-English feedback was translated by a large language model. Translations are generally accurate but may miss nuance. Original text is preserved alongside translations.
- **Feedback text vs. attachments**: Some respondents submit detailed positions as attached documents rather than in the free-text comment field. The pipeline captures both, but analysis should account for this variation.
- **Time period**: Initiatives published from June 2016 to February 2026.

## Working with the Data

The output files are standard JSON. You can explore them with:

- **`viewers/viewer.html`**: open in any browser, load an initiative JSON — tabbed navigation, expandable text, filtering, and color-coded respondent types
- **`viewers/feedback-viewer.html`**: open in any browser, load a clustering JSON — nested cluster tree with statistics and search
- **Python**: `json.load()` to read, then navigate the nested structure
- **jq** (command line): e.g. `jq '.publications[0].feedback | length' data/scrape/initiative_details/12970.json` to count feedback items
- **Any JSON viewer**: browser extensions, VS Code, or online tools like [jsoncrack.com](https://jsoncrack.com)

## Configuration Files

- **`pipeline.conf`** — pipeline orchestration config (remote host, SSH key, clustering schemes). Copy from `pipeline.conf.example`.

## Running the Pipeline

The pipeline is orchestrated by `pipeline.sh`. See `./pipeline.sh list` for all stages, or run `./pipeline.sh full` for the complete end-to-end pipeline. Remote GPU jobs run via `nohup` to survive SSH disconnects — use `./pipeline.sh logs` to monitor them.

## Technical Details

For a complete technical breakdown of each pipeline component, dependencies, data schemas, and instructions for running the pipeline, see [ARCHITECTURE.md](ARCHITECTURE.md).
