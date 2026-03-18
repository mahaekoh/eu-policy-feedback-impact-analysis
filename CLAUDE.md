# EU "Have Your Say" Initiative Data Pipeline

This project scrapes, processes, and enriches EU Better Regulation "Have Your Say" initiative data. Through a comprehensive data pipeline (scraping, OCR, translation, LLM summarization, embedding-based clustering, classification) and a web-based visualization platform, it enables citizens, researchers, and journalists to explore what the public is telling the Commission and how the Commission's documents evolve after public consultation.

## Requirements

- **Python**: >= 3.12 (see `.python-version`)
- **Package manager**: [uv](https://docs.astral.sh/uv/) (lockfile: `uv.lock`)
- **Node.js**: Required for the webapp (Next.js 16)
- **GPU**: Remote H100 required for OCR (EasyOCR/CUDA), translation, summarization, clustering, and classification (vLLM with `unsloth/gpt-oss-120b`)

## Data directory structure

All data lives under `data/`:

```
data/
  scrape/                          # Scraped raw data (source of truth)
    eu_initiatives.csv
    eu_initiatives_raw.json
    initiative_details/            # Per-initiative JSONs (mutated by merges)
    doc_cache/
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
  embeddings/                      # Cached sentence embeddings (per-model subdirs)
  classification/                  # Classification output
  cluster_summaries/               # Cluster summary output (per-scheme subdirs)
  webapp/                          # Pre-computed webapp data
    initiative_index.json          # Pre-built initiative index
    global_stats.json              # Aggregate cross-initiative statistics
    country_stats.json             # Per-country drill-down statistics
    initiative_details/            # Stripped copies (no attachment extracted_text)
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
merge_summaries.py                → updates data/scrape/initiative_details/*.json in-place (doc + attachment summaries)
build_unit_summaries.py           → data/analysis/unit_summaries/*.json
summarize_changes.py              → data/analysis/change_summaries/*.json
merge_change_summaries.py         → updates data/scrape/initiative_details/*.json in-place
cluster_all_initiatives.py        → data/clustering/<scheme>/*.json
classify_initiative_and_feedback.py → data/classification/*.json
summarize_clusters.py             → data/cluster_summaries/<scheme>/*.json + _cluster_cache.json
merge_cluster_feedback_summaries.py → updates data/scrape/initiative_details/*.json in-place
build_webapp_index.py             → data/webapp/initiative_index.json + global_stats.json + country_stats.json + initiative_details/
```

## Python dependencies (`pyproject.toml`)

Dependencies are split into two groups:

- **Base** (`uv sync`): pymupdf, pymupdf4llm, pypandoc, pypandoc_binary, docx2md, huggingface-hub. Needed for local scraping, merging, text extraction, and index building.
- **GPU optional** (`uv sync --extra gpu` or `pip install`): vllm, openai-harmony, easyocr, sentence-transformers, scikit-learn, hdbscan, torch, numpy. Needed on the remote GPU host for OCR, translation, summarization, clustering, and classification.

## Pipeline orchestration

`pipeline.sh` orchestrates the full pipeline. Copy `pipeline.conf.example` to `pipeline.conf` and fill in remote host details. Run `./pipeline.sh setup` and `./pipeline.sh setup-remote` once before the first pipeline run to install dependencies and log in to Hugging Face.

### Configuration (`pipeline.conf`)

Required variables:

| Variable | Default | Description |
|---|---|---|
| `REMOTE_HOST` | — | SSH host (e.g. `user@gpu-host`) |
| `REMOTE_DIR` | — | Remote working directory |
| `SSH_KEY` | — | Path to SSH private key |
| `PYTHON` | `python3` | Python executable on remote |
| `CLUSTER_SCHEMES` | — | Space-separated clustering scheme directory names. Each name encodes algorithm and parameters; `pipeline.sh` parses them into `cluster_all_initiatives.py` CLI flags. Example: `"agglomerative_google_embeddinggemma-300m_distance_threshold=0.75_linkage=average_max_cluster_size=20_max_depth=3_sub_cluster_scale=0.75"` |

### Commands

```
./pipeline.sh setup                    # install local Python deps (uv sync) + Hugging Face login
./pipeline.sh setup-remote             # deploy code + install remote GPU deps (pip) + HF login
./pipeline.sh list                     # show all stages
./pipeline.sh <stage> [extra-args...]  # run a single stage
./pipeline.sh full                     # full pipeline (all 28 stages in order)
./pipeline.sh deploy                   # rsync src/ to remote
./pipeline.sh remote <step> [args...]  # run a GPU step on remote
./pipeline.sh push <target>            # rsync data to remote
./pipeline.sh pull <target>            # rsync results back
./pipeline.sh logs                     # list recent remote logs
./pipeline.sh logs tail [step]         # tail a specific step's log
./pipeline.sh clean-batches <target>   # delete batch files on remote
./pipeline.sh recover                 # pull all outputs from remote + merge locally + rebuild index
```

### Full pipeline stage order (28 stages)

1. `scrape` — Runs both scrape scripts locally
2. `find-short-pdfs` — Finds short PDF extractions
3. `deploy` — Deploys src/ to remote
4. `push ocr` — Uploads OCR data to remote
5. `remote ocr` — Runs OCR on remote GPU
6. `pull ocr` — Downloads OCR results
7. `merge-ocr` — Merges OCR results locally
8. `find-nonenglish` — Finds non-English attachments
9. `push translation` — Uploads translation data
10. `remote translate` — Runs translation on remote GPU
11. `pull translation` — Downloads translation results
12. `merge-translations` — Merges translations locally
13. `analyze` — Runs initiative_stats locally
14. `push analysis` — Uploads analysis data
15. `remote summarize` — Runs document summarization on remote GPU
16. `pull summaries` — Downloads summaries
17. `build-summaries` — Runs build_unit_summaries locally
18. `push unit-summaries` — Uploads unit summaries
19. `remote cluster` — Runs clustering on remote GPU (multi-GPU via RAPIDS)
20. `pull clustering` — Downloads clustering results
21. `build-index` — Builds webapp index locally
22. `push clustering` — Uploads clustering data
23. `remote summarize-clusters` — Runs cluster summarization on remote GPU (per scheme)
24. `pull cluster-summaries` — Downloads cluster summaries
25. `merge-cluster-feedback-summaries` — Merges cluster summaries locally (per scheme)
26. `remote summarize-changes` — Runs change summarization on remote GPU
27. `pull change-summaries` — Downloads change summaries
28. `merge-change-summaries` — Merges change summaries locally

### Push targets

`ocr`, `translation`, `analysis`, `unit-summaries`, `clustering`, `all`

### Pull targets

`ocr`, `translation`, `summaries`, `classification`, `clustering`, `embeddings`, `cluster-summaries`, `change-summaries`, `logs`, `all`

### Remote GPU steps

`ocr`, `translate`, `summarize`, `classify`, `cluster`, `summarize-clusters`, `summarize-changes`

### Remote execution model

Remote commands run via `nohup` with stdout/stderr piped to log files under `logs/` on the remote host. This ensures long-running GPU jobs survive SSH disconnects. The local terminal tails the log in real-time and reads the exit code from a `.exit` status file when the job completes. Batch directories (`_batches*`) are auto-cleaned on successful completion.

Push/pull operations use parallel rsync (4 streams by default) with `--files-from` chunking for directory transfers. Pull behavior varies by target: immutable LLM output directories (summaries, classification, cluster-summaries, change-summaries) use `--ignore-existing` to skip already-downloaded files. Targets that are overwritten on every run (clustering, embeddings, single files like OCR/translation reports) use plain rsync without `--ignore-existing` to ensure local copies stay current.

## Scripts

### Scraping

**`src/scrape_eu_initiatives.py`** — Scrapes all EU "Have Your Say" initiatives from the Better Regulation API (no date filter — fetches everything available). Fetches all pages in parallel (10 workers). Outputs `data/scrape/eu_initiatives.csv` (flat extracted fields) and `data/scrape/eu_initiatives_raw.json` (full API data for each initiative).

| Argument | Type | Default | Description |
|---|---|---|---|
| `-o, --output` | str | `None` | Custom output CSV path (default: `eu_initiatives.csv`) |

**`src/scrape_eu_initiative_details.py`** — Fetches detailed data for each initiative (publications, feedback, attachments) and extracts text from attached files. Uses 4 thread pools: initiatives (20), feedback (20), PDF (40), page fetch (80). For `.doc/.docx/.odt/.rtf` files, tries PDF extraction first (many uploads are mislabeled PDFs), then falls back to the format-specific pipeline. Supports PDF (pymupdf with OCR fallback at 300 DPI via tesseract), DOCX (docx2md), DOC (macOS textutil), RTF/ODT (pypandoc), and TXT. Outputs per-initiative JSON files to `data/scrape/initiative_details/`. Each output JSON includes a `last_cached_at` ISO 8601 timestamp.

| Argument | Type | Default | Description |
|---|---|---|---|
| `initiative_id` | int (positional, optional) | `None` | Scrape a single initiative by ID and print JSON to stdout |
| `-o, --out-dir` | str | `None` | Output directory for per-initiative JSON files (defaults to `initiative_details/`) |
| `-c, --cache-dir` | str | `None` | Cache downloaded publication document files to disk (as `{cache_dir}/{init_id}/pub{pub_id}_doc{doc_id}_{filename}`). Re-runs reuse cached files. Only publication-level documents are cached, not feedback attachments. |
| `--max-age` | float | `48` | Max age in hours before re-fetching a cached initiative. Set to 0 to force update all. |

Incremental update behavior: initiatives cached more recently than `max_age` hours are skipped; stale initiatives are re-fetched from the API with a merge strategy that preserves derived fields (`extracted_text`, `extracted_text_without_ocr`, `extracted_text_before_translation`, `summary`, `cluster_feedback_summary`, etc.) on documents and attachments whose source material (pages, size_bytes, document_id, feedback_text) hasn't changed. Terminal stages (SUSPENDED, ABANDONED) and ADOPTION_WORKFLOW initiatives with all-closed feedback are never re-checked regardless of age. Top-level derived fields (e.g. `change_summary`, `diff`, `cluster_policy_summary`, `cluster_summaries`) are also preserved from the old record when re-scraping.

Corrupt JSON handling: if a cached initiative JSON file is corrupt (truncated write, encoding error), the scraper logs a warning and re-fetches the initiative from scratch instead of crashing. This handles files left in a broken state by interrupted previous runs.

### Analysis / reporting

**`src/find_missing_initiatives.py`** — Reports initiative IDs present in the CSV but missing from `data/scrape/initiative_details/`, or with incomplete feedback data (e.g. 400-status API errors). No arguments (reads from hardcoded paths).

**`src/find_initiative_by_pub.py`** — Lookup utility: finds which initiative contains a given publication ID. Uses `sys.argv` directly: `python find_initiative_by_pub.py <publication_id> <initiative_details_dir>`.

**`src/find_missing_extracted_text.py`** — Scans initiative data for publication documents and feedback attachments that have no `extracted_text`. Prints a table with error info.

| Argument | Type | Default | Description |
|---|---|---|---|
| `source` | str (positional) | — | Path to `initiative_details/` directory |
| `-f, --filter` | str | `None` | Path to newline-delimited file of initiative IDs to include |

**`src/find_non_english_feedback_attachments.py`** — Finds feedback attachments where the feedback language is not English.

| Argument | Type | Default | Description |
|---|---|---|---|
| `source` | str (positional) | — | Path to `initiative_details/` directory |
| `-o, --output` | str | `None` | Output path for JSON file with attachment records for translation |
| `-f, --filter` | str | `None` | File with newline-delimited initiative IDs to include |
| `-r, --repair-report` | str | `None` | Path to `repair_report.json`. When set, only check feedback attachments listed in the report. |

**`src/find_short_pdf_extractions.py`** — Finds attachments where `extracted_text` is suspiciously short (<100 chars). Checks all attachment types regardless of file extension (since many non-PDF extensions are actually PDFs). Downloads files in parallel (20 workers).

| Argument | Type | Default | Description |
|---|---|---|---|
| `source` | str (positional) | — | Path to `initiative_details/` directory |
| `-o, --out-dir` | str | `None` | Output directory (writes `{out_dir}/short_pdf_report.json` and downloads PDFs to `{out_dir}/pdfs/`) |
| `-f, --filter` | str | `None` | File with newline-delimited initiative IDs to include |
| `-r, --repair-report` | str | `None` | Path to `repair_report.json`. When set, only check feedback attachments listed in the report (any extension). |

### OCR pipeline

**`src/ocr_short_pdfs.py`** — GPU-accelerated OCR using EasyOCR with CUDA. Takes the output directory from `find_short_pdf_extractions.py` (reads `short_pdf_report.json` and `pdfs/` from it). Renders PDF pages to 300 DPI images via pymupdf, runs OCR. Multi-GPU support: spawns a separate subprocess per GPU with `CUDA_VISIBLE_DEVICES` isolation.

| Argument | Type | Default | Description |
|---|---|---|---|
| `input_dir` | str (positional) | — | Directory produced by `find_short_pdf_extractions.py` (contains `short_pdf_report.json` and `pdfs/`) |
| `-o, --output` | str | `None` | Output path for updated JSON report with OCR results (defaults to `{input_dir}/short_pdf_report_ocr.json`) |
| `--languages` | str | `"en"` | Comma-separated EasyOCR language codes |
| `--worker` | (hidden) | — | Internal: subprocess mode (shard, output path, languages) |

**`src/merge_ocr_results.py`** — Merges OCR results back into initiative JSON files. Replaces `extracted_text`, preserves original as `extracted_text_without_ocr`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `report` | str (positional) | — | Path to OCR report JSON (output of `ocr_short_pdfs.py`) |
| `details_dir` | str (positional) | — | Directory of per-initiative JSON files |
| `--dry-run` | flag | — | Print changes without modifying files |

### Translation pipeline

**`src/translate_attachments.py`** — Translates non-English feedback attachment texts to English using vLLM batch inference with `unsloth/gpt-oss-120b`. Uses `openai_harmony` for structured prompts with `ReasoningEffort.MEDIUM`. Long documents are chunked at sentence boundaries. Chunks returning "NO TRANSLATION NEEDED" are replaced with the original text during reassembly.

| Argument | Type | Default | Description |
|---|---|---|---|
| `input` | str (positional) | — | Path to JSON from `find_non_english_feedback_attachments.py` |
| `-o, --output` | str | — (required) | Output path for JSON with translations |
| `--model` | str | `unsloth/gpt-oss-120b` | Model name or path |
| `--max-tokens` | int | `131072` (32768×4) | Max output tokens per translation |
| `--max-model-len` | int | `None` | Max model context length. Set lower to reduce GPU memory. |
| `--temperature` | float | `0.15` | Sampling temperature |
| `--chunk-size` | int | `16384` | Max chars per chunk before splitting |
| `--batch-size` | int | `4096` | Number of prompts per inference batch |

**`src/merge_translations.py`** — Merges translations back into initiative JSON files. Replaces `extracted_text`, preserves original as `extracted_text_before_translation`. Skips records containing "NO TRANSLATION NEEDED". Supports two input modes: a combined JSON file, or a batch directory containing per-batch files.

| Argument | Type | Default | Description |
|---|---|---|---|
| `report` | str (positional) | — | Path to combined translated JSON, or path to batch directory |
| `details_dir` | str (positional) | — | Directory of per-initiative JSON files |
| `--input-records` | str | `None` | Path to original input JSON (for old batch files missing `publication_id`) |
| `--chunk-size` | int | `5000` | Chunk size used during translation (only relevant when using batch directory) |
| `--dry-run` | flag | — | Print proposed changes without modifying files |

### Summarization pipeline

**`src/initiative_stats.py`** — Analyzes initiative publication/feedback structure for all initiatives in the details directory. Identifies the first feedback publication, final post-feedback publication, and separates feedback periods.

| Argument | Type | Default | Description |
|---|---|---|---|
| `details_dir` | str (positional) | — | Directory of per-initiative JSON files |
| `-o, --output-dir` | str | `None` | Output directory for modified initiative JSONs (all initiatives with feedback). Adds `documents_before_feedback`, `documents_after_feedback` (empty when no post-feedback docs exist), and `middle_feedback` attributes. |
| `-v, --verbose` | flag | — | Print detailed per-initiative publication breakdowns |

**`src/summarize_documents.py`** — Summarizes publication documents and feedback attachments using vLLM batch inference with `unsloth/gpt-oss-120b`. Takes the output of `initiative_stats.py -o`. Uses a two-pass architecture: (1) chunk summaries — long texts are split into chunks at sentence boundaries, each chunk summarized into up to 10 paragraphs; (2) combine — multi-chunk summaries are recursively combined using a character budget until a single summary remains. Adds `summary` field to each document and attachment object.

| Argument | Type | Default | Description |
|---|---|---|---|
| `input_dir` | str (positional) | — | Directory of per-initiative JSON files (output of `initiative_stats.py -o`) |
| `-o, --output` | str | — (required) | Output directory for initiative JSONs with summaries added |
| `--model` | str | `unsloth/gpt-oss-120b` | Model name or path |
| `--max-tokens` | int | `131072` (32768×4) | Max output tokens per summary |
| `--max-model-len` | int | `131072` (32768×4) | Max model context length. Set lower to reduce GPU memory. |
| `--temperature` | float | `0.15` | Sampling temperature |
| `--chunk-size` | int | `16384` | Max chars per chunk before splitting |
| `--combine-budget` | int | `65536` (16384×4) | Max chars when combining summaries per inference call |
| `--batch-size` | int | `8192` | Number of prompts per inference batch |
| `--initiative-batch-size` | int | `128` | Number of initiative files to load at a time |
| `--prev-output` | str | `None` | Previous output directory to reuse summaries from. Items with existing summaries are skipped. |

Supports resume: skips initiative files whose output already exists (model is not loaded if there is no work). Within a run, per-batch result files in `_batches_pass1/group_NNNN/` and `_batches_pass2/group_NNNN/` provide crash recovery for incomplete groups.

**`src/merge_summaries.py`** — Merges document and attachment summaries back into initiative detail JSON files. Matches documents by `doc_id` and attachments by `(feedback_id, attachment_id)`. Sets `summary` field on each matched document in `publications[].documents[]` and each matched attachment in `publications[].feedback[].attachments[]`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `summary_dir` | str (positional) | — | Directory of summary JSON files (output of `summarize_documents.py`) |
| `details_dir` | str (positional) | — | Directory of per-initiative JSON files |
| `--dry-run` | flag | — | Print proposed changes without modifying files |

**`src/build_unit_summaries.py`** — Consolidates individual document and attachment summaries into per-initiative unified summary fields. Takes the output of `summarize_documents.py`. Adds `before_feedback_summary` (concatenation of document summaries from before feedback), `after_feedback_summary` (from after feedback), and `combined_feedback_summary` (feedback text + attachment summaries) on each middle feedback item. All concatenation joins on `\n\n`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `input_dir` | str (positional) | — | Directory of per-initiative JSON files (output of `summarize_documents.py`) |
| `-o, --output` | str | — (required) | Output directory for initiative JSONs with unified summaries |

**`src/summarize_changes.py`** — Summarizes substantive changes between before- and after-feedback documents using vLLM batch inference with `unsloth/gpt-oss-120b`. Takes the output of `build_unit_summaries.py`. For each initiative with both `before_feedback_summary` and `after_feedback_summary`, computes a unified diff and asks the LLM to describe what changed in up to 10 paragraphs. Adds `change_summary` field at top level. Initiatives missing either summary are copied through unchanged. Uses a two-pass architecture for large diffs with recursive combining.

| Argument | Type | Default | Description |
|---|---|---|---|
| `input_dir` | str (positional) | — | Directory of per-initiative JSON files (output of `build_unit_summaries.py`) |
| `-o, --output` | str | — (required) | Output directory for initiative JSONs with `change_summary` added |
| `--model` | str | `unsloth/gpt-oss-120b` | Model name or path |
| `--max-tokens` | int | `131072` (32768×4) | Max output tokens per summary |
| `--max-model-len` | int | `131072` (32768×4) | Max model context length |
| `--temperature` | float | `0.15` | Sampling temperature |
| `--batch-size` | int | `2048` | Number of prompts per inference batch |
| `--chunk-size` | int | `16384` | Max chars per chunk for splitting large inputs |
| `--combine-budget` | int | `65536` (16384×4) | Max chars when grouping summaries for combining |

Supports resume: skips initiative files whose output already exists (model is not loaded if there is no work). Per-batch result files in `_batches/` provide crash recovery.

**`src/merge_change_summaries.py`** — Merges change summaries back into initiative detail JSON files. Sets `change_summary` and `diff` at the top level of each initiative JSON.

| Argument | Type | Default | Description |
|---|---|---|---|
| `summary_dir` | str (positional) | — | Directory of change summary JSON files (output of `summarize_changes.py`) |
| `details_dir` | str (positional) | — | Directory of per-initiative JSON files |
| `--dry-run` | flag | — | Print proposed changes without modifying files |

### Clustering & classification

**`src/cluster_all_initiatives.py`** — Clusters feedback across initiatives using sentence embeddings. Supports agglomerative and HDBSCAN algorithms with configurable parameters. Reads from `data/analysis/unit_summaries/`, writes per-scheme output to `data/clustering/<scheme>/`. Scheme names encode the algorithm, model, and parameters (e.g. `agglomerative_google_embeddinggemma-300m_distance_threshold=0.96_linkage=average_...`).

Uses a three-pass architecture: (1) load all initiatives and collect texts, (2) batch-encode all texts at once with multi-GPU support via SentenceTransformer's multi-process pool, (3) cluster each initiative from pre-computed embeddings. Large clusters (>`max_cluster_size`) are recursively sub-clustered with hierarchical labels (e.g. "3.1.2"). Noise points at sub-levels are absorbed into the parent cluster.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--summaries-dir` | str | `data/analysis/unit_summaries` | Directory containing per-initiative unit summary JSONs |
| `-o, --output-dir` | str | `data/clustering` | Directory to write clustering result JSONs |
| `--model` | str | `google/embeddinggemma-300m` | SentenceTransformer model name |
| `--algorithm` | str | `agglomerative` | Clustering algorithm (`agglomerative` or `hdbscan`) |
| `--distance-threshold` | float | `0.96` | AgglomerativeClustering distance_threshold |
| `--linkage` | str | `average` | AgglomerativeClustering linkage (`ward`, `complete`, `average`, `single`) |
| `--sub-cluster-scale` | float | `0.75` | Distance threshold multiplier per recursion level (e.g. 0.75 = 25% reduction each level) |
| `--min-cluster-size` | int | `5` | HDBSCAN `min_cluster_size` parameter |
| `--min-samples` | int | `3` | HDBSCAN `min_samples` parameter |
| `--max-cluster-size` | int | `20` | Clusters larger than this are recursively sub-clustered |
| `--max-depth` | int | `4` | Maximum recursion depth for sub-clustering |
| `--skip-existing` | flag | — | Skip initiatives whose output file already exists |
| `--embeddings-cache-dir` | str | `None` | Directory to cache embeddings per initiative+model (speeds up re-runs) |
| `--cpu-cluster` | flag | — | Force CPU clustering even if cuML is available |
| `--max-items` | int | `0` | Skip initiatives with more feedback items than this (0 = no limit) |

GPU support: embedding via SentenceTransformer (CUDA), clustering optionally via cuML (RAPIDS).

**`src/classify_initiative_and_feedback.py`** — Classifies initiatives and their feedback using vLLM batch inference with `unsloth/gpt-oss-120b`. Takes unit summaries as input, writes per-initiative classification JSONs to `data/classification/`. Uses a multi-step classification hierarchy: Step 0 (unit of analysis — aggregate by initiative), Step 1 (relevance filter — nuclear + energy context), Step 2 (Commission stance: +1 inclusion, 0 neutral, −1 exclusion), Step 3 (legitimacy logic: Technocratic, Input, Procedural-Institutional), Step 4 (dominance rules and output parsing). Valid output labels: `DOES NOT MENTION`, `SUPPORT`, `OPPOSE`, `NEUTRAL`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `unit_summaries_dir` | str (positional) | — | Directory of per-initiative unit summary JSONs |
| `-o, --output` | str | — (required) | Output directory for classification JSONs |
| `--model` | str | `unsloth/gpt-oss-120b` | Model name or path |
| `--max-tokens` | int | `32768` | Max output tokens per classification |
| `--max-model-len` | int | `None` | Max model context length |
| `--temperature` | float | `0.15` | Sampling temperature |
| `--batch-size` | int | `2048` | Number of prompts per inference batch |

**`src/summarize_clusters.py`** — Summarizes feedback clusters using vLLM batch inference with `unsloth/gpt-oss-120b`. Takes clustering output from `data/clustering/<scheme>/`, writes cluster summaries to `data/cluster_summaries/<scheme>/`. Produces titled summaries at three levels: (1) a policy summary from initiative documents, (2) a titled summary per feedback item (feedback text + attachments), (3) recursive bottom-up cluster summaries that greedily combine child summaries within a character budget. Long texts are chunked at sentence boundaries. Title + summary format: title on first line, blank line, then summary body.

| Argument | Type | Default | Description |
|---|---|---|---|
| `clustering_dir` | str (positional) | — | Input directory from `cluster_all_initiatives.py` |
| `-o, --output` | str | — (required) | Output directory for cluster summary JSONs |
| `-f, --filter` | str | `None` | Initiative ID whitelist (newline-delimited file) |
| `--model` | str | `unsloth/gpt-oss-120b` | Model name or path |
| `--max-tokens` | int | `131072` (32768×4) | Max output tokens per summary |
| `--max-model-len` | int | `None` | Max model context length |
| `--temperature` | float | `0.15` | Sampling temperature |
| `--batch-size` | int | `2048` | Number of prompts per inference batch |
| `--chunk-size` | int | `16384` | Max chars per chunk before splitting |
| `--combine-budget` | int | `65536` (16384×4) | Max chars when combining cluster summaries |

Supports resume: skips initiatives whose output already exists. Per-batch result files in `_batches_p1/`, `_batches_p2/`, `_batches_p3/` provide crash recovery. Phase 1 reuses `cluster_feedback_summary` from initiative_details (set by `merge_cluster_feedback_summaries.py`) when available, avoiding redundant LLM calls for feedback items whose summaries survive re-scrapes. Phase 3 uses a content-addressed cache (`_cluster_cache.json`) keyed by SHA-256 of sorted feedback IDs under each cluster label, so clusters with unchanged membership skip LLM inference.

**`src/merge_cluster_feedback_summaries.py`** — Merges cluster summary output back into initiative detail JSON files. Sets three fields: (1) `cluster_feedback_summary` (with `title` and `summary`) on each feedback item, (2) `cluster_policy_summary` (with `title` and `summary`) at the initiative top level, (3) `cluster_summaries` (per-cluster aggregate summaries keyed by cluster label) at the initiative top level. The `cluster_feedback_summary` field survives re-scrapes because the scrape merge strategy preserves derived fields on feedback items when `feedback_text` is unchanged. Top-level derived fields are also preserved.

| Argument | Type | Default | Description |
|---|---|---|---|
| `summary_dir` | str (positional) | — | Directory of cluster summary JSON files (output of `summarize_clusters.py`) |
| `details_dir` | str (positional) | — | Directory of per-initiative JSON files |
| `--dry-run` | flag | — | Print proposed changes without modifying files |

### Webapp index

**`src/build_webapp_index.py`** — Pre-computes the webapp initiative index, aggregate statistics, and per-country drill-down data. Reads from `data/scrape/initiative_details/`, writes:
- `data/webapp/initiative_index.json` — single JSON array of all initiative summaries with pre-computed country counts, user type counts, feedback timeline (20-bucket histogram), feedback IDs, etc.
- `data/webapp/global_stats.json` — aggregate cross-initiative statistics (totals, by-country, by-topic, by-user-type, by-department, by-stage, cross-tabs, monthly time series).
- `data/webapp/country_stats.json` — per-country drill-down statistics: top 20 topics, user type breakdown, top 20 initiatives, 20 most recent feedback items (with attachment links), and top-5 topic timeline.
- `data/webapp/initiative_details/*.json` — stripped copies with `extracted_text`, `extracted_text_without_ocr`, and `extracted_text_before_translation` removed from feedback attachments to reduce file sizes.

Deduplicates initiatives sharing identical sorted feedback ID sets, keeping the one with the most feedback.

| Argument | Type | Default | Description |
|---|---|---|---|
| `details_dir` | str (positional) | — | Path to `initiative_details/` directory |
| `-o, --output` | str | `data/webapp/initiative_index.json` | Output index path |

### Webapp

A Next.js 16 web application (`webapp/`) for browsing initiatives and feedback interactively. See `webapp/AUTH.md` for Google sign-in setup.

**Tech stack:** Next.js 16.1.6, React 19.2.3, Tailwind CSS 4, shadcn/ui (Radix UI 1.4.3 + Lucide React 0.575.0), next-auth v5.0.0-beta.30 (Google OAuth, JWT sessions), react-markdown 10.1.0 + remark-gfm 4.0.1, clsx, class-variance-authority, tailwind-merge.

**Dev dependencies:** TypeScript 5, ESLint 9, shadcn 3.8.5, tw-animate-css 1.4.0.

**npm scripts:** `dev` (start dev server), `build` (production build), `start` (run production), `lint` (ESLint).

#### Pages

- `/` — Initiative index with search, sort, filters, pagination (50/page). Deduplicates initiatives sharing identical feedback IDs. Force-dynamic rendering.
- `/initiative/[id]` — Initiative detail with publications view (documents + feedback) and cluster view. Computes feedback timeline, country/user-type counts. Loads available clustering schemes and pre-loads first scheme's cluster data. Empty/disabled publications shown as compact links.
- `/charts` — Aggregate feedback statistics with country drill-down. Global view shows time series, country/topic/user-type breakdowns, cross-tabs. Country dropdown (`?country=CODE` URL param) switches to per-country view with top topics, top initiatives, topic timeline, and recent feedback with attachment links.

#### API routes

- `/api/auth/[...nextauth]` — NextAuth OAuth handlers
- `/api/clusters/[id]` — Fetch clustering data for an initiative. Query param: `?scheme=<scheme_name>`. Returns `ClusterData` JSON or 404.

#### Key lib files

- **`src/lib/data.ts`** — Server-side data loading with 5-minute in-memory cache TTL (`CACHE_TTL_MS = 300000`). Exports:
  - `getInitiativeIndex()` → `InitiativeSummary[]` (cached)
  - `getGlobalStats()` → `GlobalStats` (cached)
  - `getCountryStats()` → `CountryStats` (cached)
  - `getInitiativeDetail(id)` → `Initiative | null` (no cache, per-file read)
  - `getClusteringSchemesForInitiative(id)` → `string[]` (lists available scheme directories)
  - `getClusterData(id, scheme)` → `ClusterData | null` (loads clustering JSON + merges cluster summaries from initiative detail)

  Helper `findClusteringFile(id, schemeDir)` searches for `{id}.json` or `{id}_{params}.json` in scheme directories.

- **`src/lib/types.ts`** — TypeScript interfaces and utility functions:
  - **Data interfaces**: `Attachment`, `Feedback`, `Document`, `Publication`, `Initiative`, `InitiativeSummary`, `GlobalStats`, `CountryStatsEntry`, `CountryStats`, `ClusterData`, `ClusterNode`, `ClusterSummaryEntry`
  - **Constants**: `USER_TYPE_COLORS`, `USER_TYPE_BAR_COLORS`, `USER_TYPE_SHORT` (color/label mappings), `COUNTRY_BAR_COLORS` (palette), `ISO3_TO_ISO2` (country code mapping)
  - **Utility functions**: `countryToFlag()`, `getUserTypeColor()`, `formatUserType()`, `buildClusterTree()`, `computeClusterStats()`

- **`src/auth.ts`** — Auth.js config with Google provider
- **`src/proxy.ts`** — Session cookie refresh (Next.js 16 middleware proxy)

#### Components (`webapp/src/components/`)

| Component | Description |
|---|---|
| `header.tsx` | Navigation header with user menu |
| `user-menu.tsx` | Sign-in/sign-out menu |
| `initiative-list.tsx` | Initiative table with search, sort, filter, pagination |
| `initiative-card.tsx` | Single initiative card display |
| `initiative-detail.tsx` | Main detail view (publications + cluster tabs) |
| `publication-section.tsx` | Expandable section for each publication |
| `document-card.tsx` | Publication document display |
| `feedback-card.tsx` | Single feedback item with attachments |
| `feedback-list.tsx` | Feedback list with filters and infinite scroll |
| `charts.tsx` | Statistical visualizations (time series, breakdowns, cross-tabs) |
| `cluster-view.tsx` | Cluster visualization and navigation |
| `cluster-node.tsx` | Individual cluster node in tree |
| `cluster-stats-bar.tsx` | Country/user-type bar charts for clusters |
| `expandable-text.tsx` | Collapsible text blocks for summaries |
| `ui/*` | shadcn/ui primitives (Badge, Button, Card, Input, Select) |

#### Authentication setup

Required environment variables in `webapp/.env.local`:

| Variable | Description |
|---|---|
| `AUTH_SECRET` | JWT signing key (generate with `npx auth secret`) |
| `AUTH_GOOGLE_ID` | Google OAuth 2.0 Client ID |
| `AUTH_GOOGLE_SECRET` | Google OAuth 2.0 Client Secret |

The app is fully accessible without signing in (authentication is optional).

- **Client-side session**: `const { data: session } = useSession()` → `session.user.name/email/image`
- **Server-side session**: `const session = await auth()` → `session.user.email`

#### Data paths used at runtime (all relative to `webapp/`)

- `../data/webapp/initiative_index.json` — initiative list for index page
- `../data/webapp/global_stats.json` — aggregate stats for charts page
- `../data/webapp/country_stats.json` — per-country stats for charts country drill-down
- `../data/webapp/initiative_details/*.json` — stripped initiative details for detail pages; also provides `cluster_policy_summary` and `cluster_summaries` for the cluster view (merged by `merge_cluster_feedback_summaries.py`)
- `../data/clustering/<scheme>/*.json` — cluster assignments for cluster view

**Running:** `cd webapp && npm run dev`. Requires `build_webapp_index.py` to have been run first.

### Viewers

**`viewers/viewer.html`** — Standalone HTML file (no dependencies) for interactively browsing per-initiative JSON files in the browser. Supports file loading via browser file picker. Shows initiative metadata, tabbed navigation (Before Feedback, After Feedback, Feedback, Publications), document download links, feedback portal links, attachment download links, expandable text blocks (summaries, extracted text, pre-translation/pre-OCR originals), user type color coding, feedback filtering by type/search/empty, and chunked infinite scroll for large feedback lists.

**`viewers/feedback-viewer.html`** — Standalone HTML file (no dependencies) for browsing clustered feedback results. Loads per-initiative clustering JSON files (from `data/clustering/<scheme>/`) via file picker. Displays cluster metadata (algorithm, model, parameters, silhouette score), nested cluster tree with expandable sub-clusters, per-cluster country and user-type distribution bars, feedback text search, sorting (by size, alphabetical), and individual feedback items with attachments and extracted text.

### Utilities

**`src/text_utils.py`** — Shared text processing library.

| Function | Signature | Description |
|---|---|---|
| `should_skip_text` | `(text: str, label: str = "") -> bool` | Returns True if text should be skipped: empty, PDF binary content, or <50% printable characters (`MIN_PRINTABLE_RATIO = 0.5`). |
| `group_by_char_budget` | `(texts: list, max_chars: int) -> list` | Groups texts greedily so each group's combined length ≤ `max_chars`. Guarantees ≥2 items per group when possible (for recursive combining). Force-pairs singletons if all groups are singletons (ensures convergence). |
| `split_into_chunks` | `(text: str, max_chars: int, label: str = "") -> list` | Splits text at sentence boundaries using regex `(?<=[.!?])\s+`. Falls back to newline splits, then word boundaries, then hard-splits for oversized items. Returns list of chunks ≤ `max_chars` each. |

**`src/inference_utils.py`** — Shared vLLM batch inference helpers.

| Function | Signature | Description |
|---|---|---|
| `build_prefill` | `(encoding, text, prompt_prefix, reasoning_effort, identity_prompt, max_prompt_tokens=None) -> dict \| None` | Builds `prompt_token_ids` prefill dict using `openai_harmony`. Progressively truncates text from end if exceeds `max_prompt_tokens`. Returns dict with `"prompt_token_ids"` key or `None` on error. |
| `extract_final_texts` | `(outputs, encoding) -> list` | Extracts `'final'` channel text from vLLM outputs. Returns list of strings (or `None` for failures). |
| `run_batch_inference` | `(llm, sampling_params, encoding, prompts, prompt_texts, prompt_map, batch_size, batch_dir, summary_cache, batch_num_start=0, label="") -> tuple` | Runs batched vLLM inference with cross-batch + intra-batch deduplication. Writes per-batch JSON files to `batch_dir` for crash recovery/resume. Returns `(summarized_chunks dict, failed_prompts list, stats dict)`. |

Used by `summarize_documents.py`, `summarize_clusters.py`, `summarize_changes.py`, and `translate_attachments.py`.

**`src/print_chunk.py`** — Debug utility to print a specific chunk of a feedback attachment.

| Argument | Type | Default | Description |
|---|---|---|---|
| `spec` | str (positional) | — | Attachment spec, e.g. `"init=12096 fb=503089 att=6276475 chunk=5/15"` |
| `details_dir` | str (positional) | — | Directory of per-initiative JSON files |
| `--chunk-size` | int | `5000` | Max chars per chunk |

## Key dependencies

### Python

- **pymupdf** / **pymupdf4llm** — PDF text extraction with OCR fallback (tesseract at 300 DPI)
- **docx2md** — DOCX text extraction
- **pypandoc** / **pypandoc_binary** — RTF and ODT text extraction
- **easyocr** — GPU-accelerated OCR (CUDA)
- **vllm** — LLM batch inference engine
- **openai_harmony** — Structured prompt encoding for gpt-oss models (reasoning effort, stop tokens, output parsing)
- **sentence_transformers** — Sentence embedding models (e.g. `google/embeddinggemma-300m`)
- **scikit-learn** — AgglomerativeClustering
- **hdbscan** — HDBSCAN clustering
- **cuML** (optional) — GPU-accelerated clustering via RAPIDS

### JavaScript / TypeScript (webapp)

- **next** 16.1.6 — React framework
- **react** 19.2.3 — UI library
- **next-auth** 5.0.0-beta.30 — Authentication (Google OAuth)
- **radix-ui** 1.4.3 — Accessible UI primitives (shadcn/ui)
- **lucide-react** 0.575.0 — Icon library
- **react-markdown** 10.1.0 + **remark-gfm** 4.0.1 — Markdown rendering
- **tailwindcss** 4 — Utility-first CSS
- **clsx**, **class-variance-authority**, **tailwind-merge** — Class name utilities
