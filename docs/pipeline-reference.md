# Pipeline Reference

Comprehensive reference for the EU Policy Feedback Transparency Platform data pipeline. This document covers all 30 stages, their inputs, outputs, resume behavior, orchestration, and internal parameters.

## Pipeline Overview

The pipeline transforms the European Commission's "Have Your Say" portal data into an interactive exploration platform. It runs 30 stages that alternate between local processing and remote GPU computation, using a 120-billion-parameter language model (`unsloth/gpt-oss-120b`) via vLLM batch inference for translation, summarization, classification, cluster summarization, cluster rewriting, and change detection.

The pipeline is orchestrated by `pipeline.sh`, which manages SSH-based remote execution, parallel rsync transfers, log tailing, crash recovery, and batch cleanup. All LLM stages use file-level resume (skip existing output files) and batch-level crash recovery (per-batch JSON files on disk). The scraper preserves derived fields across re-scrapes, so incremental updates do not require re-running all LLM inference.

## Full Pipeline Stage Table

| # | Stage | Location | Description |
|---|---|---|---|
| 1 | `scrape` | local | Scrape initiative list and per-initiative details |
| 2 | `find-short-pdfs` | local | Find PDFs with suspiciously short extracted text |
| 3 | `deploy` | local->remote | Sync source code to remote GPU host |
| 4 | `push ocr` | local->remote | Upload OCR data to remote |
| 5 | `remote ocr` | remote GPU | OCR scanned PDFs (EasyOCR, multi-GPU) |
| 6 | `pull ocr` | remote->local | Download OCR results |
| 7 | `merge-ocr` | local | Merge OCR results back into initiative JSONs |
| 8 | `find-nonenglish` | local | Find non-English feedback attachments |
| 9 | `push translation` | local->remote | Upload translation data to remote |
| 10 | `remote translate` | remote GPU | Translate to English (120B LLM) |
| 11 | `pull translation` | remote->local | Download translation results |
| 12 | `merge-translations` | local | Merge translations back into initiative JSONs |
| 13 | `analyze` | local | Compute before/after feedback structure |
| 14 | `push analysis` | local->remote | Upload analysis data to remote |
| 15 | `remote summarize` | remote GPU | Summarize documents and feedback (120B LLM) |
| 16 | `pull summaries` | remote->local | Download summaries |
| 17 | `build-summaries` | local | Consolidate per-document summaries into unit summaries |
| 18 | `push unit-summaries` | local->remote | Upload unit summaries to remote |
| 19 | `remote cluster` | remote GPU | Cluster feedback (sentence embeddings, multi-GPU) |
| 20 | `pull clustering` | remote->local | Download clustering results |
| 21 | `build-index` | local | Pre-compute webapp index and statistics |
| 22 | `push clustering` | local->remote | Upload clustering data to remote |
| 23 | `remote summarize-clusters` | remote GPU | Summarize clusters (120B LLM, per scheme) |
| 24 | `pull cluster-summaries` | remote->local | Download cluster summaries |
| 25 | `merge-cluster-feedback-summaries` | local | Merge cluster summaries back (per scheme) |
| 26 | `remote rewrite-clusters` | remote GPU | Rewrite cluster summaries into shorter formats (120B LLM, per scheme) |
| 27 | `merge-cluster-rewrites` | local | Merge cluster rewrites back (per format, per scheme) |
| 28 | `remote summarize-changes` | remote GPU | Detect before/after document changes (120B LLM) |
| 29 | `pull change-summaries` | remote->local | Download change summaries |
| 30 | `merge-change-summaries` | local | Merge change summaries back into initiative JSONs |

All LLM stages use `unsloth/gpt-oss-120b` via vLLM batch inference. LLM stages (summarize, classify, summarize-clusters, rewrite-clusters, summarize-changes) use file-level resume -- they skip initiatives whose output already exists and don't load the model if there's no work. All LLM stages also write per-batch result files for crash recovery within a run. See [Resume and Recovery Patterns](#resume-and-recovery-patterns) for details.

## Pipeline Step Reference Table

Each pipeline step reads from the previous step's output and writes its own. The table below shows every step, its inputs, outputs, resume behavior, and whether output files are overwritten on re-runs.

| Step | Script | Input | Output | Resume behavior | Overwritten on re-run? |
|---|---|---|---|---|---|
| **Scrape list** | `scrape_eu_initiatives.py` | EU Better Regulation API | `data/scrape/eu_initiatives.csv`, `data/scrape/eu_initiatives_raw.json` | None -- always re-fetches all pages | Yes, regenerated every run |
| **Scrape details** | `scrape_eu_initiative_details.py` | API + CSV list | `data/scrape/initiative_details/{id}.json` | Skips initiatives cached within `--max-age` hours (default 48). Terminal stages (SUSPENDED, ABANDONED) and closed ADOPTION_WORKFLOW initiatives are never re-checked. Corrupt JSON files are detected and re-fetched from scratch. | Stale files are re-fetched with a **merge strategy** that preserves derived fields (`summary`, `extracted_text_without_ocr`, `extracted_text_before_translation`, `cluster_feedback_summary`, `change_summary`, `diff`, `cluster_policy_summary`, `cluster_summaries`) on documents/attachments whose source material (pages, size_bytes, document_id, feedback_text) hasn't changed. |
| **Find short PDFs** | `find_short_pdf_extractions.py` | `data/scrape/initiative_details/` | `data/ocr/short_pdf_report.json`, `data/ocr/pdfs/{filename}` | None | Yes |
| **OCR** | `ocr_short_pdfs.py` | `data/ocr/short_pdf_report.json` + `data/ocr/pdfs/` | `data/ocr/short_pdf_report_ocr.json` | None | Yes, single file regenerated |
| **Merge OCR** | `merge_ocr_results.py` | OCR report + `initiative_details/` | Updates `data/scrape/initiative_details/{id}.json` **in place** | None -- applies all records every run | In-place mutation: sets `extracted_text` and preserves original as `extracted_text_without_ocr` |
| **Find non-English** | `find_non_english_feedback_attachments.py` | `data/scrape/initiative_details/` | `data/translation/non_english_attachments.json` | None | Yes |
| **Translate** | `translate_attachments.py` | `data/translation/non_english_attachments.json` | `data/translation/non_english_attachments_translated.json` | Per-batch file resume: existing batch files in `_batches/` are loaded instead of re-running inference. | Yes, combined output file regenerated. Batch files are append-only. |
| **Merge translations** | `merge_translations.py` | Translation output + `initiative_details/` | Updates `data/scrape/initiative_details/{id}.json` **in place** | None | In-place mutation: sets `extracted_text` and preserves original as `extracted_text_before_translation`. Skips "NO TRANSLATION NEEDED" records. |
| **Analyze** | `initiative_stats.py` | `data/scrape/initiative_details/` | `data/analysis/before_after/{id}.json` | None -- always regenerates all files | Yes |
| **Summarize docs** | `summarize_documents.py` | `data/analysis/before_after/` | `data/analysis/summaries/{id}.json` | **File-level**: skips initiatives whose output file already exists. **Item-level**: with `--prev-output`, reuses summaries from a previous output directory for items with unchanged text. **Batch-level**: per-batch files in `_batches_pass1/` and `_batches_pass2/` provide crash recovery within a run. Model is not loaded if there is no work. | No -- files are immutable once written. To regenerate, delete the output file. |
| **Build unit summaries** | `build_unit_summaries.py` | `data/analysis/summaries/` | `data/analysis/unit_summaries/{id}.json` | None -- always regenerates all files | Yes |
| **Cluster feedback** | `cluster_all_initiatives.py` | `data/analysis/unit_summaries/` | `data/clustering/{scheme}/{id}_{algo}_{model}_{params}.json`, `data/embeddings/{model}/{id}.npz` | Optional `--skip-existing` flag skips initiatives with existing output. **Not passed by pipeline.sh** -- all files are regenerated every run. Embeddings are hash-validated: cached embeddings are reused only if text hashes match. | **Yes -- files are overwritten every run** (unless `--skip-existing` is used). Embeddings overwritten when initiative data changes. |
| **Build webapp index** | `build_webapp_index.py` | `data/scrape/initiative_details/` | `data/webapp/initiative_index.json`, `data/webapp/global_stats.json`, `data/webapp/country_stats.json`, `data/webapp/initiative_details/{id}.json` | None -- always regenerates | Yes. Stripped copies (no `extracted_text`, `extracted_text_without_ocr`, `extracted_text_before_translation` on feedback attachments). |
| **Summarize clusters** | `summarize_clusters.py` | `data/clustering/{scheme}/` | `data/cluster_summaries/{scheme}/{id}.json` | **File-level**: skips initiatives whose output file already exists. **Item-level**: reuses `cluster_feedback_summary` from `initiative_details` when available (Phase 1). **Cache-level**: content-addressed cache (`_cluster_cache.json`) keyed by SHA-256 of sorted feedback IDs skips clusters with unchanged membership (Phase 3). **Batch-level**: per-batch files in `_batches_p1/`, `_batches_p2/`, `_batches_p3/` provide crash recovery. | No -- files are immutable once written. `_cluster_cache.json` is updated incrementally. |
| **Merge cluster summaries** | `merge_cluster_feedback_summaries.py` | Cluster summaries + `initiative_details/` | Updates `data/scrape/initiative_details/{id}.json` **in place** | None | In-place mutation: sets `cluster_feedback_summary` on each feedback item, `cluster_policy_summary` and `cluster_summaries` at initiative top level. |
| **Rewrite clusters** | `rewrite_cluster_summaries.py` | `data/cluster_summaries/{scheme}/` + `data/analysis/unit_summaries/` | `data/cluster_rewrites/{format}/{scheme}/{id}.json` | **File-level**: skips existing output files. **Batch-level**: per-batch files in `_batches/` provide crash recovery. Model is not loaded if there is no work. | No -- immutable once written. |
| **Merge cluster rewrites** | `merge_cluster_rewrites.py` | Cluster rewrites + `initiative_details/` | Updates `data/scrape/initiative_details/{id}.json` **in place** | None | In-place mutation: sets `cluster_summaries[label].rewrites[format]` = `{title, body}`. Additive across formats. |
| **Summarize changes** | `summarize_changes.py` | `data/analysis/unit_summaries/` | `data/analysis/change_summaries/{id}.json` | **File-level**: skips initiatives whose output file already exists. **Batch-level**: per-batch files in `_batches/` and `_batches_combine/` provide crash recovery. Model is not loaded if there is no work. | No -- files are immutable once written. To regenerate, delete the output file. |
| **Merge change summaries** | `merge_change_summaries.py` | Change summaries + `initiative_details/` | Updates `data/scrape/initiative_details/{id}.json` **in place** | None | In-place mutation: sets `change_summary` and `diff` at initiative top level. |
| **Classify** | `classify_initiative_and_feedback.py` | `data/analysis/unit_summaries/` | `data/classification/{id}.json` | **File-level**: skips initiatives whose output file already exists. **Batch-level**: per-batch files provide crash recovery. Model is not loaded if there is no work. | No -- files are immutable once written. To regenerate, delete the output file. |

## Resume and Recovery Patterns

The pipeline uses three levels of resume to avoid redundant work:

### 1. File-level resume

Used by: `summarize_documents.py`, `summarize_changes.py`, `classify_initiative_and_feedback.py`, `summarize_clusters.py`, `rewrite_cluster_summaries.py`

If the output file for an initiative already exists in the output directory, the initiative is skipped entirely. The LLM model is not loaded if all work is already done. To force regeneration, delete the specific output file(s).

### 2. Batch-level crash recovery

Used by: all vLLM-based scripts (`translate_attachments.py`, `summarize_documents.py`, `summarize_changes.py`, `classify_initiative_and_feedback.py`, `summarize_clusters.py`, `rewrite_cluster_summaries.py`)

Within a single run, inference results are written to per-batch JSON files (`_batches*/batch_NNNN.json`). If the process crashes mid-run, restarting loads completed batches from disk and resumes from where it left off. Batch directories are auto-cleaned by `pipeline.sh` after successful remote runs.

### 3. Content-level caching

Used by: `summarize_clusters.py` only

A content-addressed cache (`_cluster_cache.json`) maps SHA-256 hashes of sorted feedback ID sets to their cluster summaries. When cluster membership hasn't changed between runs, the cached summary is reused without LLM inference.

### Recovery command

`./pipeline.sh recover` pulls all available output files from the remote host and merges them into `initiative_details` locally. This is useful when a pipeline run crashes partway through -- intermediate results that were already written to disk on the remote can still be recovered and merged. Each pull and merge step is run with error tolerance so that missing remote directories (from stages that never ran) don't abort the recovery.

## Derived Field Preservation

The scraper (`scrape_eu_initiative_details.py`) uses a merge strategy when re-fetching stale initiatives. For each document and feedback attachment, if the source material (page count, file size, document ID, or feedback text) hasn't changed, all derived fields are preserved from the previous version.

| Derived field | Set by | Preserved on |
|---|---|---|
| `extracted_text_without_ocr` | `merge_ocr_results.py` | Attachments (when source unchanged) |
| `extracted_text_before_translation` | `merge_translations.py` | Attachments (when source unchanged) |
| `summary` | `summarize_documents.py` | Documents and attachments (when source unchanged) |
| `cluster_feedback_summary` | `merge_cluster_feedback_summaries.py` | Feedback items (when `feedback_text` unchanged) |
| `change_summary`, `diff` | `merge_change_summaries.py` | Initiative top level |
| `cluster_policy_summary`, `cluster_summaries` | `merge_cluster_feedback_summaries.py` | Initiative top level |
| `cluster_summaries[].rewrites` | `merge_cluster_rewrites.py` | Initiative top level (nested within `cluster_summaries`, preserved as part of that top-level derived field) |

This means running the full pipeline, re-scraping, then running it again does not require re-doing all LLM inference -- only initiatives with genuinely changed source data need reprocessing.

## Pull Behavior

When pulling results from remote, `pipeline.sh` uses different rsync strategies depending on whether output files are immutable:

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
| `cluster-rewrites` | Skip existing | File-level resume makes output files immutable |
| `change-summaries` | Skip existing | File-level resume makes output files immutable |
| `initiative-details` | Overwrite | Remote has authoritative copy with all merges applied |
| `webapp` | Overwrite | Regenerated by `build_webapp_index.py` on every run |
| `logs` | Overwrite | Log files from remote |

**Skip existing** uses `rsync --ignore-existing`, which skips files already present locally. This is safe because file-level resume makes LLM output files immutable once written.

**Overwrite** uses plain rsync without `--ignore-existing`, so local copies are always updated to match the remote.

Both strategies use parallel rsync (4 streams by default) with `--files-from` chunking for efficient large-directory transfers.

## Output Files Overview

```
data/
  scrape/
    eu_initiatives.csv                    # Overwritten every scrape run
    eu_initiatives_raw.json               # Overwritten every scrape run
    initiative_details/{id}.json          # Mutated in-place by merge scripts;
                                          #   re-scraped with field preservation
    doc_cache/{id}/pub{pub}_doc{doc}_{name}  # Cached downloaded files, never deleted
  ocr/
    short_pdf_report.json                 # Overwritten by find_short_pdf_extractions
    pdfs/{filename}                       # Downloaded PDFs for OCR
    short_pdf_report_ocr.json             # Overwritten by ocr_short_pdfs
  translation/
    non_english_attachments.json                    # Overwritten by find_non_english
    non_english_attachments_translated.json         # Overwritten by translate
    non_english_attachments_translated_batches/     # Append-only batch files
  analysis/
    before_after/{id}.json                # Overwritten by initiative_stats
    summaries/{id}.json                   # Immutable (file-level resume)
      _batches_pass1/                     # Crash recovery (auto-cleaned)
      _batches_pass2/                     # Crash recovery (auto-cleaned)
    unit_summaries/{id}.json              # Overwritten by build_unit_summaries
    change_summaries/{id}.json            # Immutable (file-level resume)
      _batches/                           # Crash recovery (auto-cleaned)
      _batches_combine/                   # Crash recovery (auto-cleaned)
  clustering/{scheme}/
    {id}_{algo}_{model}_{params}.json     # Overwritten every clustering run
  embeddings/{model}/{id}.npz             # Overwritten when data changes (hash-validated)
  classification/{id}.json                # Immutable (file-level resume)
  cluster_summaries/{scheme}/
    {id}.json                             # Immutable (file-level resume)
    _cluster_cache.json                   # Content-addressed cache (updated incrementally)
    _batches_p1/                          # Crash recovery (auto-cleaned)
    _batches_p2/                          # Crash recovery (auto-cleaned)
    _batches_p3/                          # Crash recovery (auto-cleaned)
  cluster_rewrites/{format}/{scheme}/
    {id}.json                             # Immutable (file-level resume)
    _batches/                             # Crash recovery (auto-cleaned)
  webapp/
    initiative_index.json                 # Overwritten by build_webapp_index
    global_stats.json                     # Overwritten by build_webapp_index
    country_stats.json                    # Overwritten by build_webapp_index
    initiative_details/{id}.json          # Overwritten (stripped copies, no extracted_text)
```

## Pipeline Orchestration

`pipeline.sh` orchestrates the full pipeline. Copy `pipeline.conf.example` to `pipeline.conf` and fill in your remote GPU host details.

### Configuration (`pipeline.conf`)

| Variable | Default | Description |
|---|---|---|
| `REMOTE_HOST` | -- | SSH host (e.g. `user@gpu-host`) |
| `REMOTE_DIR` | -- | Remote working directory |
| `SSH_KEY` | -- | Path to SSH private key |
| `PYTHON` | `python3` | Python executable on remote |
| `CLUSTER_SCHEMES` | -- | Space-separated clustering scheme names. Each name encodes algorithm and parameters; `pipeline.sh` parses them into CLI flags. Example: `"agglomerative_google_embeddinggemma-300m_distance_threshold=0.75_linkage=average_max_cluster_size=20_max_depth=3_sub_cluster_scale=0.75"` |

### Commands

```bash
./pipeline.sh setup                    # Install local Python deps (uv sync) + Hugging Face login
./pipeline.sh setup-remote             # Deploy code + install remote GPU deps (pip) + HF login
./pipeline.sh list                     # Show all stages
./pipeline.sh full                     # Run entire pipeline end-to-end (all 30 stages)
./pipeline.sh <stage>                  # Run a single stage
./pipeline.sh deploy                   # Sync src/ to remote
./pipeline.sh remote <step>            # Run GPU step on remote
./pipeline.sh push <target>            # Upload data to remote
./pipeline.sh pull <target>            # Download results from remote
./pipeline.sh logs                     # List recent remote logs
./pipeline.sh logs tail <step>         # Tail a specific step's log
./pipeline.sh clean-batches <target>   # Delete batch recovery files on remote
./pipeline.sh recover                  # Pull all outputs from remote + merge locally + rebuild index
```

### Push targets

`initiative-details`, `ocr`, `translation`, `analysis`, `unit-summaries`, `clustering`, `cluster-rewrites`, `all`

### Pull targets

`initiative-details`, `ocr`, `translation`, `summaries`, `classification`, `clustering`, `embeddings`, `cluster-summaries`, `cluster-rewrites`, `change-summaries`, `webapp`, `logs`, `all`

### Remote GPU steps

`ocr`, `translate`, `summarize`, `classify`, `cluster`, `summarize-clusters`, `rewrite-clusters`, `summarize-changes`

### Composite remote pipeline steps

These chain multiple operations (find/merge with GPU steps) into a single remote execution, useful for the full pipeline:

| Composite step | Operations chained |
|---|---|
| `ocr-pipeline` | Find short PDFs + GPU OCR + merge OCR |
| `translate-pipeline` | Find non-English + GPU translate + merge translations |
| `summarize-pipeline` | Analyze + GPU summarize + build unit summaries |
| `cluster-summarize-pipeline` | GPU cluster summarize + merge cluster summaries (per scheme) |
| `rewrite-clusters-pipeline` | GPU rewrite clusters + merge rewrites (per format, per scheme) |
| `change-summarize-pipeline` | GPU change summarize + merge change summaries |
| `build-index` | Build webapp index on remote |

### Clean-batches targets

`summaries`, `cluster-summaries`, `cluster-rewrites`, `change-summaries`, `translation`, `all`

### Remote execution model

- GPU jobs run via `nohup` with stdout/stderr piped to log files under `logs/` on the remote host
- Long-running jobs survive SSH disconnects; the local terminal tails the log in real-time
- Exit code is read from a `.exit` status file when the job completes
- Batch recovery directories (`_batches*`) are auto-cleaned after successful runs
- Push/pull operations use parallel rsync (4 streams) with `--files-from` chunking for efficient large-directory transfers
- Pull behavior varies by target: immutable LLM outputs use `--ignore-existing` (skip already-downloaded files), while targets overwritten every run (clustering, embeddings, single files) use plain rsync to keep local copies current

### Full pipeline execution flow

When running `./pipeline.sh full`, the pipeline executes these phases in order:

1. **Local scraping** -- scrape initiative list and per-initiative details
2. **Deploy + push** -- sync source code and initiative data to remote
3. **OCR pipeline** (remote) -- find short PDFs, run GPU OCR, merge results
4. **Translation pipeline** (remote) -- find non-English, run GPU translation, merge results
5. **Summarization pipeline** (remote) -- analyze before/after structure, run GPU summarization, build unit summaries
6. **Clustering** (remote) -- run GPU clustering with sentence embeddings (multi-GPU)
7. **Cluster summarization pipeline** (remote) -- run GPU cluster summarization + merge (per scheme)
8. **Cluster rewrite pipeline** (remote) -- run GPU cluster rewriting + merge (per format, per scheme)
9. **Change summarization pipeline** (remote) -- run GPU change summarization + merge
10. **Build webapp index** (remote) -- pre-compute initiative index and statistics
11. **Pull results** -- download initiative details, clustering, embeddings, webapp data, summaries, cluster summaries, cluster rewrites, change summaries

## Internal Parameters Reference

The pipeline uses many internal parameters that are not exposed as CLI arguments but affect performance, quality, and resource usage. These are documented here for tuning and debugging.

### Scraping

| Parameter | Value | File | Description |
|---|---|---|---|
| `PAGE_SIZE` | 10 | `scrape_eu_initiatives.py` | API page size for initiative list pagination |
| `PAGE_WORKERS` | 10 | `scrape_eu_initiatives.py` | Parallel workers for fetching initiative list pages |
| `FEEDBACK_PAGE_SIZE` | 500 | `scrape_eu_initiative_details.py` | API page size for feedback pagination |
| `INITIATIVE_WORKERS` | 20 | `scrape_eu_initiative_details.py` | Thread pool size for initiative detail fetching |
| `FEEDBACK_WORKERS` | 20 | `scrape_eu_initiative_details.py` | Thread pool size for feedback orchestration |
| `PDF_WORKERS` | 40 | `scrape_eu_initiative_details.py` | Thread pool size for PDF/attachment text extraction |
| `PAGE_WORKERS` | 80 | `scrape_eu_initiative_details.py` | Thread pool size for feedback page fetching |
| `DOWNLOAD_WORKERS` | 20 | `find_short_pdf_extractions.py` | Thread pool size for PDF downloads |
| API timeout | 30s | `scrape_eu_initiatives.py` | HTTP timeout for initiative list API calls |
| API timeout | 60s | `scrape_eu_initiative_details.py` | HTTP timeout for initiative detail API calls |
| Download timeout | 120s | `scrape_eu_initiative_details.py` | HTTP timeout for file downloads |
| Retry attempts | 3 | both scrape scripts | Number of retries with exponential backoff (2, 4s) |
| Slow request threshold | 5s | `scrape_eu_initiative_details.py` | Requests slower than this are logged as warnings |
| `--max-age` default | 48h | `scrape_eu_initiative_details.py` | Hours before re-fetching a cached initiative (CLI arg) |

### Text extraction and filtering

| Parameter | Value | File | Description |
|---|---|---|---|
| `OCR_DPI` | 300 | `scrape_eu_initiative_details.py`, `ocr_short_pdfs.py` | DPI for rendering PDF pages for OCR |
| `OCR_MIN_CHARS` | 100 | `scrape_eu_initiative_details.py` | Extracted text shorter than this triggers OCR fallback |
| `OCR_MIN_FILE_BYTES` | 2048 | `scrape_eu_initiative_details.py` | Files smaller than this skip OCR (likely empty or corrupt) |
| `MIN_PRINTABLE_RATIO` | 0.5 | `text_utils.py` | Texts with fewer than 50% printable characters are skipped as garbled |

### LLM inference (shared across all vLLM-based scripts)

| Parameter | Value | File(s) | Description |
|---|---|---|---|
| Default model | `unsloth/gpt-oss-120b` | all LLM scripts | Default model for translation, summarization, classification, cluster summarization, rewriting |
| Temperature | 0.15 | all LLM scripts | Sampling temperature (low for deterministic output) |
| Token-to-char ratio | 4.8 | `inference_utils.py` | Estimated characters per token for prompt truncation (with 20% margin) |
| Minimum trim | 200 chars | `inference_utils.py` | Minimum characters to remove per truncation step |

### Summarization

| Parameter | Value | File | Description |
|---|---|---|---|
| `CHUNK_SIZE` | 16,384 | `summarize_documents.py` | Max characters per text chunk before sentence-boundary splitting |
| `COMBINE_BUDGET` | 65,536 | `summarize_documents.py` | Max characters when grouping chunk summaries for recursive combining |
| `MAX_OUTPUT_TOKENS` | 131,072 | `summarize_documents.py` | Max output tokens per LLM call (32768 x 4) |
| `INITIATIVE_BATCH_SIZE` | 128 | `summarize_documents.py` | Number of initiative files loaded into memory at once |
| `--batch-size` default | 8,192 | `summarize_documents.py` | Prompts per vLLM inference batch |

### Translation

| Parameter | Value | File | Description |
|---|---|---|---|
| `CHUNK_SIZE` | 16,384 | `translate_attachments.py` | Max characters per translation chunk |
| `MAX_OUTPUT_TOKENS` | 131,072 | `translate_attachments.py` | Max output tokens per LLM call |
| `--batch-size` default | 4,096 | `translate_attachments.py` | Prompts per vLLM inference batch |
| Merge chunk size | 5,000 | `merge_translations.py` | Chunk size for reassembling translated text from batch files |

### Change summarization

| Parameter | Value | File | Description |
|---|---|---|---|
| `CHUNK_SIZE` | 16,384 | `summarize_changes.py` | Max characters per chunk for large diffs |
| `COMBINE_BUDGET` | 65,536 | `summarize_changes.py` | Max characters when combining change summaries |
| `MAX_OUTPUT_TOKENS` | 131,072 | `summarize_changes.py` | Max output tokens per LLM call |
| `--batch-size` default | 2,048 | `summarize_changes.py` | Prompts per vLLM inference batch |

### Cluster summarization

| Parameter | Value | File | Description |
|---|---|---|---|
| `CHUNK_SIZE` | 16,384 | `summarize_clusters.py` | Max characters per chunk |
| `COMBINE_BUDGET` | 65,536 | `summarize_clusters.py` | Max characters when combining cluster summaries |
| `MAX_OUTPUT_TOKENS` | 131,072 | `summarize_clusters.py` | Max output tokens per LLM call |
| `--batch-size` default | 8,192 | `summarize_clusters.py` | Prompts per vLLM inference batch |
| `--min-noise-summarize-chars` | 1,000 | `summarize_clusters.py` | Minimum text length for noise feedback to get summarized |

### Cluster rewriting

| Parameter | Value | File | Description |
|---|---|---|---|
| `MAX_OUTPUT_TOKENS` | 4,096 | `rewrite_cluster_summaries.py` | Max output tokens (rewrites are short) |
| `--batch-size` default | 2,048 | `rewrite_cluster_summaries.py` | Prompts per vLLM inference batch |

Available formats: `reddit` (concise, punchy summaries -- clear, direct, no fluff). The format registry is defined in the `FORMATS` dict within `rewrite_cluster_summaries.py`. For noise clusters (label starting with `-1:`), the rewriter prefers `combined_feedback_summary` from unit summaries as richer input, falling back to the cluster summary if not found.

### Classification

| Parameter | Value | File | Description |
|---|---|---|---|
| `MAX_OUTPUT_TOKENS` | 32,768 | `classify_initiative_and_feedback.py` | Max output tokens (lower than other tasks -- classification outputs are short) |
| `--batch-size` default | 1,024 | `classify_initiative_and_feedback.py` | Prompts per vLLM inference batch |

### Clustering

| Parameter | Value | File | Description |
|---|---|---|---|
| `SILHOUETTE_SAMPLE_SIZE` | 2,000 | `cluster_all_initiatives.py` | Sample size for silhouette score (avoids O(n^2) on large initiatives) |
| Embedding batch size | 256 | `cluster_all_initiatives.py` | Batch size for SentenceTransformer encoding |

### Webapp index building

| Parameter | Value | File | Description |
|---|---|---|---|
| `TIMELINE_BUCKETS` | 20 | `build_webapp_index.py` | Number of time buckets for feedback timeline histograms |
| Top topics for time series | 10 | `build_webapp_index.py` | Number of topics shown in global time series |
| Top countries | 15 | `build_webapp_index.py` | Number of countries in global stats breakdown |
| Top topics per country | 20 | `build_webapp_index.py` | Number of topics in per-country drill-down |
| Topic timeline | 5 | `build_webapp_index.py` | Number of topics in per-country topic timeline |

### Webapp

| Parameter | Value | File | Description |
|---|---|---|---|
| `CACHE_TTL_MS` | 300,000 (5 min) | `data.ts` | In-memory cache TTL for initiative index, global stats, and country stats |
| `ITEMS_PER_PAGE` | 50 | `initiative-list.tsx` | Initiatives per page on the index |
| `CHUNK_SIZE` | 50 | `feedback-list.tsx` | Feedback items loaded per infinite scroll chunk |
| `TIMELINE_BUCKETS` | 20 | `publication-section.tsx`, `initiative/[id]/page.tsx` | Time buckets for feedback sparklines |
| `SPARKLINE_BUCKETS` | 20 | `cluster-view.tsx` | Time buckets for cluster sparklines |
| `INITIAL_SHOW` | 5 | `cluster-node.tsx` | Feedback items shown per cluster before "show more" |

### Pipeline orchestration

| Parameter | Value | File | Description |
|---|---|---|---|
| `PARALLEL_JOBS` | 4 | `pipeline.sh` | Number of parallel rsync streams for push/pull operations |
