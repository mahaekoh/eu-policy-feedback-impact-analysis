# src/ — Pipeline Scripts Reference

All Python scripts for the EU Policy Feedback data pipeline. Scripts are organized below by pipeline stage.

---

## Scraping

### `scrape_eu_initiatives.py`

Scrapes all EU "Have Your Say" initiatives from the Better Regulation API. Fetches all pages in parallel (10 workers). Outputs `data/scrape/eu_initiatives.csv` (flat extracted fields) and `data/scrape/eu_initiatives_raw.json` (full API response).

```bash
python3 src/scrape_eu_initiatives.py
python3 src/scrape_eu_initiatives.py -o custom_output.csv
```

### `scrape_eu_initiative_details.py`

Fetches detailed data for each initiative (publications, feedback, attachments) and extracts text from attached files. Supports PDF (pymupdf with OCR fallback), DOCX (docx2md), DOC (macOS textutil), RTF/ODT (pypandoc), and TXT. Uses 4 thread pools for parallelism. Outputs per-initiative JSON files to `data/scrape/initiative_details/`.

```bash
# Scrape all initiatives (skip those cached within 48 hours)
python3 src/scrape_eu_initiative_details.py

# Scrape a single initiative by ID (prints JSON to stdout)
python3 src/scrape_eu_initiative_details.py 12096

# Force re-fetch everything, cache downloaded docs
python3 src/scrape_eu_initiative_details.py --max-age 0 -c data/scrape/doc_cache
```

---

## Data Quality / Analysis

### `find_missing_initiatives.py`

Reports initiative IDs present in the CSV but missing from `data/scrape/initiative_details/`, or with incomplete feedback data (e.g. 400-status API errors). No arguments.

```bash
python3 src/find_missing_initiatives.py
```

### `find_missing_extracted_text.py`

Scans initiative data for publication documents and feedback attachments that have no `extracted_text`. Prints a table with error info.

```bash
python3 src/find_missing_extracted_text.py data/scrape/initiative_details/
python3 src/find_missing_extracted_text.py data/scrape/initiative_details/ -f filter_ids.txt
```

### `find_initiative_by_pub.py`

Lookup utility: finds which initiative contains a given publication ID.

```bash
python3 src/find_initiative_by_pub.py 123456 data/scrape/initiative_details/
```

### `find_short_pdf_extractions.py`

Finds attachments where `extracted_text` is suspiciously short (<100 chars). Downloads the original files in parallel for OCR processing.

```bash
python3 src/find_short_pdf_extractions.py data/scrape/initiative_details/ -o data/ocr/
```

### `find_non_english_feedback_attachments.py`

Finds feedback attachments where the feedback language is not English. Outputs a JSON file for the translation pipeline.

```bash
python3 src/find_non_english_feedback_attachments.py data/scrape/initiative_details/ -o data/translation/non_english_attachments.json
```

---

## OCR Pipeline

### `ocr_short_pdfs.py`

GPU-accelerated OCR using EasyOCR with CUDA. Renders PDF pages to 300 DPI images, runs OCR. Supports multi-GPU (spawns one subprocess per GPU).

```bash
python3 src/ocr_short_pdfs.py data/ocr/
python3 src/ocr_short_pdfs.py data/ocr/ -o data/ocr/short_pdf_report_ocr.json --languages en,fr,de
```

### `merge_ocr_results.py`

Merges OCR results back into initiative JSON files. Replaces `extracted_text`, preserves original as `extracted_text_without_ocr`.

```bash
python3 src/merge_ocr_results.py data/ocr/short_pdf_report_ocr.json data/scrape/initiative_details/
python3 src/merge_ocr_results.py data/ocr/short_pdf_report_ocr.json data/scrape/initiative_details/ --dry-run
```

---

## Translation Pipeline

### `translate_attachments.py`

Translates non-English feedback attachment texts to English using vLLM batch inference with `unsloth/gpt-oss-120b`. Long documents are chunked at sentence boundaries. Requires GPU.

```bash
python3 src/translate_attachments.py data/translation/non_english_attachments.json \
    -o data/translation/non_english_attachments_translated.json
```

### `merge_translations.py`

Merges translations back into initiative JSON files. Replaces `extracted_text`, preserves original as `extracted_text_before_translation`.

```bash
python3 src/merge_translations.py data/translation/non_english_attachments_translated.json data/scrape/initiative_details/
python3 src/merge_translations.py data/translation/non_english_attachments_translated.json data/scrape/initiative_details/ --dry-run
```

---

## Summarization Pipeline

### `initiative_stats.py`

Analyzes initiative publication/feedback structure. Identifies the first feedback publication, final post-feedback publication, and separates feedback periods. Adds `documents_before_feedback`, `documents_after_feedback`, and `middle_feedback` to each initiative.

```bash
python3 src/initiative_stats.py data/scrape/initiative_details/ -o data/analysis/before_after/
python3 src/initiative_stats.py data/scrape/initiative_details/ -o data/analysis/before_after/ -v
```

### `summarize_documents.py`

Summarizes publication documents and feedback attachments using vLLM batch inference with `unsloth/gpt-oss-120b`. Two-pass architecture: (1) chunk summaries for long texts, (2) recursive combining into a single summary. Requires GPU.

```bash
python3 src/summarize_documents.py data/analysis/before_after/ -o data/analysis/summaries/
python3 src/summarize_documents.py data/analysis/before_after/ -o data/analysis/summaries/ \
    --prev-output data/analysis/summaries/  # reuse existing summaries
```

### `merge_summaries.py`

Merges document and attachment summaries back into initiative detail JSON files. Sets `summary` on each matched document and attachment.

```bash
python3 src/merge_summaries.py data/analysis/summaries/ data/scrape/initiative_details/
python3 src/merge_summaries.py data/analysis/summaries/ data/scrape/initiative_details/ --dry-run
```

### `build_unit_summaries.py`

Consolidates individual document and attachment summaries into per-initiative unified summary fields: `before_feedback_summary`, `after_feedback_summary`, and `combined_feedback_summary`.

```bash
python3 src/build_unit_summaries.py data/analysis/summaries/ -o data/analysis/unit_summaries/
```

### `summarize_changes.py`

Summarizes substantive changes between before- and after-feedback documents using vLLM batch inference. Computes a unified diff and asks the LLM to describe what changed. Requires GPU.

```bash
python3 src/summarize_changes.py data/analysis/unit_summaries/ -o data/analysis/change_summaries/
```

### `merge_change_summaries.py`

Merges change summaries back into initiative detail JSON files. Sets `change_summary` and `diff` at the top level.

```bash
python3 src/merge_change_summaries.py data/analysis/change_summaries/ data/scrape/initiative_details/
python3 src/merge_change_summaries.py data/analysis/change_summaries/ data/scrape/initiative_details/ --dry-run
```

---

## Clustering & Classification

### `cluster_all_initiatives.py`

Clusters feedback across initiatives using sentence embeddings. Supports agglomerative and HDBSCAN algorithms with configurable parameters. Large clusters are recursively sub-clustered. Multi-GPU support for embedding; optional cuML (RAPIDS) for GPU-accelerated clustering.

```bash
python3 src/cluster_all_initiatives.py \
    --summaries-dir data/analysis/unit_summaries \
    -o data/clustering \
    --model google/embeddinggemma-300m \
    --algorithm agglomerative \
    --distance-threshold 0.75 \
    --max-cluster-size 20 \
    --max-depth 3 \
    --embeddings-cache-dir data/embeddings
```

### `classify_initiative_and_feedback.py`

Classifies initiatives and their feedback using vLLM batch inference with `unsloth/gpt-oss-120b`. Multi-step classification hierarchy (relevance, stance, legitimacy logic, dominance). Requires GPU.

```bash
python3 src/classify_initiative_and_feedback.py data/analysis/unit_summaries/ -o data/classification/
```

### `summarize_clusters.py`

Summarizes feedback clusters using vLLM batch inference. Three-phase architecture: (1) policy summary from initiative documents, (2) titled summary per feedback item, (3) recursive bottom-up cluster summaries. Requires GPU.

```bash
python3 src/summarize_clusters.py data/clustering/<scheme>/ -o data/cluster_summaries/<scheme>/
```

### `merge_cluster_feedback_summaries.py`

Merges cluster summary output back into initiative detail JSON files. Sets `cluster_feedback_summary` on each feedback item, `cluster_policy_summary` at initiative top level, and `cluster_summaries` (per-cluster aggregates) at initiative top level.

```bash
python3 src/merge_cluster_feedback_summaries.py data/cluster_summaries/<scheme>/ data/scrape/initiative_details/
python3 src/merge_cluster_feedback_summaries.py data/cluster_summaries/<scheme>/ data/scrape/initiative_details/ --dry-run
```

### `rewrite_cluster_summaries.py`

Rewrites cluster summaries into shorter, format-specific versions (e.g. "reddit" -- concise, punchy summaries) using vLLM batch inference. Single-pass inference. For noise clusters, uses the feedback item's `combined_feedback_summary` from unit summaries as richer input. Requires GPU.

```bash
python3 src/rewrite_cluster_summaries.py \
    data/cluster_summaries/<scheme>/ data/analysis/unit_summaries/ \
    -o data/cluster_rewrites/reddit/<scheme>/ --format reddit
```

### `merge_cluster_rewrites.py`

Merges cluster summary rewrites back into initiative detail JSON files. Sets `cluster_summaries[label]["rewrites"][format] = {title, body}`. Multiple format merges accumulate additively.

```bash
python3 src/merge_cluster_rewrites.py \
    data/cluster_rewrites/reddit/<scheme>/ data/scrape/initiative_details/ \
    --format reddit

# Dry run
python3 src/merge_cluster_rewrites.py \
    data/cluster_rewrites/reddit/<scheme>/ data/scrape/initiative_details/ \
    --format reddit --dry-run
```

---

## Webapp

### `build_webapp_index.py`

Pre-computes the webapp initiative index, aggregate statistics, per-country drill-down data, and stripped initiative detail copies. Reads from `data/scrape/initiative_details/`, writes to `data/webapp/`.

```bash
python3 src/build_webapp_index.py data/scrape/initiative_details/
python3 src/build_webapp_index.py data/scrape/initiative_details/ -o data/webapp/initiative_index.json
```

---

## Utilities

### `text_utils.py`

Shared text processing library. Exports:
- `should_skip_text(text, label)` -- returns True if text is empty, binary, or mostly unprintable.
- `group_by_char_budget(texts, max_chars)` -- groups texts greedily within a character budget.
- `split_into_chunks(text, max_chars, label)` -- splits text at sentence boundaries into chunks.

### `inference_utils.py`

Shared vLLM batch inference helpers. Exports:
- `build_prefill(encoding, text, prompt_prefix, reasoning_effort, identity_prompt, max_prompt_tokens)` -- builds prompt token IDs with progressive truncation.
- `extract_final_texts(outputs, encoding)` -- extracts final channel text from vLLM outputs.
- `run_batch_inference(llm, sampling_params, encoding, prompts, prompt_texts, prompt_map, batch_size, batch_dir, summary_cache, ...)` -- runs batched vLLM inference with deduplication and crash recovery.

### `print_chunk.py`

Debug utility to print a specific chunk of a feedback attachment.

```bash
python3 src/print_chunk.py "init=12096 fb=503089 att=6276475 chunk=5/15" data/scrape/initiative_details/ --chunk-size 5000
```
