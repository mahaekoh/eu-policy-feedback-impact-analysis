#!/usr/bin/env bash
# One-time migration: move existing data files into the new data/ hierarchy.
# Safe to re-run — skips anything that doesn't exist.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

moved=0

move_if_exists() {
    local src="$1" dst="$2"
    if [ -e "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        mv "$src" "$dst"
        echo "  $src -> $dst"
        moved=$((moved + 1))
    fi
}

# Move contents of a directory (not the directory itself) into a target dir.
# Creates target dir if needed. Skips if source doesn't exist.
move_dir_contents() {
    local src="$1" dst="$2"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        # Use find to handle both files and subdirs at top level
        for item in "$src"/*; do
            [ -e "$item" ] || continue
            mv "$item" "$dst/"
            echo "  $item -> $dst/$(basename "$item")"
            moved=$((moved + 1))
        done
        # Remove now-empty source dir
        rmdir "$src" 2>/dev/null || true
    fi
}

echo "=== Data migration ==="
echo "Project: $PROJECT_DIR"
echo

# ── Create target structure ──
mkdir -p data/scrape data/repair data/ocr data/translation
mkdir -p data/analysis/before_after data/analysis/summaries data/analysis/unit_summaries
mkdir -p data/clustering data/classification data/cluster_summaries
mkdir -p config

# ── Scrape data ──
echo "--- Scrape data ---"
move_if_exists eu_initiatives.csv          data/scrape/eu_initiatives.csv
move_if_exists eu_initiatives_raw.json     data/scrape/eu_initiatives_raw.json
move_if_exists eu_initiatives_cache        data/scrape/eu_initiatives_cache
move_if_exists initiative_details          data/scrape/initiative_details
move_if_exists doc_cache                   data/scrape/doc_cache

# ── Repair data ──
echo "--- Repair data ---"
move_if_exists repair_report.json          data/repair/repair_report.json

# ── OCR data (latest versions) ──
echo "--- OCR data ---"
move_if_exists short_pdf_report.json       data/ocr/short_pdf_report.json
move_if_exists short_pdf_report_ocr.json   data/ocr/short_pdf_report_ocr.json
# short_pdfs/ -> data/ocr/pdfs/
if [ -d short_pdfs ] && [ ! -d data/ocr/pdfs ]; then
    mv short_pdfs data/ocr/pdfs
    echo "  short_pdfs -> data/ocr/pdfs"
    moved=$((moved + 1))
fi

# ── Translation data ──
echo "--- Translation data ---"
move_if_exists translation_tasks.json      data/translation/non_english_attachments.json
move_if_exists non_english_attachments_translated_repair.json data/translation/non_english_attachments_translated.json
move_if_exists translation_output_batches_batches data/translation/translation_batches

# ── Analysis data (latest versions) ──
echo "--- Analysis data ---"
move_dir_contents before_after_analysis_v3 data/analysis/before_after
move_dir_contents summaries_output_v9      data/analysis/summaries
move_dir_contents unit_summaries           data/analysis/unit_summaries

# ── Clustering data ──
echo "--- Clustering data ---"
move_dir_contents clustering_output        data/clustering

# ── Classification data ──
echo "--- Classification data ---"
move_dir_contents classification_results_v4 data/classification

# ── Config files ──
echo "--- Config files ---"
move_if_exists initiative-whitelist-145.txt          config/initiative-whitelist-145.txt
move_if_exists init-no-response-blacklist-19.txt     config/init-no-response-blacklist-19.txt

echo
echo "=== Migration complete: $moved items moved ==="
echo
echo "Old versioned directories left in place (archive/delete manually):"
for d in before_after_analysis before_after_analysis_v2 \
         summaries_output summaries_output_v8.zip summaries_output_v9.zip \
         classification_results_v3 classification_results_v4.zip \
         short_pdfs_v3 short_pdfs_repair short_pdfs_repair_v2_pdfs \
         initiative_details_missingpdfs; do
    [ -e "$d" ] && echo "  $d"
done
