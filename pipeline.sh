#!/usr/bin/env bash
# EU Policy Feedback Impact Analysis — Pipeline Orchestrator
#
# Usage:
#   ./pipeline.sh <stage> [extra-args...]
#   ./pipeline.sh list                     # show all stages
#   ./pipeline.sh full                     # full pipeline
#   ./pipeline.sh scrape                   # just scraping
#   ./pipeline.sh deploy                   # rsync code to remote
#   ./pipeline.sh remote summarize         # ssh + run summarize on remote
#   ./pipeline.sh pull summaries           # rsync results back
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load config ──────────────────────────────────────────────────────────────

CONF_FILE="${SCRIPT_DIR}/pipeline.conf"
if [ ! -f "$CONF_FILE" ]; then
    echo "ERROR: pipeline.conf not found."
    echo "Copy pipeline.conf.example to pipeline.conf and fill in your values."
    exit 1
fi
# shellcheck source=pipeline.conf.example
source "$CONF_FILE"

: "${REMOTE_HOST:?}"
: "${REMOTE_DIR:?}"
: "${SSH_KEY:?}"
: "${PYTHON:=python3}"
: "${CLUSTER_SCHEMES:=}"

SSH_CMD="ssh -i ${SSH_KEY} ${REMOTE_HOST}"
RSYNC_SSH="-e ssh -i ${SSH_KEY}"

# ── Helpers ──────────────────────────────────────────────────────────────────

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

stage_start() {
    echo ""
    echo "============================================================"
    echo "[$(timestamp)] START: $1"
    echo "============================================================"
}

stage_end() {
    echo "[$(timestamp)] DONE:  $1"
    echo "============================================================"
}

run_local() {
    local name="$1"; shift
    stage_start "$name"
    "$@"
    stage_end "$name"
}

run_remote() {
    local name="$1"; shift
    stage_start "remote $name"
    # shellcheck disable=SC2029
    $SSH_CMD "cd ${REMOTE_DIR} && $*"
    stage_end "remote $name"
}

rsync_to_remote() {
    local local_path="$1" remote_path="$2"
    rsync -avz -e "ssh -i ${SSH_KEY}" \
        "$local_path" "${REMOTE_HOST}:${REMOTE_DIR}/${remote_path}"
}

rsync_from_remote() {
    local remote_path="$1" local_path="$2"
    mkdir -p "$(dirname "$local_path")"
    rsync -avz --ignore-existing -e "ssh -i ${SSH_KEY}" \
        "${REMOTE_HOST}:${REMOTE_DIR}/${remote_path}" "$local_path"
}

# Parse a scheme name like "agglomerative_google_embeddinggemma-300m_k1=v1_k2=v2"
# into algorithm, model, and key=value parameters.
parse_scheme() {
    local scheme="$1"
    local IFS="_"
    # shellcheck disable=SC2206
    local parts=($scheme)

    SCHEME_ALGO="${parts[0]}"
    # Model name contains a slash: "google/embeddinggemma-300m"
    SCHEME_MODEL="${parts[1]}/${parts[2]}"

    SCHEME_FLAGS=()
    for (( i=3; i<${#parts[@]}; i++ )); do
        local part="${parts[$i]}"
        local key="${part%%=*}"
        local val="${part#*=}"
        # Convert underscore-style to CLI flag: distance_threshold -> --distance-threshold
        local flag="--${key//_/-}"
        SCHEME_FLAGS+=("$flag" "$val")
    done
}

# Build the output subdirectory name for a scheme (same as the scheme name).
scheme_output_dir() {
    echo "data/clustering/$1"
}

# ── Stages ───────────────────────────────────────────────────────────────────

do_scrape() {
    run_local "scrape initiatives" \
        $PYTHON src/scrape_eu_initiatives.py "$@"
    run_local "scrape initiative details" \
        $PYTHON src/scrape_eu_initiative_details.py "$@"
}

do_repair() {
    run_local "repair broken attachments" \
        $PYTHON src/repair_broken_attachments.py -o data/repair/ "$@"
}

do_find_short_pdfs() {
    run_local "find short PDF extractions" \
        $PYTHON src/find_short_pdf_extractions.py \
            -i data/scrape/initiative_details \
            -o data/ocr/ "$@"
}

do_find_nonenglish() {
    run_local "find non-English attachments" \
        $PYTHON src/find_non_english_feedback_attachments.py \
            data/scrape/initiative_details \
            -o data/translation/non_english_attachments.json "$@"
}

do_merge_ocr() {
    run_local "merge OCR results" \
        $PYTHON src/merge_ocr_results.py \
            data/ocr/short_pdf_report_ocr.json \
            data/scrape/initiative_details "$@"
}

do_merge_translations() {
    run_local "merge translations" \
        $PYTHON src/merge_translations.py \
            data/translation/non_english_attachments_translated.json \
            data/scrape/initiative_details "$@"
}

do_analyze() {
    run_local "initiative stats / before-after analysis" \
        $PYTHON src/initiative_stats.py \
            data/scrape/initiative_details \
            -o data/analysis/before_after/ "$@"
}

do_build_summaries() {
    run_local "build unit summaries" \
        $PYTHON src/build_unit_summaries.py \
            data/analysis/summaries/ \
            -o data/analysis/unit_summaries/ "$@"
}

do_cluster() {
    if [ -z "$CLUSTER_SCHEMES" ]; then
        echo "ERROR: CLUSTER_SCHEMES not set in pipeline.conf"
        exit 1
    fi
    for scheme in $CLUSTER_SCHEMES; do
        parse_scheme "$scheme"
        local out_dir
        out_dir="$(scheme_output_dir "$scheme")"
        run_local "cluster ($SCHEME_ALGO)" \
            $PYTHON src/cluster_all_initiatives.py \
                --algorithm "$SCHEME_ALGO" \
                --model "$SCHEME_MODEL" \
                --output-dir "$out_dir" \
                "${SCHEME_FLAGS[@]}" \
                "$@"
    done
}

# ── Deploy / sync ────────────────────────────────────────────────────────────

do_deploy() {
    stage_start "deploy code to remote"
    rsync -avz \
        --exclude='.git/' \
        --exclude='data/' \
        --exclude='__pycache__/' \
        --exclude='.venv/' \
        --exclude='.idea/' \
        --exclude='*.pyc' \
        --exclude='.DS_Store' \
        --exclude='*.ipynb' \
        --exclude='pipeline.conf' \
        -e "ssh -i ${SSH_KEY}" \
        "${SCRIPT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
    stage_end "deploy code to remote"
}

do_push() {
    local target="${1:?Usage: pipeline.sh push <target>}"
    shift
    case "$target" in
        ocr)
            stage_start "push ocr data"
            rsync_to_remote data/ocr/ data/ocr/
            stage_end "push ocr data"
            ;;
        translation)
            stage_start "push translation input"
            rsync_to_remote data/translation/non_english_attachments.json \
                data/translation/non_english_attachments.json
            stage_end "push translation input"
            ;;
        analysis)
            stage_start "push before-after analysis"
            rsync_to_remote data/analysis/before_after/ data/analysis/before_after/
            stage_end "push before-after analysis"
            ;;
        unit-summaries)
            stage_start "push unit summaries"
            rsync_to_remote data/analysis/unit_summaries/ data/analysis/unit_summaries/
            stage_end "push unit summaries"
            ;;
        clustering)
            stage_start "push clustering data"
            rsync_to_remote data/clustering/ data/clustering/
            stage_end "push clustering data"
            ;;
        all)
            do_push ocr
            do_push translation
            do_push analysis
            do_push unit-summaries
            do_push clustering
            ;;
        *)
            echo "ERROR: Unknown push target: $target"
            echo "Valid targets: ocr, translation, analysis, unit-summaries, clustering, all"
            exit 1
            ;;
    esac
}

do_pull() {
    local target="${1:?Usage: pipeline.sh pull <target>}"
    shift
    case "$target" in
        ocr)
            stage_start "pull OCR results"
            rsync_from_remote data/ocr/short_pdf_report_ocr.json \
                data/ocr/short_pdf_report_ocr.json
            stage_end "pull OCR results"
            ;;
        translation)
            stage_start "pull translation results"
            rsync_from_remote data/translation/non_english_attachments_translated.json \
                data/translation/non_english_attachments_translated.json
            rsync_from_remote data/translation/translation_batches/ \
                data/translation/translation_batches/
            stage_end "pull translation results"
            ;;
        summaries)
            stage_start "pull document summaries"
            rsync_from_remote data/analysis/summaries/ data/analysis/summaries/
            stage_end "pull document summaries"
            ;;
        classification)
            stage_start "pull classification results"
            rsync_from_remote data/classification/ data/classification/
            stage_end "pull classification results"
            ;;
        cluster-summaries)
            stage_start "pull cluster summaries"
            rsync_from_remote data/cluster_summaries/ data/cluster_summaries/
            stage_end "pull cluster summaries"
            ;;
        all)
            do_pull ocr
            do_pull translation
            do_pull summaries
            do_pull classification
            do_pull cluster-summaries
            ;;
        *)
            echo "ERROR: Unknown pull target: $target"
            echo "Valid targets: ocr, translation, summaries, classification, cluster-summaries, all"
            exit 1
            ;;
    esac
}

# ── Remote execution ─────────────────────────────────────────────────────────

do_remote() {
    local step="${1:?Usage: pipeline.sh remote <step> [extra-args...]}"
    shift
    case "$step" in
        ocr)
            run_remote "ocr" \
                $PYTHON src/ocr_short_pdfs.py data/ocr/ "$@"
            ;;
        translate)
            run_remote "translate" \
                $PYTHON src/translate_attachments.py \
                    data/translation/non_english_attachments.json \
                    -o data/translation/non_english_attachments_translated.json "$@"
            ;;
        summarize)
            run_remote "summarize" \
                $PYTHON src/summarize_documents.py \
                    data/analysis/before_after/ \
                    -o data/analysis/summaries/ "$@"
            ;;
        classify)
            run_remote "classify" \
                $PYTHON src/classify_initiative_and_feedback.py \
                    data/analysis/unit_summaries/ \
                    -o data/classification/ "$@"
            ;;
        summarize-clusters)
            if [ -z "$CLUSTER_SCHEMES" ]; then
                echo "ERROR: CLUSTER_SCHEMES not set in pipeline.conf"
                exit 1
            fi
            for scheme in $CLUSTER_SCHEMES; do
                local cluster_dir="data/clustering/${scheme}"
                local summary_dir="data/cluster_summaries/${scheme}"
                run_remote "summarize-clusters ($scheme)" \
                    $PYTHON src/summarize_clusters.py \
                        "$cluster_dir" \
                        -o "$summary_dir" "$@"
            done
            ;;
        *)
            echo "ERROR: Unknown remote step: $step"
            echo "Valid steps: ocr, translate, summarize, classify, summarize-clusters"
            exit 1
            ;;
    esac
}

# ── Full pipeline ────────────────────────────────────────────────────────────

do_full() {
    # Scrape
    do_scrape "$@"

    # OCR pipeline: find short extractions → remote OCR → merge back
    do_find_short_pdfs "$@"
    do_deploy
    do_push ocr
    do_remote ocr "$@"
    do_pull ocr
    do_merge_ocr "$@"

    # Translation pipeline: find non-English (after OCR merge) → remote translate → merge back
    do_find_nonenglish "$@"
    do_push translation
    do_remote translate "$@"
    do_pull translation
    do_merge_translations "$@"

    # Analysis
    do_analyze "$@"
    do_push analysis

    # Remote summarization -> pull
    do_remote summarize "$@"
    do_pull summaries

    # Build unit summaries + clustering
    do_build_summaries "$@"
    do_cluster "$@"

    # Push for remote cluster summarization
    do_push unit-summaries
    do_push clustering

    # Remote cluster summarization -> pull
    do_remote summarize-clusters "$@"
    do_pull cluster-summaries

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] FULL PIPELINE COMPLETE"
    echo "============================================================"
}

# ── Stage listing ────────────────────────────────────────────────────────────

do_list() {
    cat <<'EOF'
Available stages:

  Local data prep:
    scrape            Scrape initiatives + details
    repair            Repair broken attachments
    find-short-pdfs   Find short PDF extractions for OCR
    find-nonenglish   Find non-English feedback attachments
    merge-ocr         Merge OCR results into initiative_details
    merge-translations  Merge translations into initiative_details
    analyze           Run initiative_stats (before/after analysis)
    build-summaries   Build unit summaries from document summaries
    cluster           Cluster all initiatives (per configured schemes)

  Deploy / sync:
    deploy            Rsync code to remote
    push <target>     Push data to remote (ocr|translation|analysis|unit-summaries|clustering|all)
    pull <target>     Pull results from remote (ocr|translation|summaries|classification|cluster-summaries|all)

  Remote execution:
    remote <step>     Run step on remote (ocr|translate|summarize|classify|summarize-clusters)

  Composite:
    full              Full pipeline in dependency order

  Extra args are passed through to the underlying Python scripts.
EOF
}

# ── Dispatch ─────────────────────────────────────────────────────────────────

STAGE="${1:?Usage: pipeline.sh <stage> [extra-args...]}"
shift

case "$STAGE" in
    list)               do_list ;;
    scrape)             do_scrape "$@" ;;
    repair)             do_repair "$@" ;;
    find-short-pdfs)    do_find_short_pdfs "$@" ;;
    find-nonenglish)    do_find_nonenglish "$@" ;;
    merge-ocr)          do_merge_ocr "$@" ;;
    merge-translations) do_merge_translations "$@" ;;
    analyze)            do_analyze "$@" ;;
    build-summaries)    do_build_summaries "$@" ;;
    cluster)            do_cluster "$@" ;;
    deploy)             do_deploy ;;
    push)               do_push "$@" ;;
    pull)               do_pull "$@" ;;
    remote)             do_remote "$@" ;;
    full)               do_full "$@" ;;
    *)
        echo "ERROR: Unknown stage: $STAGE"
        echo "Run './pipeline.sh list' to see available stages."
        exit 1
        ;;
esac
