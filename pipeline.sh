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
    local log_name
    log_name="$(echo "$name" | tr ' ()' '_')"
    local log_file="logs/${log_name}_$(date +%Y%m%d_%H%M%S).log"
    local status_file="${log_file}.exit"
    stage_start "remote $name"

    # Launch via nohup so SSH disconnects don't kill the process.
    # The wrapper runs the command, captures its exit code to a status file,
    # then exits. We get the wrapper's PID back.
    local remote_cmd="$*"
    # shellcheck disable=SC2029
    local remote_pid
    remote_pid=$($SSH_CMD "cd ${REMOTE_DIR} && mkdir -p logs \
        && nohup bash -c '${remote_cmd} > ${log_file} 2>&1; echo \$? > ${status_file}' \
           </dev/null >/dev/null 2>&1 & echo \$!")
    echo "Remote PID: ${remote_pid}, log: ${REMOTE_DIR}/${log_file}"

    # Tail the log locally until the remote process exits.
    # Poll for the status file since --pid doesn't work across SSH sessions.
    # shellcheck disable=SC2029
    $SSH_CMD "tail -n 20 -f ${REMOTE_DIR}/${log_file} &
        TAIL_PID=\$!
        while [ ! -f ${REMOTE_DIR}/${status_file} ]; do sleep 2; done
        sleep 1
        kill \$TAIL_PID 2>/dev/null
        wait \$TAIL_PID 2>/dev/null" || true

    # Read exit code from status file
    # shellcheck disable=SC2029
    local exit_code
    exit_code=$($SSH_CMD "cat ${REMOTE_DIR}/${status_file} 2>/dev/null || echo 1")
    exit_code=$(echo "$exit_code" | tr -d '[:space:]')

    if [ "$exit_code" != "0" ]; then
        echo "ERROR: remote $name exited with code ${exit_code}"
        echo "Full log: ${REMOTE_HOST}:${REMOTE_DIR}/${log_file}"
        return 1
    fi

    stage_end "remote $name"
}

rsync_to_remote() {
    local local_path="$1" remote_path="$2"
    if [[ "$remote_path" == */ ]]; then
        $SSH_CMD "mkdir -p ${REMOTE_DIR}/${remote_path}"
    else
        $SSH_CMD "mkdir -p $(dirname "${REMOTE_DIR}/${remote_path}")"
    fi
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

do_find_short_pdfs() {
    run_local "find short PDF extractions" \
        $PYTHON src/find_short_pdf_extractions.py \
            data/scrape/initiative_details \
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

do_build_index() {
    run_local "build webapp index" \
        $PYTHON src/build_webapp_index.py \
            data/scrape/initiative_details \
            -o data/webapp/initiative_index.json "$@"
}

# ── Deploy / sync ────────────────────────────────────────────────────────────

do_deploy() {
    stage_start "deploy code to remote"
    $SSH_CMD "mkdir -p ${REMOTE_DIR}/src"
    rsync -avz --exclude='__pycache__' \
        -e "ssh -i ${SSH_KEY}" \
        "${SCRIPT_DIR}/src/" "${REMOTE_HOST}:${REMOTE_DIR}/src/"
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
            rsync_from_remote data/translation/non_english_attachments_translated_batches/ \
                data/translation/non_english_attachments_translated_batches/
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
        change-summaries)
            stage_start "pull change summaries"
            rsync_from_remote data/analysis/change_summaries/ data/analysis/change_summaries/
            stage_end "pull change summaries"
            ;;
        all)
            do_pull ocr
            do_pull translation
            do_pull summaries
            do_pull classification
            do_pull cluster-summaries
            do_pull change-summaries
            ;;
        *)
            echo "ERROR: Unknown pull target: $target"
            echo "Valid targets: ocr, translation, summaries, classification, cluster-summaries, change-summaries, all"
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
        summarize-changes)
            run_remote "summarize-changes" \
                $PYTHON src/summarize_changes.py \
                    data/analysis/unit_summaries/ \
                    -o data/analysis/change_summaries/ "$@"
            ;;
        *)
            echo "ERROR: Unknown remote step: $step"
            echo "Valid steps: ocr, translate, summarize, classify, summarize-clusters, summarize-changes"
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
    do_build_index "$@"

    # Push for remote cluster summarization
    do_push unit-summaries
    do_push clustering

    # Remote cluster summarization -> pull
    do_remote summarize-clusters "$@"
    do_pull cluster-summaries

    # Remote change summarization -> pull
    do_remote summarize-changes "$@"
    do_pull change-summaries

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] FULL PIPELINE COMPLETE"
    echo "============================================================"
}

# ── Log tailing ──────────────────────────────────────────────────────────────

do_logs() {
    local target="${1:-list}"
    shift 2>/dev/null || true
    case "$target" in
        list)
            echo "Remote logs:"
            # shellcheck disable=SC2029
            $SSH_CMD "ls -lt ${REMOTE_DIR}/logs/*.log 2>/dev/null | head -20" || echo "  (no logs)"
            ;;
        tail)
            local step="${1:-}"
            if [ -z "$step" ]; then
                # Tail the most recent log
                echo "Tailing most recent remote log..."
                # shellcheck disable=SC2029
                $SSH_CMD "tail -n 20 -f \$(ls -t ${REMOTE_DIR}/logs/*.log 2>/dev/null | head -1)"
            else
                # Tail the most recent log matching the step name
                local pattern
                pattern="$(echo "$step" | tr ' ()' '_')"
                echo "Tailing most recent '$step' log..."
                # shellcheck disable=SC2029
                $SSH_CMD "tail -n 20 -f \$(ls -t ${REMOTE_DIR}/logs/${pattern}*.log 2>/dev/null | head -1)"
            fi
            ;;
        ocr|translate|summarize|classify|summarize-clusters|summarize-changes)
            local pattern
            pattern="$(echo "$target" | tr ' ()' '_')"
            echo "Tailing most recent '$target' log..."
            # shellcheck disable=SC2029
            $SSH_CMD "tail -n 20 -f \$(ls -t ${REMOTE_DIR}/logs/${pattern}*.log 2>/dev/null | head -1)"
            ;;
        *)
            echo "Usage: pipeline.sh logs [list|tail [step]|ocr|translate|summarize|classify|summarize-clusters|summarize-changes]"
            exit 1
            ;;
    esac
}

# ── Stage listing ────────────────────────────────────────────────────────────

do_list() {
    cat <<'EOF'
Full pipeline (./pipeline.sh full):

   #  Command                   Description
   1  scrape                    Scrape initiatives + details
   2  find-short-pdfs           Find short PDF extractions for OCR
   3  deploy                    Rsync code to remote
   4  push ocr                  Push OCR input to remote
   5  remote ocr                Run GPU OCR on remote
   6  pull ocr                  Pull OCR results back
   7  merge-ocr                 Merge OCR results into initiative_details
   8  find-nonenglish           Find non-English feedback attachments
   9  push translation          Push translation input to remote
  10  remote translate          Run GPU translation on remote
  11  pull translation          Pull translation results back
  12  merge-translations        Merge translations into initiative_details
  13  analyze                   Run initiative_stats (before/after analysis)
  14  push analysis             Push before-after analysis to remote
  15  remote summarize          Run GPU document summarization on remote
  16  pull summaries            Pull document summaries back
  17  build-summaries           Build unit summaries from document summaries
  18  cluster                   Cluster all initiatives (per configured schemes)
  19  build-index               Pre-compute webapp initiative index
  20  push unit-summaries       Push unit summaries to remote
  21  push clustering           Push clustering data to remote
  22  remote summarize-clusters Run GPU cluster summarization on remote
  23  pull cluster-summaries    Pull cluster summaries back
  24  remote summarize-changes  Run GPU change summarization on remote
  25  pull change-summaries     Pull change summaries back

Other commands:
  remote classify          Run GPU classification on remote
  pull classification      Pull classification results back
  clean-batches <target>   Delete batch files on remote (summaries, cluster-summaries,
                           change-summaries, translation, all)
  logs                     List recent remote logs
  logs tail [step]         Tail most recent log (optionally filtered by step name)

Extra args are passed through to the underlying Python scripts.
EOF
}

# ── Clean batch files ─────────────────────────────────────────────────────────

do_clean_batches() {
    local target="${1:?Usage: pipeline.sh clean-batches <target>}"
    shift
    case "$target" in
        summaries)
            stage_start "clean summary batch files (remote)"
            $SSH_CMD "rm -rf ${REMOTE_DIR}/data/analysis/summaries/_batches_pass1 \
                             ${REMOTE_DIR}/data/analysis/summaries/_batches_pass2"
            stage_end "clean summary batch files (remote)"
            ;;
        cluster-summaries)
            if [ -z "$CLUSTER_SCHEMES" ]; then
                echo "ERROR: CLUSTER_SCHEMES not set in pipeline.conf"
                exit 1
            fi
            for scheme in $CLUSTER_SCHEMES; do
                stage_start "clean cluster summary batch files ($scheme, remote)"
                $SSH_CMD "rm -rf ${REMOTE_DIR}/data/cluster_summaries/${scheme}/_batches*"
                stage_end "clean cluster summary batch files ($scheme, remote)"
            done
            ;;
        change-summaries)
            stage_start "clean change summary batch files (remote)"
            $SSH_CMD "rm -rf ${REMOTE_DIR}/data/analysis/change_summaries/_batches"
            stage_end "clean change summary batch files (remote)"
            ;;
        translation)
            stage_start "clean translation batch files (remote)"
            $SSH_CMD "rm -rf ${REMOTE_DIR}/data/translation/non_english_attachments_translated_batches"
            stage_end "clean translation batch files (remote)"
            ;;
        all)
            do_clean_batches summaries
            do_clean_batches cluster-summaries
            do_clean_batches change-summaries
            do_clean_batches translation
            ;;
        *)
            echo "ERROR: Unknown clean-batches target: $target"
            echo "Valid targets: summaries, cluster-summaries, change-summaries, translation, all"
            exit 1
            ;;
    esac
}

# ── Dispatch ─────────────────────────────────────────────────────────────────

STAGE="${1:?Usage: pipeline.sh <stage> [extra-args...]}"
shift

case "$STAGE" in
    list)               do_list ;;
    scrape)             do_scrape "$@" ;;
    find-short-pdfs)    do_find_short_pdfs "$@" ;;
    find-nonenglish)    do_find_nonenglish "$@" ;;
    merge-ocr)          do_merge_ocr "$@" ;;
    merge-translations) do_merge_translations "$@" ;;
    analyze)            do_analyze "$@" ;;
    build-summaries)    do_build_summaries "$@" ;;
    cluster)            do_cluster "$@" ;;
    build-index)        do_build_index "$@" ;;
    deploy)             do_deploy ;;
    push)               do_push "$@" ;;
    pull)               do_pull "$@" ;;
    remote)             do_remote "$@" ;;
    logs)               do_logs "$@" ;;
    clean-batches)      do_clean_batches "$@" ;;
    full)               do_full "$@" ;;
    *)
        echo "ERROR: Unknown stage: $STAGE"
        echo "Run './pipeline.sh list' to see available stages."
        exit 1
        ;;
esac
