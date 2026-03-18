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

# Count JSON files in a directory (excluding _* batch dirs/files).
count_json() {
    local dir="$1"
    local label="${2:-input files}"
    if [ -d "$dir" ]; then
        local n
        n=$(find "$dir" -maxdepth 1 -name '*.json' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')
        echo "  Tasks: $n $label"
    fi
}

# Count records in a JSON array file.
count_json_records() {
    local file="$1"
    local label="${2:-records}"
    if [ -f "$file" ]; then
        local n
        n=$($PYTHON -c "import json; print(len(json.load(open('$file'))))" 2>/dev/null || echo "?")
        echo "  Tasks: $n $label"
    fi
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
        && nohup bash -c '{ ${remote_cmd}; } > ${log_file} 2>&1; echo \$? > ${status_file}' \
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

    # Clean up batch files after successful run
    if [ -n "${REMOTE_BATCH_DIRS:-}" ]; then
        echo "Cleaning up batch files..."
        # shellcheck disable=SC2029
        $SSH_CMD "cd ${REMOTE_DIR} && for pat in ${REMOTE_BATCH_DIRS}; do rm -rf \$pat 2>/dev/null && echo \"  Removed \$pat\" || true; done"
        REMOTE_BATCH_DIRS=""
    fi

    stage_end "remote $name"
}

RSYNC_OPTS=(-avz -e "ssh -i ${SSH_KEY}")
PARALLEL_JOBS=4

# Split a file list into N chunks and run parallel rsync with --files-from.
# Usage: parallel_rsync <file_list_file> <src_base> <dst_spec> [extra_rsync_args...]
parallel_rsync() {
    local file_list="$1" src_base="$2" dst_spec="$3"
    shift 3
    local n_files
    n_files=$(wc -l < "$file_list" | tr -d ' ')
    if [ "$n_files" -eq 0 ]; then
        echo "Nothing to transfer"
        return
    fi
    local chunk_size=$(( (n_files + PARALLEL_JOBS - 1) / PARALLEL_JOBS ))
    echo "Transferring $n_files files ($PARALLEL_JOBS parallel streams)"
    local tmpdir
    tmpdir=$(mktemp -d)
    split -l "$chunk_size" "$file_list" "${tmpdir}/chunk_"
    local pids=()
    for chunk in "${tmpdir}"/chunk_*; do
        rsync "${RSYNC_OPTS[@]}" "$@" --files-from="$chunk" \
            "$src_base" "$dst_spec" &
        pids+=($!)
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=$((failed + 1))
    done
    rm -rf "$tmpdir"
    if [ "$failed" -gt 0 ]; then
        echo "WARNING: $failed rsync streams had errors"
        return 1
    fi
}

# Push a local directory or file to remote.
rsync_to_remote() {
    local local_path="$1" remote_path="$2"
    if [[ -d "$local_path" || "$local_path" == */ ]]; then
        local src_dir="${local_path%/}"
        $SSH_CMD "mkdir -p ${REMOTE_DIR}/${remote_path}"
        local tmpfile
        tmpfile=$(mktemp)
        (cd "$src_dir" && find . -type f) | sed 's|^\./||' | sort > "$tmpfile"
        parallel_rsync "$tmpfile" "$src_dir/" "${REMOTE_HOST}:${REMOTE_DIR}/${remote_path%/}/"
        rm -f "$tmpfile"
    else
        $SSH_CMD "mkdir -p $(dirname "${REMOTE_DIR}/${remote_path}")"
        rsync "${RSYNC_OPTS[@]}" \
            "$local_path" "${REMOTE_HOST}:${REMOTE_DIR}/${remote_path}"
    fi
}

# Pull a remote directory or file, skipping files that already exist locally.
rsync_from_remote() {
    local remote_path="$1" local_path="$2"
    mkdir -p "$(dirname "$local_path")"
    if [[ "$remote_path" == */ ]]; then
        mkdir -p "${local_path%/}"
        local remote_dir="${remote_path%/}"
        local local_dir="${local_path%/}"
        local tmpfile
        tmpfile=$(mktemp)
        rsync "${RSYNC_OPTS[@]}" --ignore-existing --list-only \
            "${REMOTE_HOST}:${REMOTE_DIR}/${remote_dir}/" "${local_dir}/" 2>/dev/null | \
            awk '/^-/ {print $NF}' | sort > "$tmpfile"
        parallel_rsync "$tmpfile" "${REMOTE_HOST}:${REMOTE_DIR}/${remote_dir}/" "${local_dir}/" --ignore-existing
        rm -f "$tmpfile"
    else
        # Single files: always overwrite (they may be regenerated on re-runs)
        rsync "${RSYNC_OPTS[@]}" \
            "${REMOTE_HOST}:${REMOTE_DIR}/${remote_path}" "$local_path"
    fi
}

# Pull from remote with exclude patterns, using parallel rsync.
rsync_from_remote_exclude() {
    local remote_path="$1" local_path="$2"
    shift 2
    local excludes=()
    for pat in "$@"; do
        excludes+=(--exclude "$pat")
    done
    mkdir -p "$(dirname "$local_path")"
    if [[ "$remote_path" == */ ]]; then
        mkdir -p "${local_path%/}"
        local remote_dir="${remote_path%/}"
        local local_dir="${local_path%/}"
        local tmpfile
        tmpfile=$(mktemp)
        rsync "${RSYNC_OPTS[@]}" --ignore-existing "${excludes[@]}" --list-only \
            "${REMOTE_HOST}:${REMOTE_DIR}/${remote_dir}/" "${local_dir}/" 2>/dev/null | \
            awk '/^-/ {print $NF}' | sort > "$tmpfile"
        parallel_rsync "$tmpfile" "${REMOTE_HOST}:${REMOTE_DIR}/${remote_dir}/" "${local_dir}/" --ignore-existing "${excludes[@]}"
        rm -f "$tmpfile"
    else
        rsync "${RSYNC_OPTS[@]}" --ignore-existing "${excludes[@]}" \
            "${REMOTE_HOST}:${REMOTE_DIR}/${remote_path}" "$local_path"
    fi
}

# Parse a scheme name like "agglomerative_google_embeddinggemma-300m_k1=v1_k2=v2"
# into algorithm, model, and key=value parameters.
# Multi-word keys like "max_cluster_size=20" are reassembled from underscore-split
# parts by accumulating until a part contains "=".
parse_scheme() {
    local scheme="$1"
    local IFS="_"
    # shellcheck disable=SC2206
    local parts=($scheme)

    SCHEME_ALGO="${parts[0]}"
    # Model name contains a slash: "google/embeddinggemma-300m"
    SCHEME_MODEL="${parts[1]}/${parts[2]}"

    # Reassemble key=value pairs: accumulate parts until one contains "="
    local params=()
    local accum=""
    for (( i=3; i<${#parts[@]}; i++ )); do
        local part="${parts[$i]}"
        if [[ "$part" == *=* ]]; then
            if [ -n "$accum" ]; then
                accum="${accum}_${part}"
            else
                accum="$part"
            fi
            params+=("$accum")
            accum=""
        else
            if [ -n "$accum" ]; then
                accum="${accum}_${part}"
            else
                accum="$part"
            fi
        fi
    done

    SCHEME_FLAGS=()
    for param in "${params[@]}"; do
        local key="${param%%=*}"
        local val="${param#*=}"
        # Convert underscore-style to CLI flag: distance_threshold -> --distance-threshold
        local flag="--${key//_/-}"
        SCHEME_FLAGS+=("$flag" "$val")
    done
}

# Build the output subdirectory name for a scheme (same as the scheme name).
scheme_output_dir() {
    echo "data/clustering/$1"
}

# ── Setup ────────────────────────────────────────────────────────────────────

do_setup() {
    stage_start "local setup"
    echo "Installing Python dependencies (local)..."
    uv sync
    if huggingface-cli whoami &>/dev/null; then
        echo "Hugging Face token already configured, skipping login."
    else
        echo ""
        echo "Logging in to Hugging Face (for model downloads)..."
        echo "Get your token at https://huggingface.co/settings/tokens"
        huggingface-cli login
    fi
    stage_end "local setup"
}

do_setup_remote() {
    stage_start "remote setup"

    # Deploy source code
    do_deploy

    # Install Python dependencies on remote
    echo "Installing Python dependencies (remote)..."
    # shellcheck disable=SC2029
    $SSH_CMD "cd ${REMOTE_DIR} && $PYTHON -m pip install \
        --extra-index-url https://pypi.nvidia.com \
        vllm openai-harmony \
        easyocr \
        sentence-transformers scikit-learn hdbscan \
        cuml-cu12 \
        torch numpy \
        huggingface-hub"

    # Interactive HF login on remote (needs a TTY)
    # shellcheck disable=SC2029
    if $SSH_CMD "huggingface-cli whoami" &>/dev/null; then
        echo "Hugging Face token already configured on remote, skipping login."
    else
        echo ""
        echo "Logging in to Hugging Face on remote..."
        echo "Get your token at https://huggingface.co/settings/tokens"
        ssh -t -i "${SSH_KEY}" "${REMOTE_HOST}" \
            "cd ${REMOTE_DIR} && huggingface-cli login"
    fi

    stage_end "remote setup"
}

# ── Stages ───────────────────────────────────────────────────────────────────

do_scrape() {
    run_local "scrape initiatives" \
        $PYTHON src/scrape_eu_initiatives.py "$@"
    count_json data/scrape/initiative_details "cached initiative details"
    run_local "scrape initiative details" \
        $PYTHON src/scrape_eu_initiative_details.py "$@"
}

do_find_short_pdfs() {
    count_json data/scrape/initiative_details "initiative files to scan"
    run_local "find short PDF extractions" \
        $PYTHON src/find_short_pdf_extractions.py \
            data/scrape/initiative_details \
            -o data/ocr/ "$@"
}

do_find_nonenglish() {
    count_json data/scrape/initiative_details "initiative files to scan"
    run_local "find non-English attachments" \
        $PYTHON src/find_non_english_feedback_attachments.py \
            data/scrape/initiative_details \
            -o data/translation/non_english_attachments.json "$@"
}

do_merge_ocr() {
    count_json_records data/ocr/short_pdf_report_ocr.json "OCR records"
    run_local "merge OCR results" \
        $PYTHON src/merge_ocr_results.py \
            data/ocr/short_pdf_report_ocr.json \
            data/scrape/initiative_details "$@"
}

do_merge_translations() {
    count_json_records data/translation/non_english_attachments_translated.json "translation records"
    run_local "merge translations" \
        $PYTHON src/merge_translations.py \
            data/translation/non_english_attachments_translated.json \
            data/scrape/initiative_details "$@"
}

do_analyze() {
    count_json data/scrape/initiative_details "initiative files to analyze"
    run_local "initiative stats / before-after analysis" \
        $PYTHON src/initiative_stats.py \
            data/scrape/initiative_details \
            -o data/analysis/before_after/ "$@"
}

do_build_summaries() {
    count_json data/analysis/summaries "summary files to process"
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
    count_json data/analysis/unit_summaries "unit summary files to cluster"
    for scheme in $CLUSTER_SCHEMES; do
        parse_scheme "$scheme"
        local out_dir
        out_dir="$(scheme_output_dir "$scheme")"
        run_local "cluster ($SCHEME_ALGO)" \
            $PYTHON src/cluster_all_initiatives.py \
                --algorithm "$SCHEME_ALGO" \
                --model "$SCHEME_MODEL" \
                --output-dir "$out_dir" \
                --embeddings-cache-dir data/embeddings \
                "${SCHEME_FLAGS[@]}" \
                "$@"
    done
}

do_build_index() {
    count_json data/scrape/initiative_details "initiative files to index"
    run_local "build webapp index" \
        $PYTHON src/build_webapp_index.py \
            data/scrape/initiative_details \
            -o data/webapp/initiative_index.json "$@"
}

do_merge_summaries() {
    count_json data/analysis/summaries "document summary files"
    run_local "merge document/attachment summaries" \
        $PYTHON src/merge_summaries.py \
            data/analysis/summaries \
            data/scrape/initiative_details "$@"
}

do_merge_change_summaries() {
    count_json data/analysis/change_summaries "change summary files"
    run_local "merge change summaries" \
        $PYTHON src/merge_change_summaries.py \
            data/analysis/change_summaries \
            data/scrape/initiative_details "$@"
}

do_merge_cluster_feedback_summaries() {
    if [ -z "$CLUSTER_SCHEMES" ]; then
        echo "ERROR: CLUSTER_SCHEMES not set in pipeline.conf"
        exit 1
    fi
    for scheme in $CLUSTER_SCHEMES; do
        count_json "data/cluster_summaries/${scheme}" "cluster summary files ($scheme)"
        run_local "merge cluster feedback summaries ($scheme)" \
            $PYTHON src/merge_cluster_feedback_summaries.py \
                "data/cluster_summaries/${scheme}" \
                data/scrape/initiative_details "$@"
    done
}

# ── Deploy / sync ────────────────────────────────────────────────────────────

do_deploy() {
    stage_start "deploy code to remote"
    $SSH_CMD "mkdir -p ${REMOTE_DIR}/src"
    rsync "${RSYNC_OPTS[@]}" --exclude='__pycache__' \
        "${SCRIPT_DIR}/src/" "${REMOTE_HOST}:${REMOTE_DIR}/src/"
    stage_end "deploy code to remote"
}

do_push() {
    local target="${1:?Usage: pipeline.sh push <target>}"
    shift
    case "$target" in
        initiative-details)
            stage_start "push initiative details"
            count_json data/scrape/initiative_details "initiative files"
            rsync_to_remote data/scrape/initiative_details/ data/scrape/initiative_details/
            stage_end "push initiative details"
            ;;
        ocr)
            stage_start "push ocr data"
            count_json data/ocr "OCR files"
            rsync_to_remote data/ocr/ data/ocr/
            stage_end "push ocr data"
            ;;
        translation)
            stage_start "push translation input"
            count_json_records data/translation/non_english_attachments.json "attachment records"
            rsync_to_remote data/translation/non_english_attachments.json \
                data/translation/non_english_attachments.json
            stage_end "push translation input"
            ;;
        analysis)
            stage_start "push before-after analysis"
            count_json data/analysis/before_after "before-after files"
            rsync_to_remote data/analysis/before_after/ data/analysis/before_after/
            stage_end "push before-after analysis"
            ;;
        unit-summaries)
            stage_start "push unit summaries"
            count_json data/analysis/unit_summaries "unit summary files"
            rsync_to_remote data/analysis/unit_summaries/ data/analysis/unit_summaries/
            stage_end "push unit summaries"
            ;;
        clustering)
            stage_start "push clustering data"
            count_json data/clustering "clustering scheme dirs"
            rsync_to_remote data/clustering/ data/clustering/
            stage_end "push clustering data"
            ;;
        all)
            do_push initiative-details
            do_push ocr
            do_push translation
            do_push analysis
            do_push unit-summaries
            do_push clustering
            ;;
        *)
            echo "ERROR: Unknown push target: $target"
            echo "Valid targets: initiative-details, ocr, translation, analysis, unit-summaries, clustering, all"
            exit 1
            ;;
    esac
}

do_pull() {
    local target="${1:?Usage: pipeline.sh pull <target>}"
    shift
    case "$target" in
        initiative-details)
            stage_start "pull initiative details"
            # Overwrite local: remote has authoritative copy with all merges applied
            mkdir -p data/scrape/initiative_details
            rsync "${RSYNC_OPTS[@]}" \
                "${REMOTE_HOST}:${REMOTE_DIR}/data/scrape/initiative_details/" \
                data/scrape/initiative_details/
            stage_end "pull initiative details"
            ;;
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
            rsync_from_remote_exclude data/analysis/summaries/ data/analysis/summaries/ '_batches*'
            stage_end "pull document summaries"
            ;;
        classification)
            stage_start "pull classification results"
            rsync_from_remote data/classification/ data/classification/
            stage_end "pull classification results"
            ;;
        clustering)
            stage_start "pull clustering results"
            rsync_from_remote data/clustering/ data/clustering/
            stage_end "pull clustering results"
            ;;
        embeddings)
            stage_start "pull embeddings cache"
            # No --ignore-existing: embeddings are overwritten when initiative data changes
            mkdir -p data/embeddings
            rsync "${RSYNC_OPTS[@]}" \
                "${REMOTE_HOST}:${REMOTE_DIR}/data/embeddings/" data/embeddings/
            stage_end "pull embeddings cache"
            ;;
        cluster-summaries)
            stage_start "pull cluster summaries"
            rsync_from_remote_exclude data/cluster_summaries/ data/cluster_summaries/ '_batches*'
            stage_end "pull cluster summaries"
            ;;
        change-summaries)
            stage_start "pull change summaries"
            rsync_from_remote_exclude data/analysis/change_summaries/ data/analysis/change_summaries/ '_batches*'
            stage_end "pull change summaries"
            ;;
        webapp)
            stage_start "pull webapp data"
            mkdir -p data/webapp
            rsync "${RSYNC_OPTS[@]}" \
                "${REMOTE_HOST}:${REMOTE_DIR}/data/webapp/" data/webapp/
            stage_end "pull webapp data"
            ;;
        logs)
            stage_start "pull remote logs"
            mkdir -p data/logs
            rsync "${RSYNC_OPTS[@]}" \
                "${REMOTE_HOST}:${REMOTE_DIR}/logs/" data/logs/
            stage_end "pull remote logs"
            ;;
        all)
            do_pull initiative-details
            do_pull ocr
            do_pull translation
            do_pull summaries
            do_pull classification
            do_pull clustering
            do_pull embeddings
            do_pull cluster-summaries
            do_pull change-summaries
            do_pull webapp
            ;;
        *)
            echo "ERROR: Unknown pull target: $target"
            echo "Valid targets: initiative-details, ocr, translation, summaries, classification, clustering, embeddings, cluster-summaries, change-summaries, webapp, logs, all"
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
            REMOTE_BATCH_DIRS="data/translation/non_english_attachments_translated_batches"
            run_remote "translate" \
                $PYTHON src/translate_attachments.py \
                    data/translation/non_english_attachments.json \
                    -o data/translation/non_english_attachments_translated.json "$@"
            ;;
        summarize)
            REMOTE_BATCH_DIRS="data/analysis/summaries/_batches_pass1 data/analysis/summaries/_batches_pass2"
            local summarize_prev=""
            if [ -d "data/analysis/summaries" ]; then
                summarize_prev="--prev-output data/analysis/summaries"
            fi
            run_remote "summarize" \
                $PYTHON src/summarize_documents.py \
                    data/analysis/before_after/ \
                    -o data/analysis/summaries/ $summarize_prev "$@"
            ;;
        classify)
            run_remote "classify" \
                $PYTHON src/classify_initiative_and_feedback.py \
                    data/analysis/unit_summaries/ \
                    -o data/classification/ "$@"
            ;;
        cluster)
            if [ -z "$CLUSTER_SCHEMES" ]; then
                echo "ERROR: CLUSTER_SCHEMES not set in pipeline.conf"
                exit 1
            fi
            # cuML needs RAPIDS native libraries on LD_LIBRARY_PATH; unbuffered for live log output
            local rapids_ld='export PYTHONUNBUFFERED=1; export LD_LIBRARY_PATH=$(python3 -c "import site,os,glob;ps=[os.path.expanduser(\"~/.local/lib/python3.10/site-packages\")];print(\":\".join(d for p in ps for d in glob.glob(os.path.join(p,\"lib*/lib64\"))))" 2>/dev/null):$LD_LIBRARY_PATH;'
            local pids=()
            for scheme in $CLUSTER_SCHEMES; do
                parse_scheme "$scheme"
                local out_dir
                out_dir="$(scheme_output_dir "$scheme")"
                run_remote "cluster ($SCHEME_ALGO)" \
                    "$rapids_ld" $PYTHON src/cluster_all_initiatives.py \
                        --algorithm "$SCHEME_ALGO" \
                        --model "$SCHEME_MODEL" \
                        --output-dir "$out_dir" \
                        --embeddings-cache-dir data/embeddings \
                        "${SCHEME_FLAGS[@]}" \
                        "$@" &
                pids+=($!)
            done
            local failed=0
            for pid in "${pids[@]}"; do
                wait "$pid" || failed=$((failed + 1))
            done
            if [ "$failed" -gt 0 ]; then
                echo "ERROR: $failed clustering scheme(s) failed"
                return 1
            fi
            ;;
        summarize-clusters)
            if [ -z "$CLUSTER_SCHEMES" ]; then
                echo "ERROR: CLUSTER_SCHEMES not set in pipeline.conf"
                exit 1
            fi
            for scheme in $CLUSTER_SCHEMES; do
                local cluster_dir="data/clustering/${scheme}"
                local summary_dir="data/cluster_summaries/${scheme}"
                REMOTE_BATCH_DIRS="data/cluster_summaries/${scheme}/_batches_p1_* data/cluster_summaries/${scheme}/_batches_p2_* data/cluster_summaries/${scheme}/_batches_p3_*"
                local prev_arg=""
                if [ -d "$summary_dir" ]; then
                    prev_arg="--prev-output $summary_dir"
                fi
                run_remote "summarize-clusters ($scheme)" \
                    $PYTHON src/summarize_clusters.py \
                        "$cluster_dir" \
                        -o "$summary_dir" $prev_arg "$@"
            done
            ;;
        summarize-changes)
            REMOTE_BATCH_DIRS="data/analysis/change_summaries/_batches data/analysis/change_summaries/_batches_combine"
            run_remote "summarize-changes" \
                $PYTHON src/summarize_changes.py \
                    data/analysis/unit_summaries/ \
                    -o data/analysis/change_summaries/ "$@"
            ;;
        # ── Composite pipeline steps (chain find/merge with GPU steps on remote) ──
        ocr-pipeline)
            run_remote "ocr-pipeline" \
                "$PYTHON src/find_short_pdf_extractions.py data/scrape/initiative_details -o data/ocr/" \
                "&& $PYTHON src/ocr_short_pdfs.py data/ocr/" \
                "&& $PYTHON src/merge_ocr_results.py data/ocr/short_pdf_report_ocr.json data/scrape/initiative_details"
            ;;
        translate-pipeline)
            REMOTE_BATCH_DIRS="data/translation/non_english_attachments_translated_batches"
            run_remote "translate-pipeline" \
                "$PYTHON src/find_non_english_feedback_attachments.py data/scrape/initiative_details -o data/translation/non_english_attachments.json" \
                "&& $PYTHON src/translate_attachments.py data/translation/non_english_attachments.json -o data/translation/non_english_attachments_translated.json" \
                "&& $PYTHON src/merge_translations.py data/translation/non_english_attachments_translated.json data/scrape/initiative_details"
            ;;
        summarize-pipeline)
            REMOTE_BATCH_DIRS="data/analysis/summaries/_batches_pass1 data/analysis/summaries/_batches_pass2"
            run_remote "summarize-pipeline" \
                "$PYTHON src/initiative_stats.py data/scrape/initiative_details -o data/analysis/before_after/" \
                "&& $PYTHON src/summarize_documents.py data/analysis/before_after/ -o data/analysis/summaries/ --prev-output data/analysis/summaries" \
                "&& $PYTHON src/build_unit_summaries.py data/analysis/summaries/ -o data/analysis/unit_summaries/"
            ;;
        cluster-summarize-pipeline)
            if [ -z "$CLUSTER_SCHEMES" ]; then
                echo "ERROR: CLUSTER_SCHEMES not set in pipeline.conf"
                exit 1
            fi
            for scheme in $CLUSTER_SCHEMES; do
                local cluster_dir="data/clustering/${scheme}"
                local summary_dir="data/cluster_summaries/${scheme}"
                REMOTE_BATCH_DIRS="data/cluster_summaries/${scheme}/_batches_p1_* data/cluster_summaries/${scheme}/_batches_p2_* data/cluster_summaries/${scheme}/_batches_p3_*"
                run_remote "cluster-summarize ($scheme)" \
                    "$PYTHON src/summarize_clusters.py $cluster_dir -o $summary_dir" \
                    "&& $PYTHON src/merge_cluster_feedback_summaries.py $summary_dir data/scrape/initiative_details"
            done
            ;;
        change-summarize-pipeline)
            REMOTE_BATCH_DIRS="data/analysis/change_summaries/_batches data/analysis/change_summaries/_batches_combine"
            run_remote "change-summarize-pipeline" \
                "$PYTHON src/summarize_changes.py data/analysis/unit_summaries/ -o data/analysis/change_summaries/" \
                "&& $PYTHON src/merge_change_summaries.py data/analysis/change_summaries data/scrape/initiative_details"
            ;;
        build-index)
            run_remote "build-index" \
                $PYTHON src/build_webapp_index.py \
                    data/scrape/initiative_details \
                    -o data/webapp/initiative_index.json "$@"
            ;;
        *)
            echo "ERROR: Unknown remote step: $step"
            echo "Valid steps: ocr, translate, summarize, classify, cluster, summarize-clusters, summarize-changes,"
            echo "             ocr-pipeline, translate-pipeline, summarize-pipeline,"
            echo "             cluster-summarize-pipeline, change-summarize-pipeline, build-index"
            exit 1
            ;;
    esac
}

# ── Full pipeline ────────────────────────────────────────────────────────────

do_full() {
    # Phase 1: Local scraping
    do_scrape "$@"

    # Phase 2: Deploy code + push initiative data to remote
    do_deploy
    do_push initiative-details

    # Phase 3: OCR pipeline (find + GPU OCR + merge, all on remote)
    do_remote ocr-pipeline

    # Phase 4: Translation pipeline (find + GPU translate + merge, all on remote)
    do_remote translate-pipeline

    # Phase 5: Summarization pipeline (analyze + GPU summarize + build summaries, all on remote)
    do_remote summarize-pipeline

    # Phase 6: Clustering (on remote, uses unit_summaries produced in phase 5)
    do_remote cluster

    # Phase 7: Cluster summarization + merge (on remote, per scheme)
    do_remote cluster-summarize-pipeline

    # Phase 8: Change summarization + merge (on remote)
    do_remote change-summarize-pipeline

    # Phase 9: Build webapp index (on remote, after all merges)
    do_remote build-index

    # Phase 10: Pull all results back
    do_pull initiative-details
    do_pull clustering
    do_pull embeddings
    do_pull webapp
    do_pull summaries
    do_pull cluster-summaries
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
        ocr|translate|summarize|classify|cluster|summarize-clusters|summarize-changes)
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

   #  Command                              Description
   1  scrape                               Scrape initiatives + details (local)
   2  deploy                               Rsync code to remote
   3  push initiative-details              Push initiative data to remote
   4  remote ocr-pipeline                  Find short PDFs + GPU OCR + merge (remote)
   5  remote translate-pipeline            Find non-English + GPU translate + merge (remote)
   6  remote summarize-pipeline            Analyze + GPU summarize + build summaries (remote)
   7  remote cluster                       GPU clustering (remote, multi-GPU)
   8  remote cluster-summarize-pipeline    GPU cluster summarize + merge (remote, per scheme)
   9  remote change-summarize-pipeline     GPU change summarize + merge (remote)
  10  remote build-index                   Build webapp index (remote)
  11  pull initiative-details              Pull initiative data with all merges applied
  12  pull clustering                      Pull clustering results
  13  pull embeddings                      Pull embeddings cache
  14  pull webapp                          Pull webapp data
  15  pull summaries                       Pull document summaries
  16  pull cluster-summaries               Pull cluster summaries
  17  pull change-summaries                Pull change summaries

Setup (run once before first pipeline run):
  setup                    Install local Python deps (uv sync) + Hugging Face login
  setup-remote             Deploy code + install remote Python deps (pip) + Hugging Face login

Individual stages (for ad-hoc use):
  scrape                   Scrape initiatives + details (local)
  find-short-pdfs          Find short PDF extractions (local)
  find-nonenglish          Find non-English feedback attachments (local)
  merge-ocr                Merge OCR results into initiative_details (local)
  merge-translations       Merge translations into initiative_details (local)
  analyze                  Run initiative_stats (local)
  build-summaries          Build unit summaries (local)
  build-index              Build webapp index (local)
  cluster                  Cluster all initiatives locally
  merge-summaries          Merge doc/attachment summaries (local)
  merge-change-summaries   Merge change summaries (local)
  merge-cluster-feedback-summaries  Merge cluster feedback summaries (local)
  remote ocr               Run GPU OCR on remote
  remote translate         Run GPU translation on remote
  remote summarize         Run GPU document summarization on remote
  remote classify          Run GPU classification on remote
  remote cluster           Run GPU clustering on remote
  remote summarize-clusters  Run GPU cluster summarization on remote
  remote summarize-changes   Run GPU change summarization on remote

Recovery:
  recover                  Pull all outputs from remote + merge locally + rebuild index
                           Use after a crash to salvage intermediate results

Other commands:
  deploy                   Rsync code to remote
  push <target>            Push data to remote
  pull <target>            Pull data from remote
  clean-batches <target>   Delete batch files on remote
  logs                     List recent remote logs
  logs tail [step]         Tail most recent log

Push targets: initiative-details, ocr, translation, analysis, unit-summaries, clustering, all
Pull targets: initiative-details, ocr, translation, summaries, classification, clustering,
              embeddings, cluster-summaries, change-summaries, webapp, logs, all

Push/pull use parallel rsync (4 streams). Extra args are passed through to Python scripts.
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

# ── Recover ──────────────────────────────────────────────────────────────────

# Pull all available output files from remote and merge them into
# initiative_details locally.  Useful when a pipeline run crashes partway
# through — intermediate results that were already written to disk on the
# remote can still be recovered and merged.
#
# Each pull and merge step is run with || true so that missing remote
# directories (from stages that never ran) don't abort the recovery.

do_recover() {
    echo ""
    echo "============================================================"
    echo "[$(timestamp)] RECOVER: pulling outputs from remote"
    echo "============================================================"

    # Pull initiative_details (authoritative copy with any merges applied on remote)
    echo ""
    echo "--- Pulling initiative_details ---"
    do_pull initiative-details || echo "  (skipped — not available on remote)"

    # Pull each pipeline output category
    echo ""
    echo "--- Pulling OCR results ---"
    do_pull ocr || echo "  (skipped — not available on remote)"

    echo ""
    echo "--- Pulling translation results ---"
    do_pull translation || echo "  (skipped — not available on remote)"

    echo ""
    echo "--- Pulling document summaries ---"
    do_pull summaries || echo "  (skipped — not available on remote)"

    echo ""
    echo "--- Pulling change summaries ---"
    do_pull change-summaries || echo "  (skipped — not available on remote)"

    echo ""
    echo "--- Pulling cluster summaries ---"
    do_pull cluster-summaries || echo "  (skipped — not available on remote)"

    echo ""
    echo "--- Pulling clustering results ---"
    do_pull clustering || echo "  (skipped — not available on remote)"

    echo ""
    echo "--- Pulling embeddings ---"
    do_pull embeddings || echo "  (skipped — not available on remote)"

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] RECOVER: merging outputs into initiative_details"
    echo "============================================================"

    # Run all merge scripts locally.  Each tolerates missing input gracefully.
    echo ""
    echo "--- Merging OCR results ---"
    if [ -f data/ocr/short_pdf_report_ocr.json ]; then
        do_merge_ocr
    else
        echo "  (skipped — no OCR report)"
    fi

    echo ""
    echo "--- Merging translations ---"
    if [ -f data/translation/non_english_attachments_translated.json ]; then
        do_merge_translations
    elif [ -d data/translation/non_english_attachments_translated_batches ]; then
        run_local "merge translations (from batches)" \
            $PYTHON src/merge_translations.py \
                data/translation/non_english_attachments_translated_batches \
                data/scrape/initiative_details \
                --chunk-size 16384
    else
        echo "  (skipped — no translation output)"
    fi

    echo ""
    echo "--- Merging document/attachment summaries ---"
    if [ -d data/analysis/summaries ]; then
        do_merge_summaries
    else
        echo "  (skipped — no summaries directory)"
    fi

    echo ""
    echo "--- Merging change summaries ---"
    if [ -d data/analysis/change_summaries ]; then
        do_merge_change_summaries
    else
        echo "  (skipped — no change summaries directory)"
    fi

    echo ""
    echo "--- Merging cluster feedback summaries ---"
    if [ -n "$CLUSTER_SCHEMES" ]; then
        for scheme in $CLUSTER_SCHEMES; do
            if [ -d "data/cluster_summaries/${scheme}" ]; then
                count_json "data/cluster_summaries/${scheme}" "cluster summary files ($scheme)"
                run_local "merge cluster feedback summaries ($scheme)" \
                    $PYTHON src/merge_cluster_feedback_summaries.py \
                        "data/cluster_summaries/${scheme}" \
                        data/scrape/initiative_details
            else
                echo "  (skipped — no cluster summaries for $scheme)"
            fi
        done
    else
        echo "  (skipped — CLUSTER_SCHEMES not set)"
    fi

    echo ""
    echo "--- Rebuilding webapp index ---"
    do_build_index

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] RECOVER COMPLETE"
    echo "============================================================"
}

# ── Dispatch ─────────────────────────────────────────────────────────────────

STAGE="${1:?Usage: pipeline.sh <stage> [extra-args...]}"
shift

case "$STAGE" in
    list)               do_list ;;
    setup)              do_setup ;;
    setup-remote)       do_setup_remote ;;
    scrape)             do_scrape "$@" ;;
    find-short-pdfs)    do_find_short_pdfs "$@" ;;
    find-nonenglish)    do_find_nonenglish "$@" ;;
    merge-ocr)          do_merge_ocr "$@" ;;
    merge-translations) do_merge_translations "$@" ;;
    analyze)            do_analyze "$@" ;;
    build-summaries)    do_build_summaries "$@" ;;
    cluster)            do_cluster "$@" ;;
    build-index)        do_build_index "$@" ;;
    merge-summaries) do_merge_summaries "$@" ;;
    merge-change-summaries) do_merge_change_summaries "$@" ;;
    merge-cluster-feedback-summaries) do_merge_cluster_feedback_summaries "$@" ;;
    deploy)             do_deploy ;;
    push)               do_push "$@" ;;
    pull)               do_pull "$@" ;;
    remote)             do_remote "$@" ;;
    logs)               do_logs "$@" ;;
    clean-batches)      do_clean_batches "$@" ;;
    full)               do_full "$@" ;;
    recover)            do_recover ;;
    *)
        echo "ERROR: Unknown stage: $STAGE"
        echo "Run './pipeline.sh list' to see available stages."
        exit 1
        ;;
esac
