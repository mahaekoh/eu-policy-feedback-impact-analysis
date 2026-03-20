# Deployment and Setup Guide

This guide covers everything needed to set up and run the EU Policy Feedback Transparency Platform from scratch, including local development, remote GPU configuration, and the web application.

## Table of Contents

- [Requirements](#requirements)
- [Python Dependencies](#python-dependencies)
- [Local Setup](#local-setup)
- [Remote GPU Setup](#remote-gpu-setup)
- [Configuration Reference](#configuration-reference-pipelineconf)
- [Webapp Setup](#webapp-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Remote Execution Model](#remote-execution-model)
- [Recovery](#recovery)
- [Troubleshooting](#troubleshooting)

---

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.12 | See `.python-version` in the repository root |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager. Lockfile: `uv.lock` |
| Node.js | >= 18 | Required only for the webapp (Next.js 16) |
| NVIDIA GPU | H100 recommended | Required for OCR, translation, summarization, clustering, and classification. Not needed for scraping, merging, or the webapp. |
| macOS `textutil` | (system) | For `.doc` text extraction. Optional; only available on macOS. |
| Hugging Face account | -- | Required for downloading models (`unsloth/gpt-oss-120b`, `google/embeddinggemma-300m`). Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |

The local machine handles scraping, merging, text extraction, index building, and the webapp. The remote GPU host handles OCR, translation, summarization, clustering, classification, and change summarization. You can run the full pipeline with just a local machine if you have the necessary GPU hardware, but the standard workflow assumes a remote GPU server.

---

## Python Dependencies

Dependencies are split into two groups in `pyproject.toml`:

### Base dependencies (local machine)

Installed with `uv sync`. These are sufficient for scraping, merging, text extraction, and index building.

| Package | Purpose |
|---|---|
| pymupdf / pymupdf4llm | PDF text extraction with tesseract OCR fallback (300 DPI) |
| docx2md | DOCX text extraction |
| pypandoc / pypandoc_binary | RTF and ODT text extraction |
| huggingface-hub | Model downloads and Hugging Face CLI |

### GPU dependencies (remote host)

Installed with `uv sync --extra gpu` or `pip install` on the remote GPU host. These are needed for all GPU-accelerated pipeline stages.

| Package | Purpose |
|---|---|
| vllm | LLM batch inference engine |
| openai-harmony | Structured prompt encoding for `unsloth/gpt-oss-120b` (reasoning effort, stop tokens, output parsing) |
| easyocr | GPU-accelerated OCR (CUDA) |
| sentence-transformers | Sentence embeddings (`google/embeddinggemma-300m`) |
| scikit-learn | AgglomerativeClustering algorithm |
| hdbscan | HDBSCAN clustering algorithm |
| cuml-cu12 (optional) | GPU-accelerated clustering via RAPIDS. Installed via `pip install --extra-index-url https://pypi.nvidia.com cuml-cu12`. |
| torch | PyTorch (CUDA) |
| numpy | Numerical operations |

---

## Local Setup

### Automated setup

```bash
./pipeline.sh setup
```

This command:

1. Runs `uv sync` to install all base Python dependencies from the lockfile
2. Checks whether you are already logged in to Hugging Face
3. If not logged in, runs `huggingface-cli login` interactively (you will need your Hugging Face token)

### Manual setup

If you prefer to set up manually:

```bash
# Install Python dependencies
uv sync

# Log in to Hugging Face (required for model downloads on first use)
huggingface-cli login
```

After setup, you can immediately run scraping, merging, index building, and the webapp without any GPU.

---

## Remote GPU Setup

The remote GPU host runs all compute-intensive stages: OCR (EasyOCR/CUDA), translation (120B LLM), summarization (120B LLM), clustering (sentence embeddings + cuML), classification (120B LLM), and change summarization (120B LLM).

### Step 1: Create the configuration file

```bash
cp pipeline.conf.example pipeline.conf
```

### Step 2: Fill in your remote host details

Edit `pipeline.conf` with your values:

```bash
REMOTE_HOST="user@gpu-host"
REMOTE_DIR="/home/user/eu-pipeline"
SSH_KEY="~/.ssh/id_ed25519"
PYTHON="python3"

CLUSTER_SCHEMES="agglomerative_google_embeddinggemma-300m_distance_threshold=0.75_linkage=average_max_cluster_size=20_max_depth=3_sub_cluster_scale=0.75"
```

See [Configuration Reference](#configuration-reference-pipelineconf) below for details on each variable.

### Step 3: Run setup-remote

```bash
./pipeline.sh setup-remote
```

This command performs three actions:

1. **Deploys source code**: rsyncs the `src/` directory to `REMOTE_DIR/src/` on the remote host
2. **Installs GPU Python dependencies**: runs `pip install` on the remote host with the NVIDIA extra index URL. The full install command is:
   ```bash
   python3 -m pip install \
       --extra-index-url https://pypi.nvidia.com \
       vllm openai-harmony \
       easyocr \
       sentence-transformers scikit-learn hdbscan \
       cuml-cu12 \
       torch numpy \
       huggingface-hub
   ```
3. **Logs in to Hugging Face on the remote**: if not already logged in, opens an interactive SSH session for `huggingface-cli login` (requires a terminal with TTY support)

### Prerequisites for the remote host

- SSH access with key-based authentication
- Python >= 3.12 installed (or the path specified in `PYTHON`)
- NVIDIA GPU drivers and CUDA toolkit installed
- Sufficient disk space for model weights (~240 GB for `unsloth/gpt-oss-120b`) and pipeline data
- `pip` available for the Python executable

---

## Configuration Reference (`pipeline.conf`)

The configuration file `pipeline.conf` is sourced by `pipeline.sh` as a Bash script. It is gitignored.

### Required variables

| Variable | Default | Description |
|---|---|---|
| `REMOTE_HOST` | -- (required) | SSH host in `user@hostname` format. Used for all SSH and rsync operations. |
| `REMOTE_DIR` | -- (required) | Absolute path to the working directory on the remote host. All pipeline data and source code are stored here. |
| `SSH_KEY` | -- (required) | Path to the SSH private key for connecting to the remote host. Supports `~` expansion. |
| `PYTHON` | `python3` | Python executable name or path on the remote host. Override if the GPU host uses a different Python binary (e.g. `python3.12`). |
| `CLUSTER_SCHEMES` | -- | Space-separated clustering scheme directory names. Required for clustering and cluster summarization stages. |

### CLUSTER_SCHEMES format

Each scheme name is a single string that encodes the clustering algorithm, embedding model, and all parameters. `pipeline.sh` parses the name by splitting on underscores and extracts:

1. **Algorithm** (first token): `agglomerative` or `hdbscan`
2. **Model** (second and third tokens, joined with `/`): e.g. `google/embeddinggemma-300m`
3. **Parameters** (remaining tokens): `key=value` pairs, where multi-word keys (e.g. `max_cluster_size`) are reassembled from underscore-separated parts

The parsed values are converted to CLI flags for `cluster_all_initiatives.py`. For example, `distance_threshold=0.75` becomes `--distance-threshold 0.75`.

#### Full example with two schemes

```bash
CLUSTER_SCHEMES="agglomerative_google_embeddinggemma-300m_distance_threshold=0.75_linkage=average_max_cluster_size=20_max_depth=3_sub_cluster_scale=0.75 hdbscan_google_embeddinggemma-300m_min_cluster_size=5_min_samples=3_max_cluster_size=20_max_depth=4"
```

This defines two clustering schemes:

**Scheme 1** (agglomerative):
- Algorithm: `agglomerative`
- Model: `google/embeddinggemma-300m`
- `--distance-threshold 0.75`
- `--linkage average`
- `--max-cluster-size 20`
- `--max-depth 3`
- `--sub-cluster-scale 0.75`

**Scheme 2** (HDBSCAN):
- Algorithm: `hdbscan`
- Model: `google/embeddinggemma-300m`
- `--min-cluster-size 5`
- `--min-samples 3`
- `--max-cluster-size 20`
- `--max-depth 4`

The scheme name also becomes the output subdirectory under `data/clustering/` and `data/cluster_summaries/`, so each scheme's results are kept separate.

---

## Webapp Setup

The webapp is a Next.js 16 application that provides an interactive interface for browsing initiatives, feedback, summaries, and clusters.

### Prerequisites

1. **Node.js >= 18** installed
2. **Pre-computed data**: the webapp reads data from `data/webapp/` and `data/clustering/`, which must be generated by running the pipeline (specifically `build_webapp_index.py`)

### Installation and development

```bash
cd webapp
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). No sign-in is required -- the app is fully accessible without authentication.

### Production build

```bash
cd webapp
npm run build
npm start
```

### Optional: Google OAuth

Authentication is optional. If you want to enable Google sign-in:

1. Go to [Google Cloud Console > APIs & Services > Credentials](https://console.cloud.google.com/apis/credentials)
2. Create an **OAuth 2.0 Client ID** (application type: Web application)
3. Add authorized redirect URIs:
   - Local development: `http://localhost:3000/api/auth/callback/google`
   - Production: `https://<your-domain>/api/auth/callback/google`
4. Create the environment file `webapp/.env.local` with:

```bash
# Generate with: npx auth secret
AUTH_SECRET="your-jwt-signing-key"

# From Google Cloud Console
AUTH_GOOGLE_ID="your-google-client-id"
AUTH_GOOGLE_SECRET="your-google-client-secret"
```

Or use the quick setup:

```bash
cd webapp
npx auth secret          # writes AUTH_SECRET to .env.local automatically
# Then manually add AUTH_GOOGLE_ID and AUTH_GOOGLE_SECRET to .env.local
```

See `webapp/AUTH.md` for the full authentication architecture and session access patterns.

### Data paths at runtime

The webapp reads from these paths (all relative to `webapp/`):

| Path | Source | Content |
|---|---|---|
| `../data/webapp/initiative_index.json` | `build_webapp_index.py` | Initiative list for the index page |
| `../data/webapp/global_stats.json` | `build_webapp_index.py` | Aggregate statistics for the charts page |
| `../data/webapp/country_stats.json` | `build_webapp_index.py` | Per-country statistics for charts drill-down |
| `../data/webapp/initiative_details/*.json` | `build_webapp_index.py` | Stripped initiative details (no `extracted_text` on attachments) |
| `../data/clustering/<scheme>/*.json` | `cluster_all_initiatives.py` | Cluster assignments for the cluster view |

Data is loaded server-side with a 5-minute in-memory cache (`CACHE_TTL_MS = 300000`).

---

## Running the Pipeline

### Quick reference

| Command | Description |
|---|---|
| `./pipeline.sh full` | Run the entire pipeline end-to-end (all stages in order) |
| `./pipeline.sh list` | Show all stages with descriptions |
| `./pipeline.sh <stage>` | Run a single stage |
| `./pipeline.sh deploy` | Sync source code to the remote host |
| `./pipeline.sh remote <step>` | Run a GPU step on the remote host |
| `./pipeline.sh push <target>` | Upload data to the remote host |
| `./pipeline.sh pull <target>` | Download results from the remote host |
| `./pipeline.sh logs` | List recent remote log files |
| `./pipeline.sh logs tail [step]` | Tail a specific step's log in real-time |
| `./pipeline.sh clean-batches <target>` | Delete batch recovery files on the remote host |
| `./pipeline.sh recover` | Pull all outputs from remote, merge locally, rebuild index |

### Push targets

| Target | Data pushed |
|---|---|
| `initiative-details` | `data/scrape/initiative_details/` |
| `ocr` | `data/ocr/` (report + PDFs) |
| `translation` | `data/translation/non_english_attachments.json` |
| `analysis` | `data/analysis/before_after/` |
| `unit-summaries` | `data/analysis/unit_summaries/` |
| `clustering` | `data/clustering/` |
| `cluster-rewrites` | `data/cluster_rewrites/` |
| `all` | All of the above |

### Pull targets

| Target | Data pulled | rsync strategy |
|---|---|---|
| `initiative-details` | `data/scrape/initiative_details/` | Overwrite (remote is authoritative) |
| `ocr` | `data/ocr/short_pdf_report_ocr.json` | Overwrite (single file, regenerated each run) |
| `translation` | Combined JSON + batch directory | Overwrite for combined file; skip existing for batches |
| `summaries` | `data/analysis/summaries/` | Skip existing (immutable output files) |
| `classification` | `data/classification/` | Skip existing (immutable output files) |
| `clustering` | `data/clustering/` | Skip existing |
| `embeddings` | `data/embeddings/` | Overwrite (files updated when data changes) |
| `cluster-summaries` | `data/cluster_summaries/` | Skip existing (immutable output files) |
| `cluster-rewrites` | `data/cluster_rewrites/` | Skip existing (immutable output files) |
| `change-summaries` | `data/analysis/change_summaries/` | Skip existing (immutable output files) |
| `webapp` | `data/webapp/` | Overwrite |
| `logs` | Remote `logs/` directory | Overwrite |
| `all` | All of the above (except `logs`) | Mixed |

### Remote GPU steps

| Step | Script | Description |
|---|---|---|
| `ocr` | `ocr_short_pdfs.py` | GPU-accelerated OCR (EasyOCR, multi-GPU) |
| `translate` | `translate_attachments.py` | Translate non-English feedback (120B LLM) |
| `summarize` | `summarize_documents.py` | Summarize documents and attachments (120B LLM) |
| `classify` | `classify_initiative_and_feedback.py` | Classify initiatives and feedback (120B LLM) |
| `cluster` | `cluster_all_initiatives.py` | Cluster feedback by topic (sentence embeddings, multi-GPU) |
| `summarize-clusters` | `summarize_clusters.py` | Summarize feedback clusters (120B LLM, per scheme) |
| `rewrite-clusters` | `rewrite_cluster_summaries.py` | Rewrite cluster summaries (120B LLM, per scheme) |
| `summarize-changes` | `summarize_changes.py` | Summarize before/after document changes (120B LLM) |

### Composite remote pipeline steps

For convenience, `pipeline.sh` also supports composite steps that chain multiple operations on the remote host into a single SSH session. These run the find/analyze step, the GPU step, and the merge step all on the remote host sequentially:

| Composite step | Chains |
|---|---|
| `ocr-pipeline` | find short PDFs + GPU OCR + merge OCR results |
| `translate-pipeline` | find non-English + GPU translate + merge translations |
| `summarize-pipeline` | analyze (initiative stats) + GPU summarize + build unit summaries |
| `cluster-summarize-pipeline` | GPU cluster summarize + merge cluster summaries (per scheme) |
| `rewrite-clusters-pipeline` | GPU rewrite clusters + merge rewrites (per scheme) |
| `change-summarize-pipeline` | GPU change summarize + merge change summaries |

The `full` pipeline uses these composite steps to minimize SSH round trips and data transfers.

### Running the full pipeline

```bash
# One-time setup
./pipeline.sh setup
./pipeline.sh setup-remote

# Run everything
./pipeline.sh full
```

The full pipeline follows this sequence:

1. **Scrape** locally (initiatives list + details with text extraction)
2. **Deploy** source code to remote
3. **Push** initiative details to remote
4. **OCR pipeline** on remote (find + OCR + merge)
5. **Translation pipeline** on remote (find + translate + merge)
6. **Summarization pipeline** on remote (analyze + summarize + build summaries)
7. **Clustering** on remote (sentence embeddings, per scheme)
8. **Cluster summarization pipeline** on remote (summarize + merge, per scheme)
9. **Cluster rewrite pipeline** on remote (rewrite + merge, per scheme)
10. **Change summarization pipeline** on remote (summarize + merge)
11. **Build webapp index** on remote
12. **Pull** all results back to local

### Running individual stages

You can run any stage independently. Common workflows:

```bash
# Re-scrape and update just the initiative data
./pipeline.sh scrape

# Deploy code changes to remote
./pipeline.sh deploy

# Run just the summarization step on remote
./pipeline.sh remote summarize

# Pull just the summaries
./pipeline.sh pull summaries

# Merge summaries locally and rebuild the index
./pipeline.sh merge-summaries
./pipeline.sh build-index
```

Extra arguments are passed through to the underlying Python scripts:

```bash
# Force re-scrape all initiatives (max-age=0)
./pipeline.sh scrape --max-age 0

# Run summarization with reduced context length to save GPU memory
./pipeline.sh remote summarize --max-model-len 65536
```

---

## Remote Execution Model

Understanding how remote GPU jobs are managed is important for operating the pipeline reliably.

### How remote jobs run

1. `pipeline.sh` connects via SSH and launches the Python script inside `nohup`, so the process survives SSH disconnects
2. stdout and stderr are redirected to a log file under `logs/` on the remote host (e.g. `logs/summarize_20250115_143022.log`)
3. When the Python script exits, its exit code is written to a `.exit` status file alongside the log
4. The local terminal tails the remote log file in real-time via a second SSH channel
5. When the `.exit` file appears, the local terminal reads the exit code and reports success or failure
6. On success, any batch recovery directories (`_batches*`) are automatically cleaned up on the remote

### SSH resilience

If your SSH connection drops during a remote job:

- The GPU process continues running on the remote host (thanks to `nohup`)
- No work is lost
- To monitor progress, reconnect and tail the log:
  ```bash
  ./pipeline.sh logs tail summarize
  ```
- Or list all recent logs:
  ```bash
  ./pipeline.sh logs
  ```

### Log files

Remote log files are stored at `REMOTE_DIR/logs/` with the naming pattern `{step}_{YYYYMMDD_HHMMSS}.log`. You can pull all logs locally:

```bash
./pipeline.sh pull logs
# Logs are downloaded to data/logs/
```

### Batch recovery files

All vLLM-based scripts (summarize, translate, classify, summarize-clusters, summarize-changes) write intermediate results to per-batch JSON files in `_batches*/` directories. If a GPU job crashes mid-run:

1. Restart the same command -- it will detect existing batch files and resume from where it left off
2. Only the incomplete batch needs to be re-run

Batch directories are auto-cleaned by `pipeline.sh` after successful remote runs. You can also clean them manually:

```bash
./pipeline.sh clean-batches summaries
./pipeline.sh clean-batches cluster-summaries
./pipeline.sh clean-batches change-summaries
./pipeline.sh clean-batches translation
./pipeline.sh clean-batches all
```

### Parallel data transfer

Push and pull operations use parallel rsync with 4 streams (`PARALLEL_JOBS=4`). Large directory transfers are split into chunks using `--files-from`, with each chunk transferred by a separate rsync process. This significantly speeds up transfers of directories with thousands of small JSON files.

---

## Recovery

The `recover` command is designed for situations where a pipeline run has partially completed -- either due to a crash, SSH disconnect, or manual interruption. It pulls whatever results exist on the remote host and merges them locally.

### What `./pipeline.sh recover` does

1. **Pulls initiative details** from the remote (the authoritative copy with any merges applied during the remote pipeline steps)
2. **Pulls all pipeline outputs** (OCR, translation, summaries, change summaries, cluster summaries, cluster rewrites, clustering, embeddings). Each pull step uses `|| true` so missing directories (from stages that never ran) do not abort the recovery.
3. **Runs all merge scripts locally** to integrate pulled results into initiative_details:
   - Merge OCR results (if OCR report exists)
   - Merge translations (from combined file or batch directory)
   - Merge document/attachment summaries
   - Merge change summaries
   - Merge cluster feedback summaries (per scheme)
   - Merge cluster rewrites (per format and scheme)
4. **Rebuilds the webapp index** from the updated initiative_details

### When to use recover

- A remote GPU job crashed partway through and you want to salvage the results that were written before the crash
- The full pipeline was interrupted and you want to collect everything that completed successfully
- You ran individual remote steps manually and need to sync results back and merge

### Usage

```bash
./pipeline.sh recover
```

The command is idempotent -- running it multiple times is safe. Merge scripts that find no input data are skipped with a message.

---

## Troubleshooting

### SSH connection drops during a remote job

GPU jobs run via `nohup` and are not affected by SSH disconnects. The process continues running on the remote host.

To check on the job:

```bash
# List recent logs
./pipeline.sh logs

# Tail the specific step's log
./pipeline.sh logs tail summarize

# Or tail the most recent log (any step)
./pipeline.sh logs tail
```

When the job finishes, its exit code is written to a `.exit` file. You can then pull results normally.

### Out of GPU memory

If a vLLM-based step runs out of GPU memory, reduce the model context length:

```bash
./pipeline.sh remote summarize --max-model-len 65536
./pipeline.sh remote translate --max-model-len 65536
./pipeline.sh remote summarize-changes --max-model-len 65536
./pipeline.sh remote summarize-clusters --max-model-len 65536
```

The default `--max-model-len` is `131072` (128K tokens) for most steps. Reducing it to 65536 or lower reduces GPU memory usage at the cost of truncating very long inputs.

### Corrupt JSON files in initiative_details

The scraper (`scrape_eu_initiative_details.py`) automatically detects corrupt JSON files (from truncated writes during interrupted previous runs). When it encounters a corrupt file, it logs a warning and re-fetches the initiative from the API from scratch instead of crashing.

If you encounter corrupt files outside of scraping, you can delete them and re-run the relevant stage. The pipeline's file-level resume logic will regenerate the missing output.

### Missing Python dependencies (local vs remote)

The local and remote hosts have different dependency sets:

| Error | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pymupdf'` | Base deps not installed locally | Run `uv sync` or `./pipeline.sh setup` |
| `ModuleNotFoundError: No module named 'vllm'` | GPU deps not installed on remote | Run `./pipeline.sh setup-remote` |
| `ModuleNotFoundError: No module named 'cuml'` | cuML (RAPIDS) not installed on remote | Install via `pip install --extra-index-url https://pypi.nvidia.com cuml-cu12`. This is optional -- clustering falls back to CPU scikit-learn without it. |

### Hugging Face model download failures

If model downloads fail on the remote host:

1. Verify the Hugging Face login: `ssh -i ~/.ssh/your_key user@gpu-host "huggingface-cli whoami"`
2. Re-run login if needed: `ssh -t -i ~/.ssh/your_key user@gpu-host "huggingface-cli login"`
3. Ensure the token has read access to the required models (`unsloth/gpt-oss-120b`, `google/embeddinggemma-300m`)

### Stale initiative data after re-scraping

The scraper uses a merge strategy that preserves derived fields (summaries, translations, OCR results) when re-fetching initiatives, as long as the source material (page count, file size, feedback text) has not changed. If you want to force a full re-scrape without preserving any cached data:

```bash
# Force re-fetch all initiatives regardless of age
./pipeline.sh scrape --max-age 0
```

Note that this will still preserve derived fields on unchanged documents. To fully regenerate summaries or other derived data, delete the relevant output files and re-run the corresponding stage.

### LLM stages show "no work to do" and exit immediately

This is expected behavior. All LLM-based stages (summarize, classify, summarize-clusters, summarize-changes) use file-level resume: they skip initiatives whose output file already exists. The model is not loaded if there is no work.

To force regeneration, delete the specific output file(s) and re-run:

```bash
# Regenerate summaries for a specific initiative
rm data/analysis/summaries/12345.json
./pipeline.sh remote summarize

# Regenerate all summaries (caution: full re-run)
rm -rf data/analysis/summaries/*.json
./pipeline.sh remote summarize
```

### Rsync transfer errors

Push/pull operations use parallel rsync (4 streams). If one stream fails, a warning is printed but the overall operation may still succeed for other files. Re-run the push or pull command to retry failed transfers -- rsync is idempotent and will only transfer files that are missing or different.

If transfers are slow or unreliable, check that `ServerAliveInterval` and `ServerAliveCountMax` are set in your SSH config (pipeline.sh sets these on its rsync connections: `ServerAliveInterval=30`, `ServerAliveCountMax=5`).
