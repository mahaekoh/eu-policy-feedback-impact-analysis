"""Rewrite cluster summaries into shorter formats using vLLM batch inference.

Takes cluster summary output (from summarize_clusters.py) and unit summaries
(from build_unit_summaries.py), then rewrites each cluster summary in a specified
format (e.g. "reddit" — concise, punchy summaries).

For regular clusters, rewrites the cluster summary.
For noise clusters (-1:feedback_id), uses the feedback item's
combined_feedback_summary from unit summaries as richer input, falling back
to the cluster summary if not found.

Output: one JSON per initiative with format name and cluster_rewrites dict.

Usage:
    python3 src/rewrite_cluster_summaries.py \\
        data/cluster_summaries/<scheme>/ data/analysis/unit_summaries/ \\
        -o data/cluster_rewrites/reddit/<scheme>/ --format reddit
"""

import argparse
import json
import os
import time

import torch
from openai_harmony import (
    HarmonyEncodingName,
    ReasoningEffort,
    load_harmony_encoding,
)
from vllm import LLM, SamplingParams

from inference_utils import build_prefill, run_batch_inference

DEFAULT_MODEL = "unsloth/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 4096  # Rewrites are short

# ── Format registry ──

FORMATS = {
    "reddit": {
        "identity": (
            "You are a sharp policy analyst who writes like a top Reddit "
            "commenter -- clear, direct, no fluff."
        ),
        "prompt_prefix": (
            "Rewrite this cluster summary of public feedback on an EU policy initiative.\n\n"
            "Rules:\n"
            "- SHORT title on the first line (max 15 words, no markdown, no quotes)\n"
            "- Blank line\n"
            "- 2-4 sentences: core position and key arguments\n"
            "- Optionally: blank line + 3-5 bullet points with one specific detail each\n"
            "- Strip formality, meta-commentary, hedging\n"
            "- Do NOT repeat info from the policy summary — focus on what feedback ADDS\n"
            "- Do NOT invent details not in the original\n"
            "- This cluster contains {feedback_count} feedback item(s)\n\n"
            "=== POLICY CONTEXT (do not repeat) ===\n"
            "{before_feedback_summary}\n\n"
            "=== CLUSTER SUMMARY TO REWRITE ===\n"
        ),
    },
}


def parse_title_and_body(text: str) -> tuple[str, str]:
    """Split LLM output into (title, body).

    Expected format: first line = title, blank line, rest = body.
    """
    text = text.strip()
    if not text:
        return ("", "")

    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        title = parts[0].strip()
        body = parts[1].strip()
        if title and body:
            return (title, body)

    # Fallback: first line = title, rest = body
    lines = text.split("\n", 1)
    if len(lines) == 2:
        return (lines[0].strip(), lines[1].strip())

    return (text, text)


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite cluster summaries into shorter formats using vLLM."
    )
    parser.add_argument(
        "cluster_summary_dir",
        help="Directory of cluster summary JSON files (output of summarize_clusters.py)",
    )
    parser.add_argument(
        "unit_summaries_dir",
        help="Directory of unit summary JSON files (output of build_unit_summaries.py)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for rewrite JSONs.",
    )
    parser.add_argument(
        "--format", required=True, choices=list(FORMATS.keys()),
        help="Rewrite format name.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_OUTPUT_TOKENS,
        help=f"Max output tokens per rewrite (default: {MAX_OUTPUT_TOKENS}).",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Max model context length.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.15,
        help="Sampling temperature (default: 0.15).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048,
        help="Number of prompts per inference batch (default: 2048).",
    )
    args = parser.parse_args()

    fmt = FORMATS[args.format]
    identity_prompt = fmt["identity"]
    prompt_prefix_template = fmt["prompt_prefix"]

    # Initialize encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    reasoning_effort = ReasoningEffort.MEDIUM

    # Discover cluster summary files
    summary_files = sorted(
        f for f in os.listdir(args.cluster_summary_dir)
        if f.endswith(".json") and not f.startswith("_")
    )
    print(f"Found {len(summary_files)} cluster summary files in {args.cluster_summary_dir}/")

    # Resume: skip existing outputs
    os.makedirs(args.output, exist_ok=True)
    pending = []
    for f in summary_files:
        init_id = f.replace(".json", "")
        out_path = os.path.join(args.output, f"{init_id}.json")
        if not os.path.isfile(out_path):
            pending.append((init_id, f))

    skipped = len(summary_files) - len(pending)
    if skipped:
        print(f"Resume: {skipped}/{len(summary_files)} already exist, "
              f"{len(pending)} remaining")

    if not pending:
        print("\nAll output files already exist. Nothing to do.")
        return

    # Load unit summaries for before_feedback_summary and noise fallback
    print(f"\nLoading unit summaries from {args.unit_summaries_dir}/...")
    unit_data = {}  # init_id -> {before_feedback_summary, feedback_lookup}
    for init_id, _ in pending:
        unit_path = os.path.join(args.unit_summaries_dir, f"{init_id}.json")
        if not os.path.isfile(unit_path):
            continue
        with open(unit_path, encoding="utf-8") as fh:
            unit = json.load(fh)
        before_summary = unit.get("before_feedback_summary", "") or ""
        fb_lookup = {}
        for fb in unit.get("middle_feedback", []):
            fb_lookup[str(fb["id"])] = fb
        unit_data[init_id] = {
            "before_feedback_summary": before_summary,
            "feedback_lookup": fb_lookup,
        }
    print(f"Loaded unit summaries for {len(unit_data)}/{len(pending)} initiatives")

    # Build prompts
    print("\nBuilding prompts...")
    prompts = []
    prompt_texts = []
    prompt_map = []
    init_cluster_labels = {}  # init_id -> set of cluster labels

    for init_id, filename in pending:
        filepath = os.path.join(args.cluster_summary_dir, filename)
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)

        cluster_summaries = data.get("cluster_summaries", {})
        if not cluster_summaries:
            continue

        init_cluster_labels[init_id] = set(cluster_summaries.keys())
        unit = unit_data.get(init_id, {})
        before_summary = unit.get("before_feedback_summary", "")
        fb_lookup = unit.get("feedback_lookup", {})

        for label, cs in cluster_summaries.items():
            summary_text = cs.get("summary", "")
            feedback_count = cs.get("feedback_count", 1)

            # For noise clusters, prefer combined_feedback_summary from unit summaries
            if label.startswith("-1:"):
                fb_id = label.split(":", 1)[1]
                fb = fb_lookup.get(fb_id)
                if fb and fb.get("combined_feedback_summary"):
                    summary_text = fb["combined_feedback_summary"]

            if not summary_text:
                continue

            prefix = prompt_prefix_template.format(
                before_feedback_summary=before_summary or "(not available)",
                feedback_count=feedback_count,
            )
            prefill = build_prefill(
                encoding, summary_text, prefix,
                reasoning_effort, identity_prompt,
                args.max_model_len,
            )
            if prefill:
                prompts.append(prefill)
                prompt_texts.append(prefix + summary_text)
                prompt_map.append((f"rw:{init_id}:{label}", 0))

    print(f"Collected {len(prompts)} prompts across "
          f"{len(init_cluster_labels)} initiatives")

    if not prompts:
        print("\nNo prompts to process. Writing empty output files.")
        for init_id, filename in pending:
            filepath = os.path.join(args.cluster_summary_dir, filename)
            with open(filepath, encoding="utf-8") as fh:
                data = json.load(fh)
            out = {
                "initiative_id": data.get("initiative_id", init_id),
                "format": args.format,
                "cluster_rewrites": {},
            }
            out_path = os.path.join(args.output, f"{init_id}.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
        return

    # Initialize vLLM
    tp_size = torch.cuda.device_count()
    print(f"\nCUDA device count: {tp_size}")
    print(f"Loading model {args.model} (tp={tp_size})...")
    t0 = time.time()
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": tp_size,
        "max_num_seqs": 128,
        "async_scheduling": True,
    }
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=encoding.stop_tokens_for_assistant_actions(),
    )

    # Run batch inference
    batch_dir = os.path.join(args.output, "_batches")
    os.makedirs(batch_dir, exist_ok=True)
    summary_cache = {}

    t_start = time.time()
    results, failed, stats = run_batch_inference(
        llm, sampling_params, encoding,
        prompts, prompt_texts, prompt_map,
        args.batch_size, batch_dir, summary_cache,
        label="[rewrite] ",
    )
    elapsed = time.time() - t_start
    print(f"\nInference complete in {elapsed:.1f}s")
    if failed:
        print(f"  {len(failed)} prompts FAILED")

    # Parse results and write output files
    print("\nWriting output files...")
    n_written = 0
    n_rewrites = 0

    for init_id, filename in pending:
        labels = init_cluster_labels.get(init_id, set())
        rewrites = {}

        for label in labels:
            key = f"rw:{init_id}:{label}"
            chunks = results.get(key, {})
            if 0 in chunks:
                title, body = parse_title_and_body(chunks[0])
                if title and body:
                    rewrites[label] = {"title": title, "body": body}
                    n_rewrites += 1

        filepath = os.path.join(args.cluster_summary_dir, filename)
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)

        out = {
            "initiative_id": data.get("initiative_id", init_id),
            "format": args.format,
            "cluster_rewrites": rewrites,
        }
        out_path = os.path.join(args.output, f"{init_id}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        n_written += 1

    print(f"Wrote {n_written} files, {n_rewrites} cluster rewrites total")

    if failed:
        failed_path = os.path.join(args.output, "_all_failed.json")
        with open(failed_path, "w", encoding="utf-8") as fh:
            json.dump(failed, fh, ensure_ascii=False, indent=2)
        print(f"FAILED: {len(failed)} prompts. Wrote {failed_path}")

    total_elapsed = time.time() - t_start
    print(f"\nDONE in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
