"""Summarize changes between before- and after-feedback documents using vLLM batch inference.

Takes the output directory from build_unit_summaries.py, which contains per-initiative
JSON files with before_feedback_summary and after_feedback_summary fields. For each
initiative that has both summaries, computes a unified diff and asks the LLM to
summarize the substantive policy changes. Adds a 'change_summary' field at top level.

Initiatives missing either summary are copied through unchanged.

Usage:
    python3 src/summarize_changes.py data/analysis/unit_summaries/ -o data/analysis/change_summaries/
    python3 src/summarize_changes.py data/analysis/unit_summaries/ -o data/analysis/change_summaries/ --batch-size 16
"""

import argparse
import difflib
import json
import os
import sys
import time

import torch
from openai_harmony import (
    HarmonyEncodingName,
    ReasoningEffort,
    load_harmony_encoding,
)
from vllm import LLM, SamplingParams

from inference_utils import build_prefill, extract_final_texts, run_batch_inference

DEFAULT_MODEL = "unsloth/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 32768 * 4

IDENTITY_PROMPT = (
    "You are a policy analyst who compares EU regulatory documents "
    "before and after public consultation feedback."
)

CHANGE_SUMMARY_PROMPT = (
    "Below are summaries of EU policy documents published BEFORE and AFTER "
    "a public feedback period, along with a unified diff showing the changes.\n\n"
    "=== BEFORE FEEDBACK ===\n{before}\n\n"
    "=== AFTER FEEDBACK ===\n{after}\n\n"
    "=== DIFF ===\n{diff}\n\n"
    "Summarize the substantive changes between the before and after documents "
    "in up to 10 paragraphs. Focus on policy changes, not formatting. "
    "Be specific about what was added, removed, or modified. "
    "If any, preserve all points about nuclear energy, nuclear plants, "
    "or small modular reactors. Do not generate any mete commentary "
    "(for example stating that there are no nuclear-related points)."
)


def compute_diff(before: str, after: str) -> str:
    """Compute a unified diff between before and after summaries."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines, after_lines,
        fromfile="before_feedback", tofile="after_feedback",
    )
    return "".join(diff)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize changes between before- and after-feedback documents using vLLM."
    )
    parser.add_argument(
        "input_dir",
        help="Directory of per-initiative JSON files (output of build_unit_summaries.py)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for initiative JSONs with change_summary added.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_OUTPUT_TOKENS,
        help=f"Max output tokens per summary (default: {MAX_OUTPUT_TOKENS}).",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=32768*4,
        help="Max model context length. Set lower to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.15,
        help="Sampling temperature (default: 0.15).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of prompts per inference batch (default: 32).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Initialize openai_harmony encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    reasoning_effort = ReasoningEffort.MEDIUM

    # List all initiative files
    initiative_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith(".json")
    )
    print(f"Found {len(initiative_files)} initiative files in {args.input_dir}/")

    if not initiative_files:
        print("Nothing to do.")
        return

    # Resume: skip files whose output already exists
    os.makedirs(args.output, exist_ok=True)
    pending_files = [
        f for f in initiative_files
        if not os.path.isfile(os.path.join(args.output, f))
    ]
    skipped_files = len(initiative_files) - len(pending_files)
    if skipped_files:
        print(f"Resume: {skipped_files}/{len(initiative_files)} output files already exist, "
              f"{len(pending_files)} remaining")

    if not pending_files:
        print("\nAll output files already exist. Nothing to do.")
        return

    # Load pending files and build prompts
    prompts = []
    prompt_texts = []
    prompt_map = []       # (file_index, 0) — one prompt per initiative
    copy_through = []     # file indices that need no inference
    initiatives = []      # loaded JSON data, indexed by file_index
    diffs = {}            # file_index -> diff string

    for file_index, filename in enumerate(pending_files):
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            initiative = json.load(f)
        initiatives.append(initiative)

        before = initiative.get("before_feedback_summary", "")
        after = initiative.get("after_feedback_summary", "")

        if not before or not after:
            copy_through.append(file_index)
            continue

        diff = compute_diff(before, after)
        diffs[file_index] = diff
        prompt_text = CHANGE_SUMMARY_PROMPT.format(
            before=before, after=after, diff=diff,
        )

        prefill = build_prefill(encoding, prompt_text, "", reasoning_effort, IDENTITY_PROMPT)
        if prefill is None:
            print(f"  {filename}: SKIPPED (encoding failed)")
            copy_through.append(file_index)
            continue

        n_tokens = len(prefill["prompt_token_ids"])
        print(f"  {filename}: {len(prompt_text)} chars, {n_tokens} tokens")
        prompts.append(prefill)
        prompt_texts.append(prompt_text)
        prompt_map.append((file_index, 0))

    n_with_both = len(prompts)
    n_copy = len(copy_through)
    print(f"\nInitiatives with both summaries: {n_with_both}")
    print(f"Initiatives to copy through (missing before or after): {n_copy}")

    # If no prompts, copy everything through without loading the model
    if not prompts:
        for file_index, filename in enumerate(pending_files):
            out_path = os.path.join(args.output, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(initiatives[file_index], f, ensure_ascii=False, indent=2)
        print(f"\nNo change summaries needed. Copied {len(pending_files)} files unchanged.")
        return

    # Initialize vLLM (deferred until we know there is work)
    tp_size = torch.cuda.device_count()
    print(f"\nCUDA device count: {tp_size}")
    print(f"Loading model {args.model} (tp={tp_size})...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp_size,
        max_num_seqs=128,
        async_scheduling=True,
        max_model_len=args.max_model_len,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=encoding.stop_tokens_for_assistant_actions(),
    )

    # Run inference — single pass, no chunking
    batch_dir = os.path.join(args.output, "_batches")
    os.makedirs(batch_dir, exist_ok=True)

    summary_cache = {}
    summarized_chunks, failed_prompts, stats = run_batch_inference(
        llm, sampling_params, encoding,
        prompts, prompt_texts, prompt_map,
        args.batch_size, batch_dir, summary_cache,
        batch_num_start=0, label="",
    )

    # Write output files
    written = 0
    summaries_added = 0
    for file_index, filename in enumerate(pending_files):
        initiative = initiatives[file_index]

        # Add change_summary and diff if we got one
        result = summarized_chunks.get(file_index, {}).get(0)
        if result is not None:
            initiative["change_summary"] = result
            initiative["diff"] = diffs[file_index]
            summaries_added += 1

        out_path = os.path.join(args.output, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(initiative, f, ensure_ascii=False, indent=2)
        written += 1

    # Stats
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"Initiative files: {written + skipped_files}/{len(initiative_files)} "
          f"({written} written, {skipped_files} already existed)")
    print(f"Change summaries added: {summaries_added}/{n_with_both}")
    print(f"Copied through unchanged: {n_copy}")

    if failed_prompts:
        failed_file = os.path.join(args.output, "_failed.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_prompts, f, ensure_ascii=False, indent=2)
        print(f"FAILED: {len(failed_prompts)} prompts could not be parsed. Wrote {failed_file}")


if __name__ == "__main__":
    main()
