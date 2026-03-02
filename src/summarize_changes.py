"""Summarize changes between before- and after-feedback documents using vLLM batch inference.

Takes the output directory from build_unit_summaries.py, which contains per-initiative
JSON files with before_feedback_summary and after_feedback_summary fields. For each
initiative that has both summaries, computes a unified diff and asks the LLM to
summarize the substantive policy changes. Adds a 'change_summary' field at top level.

When the combined prompt (before + after + diff) exceeds the chunk size, the before and
after texts are split into aligned chunks, diffs are computed per chunk pair, and change
summaries are recursively combined using a character budget — the same strategy used by
summarize_documents.py.

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

from inference_utils import build_prefill, run_batch_inference
from text_utils import group_by_char_budget, split_into_chunks

DEFAULT_MODEL = "unsloth/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 32768 * 4
CHUNK_SIZE = 16384
COMBINE_BUDGET = CHUNK_SIZE * 4

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

COMBINE_PREFIX = (
    "The following are change summaries from consecutive sections of EU policy "
    "documents before and after a public feedback period. Combine them into a "
    "single summary up to 10 paragraphs. Focus on policy changes, not formatting. "
    "Be specific about what was added, removed, or modified. "
    "If any, preserve all points about nuclear energy, nuclear plants, "
    "or small modular reactors. Do not generate any mete commentary "
    "(for example stating that there are no nuclear-related points).\n\n"
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
        "--batch-size", type=int, default=2048,
        help="Number of prompts per inference batch (default: 2048).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Max chars per chunk for splitting large inputs (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--combine-budget", type=int, default=COMBINE_BUDGET,
        help=f"Max chars when grouping summaries for combining (default: {COMBINE_BUDGET}).",
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
    # prompt_map uses (file_index, chunk_index) — chunk_index=0 for single-chunk items
    prompts = []
    prompt_texts = []
    prompt_map = []
    item_chunk_counts = {}  # file_index -> number of chunks
    copy_through = []       # file indices that need no inference
    initiatives = []        # loaded JSON data, indexed by file_index
    diffs = {}              # file_index -> diff string (for single-chunk items)

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
        prompt_text = CHANGE_SUMMARY_PROMPT.format(
            before=before, after=after, diff=diff,
        )

        # Try single-prompt approach first
        max_prompt_tokens = args.max_model_len
        prefill = build_prefill(encoding, prompt_text, "", reasoning_effort,
                                IDENTITY_PROMPT, max_prompt_tokens)
        if prefill is not None:
            n_tokens = len(prefill["prompt_token_ids"])
            print(f"  {filename}: {len(prompt_text)} chars, {n_tokens} tokens")
            prompts.append(prefill)
            prompt_texts.append(prompt_text)
            prompt_map.append((file_index, 0))
            item_chunk_counts[file_index] = 1
            diffs[file_index] = diff
            continue

        # Too large for a single prompt — chunk before and after, diff each pair
        print(f"  {filename}: too large ({len(prompt_text)} chars), chunking...")
        before_chunks = split_into_chunks(before, args.chunk_size, label=filename)
        after_chunks = split_into_chunks(after, args.chunk_size, label=filename)

        # Align chunks: pad the shorter list with empty strings
        n_chunks = max(len(before_chunks), len(after_chunks))
        while len(before_chunks) < n_chunks:
            before_chunks.append("")
        while len(after_chunks) < n_chunks:
            after_chunks.append("")

        chunk_count = 0
        for ci in range(n_chunks):
            b_chunk = before_chunks[ci]
            a_chunk = after_chunks[ci]
            if not b_chunk.strip() and not a_chunk.strip():
                continue
            chunk_diff = compute_diff(b_chunk, a_chunk) if b_chunk and a_chunk else ""
            chunk_prompt_text = CHANGE_SUMMARY_PROMPT.format(
                before=b_chunk, after=a_chunk, diff=chunk_diff,
            )
            chunk_prefill = build_prefill(
                encoding, chunk_prompt_text, "", reasoning_effort,
                IDENTITY_PROMPT, max_prompt_tokens,
            )
            if chunk_prefill is None:
                print(f"    chunk {ci+1}/{n_chunks}: SKIPPED (encoding failed)")
                continue
            n_tokens = len(chunk_prefill["prompt_token_ids"])
            print(f"    chunk {ci+1}/{n_chunks}: {len(chunk_prompt_text)} chars, {n_tokens} tokens")
            prompts.append(chunk_prefill)
            prompt_texts.append(chunk_prompt_text)
            prompt_map.append((file_index, chunk_count))
            chunk_count += 1

        if chunk_count > 0:
            item_chunk_counts[file_index] = chunk_count
            diffs[file_index] = diff
        else:
            print(f"    {filename}: all chunks failed, copying through")
            copy_through.append(file_index)

    n_with_both = len(item_chunk_counts)
    n_copy = len(copy_through)
    n_multi_chunk = sum(1 for c in item_chunk_counts.values() if c > 1)
    print(f"\nInitiatives with both summaries: {n_with_both}"
          f" ({n_multi_chunk} multi-chunk)")
    print(f"Initiatives to copy through (missing before or after): {n_copy}")
    print(f"Total prompts (pass 1): {len(prompts)}")

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

    # === Pass 1: summarize each chunk ===
    batch_dir_p1 = os.path.join(args.output, "_batches")
    os.makedirs(batch_dir_p1, exist_ok=True)

    summary_cache = {}
    all_failed = []
    summarized_chunks, failed_p1, stats_p1 = run_batch_inference(
        llm, sampling_params, encoding,
        prompts, prompt_texts, prompt_map,
        args.batch_size, batch_dir_p1, summary_cache,
        batch_num_start=0, label="[P1] ",
    )
    all_failed.extend(failed_p1)

    # Separate single-chunk finals from multi-chunk items needing combining
    final_summaries = {}  # file_index -> final change summary
    current_parts = {}    # file_index -> [chunk_summaries...] for combining

    for file_index, n_chunks in item_chunk_counts.items():
        chunks_dict = summarized_chunks.get(file_index, {})
        if n_chunks == 1:
            if 0 in chunks_dict:
                final_summaries[file_index] = chunks_dict[0]
        else:
            parts = []
            for ci in range(n_chunks):
                if ci in chunks_dict:
                    parts.append(chunks_dict[ci])
            if parts:
                if len(parts) == 1:
                    final_summaries[file_index] = parts[0]
                else:
                    current_parts[file_index] = parts

    print(f"\nPass 1 results: {len(final_summaries)} single-chunk items done, "
          f"{len(current_parts)} multi-chunk items need combining")

    # === Pass 2: Recursively combine chunk summaries ===
    combine_budget = args.combine_budget
    combine_level = 0
    batch_dir_p2 = os.path.join(args.output, "_batches_combine")
    os.makedirs(batch_dir_p2, exist_ok=True)
    batch_num_p2 = 0

    while current_parts:
        combine_level += 1

        p2_prompts = []
        p2_texts = []
        p2_prompt_map = []

        for file_index, parts in current_parts.items():
            chunk_groups = group_by_char_budget(parts, combine_budget)
            for gi, group in enumerate(chunk_groups):
                if len(group) == 1:
                    continue
                combined = "\n\n".join(group)
                prefill = build_prefill(
                    encoding, combined, COMBINE_PREFIX, reasoning_effort,
                    IDENTITY_PROMPT, args.max_model_len,
                )
                if prefill is None:
                    print(f"  WARNING: combine prefill failed for file_index={file_index} group={gi}")
                    continue
                p2_prompts.append(prefill)
                p2_texts.append(combined)
                p2_prompt_map.append((file_index, gi))

        print(f"\n--- Combine level {combine_level}: {len(p2_prompts)} prompts "
              f"from {len(current_parts)} items ---\n")

        p2_results = {}
        if p2_prompts:
            p2_results, failed_p2, stats_p2 = run_batch_inference(
                llm, sampling_params, encoding,
                p2_prompts, p2_texts, p2_prompt_map,
                args.batch_size, batch_dir_p2, summary_cache,
                batch_num_start=batch_num_p2, label=f"[C{combine_level}] ",
            )
            batch_num_p2 = stats_p2["batch_num"]
            all_failed.extend(failed_p2)

        next_parts = {}
        for file_index, parts in current_parts.items():
            chunk_groups = group_by_char_budget(parts, combine_budget)
            new_parts = []
            for gi, group in enumerate(chunk_groups):
                if len(group) == 1:
                    new_parts.append(group[0])
                elif file_index in p2_results and gi in p2_results[file_index]:
                    new_parts.append(p2_results[file_index][gi])

            if len(new_parts) == 1:
                final_summaries[file_index] = new_parts[0]
            elif len(new_parts) > 1:
                next_parts[file_index] = new_parts

        current_parts = next_parts

    # Write output files
    written = 0
    summaries_added = 0
    for file_index, filename in enumerate(pending_files):
        initiative = initiatives[file_index]

        if file_index in final_summaries:
            initiative["change_summary"] = final_summaries[file_index]
            if file_index in diffs:
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

    if all_failed:
        failed_file = os.path.join(args.output, "_failed.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, ensure_ascii=False, indent=2)
        print(f"FAILED: {len(all_failed)} prompts could not be parsed. Wrote {failed_file}")


if __name__ == "__main__":
    main()
