"""Summarize EU initiative documents and feedback attachments using vLLM batch inference.

Takes the output directory from initiative_stats.py -o, which contains per-initiative
JSON files with documents_before_feedback, documents_after_feedback, and middle_feedback.
Adds a 'summary' field to each document and feedback attachment.

Long texts are split into chunks at sentence boundaries, each chunk is summarized
independently (pass 1), then multi-chunk summaries are combined into a single
final summary via a second inference pass (pass 2).

Initiatives are processed in groups (--initiative-batch-size) so the full dataset
does not need to be loaded into memory at once.

Usage:
    python3 src/summarize_documents.py before_after_analysis_v2/ -o summaries_output/
    python3 src/summarize_documents.py before_after_analysis_v2/ -o summaries_output/ --batch-size 16
"""

import argparse
import json
import os
import sys
import time

import torch
from openai_harmony import (
    HarmonyEncoding,
    HarmonyEncodingName,
    ReasoningEffort,
    load_harmony_encoding,
)
from vllm import LLM, SamplingParams

from inference_utils import build_prefill, extract_final_texts, run_batch_inference
from text_utils import should_skip_text, split_into_chunks

DEFAULT_MODEL = "unsloth/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 32768 * 4
CHUNK_SIZE = 16384
INITIATIVE_BATCH_SIZE = 128

IDENTITY_PROMPT = (
    "You are a policy analyst who summarizes EU regulatory documents "
    "clearly and concisely."
)

DOCUMENT_PROMPT_PREFIX = (
    "The following is a section of a publication document from an EU policy initiative. "
    "Summarize it into a text up to 10 paragraphs. Be as specific and detailed as possible. If any, preserve all points about nuclear energy, nuclear plants, or small modular reactors. Do not generate any mete commentary (for example stating that there are no nuclear-related points).\n\n"
)

FEEDBACK_ATTACHMENT_PROMPT_PREFIX = (
    "The following is a section of a feedback attachment submitted in response to an EU policy initiative. "
    "Summarize it into a text up to 10 paragraphs. Be as specific and detailed as possible. If any, preserve all points about nuclear energy, nuclear plants, or small modular reactors. Do not generate any mete commentary (for example stating that there are no nuclear-related points).\n\n"
)

DOCUMENT_COMBINE_PREFIX = (
    "The following are summaries of consecutive sections of a publication document "
    "from an EU policy initiative. Combine them into a single summary up to 10 paragraphs. Be as specific and detailed as possible. If any, preserve all points about nuclear energy, nuclear plants, or small modular reactors. Do not generate any mete commentary (for example stating that there are no nuclear-related points).\n\n"
)

FEEDBACK_COMBINE_PREFIX = (
    "The following are summaries of consecutive sections of a feedback attachment "
    "submitted in response to an EU policy initiative. Combine them into a single summary up to 10 paragraphs. Be as specific and detailed as possible. If any, preserve all points about nuclear energy, nuclear plants, or small modular reactors. Do not generate any mete commentary (for example stating that there are no nuclear-related points).\n\n"
)




def collect_prompts(input_dir, filenames, encoding, reasoning_effort, chunk_size):
    """Collect chunk-level prompts for a subset of initiative files.

    Returns:
        prompts: list of {"prompt_token_ids": [...]}
        chunk_texts: list of raw chunk text strings (for dedup)
        prompt_map: list of (item_index, chunk_index) tuples
        item_locations: list of location dicts (indexed by item_index)
        item_chunk_counts: dict of item_index -> number of chunks
        item_is_feedback: dict of item_index -> bool
    """
    prompts = []
    chunk_texts = []
    prompt_map = []
    item_locations = []
    item_chunk_counts = {}
    item_is_feedback = {}

    item_index = 0
    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            initiative = json.load(f)

        # documents_before_feedback and documents_after_feedback
        for list_name in ("documents_before_feedback", "documents_after_feedback"):
            for doc_idx, doc in enumerate(initiative.get(list_name, [])):
                text = doc.get("extracted_text", "")
                if should_skip_text(text):
                    continue
                text = text.strip()
                init_id = filename.replace(".json", "")
                label = f"init={init_id} {list_name}[{doc_idx}] {doc.get('filename', '?')}"
                chunks = split_into_chunks(text, chunk_size, label=label)
                item_chunk_counts[item_index] = len(chunks)
                item_is_feedback[item_index] = False
                item_locations.append({
                    "file": filename,
                    "list": list_name,
                    "doc_idx": doc_idx,
                    "filename": doc.get("filename", "?"),
                })
                for ci, chunk in enumerate(chunks):
                    prefill = build_prefill(
                        encoding, chunk, DOCUMENT_PROMPT_PREFIX, reasoning_effort,
                        IDENTITY_PROMPT,
                    )
                    if prefill is None:
                        print(f"  {filename} {list_name}[{doc_idx}] chunk {ci+1}/{len(chunks)}: "
                              f"{len(chunk)} chars — SKIPPED (encoding failed)")
                        continue
                    n_tokens = len(prefill["prompt_token_ids"])
                    print(f"  {filename} {list_name}[{doc_idx}] chunk {ci+1}/{len(chunks)}: "
                          f"{len(chunk)} chars, {n_tokens} tokens")
                    prompts.append(prefill)
                    chunk_texts.append(chunk)
                    prompt_map.append((item_index, ci))
                item_index += 1

        # middle_feedback -> attachments
        for fb_idx, fb in enumerate(initiative.get("middle_feedback", [])):
            for att_idx, att in enumerate(fb.get("attachments", [])):
                text = att.get("extracted_text", "")
                if should_skip_text(text):
                    continue
                text = text.strip()
                label = f"init={filename.replace('.json', '')} fb={fb.get('id', '?')} att={att.get('id', '?')}"
                chunks = split_into_chunks(text, chunk_size, label=label)
                item_chunk_counts[item_index] = len(chunks)
                item_is_feedback[item_index] = True
                item_locations.append({
                    "file": filename,
                    "list": "middle_feedback",
                    "fb_idx": fb_idx,
                    "att_idx": att_idx,
                    "feedback_id": fb.get("id", "?"),
                    "attachment_id": att.get("id", "?"),
                    "filename": att.get("filename", "?"),
                })
                for ci, chunk in enumerate(chunks):
                    prefill = build_prefill(
                        encoding, chunk, FEEDBACK_ATTACHMENT_PROMPT_PREFIX, reasoning_effort,
                        IDENTITY_PROMPT,
                    )
                    if prefill is None:
                        print(f"  {filename} fb={fb.get('id','?')} att={att.get('id','?')} "
                              f"chunk {ci+1}/{len(chunks)}: {len(chunk)} chars — SKIPPED (encoding failed)")
                        continue
                    n_tokens = len(prefill["prompt_token_ids"])
                    print(f"  {filename} fb={fb.get('id','?')} att={att.get('id','?')} "
                          f"chunk {ci+1}/{len(chunks)}: {len(chunk)} chars, {n_tokens} tokens")
                    prompts.append(prefill)
                    chunk_texts.append(chunk)
                    prompt_map.append((item_index, ci))
                item_index += 1

    return (prompts, chunk_texts, prompt_map, item_locations,
            item_chunk_counts, item_is_feedback)


def write_output(input_dir, output_dir, item_locations, final_summaries, filenames):
    """Load initiative JSONs, add summaries, write to output directory."""
    # Group summaries by file
    by_file = {}
    for item_idx, summary in final_summaries.items():
        loc = item_locations[item_idx]
        by_file.setdefault(loc["file"], []).append((loc, summary))

    written = 0
    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            initiative = json.load(f)

        for loc, summary in by_file.get(filename, []):
            list_name = loc["list"]
            if list_name in ("documents_before_feedback", "documents_after_feedback"):
                initiative[list_name][loc["doc_idx"]]["summary"] = summary
            elif list_name == "middle_feedback":
                initiative["middle_feedback"][loc["fb_idx"]]["attachments"][loc["att_idx"]]["summary"] = summary

        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(initiative, f, ensure_ascii=False, indent=2)
        written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Summarize EU initiative documents and feedback attachments using vLLM."
    )
    parser.add_argument(
        "input_dir",
        help="Directory of per-initiative JSON files (output of initiative_stats.py -o)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for initiative JSONs with summaries added.",
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
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Max chars per chunk before splitting (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048,
        help="Number of prompts per inference batch (default: 2048).",
    )
    parser.add_argument(
        "--max-combine-chunks", type=int, default=4,
        help="Max summaries to combine per inference call in pass 2 (default: 4).",
    )
    parser.add_argument(
        "--initiative-batch-size", type=int, default=INITIATIVE_BATCH_SIZE,
        help=f"Number of initiative files to load at a time (default: {INITIATIVE_BATCH_SIZE}).",
    )
    args = parser.parse_args()

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

    # Initialize vLLM (deferred until we know there is work)
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

    # Shared state across initiative groups
    summary_cache = {}   # text -> summary (persists for dedup across groups)
    grand_items = 0
    grand_prompts = 0
    grand_summaries = 0
    grand_written = 0
    all_failed = []

    t_total_start = time.time()

    n_groups = (len(pending_files) + args.initiative_batch_size - 1) // args.initiative_batch_size

    for group_idx in range(n_groups):
        group_start = group_idx * args.initiative_batch_size
        group_end = min(group_start + args.initiative_batch_size, len(pending_files))
        group_files = pending_files[group_start:group_end]

        # Per-group batch directories (local batch numbering, groups are independent)
        batch_dir_p1 = os.path.join(args.output, "_batches_pass1", f"group_{group_idx:04d}")
        batch_dir_p2 = os.path.join(args.output, "_batches_pass2", f"group_{group_idx:04d}")
        os.makedirs(batch_dir_p1, exist_ok=True)
        os.makedirs(batch_dir_p2, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Group {group_idx+1}/{n_groups}: "
              f"files {group_start+1}-{group_end}/{len(pending_files)} "
              f"({len(group_files)} files)")
        print(f"{'='*60}\n")

        # Collect prompts for this group
        (prompts, chunk_texts, prompt_map, item_locations,
         item_chunk_counts, item_is_feedback) = collect_prompts(
            args.input_dir, group_files, encoding, reasoning_effort, args.chunk_size
        )

        n_prompts = len(prompts)
        n_items = len(item_locations)
        n_docs = sum(1 for v in item_is_feedback.values() if not v)
        n_fb_att = sum(1 for v in item_is_feedback.values() if v)
        n_chunked = sum(1 for c in item_chunk_counts.values() if c > 1)
        grand_items += n_items
        grand_prompts += n_prompts

        print(f"Items: {n_items} ({n_docs} documents, {n_fb_att} feedback attachments)")
        print(f"Prompts: {n_prompts} ({n_chunked} items split into multiple chunks)")

        if n_prompts == 0:
            write_output(args.input_dir, args.output, item_locations, {}, group_files)
            grand_written += len(group_files)
            print(f"No text to summarize, wrote {len(group_files)} files unchanged.")
            continue

        text_lens = [len(t) for t in chunk_texts]
        print(f"Chunk sizes: min={min(text_lens)}, max={max(text_lens)}, "
              f"avg={sum(text_lens) // len(text_lens)}, total={sum(text_lens)}")

        # === Pass 1: Summarize each chunk ===
        print(f"\n--- Pass 1: Summarize {n_prompts} chunks from {n_items} items ---\n")

        summarized_chunks, failed_p1, stats_p1 = run_batch_inference(
            llm, sampling_params, encoding,
            prompts, chunk_texts, prompt_map,
            args.batch_size, batch_dir_p1, summary_cache,
            batch_num_start=0, label="[P1] ",
        )
        all_failed.extend(failed_p1)

        # === Assemble final summaries ===
        final_summaries = {}
        current_parts = {}  # item_idx -> list of summaries at current level

        for item_idx in range(n_items):
            n_chunks = item_chunk_counts[item_idx]
            chunks_dict = summarized_chunks.get(item_idx, {})

            if n_chunks == 1:
                if 0 in chunks_dict:
                    final_summaries[item_idx] = chunks_dict[0]
            else:
                parts = []
                complete = True
                for ci in range(n_chunks):
                    if ci in chunks_dict:
                        parts.append(chunks_dict[ci])
                    else:
                        complete = False
                        break
                if complete and parts:
                    current_parts[item_idx] = parts

        print(f"\nPass 1 results: {len(final_summaries)} single-chunk items done, "
              f"{len(current_parts)} multi-chunk items need combining")

        # === Pass 2: Recursively combine chunk summaries ===
        max_combine = args.max_combine_chunks
        combine_level = 0
        batch_num_p2 = 0

        while current_parts:
            combine_level += 1

            # Build prompts: for each item, group summaries into groups of <= max_combine
            p2_prompts = []
            p2_texts = []
            p2_prompt_map = []

            for item_idx, parts in current_parts.items():
                chunk_groups = [parts[i:i + max_combine] for i in range(0, len(parts), max_combine)]
                for gi, group in enumerate(chunk_groups):
                    if len(group) == 1:
                        # Single-element group: pass through without inference
                        continue
                    combined = "\n\n".join(group)
                    is_fb = item_is_feedback[item_idx]
                    prefix = FEEDBACK_COMBINE_PREFIX if is_fb else DOCUMENT_COMBINE_PREFIX
                    p2_prompts.append(build_prefill(encoding, combined, prefix, reasoning_effort, IDENTITY_PROMPT))
                    p2_texts.append(combined)
                    p2_prompt_map.append((item_idx, gi))

            print(f"\n--- Combine level {combine_level}: {len(p2_prompts)} prompts "
                  f"from {len(current_parts)} items ---\n")

            # Run inference (if any prompts need it)
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

            # Collect results into next level
            next_parts = {}
            for item_idx, parts in current_parts.items():
                chunk_groups = [parts[i:i + max_combine] for i in range(0, len(parts), max_combine)]
                new_parts = []
                for gi, group in enumerate(chunk_groups):
                    if len(group) == 1:
                        new_parts.append(group[0])  # pass-through
                    elif item_idx in p2_results and gi in p2_results[item_idx]:
                        new_parts.append(p2_results[item_idx][gi])
                    # else: failed, skip this group

                if len(new_parts) == 1:
                    final_summaries[item_idx] = new_parts[0]
                elif len(new_parts) > 1:
                    next_parts[item_idx] = new_parts
                # else: all groups failed, item has no summary

            current_parts = next_parts

        # Write output for this group
        written = write_output(args.input_dir, args.output, item_locations,
                               final_summaries, group_files)
        grand_written += written
        grand_summaries += len(final_summaries)
        print(f"\nGroup done: {len(final_summaries)}/{n_items} summaries, wrote {written} files")

    # Grand totals
    total_elapsed = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"Initiative files: {grand_written + skipped_files}/{len(initiative_files)} "
          f"({grand_written} written, {skipped_files} already existed)")
    print(f"Items summarized: {grand_summaries}/{grand_items}")
    print(f"Dedup cache size: {len(summary_cache)}")

    if all_failed:
        failed_file = os.path.join(args.output, "_failed.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, ensure_ascii=False, indent=2)
        print(f"FAILED: {len(all_failed)} prompts could not be parsed. Wrote {failed_file}")


if __name__ == "__main__":
    main()
