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
import datetime
import json
import os
import sys
import time

import torch
from openai_harmony import (
    Conversation,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)
from vllm import LLM, SamplingParams

from text_utils import split_into_chunks

DEFAULT_MODEL = "unsloth/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 2048
CHUNK_SIZE = 5000
INITIATIVE_BATCH_SIZE = 10

IDENTITY_PROMPT = (
    "You are a policy analyst who summarizes EU regulatory documents "
    "clearly and concisely."
)

DOCUMENT_PROMPT_PREFIX = (
    "The following is a section of a publication document from an EU policy initiative. "
    "Summarize it into exactly 5 paragraphs of exactly 5 sentences each. If any, preserve all points about nuclear energy, nuclear plants, or small modular reactors.\n\n"
)

FEEDBACK_ATTACHMENT_PROMPT_PREFIX = (
    "The following is a section of a feedback attachment submitted in response to an EU policy initiative. "
    "Summarize it into at most 5 paragraphs of at most 5 sentences each. If any, preserve all points about nuclear energy, nuclear plants, or small modular reactors.\n\n"
)

DOCUMENT_COMBINE_PREFIX = (
    "The following are summaries of consecutive sections of a publication document "
    "from an EU policy initiative. Combine them into a single summary of exactly "
    "5 paragraphs of exactly 5 sentences each. If any, preserve all points about nuclear energy, nuclear plants, or small modular reactors.\n\n"
)

FEEDBACK_COMBINE_PREFIX = (
    "The following are summaries of consecutive sections of a feedback attachment "
    "submitted in response to an EU policy initiative. Combine them into a single "
    "summary of at most 5 paragraphs of at most 5 sentences each. If any, preserve all points about nuclear energy, nuclear plants, or small modular reactors.\n\n"
)


def build_prefill(
    encoding: HarmonyEncoding,
    text: str,
    prompt_prefix: str,
    reasoning_effort: ReasoningEffort,
) -> dict:
    """Build a prompt_token_ids prefill dict for vLLM using openai_harmony."""
    user_prompt = prompt_prefix + text
    convo = Conversation.from_messages([
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_conversation_start_date(
                datetime.datetime.now().strftime('%Y-%m-%d %A')
            ).with_reasoning_effort(reasoning_effort).with_model_identity(IDENTITY_PROMPT)
        ),
        Message.from_role_and_content(Role.USER, user_prompt),
    ])
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    return {"prompt_token_ids": prefill_ids}


_EXTRACT_ERROR = object()  # sentinel for failed extractions


def extract_final_texts(outputs, encoding: HarmonyEncoding) -> list:
    """Extract the 'final' channel text from each vLLM output.

    Returns a list with one entry per output. Successful entries are strings.
    Failed entries are the _EXTRACT_ERROR sentinel.
    """
    results = []
    for i, output in enumerate(outputs):
        try:
            gen = output.outputs[0]
            output_tokens = gen.token_ids
            entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
            final_message = None
            for message in entries:
                if message.channel == "final":
                    final_message = message.content[0].text
            results.append(final_message if final_message else gen.text.strip())
        except Exception as e:
            print(f"  WARNING: failed to parse output {i}: {e}")
            results.append(_EXTRACT_ERROR)
    return results


def should_skip_text(text):
    """Return True if text should not be summarized."""
    if not text or not text.strip():
        return True
    if text.startswith("%PDF-"):
        return True
    return False


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
                chunks = split_into_chunks(text, chunk_size)
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
                        encoding, chunk, DOCUMENT_PROMPT_PREFIX, reasoning_effort
                    )
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
                chunks = split_into_chunks(text, chunk_size)
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
                        encoding, chunk, FEEDBACK_ATTACHMENT_PROMPT_PREFIX, reasoning_effort
                    )
                    n_tokens = len(prefill["prompt_token_ids"])
                    print(f"  {filename} fb={fb.get('id','?')} att={att.get('id','?')} "
                          f"chunk {ci+1}/{len(chunks)}: {len(chunk)} chars, {n_tokens} tokens")
                    prompts.append(prefill)
                    chunk_texts.append(chunk)
                    prompt_map.append((item_index, ci))
                item_index += 1

    return (prompts, chunk_texts, prompt_map, item_locations,
            item_chunk_counts, item_is_feedback)


def run_batch_inference(llm, sampling_params, encoding, prompts, prompt_texts,
                        prompt_map, batch_size, batch_dir, summary_cache,
                        batch_num_start=0, label=""):
    """Run batched inference with dedup, resume, and per-batch file output.

    Args:
        llm: vLLM LLM instance.
        sampling_params: vLLM SamplingParams.
        encoding: HarmonyEncoding for output parsing.
        prompts: list of prefill dicts.
        prompt_texts: list of raw text strings (for dedup).
        prompt_map: list of (item_key, chunk_index) tuples.
        batch_size: number of prompts per batch.
        batch_dir: directory for per-batch output files.
        summary_cache: dict of text -> summary (shared across calls for dedup).
        batch_num_start: starting batch file number.
        label: prefix for log messages (e.g. "[P1] ", "[P2] ").

    Returns:
        summarized_chunks: dict of item_key -> {chunk_index: summary_text}
        failed_prompts: list of failed prompt info dicts.
        stats: dict with batch_num, skipped_batches, dedup_total.
    """
    n_prompts = len(prompts)
    text_lens = [len(t) for t in prompt_texts]
    summarized_chunks = {}
    failed_prompts = []
    batch_num = batch_num_start
    skipped_batches = 0
    dedup_total = 0

    print(f"{label}Running inference on {n_prompts} prompts...")
    t_start = time.time()

    for batch_start in range(0, n_prompts, batch_size):
        batch_end = min(batch_start + batch_size, n_prompts)
        batch_file = os.path.join(batch_dir, f"batch_{batch_num:04d}.json")

        # Resume: if batch file exists, load from it
        if os.path.isfile(batch_file):
            with open(batch_file, encoding="utf-8") as f:
                cached_results = json.load(f)
            for entry in cached_results:
                item_key = entry["item_key"]
                chunk_idx = entry["chunk_index"]
                summarized_chunks.setdefault(item_key, {})[chunk_idx] = entry["summary"]
            # Populate summary_cache
            for j in range(batch_start, batch_end):
                item_key, chunk_idx = prompt_map[j]
                ct = prompt_texts[j]
                result = summarized_chunks.get(item_key, {}).get(chunk_idx)
                if result is not None and ct not in summary_cache:
                    summary_cache[ct] = result
            skipped_batches += 1
            print(f"  {label}Batch [{batch_start+1}-{batch_end}/{n_prompts}]: "
                  f"loaded from {batch_file} ({len(cached_results)} results)")
            batch_num += 1
            continue

        # Dedup: cross-batch cache + intra-batch
        infer_indices = []
        infer_prompts = []
        infer_seen = {}
        dedup_indices = []
        dedup_count = 0
        batch_results = []

        for j in range(batch_start, batch_end):
            ct = prompt_texts[j]
            if ct in summary_cache:
                dedup_indices.append(j)
                dedup_count += 1
            elif ct in infer_seen:
                dedup_indices.append(j)
                dedup_count += 1
            else:
                infer_seen[ct] = len(infer_indices)
                infer_indices.append(j)
                infer_prompts.append(prompts[j])

        dedup_total += dedup_count
        batch_sizes = [text_lens[j] for j in range(batch_start, batch_end)]

        # Log batch info
        print(f"  {label}Batch [{batch_start+1}-{batch_end}/{n_prompts}]"
              f"{f' ({dedup_count} deduped, {len(infer_prompts)} to infer)' if dedup_count else ''}:")
        for j in range(batch_start, batch_end):
            item_key, chunk_idx = prompt_map[j]
            snippet = prompt_texts[j][:80].replace("\n", " ").strip()
            dedup_tag = " [dedup]" if j in set(dedup_indices) else ""
            print(f"    item={item_key} chunk={chunk_idx} "
                  f"({text_lens[j]} chars) \"{snippet}...\"{dedup_tag}")

        # Run inference
        if infer_prompts:
            t0 = time.time()
            batch_outputs = llm.generate(infer_prompts, sampling_params)
            elapsed = time.time() - t0

            batch_summaries = extract_final_texts(batch_outputs, encoding)
            batch_errors = 0
            for k, summary in enumerate(batch_summaries):
                j = infer_indices[k]
                item_key, chunk_idx = prompt_map[j]
                if summary is _EXTRACT_ERROR:
                    batch_errors += 1
                    failed_prompts.append({
                        "item_key": item_key,
                        "chunk_index": chunk_idx,
                        "text_length": text_lens[j],
                    })
                    continue
                summarized_chunks.setdefault(item_key, {})[chunk_idx] = summary
                summary_cache[prompt_texts[j]] = summary
                batch_results.append({
                    "item_key": item_key,
                    "chunk_index": chunk_idx,
                    "summary": summary,
                })
        else:
            elapsed = 0.0
            batch_errors = 0

        # Resolve deduped prompts
        for j in dedup_indices:
            ct = prompt_texts[j]
            summary = summary_cache.get(ct)
            item_key, chunk_idx = prompt_map[j]
            if summary is not None:
                summarized_chunks.setdefault(item_key, {})[chunk_idx] = summary
                batch_results.append({
                    "item_key": item_key,
                    "chunk_index": chunk_idx,
                    "summary": summary,
                })
            else:
                failed_prompts.append({
                    "item_key": item_key,
                    "chunk_index": chunk_idx,
                    "text_length": text_lens[j],
                })

        # Write per-batch file
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)

        error_note = f", {batch_errors} FAILED" if batch_errors else ""
        dedup_note = f", {dedup_count} deduped" if dedup_count else ""
        print(f"  {label}[{batch_end}/{n_prompts}] batch of {batch_end - batch_start} done in {elapsed:.1f}s "
              f"(input: {min(batch_sizes)}-{max(batch_sizes)} chars{dedup_note}{error_note})")
        print(f"  Wrote {batch_file}")
        batch_num += 1

    total_elapsed = time.time() - t_start
    if n_prompts > 0:
        print(f"{label}Inference done in {total_elapsed:.1f}s ({total_elapsed / n_prompts:.2f}s per prompt)")
    if skipped_batches:
        print(f"  ({skipped_batches} batches loaded from existing files)")
    if dedup_total:
        print(f"  ({dedup_total} prompts skipped via deduplication)")

    return summarized_chunks, failed_prompts, {
        "batch_num": batch_num,
        "skipped_batches": skipped_batches,
        "dedup_total": dedup_total,
    }


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
        "--max-model-len", type=int, default=None,
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
        "--batch-size", type=int, default=32,
        help="Number of prompts per inference batch (default: 32).",
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

    # Initialize vLLM
    tp_size = torch.cuda.device_count()
    print(f"CUDA device count: {tp_size}")
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

    # Prepare output and batch directories
    os.makedirs(args.output, exist_ok=True)
    batch_dir_p1 = os.path.join(args.output, "_batches_pass1")
    batch_dir_p2 = os.path.join(args.output, "_batches_pass2")
    os.makedirs(batch_dir_p1, exist_ok=True)
    os.makedirs(batch_dir_p2, exist_ok=True)

    # Shared state across initiative groups
    summary_cache = {}   # text -> summary (persists for dedup across groups)
    batch_num_p1 = 0
    batch_num_p2 = 0
    grand_items = 0
    grand_prompts = 0
    grand_summaries = 0
    grand_written = 0
    all_failed = []

    t_total_start = time.time()
    n_groups = (len(initiative_files) + args.initiative_batch_size - 1) // args.initiative_batch_size

    for group_idx, group_start in enumerate(range(0, len(initiative_files), args.initiative_batch_size)):
        group_files = initiative_files[group_start:group_start + args.initiative_batch_size]
        group_end = group_start + len(group_files)

        print(f"\n{'='*60}")
        print(f"Group {group_idx+1}/{n_groups}: initiatives {group_start+1}-{group_end}/{len(initiative_files)} "
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
            batch_num_start=batch_num_p1, label="[P1] ",
        )
        batch_num_p1 = stats_p1["batch_num"]
        all_failed.extend(failed_p1)

        # === Assemble final summaries ===
        final_summaries = {}
        pass2_items = []

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
                    pass2_items.append((item_idx, "\n\n".join(parts)))

        print(f"\nPass 1 results: {len(final_summaries)} single-chunk items done, "
              f"{len(pass2_items)} multi-chunk items need pass 2")

        # === Pass 2: Combine chunk summaries for multi-chunk items ===
        if pass2_items:
            p2_prompts = []
            p2_texts = []
            p2_prompt_map = []

            for p2_idx, (item_idx, combined_text) in enumerate(pass2_items):
                is_fb = item_is_feedback[item_idx]
                prefix = FEEDBACK_COMBINE_PREFIX if is_fb else DOCUMENT_COMBINE_PREFIX
                p2_prompts.append(build_prefill(encoding, combined_text, prefix, reasoning_effort))
                p2_texts.append(combined_text)
                p2_prompt_map.append((item_idx, 0))

            print(f"\n--- Pass 2: Combine summaries for {len(pass2_items)} multi-chunk items ---\n")

            p2_results, failed_p2, stats_p2 = run_batch_inference(
                llm, sampling_params, encoding,
                p2_prompts, p2_texts, p2_prompt_map,
                args.batch_size, batch_dir_p2, summary_cache,
                batch_num_start=batch_num_p2, label="[P2] ",
            )
            batch_num_p2 = stats_p2["batch_num"]
            all_failed.extend(failed_p2)

            for item_idx in p2_results:
                if 0 in p2_results[item_idx]:
                    final_summaries[item_idx] = p2_results[item_idx][0]

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
    print(f"Initiative files: {grand_written}/{len(initiative_files)} written to {args.output}/")
    print(f"Items summarized: {grand_summaries}/{grand_items}")
    print(f"Batch files: pass1={batch_num_p1}, pass2={batch_num_p2}")
    print(f"Dedup cache size: {len(summary_cache)}")

    if all_failed:
        failed_file = os.path.join(args.output, "_failed.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, ensure_ascii=False, indent=2)
        print(f"FAILED: {len(all_failed)} prompts could not be parsed. Wrote {failed_file}")


if __name__ == "__main__":
    main()
