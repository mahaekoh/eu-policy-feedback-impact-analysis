"""Shared vLLM batch inference helpers.

Provides build_prefill(), extract_final_texts(), and run_batch_inference() used by
summarize_documents.py, summarize_clusters.py, and summarize_changes.py.
"""

import datetime
import json
import os
import time

from openai_harmony import (
    Conversation,
    HarmonyEncoding,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
)


def build_prefill(
    encoding: HarmonyEncoding,
    text: str,
    prompt_prefix: str,
    reasoning_effort: ReasoningEffort,
    identity_prompt: str,
) -> dict | None:
    """Build a prompt_token_ids prefill dict for vLLM using openai_harmony."""
    user_prompt = prompt_prefix + text
    convo = Conversation.from_messages([
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_conversation_start_date(
                datetime.datetime.now().strftime('%Y-%m-%d %A')
            ).with_reasoning_effort(reasoning_effort).with_model_identity(identity_prompt)
        ),
        Message.from_role_and_content(Role.USER, user_prompt),
    ])
    try:
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    except BaseException as e:
        print(f"ERROR in render_conversation_for_completion: {type(e).__name__}: {e}")
        print(f"  prompt_prefix: {prompt_prefix[:100]!r}")
        print(f"  text length: {len(text)} chars")
        print(f"  text repr (first 500): {text[:500]!r}")
        print(f"  text repr (last 500):  {text[-500:]!r}")
        return None
    return {"prompt_token_ids": prefill_ids}


def extract_final_texts(outputs, encoding: HarmonyEncoding) -> list:
    """Extract the 'final' channel text from each vLLM output.

    Returns a list with one entry per output. Successful entries are strings.
    Failed entries are None.
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
            if final_message is None:
                print(f"  WARNING: no 'final' channel in output {i}")
            results.append(final_message)
        except Exception as e:
            print(f"  WARNING: failed to parse output {i}: {e}")
            results.append(None)
    return results


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
                if summary is None:
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
