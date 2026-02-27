"""Translate non-English feedback attachment texts to English using vLLM batch inference.

Takes the JSON output from find_non_english_feedback_attachments.py, translates
each item's extracted_text using unsloth/gpt-oss-120b, and writes a copy of
the JSON with extracted_text_translated added.

Long documents (>16384 chars) are split at sentence boundaries into chunks,
translated separately, and reassembled.

Usage:
    python3 src/translate_attachments.py non_english_attachments.json -o non_english_attachments_translated.json
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
MAX_OUTPUT_TOKENS = 32768 * 4
CHUNK_SIZE = 16384


IDENTITY_PROMPT = "You are a professional translator. You translate text to English accurately and faithfully."

USER_PROMPT_PREFIX = '''The following is a piece of feedback for an EU policy proposal. If it's in English, simply generate "NO TRANSLATION NEEDED." If it's not in English, generate an English translation of the text. No commentary at the beginning or end.\n\n'''


def build_prefill(encoding: HarmonyEncoding, text: str, reasoning_effort: ReasoningEffort):
    """Build a prompt_token_ids prefill dict for vLLM using openai_harmony.

    Returns None if encoding fails (e.g. bad text causing stack overflow).
    """
    user_prompt = USER_PROMPT_PREFIX + text
    convo = Conversation.from_messages([
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_conversation_start_date(
                datetime.datetime.now().strftime('%Y-%m-%d %A')
            ).with_reasoning_effort(reasoning_effort).with_model_identity(IDENTITY_PROMPT)
        ),
        Message.from_role_and_content(Role.USER, user_prompt),
    ])
    try:
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    except BaseException as e:
        print(f"ERROR in render_conversation_for_completion: {type(e).__name__}: {e}")
        return None
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
        except BaseException as e:
            print(f"  WARNING: failed to parse output {i}: {type(e).__name__}: {e}")
            results.append(_EXTRACT_ERROR)
    return results



def main():
    parser = argparse.ArgumentParser(
        description="Translate non-English feedback attachments using vLLM."
    )
    parser.add_argument(
        "input", help="Path to JSON from find_non_english_feedback_attachments.py"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output path for JSON with translations.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_OUTPUT_TOKENS,
        help=f"Max output tokens per translation (default: {MAX_OUTPUT_TOKENS}).",
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
        "--batch-size", type=int, default=4096,
        help="Number of prompts per inference batch (default: 4096).",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    # Initialize openai_harmony encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    reasoning_effort = ReasoningEffort.MEDIUM

    # Build chunk index (no tokenization yet — that happens per-batch)
    chunk_texts = []    # raw chunk texts
    prompt_map = []     # list of (record_index, chunk_index)
    record_chunk_counts = {}  # record_index -> number of chunks

    for i, rec in enumerate(records):
        text = rec.get("extracted_text", "")
        if not text or not text.strip() or text.startswith("%PDF-"):
            continue
        chunks = split_into_chunks(text.strip(), args.chunk_size)
        record_chunk_counts[i] = len(chunks)
        for ci, chunk in enumerate(chunks):
            chunk_texts.append(chunk)
            prompt_map.append((i, ci))

    total = len(records)
    n_records_with_text = len(record_chunk_counts)
    n_entries = len(chunk_texts)
    n_chunked = sum(1 for c in record_chunk_counts.values() if c > 1)
    print(f"Records: {total} total, {n_records_with_text} with text, {total - n_records_with_text} without text")
    chunk_lens = [len(ct) for ct in chunk_texts]
    print(f"Chunks: {n_entries} ({n_chunked} records were split into multiple chunks)")
    if chunk_lens:
        print(f"Chunk sizes: min={min(chunk_lens)}, max={max(chunk_lens)}, "
              f"avg={sum(chunk_lens) // len(chunk_lens)}, total={sum(chunk_lens)}")

    # Log the largest chunks
    if chunk_lens:
        top_n = min(10, n_entries)
        indices_by_size = sorted(range(n_entries), key=lambda i: chunk_lens[i], reverse=True)
        print(f"Top {top_n} largest chunks:")
        for rank, pi in enumerate(indices_by_size[:top_n]):
            rec_idx, chunk_idx = prompt_map[pi]
            rec = records[rec_idx]
            print(f"  {rank+1}. {chunk_lens[pi]} chars — "
                  f"initiative {rec.get('initiative_id','?')}, "
                  f"fb {rec.get('feedback_id','?')}, "
                  f"chunk {chunk_idx}/{record_chunk_counts[rec_idx]}, "
                  f"{rec.get('filename','?')}")

    if n_entries == 0:
        print("Nothing to translate.")
        for rec in records:
            rec["extracted_text_translated"] = None
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.output}")
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

    # Build lookup from (rec_idx, chunk_idx) -> original chunk text
    original_chunks = {}
    for j, (rec_idx, chunk_idx) in enumerate(prompt_map):
        original_chunks[(rec_idx, chunk_idx)] = chunk_texts[j]

    # Prepare output directory for per-batch files
    output_base, output_ext = os.path.splitext(args.output)
    batch_dir = output_base + "_batches"
    os.makedirs(batch_dir, exist_ok=True)

    # Run batch inference, writing a file after each batch
    # translation_cache: chunk_text -> translation (dedup identical texts)
    translation_cache = {}
    print(f"Running inference on {n_entries} chunks...")
    t_start = time.time()
    translated_chunks = {}  # record_index -> {chunk_index: text}
    failed_prompts = []     # prompts that failed extraction, for retry
    batch_num = 0
    skipped_batches = 0
    dedup_total = 0

    for batch_start in range(0, n_entries, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n_entries)
        batch_file = os.path.join(batch_dir, f"batch_{batch_num:04d}{output_ext}")

        # If batch file already exists, load results from it instead of re-running
        if os.path.isfile(batch_file):
            with open(batch_file, encoding="utf-8") as f:
                cached_results = json.load(f)
            for entry in cached_results:
                rec_idx = entry["record_index"]
                chunk_idx = entry["chunk_index"]
                translated_chunks.setdefault(rec_idx, {})[chunk_idx] = entry["translation"]
            # Populate translation cache from loaded batch
            for j in range(batch_start, batch_end):
                rec_idx, chunk_idx = prompt_map[j]
                ct = chunk_texts[j]
                result = translated_chunks.get(rec_idx, {}).get(chunk_idx)
                if result is not None and ct not in translation_cache:
                    translation_cache[ct] = result
            skipped_batches += 1
            print(f"  Batch [{batch_start+1}-{batch_end}/{n_entries}]: loaded from {batch_file} ({len(cached_results)} results)")
            batch_num += 1
            continue

        # Tokenize this batch and separate into inference vs cached/intra-batch duplicates
        infer_indices = []   # prompt indices that need inference
        infer_prompts = []
        infer_seen = {}      # chunk_text -> index in infer_indices (first occurrence)
        dedup_indices = []   # prompt indices resolved via cache or intra-batch dedup
        dedup_count = 0
        batch_results = []

        for j in range(batch_start, batch_end):
            ct = chunk_texts[j]
            if ct in translation_cache:
                # Reuse from cross-batch cache
                dedup_indices.append(j)
                dedup_count += 1
            elif ct in infer_seen:
                # Duplicate within this batch — will be resolved after inference
                dedup_indices.append(j)
                dedup_count += 1
            else:
                prefill = build_prefill(encoding, ct, reasoning_effort)
                if prefill is None:
                    rec_idx, chunk_idx = prompt_map[j]
                    rec = records[rec_idx]
                    print(f"  WARNING: skipping chunk {chunk_idx}/{record_chunk_counts[rec_idx]} "
                          f"for record {rec_idx} "
                          f"(initiative {rec.get('initiative_id','?')}, "
                          f"fb {rec.get('feedback_id','?')}, "
                          f"att {rec.get('attachment_id','?')}): encoding failed")
                    continue
                infer_seen[ct] = len(infer_indices)
                infer_indices.append(j)
                infer_prompts.append(prefill)

        dedup_total += dedup_count
        dedup_set = set(dedup_indices)
        batch_sizes = [chunk_lens[i] for i in range(batch_start, batch_end)]

        # Log IDs for each prompt in the batch
        print(f"  Batch [{batch_start+1}-{batch_end}/{n_entries}]"
              f"{f' ({dedup_count} deduped, {len(infer_prompts)} to infer)' if dedup_count else ''}:")
        for j in range(batch_start, batch_end):
            rec_idx, chunk_idx = prompt_map[j]
            rec = records[rec_idx]
            size = chunk_lens[j]
            snippet = chunk_texts[j][:80].replace("\n", " ").strip() if chunk_texts[j] else ""
            label = (f"    init={rec.get('initiative_id','?')}"
                     f" fb={rec.get('feedback_id','?')}"
                     f" att={rec.get('attachment_id','?')}")
            if record_chunk_counts[rec_idx] > 1:
                label += f" chunk={chunk_idx}/{record_chunk_counts[rec_idx]}"
            dedup_tag = " [dedup]" if j in dedup_set else ""
            print(f"{label} ({size} chars) \"{snippet}...\"{dedup_tag}")

        if infer_prompts:
            t0 = time.time()
            try:
                batch_outputs = llm.generate(infer_prompts, sampling_params)
            except BaseException as e:
                elapsed = time.time() - t0
                print(f"  BATCH FAILED ({type(e).__name__}: {e}) — marking all {len(infer_prompts)} prompts as failed")
                batch_errors = len(infer_prompts)
                for k in range(len(infer_prompts)):
                    j = infer_indices[k]
                    rec_idx, chunk_idx = prompt_map[j]
                    failed_prompts.append({
                        "record_index": rec_idx,
                        "chunk_index": chunk_idx,
                        "initiative_id": records[rec_idx].get("initiative_id"),
                        "feedback_id": records[rec_idx].get("feedback_id"),
                        "attachment_id": records[rec_idx].get("attachment_id"),
                        "chunk_text": chunk_texts[j],
                        "batch_error": f"{type(e).__name__}: {e}",
                    })
                batch_outputs = None

            if batch_outputs is not None:
                elapsed = time.time() - t0
                batch_translations = extract_final_texts(batch_outputs, encoding)
                batch_errors = 0
                for k, translation in enumerate(batch_translations):
                    j = infer_indices[k]
                    rec_idx, chunk_idx = prompt_map[j]
                    if translation is _EXTRACT_ERROR:
                        batch_errors += 1
                        failed_prompts.append({
                            "record_index": rec_idx,
                            "chunk_index": chunk_idx,
                            "initiative_id": records[rec_idx].get("initiative_id"),
                            "feedback_id": records[rec_idx].get("feedback_id"),
                            "attachment_id": records[rec_idx].get("attachment_id"),
                            "chunk_text": chunk_texts[j],
                        })
                        continue
                    translated_chunks.setdefault(rec_idx, {})[chunk_idx] = translation
                    translation_cache[chunk_texts[j]] = translation
                    batch_results.append({
                        "record_index": rec_idx,
                        "chunk_index": chunk_idx,
                        "initiative_id": records[rec_idx].get("initiative_id"),
                        "publication_id": records[rec_idx].get("publication_id"),
                        "feedback_id": records[rec_idx].get("feedback_id"),
                        "attachment_id": records[rec_idx].get("attachment_id"),
                        "translation": translation,
                    })
        else:
            elapsed = 0.0
            batch_errors = 0

        # Resolve deduped prompts (cross-batch cache hits + intra-batch duplicates)
        for j in dedup_indices:
            ct = chunk_texts[j]
            translation = translation_cache.get(ct)
            rec_idx, chunk_idx = prompt_map[j]
            if translation is not None:
                translated_chunks.setdefault(rec_idx, {})[chunk_idx] = translation
                batch_results.append({
                    "record_index": rec_idx,
                    "chunk_index": chunk_idx,
                    "initiative_id": records[rec_idx].get("initiative_id"),
                    "publication_id": records[rec_idx].get("publication_id"),
                    "feedback_id": records[rec_idx].get("feedback_id"),
                    "attachment_id": records[rec_idx].get("attachment_id"),
                    "translation": translation,
                })
            else:
                # Intra-batch duplicate whose first occurrence failed extraction
                failed_prompts.append({
                    "record_index": rec_idx,
                    "chunk_index": chunk_idx,
                    "initiative_id": records[rec_idx].get("initiative_id"),
                    "publication_id": records[rec_idx].get("publication_id"),
                    "feedback_id": records[rec_idx].get("feedback_id"),
                    "attachment_id": records[rec_idx].get("attachment_id"),
                    "chunk_text": ct,
                })

        # Write per-batch file
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)

        rec_idx_sample, _ = prompt_map[batch_start]
        rec_sample = records[rec_idx_sample]
        error_note = f", {batch_errors} FAILED" if batch_errors else ""
        dedup_note = f", {dedup_count} deduped" if dedup_count else ""
        print(f"  [{batch_end}/{n_entries}] batch of {batch_end - batch_start} done in {elapsed:.1f}s "
              f"(input: {min(batch_sizes)}-{max(batch_sizes)} chars{dedup_note}{error_note}, "
              f"e.g. initiative {rec_sample.get('initiative_id','?')})")
        print(f"  Wrote {batch_file}")
        batch_num += 1

    total_elapsed = time.time() - t_start
    print(f"Inference done in {total_elapsed:.1f}s ({total_elapsed / n_entries:.2f}s per prompt)")
    if skipped_batches:
        print(f"  ({skipped_batches} batches loaded from existing files)")
    if dedup_total:
        print(f"  ({dedup_total} prompts skipped via deduplication)")

    # Reassemble and write combined output
    for rec in records:
        rec["extracted_text_translated"] = None

    for rec_idx, chunks_dict in translated_chunks.items():
        n_chunks = record_chunk_counts[rec_idx]
        parts = []
        for ci in range(n_chunks):
            translation = chunks_dict.get(ci, "")
            if "NO TRANSLATION NEEDED" in translation:
                parts.append(original_chunks.get((rec_idx, ci), ""))
            else:
                parts.append(translation)
        combined = "\n\n".join(parts)
        records[rec_idx]["extracted_text_translated"] = combined
        records[rec_idx]["extracted_text_translated_chars"] = len(combined)
        records[rec_idx]["extracted_text_chunks"] = n_chunks

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    translated = sum(1 for r in records if r.get("extracted_text_translated"))
    print(f"\nDone. Translated {translated}/{total} records.")
    print(f"Batch files: {batch_dir}/ ({batch_num} files)")
    print(f"Combined output: {args.output}")

    if failed_prompts:
        failed_file = output_base + "_failed" + output_ext
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_prompts, f, ensure_ascii=False, indent=2)
        print(f"FAILED: {len(failed_prompts)} prompts could not be parsed. Wrote {failed_file}")


if __name__ == "__main__":
    main()
