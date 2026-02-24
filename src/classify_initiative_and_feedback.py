"""Classify EU initiative summaries by nuclear energy stance using vLLM batch inference.

Takes the output directory from build_unit_summaries.py, which contains per-initiative
JSON files with before_feedback_summary and after_feedback_summary fields.
Classifies each summary as SUPPORT, OPPOSE, NEUTRAL, or DOES NOT MENTION with respect
to nuclear energy.

Usage:
    python3 src/classify_initiative_and_feedback.py unit_summaries/ -o classified_output/
    python3 src/classify_initiative_and_feedback.py unit_summaries/ -o classified_output/ --batch-size 16
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

DEFAULT_MODEL = "unsloth/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 32768

VALID_LABELS = ["DOES NOT MENTION", "SUPPORT", "OPPOSE", "NEUTRAL"]

IDENTITY_PROMPT = (
    "You are a policy analyst who classifies EU regulatory documents "
    "by their stance on nuclear energy."
)

CLASSIFICATION_PROMPT_PREFIX = (
    'The following is a summary of an EU policy proposal. Generate "SUPPORT" or '
    '"OPPOSE" or "NEUTRAL" or "DOES NOT MENTION" based on whether the piece of '
    "proposal is in support of nuclear energy (and the construction of nuclear "
    "power plants and/or small modular reactors) or in opposition, or neutral, "
    "or none of the above. Do not generate any other meta commentary.\n\n"
)

FEEDBACK_CLASSIFICATION_PROMPT_PREFIX = (
    'The following is a summary of a piece of feedback to an EU policy proposal. Generate "SUPPORT" or '
    '"OPPOSE" or "NEUTRAL" or "DOES NOT MENTION" based on whether the piece of '
    "proposal is in support of nuclear energy (and the construction of nuclear "
    "power plants and/or small modular reactors) or in opposition, or neutral, "
    "or none of the above. Do not generate any other meta commentary.\n\n"
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


def extract_outputs(outputs, encoding: HarmonyEncoding) -> list:
    """Extract the 'analysis' and 'final' channel texts from each vLLM output.

    Returns a list with one entry per output. Successful entries are
    (analysis_text, final_text) tuples. Failed entries are None.
    """
    results = []
    for i, output in enumerate(outputs):
        try:
            gen = output.outputs[0]
            output_tokens = gen.token_ids
            entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
            analysis_message = None
            final_message = None
            for message in entries:
                if message.channel == "analysis":
                    analysis_message = message.content[0].text
                elif message.channel == "final":
                    final_message = message.content[0].text
            if final_message is None:
                print(f"  WARNING: no 'final' channel in output {i}")
                results.append(None)
            else:
                results.append((analysis_message, final_message))
        except Exception as e:
            print(f"  WARNING: failed to parse output {i}: {e}")
            results.append(None)
    return results


def extract_label(text: str) -> str | None:
    """Extract a valid classification label from model output.

    Searches for the first occurrence of a valid label (checking longest first).
    Returns None if no valid label found.
    """
    if not text:
        return None
    upper = text.upper()
    for label in VALID_LABELS:  # DOES NOT MENTION checked first (longest)
        if label in upper:
            return label
    return None


def run_batch_inference(llm, sampling_params, encoding, prompts, prompt_texts,
                        prompt_map, batch_size, batch_dir, result_cache,
                        batch_num_start=0, label=""):
    """Run batched inference with dedup, resume, and per-batch file output.

    Args:
        llm: vLLM LLM instance.
        sampling_params: vLLM SamplingParams.
        encoding: HarmonyEncoding for output parsing.
        prompts: list of prefill dicts.
        prompt_texts: list of raw text strings (for dedup).
        prompt_map: list of prompt key strings.
        batch_size: number of prompts per batch.
        batch_dir: directory for per-batch output files.
        result_cache: dict of text -> label (shared across calls for dedup).
        batch_num_start: starting batch file number.
        label: prefix for log messages.

    Returns:
        results: dict of prompt_key -> {"label": str, "reasoning": str|None}
        failed_prompts: list of failed prompt info dicts.
        stats: dict with batch_num, skipped_batches, dedup_total.
    """
    n_prompts = len(prompts)
    text_lens = [len(t) for t in prompt_texts]
    results = {}
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
                results[entry["key"]] = {
                    "label": entry["label"],
                    "reasoning": entry.get("reasoning"),
                }
            # Populate result_cache
            for j in range(batch_start, batch_end):
                key = prompt_map[j]
                ct = prompt_texts[j]
                result = results.get(key)
                if result is not None and ct not in result_cache:
                    result_cache[ct] = result
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
            if ct in result_cache:
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
            key = prompt_map[j]
            snippet = prompt_texts[j][:80].replace("\n", " ").strip()
            dedup_tag = " [dedup]" if j in set(dedup_indices) else ""
            print(f"    key={key} ({text_lens[j]} chars) \"{snippet}...\"{dedup_tag}")

        # Run inference
        if infer_prompts:
            t0 = time.time()
            batch_outputs = llm.generate(infer_prompts, sampling_params)
            elapsed = time.time() - t0

            parsed_outputs = extract_outputs(batch_outputs, encoding)
            batch_errors = 0
            for k, parsed in enumerate(parsed_outputs):
                j = infer_indices[k]
                key = prompt_map[j]
                if parsed is None:
                    batch_errors += 1
                    failed_prompts.append({
                        "key": key,
                        "text_length": text_lens[j],
                    })
                    continue
                analysis_text, final_text = parsed
                parsed_label = extract_label(final_text)
                if parsed_label is None:
                    print(f"  WARNING: no valid label found in output for {key}: {final_text!r}")
                    failed_prompts.append({
                        "key": key,
                        "text_length": text_lens[j],
                        "raw_output": final_text,
                    })
                    continue
                entry = {"label": parsed_label, "reasoning": analysis_text}
                results[key] = entry
                result_cache[prompt_texts[j]] = entry
                batch_results.append({
                    "key": key,
                    "label": parsed_label,
                    "reasoning": analysis_text,
                    "raw_output": final_text,
                })
        else:
            elapsed = 0.0
            batch_errors = 0

        # Resolve deduped prompts
        for j in dedup_indices:
            ct = prompt_texts[j]
            cached_entry = result_cache.get(ct)
            key = prompt_map[j]
            if cached_entry is not None:
                results[key] = cached_entry
                batch_results.append({
                    "key": key,
                    "label": cached_entry["label"],
                    "reasoning": cached_entry["reasoning"],
                })
            else:
                failed_prompts.append({
                    "key": key,
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

    return results, failed_prompts, {
        "batch_num": batch_num,
        "skipped_batches": skipped_batches,
        "dedup_total": dedup_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify EU initiative summaries by nuclear energy stance using vLLM."
    )
    parser.add_argument(
        "input_dir",
        help="Directory of per-initiative JSON files (output of build_unit_summaries.py)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for initiative JSONs with classification labels added.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_OUTPUT_TOKENS,
        help=f"Max output tokens per classification (default: {MAX_OUTPUT_TOKENS}).",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=32768 * 4,
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
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of independent classification runs per prompt (default: 1).",
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

    # Collect all prompts from pending files
    print(f"\nCollecting prompts from {len(pending_files)} initiative files...")
    prompts = []
    prompt_texts = []
    prompt_map = []  # list of string keys like "12096:before", "12096:after", "12096:fb:0"
    prompt_file_map = {}  # key -> filename

    for filename in pending_files:
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            initiative = json.load(f)

        init_id = os.path.splitext(filename)[0]

        # Initiative-level: before/after feedback summaries
        for field, key_suffix in [
            ("before_feedback_summary", "before"),
            ("after_feedback_summary", "after"),
        ]:
            text = initiative.get(field, "")
            if not text or not text.strip():
                continue
            text = text.strip()
            key = f"{init_id}:{key_suffix}"
            prefill = build_prefill(encoding, text, CLASSIFICATION_PROMPT_PREFIX, reasoning_effort)
            if prefill is None:
                print(f"  {filename} {field}: {len(text)} chars — SKIPPED (encoding failed)")
                continue
            n_tokens = len(prefill["prompt_token_ids"])
            print(f"  {filename} {field}: {len(text)} chars, {n_tokens} tokens")
            prompts.append(prefill)
            prompt_texts.append(text)
            prompt_map.append(key)
            prompt_file_map[key] = filename

        # Feedback-level: each middle_feedback combined_feedback_summary
        for fb_idx, fb in enumerate(initiative.get("middle_feedback", [])):
            text = fb.get("combined_feedback_summary", "")
            if not text or not text.strip():
                continue
            text = text.strip()
            key = f"{init_id}:fb:{fb_idx}"
            prefill = build_prefill(encoding, text, FEEDBACK_CLASSIFICATION_PROMPT_PREFIX, reasoning_effort)
            if prefill is None:
                print(f"  {filename} middle_feedback[{fb_idx}]: {len(text)} chars — SKIPPED (encoding failed)")
                continue
            n_tokens = len(prefill["prompt_token_ids"])
            print(f"  {filename} middle_feedback[{fb_idx}]: {len(text)} chars, {n_tokens} tokens")
            prompts.append(prefill)
            prompt_texts.append(text)
            prompt_map.append(key)
            prompt_file_map[key] = filename

    n_prompts = len(prompts)
    print(f"\nTotal prompts: {n_prompts}")

    if n_prompts == 0:
        # No summaries to classify, just copy files through
        for filename in pending_files:
            filepath = os.path.join(args.input_dir, filename)
            with open(filepath, encoding="utf-8") as f:
                initiative = json.load(f)
            out_path = os.path.join(args.output, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(initiative, f, ensure_ascii=False, indent=2)
        print(f"No summaries to classify, wrote {len(pending_files)} files unchanged.")
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

    # Run inference (one pass per run, each with its own batch dir and cache)
    n_runs = args.runs
    all_run_results = []  # list of per-run result dicts
    all_failed = []
    t_total_start = time.time()

    for run_idx in range(n_runs):
        run_label = f"[run {run_idx+1}/{n_runs}] " if n_runs > 1 else ""
        if n_runs > 1:
            print(f"\n{'='*60}")
            print(f"Run {run_idx+1}/{n_runs}")
            print(f"{'='*60}\n")

        batch_dir = os.path.join(args.output, f"_batches_run_{run_idx}")
        os.makedirs(batch_dir, exist_ok=True)

        result_cache = {}  # fresh cache per run for independent samples

        results, failed_prompts, stats = run_batch_inference(
            llm, sampling_params, encoding,
            prompts, prompt_texts, prompt_map,
            args.batch_size, batch_dir, result_cache,
            batch_num_start=0, label=run_label,
        )
        all_run_results.append(results)
        all_failed.extend(failed_prompts)

    total_elapsed = time.time() - t_total_start

    # Write output files
    print(f"\nWriting output files...")
    written = 0
    for filename in pending_files:
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            initiative = json.load(f)

        init_id = os.path.splitext(filename)[0]

        before_key = f"{init_id}:before"
        after_key = f"{init_id}:after"

        if n_runs == 1:
            results = all_run_results[0]
            if before_key in results:
                initiative["before_feedback_nuclear_stance"] = results[before_key]["label"]
                initiative["before_feedback_nuclear_stance_reasoning"] = results[before_key]["reasoning"]
            if after_key in results:
                initiative["after_feedback_nuclear_stance"] = results[after_key]["label"]
                initiative["after_feedback_nuclear_stance_reasoning"] = results[after_key]["reasoning"]
            for fb_idx, fb in enumerate(initiative.get("middle_feedback", [])):
                fb_key = f"{init_id}:fb:{fb_idx}"
                if fb_key in results:
                    fb["nuclear_stance"] = results[fb_key]["label"]
                    fb["nuclear_stance_reasoning"] = results[fb_key]["reasoning"]
        else:
            # Collect across runs into lists
            before_entries = [r[before_key] for r in all_run_results if before_key in r]
            if before_entries:
                initiative["before_feedback_nuclear_stance"] = [e["label"] for e in before_entries]
                initiative["before_feedback_nuclear_stance_reasoning"] = [e["reasoning"] for e in before_entries]
            after_entries = [r[after_key] for r in all_run_results if after_key in r]
            if after_entries:
                initiative["after_feedback_nuclear_stance"] = [e["label"] for e in after_entries]
                initiative["after_feedback_nuclear_stance_reasoning"] = [e["reasoning"] for e in after_entries]
            for fb_idx, fb in enumerate(initiative.get("middle_feedback", [])):
                fb_key = f"{init_id}:fb:{fb_idx}"
                fb_entries = [r[fb_key] for r in all_run_results if fb_key in r]
                if fb_entries:
                    fb["nuclear_stance"] = [e["label"] for e in fb_entries]
                    fb["nuclear_stance_reasoning"] = [e["reasoning"] for e in fb_entries]

        out_path = os.path.join(args.output, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(initiative, f, ensure_ascii=False, indent=2)
        written += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_elapsed:.1f}s ({n_runs} run{'s' if n_runs > 1 else ''})")
    print(f"{'='*60}")
    print(f"Initiative files: {written + skipped_files}/{len(initiative_files)} "
          f"({written} written, {skipped_files} already existed)")
    total_classifications = sum(len(r) for r in all_run_results)
    print(f"Classifications: {total_classifications}/{n_prompts * n_runs}")

    # Print label distribution (aggregated across all runs)
    label_counts = {}
    for results in all_run_results:
        for entry in results.values():
            lbl = entry["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"Label distribution: {label_counts}")

    if all_failed:
        failed_file = os.path.join(args.output, "_failed.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, ensure_ascii=False, indent=2)
        print(f"FAILED: {len(all_failed)} prompts could not be classified. Wrote {failed_file}")


if __name__ == "__main__":
    main()
