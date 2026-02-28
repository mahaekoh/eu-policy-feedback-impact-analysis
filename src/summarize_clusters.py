"""Summarize clustered EU initiative feedback using vLLM batch inference.

Takes clustering output (from cluster_all_initiatives.py) and produces titled
summaries at every level:
  1. A titled 10-paragraph summary for each policy (initiative documents)
  2. A titled 10-paragraph summary for each feedback item (feedback text + attachments)
  3. Recursive bottom-up summaries for each cluster and sub-cluster

Output: one JSON per initiative with policy_summary, feedback_summaries, and
cluster_summaries (at every hierarchy level).

Usage:
    python3 src/summarize_clusters.py clustering_output/ -o cluster_summaries/
    python3 src/summarize_clusters.py clustering_output/ -o cluster_summaries/ -f whitelist.txt
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
from text_utils import split_into_chunks

DEFAULT_MODEL = "unsloth/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 32768 * 4
MAX_MODEL_LEN = 32768 * 4
CHUNK_SIZE = 16384

IDENTITY_PROMPT = (
    "You are a policy analyst who summarizes EU regulatory documents "
    "clearly and concisely."
)

NUCLEAR_INSTRUCTION = (
    " Be as specific and detailed as possible."
    " If any, preserve all points about nuclear energy, nuclear plants,"
    " or small modular reactors."
    " Do not generate any mete commentary"
    " (for example stating that there are no nuclear-related points)."
)

POLICY_SUMMARY_PREFIX = (
    "Summarize this EU policy initiative document section."
    " Include the policy's objectives, scope, key measures, and regulatory approach."
    " Produce a title on the first line, then a blank line,"
    " then a summary up to 10 paragraphs."
    + NUCLEAR_INSTRUCTION + "\n\n"
)

POLICY_COMBINE_PREFIX = (
    "Combine these summaries of consecutive sections of an EU policy initiative"
    " into a single titled summary."
    " Produce a title on the first line, then a blank line,"
    " then a summary up to 10 paragraphs."
    + NUCLEAR_INSTRUCTION + "\n\n"
)

FEEDBACK_SUMMARY_PREFIX = (
    "Summarize this feedback submission on an EU policy initiative."
    " Include the submitter's position, key arguments, specific recommendations,"
    " and any supporting evidence."
    " Produce a title on the first line, then a blank line,"
    " then a summary up to 10 paragraphs."
    + NUCLEAR_INSTRUCTION + "\n\n"
)

FEEDBACK_COMBINE_PREFIX = (
    "Combine these summaries of consecutive sections of a feedback submission"
    " into a single titled summary."
    " Produce a title on the first line, then a blank line,"
    " then a summary up to 10 paragraphs."
    + NUCLEAR_INSTRUCTION + "\n\n"
)

CLUSTER_COMBINE_PREFIX = (
    "The following are titled summaries of individual feedback submissions"
    " (or sub-groups) on an EU policy initiative, all belonging to the same"
    " thematic cluster. Combine them into a single titled summary that captures"
    " the common themes, range of positions, and key arguments."
    " Produce a title on the first line, then a blank line,"
    " then a summary up to 10 paragraphs."
    + NUCLEAR_INSTRUCTION + "\n\n"
)


# ── Title/summary parsing ──


def parse_title_and_summary(text: str) -> tuple[str, str]:
    """Split LLM output into (title, summary).

    Expected format: first line = title, blank line, rest = summary paragraphs.
    Fallback: if no blank line found, first sentence = title, rest = summary.
    """
    text = text.strip()
    if not text:
        return ("", "")

    # Try splitting on first blank line
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        title = parts[0].strip()
        summary = parts[1].strip()
        if title and summary:
            return (title, summary)

    # Fallback: first sentence = title
    for sep in (". ", "! ", "? "):
        idx = text.find(sep)
        if idx != -1:
            return (text[:idx + 1].strip(), text[idx + 2:].strip())

    # Last resort: entire text as both
    return (text, text)


# ── Text extraction helpers ──


def should_skip_text(text: str) -> bool:
    """Return True if text should not be summarized."""
    if not text or not text.strip():
        return True
    if text.startswith("%PDF-"):
        return True
    return False


def get_policy_text(initiative: dict) -> str:
    """Concatenate all publication document texts into a single policy text block."""
    parts = []
    for pub in initiative.get("publications", []):
        for doc in pub.get("documents", []):
            title = doc.get("title", "") or ""
            extracted = doc.get("extracted_text", "") or ""
            summary = doc.get("summary", "") or ""

            text = extracted.strip() if not should_skip_text(extracted) else summary.strip()
            if not text:
                continue

            if title.strip():
                parts.append(title.strip() + "\n\n" + text)
            else:
                parts.append(text)

    return "\n\n---\n\n".join(parts)


def get_feedback_text(feedback_item: dict) -> str:
    """Concatenate feedback_text + all attachment texts for a single feedback item."""
    parts = []
    fb_text = (feedback_item.get("feedback_text", "") or "").strip()
    if fb_text:
        parts.append(fb_text)

    for att in feedback_item.get("attachments", []):
        extracted = att.get("extracted_text", "") or ""
        summary = att.get("summary", "") or ""
        text = extracted.strip() if not should_skip_text(extracted) else summary.strip()
        if text:
            parts.append(text)

    return "\n\n".join(parts)


# ── Cluster tree building ──


def build_cluster_tree(cluster_assignments: dict) -> dict:
    """Build a tree from dot-separated cluster labels.

    Args:
        cluster_assignments: dict mapping feedback_id (str) -> label (str)

    Returns:
        dict: label -> {"children": set of direct child labels,
                        "feedback_ids": list of feedback_ids assigned to exactly this level}
    """
    tree = {}

    # Collect all labels and their direct feedback
    for feedback_id, label in cluster_assignments.items():
        if label not in tree:
            tree[label] = {"children": set(), "feedback_ids": []}
        tree[label]["feedback_ids"].append(feedback_id)

    # Build parent-child relationships: walk each label up to its root,
    # creating intermediate nodes as needed
    for label in list(tree.keys()):
        current = label
        while "." in current:
            parent = current.rsplit(".", 1)[0]
            if parent not in tree:
                tree[parent] = {"children": set(), "feedback_ids": []}
            tree[parent]["children"].add(current)
            current = parent

    return tree


def get_depth(label: str) -> int:
    """Return the depth of a dot-separated label (0 for top-level)."""
    if not label or label == "-1":
        return 0
    return label.count(".")


# ── Summarization logic ──


def summarize_single_text(text, prompt_prefix, combine_prefix, chunk_size,
                          encoding, reasoning_effort, llm, sampling_params,
                          batch_size, batch_dir, summary_cache, max_combine,
                          item_key, batch_num_start=0, label=""):
    """Handle chunking + combining for one text block. Returns (title, summary) or None.

    Runs Phase 1 (chunk-level) and Phase 2 (combining) for a single text item.
    """
    if should_skip_text(text):
        return None

    text = text.strip()
    chunks = split_into_chunks(text, chunk_size, label=label)
    n_chunks = len(chunks)

    # Phase 1: chunk-level summaries
    prompts = []
    prompt_texts = []
    prompt_map = []

    for ci, chunk in enumerate(chunks):
        prefill = build_prefill(encoding, chunk, prompt_prefix, reasoning_effort, IDENTITY_PROMPT)
        if prefill is None:
            print(f"  {label}item={item_key} chunk {ci+1}/{n_chunks}: SKIPPED (encoding failed)")
            continue
        prompts.append(prefill)
        prompt_texts.append(chunk)
        prompt_map.append((item_key, ci))

    if not prompts:
        return None

    summarized_chunks, failed, stats = run_batch_inference(
        llm, sampling_params, encoding,
        prompts, prompt_texts, prompt_map,
        batch_size, batch_dir, summary_cache,
        batch_num_start=batch_num_start, label=label,
    )
    batch_num = stats["batch_num"]

    # Collect chunk results
    chunks_dict = summarized_chunks.get(item_key, {})

    if n_chunks == 1:
        result = chunks_dict.get(0)
        if result is None:
            return None
        return parse_title_and_summary(result)

    # Multi-chunk: need combining
    parts = []
    for ci in range(n_chunks):
        if ci in chunks_dict:
            parts.append(chunks_dict[ci])
        else:
            print(f"  {label}item={item_key} chunk {ci} missing, skipping")

    if not parts:
        return None

    # Recursively combine
    title, summary = combine_summaries_recursive(
        parts, combine_prefix, max_combine,
        encoding, reasoning_effort, llm, sampling_params,
        batch_size, batch_dir, summary_cache,
        batch_num_start=batch_num, label=label,
    )
    return (title, summary)


def combine_summaries_recursive(texts, combine_prefix, max_combine,
                                encoding, reasoning_effort, llm, sampling_params,
                                batch_size, batch_dir, summary_cache,
                                batch_num_start=0, label=""):
    """Recursively group and combine texts until 1 remains. Returns (title, summary)."""
    current = list(texts)
    batch_num = batch_num_start
    combine_level = 0

    while len(current) > 1:
        combine_level += 1
        groups = [current[i:i + max_combine] for i in range(0, len(current), max_combine)]

        prompts = []
        prompt_texts = []
        prompt_map = []
        single_pass_indices = []

        for gi, group in enumerate(groups):
            if len(group) == 1:
                single_pass_indices.append(gi)
                continue
            combined = "\n\n".join(group)
            prefill = build_prefill(encoding, combined, combine_prefix, reasoning_effort, IDENTITY_PROMPT)
            if prefill is None:
                single_pass_indices.append(gi)
                continue
            prompts.append(prefill)
            prompt_texts.append(combined)
            prompt_map.append(("combine", gi))

        results = {}
        if prompts:
            summarized, failed, stats = run_batch_inference(
                llm, sampling_params, encoding,
                prompts, prompt_texts, prompt_map,
                batch_size, batch_dir, summary_cache,
                batch_num_start=batch_num,
                label=f"{label}[C{combine_level}] ",
            )
            batch_num = stats["batch_num"]
            results = summarized.get("combine", {})

        # Assemble next level
        next_level = []
        for gi, group in enumerate(groups):
            if gi in single_pass_indices:
                next_level.append(group[0])
            elif gi in results:
                next_level.append(results[gi])
            else:
                # Failed, keep first item as fallback
                next_level.append(group[0])

        current = next_level

    return parse_title_and_summary(current[0])


# ── Main processing ──


def find_feedback_by_id(initiative: dict) -> dict:
    """Build a lookup dict: feedback_id (str) -> feedback item dict.

    Searches middle_feedback first (preferred), then publications.feedback.
    """
    lookup = {}
    for fb in initiative.get("middle_feedback", []):
        lookup[str(fb["id"])] = fb
    # Also check publications.feedback as fallback
    for pub in initiative.get("publications", []):
        for fb in pub.get("feedback", []):
            fb_id = str(fb["id"])
            if fb_id not in lookup:
                lookup[fb_id] = fb
    return lookup


def process_initiative(initiative, llm, sampling_params, encoding, reasoning_effort,
                       batch_size, chunk_size, max_combine, output_dir, init_id):
    """Orchestrate all 3 phases for one initiative. Writes output JSON."""
    cluster_assignments = initiative.get("cluster_assignments", {})
    if not cluster_assignments:
        print(f"  No cluster_assignments, skipping")
        return

    batch_dir_p1 = os.path.join(output_dir, "_batches_p1", str(init_id))
    batch_dir_p2 = os.path.join(output_dir, "_batches_p2", str(init_id))
    batch_dir_p3 = os.path.join(output_dir, "_batches_p3", str(init_id))
    os.makedirs(batch_dir_p1, exist_ok=True)
    os.makedirs(batch_dir_p2, exist_ok=True)
    os.makedirs(batch_dir_p3, exist_ok=True)

    summary_cache = {}
    all_failed = []

    # ── Phase 1+2: Policy summary ──

    print(f"\n  --- Policy summary ---")
    policy_text = get_policy_text(initiative)
    policy_summary = None
    if policy_text and not should_skip_text(policy_text):
        result = summarize_single_text(
            policy_text, POLICY_SUMMARY_PREFIX, POLICY_COMBINE_PREFIX,
            chunk_size, encoding, reasoning_effort, llm, sampling_params,
            batch_size, batch_dir_p1, summary_cache, max_combine,
            item_key="policy", label="[POL] ",
        )
        if result:
            policy_summary = {"title": result[0], "summary": result[1]}
            print(f"  Policy summary: {result[0][:80]}")
        else:
            print(f"  Policy summary: FAILED")
    else:
        print(f"  No policy text to summarize")

    # ── Phase 1+2: Feedback summaries ──

    print(f"\n  --- Feedback summaries ({len(cluster_assignments)} items) ---")
    fb_lookup = find_feedback_by_id(initiative)
    feedback_summaries = {}  # feedback_id -> {"title": ..., "summary": ...}

    # Collect all feedback texts and build chunk-level prompts
    fb_prompts = []
    fb_prompt_texts = []
    fb_prompt_map = []  # (feedback_id, chunk_index)
    fb_chunk_counts = {}  # feedback_id -> n_chunks

    for feedback_id in cluster_assignments:
        fb = fb_lookup.get(feedback_id)
        if fb is None:
            print(f"  WARNING: feedback {feedback_id} not found in initiative data")
            continue

        text = get_feedback_text(fb)
        if should_skip_text(text):
            continue

        text = text.strip()
        label = f"init={init_id} fb={feedback_id}"
        chunks = split_into_chunks(text, chunk_size, label=label)
        fb_chunk_counts[feedback_id] = len(chunks)

        for ci, chunk in enumerate(chunks):
            prefill = build_prefill(encoding, chunk, FEEDBACK_SUMMARY_PREFIX, reasoning_effort, IDENTITY_PROMPT)
            if prefill is None:
                print(f"  fb={feedback_id} chunk {ci+1}/{len(chunks)}: SKIPPED (encoding failed)")
                continue
            fb_prompts.append(prefill)
            fb_prompt_texts.append(chunk)
            fb_prompt_map.append((feedback_id, ci))

    if fb_prompts:
        print(f"  Phase 1: {len(fb_prompts)} chunk prompts for {len(fb_chunk_counts)} feedback items")

        fb_batch_dir = os.path.join(batch_dir_p1, "feedback")
        os.makedirs(fb_batch_dir, exist_ok=True)

        fb_summarized, fb_failed, fb_stats = run_batch_inference(
            llm, sampling_params, encoding,
            fb_prompts, fb_prompt_texts, fb_prompt_map,
            batch_size, fb_batch_dir, summary_cache,
            label="[FB-P1] ",
        )
        all_failed.extend(fb_failed)

        # Assemble single-chunk results and collect multi-chunk items
        fb_multi_chunk = {}  # feedback_id -> [chunk summaries in order]

        for feedback_id, n_chunks in fb_chunk_counts.items():
            chunks_dict = fb_summarized.get(feedback_id, {})

            if n_chunks == 1:
                if 0 in chunks_dict:
                    feedback_summaries[feedback_id] = dict(
                        zip(("title", "summary"), parse_title_and_summary(chunks_dict[0]))
                    )
            else:
                parts = []
                for ci in range(n_chunks):
                    if ci in chunks_dict:
                        parts.append(chunks_dict[ci])
                if parts:
                    fb_multi_chunk[feedback_id] = parts

        # Phase 2: combine multi-chunk feedback summaries
        if fb_multi_chunk:
            print(f"\n  Phase 2: combining {len(fb_multi_chunk)} multi-chunk feedback items")

            p2_prompts = []
            p2_texts = []
            p2_map = []
            p2_chunk_groups = {}  # feedback_id -> list of groups
            batch_num_p2 = 0

            current_parts = dict(fb_multi_chunk)

            while current_parts:
                p2_prompts = []
                p2_texts = []
                p2_map = []

                for feedback_id, parts in current_parts.items():
                    groups = [parts[i:i + max_combine] for i in range(0, len(parts), max_combine)]
                    p2_chunk_groups[feedback_id] = groups
                    for gi, group in enumerate(groups):
                        if len(group) == 1:
                            continue
                        combined = "\n\n".join(group)
                        prefill = build_prefill(encoding, combined, FEEDBACK_COMBINE_PREFIX, reasoning_effort, IDENTITY_PROMPT)
                        if prefill is None:
                            continue
                        p2_prompts.append(prefill)
                        p2_texts.append(combined)
                        p2_map.append((feedback_id, gi))

                if p2_prompts:
                    fb_p2_dir = os.path.join(batch_dir_p2, "feedback")
                    os.makedirs(fb_p2_dir, exist_ok=True)

                    p2_results, p2_failed, p2_stats = run_batch_inference(
                        llm, sampling_params, encoding,
                        p2_prompts, p2_texts, p2_map,
                        batch_size, fb_p2_dir, summary_cache,
                        batch_num_start=batch_num_p2, label="[FB-C] ",
                    )
                    batch_num_p2 = p2_stats["batch_num"]
                    all_failed.extend(p2_failed)
                else:
                    p2_results = {}

                # Collect into next level
                next_parts = {}
                for feedback_id, parts in current_parts.items():
                    groups = p2_chunk_groups[feedback_id]
                    new_parts = []
                    for gi, group in enumerate(groups):
                        if len(group) == 1:
                            new_parts.append(group[0])
                        else:
                            result = p2_results.get(feedback_id, {}).get(gi)
                            if result:
                                new_parts.append(result)
                            else:
                                new_parts.append(group[0])  # fallback

                    if len(new_parts) == 1:
                        feedback_summaries[feedback_id] = dict(
                            zip(("title", "summary"), parse_title_and_summary(new_parts[0]))
                        )
                    elif len(new_parts) > 1:
                        next_parts[feedback_id] = new_parts

                current_parts = next_parts
                p2_chunk_groups = {}

    print(f"  Feedback summaries: {len(feedback_summaries)}/{len(cluster_assignments)}")

    # ── Phase 3: Bottom-up cluster summaries ──

    print(f"\n  --- Cluster summaries ---")
    tree = build_cluster_tree(cluster_assignments)

    # Determine processing order: deepest labels first
    all_labels = list(tree.keys())
    max_depth_val = max(get_depth(label) for label in all_labels) if all_labels else 0
    labels_by_depth = {}
    for label in all_labels:
        d = get_depth(label)
        labels_by_depth.setdefault(d, []).append(label)

    print(f"  Cluster tree: {len(all_labels)} labels, max depth {max_depth_val}")
    for d in sorted(labels_by_depth.keys()):
        print(f"    depth {d}: {len(labels_by_depth[d])} labels")

    cluster_summaries = {}  # label -> {"title": ..., "summary": ..., "feedback_count": ...}
    batch_num_p3 = 0

    # Process bottom-up
    for depth in range(max_depth_val, -1, -1):
        labels_at_depth = labels_by_depth.get(depth, [])
        if not labels_at_depth:
            continue

        print(f"\n  Processing depth {depth}: {len(labels_at_depth)} labels")

        # Collect all combine prompts for this depth level
        p3_prompts = []
        p3_texts = []
        p3_map = []  # (label, group_index)
        label_groups = {}  # label -> list of text groups
        label_texts = {}  # label -> list of all texts to combine
        label_fb_counts = {}  # label -> total feedback count

        for label in labels_at_depth:
            node = tree[label]
            texts_to_combine = []

            # Direct feedback summaries
            fb_count = 0
            for fb_id in node["feedback_ids"]:
                fb_count += 1
                if fb_id in feedback_summaries:
                    fs = feedback_summaries[fb_id]
                    texts_to_combine.append(fs["title"] + "\n\n" + fs["summary"])

            # Child cluster summaries
            for child_label in sorted(node["children"]):
                if child_label in cluster_summaries:
                    cs = cluster_summaries[child_label]
                    fb_count += cs["feedback_count"]
                    texts_to_combine.append(cs["title"] + "\n\n" + cs["summary"])

            label_fb_counts[label] = fb_count

            if not texts_to_combine:
                print(f"    label={label}: no texts to combine (0 feedback)")
                continue

            if len(texts_to_combine) == 1:
                # Single item: use directly
                title, summary = parse_title_and_summary(texts_to_combine[0])
                cluster_summaries[label] = {
                    "title": title,
                    "summary": summary,
                    "feedback_count": fb_count,
                }
                print(f"    label={label}: single item, used directly ({fb_count} feedback)")
                continue

            label_texts[label] = texts_to_combine
            # Group for combining
            groups = [texts_to_combine[i:i + max_combine]
                      for i in range(0, len(texts_to_combine), max_combine)]
            label_groups[label] = groups

            for gi, group in enumerate(groups):
                if len(group) == 1:
                    continue
                combined = "\n\n".join(group)
                prefill = build_prefill(encoding, combined, CLUSTER_COMBINE_PREFIX, reasoning_effort, IDENTITY_PROMPT)
                if prefill is None:
                    continue
                p3_prompts.append(prefill)
                p3_texts.append(combined)
                p3_map.append((label, gi))

        if not p3_prompts and not label_groups:
            continue

        # Run inference if needed
        p3_results = {}
        if p3_prompts:
            p3_batch_dir = os.path.join(batch_dir_p3, f"depth_{depth}")
            os.makedirs(p3_batch_dir, exist_ok=True)

            p3_summarized, p3_failed, p3_stats = run_batch_inference(
                llm, sampling_params, encoding,
                p3_prompts, p3_texts, p3_map,
                batch_size, p3_batch_dir, summary_cache,
                batch_num_start=batch_num_p3,
                label=f"[CL-D{depth}] ",
            )
            batch_num_p3 = p3_stats["batch_num"]
            all_failed.extend(p3_failed)
            p3_results = p3_summarized

        # Assemble results and recursively combine if needed
        pending_labels = {}
        for label, groups in label_groups.items():
            new_parts = []
            for gi, group in enumerate(groups):
                if len(group) == 1:
                    new_parts.append(group[0])
                else:
                    result = p3_results.get(label, {}).get(gi)
                    if result:
                        new_parts.append(result)
                    else:
                        new_parts.append(group[0])  # fallback

            if len(new_parts) == 1:
                title, summary = parse_title_and_summary(new_parts[0])
                cluster_summaries[label] = {
                    "title": title,
                    "summary": summary,
                    "feedback_count": label_fb_counts[label],
                }
                print(f"    label={label}: combined ({label_fb_counts[label]} feedback)")
            else:
                pending_labels[label] = new_parts

        # Handle labels that still need more combining rounds
        for label, parts in pending_labels.items():
            p3_extra_dir = os.path.join(batch_dir_p3, f"depth_{depth}_extra_{label}")
            os.makedirs(p3_extra_dir, exist_ok=True)

            title, summary = combine_summaries_recursive(
                parts, CLUSTER_COMBINE_PREFIX, max_combine,
                encoding, reasoning_effort, llm, sampling_params,
                batch_size, p3_extra_dir, summary_cache,
                label=f"[CL-{label}] ",
            )
            cluster_summaries[label] = {
                "title": title,
                "summary": summary,
                "feedback_count": label_fb_counts[label],
            }
            print(f"    label={label}: combined recursively ({label_fb_counts[label]} feedback)")

    print(f"\n  Cluster summaries: {len(cluster_summaries)} labels")

    # ── Write output ──

    output = {
        "initiative_id": initiative.get("id"),
        "short_title": initiative.get("short_title", ""),
        "policy_summary": policy_summary,
        "feedback_summaries": feedback_summaries,
        "cluster_summaries": {
            label: cs for label, cs in sorted(cluster_summaries.items())
        },
    }

    out_path = os.path.join(output_dir, f"{init_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Wrote {out_path}")

    if all_failed:
        failed_path = os.path.join(output_dir, f"_failed_{init_id}.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, ensure_ascii=False, indent=2)
        print(f"  FAILED: {len(all_failed)} prompts. Wrote {failed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize clustered EU initiative feedback using vLLM."
    )
    parser.add_argument(
        "input_dir",
        help="Directory of clustering output JSON files (from cluster_all_initiatives.py)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for summary JSONs.",
    )
    parser.add_argument(
        "-f", "--filter",
        help="Optional whitelist file of initiative IDs (one per line).",
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
        "--max-model-len", type=int, default=MAX_MODEL_LEN,
        help="Max model context length.",
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
        help="Max summaries to combine per inference call (default: 4).",
    )
    args = parser.parse_args()

    # Load whitelist if specified
    whitelist = None
    if args.filter:
        with open(args.filter, encoding="utf-8") as f:
            whitelist = set(line.strip() for line in f if line.strip())
        print(f"Whitelist: {len(whitelist)} initiative IDs from {args.filter}")

    # Initialize openai_harmony encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    reasoning_effort = ReasoningEffort.MEDIUM

    # Discover input files
    input_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith(".json") and not f.startswith("_")
    )
    print(f"Found {len(input_files)} JSON files in {args.input_dir}/")

    # Extract initiative ID from filename (first numeric segment or whole stem)
    def extract_init_id(filename):
        stem = filename.replace(".json", "")
        # Filenames may be plain "12096.json" or "12096_agglomerative_...json"
        parts = stem.split("_")
        return parts[0]

    # Filter by whitelist and deduplicate (take first file per initiative)
    seen_ids = set()
    selected_files = []
    for f in input_files:
        init_id = extract_init_id(f)
        if whitelist and init_id not in whitelist:
            continue
        if init_id in seen_ids:
            continue
        seen_ids.add(init_id)
        selected_files.append((init_id, f))

    print(f"Selected {len(selected_files)} initiatives to process")

    if not selected_files:
        print("Nothing to do.")
        return

    # Resume: skip initiatives whose output already exists
    os.makedirs(args.output, exist_ok=True)
    pending = [
        (init_id, f) for init_id, f in selected_files
        if not os.path.isfile(os.path.join(args.output, f"{init_id}.json"))
    ]
    skipped = len(selected_files) - len(pending)
    if skipped:
        print(f"Resume: {skipped}/{len(selected_files)} output files already exist, "
              f"{len(pending)} remaining")

    if not pending:
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

    # Process each initiative
    t_total_start = time.time()

    for idx, (init_id, filename) in enumerate(pending, 1):
        filepath = os.path.join(args.input_dir, filename)

        print(f"\n{'='*60}")
        print(f"[{idx}/{len(pending)}] Initiative {init_id} ({filename})")
        print(f"{'='*60}")

        with open(filepath, encoding="utf-8") as f:
            initiative = json.load(f)

        process_initiative(
            initiative, llm, sampling_params, encoding, reasoning_effort,
            args.batch_size, args.chunk_size, args.max_combine_chunks,
            args.output, init_id,
        )

    total_elapsed = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"Processed: {len(pending)}, Skipped: {skipped}, Total: {len(selected_files)}")


if __name__ == "__main__":
    main()
