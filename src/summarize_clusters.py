"""Summarize clustered EU initiative feedback using vLLM batch inference.

Takes clustering output (from cluster_all_initiatives.py) and produces titled
summaries at every level:
  1. A titled 10-paragraph summary for each policy (initiative documents)
  2. A titled 10-paragraph summary for each feedback item (feedback text + attachments)
  3. Recursive bottom-up summaries for each cluster and sub-cluster

Prompts are collected across all initiatives and batched together for inference,
avoiding many tiny per-initiative batches.

Output: one JSON per initiative with policy_summary, feedback_summaries, and
cluster_summaries (at every hierarchy level).

Usage:
    python3 src/summarize_clusters.py clustering_output/ -o cluster_summaries/
    python3 src/summarize_clusters.py clustering_output/ -o cluster_summaries/ -f whitelist.txt
"""

import argparse
import hashlib
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
from text_utils import group_by_char_budget, should_skip_text, split_into_chunks

DEFAULT_MODEL = "unsloth/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 32768 * 4
MAX_MODEL_LEN = 32768 * 4
CHUNK_SIZE = 16384
COMBINE_BUDGET = CHUNK_SIZE * 4

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


def get_policy_text(initiative: dict, label: str = "") -> str:
    """Concatenate all publication document texts into a single policy text block."""
    parts = []
    for pub in initiative.get("publications", []):
        pub_id = pub.get("id", "?")
        for doc in pub.get("documents", []):
            doc_id = doc.get("document_id", "") or doc.get("id", "?")
            doc_label = f"{label} pub={pub_id} doc={doc_id}" if label else f"pub={pub_id} doc={doc_id}"
            title = doc.get("title", "") or ""
            extracted = doc.get("extracted_text", "") or ""
            summary = doc.get("summary", "") or ""

            text = extracted.strip() if not should_skip_text(extracted, label=doc_label) else summary.strip()
            if not text:
                continue

            if title.strip():
                parts.append(title.strip() + "\n\n" + text)
            else:
                parts.append(text)

    return "\n\n---\n\n".join(parts)


def get_feedback_text(feedback_item: dict, label: str = "") -> str:
    """Concatenate feedback_text + all attachment texts for a single feedback item."""
    parts = []
    fb_text = (feedback_item.get("feedback_text", "") or "").strip()
    if fb_text:
        parts.append(fb_text)

    for att in feedback_item.get("attachments", []):
        att_id = att.get("id", "?")
        att_label = f"{label} att={att_id}" if label else f"att={att_id}"
        extracted = att.get("extracted_text", "") or ""
        summary = att.get("summary", "") or ""
        text = extracted.strip() if not should_skip_text(extracted, label=att_label) else summary.strip()
        if text:
            parts.append(text)

    return "\n\n".join(parts)


# ── Cluster tree building ──


def build_cluster_tree(cluster_assignments: dict) -> dict:
    """Build a tree from dot-separated cluster labels.

    Returns:
        dict: label -> {"children": set of direct child labels,
                        "feedback_ids": list of feedback_ids assigned to exactly this level}
    """
    tree = {}

    for feedback_id, label in cluster_assignments.items():
        if label not in tree:
            tree[label] = {"children": set(), "feedback_ids": []}
        tree[label]["feedback_ids"].append(feedback_id)

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


def find_feedback_by_id(initiative: dict) -> dict:
    """Build a lookup dict: feedback_id (str) -> feedback item dict."""
    lookup = {}
    for fb in initiative.get("middle_feedback", []):
        lookup[str(fb["id"])] = fb
    for pub in initiative.get("publications", []):
        for fb in pub.get("feedback", []):
            fb_id = str(fb["id"])
            if fb_id not in lookup:
                lookup[fb_id] = fb
    return lookup


# ── Cluster cache helpers ──


def compute_cluster_cache_key(cluster_assignments: dict, label: str) -> str:
    """Compute a content-addressed cache key for a cluster label.

    The key is the SHA-256 of the comma-joined sorted feedback IDs that belong
    to this label or any of its descendants (transitive).
    """
    prefix = label + "."
    feedback_ids = sorted(
        fb_id for fb_id, lbl in cluster_assignments.items()
        if lbl == label or lbl.startswith(prefix)
    )
    return hashlib.sha256(",".join(feedback_ids).encode()).hexdigest()


def load_cluster_cache(output_dir: str) -> dict:
    """Load the cluster summary cache from disk."""
    cache_path = os.path.join(output_dir, "_cluster_cache.json")
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cluster_cache(output_dir: str, cache: dict) -> None:
    """Write the cluster summary cache to disk."""
    cache_path = os.path.join(output_dir, "_cluster_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── Cross-initiative batched processing ──


def process_all_initiatives(init_data, llm, sampling_params, encoding,
                            reasoning_effort, batch_size, chunk_size,
                            combine_max_chars, min_noise_summarize_chars,
                            max_prompt_tokens, output_dir,
                            prev_summaries=None):
    """Process all initiatives with cross-initiative batching.

    Instead of processing one initiative at a time (producing many tiny batches),
    collects prompts across all initiatives and runs large batches.

    prev_summaries: optional dict of init_id -> {"policy_summary", "feedback_summaries"}
        from a previous run.  Policy and feedback summaries are reused when the
        source text hasn't changed, avoiding redundant LLM inference.
    """
    if prev_summaries is None:
        prev_summaries = {}
    summary_cache = {}
    all_failed = []

    # Batch directory suffix from pending init IDs (for safe resume across runs)
    run_hash = hashlib.md5(
        ",".join(sorted(init_data.keys())).encode()
    ).hexdigest()[:8]

    batch_dir_p1 = os.path.join(output_dir, f"_batches_p1_{run_hash}")
    batch_dir_p2 = os.path.join(output_dir, f"_batches_p2_{run_hash}")
    batch_dir_p3 = os.path.join(output_dir, f"_batches_p3_{run_hash}")

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Chunk-level summaries (policy + feedback) across all initiatives
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"Phase 1: Chunk-level summaries across {len(init_data)} initiatives")
    print(f"{'='*60}")

    p1_prompts = []
    p1_texts = []
    p1_map = []

    policy_chunk_counts = {}  # init_id -> n_chunks
    fb_chunk_counts = {}  # (init_id, feedback_id) -> n_chunks
    fb_dedup_maps = {}  # init_id -> {dup_id: canonical_id}
    fb_dedup_total = 0
    fb_noise_raw_total = 0
    noise_raw_summaries = {}  # init_id -> {feedback_id -> {"title", "summary"}}
    cluster_assignments_all = {}  # init_id -> cluster_assignments
    reused_policy = {}  # init_id -> {"title", "summary"} from prev
    reused_feedback = {}  # init_id -> {feedback_id -> {"title", "summary"}} from prev
    reused_policy_count = 0
    reused_feedback_count = 0
    reused_from_details_count = 0

    # Pre-compute noise feedback IDs so Phase 1 can skip short ones
    noise_ids_all = {}  # init_id -> set of feedback_ids with label "-1"
    for init_id in sorted(init_data.keys()):
        ca = init_data[init_id].get("cluster_assignments", {})
        noise = {fb_id for fb_id, label in ca.items() if label == "-1"}
        if noise:
            noise_ids_all[init_id] = noise

    for init_id in sorted(init_data.keys()):
        initiative = init_data[init_id]
        cluster_assignments = initiative.get("cluster_assignments", {})
        cluster_assignments_all[init_id] = cluster_assignments

        # Policy prompts — reuse from previous run if available
        prev = prev_summaries.get(init_id, {})
        prev_ps = prev.get("policy_summary")
        if prev_ps and prev_ps.get("summary"):
            reused_policy[init_id] = prev_ps
            reused_policy_count += 1
        else:
            policy_text = get_policy_text(initiative, label=f"init={init_id}")
            if policy_text and not should_skip_text(policy_text, label=f"init={init_id} policy"):
                chunks = split_into_chunks(policy_text, chunk_size,
                                           label=f"init={init_id} policy")
                policy_chunk_counts[init_id] = len(chunks)
                for ci, chunk in enumerate(chunks):
                    prefill = build_prefill(encoding, chunk, POLICY_SUMMARY_PREFIX,
                                            reasoning_effort, IDENTITY_PROMPT,
                                            max_prompt_tokens)
                    if prefill:
                        p1_prompts.append(prefill)
                        p1_texts.append(chunk)
                        p1_map.append((f"pol:{init_id}", ci))

        # Feedback prompts (with per-initiative dedup)
        prev_fb = prev.get("feedback_summaries", {})
        fb_lookup = find_feedback_by_id(initiative)
        fb_text_to_canonical = {}
        fb_dedup_map = {}

        for feedback_id in sorted(cluster_assignments.keys()):
            # Reuse previous feedback summary if available
            prev_fs = prev_fb.get(feedback_id) or prev_fb.get(str(feedback_id))
            if prev_fs and prev_fs.get("summary"):
                reused_feedback.setdefault(init_id, {})[feedback_id] = prev_fs
                reused_feedback_count += 1
                continue

            # Reuse cluster_feedback_summary from initiative_details (durable)
            fb = fb_lookup.get(feedback_id)
            if fb is not None:
                cfs = fb.get("cluster_feedback_summary")
                if cfs and cfs.get("summary"):
                    reused_feedback.setdefault(init_id, {})[feedback_id] = {
                        "title": cfs.get("title", ""),
                        "summary": cfs["summary"],
                    }
                    reused_from_details_count += 1
                    continue

            if fb is None:
                continue
            label = f"init={init_id} fb={feedback_id}"
            text = get_feedback_text(fb, label=label)
            if should_skip_text(text, label=label):
                continue
            text = text.strip()

            # Short noise items: use raw text directly instead of LLM summary
            is_noise = feedback_id in noise_ids_all.get(init_id, set())
            if is_noise and min_noise_summarize_chars and len(text) < min_noise_summarize_chars:
                first_line = text.split("\n", 1)[0][:120].strip()
                noise_raw_summaries.setdefault(init_id, {})[feedback_id] = {
                    "title": first_line,
                    "summary": text,
                }
                fb_noise_raw_total += 1
                continue

            # Deduplicate identical texts within this initiative
            if text in fb_text_to_canonical:
                fb_dedup_map[feedback_id] = fb_text_to_canonical[text]
                fb_dedup_total += 1
                continue
            fb_text_to_canonical[text] = feedback_id

            chunks = split_into_chunks(text, chunk_size, label=label)
            fb_chunk_counts[(init_id, feedback_id)] = len(chunks)

            for ci, chunk in enumerate(chunks):
                prefill = build_prefill(encoding, chunk, FEEDBACK_SUMMARY_PREFIX,
                                        reasoning_effort, IDENTITY_PROMPT,
                                        max_prompt_tokens)
                if prefill:
                    p1_prompts.append(prefill)
                    p1_texts.append(chunk)
                    p1_map.append((f"fb:{init_id}:{feedback_id}", ci))

        fb_dedup_maps[init_id] = fb_dedup_map

    n_policy = len(policy_chunk_counts)
    n_feedback = len(fb_chunk_counts)
    print(f"Collected {len(p1_prompts)} prompts "
          f"({n_policy} policy items, {n_feedback} unique feedback items)")
    if reused_policy_count or reused_feedback_count or reused_from_details_count:
        print(f"  ({reused_policy_count} policy + {reused_feedback_count} feedback "
              f"summaries reused from previous run"
              f"{f', {reused_from_details_count} from initiative_details' if reused_from_details_count else ''}"
              f")")
    if fb_noise_raw_total:
        print(f"  ({fb_noise_raw_total} short noise items using raw text, "
              f"skipped LLM summarization)")
    if fb_dedup_total:
        print(f"  ({fb_dedup_total} feedback items deduplicated within initiatives)")

    # Run Phase 1 inference
    os.makedirs(batch_dir_p1, exist_ok=True)
    p1_results = {}
    if p1_prompts:
        p1_results, p1_failed, _ = run_batch_inference(
            llm, sampling_params, encoding,
            p1_prompts, p1_texts, p1_map,
            batch_size, batch_dir_p1, summary_cache,
            label="[P1] ",
        )
        all_failed.extend(p1_failed)

    # Extract policy results
    policy_summaries = {}  # init_id -> {"title", "summary"}
    policy_multi_chunk = {}  # init_id -> [parts]

    for init_id, n_chunks in policy_chunk_counts.items():
        key = f"pol:{init_id}"
        chunks_dict = p1_results.get(key, {})
        if n_chunks == 1:
            if 0 in chunks_dict:
                t, s = parse_title_and_summary(chunks_dict[0])
                policy_summaries[init_id] = {"title": t, "summary": s}
        else:
            parts = [chunks_dict[ci] for ci in range(n_chunks) if ci in chunks_dict]
            if parts:
                policy_multi_chunk[init_id] = parts

    # Extract feedback results
    feedback_summaries = {}  # init_id -> {feedback_id -> {"title", "summary"}}
    fb_multi_chunk = {}  # (init_id, feedback_id) -> [parts]

    for (init_id, feedback_id), n_chunks in fb_chunk_counts.items():
        key = f"fb:{init_id}:{feedback_id}"
        chunks_dict = p1_results.get(key, {})
        if n_chunks == 1:
            if 0 in chunks_dict:
                t, s = parse_title_and_summary(chunks_dict[0])
                feedback_summaries.setdefault(init_id, {})[feedback_id] = {
                    "title": t, "summary": s,
                }
        else:
            parts = [chunks_dict[ci] for ci in range(n_chunks) if ci in chunks_dict]
            if parts:
                fb_multi_chunk[(init_id, feedback_id)] = parts

    # Merge raw noise summaries into feedback_summaries
    for init_id, fb_map in noise_raw_summaries.items():
        feedback_summaries.setdefault(init_id, {}).update(fb_map)

    # Merge reused summaries from previous run
    policy_summaries.update(reused_policy)
    for init_id, fb_map in reused_feedback.items():
        feedback_summaries.setdefault(init_id, {}).update(fb_map)

    fb_done = sum(len(v) for v in feedback_summaries.values())
    print(f"\nPhase 1 results:")
    print(f"  Policy: {len(policy_summaries)} done, {len(policy_multi_chunk)} multi-chunk")
    print(f"  Feedback: {fb_done} done ({fb_noise_raw_total} raw noise), "
          f"{len(fb_multi_chunk)} multi-chunk")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Combine multi-chunk summaries (policy + feedback)
    # ═══════════════════════════════════════════════════════════

    if policy_multi_chunk or fb_multi_chunk:
        print(f"\n{'='*60}")
        print(f"Phase 2: Combining multi-chunk summaries")
        print(f"{'='*60}")

        os.makedirs(batch_dir_p2, exist_ok=True)
        batch_num_p2 = 0

        # current_parts: key -> (parts_list, combine_prefix)
        current_parts = {}
        for init_id, parts in sorted(policy_multi_chunk.items()):
            current_parts[f"pol:{init_id}"] = (parts, POLICY_COMBINE_PREFIX)
        for (init_id, feedback_id), parts in sorted(fb_multi_chunk.items()):
            current_parts[f"fb:{init_id}:{feedback_id}"] = (parts, FEEDBACK_COMBINE_PREFIX)

        combine_round = 0
        while current_parts:
            combine_round += 1
            print(f"\n  Combine round {combine_round}: {len(current_parts)} items")

            p2_prompts = []
            p2_texts = []
            p2_map = []
            key_groups = {}

            for key in sorted(current_parts.keys()):
                parts, prefix = current_parts[key]
                groups = group_by_char_budget(parts, combine_max_chars)
                key_groups[key] = groups
                for gi, group in enumerate(groups):
                    if len(group) == 1:
                        continue
                    combined = "\n\n".join(group)
                    prefill = build_prefill(encoding, combined, prefix,
                                            reasoning_effort, IDENTITY_PROMPT,
                                            max_prompt_tokens)
                    if prefill:
                        p2_prompts.append(prefill)
                        p2_texts.append(combined)
                        p2_map.append((key, gi))

            p2_results = {}
            if p2_prompts:
                p2_summarized, p2_failed, p2_stats = run_batch_inference(
                    llm, sampling_params, encoding,
                    p2_prompts, p2_texts, p2_map,
                    batch_size, batch_dir_p2, summary_cache,
                    batch_num_start=batch_num_p2,
                    label=f"[P2-R{combine_round}] ",
                )
                batch_num_p2 = p2_stats["batch_num"]
                all_failed.extend(p2_failed)
                p2_results = p2_summarized

            next_parts = {}
            for key, (parts, prefix) in current_parts.items():
                groups = key_groups[key]
                new_parts = []
                for gi, group in enumerate(groups):
                    if len(group) == 1:
                        new_parts.append(group[0])
                    else:
                        result = p2_results.get(key, {}).get(gi)
                        new_parts.append(result if result else group[0])

                if len(new_parts) == 1:
                    t, s = parse_title_and_summary(new_parts[0])
                    if key.startswith("pol:"):
                        init_id = key.split(":", 1)[1]
                        policy_summaries[init_id] = {"title": t, "summary": s}
                    elif key.startswith("fb:"):
                        _, init_id, feedback_id = key.split(":", 2)
                        feedback_summaries.setdefault(init_id, {})[feedback_id] = {
                            "title": t, "summary": s,
                        }
                else:
                    next_parts[key] = (new_parts, prefix)

            current_parts = next_parts

    # Copy summaries to deduplicated feedback IDs
    for init_id, fb_dedup_map in fb_dedup_maps.items():
        init_fb = feedback_summaries.get(init_id, {})
        for dup_id, canonical_id in fb_dedup_map.items():
            if canonical_id in init_fb:
                init_fb[dup_id] = init_fb[canonical_id]

    total_fb = sum(len(v) for v in feedback_summaries.values())
    total_ca = sum(len(ca) for ca in cluster_assignments_all.values())
    print(f"\nFeedback summaries: {total_fb}/{total_ca}"
          f"{f' ({fb_dedup_total} deduplicated)' if fb_dedup_total else ''}")
    print(f"Policy summaries: {len(policy_summaries)}/{len(init_data)}")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Bottom-up cluster summaries across all initiatives
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"Phase 3: Bottom-up cluster summaries across all initiatives")
    print(f"{'='*60}")

    cluster_trees = {}
    max_depth_global = 0
    for init_id, ca in cluster_assignments_all.items():
        if not ca:
            continue
        tree = build_cluster_tree(ca)
        cluster_trees[init_id] = tree
        if tree:
            md = max(get_depth(label) for label in tree.keys())
            max_depth_global = max(max_depth_global, md)

    total_labels = sum(len(t) for t in cluster_trees.values())
    print(f"Cluster trees: {len(cluster_trees)} initiatives, "
          f"{total_labels} labels, max depth {max_depth_global}")

    cluster_summaries = {}  # init_id -> {label -> {"title", "summary", "feedback_count"}}

    # Load content-addressed cluster cache
    cluster_cache = load_cluster_cache(output_dir)
    n_cache_hit = 0

    # Noise handling: each "-1" feedback item becomes its own cluster summary
    # (no combining — noise items are unrelated by definition)
    n_noise_total = 0
    for init_id, tree in cluster_trees.items():
        if "-1" not in tree:
            continue
        node = tree["-1"]
        init_fb = feedback_summaries.get(init_id, {})
        for fb_id in node["feedback_ids"]:
            if fb_id in init_fb:
                fs = init_fb[fb_id]
                cluster_summaries.setdefault(init_id, {})[f"-1:{fb_id}"] = {
                    "title": fs["title"],
                    "summary": fs["summary"],
                    "feedback_count": 1,
                }
            n_noise_total += 1
        del tree["-1"]

    if n_noise_total:
        print(f"Noise: {n_noise_total} feedback items promoted to "
              f"individual cluster summaries (no combining)")

    batch_num_p3 = 0

    for depth in range(max_depth_global, -1, -1):
        # Collect all labels at this depth across all initiatives
        labels_at_depth = {}  # init_id -> [labels]
        n_labels = 0
        for init_id, tree in cluster_trees.items():
            labels = [label for label in tree if get_depth(label) == depth]
            if labels:
                labels_at_depth[init_id] = labels
                n_labels += len(labels)

        if not labels_at_depth:
            continue

        print(f"\n  Depth {depth}: {n_labels} labels across "
              f"{len(labels_at_depth)} initiatives")

        # Build combine prompts for this depth
        p3_prompts = []
        p3_texts = []
        p3_map = []
        label_groups = {}  # (init_id, label) -> groups
        label_fb_counts = {}  # (init_id, label) -> fb_count
        n_single = 0

        for init_id in sorted(labels_at_depth.keys()):
            labels = labels_at_depth[init_id]
            tree = cluster_trees[init_id]
            init_fb = feedback_summaries.get(init_id, {})
            init_cs = cluster_summaries.get(init_id, {})

            for label in sorted(labels):
                node = tree[label]
                texts_to_combine = []
                fb_count = 0

                for fb_id in node["feedback_ids"]:
                    fb_count += 1
                    if fb_id in init_fb:
                        fs = init_fb[fb_id]
                        texts_to_combine.append(
                            fs["title"] + "\n\n" + fs["summary"])

                for child_label in sorted(node["children"]):
                    if child_label in init_cs:
                        cs = init_cs[child_label]
                        fb_count += cs["feedback_count"]
                        texts_to_combine.append(
                            cs["title"] + "\n\n" + cs["summary"])

                label_fb_counts[(init_id, label)] = fb_count

                if not texts_to_combine:
                    continue

                # Check content-addressed cache
                ca = cluster_assignments_all[init_id]
                cache_key = compute_cluster_cache_key(ca, label)
                cached = cluster_cache.get(cache_key)
                if cached and cached.get("summary"):
                    cluster_summaries.setdefault(init_id, {})[label] = {
                        "title": cached["title"],
                        "summary": cached["summary"],
                        "feedback_count": fb_count,
                    }
                    n_cache_hit += 1
                    continue

                if len(texts_to_combine) == 1:
                    t, s = parse_title_and_summary(texts_to_combine[0])
                    cluster_summaries.setdefault(init_id, {})[label] = {
                        "title": t, "summary": s, "feedback_count": fb_count,
                    }
                    n_single += 1
                    continue

                groups = group_by_char_budget(texts_to_combine, combine_max_chars)
                label_groups[(init_id, label)] = groups

                for gi, group in enumerate(groups):
                    if len(group) == 1:
                        continue
                    combined = "\n\n".join(group)
                    prefill = build_prefill(encoding, combined,
                                            CLUSTER_COMBINE_PREFIX,
                                            reasoning_effort, IDENTITY_PROMPT,
                                            max_prompt_tokens)
                    if prefill:
                        p3_prompts.append(prefill)
                        p3_texts.append(combined)
                        p3_map.append((f"cl:{init_id}:{label}", gi))

        if n_single:
            print(f"    {n_single} single-item labels used directly")

        if not p3_prompts and not label_groups:
            continue

        # Run inference for this depth
        p3_results = {}
        if p3_prompts:
            p3_dir = os.path.join(batch_dir_p3, f"depth_{depth}")
            os.makedirs(p3_dir, exist_ok=True)

            p3_summarized, p3_failed, p3_stats = run_batch_inference(
                llm, sampling_params, encoding,
                p3_prompts, p3_texts, p3_map,
                batch_size, p3_dir, summary_cache,
                batch_num_start=batch_num_p3,
                label=f"[P3-D{depth}] ",
            )
            batch_num_p3 = p3_stats["batch_num"]
            all_failed.extend(p3_failed)
            p3_results = p3_summarized

        # Assemble first-round results
        pending_combine = {}  # (init_id, label) -> [parts]

        for (init_id, label), groups in label_groups.items():
            key = f"cl:{init_id}:{label}"
            new_parts = []
            for gi, group in enumerate(groups):
                if len(group) == 1:
                    new_parts.append(group[0])
                else:
                    result = p3_results.get(key, {}).get(gi)
                    new_parts.append(result if result else group[0])

            if len(new_parts) == 1:
                t, s = parse_title_and_summary(new_parts[0])
                cluster_summaries.setdefault(init_id, {})[label] = {
                    "title": t, "summary": s,
                    "feedback_count": label_fb_counts[(init_id, label)],
                }
            else:
                pending_combine[(init_id, label)] = new_parts

        # Extra combining rounds for labels that still have multiple parts
        combine_round = 0
        while pending_combine:
            combine_round += 1
            print(f"    Extra combine round {combine_round}: "
                  f"{len(pending_combine)} labels")

            ex_prompts = []
            ex_texts = []
            ex_map = []
            ex_groups = {}

            for (init_id, label) in sorted(pending_combine.keys()):
                parts = pending_combine[(init_id, label)]
                groups = group_by_char_budget(parts, combine_max_chars)
                ex_groups[(init_id, label)] = groups

                for gi, group in enumerate(groups):
                    if len(group) == 1:
                        continue
                    combined = "\n\n".join(group)
                    prefill = build_prefill(encoding, combined,
                                            CLUSTER_COMBINE_PREFIX,
                                            reasoning_effort, IDENTITY_PROMPT,
                                            max_prompt_tokens)
                    if prefill:
                        ex_prompts.append(prefill)
                        ex_texts.append(combined)
                        ex_map.append((f"cl:{init_id}:{label}", gi))

            ex_results = {}
            if ex_prompts:
                ex_dir = os.path.join(batch_dir_p3,
                                      f"depth_{depth}_x{combine_round}")
                os.makedirs(ex_dir, exist_ok=True)

                ex_summarized, ex_failed, ex_stats = run_batch_inference(
                    llm, sampling_params, encoding,
                    ex_prompts, ex_texts, ex_map,
                    batch_size, ex_dir, summary_cache,
                    batch_num_start=batch_num_p3,
                    label=f"[P3-D{depth}-X{combine_round}] ",
                )
                batch_num_p3 = ex_stats["batch_num"]
                all_failed.extend(ex_failed)
                ex_results = ex_summarized

            next_pending = {}
            for (init_id, label), parts in pending_combine.items():
                key = f"cl:{init_id}:{label}"
                groups = ex_groups[(init_id, label)]
                new_parts = []
                for gi, group in enumerate(groups):
                    if len(group) == 1:
                        new_parts.append(group[0])
                    else:
                        result = ex_results.get(key, {}).get(gi)
                        new_parts.append(result if result else group[0])

                if len(new_parts) == 1:
                    t, s = parse_title_and_summary(new_parts[0])
                    cluster_summaries.setdefault(init_id, {})[label] = {
                        "title": t, "summary": s,
                        "feedback_count": label_fb_counts[(init_id, label)],
                    }
                else:
                    next_pending[(init_id, label)] = new_parts

            pending_combine = next_pending

    total_cs = sum(len(v) for v in cluster_summaries.values())
    print(f"\nCluster summaries: {total_cs} labels across "
          f"{len(cluster_summaries)} initiatives"
          f"{f' ({n_cache_hit} from cache)' if n_cache_hit else ''}")

    # Update cluster cache with new entries (non-noise labels only)
    n_new_cache = 0
    for init_id, cs_map in cluster_summaries.items():
        ca = cluster_assignments_all.get(init_id, {})
        for label, cs in cs_map.items():
            if label.startswith("-1:"):
                continue
            cache_key = compute_cluster_cache_key(ca, label)
            if cache_key not in cluster_cache:
                cluster_cache[cache_key] = {
                    "title": cs["title"],
                    "summary": cs["summary"],
                }
                n_new_cache += 1
    if cluster_cache:
        save_cluster_cache(output_dir, cluster_cache)
        print(f"Cluster cache: {len(cluster_cache)} entries "
              f"({n_new_cache} new, {n_cache_hit} hits)")

    # ═══════════════════════════════════════════════════════════
    # Write output files
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"Writing output files")
    print(f"{'='*60}")

    for init_id in sorted(init_data.keys()):
        initiative = init_data[init_id]
        init_fb = feedback_summaries.get(init_id, {})
        init_cs = cluster_summaries.get(init_id, {})

        output = {
            "initiative_id": initiative.get("id"),
            "short_title": initiative.get("short_title", ""),
            "policy_summary": policy_summaries.get(init_id),
            "feedback_summaries": init_fb,
            "cluster_summaries": {
                label: cs for label, cs in sorted(init_cs.items())
            },
        }

        out_path = os.path.join(output_dir, f"{init_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(init_data)} output files to {output_dir}/")

    if all_failed:
        failed_path = os.path.join(output_dir, "_all_failed.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, ensure_ascii=False, indent=2)
        print(f"FAILED: {len(all_failed)} prompts. Wrote {failed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize clustered EU initiative feedback using vLLM."
    )
    parser.add_argument(
        "input_dir",
        help="Directory of clustering output JSON files "
             "(from cluster_all_initiatives.py)",
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
        "--batch-size", type=int, default=8192,
        help="Number of prompts per inference batch (default: 8192).",
    )
    parser.add_argument(
        "--combine-budget", type=int, default=COMBINE_BUDGET,
        help=f"Max chars when combining summaries per inference call "
             f"(default: {COMBINE_BUDGET}).",
    )
    parser.add_argument(
        "--min-noise-summarize-chars", type=int, default=1000,
        help="Noise feedback shorter than this uses raw text instead of "
             "LLM summary (default: 1000, 0=summarize all).",
    )
    parser.add_argument(
        "--prev-output", type=str, default=None,
        help="Previous output directory to reuse policy/feedback summaries from. "
             "Summaries for items with unchanged source text are carried forward, "
             "skipping LLM inference for those items.",
    )
    args = parser.parse_args()

    # Load whitelist
    whitelist = None
    if args.filter:
        with open(args.filter, encoding="utf-8") as f:
            whitelist = set(line.strip() for line in f if line.strip())
        print(f"Whitelist: {len(whitelist)} initiative IDs from {args.filter}")

    # Initialize encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    reasoning_effort = ReasoningEffort.MEDIUM

    # Discover input files
    input_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith(".json") and not f.startswith("_")
    )
    print(f"Found {len(input_files)} JSON files in {args.input_dir}/")

    def extract_init_id(filename):
        stem = filename.replace(".json", "")
        return stem.split("_")[0]

    # Filter and deduplicate
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

    # Resume: skip initiatives with existing output
    os.makedirs(args.output, exist_ok=True)
    pending = [
        (init_id, f) for init_id, f in selected_files
        if not os.path.isfile(os.path.join(args.output, f"{init_id}.json"))
    ]
    skipped = len(selected_files) - len(pending)
    if skipped:
        print(f"Resume: {skipped}/{len(selected_files)} already exist, "
              f"{len(pending)} remaining")

    if not pending:
        print("\nAll output files already exist. Nothing to do.")
        return

    # Load all pending initiative data
    print(f"\nLoading {len(pending)} initiative files...")
    t_load = time.time()
    init_data = {}
    for init_id, filename in pending:
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            initiative = json.load(f)
        ca = initiative.get("cluster_assignments", {})
        if not ca:
            print(f"  {init_id}: no cluster_assignments, skipping")
            continue
        init_data[init_id] = initiative
    print(f"Loaded {len(init_data)} initiatives in {time.time() - t_load:.1f}s")

    if not init_data:
        print("\nNo initiatives with cluster assignments. Nothing to do.")
        return

    # Load previous output for reuse of policy/feedback summaries
    prev_summaries = {}  # init_id -> {"policy_summary", "feedback_summaries"}
    if args.prev_output and os.path.isdir(args.prev_output):
        print(f"\nLoading previous summaries from {args.prev_output}/...")
        n_prev = 0
        for init_id in init_data:
            prev_path = os.path.join(args.prev_output, f"{init_id}.json")
            if not os.path.isfile(prev_path):
                continue
            with open(prev_path, encoding="utf-8") as f:
                prev = json.load(f)
            prev_summaries[init_id] = {
                "policy_summary": prev.get("policy_summary"),
                "feedback_summaries": prev.get("feedback_summaries", {}),
            }
            n_prev += 1
        print(f"Loaded previous summaries for {n_prev}/{len(init_data)} initiatives")

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

    # Process all initiatives with cross-initiative batching
    t_total = time.time()
    process_all_initiatives(
        init_data, llm, sampling_params, encoding, reasoning_effort,
        args.batch_size, args.chunk_size, args.combine_budget,
        args.min_noise_summarize_chars, args.max_model_len, args.output,
        prev_summaries=prev_summaries,
    )

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"Processed: {len(init_data)}, Skipped: {skipped}, "
          f"Total: {len(selected_files)}")


if __name__ == "__main__":
    main()
