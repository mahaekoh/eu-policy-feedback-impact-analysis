#!/usr/bin/env python3
"""Cluster all initiatives in unit_summaries/ using Agglomerative or HDBSCAN
clustering with recursive sub-clustering for large clusters.

Agglomerative mode: distance_threshold=0.96, linkage="average".
HDBSCAN mode: min_cluster_size=5, min_samples=3; noise at top level labeled
"-1", noise at sub-levels absorbed back into parent cluster.

Large clusters (>20 items) are recursively sub-clustered up to 4 levels deep.
Sub-cluster labels are hierarchical (e.g. "3.1.2").

Loads the SentenceTransformer model once, then iterates over every initiative
JSON, embeds feedback text, clusters, and writes one output JSON per initiative.

Usage:
    python src/cluster_all_initiatives.py
    python src/cluster_all_initiatives.py --algorithm hdbscan --min-cluster-size 5
    python src/cluster_all_initiatives.py --summaries-dir unit_summaries --output-dir clustering_output
    python src/cluster_all_initiatives.py --model all-mpnet-base-v2 --distance-threshold 1.2
    python src/cluster_all_initiatives.py --max-cluster-size 30 --sub-cluster-scale 0.75 --max-depth 3
"""

import argparse
import copy
import json
import os
import sys
import time

import hdbscan
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# ── Defaults ───────────────────────────────────────────────────────────────
ALGORITHM = "agglomerative"
MODEL_NAME = "google/embeddinggemma-300m"
DISTANCE_THRESHOLD = 0.96
LINKAGE = "average"
SUMMARIES_DIR = "data/analysis/unit_summaries"
OUTPUT_DIR = "data/clustering"
MAX_CLUSTER_SIZE = 20
SUB_CLUSTER_SCALE = 0.75
MAX_DEPTH = 4
# HDBSCAN defaults
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries-dir", default=SUMMARIES_DIR,
                   help="Directory containing per-initiative unit summary JSONs")
    p.add_argument("--output-dir", "-o", default=OUTPUT_DIR,
                   help="Directory to write clustering result JSONs")
    p.add_argument("--model", default=MODEL_NAME,
                   help="SentenceTransformer model name")
    p.add_argument("--algorithm", default=ALGORITHM, choices=["agglomerative", "hdbscan"],
                   help="Clustering algorithm (default: agglomerative)")
    # Agglomerative-specific
    p.add_argument("--distance-threshold", type=float, default=DISTANCE_THRESHOLD,
                   help="AgglomerativeClustering distance_threshold")
    p.add_argument("--linkage", default=LINKAGE,
                   help="AgglomerativeClustering linkage (ward, complete, average, single)")
    p.add_argument("--sub-cluster-scale", type=float, default=SUB_CLUSTER_SCALE,
                   help="Distance threshold multiplier per recursion level (e.g. 0.75 = 25%% reduction each level)")
    # HDBSCAN-specific
    p.add_argument("--min-cluster-size", type=int, default=HDBSCAN_MIN_CLUSTER_SIZE,
                   help="HDBSCAN min_cluster_size parameter")
    p.add_argument("--min-samples", type=int, default=HDBSCAN_MIN_SAMPLES,
                   help="HDBSCAN min_samples parameter")
    # Shared recursive params
    p.add_argument("--max-cluster-size", type=int, default=MAX_CLUSTER_SIZE,
                   help="Clusters larger than this are recursively sub-clustered")
    p.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                   help="Maximum recursion depth for sub-clustering")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip initiatives whose output file already exists")
    return p.parse_args()


def load_feedback(initiative):
    """Extract feedback rows from a unit_summaries-format initiative JSON."""
    rows = []
    for fb in initiative.get("middle_feedback", []):
        raw_fb = fb.get("feedback_text", "") or ""
        sum_att_parts = []
        raw_att_parts = []
        for att in fb.get("attachments", []):
            raw_att_parts.append(att.get("extracted_text", "") or "")
            sum_att_parts.append(att.get("summary", "") or "")
        raw_att = "\n\n".join(filter(None, raw_att_parts))
        sum_att = "\n\n".join(filter(None, sum_att_parts))

        cfs = fb.get("combined_feedback_summary", "") or ""
        if cfs:
            combined = cfs
        else:
            combined = "\n\n".join(filter(None, [raw_fb, sum_att or raw_att]))

        if not combined.strip():
            continue  # skip empty feedback

        rows.append({
            "feedback_id": fb["id"],
            "combined_text": combined,
        })
    return rows


def cluster_recursive(indices, embeddings, base_dt, linkage, scale, max_size,
                      max_depth, depth, prefix):
    """Recursively cluster a subset of embeddings.

    Args:
        indices: array of integer indices into the full embeddings matrix
        embeddings: the full embeddings matrix (indexed by indices)
        base_dt: distance threshold at the top level
        linkage: linkage method
        scale: proportional multiplier per level (e.g. 0.75)
        max_size: cluster size threshold for sub-clustering
        max_depth: maximum recursion depth
        depth: current depth (0-based)
        prefix: label prefix for this level (e.g. "3" or "3.1")

    Returns:
        dict mapping index -> hierarchical label string
    """
    sub_emb = embeddings[indices]
    dt = base_dt * (scale ** depth)

    # Stop recursion: dt too low, only 1 item, or at max depth already
    if dt <= 0 or len(indices) < 2:
        return {idx: prefix for idx in indices}

    ag = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dt,
        linkage=linkage,
    )
    sub_labels = ag.fit_predict(sub_emb)

    # If clustering didn't split (everything in one cluster), just return as-is
    if len(set(sub_labels)) <= 1:
        return {idx: prefix for idx in indices}

    assignments = {}
    for cl in sorted(set(sub_labels)):
        cl_mask = sub_labels == cl
        cl_indices = indices[cl_mask]
        label = f"{prefix}.{cl}" if prefix else str(cl)

        if len(cl_indices) > max_size and depth < max_depth:
            # Recurse
            sub_assignments = cluster_recursive(
                cl_indices, embeddings, base_dt, linkage, scale,
                max_size, max_depth, depth + 1, label,
            )
            assignments.update(sub_assignments)
        else:
            for idx in cl_indices:
                assignments[idx] = label

    return assignments


def cluster_recursive_hdbscan(indices, embeddings, min_cluster_size, min_samples,
                               max_size, max_depth, depth, prefix):
    """Recursively cluster a subset of embeddings using HDBSCAN.

    Noise items at sub-levels are absorbed back into the parent cluster
    (they keep the parent's label). Only top-level noise gets the "-1" label.

    Args:
        indices: array of integer indices into the full embeddings matrix
        embeddings: the full embeddings matrix (indexed by indices)
        min_cluster_size: HDBSCAN min_cluster_size
        min_samples: HDBSCAN min_samples
        max_size: cluster size threshold for sub-clustering
        max_depth: maximum recursion depth
        depth: current depth (0-based)
        prefix: label prefix for this level (e.g. "3" or "3.1")

    Returns:
        dict mapping index -> hierarchical label string
    """
    sub_emb = embeddings[indices]

    if len(indices) < 2 or len(indices) < min_cluster_size:
        return {idx: prefix for idx in indices}

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min(min_samples, len(indices) - 1),
    )
    sub_labels = clusterer.fit_predict(sub_emb)

    unique_clusters = sorted(set(sub_labels) - {-1})

    # If no clusters found (all noise) or only 1 cluster, return as-is
    if len(unique_clusters) <= 1:
        return {idx: prefix for idx in indices}

    assignments = {}

    # Noise items at sub-level stay at parent label
    noise_mask = sub_labels == -1
    for idx in indices[noise_mask]:
        assignments[idx] = prefix

    for cl in unique_clusters:
        cl_mask = sub_labels == cl
        cl_indices = indices[cl_mask]
        label = f"{prefix}.{cl}" if prefix else str(cl)

        if len(cl_indices) > max_size and depth < max_depth:
            sub_assignments = cluster_recursive_hdbscan(
                cl_indices, embeddings, min_cluster_size, min_samples,
                max_size, max_depth, depth + 1, label,
            )
            assignments.update(sub_assignments)
        else:
            for idx in cl_indices:
                assignments[idx] = label

    return assignments


def save_clustering(initiative, initiative_id, rows, assignments, embeddings,
                    model_name, algorithm, params, output_dir):
    """Save a clone of the initiative JSON with clustering results."""
    result = copy.deepcopy(initiative)

    fb_assignments = {str(rows[i]["feedback_id"]): assignments[i]
                      for i in range(len(rows))}

    # Count unique labels (exclude noise label "-1" from cluster count)
    unique_labels = set(assignments.values())
    noise_labels = {"-1"}
    n_clusters = len(unique_labels - noise_labels)
    noise_count = sum(1 for a in assignments.values() if a == "-1")

    # Silhouette on top-level labels (the part before the first dot), excluding noise
    all_top = [a.split(".")[0] for a in assignments.values()]
    non_noise_mask = np.array([t != "-1" for t in all_top])
    top_labels = np.array(all_top)
    sil = None
    if non_noise_mask.all():
        top_unique = set(top_labels)
        if len(top_unique) >= 2:
            sil = float(silhouette_score(embeddings, top_labels))
    else:
        top_unique_nn = set(top_labels[non_noise_mask])
        if len(top_unique_nn) >= 2 and non_noise_mask.sum() >= 2:
            sil = float(silhouette_score(
                embeddings[non_noise_mask], top_labels[non_noise_mask]))

    result["cluster_model"] = model_name
    result["cluster_algorithm"] = algorithm
    result["cluster_params"] = params
    result["cluster_n_clusters"] = n_clusters
    result["cluster_noise_count"] = noise_count
    result["cluster_silhouette"] = sil
    result["cluster_assignments"] = fb_assignments

    model_safe = model_name.replace("/", "_")
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    filename = f"{initiative_id}_{algorithm}_{model_safe}_{param_str}.json"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path


def main():
    args = parse_args()

    # Discover initiative files
    files = sorted(f for f in os.listdir(args.summaries_dir) if f.endswith(".json"))
    print(f"Found {len(files)} initiative files in {args.summaries_dir}/")

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {args.model} (device={device})")
    model = SentenceTransformer(args.model, device=device)
    print(f"Algorithm: {args.algorithm}")
    print(f"Sub-clustering: max_size={args.max_cluster_size}, max_depth={args.max_depth}")
    if args.algorithm == "agglomerative":
        print(f"  distance_threshold={args.distance_threshold}, linkage={args.linkage}, "
              f"scale={args.sub_cluster_scale}")
    else:
        print(f"  min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}")
    print()

    # Stats
    total = len(files)
    skipped = 0
    processed = 0
    errors = 0
    t_start = time.time()

    # Build params dict based on algorithm
    if args.algorithm == "agglomerative":
        params = {
            "distance_threshold": args.distance_threshold,
            "linkage": args.linkage,
            "max_cluster_size": args.max_cluster_size,
            "sub_cluster_scale": args.sub_cluster_scale,
            "max_depth": args.max_depth,
        }
    else:
        params = {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "max_cluster_size": args.max_cluster_size,
            "max_depth": args.max_depth,
        }

    for idx, fname in enumerate(files, 1):
        init_id = fname.replace(".json", "")

        # Check skip-existing
        if args.skip_existing:
            model_safe = args.model.replace("/", "_")
            param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
            out_name = f"{init_id}_{args.algorithm}_{model_safe}_{param_str}.json"
            if os.path.exists(os.path.join(args.output_dir, out_name)):
                skipped += 1
                continue

        src_path = os.path.join(args.summaries_dir, fname)
        try:
            with open(src_path) as f:
                initiative = json.load(f)
        except Exception as e:
            print(f"[{idx}/{total}] ERROR loading {fname}: {e}")
            errors += 1
            continue

        rows = load_feedback(initiative)
        if len(rows) < 2:
            print(f"[{idx}/{total}] {init_id}: {len(rows)} feedback items (skipped, need >=2)")
            skipped += 1
            continue

        # Encode
        texts = [r["combined_text"] for r in rows]
        embeddings = model.encode(texts, show_progress_bar=False)
        embeddings = normalize(embeddings)

        all_indices = np.arange(len(rows))

        if args.algorithm == "hdbscan":
            # ── HDBSCAN path ──
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
            )
            top_labels = clusterer.fit_predict(embeddings)

            noise_mask = top_labels == -1
            n_noise = int(noise_mask.sum())
            unique_clusters = sorted(set(top_labels) - {-1})
            n_top = len(unique_clusters)

            # Build assignments with recursive sub-clustering
            assignments = {}  # index -> hierarchical label string
            sub_clustered = 0

            # Top-level noise gets label "-1"
            for i in all_indices[noise_mask]:
                assignments[i] = "-1"

            for cl in unique_clusters:
                cl_mask = top_labels == cl
                cl_indices = all_indices[cl_mask]
                label = str(cl)

                if len(cl_indices) > args.max_cluster_size and args.max_depth > 0:
                    sub_assignments = cluster_recursive_hdbscan(
                        cl_indices, embeddings, args.min_cluster_size,
                        args.min_samples, args.max_cluster_size,
                        args.max_depth, 1, label,
                    )
                    assignments.update(sub_assignments)
                    sub_labels_set = set(sub_assignments.values())
                    if len(sub_labels_set) > 1:
                        sub_clustered += 1
                else:
                    for i in cl_indices:
                        assignments[i] = label

            n_final = len(set(assignments.values()) - {"-1"})

            out_path = save_clustering(
                initiative, init_id, rows, assignments, embeddings,
                args.model, args.algorithm, params, args.output_dir,
            )

            processed += 1
            print(f"[{idx}/{total}] {init_id}: {len(rows)} items -> {n_top} top clusters "
                  f"({n_noise} noise) -> {n_final} final ({sub_clustered} sub-clustered) "
                  f"-> {os.path.basename(out_path)}")

        else:
            # ── Agglomerative path ──
            ag = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=args.distance_threshold,
                linkage=args.linkage,
            )
            top_labels = ag.fit_predict(embeddings)
            n_top = len(set(top_labels))

            # Build assignments with recursive sub-clustering
            assignments = {}  # index -> hierarchical label string
            sub_clustered = 0
            for cl in sorted(set(top_labels)):
                cl_mask = top_labels == cl
                cl_indices = all_indices[cl_mask]
                label = str(cl)

                if len(cl_indices) > args.max_cluster_size and args.max_depth > 0:
                    sub_assignments = cluster_recursive(
                        cl_indices, embeddings, args.distance_threshold,
                        args.linkage, args.sub_cluster_scale, args.max_cluster_size,
                        args.max_depth, 1, label,
                    )
                    assignments.update(sub_assignments)
                    sub_labels_set = set(sub_assignments.values())
                    if len(sub_labels_set) > 1:
                        sub_clustered += 1
                else:
                    for i in cl_indices:
                        assignments[i] = label

            n_final = len(set(assignments.values()))

            out_path = save_clustering(
                initiative, init_id, rows, assignments, embeddings,
                args.model, args.algorithm, params, args.output_dir,
            )

            processed += 1
            print(f"[{idx}/{total}] {init_id}: {len(rows)} items -> {n_top} top clusters "
                  f"-> {n_final} final ({sub_clustered} sub-clustered) "
                  f"-> {os.path.basename(out_path)}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
