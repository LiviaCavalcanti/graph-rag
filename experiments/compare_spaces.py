"""Compare embedding space quality: GIN-Struct (trained) vs Combined (norm_concat_pca).

Computes intrinsic quality metrics:
  - Silhouette score (CVE-level and CWE-level)
  - Intra-class / inter-class distance ratio (lower = better separation)
  - k-NN purity (fraction of k nearest neighbors sharing same CVE/CWE)
  - Alignment score (mean cosine similarity within same-CVE pairs)

Usage:
    python -m experiments.compare_spaces
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from src.data.autopatch import load_pairs
from src.data.split import build_split
from src.embeddings import REGISTRY
from src.embeddings.gin_struct_model import GINStructModel
from src.embeddings.wl import nx_to_pyg
from src.io.read_write import load_config
from src.training.struct_trainer import StructTripletTrainer

import torch
from torch_geometric.data import Batch


def embed_gin_struct(model, graphs):
    """Embed with trained GIN-struct (train mode for BN)."""
    model.train()
    embeddings = []
    with torch.no_grad():
        for G in graphs:
            data = nx_to_pyg(G)
            if data is None or data.x.shape[0] < 2:
                embeddings.append(np.zeros(model.out_dim, dtype=np.float32))
                continue
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            batch = Batch.from_data_list([data])
            emb = model(batch).cpu().numpy()[0]
            embeddings.append(emb)
    return np.stack(embeddings)


def knn_purity(embs, labels, k=5):
    """Fraction of k-nearest neighbors sharing the same label."""
    dists = cosine_distances(embs)
    np.fill_diagonal(dists, np.inf)
    purities = []
    for i in range(len(labels)):
        neighbors = np.argsort(dists[i])[:k]
        same = sum(1 for j in neighbors if labels[j] == labels[i])
        purities.append(same / k)
    return float(np.mean(purities))


def intra_inter_ratio(embs, labels):
    """Ratio of mean intra-class distance to mean inter-class distance (lower = better)."""
    dists = cosine_distances(embs)
    intra_dists = []
    inter_dists = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_dists.append(dists[i, j])
            else:
                inter_dists.append(dists[i, j])
    mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    mean_inter = float(np.mean(inter_dists)) if inter_dists else 1.0
    return mean_intra / mean_inter if mean_inter > 0 else float('inf')


def alignment_score(embs, labels):
    """Mean cosine similarity for same-label pairs."""
    sims = cosine_similarity(embs)
    scores = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                scores.append(sims[i, j])
    return float(np.mean(scores)) if scores else 0.0


def compute_metrics(embs, cve_labels, cwe_labels, name):
    """Compute all quality metrics for an embedding space."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Embedding shape: {embs.shape}")

    # Filter out labels with <2 samples for silhouette
    cve_counts = defaultdict(int)
    for l in cve_labels:
        cve_counts[l] += 1
    valid_cve = {l for l, c in cve_counts.items() if c >= 2}

    cwe_counts = defaultdict(int)
    for l in cwe_labels:
        cwe_counts[l] += 1
    valid_cwe = {l for l, c in cwe_counts.items() if c >= 2}

    # Silhouette (CVE)
    mask_cve = np.array([l in valid_cve for l in cve_labels])
    if mask_cve.sum() > 10:
        sil_cve = silhouette_score(embs[mask_cve], np.array(cve_labels)[mask_cve], metric="cosine")
    else:
        sil_cve = float('nan')

    # Silhouette (CWE)
    mask_cwe = np.array([l in valid_cwe for l in cwe_labels])
    if mask_cwe.sum() > 10:
        sil_cwe = silhouette_score(embs[mask_cwe], np.array(cwe_labels)[mask_cwe], metric="cosine")
    else:
        sil_cwe = float('nan')

    # k-NN purity
    knn_cve_5 = knn_purity(embs, cve_labels, k=5)
    knn_cwe_5 = knn_purity(embs, cwe_labels, k=5)
    knn_cve_10 = knn_purity(embs, cve_labels, k=10)
    knn_cwe_10 = knn_purity(embs, cwe_labels, k=10)

    # Intra/inter ratio
    ratio_cve = intra_inter_ratio(embs, cve_labels)
    ratio_cwe = intra_inter_ratio(embs, cwe_labels)

    # Alignment (mean same-CVE similarity)
    align_cve = alignment_score(embs, cve_labels)
    align_cwe = alignment_score(embs, cwe_labels)

    results = {
        "silhouette_cve": sil_cve,
        "silhouette_cwe": sil_cwe,
        "knn_purity_cve@5": knn_cve_5,
        "knn_purity_cwe@5": knn_cwe_5,
        "knn_purity_cve@10": knn_cve_10,
        "knn_purity_cwe@10": knn_cwe_10,
        "intra_inter_ratio_cve": ratio_cve,
        "intra_inter_ratio_cwe": ratio_cwe,
        "alignment_cve": align_cve,
        "alignment_cwe": align_cwe,
    }

    print(f"\n  {'Metric':<28} {'Value':>10}")
    print(f"  {'-'*40}")
    for k, v in results.items():
        print(f"  {k:<28} {v:>10.4f}")

    return results


def main():
    print("Loading data...")
    cfg = load_config("config.yaml")
    pairs = load_pairs(cfg)
    index_pairs, query_pairs, _ = build_split(pairs, cfg)
    all_pairs = index_pairs + query_pairs
    graphs = [p.G_vuln for p in all_pairs]
    cve_labels = [p.cve_id for p in all_pairs]
    cwe_labels = [p.cwe_id for p in all_pairs]

    print(f"  {len(all_pairs)} samples, {len(set(cve_labels))} CVEs, {len(set(cwe_labels))} CWEs")

    # ── 1. GIN-Struct (trained) ─────────────────────────────────────
    print("\nComputing GIN-Struct (trained) embeddings...")
    ckpt_path = Path("experiments/output/gin_struct_training/gin_struct_checkpoint.pt")
    model = StructTripletTrainer.load_checkpoint(ckpt_path)
    gin_embs = embed_gin_struct(model, graphs)

    # ── 2. Combined (norm_concat_pca) ──────────────────────────────
    print("\nComputing Combined (norm_concat_pca) embeddings...")
    emb_cfg = cfg["embeddings"]
    combined_embedder = REGISTRY["combined"](emb_cfg)
    combined_embs = combined_embedder.embed_many(graphs)

    # ── 3. Frozen GIN (baseline) ──────────────────────────────────
    print("\nComputing Frozen GIN embeddings...")
    gin_embedder = REGISTRY["gin"](emb_cfg)
    frozen_embs = gin_embedder.embed_many(graphs)

    # ── Compute metrics ────────────────────────────────────────────
    results = {}
    results["gin_struct_trained"] = compute_metrics(gin_embs, cve_labels, cwe_labels, "GIN-Struct (trained)")
    results["combined"] = compute_metrics(combined_embs, cve_labels, cwe_labels, "Combined (norm_concat_pca)")
    results["gin_frozen"] = compute_metrics(frozen_embs, cve_labels, cwe_labels, "GIN (frozen baseline)")

    # ── Summary comparison ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<28} {'GIN-Struct':>12} {'Combined':>12} {'GIN-Frozen':>12}")
    print(f"  {'-'*66}")
    for key in results["gin_struct_trained"]:
        v1 = results["gin_struct_trained"][key]
        v2 = results["combined"][key]
        v3 = results["gin_frozen"][key]
        # Mark best
        vals = [v1, v2, v3]
        # For ratio metrics, lower is better; for others, higher is better
        if "ratio" in key:
            best_idx = np.argmin(vals)
        else:
            best_idx = np.argmax(vals)
        markers = ["", "", ""]
        markers[best_idx] = " *"
        print(f"  {key:<28} {v1:>10.4f}{markers[0]:2s} {v2:>10.4f}{markers[1]:2s} {v3:>10.4f}{markers[2]:2s}")

    print(f"\n  * = best")

    # Save results
    import json
    out_path = Path("experiments/output/space_quality_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
