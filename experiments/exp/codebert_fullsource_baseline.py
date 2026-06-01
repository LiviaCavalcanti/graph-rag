#!/usr/bin/env python3
"""
CodeBERT Full-Source Baseline Experiment
========================================

Evaluates CodeBERT embeddings on the **full vulnerable function source code**
(no graph-based diff filtering). This serves as the baseline to demonstrate
the value of graph-aware code selection in the retrieval pipeline.

Comparison:
    - CodeBERT (seq):  Embeds only diff-weighted nodes (diff_weight > 0.3)
                       selected from the CPG vulnerability slice (G_vuln).
    - This script:     Embeds the ENTIRE function body reconstructed from
                       all CPG nodes (no diff awareness), or from raw source
                       files when available.

Method:
    1. For each CVE in CVE-list/, load the G_before CPG from graphml_augmented/
       (variants: original, augmented, patch_augmented).
    2. Reconstruct the full function text by concatenating ALL node CODE
       attributes in LINE_NUMBER order (same as codexglue_baseline embedder).
    3. Encode through CodeBERT [CLS] token → 768-dim embedding → L2 normalize.
    4. Evaluate via leave-one-out cosine similarity retrieval:
       - Hit@1, Hit@5: fraction of queries where a same-CVE entry is in top-k.
       - MRR: mean reciprocal rank of first same-CVE result.
       - CWE Recall: macro-averaged per-CWE hit rate in top-5.

Usage:
    python -m experiments.exp.codebert_fullsource_baseline [--output DIR]

Results (n=150, original + augmented variants):
    H@1=0.1533  H@5=0.2200  MRR=0.1826  CWE_R=0.2933

Compare with graph-filtered CodeBERT (seq) on same dataset (n=225):
    H@1=0.7534  H@5=0.8630  MRR=0.7927  CWE_R=0.4121
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.pipeline import load_cpg_dir
from src.embeddings.codexglue_baseline import extract_full_function


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/codebert-base"
DEFAULT_OUTPUT_DIR = "experiments/output/codebert_fullsource_baseline"
CVE_ROOT = "CVE-list"
GRAPHML_ROOT = "graphml_augmented"
VARIANTS = ["original", "augmented", "patch_augmented"]
MAX_SEQ_LEN = 512
BATCH_SIZE = 16


# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_pairs(cve_root: Path, graphml_root: Path, variants: list[str]) -> list[dict]:
    """Load pairs by extracting full function text from G_before CPGs.

    This avoids computing graph diffs (the expensive step) since we only
    need the raw node CODE attributes from the 'before' graph.
    """
    pairs = []
    for cve_dir in sorted(cve_root.iterdir()):
        if not cve_dir.is_dir():
            continue
        db_path = cve_dir / "out_v2" / "db_entry.json"
        if not db_path.exists():
            continue
        db = json.loads(db_path.read_text())
        cve_id = str(db.get("cve_id", cve_dir.name))
        cwe_id = str(db.get("cwe_type", ""))
        func_name = str(db.get("function_name", ""))

        for variant in variants:
            before_path = graphml_root / cve_dir.name / variant / "before"
            if not before_path.exists():
                continue
            G = load_cpg_dir(before_path)
            if G is None or G.number_of_nodes() == 0:
                continue
            code = extract_full_function(G)
            if not code.strip():
                continue
            pairs.append({
                "dir": cve_dir.name,
                "cve_id": cve_id,
                "cwe_id": cwe_id,
                "func_name": func_name,
                "code": code,
                "variant": variant,
            })
    return pairs


def encode_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_SEQ_LEN,
) -> np.ndarray:
    """Encode texts through CodeBERT and return [CLS] embeddings."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].numpy()
        all_embs.append(cls)
    return np.vstack(all_embs)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return embeddings / norms


def evaluate_loo(pairs: list[dict], embeddings: np.ndarray) -> dict:
    """Leave-one-out evaluation via cosine similarity."""
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1)  # exclude self

    n = len(pairs)
    hits_at = {1: 0, 5: 0, 10: 0}
    rr_sum = 0.0
    cwe_hits = defaultdict(lambda: {"total": 0, "hit": 0})
    raw_queries = []

    for i, qp in enumerate(pairs):
        sims = sim[i].copy()
        # Exclude exact self (same dir + variant)
        for j in range(n):
            if pairs[j]["dir"] == qp["dir"] and pairs[j]["variant"] == qp["variant"]:
                sims[j] = -2
        top_idx = np.argsort(sims)[::-1][:10]

        target_cve = qp["cve_id"]
        hit_rank = None
        for rank, j in enumerate(top_idx, 1):
            if pairs[j]["cve_id"] == target_cve:
                if hit_rank is None:
                    hit_rank = rank
                break

        for k in [1, 5, 10]:
            if any(pairs[j]["cve_id"] == target_cve for j in top_idx[:k]):
                hits_at[k] += 1

        if hit_rank is not None:
            rr_sum += 1.0 / hit_rank

        cwe = qp["cwe_id"]
        if cwe:
            cwe_hits[cwe]["total"] += 1
            if any(pairs[j]["cwe_id"] == cwe for j in top_idx[:5]):
                cwe_hits[cwe]["hit"] += 1

        raw_queries.append({
            "query_cve": target_cve,
            "query_cwe": cwe,
            "query_func": qp["func_name"],
            "query_variant": qp["variant"],
            "hit_rank": hit_rank,
            "top5_cves": [pairs[j]["cve_id"] for j in top_idx[:5]],
        })

    cwe_recall = float(np.mean(
        [v["hit"] / v["total"] for v in cwe_hits.values() if v["total"] > 0]
    ))

    return {
        "hit@1": hits_at[1] / n,
        "hit@5": hits_at[5] / n,
        "hit@10": hits_at[10] / n,
        "mrr": rr_sum / n,
        "cwe_recall": cwe_recall,
        "n": n,
        "n_with_match": sum(1 for q in raw_queries if q["hit_rank"] is not None),
        "raw_queries": raw_queries,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="CodeBERT full-source baseline (no graph diff filtering)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results JSON",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Path to CodeBERT model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cve_root = Path(CVE_ROOT)
    graphml_root = Path(GRAPHML_ROOT)

    # ── 1. Load pairs ──────────────────────────────────────────────
    print("Loading pairs from graphml (no diff computation)...")
    t0 = time.perf_counter()
    pairs = load_pairs(cve_root, graphml_root, VARIANTS)
    load_time = time.perf_counter() - t0
    print(f"  Loaded {len(pairs)} pairs in {load_time:.1f}s")
    print(f"  Unique CVEs: {len(set(p['cve_id'] for p in pairs))}")
    print(f"  Variants: {dict(sorted(defaultdict(int, {v: sum(1 for p in pairs if p['variant'] == v) for v in VARIANTS}).items()))}")

    # ── 2. Load CodeBERT ───────────────────────────────────────────
    print(f"\nLoading CodeBERT from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    # ── 3. Embed ───────────────────────────────────────────────────
    print("Embedding full-source texts...")
    t1 = time.perf_counter()
    codes = [p["code"] for p in pairs]
    embeddings = encode_batch(codes, tokenizer, model)
    embeddings = l2_normalize(embeddings)
    embed_time = time.perf_counter() - t1
    print(f"  Embedded {embeddings.shape[0]} vectors (dim={embeddings.shape[1]}) in {embed_time:.1f}s")

    # Embedding space stats
    pairwise_sim = embeddings @ embeddings.T
    np.fill_diagonal(pairwise_sim, np.nan)
    mean_sim = float(np.nanmean(pairwise_sim))
    # Effective dimensionality via PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, embeddings.shape[0] - 1))
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eff_dim = float(np.searchsorted(cumvar, 0.90) + 1)
    print(f"  Mean pairwise sim: {mean_sim:.4f}")
    print(f"  Effective dim (90% var): {eff_dim:.0f}")

    # ── 4. Evaluate ────────────────────────────────────────────────
    print("\nRunning leave-one-out evaluation...")
    results = evaluate_loo(pairs, embeddings)
    print(f"\n{'='*50}")
    print(f"  RESULTS (CodeBERT full-source baseline, n={results['n']})")
    print(f"{'='*50}")
    print(f"  Hit@1:       {results['hit@1']:.4f}")
    print(f"  Hit@5:       {results['hit@5']:.4f}")
    print(f"  Hit@10:      {results['hit@10']:.4f}")
    print(f"  MRR:         {results['mrr']:.4f}")
    print(f"  CWE Recall:  {results['cwe_recall']:.4f}")
    print(f"  Queries with match: {results['n_with_match']}/{results['n']}")

    # ── 5. Save ────────────────────────────────────────────────────
    output = {
        "experiment": "codebert_fullsource_baseline",
        "description": (
            "CodeBERT [CLS] embedding of full vulnerable function source code, "
            "reconstructed from all CPG nodes without any graph-diff filtering. "
            "Baseline to demonstrate value of graph-aware code selection."
        ),
        "config": {
            "model": args.model,
            "max_seq_len": MAX_SEQ_LEN,
            "variants": VARIANTS,
            "cve_root": CVE_ROOT,
            "graphml_root": GRAPHML_ROOT,
        },
        "timing": {
            "load_s": load_time,
            "embed_s": embed_time,
        },
        "space_stats": {
            "mean_pairwise_sim": mean_sim,
            "effective_dim_90": eff_dim,
            "embedding_dim": int(embeddings.shape[1]),
        },
        "results": {
            "hit@1": results["hit@1"],
            "hit@5": results["hit@5"],
            "hit@10": results["hit@10"],
            "mrr": results["mrr"],
            "cwe_recall": results["cwe_recall"],
            "n_pairs": results["n"],
            "n_with_match": results["n_with_match"],
        },
        "raw_queries": results["raw_queries"],
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
