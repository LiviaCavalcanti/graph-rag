#!/usr/bin/env python3
"""
Compare slicing methodologies with and without diff labelling.

Four conditions:
  1. G_before                  — full pre-patch graph, no diff info  (≈ old ID-based)
  2. G_vuln (no labels)        — fingerprint-sliced, diff attrs stripped
  3. G_vuln (labels)           — fingerprint-sliced, diff labels + weights (current)
  4. G_before (uniform labels) — full pre-patch graph, all nodes labelled 'context' w=0.2

For each condition × embedder, runs self-retrieval (hit@1/5/10, MRR)
and CWE recall, then prints a comparison table.

Usage:
    cd /home/z0050s2b/code/graph-rag
    python -m experiments.slicing_comparison [--config config.yaml]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import uuid
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.autopatch import AutoPatchDataset
from src.embeddings import build_embedders
from src.rag.hnsw import HNSWIndex
from src.rag.retriever import Retriever
from experiments.metrics import embedding_space_stats
from experiments.runner import _build_split_plan

OUTPUT_DIR = Path("experiments/output")


# ── graph variant factories ──────────────────────────────────────────

def _strip_diff_attrs(G):
    """Return a copy with diff/diff_weight removed from all nodes."""
    G2 = G.copy()
    for n in G2:
        G2.nodes[n].pop("diff", None)
        G2.nodes[n].pop("diff_weight", None)
    return G2


def _add_uniform_labels(G, label="context", weight=0.2):
    """Return a copy with uniform diff labels on every node."""
    G2 = G.copy()
    for n in G2:
        G2.nodes[n]["diff"] = label
        G2.nodes[n]["diff_weight"] = weight
    return G2


VARIANT_DEFS = {
    "G_before": {
        "desc": "Full pre-patch graph, no diff (≈ old ID-based)",
        "build": lambda pair: _strip_diff_attrs(pair.G_before),
    },
    "G_before_uniform": {
        "desc": "Full pre-patch graph, all nodes labelled context (w=0.2)",
        "build": lambda pair: _add_uniform_labels(pair.G_before),
    },
    "G_vuln_no_labels": {
        "desc": "Fingerprint-sliced, diff attrs stripped",
        "build": lambda pair: _strip_diff_attrs(pair.G_vuln),
    },
    "G_vuln": {
        "desc": "Fingerprint-sliced + diff labels + weights (current)",
        "build": lambda pair: pair.G_vuln,
    },
}


# ── experiment loop ──────────────────────────────────────────────────

def run_comparison(cfg: dict) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_slice_" + uuid.uuid4().hex[:6]
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # load data
    ds = AutoPatchDataset(cfg["data"]["autopatch"])
    pairs = ds.load_all()
    print(f"Loaded {len(pairs)} pairs")

    index_pairs, query_pairs, split_info = _build_split_plan(pairs, cfg)
    print(f"Index: {len(index_pairs)}  Query: {len(query_pairs)}")

    embedders = build_embedders(cfg)
    ks = [1, 5, 10]

    results = []

    for variant_name, variant_def in VARIANT_DEFS.items():
        print(f"\n{'='*60}")
        print(f"  Variant: {variant_name}  —  {variant_def['desc']}")
        print(f"{'='*60}")

        build_fn = variant_def["build"]

        for embedder in embedders:
            print(f"\n  [{embedder.name} / {variant_name}]")

            try:
                # embed index pairs
                t0 = time.perf_counter()
                index_graphs = [build_fn(p) for p in index_pairs]
                index_embeddings = embedder.embed_many(index_graphs)
                embed_time = time.perf_counter() - t0

                # check for degenerate embeddings (all-zero)
                norms = np.linalg.norm(index_embeddings, axis=1)
                n_zero = int(np.sum(norms < 1e-6))
                if n_zero == len(index_embeddings):
                    print(f"    SKIP — all {n_zero} embeddings are zero (no usable features)")
                    results.append({
                        "variant": variant_name, "embedder": embedder.name,
                        "hit@1": 0, "hit@5": 0, "hit@10": 0, "mrr": 0,
                        "cwe_recall": 0, "effective_dim": 0, "mean_pairwise_sim": 0,
                        "n": 0, "embed_time_s": round(embed_time, 1),
                        "error": f"all {n_zero} embeddings zero",
                    })
                    continue

                space_stats = embedding_space_stats(index_embeddings)
                eff_dim = space_stats.get("effective_dim", 0)
                if np.isnan(eff_dim):
                    eff_dim = 0.0
                    space_stats["effective_dim"] = 0.0
                print(f"    embed {embed_time:.1f}s  eff_dim={eff_dim:.1f}  "
                      f"mean_sim={space_stats['mean_pairwise_sim']:.3f}  "
                      f"({n_zero} zero vecs)")

                # build index
                idx_dir = run_dir / "indices"
                idx_dir.mkdir(exist_ok=True)
                stem = f"{embedder.name}__{variant_name}"
                index = HNSWIndex(
                    dim=embedder.dim,
                    index_path=str(idx_dir / f"{stem}__hnsw.index"),
                    metadata_path=str(idx_dir / f"{stem}__hnsw_meta.json"),
                )
                for pair, vec in zip(index_pairs, index_embeddings):
                    index.add(pair, vec, embedder.name)
                index.save()
                index.load()
                retriever = Retriever(index, top_k=max(ks))

                # self-retrieval — use same variant for queries
                query_graphs = [build_fn(p) for p in query_pairs]
                query_embeddings = embedder.embed_many(query_graphs)

                # evaluate: hit@k, MRR
                from collections import defaultdict as _dd
                hits = _dd(int)
                mrrs_list = []
                raw_queries = []
                n = 0
                for pidx, (pair, qvec) in enumerate(zip(query_pairs, query_embeddings)):
                    if np.linalg.norm(qvec) < 1e-6:
                        continue
                    res = retriever.query(qvec, top_k=max(ks))
                    for k in ks:
                        hits[k] += int(any(r['cve_id'] == pair.cve_id for r in res[:k]))
                    m = next(
                        (1.0 / (j + 1) for j, r in enumerate(res) if r['cve_id'] == pair.cve_id),
                        0.0
                    )
                    mrrs_list.append(m)
                    raw_queries.append({
                        'query_cve': pair.cve_id,
                        'query_cwe': pair.cwe_id,
                        'hit': m > 0,
                        'mrr': m,
                        'retrieved': [
                            {'rank': j+1, 'cve_id': r.get('cve_id'), 'cwe_id': r.get('cwe_id'), 'score': r.get('score')}
                            for j, r in enumerate(res)
                        ],
                    })
                    n += 1

                sr = {}
                if n > 0:
                    sr = {**{f'hit@{k}': hits[k] / n for k in ks}, 'mrr': float(np.mean(mrrs_list)), 'n': n, 'raw_queries': raw_queries}
                else:
                    sr = {'n': 0, 'raw_queries': []}
                hit1 = sr.get("hit@1", 0)
                hit5 = sr.get("hit@5", 0)
                mrr = sr.get("mrr", 0)
                n = sr.get("n", 0)

                # CWE recall — also use pre-computed embeddings
                cwe_support = _dd(int)
                for m in index.metadata:
                    cwe = m.get('cwe_id')
                    if cwe and cwe != 'UNKNOWN':
                        cwe_support[cwe] += 1

                per_cwe_scores = _dd(list)
                for pair, qvec in zip(query_pairs, query_embeddings):
                    cwe = pair.cwe_id
                    if not cwe or cwe == 'UNKNOWN':
                        continue
                    possible = min(max(ks), cwe_support.get(cwe, 0))
                    if possible <= 0:
                        continue
                    if np.linalg.norm(qvec) < 1e-6:
                        continue
                    res = retriever.query(qvec, top_k=max(ks))
                    same = sum(1 for r in res if r.get('cwe_id') == cwe)
                    per_cwe_scores[cwe].append(same / possible)

                cwe_recall = float(np.mean([np.mean(v) for v in per_cwe_scores.values()])) if per_cwe_scores else 0.0

                print(f"    hit@1={hit1:.3f}  hit@5={hit5:.3f}  MRR={mrr:.3f}  "
                      f"CWE_recall={cwe_recall:.3f}  n={n}")

            except Exception as e:
                print(f"    ERROR — {type(e).__name__}: {e}")
                hit1 = hit5 = mrr = cwe_recall = n = 0
                embed_time = 0
                space_stats = {"effective_dim": 0, "mean_pairwise_sim": 0}
                sr = {}

            results.append({
                "variant": variant_name,
                "embedder": embedder.name,
                "hit@1": round(hit1, 4),
                "hit@5": round(hit5, 4),
                "hit@10": round(sr.get("hit@10", 0), 4),
                "mrr": round(mrr, 4),
                "cwe_recall": round(cwe_recall, 4),
                "effective_dim": round(space_stats.get("effective_dim", 0), 1),
                "mean_pairwise_sim": round(space_stats.get("mean_pairwise_sim", 0), 4),
                "n": n,
                "embed_time_s": round(embed_time, 1),
            })

    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_info": split_info,
        "n_index": len(index_pairs),
        "n_query": len(query_pairs),
        "variants": list(VARIANT_DEFS.keys()),
        "embedders": [e.name for e in embedders],
        "results": results,
    }

    out_path = run_dir / "slicing_comparison.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nResults written to: {out_path}")

    _print_table(report)
    return report


def _print_table(report: dict):
    """Pretty-print comparison table."""
    results = report["results"]
    variants = report["variants"]
    embedders = report["embedders"]

    print(f"\n{'='*90}")
    print("  SLICING METHODOLOGY COMPARISON")
    print(f"{'='*90}")
    print(f"  Index: {report['n_index']}  |  Query: {report['n_query']}")
    print()

    # Per-embedder tables
    for emb in embedders:
        print(f"  ── {emb} {'─'*(70-len(emb))}")
        print(f"    {'Variant':<25s} {'hit@1':>7s} {'hit@5':>7s} {'MRR':>7s} {'CWE':>7s} {'eff_dim':>8s}")
        print(f"    {'─'*62}")
        for v in variants:
            row = next((r for r in results if r["variant"] == v and r["embedder"] == emb), None)
            if row:
                print(f"    {v:<25s} {row['hit@1']:>6.1%} {row['hit@5']:>6.1%} "
                      f"{row['mrr']:>7.3f} {row['cwe_recall']:>6.1%} {row['effective_dim']:>8.1f}")
        print()

    # Summary: best variant per embedder
    print(f"  ── Summary: hit@1 by variant {'─'*40}")
    header = f"    {'Embedder':<20s}" + "".join(f"{v:>22s}" for v in variants)
    print(header)
    for emb in embedders:
        cells = []
        for v in variants:
            row = next((r for r in results if r["variant"] == v and r["embedder"] == emb), None)
            cells.append(f"{row['hit@1']:>6.1%}" if row else "   N/A")
        print(f"    {emb:<20s}" + "".join(f"{c:>22s}" for c in cells))
    print(f"\n{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare slicing methodologies")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_comparison(cfg)


if __name__ == "__main__":
    main()
