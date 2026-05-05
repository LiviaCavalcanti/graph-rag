#!/usr/bin/env python3
"""
Compare slicing methodologies with and without diff labelling.

Four conditions (2×2 factorial: slicing × labelling):
  1. G_before                  — full pre-patch graph, no diff info  (≈ old ID-based)
  2. G_before_labeled          — full pre-patch graph + real diff labels from G_vuln
  3. G_vuln_no_labels          — fingerprint-sliced, diff attrs stripped
  4. G_vuln                    — fingerprint-sliced + diff labels + weights (current)

For each condition x embedder, runs self-retrieval (hit@1/5/10, MRR)
and CWE recall, then prints a comparison table.

Query variant:
  By default, queries use the same graph variant as the index.
  Use --query-variant to fix queries to a specific variant (e.g.
  --query-variant G_before).  The special value "runner_compat"
  reproduces the checkpoint-2 protocol where augmented queries
  use G_before and originals use G_vuln.

Usage:
    python -m experiments.slicing_comparison [--config config.yaml]
    python -m experiments.slicing_comparison --query-variant runner_compat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (build_hnsw, build_split, evaluate_cwe_recall,
                                evaluate_retrieval, load_config,
                                make_run_dir, save_json)
from src.data.autopatch import load_pairs
from src.embeddings import build_embedders
from src.metrics.metrics import embedding_space_stats

# ── graph variant factories ──────────────────────────────────────────


def _strip_diff_attrs(G):
    """Return a Graph copy with diff/diff_weight removed from all nodes."""
    G2 = G.copy()
    for n in G2:
        G2.nodes[n].pop("diff", None)
        G2.nodes[n].pop("diff_weight", None)
    return G2


def _add_labels_from_vuln(pair):
    """Full G_before with real diff labels transferred from G_vuln."""
    G2 = pair.G_before.copy()
    for n in G2:
        if n in pair.G_vuln.nodes:
            G2.nodes[n]["diff"] = pair.G_vuln.nodes[n].get("diff", "context")
            G2.nodes[n]["diff_weight"] = pair.G_vuln.nodes[n].get("diff_weight", 0.2)
        else:
            G2.nodes[n]["diff"] = "context"
            G2.nodes[n]["diff_weight"] = 0.2
    return G2


VARIANT_DEFS = {
    "G_before": {
        "desc": "Full pre-patch graph, no diff (≈ old ID-based)",
        "build": lambda pair: _strip_diff_attrs(pair.G_before),
    },
    "G_before_labeled": {
        "desc": "Full pre-patch graph + real diff labels from G_vuln",
        "build": lambda pair: _add_labels_from_vuln(pair),
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


# ── query variant helpers ────────────────────────────────────────────


def _runner_compat_query_graph(pair):
    """Reproduce checkpoint-2 runner.py query protocol:
    all queries use G_vuln (the diff graph)."""
    return pair.G_vuln


def _resolve_query_build_fn(query_variant: str | None, index_build_fn):
    """Return (build_fn, label) for query graphs."""
    if query_variant is None:
        return index_build_fn, None  # same as index
    # Special case for runner compatibility mode to reproduce checkpoint2 results where the query is always G_before (autopatch)
    if query_variant == "runner_compat":
        return _runner_compat_query_graph, "runner_compat"
    if query_variant in VARIANT_DEFS:
        return VARIANT_DEFS[query_variant]["build"], query_variant
    raise ValueError(
        f"Unknown --query-variant '{query_variant}'. "
        f"Choose from {list(VARIANT_DEFS)} or 'runner_compat'."
    )


# ── experiment loop ──────────────────────────────────────────────────


def run_comparison(cfg: dict, *, query_variant: str | None = None) -> dict:
    run_id, run_dir = make_run_dir("slice")

    # load data
    pairs = load_pairs(cfg)
    print(f"Loaded {len(pairs)} pairs")

    index_pairs, query_pairs, split_info = build_split(pairs, cfg)
    print(f"Index: {len(index_pairs)}  Query: {len(query_pairs)}")
    if query_variant:
        print(f"Query variant override: {query_variant}")

    embedders = build_embedders(cfg)
    ks = [1, 5, 10]

    results = []

    for variant_name, variant_def in VARIANT_DEFS.items():
        print(f"\n{'='*60}")
        print(f"  Variant: {variant_name}  —  {variant_def['desc']}")
        print(f"{'='*60}")

        # Reset PCA state so each variant gets its own projection
        for emb in embedders:
            if hasattr(emb, "_fitted"):
                emb._fitted = False
                emb._pca = None

        build_fn = variant_def["build"]
        q_build_fn, q_label = _resolve_query_build_fn(query_variant, build_fn)

        for embedder in embedders:
            label = f"{variant_name}→{q_label}" if q_label else variant_name
            print(f"\n  [{embedder.name} / {label}]")

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
                    print(
                        f"    SKIP — all {n_zero} embeddings are zero (no usable features)"
                    )
                    results.append(
                        {
                            "variant": variant_name,
                            "embedder": embedder.name,
                            "query_variant": q_label,
                            "hit@1": 0,
                            "hit@5": 0,
                            "hit@10": 0,
                            "mrr": 0,
                            "cve_precision": 0,
                            "cve_recall": 0,
                            "cve_f1": 0,
                            "cwe_recall": 0,
                            "effective_dim": 0,
                            "mean_pairwise_sim": 0,
                            "n": 0,
                            "embed_time_s": round(embed_time, 1),
                            "error": f"all {n_zero} embeddings zero",
                        }
                    )
                    continue

                space_stats = embedding_space_stats(index_embeddings)
                eff_dim = space_stats.get("effective_dim", 0)
                if np.isnan(eff_dim):
                    eff_dim = 0.0
                    space_stats["effective_dim"] = 0.0
                print(
                    f"    embed {embed_time:.1f}s  eff_dim={eff_dim:.1f}  "
                    f"mean_sim={space_stats['mean_pairwise_sim']:.3f}  "
                    f"({n_zero} zero vecs)"
                )

                # build index & retriever via common
                tag = f"{embedder.name}__{variant_name}"
                index, retriever = build_hnsw(
                    index_pairs,
                    index_embeddings,
                    embedder.name,
                    embedder.dim,
                    run_dir,
                    tag=tag,
                )

                # embed queries (may differ from index variant)
                query_graphs = [q_build_fn(p) for p in query_pairs]
                query_embeddings = embedder.embed_many(query_graphs)

                # evaluate via common
                sr = evaluate_retrieval(
                    query_pairs,
                    query_embeddings,
                    retriever,
                    index_pairs,
                    ks=ks,
                )
                cwe_result = evaluate_cwe_recall(
                    query_pairs,
                    query_embeddings,
                    retriever,
                    index.metadata,
                    top_k=max(ks),
                )
                cwe_recall = cwe_result["macro_avg"]

                hit1 = sr.get("hit@1", 0)
                hit5 = sr.get("hit@5", 0)
                mrr = sr.get("mrr", 0)
                n = sr.get("n", 0)

                print(
                    f"    hit@1={hit1:.3f}  hit@5={hit5:.3f}  MRR={mrr:.3f}  "
                    f"CVE_F1={sr.get('cve_f1', 0):.3f}  "
                    f"CWE_recall={cwe_recall:.3f}  n={n}"
                )

            except Exception as e:
                print(f"    ERROR — {type(e).__name__}: {e}")
                hit1 = hit5 = mrr = cwe_recall = n = 0
                embed_time = 0
                space_stats = {"effective_dim": 0, "mean_pairwise_sim": 0}
                sr = {}

            results.append(
                {
                    "variant": variant_name,
                    "embedder": embedder.name,
                    "query_variant": q_label,
                    "hit@1": round(hit1, 4),
                    "hit@5": round(hit5, 4),
                    "hit@10": round(sr.get("hit@10", 0), 4),
                    "mrr": round(mrr, 4),
                    "cve_precision": round(sr.get("cve_precision", 0), 4),
                    "cve_recall": round(sr.get("cve_recall", 0), 4),
                    "cve_f1": round(sr.get("cve_f1", 0), 4),
                    "cwe_recall": round(cwe_recall, 4),
                    "effective_dim": round(space_stats.get("effective_dim", 0), 1),
                    "mean_pairwise_sim": round(
                        space_stats.get("mean_pairwise_sim", 0), 4
                    ),
                    "n": n,
                    "embed_time_s": round(embed_time, 1),
                }
            )

    report = {
        "run_id": run_id,
        "split_info": split_info,
        "n_index": len(index_pairs),
        "n_query": len(query_pairs),
        "query_variant": query_variant,
        "variants": list(VARIANT_DEFS.keys()),
        "embedders": [e.name for e in embedders],
        "results": results,
    }

    save_json(report, run_dir / "slicing_comparison.json")
    print(f"\nResults written to: {run_dir / 'slicing_comparison.json'}")

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
        print(
            f"    {'Variant':<25s} {'hit@1':>7s} {'hit@5':>7s} {'MRR':>7s} {'CWE':>7s} {'eff_dim':>8s}"
        )
        print(f"    {'─'*62}")
        for v in variants:
            row = next(
                (r for r in results if r["variant"] == v and r["embedder"] == emb), None
            )
            if row:
                print(
                    f"    {v:<25s} {row['hit@1']:>6.1%} {row['hit@5']:>6.1%} "
                    f"{row['mrr']:>7.3f} {row['cwe_recall']:>6.1%} {row['effective_dim']:>8.1f}"
                )
        print()

    # Summary: best variant per embedder
    print(f"  ── Summary: hit@1 by variant {'─'*40}")
    header = f"    {'Embedder':<20s}" + "".join(f"{v:>22s}" for v in variants)
    print(header)
    for emb in embedders:
        cells = []
        for v in variants:
            row = next(
                (r for r in results if r["variant"] == v and r["embedder"] == emb), None
            )
            cells.append(f"{row['hit@1']:>6.1%}" if row else "   N/A")
        print(f"    {emb:<20s}" + "".join(f"{c:>22s}" for c in cells))
    print(f"\n{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare slicing methodologies")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument(
        "--query-variant",
        default=None,
        help="Fix query graphs to this variant (e.g. G_before, G_vuln, "
        "runner_compat).  Default: same as index variant.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_comparison(cfg, query_variant=args.query_variant)


if __name__ == "__main__":
    main()
