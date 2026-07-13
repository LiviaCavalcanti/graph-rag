#!/usr/bin/env python3
"""
Slice-depth sweep — retrieval quality vs. compute_graph_diff ``slice_depth``.

Sweeps the program-slice depth used to build ``G_vuln`` (the vulnerability
diff subgraph) and measures self-retrieval (hit@1/5/10, MRR) and CWE recall
for every ``slice_depth x embedder`` cell.

``G_vuln`` is recomputed from each pair's ``G_before``/``G_after`` at every
depth via ``src.data.pipeline.compute_graph_diff(..., slice_depth=d)``, so no
re-export or re-parsing of CPGs is needed — only the diff/slice is recomputed
(cheap, in-memory).  The train/query split is built once so all depths are
compared on identical index/query sets.

This complements ``slicing_comparison.py`` (which fixes the depth and varies
the graph *variant*); here we fix the variant to ``G_vuln`` and vary the depth.

Usage:
    uv run python -m experiments.exp.graph_slicing.scripts.slice_depth_sweep \
        --graphml-root graphml_selected_cves_method \
        [--config config.yaml] [--depths 1 2 3 4] [--embedders wl gin combined] \
        [--target-cwes CWE-476 CWE-787] [--max-per-cwe 20] [--sample-limit 300]

Note: ``--graphml-root`` must point at a METHOD-level CPG root (dirs named
``CVE-..._m<method_change_id>``, e.g. ``graphml_selected_cves_method``); the
default ``config.yaml`` root is file-level and will not match the loader.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Ensure the repo root is importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.common import (
    build_hnsw,
    evaluate_cwe_recall,
    evaluate_retrieval,
    load_config,
    make_run_dir,
    save_json,
)
from src.data import load_pairs
from src.data.pipeline import compute_graph_diff, graph_diff_params
from src.embeddings import build_embedders
from src.metrics.metrics import embedding_space_stats


def _diff_at_depth(pairs: list, depth: int, base_params: dict) -> list:
    """Recompute G_vuln for every pair at the given slice depth.

    Other diff parameters (change_weight, noise_types, flow_edges) come from
    ``base_params`` (the config ``graph:`` section), so only ``slice_depth``
    varies.  Falls back to the pre-computed ``G_vuln`` if a diff fails.
    """
    params = {**base_params, "slice_depth": depth}
    graphs = []
    for p in pairs:
        try:
            graphs.append(compute_graph_diff(p.G_before, p.G_after, **params))
        except Exception:
            graphs.append(p.G_vuln)
    return graphs


def _zero_result(depth: int, embedder_name: str, error: str) -> dict:
    return {
        "slice_depth": depth,
        "embedder": embedder_name,
        "hit@1": 0,
        "hit@5": 0,
        "hit@10": 0,
        "mrr": 0,
        "cve_f1": 0,
        "cwe_recall": 0,
        "effective_dim": 0,
        "mean_pairwise_sim": 0,
        "n": 0,
        "error": error,
    }


def _sample_per_cwe(pairs: list, max_per_cwe: int, seed: int) -> list:
    """Cap the number of pairs per CWE (keeps the sweep small/fast)."""
    rng = random.Random(seed)
    by_cwe: dict[str, list] = defaultdict(list)
    for p in pairs:
        by_cwe[getattr(p, "cwe_id", "") or "UNKNOWN"].append(p)
    out: list = []
    for cwe in sorted(by_cwe):
        items = by_cwe[cwe][:]
        rng.shuffle(items)
        out.extend(items[:max_per_cwe])
    rng.shuffle(out)
    return out


def _cve_aware_split(
    pairs: list, *, test_ratio: float, seed: int, min_pairs_per_cve: int = 2
) -> tuple[list, list, dict]:
    """Split so every query CVE keeps >= 1 support entry in the index.

    Pure-CVEfixes graphs need same-CVE support in the index, otherwise a query
    can have no possible positive (leakage / impossible retrieval).
    """
    rng = random.Random(seed)
    ratio = max(0.0, min(0.9, float(test_ratio)))
    by_cve: dict[str, list] = defaultdict(list)
    for p in pairs:
        by_cve[getattr(p, "cve_id", "") or "UNKNOWN"].append(p)

    index_pairs: list = []
    query_pairs: list = []
    for cve_id in sorted(by_cve):
        items = by_cve[cve_id][:]
        rng.shuffle(items)
        if len(items) >= min_pairs_per_cve:
            n_query = max(1, min(len(items) - 1, int(round(len(items) * ratio))))
            query_pairs.extend(items[:n_query])
            index_pairs.extend(items[n_query:])
        else:
            index_pairs.extend(items)

    index_cves = {p.cve_id for p in index_pairs}
    query_pairs = [p for p in query_pairs if p.cve_id in index_cves]

    rng.shuffle(index_pairs)
    rng.shuffle(query_pairs)
    split_info = {
        "enabled": True,
        "seed": seed,
        "test_ratio": ratio,
        "strategy": "cve_aware_same_cve_support",
        "n_index": len(index_pairs),
        "n_query": len(query_pairs),
    }
    return index_pairs, query_pairs, split_info


def run_sweep(
    cfg: dict,
    *,
    depths: list[int],
    embedder_names: list[str] | None = None,
    graphml_root: str | None = None,
    db_path: str | None = None,
    target_cwes: list[str] | None = None,
    max_per_cwe: int = 0,
    sample_limit: int | None = None,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> dict:
    """Run the slice-depth × embedder retrieval sweep."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("data", {}).setdefault("cvefixes", {})
    cfg["data"]["active"] = ["cvefixes"]
    if graphml_root:
        cfg["data"]["cvefixes"]["graphml_root"] = graphml_root
    if db_path:
        cfg["data"]["cvefixes"]["db_path"] = db_path
    if target_cwes is not None:
        cfg["data"]["cvefixes"]["target_cwes"] = target_cwes
    if sample_limit is not None:
        cfg["data"]["cvefixes"]["sample_limit"] = sample_limit
    if embedder_names:
        cfg["embeddings"] = {**cfg["embeddings"], "active": embedder_names}

    run_id, run_dir = make_run_dir("slice_sweep")

    pairs = load_pairs(cfg)
    # Keep only pairs with a non-empty vulnerability slice (a detectable diff).
    pairs = [
        p
        for p in pairs
        if getattr(p, "G_vuln", None) is not None and p.G_vuln.number_of_nodes() > 0
    ]
    if max_per_cwe > 0:
        pairs = _sample_per_cwe(pairs, max_per_cwe, seed)
    print(f"Usable pairs: {len(pairs)}")

    index_pairs, query_pairs, split_info = _cve_aware_split(
        pairs, test_ratio=test_ratio, seed=seed
    )
    print(
        f"Index: {len(index_pairs)}  Query: {len(query_pairs)}  "
        f"(index CWE dist={dict(Counter(p.cwe_id for p in index_pairs))})"
    )

    embedders = build_embedders(cfg)

    # Base diff parameters from config (change_weight/noise_types/flow_edges);
    # slice_depth is the swept dimension, so drop it from the base.
    base_params = graph_diff_params(cfg)
    base_params.pop("slice_depth", None)

    ks = cfg.get("experiment", {}).get("ks", [1, 5, 10])

    results: list[dict] = []

    for depth in depths:
        print(f"\n{'=' * 60}")
        print(f"  slice_depth = {depth}")
        print(f"{'=' * 60}")

        # Re-diff once per depth (embedder-independent), reused across embedders.
        t0 = time.perf_counter()
        index_graphs = _diff_at_depth(index_pairs, depth, base_params)
        query_graphs = _diff_at_depth(query_pairs, depth, base_params)
        print(
            f"  re-diff {len(index_pairs)}+{len(query_pairs)} graphs in "
            f"{time.perf_counter() - t0:.1f}s"
        )

        for embedder in embedders:
            # Fresh PCA projection per (depth, embedder) cell — otherwise a
            # projection fitted at one depth would leak into another.
            if hasattr(embedder, "_fitted"):
                embedder._fitted = False
                embedder._pca = None

            try:
                index_embeddings = embedder.embed_many(index_graphs)
                norms = np.linalg.norm(index_embeddings, axis=1)
                if int(np.sum(norms < 1e-6)) == len(index_embeddings):
                    print(f"  [{embedder.name}] SKIP — all embeddings zero")
                    results.append(
                        _zero_result(depth, embedder.name, "all embeddings zero")
                    )
                    continue

                stats = embedding_space_stats(index_embeddings)
                eff_dim = stats.get("effective_dim", 0)
                if np.isnan(eff_dim):
                    eff_dim = 0.0
                    stats["effective_dim"] = 0.0

                tag = f"{embedder.name}__d{depth}"
                index, retriever = build_hnsw(
                    index_pairs,
                    index_embeddings,
                    embedder.name,
                    embedder.dim,
                    run_dir,
                    tag=tag,
                )

                query_embeddings = embedder.embed_many(query_graphs)
                sr = evaluate_retrieval(
                    query_pairs,
                    query_embeddings,
                    retriever,
                    index_pairs,
                    ks=ks,
                )
                cwe_recall = evaluate_cwe_recall(
                    query_pairs,
                    query_embeddings,
                    retriever,
                    index.metadata,
                    top_k=max(ks),
                )["macro_avg"]

                print(
                    f"  [{embedder.name}]  hit@1={sr.get('hit@1', 0):.3f}  "
                    f"hit@5={sr.get('hit@5', 0):.3f}  MRR={sr.get('mrr', 0):.3f}  "
                    f"CWE={cwe_recall:.3f}  n={sr.get('n', 0)}"
                )

                results.append(
                    {
                        "slice_depth": depth,
                        "embedder": embedder.name,
                        "hit@1": round(sr.get("hit@1", 0), 4),
                        "hit@5": round(sr.get("hit@5", 0), 4),
                        "hit@10": round(sr.get("hit@10", 0), 4),
                        "mrr": round(sr.get("mrr", 0), 4),
                        "cve_f1": round(sr.get("cve_f1", 0), 4),
                        "cwe_recall": round(cwe_recall, 4),
                        "effective_dim": round(eff_dim, 1),
                        "mean_pairwise_sim": round(
                            stats.get("mean_pairwise_sim", 0), 4
                        ),
                        "n": sr.get("n", 0),
                    }
                )
            except Exception as e:
                print(f"  [{embedder.name}] ERROR — {type(e).__name__}: {e}")
                results.append(
                    _zero_result(depth, embedder.name, f"{type(e).__name__}: {e}")
                )

    report = {
        "run_id": run_id,
        "split_info": split_info,
        "n_index": len(index_pairs),
        "n_query": len(query_pairs),
        "depths": list(depths),
        "embedders": [e.name for e in embedders],
        "results": results,
    }

    save_json(report, run_dir / "slice_depth_sweep.json")
    print(f"\nResults written to: {run_dir / 'slice_depth_sweep.json'}")

    _print_table(report)
    return report


def _print_table(report: dict) -> None:
    """Pretty-print the sweep results, one block per embedder."""
    results = report["results"]
    depths = report["depths"]
    embedders = report["embedders"]

    print(f"\n{'=' * 72}")
    print("  SLICE-DEPTH SWEEP")
    print(f"{'=' * 72}")
    print(f"  Index: {report['n_index']}  |  Query: {report['n_query']}")

    for emb in embedders:
        print(f"\n  ── {emb} {'─' * max(0, 60 - len(emb))}")
        print(
            f"    {'depth':>6s} {'hit@1':>7s} {'hit@5':>7s} {'MRR':>7s} "
            f"{'CWE':>7s} {'eff_dim':>8s}"
        )
        print(f"    {'─' * 46}")
        for d in depths:
            row = next(
                (r for r in results if r["slice_depth"] == d and r["embedder"] == emb),
                None,
            )
            if row:
                print(
                    f"    {d:>6d} {row['hit@1']:>6.1%} {row['hit@5']:>6.1%} "
                    f"{row['mrr']:>7.3f} {row['cwe_recall']:>6.1%} "
                    f"{row['effective_dim']:>8.1f}"
                )
    print(f"\n{'=' * 72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep compute_graph_diff slice_depth for retrieval quality."
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=None,
        help="Slice depths to sweep (default: experiment.slice_depths or 1 2 3).",
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=None,
        help="Embedder names to evaluate (default: embeddings.active).",
    )
    parser.add_argument(
        "--graphml-root",
        default=None,
        help="Override data.cvefixes.graphml_root. Must be a METHOD-level root "
        "(dirs named CVE-..._m<id>), e.g. graphml_selected_cves_method.",
    )
    parser.add_argument(
        "--db-path", default=None, help="Override data.cvefixes.db_path."
    )
    parser.add_argument(
        "--target-cwes",
        nargs="*",
        default=None,
        help="CWE filter, e.g. CWE-476 CWE-787 (empty list = all CWEs).",
    )
    parser.add_argument(
        "--max-per-cwe",
        type=int,
        default=0,
        help="Cap pairs per CWE after loading (0 = no cap).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Cap DB rows scanned via SQL LIMIT (useful for quick runs).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument(
        "--test-ratio", type=float, default=0.2, help="Query fraction per CVE."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    depths = args.depths or cfg.get("experiment", {}).get("slice_depths", [1, 2, 3])
    run_sweep(
        cfg,
        depths=depths,
        embedder_names=args.embedders,
        graphml_root=args.graphml_root,
        db_path=args.db_path,
        target_cwes=args.target_cwes,
        max_per_cwe=args.max_per_cwe,
        sample_limit=args.sample_limit,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
