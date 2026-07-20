#!/usr/bin/env python3
"""
CVEfixes index-distribution study (file-level, shared query set).

════════════════════════════════════════════════════════════════════════
GOAL
════════════════════════════════════════════════════════════════════════
Isolate the effect of the knowledge-base (index) CWE distribution on
retrieval. A single SHARED query set is held constant and only the INDEX
distribution is varied:

  - proportional : the index preserves the dataset's natural CWE distribution
  - balanced     : the index is ~uniform across CWEs

Both indices:
  * have the SAME size,
  * fully support the shared query set (>= 1 same-CVE index entry per query),
  * share an identical per-query "support" backbone,

so the ONLY variable across the two runs is the index CWE distribution.
Any metric delta is therefore attributable to distribution alone.

════════════════════════════════════════════════════════════════════════
PROTOCOL
════════════════════════════════════════════════════════════════════════
  1. Load file-level pairs and perform the leakage-safe CWE→CVE split ONCE
     (reuses the file-level experiment loader). This yields a fixed query
     set and an index pool.
  2. Reserve one index pair per query CVE (the shared support backbone).
  3. Build two equal-size indices from the pool that both contain the
     support backbone but shape the remaining budget to proportional vs.
     balanced CWE targets.
  4. For each embedder, evaluate the SAME queries against each index.
  5. Emit per-variant results.json (+ dashboards) and a comparison.json
     with proportional-vs-balanced deltas.

════════════════════════════════════════════════════════════════════════
USAGE
════════════════════════════════════════════════════════════════════════
    python -m cvefixes_experiments.scripts.pipeline_verification.exp_cvefixes_index_distribution \
      --config config.yaml \
      --embedders codebert_seq codebert_pattern combined gin \
      --index-total 140
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cvefixes_experiments.scripts.performance.exp_cvefixes_retrieval_grid_file_level import (
    CVEFixesFileLevelRetrievalExperiment, _normalize_embedder_name,
    _prepare_cfg, _sync_top_level_split)
from src.embeddings import REGISTRY as EMBEDDER_REGISTRY
from src.metrics.metrics import embedding_space_stats
from src.metrics.retrieval_eval import (cve_retrieval_metrics,
                                        cwe_recall_metrics, retrieve_all)
from src.rag.hnsw import HNSWIndex
from src.rag.utils import populate_index

# ── Index construction (shared support + shaped distribution) ────────


def _pick_support(index_pool: list, query_cves: set, seed: int) -> list:
    """Reserve one index pair per query CVE.

    Guarantees the shared query set is answerable in EVERY index variant.
    The leakage-safe split ensures each query CVE has at least one pair in
    the index pool, so every query CVE is covered.
    """
    rng = random.Random(seed)
    by_cve: dict[str, list] = defaultdict(list)
    for pair in index_pool:
        by_cve[pair.cve_id].append(pair)

    support: list = []
    for cve in sorted(query_cves):
        group = by_cve.get(cve)
        if group:
            group = group[:]
            rng.shuffle(group)
            support.append(group[0])
    return support


def _build_index_variant(
    index_pool: list,
    support: list,
    *,
    mode: str,
    total: int,
    seed: int,
) -> tuple[list, dict[str, int]]:
    """Build an index of exactly ``total`` pairs, shaped to ``mode`` by CWE.

    The returned index always contains the shared ``support`` backbone; the
    remaining budget is filled to hit per-CWE targets (uniform for
    ``balanced``, natural proportions for ``proportional``). Size is forced
    to exactly ``total`` by trimming non-support surplus or filling deficit.
    """
    rng = random.Random(seed)
    support_ids = {id(p) for p in support}
    cwes = sorted({p.cwe_id for p in index_pool}) or ["UNKNOWN"]
    n_cwes = len(cwes)

    rest_by_cwe: dict[str, list] = defaultdict(list)
    for pair in index_pool:
        if id(pair) not in support_ids:
            rest_by_cwe[pair.cwe_id].append(pair)
    for cwe in cwes:
        rng.shuffle(rest_by_cwe[cwe])

    support_by_cwe = Counter(p.cwe_id for p in support)
    pool_by_cwe = Counter(p.cwe_id for p in index_pool)
    pool_total = sum(pool_by_cwe.values()) or 1

    total = max(len(support), min(int(total), len(index_pool)))

    # Per-CWE target for the FULL index (support + extra).
    if mode == "balanced":
        base, rem = divmod(total, n_cwes)
        target = {c: base + (1 if i < rem else 0) for i, c in enumerate(cwes)}
    else:  # proportional
        target = {c: int(round(total * pool_by_cwe[c] / pool_total)) for c in cwes}

    # Start from the shared support, then add non-support up to target.
    selected = list(support)
    added_by_cwe: dict[str, int] = {}
    for cwe in cwes:
        want = max(0, target[cwe] - support_by_cwe.get(cwe, 0))
        take = min(want, len(rest_by_cwe[cwe]))
        added_by_cwe[cwe] = take
        selected.extend(rest_by_cwe[cwe][:take])

    # Force size to exactly ``total``.
    diff = len(selected) - total
    if diff > 0:
        # Remove non-support pairs from the most over-target CWEs first.
        cur = Counter(p.cwe_id for p in selected)
        removable = [p for p in selected if id(p) not in support_ids]
        removable.sort(key=lambda p: target[p.cwe_id] - cur[p.cwe_id])
        drop = {id(p) for p in removable[:diff]}
        selected = [p for p in selected if id(p) not in drop]
    elif diff < 0:
        leftovers: list = []
        for cwe in cwes:
            leftovers.extend(rest_by_cwe[cwe][added_by_cwe[cwe] :])
        if mode == "proportional":
            leftovers.sort(key=lambda p: -pool_by_cwe[p.cwe_id])
        else:
            rng.shuffle(leftovers)
        selected.extend(leftovers[:-diff])

    per_cwe = dict(Counter(p.cwe_id for p in selected))
    return selected, per_cwe


# ── Per-variant retrieval ────────────────────────────────────────────


def run_variant_retrieval(
    variant: str,
    index_pairs: list,
    query_pairs: list,
    *,
    embedders: list[str],
    emb_cfg: dict,
    ks: list[int],
    out_dir: Path,
) -> list[dict]:
    """Run the HNSW retrieval grid for one index variant."""
    variant_dir = out_dir / variant
    index_dir = variant_dir / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n── index={variant}  n_index={len(index_pairs)}  n_query={len(query_pairs)} ──"
    )
    cells: list[dict] = []

    for emb_name in embedders:
        if emb_name not in EMBEDDER_REGISTRY:
            print(f"  WARNING: unknown embedder '{emb_name}', skipping")
            continue
        embedder = EMBEDDER_REGISTRY[emb_name](emb_cfg)

        t0 = time.perf_counter()
        index_embeddings = embedder.embed_many([p.G_vuln for p in index_pairs])
        embed_time = time.perf_counter() - t0

        norms = np.linalg.norm(index_embeddings, axis=1)
        n_index_degenerate = int(np.sum(norms < 1e-6))
        if index_embeddings.shape[0] and n_index_degenerate == len(index_embeddings):
            print(f"  [{emb_name}] all-zero index embeddings, skipping")
            continue

        space_stats = embedding_space_stats(index_embeddings)
        index = HNSWIndex(
            dim=index_embeddings.shape[1],
            index_path=str(index_dir / f"{embedder.name}__hnsw.index"),
            metadata_path=str(index_dir / f"{embedder.name}__hnsw_meta.json"),
        )
        retriever = populate_index(
            index, index_pairs, index_embeddings, embedder.name, top_k=max(ks)
        )

        qr = retrieve_all(query_pairs, embedder, retriever, top_k=max(ks))
        n_query_degenerate = sum(1 for _pair, results in qr if not results)

        cve_m = cve_retrieval_metrics(qr, ks=ks, index_metadata=index.metadata)
        cwe_m = cwe_recall_metrics(qr, index.metadata, top_k=max(ks))

        print(
            f"  [{emb_name:<16}] hit@1={cve_m.get('hit@1',0):.3f} "
            f"hit@5={cve_m.get('hit@5',0):.3f} mrr={cve_m.get('mrr',0):.3f} "
            f"cwe_recall={cwe_m.get('macro_avg',0):.3f} "
            f"q_dropped={n_query_degenerate}/{len(query_pairs)}"
        )

        cells.append(
            {
                "embedder": embedder.name,
                "backend": "hnsw",
                "graph_variant": "G_vuln",
                "index_distribution": variant,
                "n_samples": len(index_pairs),
                "n_query": len(query_pairs),
                "n_query_degenerate": n_query_degenerate,
                "n_index_degenerate": n_index_degenerate,
                "query_degenerate_rate": (
                    round(n_query_degenerate / len(query_pairs), 4)
                    if query_pairs
                    else 0.0
                ),
                "embed_time_s": round(embed_time, 2),
                "space_stats": space_stats,
                "cve_retrieval": cve_m,
                "cwe_recall": cwe_m,
                "leave_one_out": {},
            }
        )

    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "results.json").write_text(
        json.dumps(
            {
                "run_id": f"index_distribution_{variant}",
                "description": f"Shared-query index-distribution study — {variant} index",
                "cells": cells,
            },
            indent=2,
            default=str,
        )
    )

    try:
        from experiments.dashboard_scripts.dashboard import \
            generate_html_dashboard

        generate_html_dashboard(str(variant_dir))
        print(f"  Dashboard -> {variant_dir / 'dashboard.html'}")
    except Exception as exc:  # pragma: no cover - dashboard is best-effort
        print(f"  Dashboard generation skipped ({variant}): {exc}")

    return cells


# ── Comparison ───────────────────────────────────────────────────────


def _build_comparison(prop_cells, bal_cells, *, ks, n_query, n_index, dist, embedders):
    by = {
        "proportional": {c["embedder"]: c for c in prop_cells},
        "balanced": {c["embedder"]: c for c in bal_cells},
    }
    metrics = [f"hit@{k}" for k in ks] + ["mrr"]
    deltas = []
    for emb in embedders:
        p = by["proportional"].get(emb)
        b = by["balanced"].get(emb)
        if not p or not b:
            continue
        for metric in metrics:
            pv = p["cve_retrieval"].get(metric, 0.0)
            bv = b["cve_retrieval"].get(metric, 0.0)
            deltas.append(
                {
                    "embedder": emb,
                    "metric": metric,
                    "proportional": round(pv, 4),
                    "balanced": round(bv, 4),
                    "delta_balanced_minus_proportional": round(bv - pv, 4),
                }
            )
        pc = p["cwe_recall"].get("macro_avg", 0.0)
        bc = b["cwe_recall"].get("macro_avg", 0.0)
        deltas.append(
            {
                "embedder": emb,
                "metric": "cwe_recall",
                "proportional": round(pc, 4),
                "balanced": round(bc, 4),
                "delta_balanced_minus_proportional": round(bc - pc, 4),
            }
        )
    return {
        "run_id": "index_distribution_study",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Retrieval vs. index CWE distribution on a shared query set",
        "config": {"ks": ks, "embedders": embedders},
        "dataset_info": {
            "n_query_shared": n_query,
            "n_index_each": n_index,
            "index_cwe_distribution": dist,
        },
        "proportional_cells": prop_cells,
        "balanced_cells": bal_cells,
        "deltas": deltas,
    }


def _print_comparison(comparison: dict) -> None:
    print("\n" + "=" * 72)
    print("INDEX DISTRIBUTION  (delta = balanced - proportional)")
    print("=" * 72)
    di = comparison["dataset_info"]
    print(
        f"  shared query set: {di['n_query_shared']}   index size (each): {di['n_index_each']}"
    )
    print(
        f"  proportional index CWE dist: {di['index_cwe_distribution'].get('proportional')}"
    )
    print(
        f"  balanced     index CWE dist: {di['index_cwe_distribution'].get('balanced')}"
    )
    print("  " + "-" * 60)
    print(f"  {'embedder':<16} {'metric':<12} {'prop':>8} {'bal':>8} {'delta':>8}")
    for d in comparison["deltas"]:
        dv = d["delta_balanced_minus_proportional"]
        arrow = "▲" if dv > 0 else ("▼" if dv < 0 else "=")
        print(
            f"  {d['embedder']:<16} {d['metric']:<12} "
            f"{d['proportional']:>8.3f} {d['balanced']:>8.3f} {dv:>+8.3f} {arrow}"
        )


# ── Orchestration ────────────────────────────────────────────────────


def run_study(
    *,
    config_path: str = "config.yaml",
    output_dir: str = "cvefixes_experiments/output/index_distribution_study",
    graphml_root: str = "graphml_fixedcves_file",
    db_path: str = "data/cvefixes/CVEfixes.db",
    embedders: list[str] | None = None,
    target_cwes: list[str] | None = None,
    ks: list[int] | None = None,
    index_total: int = 140,
    seed: int = 42,
) -> dict:
    embedders = [
        _normalize_embedder_name(n)
        for n in (embedders or ["codebert_seq", "codebert_pattern", "combined", "gin"])
    ]
    unknown = [n for n in embedders if n not in EMBEDDER_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown embedders: {unknown}. Available: {sorted(EMBEDDER_REGISTRY)}"
        )
    ks = ks or [1, 5, 10]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _prepare_cfg(
        cfg,
        db_path=db_path,
        graphml_root=graphml_root,
        embedders=embedders,
        target_cwes=target_cwes,
        ks=ks,
    )
    _sync_top_level_split(cfg)
    # Drive the leakage-safe split with this run's seed so the shared query
    # set + index pool resample across seeds (multi-seed variance studies).
    cfg["experiment"]["split"]["seed"] = int(seed)
    # Sampling OFF: the split (and therefore the query set + index pool) must
    # be independent of the index-distribution being studied.
    cfg.setdefault("experiment", {})["sampling"] = {"mode": "none", "seed": seed}

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_index_distribution"
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {out_dir}")

    print("=" * 72)
    print("EXPERIMENT: CVEfixes index-distribution study (shared query set)")
    print("=" * 72)
    print(f"Config: {config_path}   GraphML: {graphml_root}")
    print(
        f"Embedders: {embedders}   KS: {ks}   index_total={index_total}   seed={seed}"
    )

    # 1) Load + leakage-safe split ONCE (shared query set + index pool).
    exp = CVEFixesFileLevelRetrievalExperiment(run_self_retrieval=True, ks=ks)
    data = exp.load_data(cfg)
    query_pairs = data["query_pairs"]
    index_pool = data["index_pairs"]
    query_cves = {p.cve_id for p in query_pairs}
    print(
        f"\nShared split: query={len(query_pairs)} ({len(query_cves)} CVEs)  index_pool={len(index_pool)}"
    )

    # 2) Shared support backbone (identical in both variants).
    support = _pick_support(index_pool, query_cves, seed)
    print(f"Support backbone (1 pair per query CVE): {len(support)} pairs")

    # 3) Two equal-size indices, differing only in CWE distribution.
    prop_index, prop_dist = _build_index_variant(
        index_pool, support, mode="proportional", total=index_total, seed=seed + 1
    )
    bal_index, bal_dist = _build_index_variant(
        index_pool, support, mode="balanced", total=index_total, seed=seed + 1
    )
    if len(prop_index) != len(bal_index):
        # Equalize defensively (should already match); trim the larger's
        # non-support pairs so both indices are exactly the same size.
        n = min(len(prop_index), len(bal_index))
        support_ids = {id(p) for p in support}

        def _trim(idx):
            keep = [p for p in idx if id(p) in support_ids]
            extra = [p for p in idx if id(p) not in support_ids]
            return keep + extra[: n - len(keep)]

        prop_index, bal_index = _trim(prop_index), _trim(bal_index)
        prop_dist = dict(Counter(p.cwe_id for p in prop_index))
        bal_dist = dict(Counter(p.cwe_id for p in bal_index))
    print(f"Proportional index: {len(prop_index)} pairs  CWE={prop_dist}")
    print(f"Balanced     index: {len(bal_index)} pairs  CWE={bal_dist}")

    emb_cfg = cfg.get("embeddings", {})
    prop_cells = run_variant_retrieval(
        "proportional",
        prop_index,
        query_pairs,
        embedders=embedders,
        emb_cfg=emb_cfg,
        ks=ks,
        out_dir=out_dir,
    )
    bal_cells = run_variant_retrieval(
        "balanced",
        bal_index,
        query_pairs,
        embedders=embedders,
        emb_cfg=emb_cfg,
        ks=ks,
        out_dir=out_dir,
    )

    comparison = _build_comparison(
        prop_cells,
        bal_cells,
        ks=ks,
        n_query=len(query_pairs),
        n_index=len(prop_index),
        dist={"proportional": prop_dist, "balanced": bal_dist},
        embedders=embedders,
    )
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str)
    )
    print(f"\nComparison → {out_dir / 'comparison.json'}")
    _print_comparison(comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CVEfixes index-distribution study (shared query set)"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--output-dir",
        default="cvefixes_experiments/output/index_distribution_study",
    )
    parser.add_argument("--graphml-root", default="graphml_fixedcves_file")
    parser.add_argument("--db-path", default="data/cvefixes/CVEfixes.db")
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=["codebert_seq", "codebert_pattern", "combined", "gin"],
    )
    parser.add_argument("--target-cwes", nargs="*", default=None)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument(
        "--index-total",
        type=int,
        default=140,
        help="Size of BOTH indices (shared). Clamped to [n_support, index_pool].",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_study(
        config_path=args.config,
        output_dir=args.output_dir,
        graphml_root=args.graphml_root,
        db_path=args.db_path,
        embedders=args.embedders,
        target_cwes=args.target_cwes,
        ks=args.ks,
        index_total=args.index_total,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
