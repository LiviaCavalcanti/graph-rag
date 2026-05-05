#!/usr/bin/env python3
"""
Repeated slicing comparison with mean ± std aggregation.

Runs the 4-variant × N-embedder grid multiple times with different
train/test splits (varying the random seed), then reports:

  • Self-retrieval: hit@1, hit@5, hit@10, MRR  (mean ± std)
  • CWE recall:     macro-average               (mean ± std)

Usage:
    python -m experiments.slicing_repeated [--runs 5] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import build_embedders
from experiments.common import (
    load_config, build_split, make_run_dir,
    build_hnsw, evaluate_retrieval, evaluate_cwe_recall, save_json,
)
from src.data.autopatch import load_pairs
from experiments.slicing_comparison import (
    VARIANT_DEFS, _resolve_query_build_fn,
    _strip_diff_attrs, _add_labels_from_vuln,
)


# ── single-run evaluation ────────────────────────────────────────────

def _evaluate_one_run(
    index_pairs: list,
    query_pairs: list,
    embedder,
    build_fn,
    variant_name: str,
    run_dir: Path,
    ks: list[int],    *,
    query_build_fn=None,) -> dict:
    """
    Run one (variant, embedder) cell. Returns a dict with separate
    retrieval and cwe_recall sections.
    """
    # ── embed index ──────────────────────────────────────────────
    index_graphs = [build_fn(p) for p in index_pairs]
    index_embeddings = embedder.embed_many(index_graphs)

    norms = np.linalg.norm(index_embeddings, axis=1)
    n_zero = int(np.sum(norms < 1e-6))
    if n_zero == len(index_embeddings):
        return {
            "variant": variant_name,
            "embedder": embedder.name,
            "retrieval": {"hit@1": 0, "hit@5": 0, "hit@10": 0, "mrr": 0,
                          "cve_precision": 0, "cve_recall": 0, "cve_f1": 0, "n": 0},
            "cwe_recall": {"macro_avg": 0, "n_cwes": 0},
            "error": f"all {n_zero} embeddings zero",
        }

    # ── build HNSW index via common ──────────────────────────────
    tag = f"{embedder.name}__{variant_name}"
    index, retriever = build_hnsw(
        index_pairs, index_embeddings, embedder.name,
        embedder.dim, run_dir, tag=tag,
    )

    # ── embed queries ────────────────────────────────────────────
    q_fn = query_build_fn if query_build_fn is not None else build_fn
    query_graphs = [q_fn(p) for p in query_pairs]
    query_embeddings = embedder.embed_many(query_graphs)

    # ── retrieval metrics via common ─────────────────────────────
    sr = evaluate_retrieval(
        query_pairs, query_embeddings, retriever, index_pairs, ks=ks,
    )

    # ── CWE recall via common ────────────────────────────────────
    cwe_result = evaluate_cwe_recall(
        query_pairs, query_embeddings, retriever, index.metadata, top_k=max(ks),
    )

    return {
        "variant": variant_name,
        "embedder": embedder.name,
        "retrieval": sr,
        "cwe_recall": cwe_result,
    }


# ── main repeated experiment ─────────────────────────────────────────

def run_repeated(cfg: dict, n_runs: int = 5, base_seed: int = 42,
                 *, query_variant: str | None = None) -> dict:
    run_id, run_dir = make_run_dir(f"repeated{n_runs}")

    # load data once
    pairs = load_pairs(cfg)
    print(f"Loaded {len(pairs)} pairs")
    if query_variant:
        print(f"Query variant override: {query_variant}")

    embedders = build_embedders(cfg)
    ks = [1, 5, 10]
    variant_names = list(VARIANT_DEFS.keys())
    embedder_names = [e.name for e in embedders]

    # collector: (variant, embedder) → list of per-run dicts
    all_runs: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        # override seed in config for this run
        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("experiment", {}).setdefault("split", {})["seed"] = seed

        index_pairs, query_pairs, split_info = build_split(pairs, run_cfg)
        print(f"\n{'━'*60}")
        print(f"  Run {run_idx + 1}/{n_runs}  seed={seed}  "
              f"index={len(index_pairs)}  query={len(query_pairs)}")
        print(f"{'━'*60}")

        sub_dir = run_dir / f"run_{run_idx}"
        sub_dir.mkdir(exist_ok=True)

        for variant_name in variant_names:
            build_fn = VARIANT_DEFS[variant_name]["build"]
            q_build_fn, _ = _resolve_query_build_fn(query_variant, build_fn)
            # Reset PCA state so each variant gets its own projection
            for emb in embedders:
                if hasattr(emb, '_fitted'):
                    emb._fitted = False
                    emb._pca = None
            for embedder in embedders:
                print(f"  [{run_idx+1}/{n_runs}] {embedder.name} / {variant_name} ... ", end="", flush=True)
                try:
                    result = _evaluate_one_run(
                        index_pairs, query_pairs, embedder,
                        build_fn, variant_name, sub_dir, ks,
                        query_build_fn=q_build_fn,
                    )
                    sr = result["retrieval"]
                    cw = result["cwe_recall"]
                    print(f"hit@1={sr.get('hit@1',0):.3f}  "
                          f"MRR={sr.get('mrr',0):.3f}  "
                          f"CVE_F1={sr.get('cve_f1',0):.3f}  "
                          f"CWE={cw.get('macro_avg',0):.3f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    result = {
                        "variant": variant_name,
                        "embedder": embedder.name,
                        "retrieval": {"hit@1": 0, "hit@5": 0, "hit@10": 0, "mrr": 0,
                                      "cve_precision": 0, "cve_recall": 0, "cve_f1": 0, "n": 0},
                        "cwe_recall": {"macro_avg": 0, "n_cwes": 0},
                        "error": str(e),
                    }
                all_runs[(variant_name, embedder.name)].append(result)

    # ── aggregate ─────────────────────────────────────────────────
    aggregated = []
    for variant_name in variant_names:
        for emb_name in embedder_names:
            runs = all_runs[(variant_name, emb_name)]
            sr_keys = ["hit@1", "hit@5", "hit@10", "mrr", "cve_precision", "cve_recall", "cve_f1"]
            sr_vals = {k: [r["retrieval"].get(k, 0) for r in runs] for k in sr_keys}
            cwe_vals = [r["cwe_recall"].get("macro_avg", 0) for r in runs]

            row = {
                "variant": variant_name,
                "embedder": emb_name,
                "n_runs": len(runs),
                "retrieval": {
                    k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                    for k, v in sr_vals.items()
                },
                "cwe_recall": {
                    "macro_avg": {
                        "mean": float(np.mean(cwe_vals)),
                        "std": float(np.std(cwe_vals)),
                        "values": cwe_vals,
                    }
                },
            }
            aggregated.append(row)

    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": n_runs,
        "base_seed": base_seed,
        "seeds": [base_seed + i for i in range(n_runs)],
        "n_pairs": len(pairs),
        "query_variant": query_variant,
        "variants": variant_names,
        "embedders": embedder_names,
        "aggregated": aggregated,
    }

    out_path = run_dir / "repeated_comparison.json"
    save_json(report, out_path)
    print(f"\nResults written to: {out_path}")

    _print_tables(report)
    return report


# ── pretty-print ─────────────────────────────────────────────────────

def _fmt(mean: float, std: float) -> str:
    """Format as 'XX.X ± Y.Y%'."""
    return f"{mean*100:5.1f}±{std*100:4.1f}%"


def _fmt_mrr(mean: float, std: float) -> str:
    return f"{mean:.3f}±{std:.3f}"


def _print_tables(report: dict):
    agg = report["aggregated"]
    variants = report["variants"]
    embedders = report["embedders"]
    n_runs = report["n_runs"]

    print(f"\n{'='*100}")
    print(f"  REPEATED SLICING COMPARISON  ({n_runs} runs, seeds {report['seeds']})")
    print(f"{'='*100}")

    # ── Retrieval table ────────────────────────────────────────
    print(f"\n  ┌─ RETRIEVAL {'─'*82}┐")
    for emb in embedders:
        print(f"\n  {emb}:")
        print(f"    {'Variant':<25s} {'hit@1':>13s} {'hit@5':>13s} {'hit@10':>13s} {'MRR':>13s} {'CVE F1':>13s}")
        print(f"    {'─'*92}")
        for v in variants:
            row = next((r for r in agg if r["variant"] == v and r["embedder"] == emb), None)
            if not row:
                continue
            sr = row["retrieval"]
            h1 = _fmt(sr["hit@1"]["mean"], sr["hit@1"]["std"])
            h5 = _fmt(sr["hit@5"]["mean"], sr["hit@5"]["std"])
            h10 = _fmt(sr["hit@10"]["mean"], sr["hit@10"]["std"])
            mrr = _fmt_mrr(sr["mrr"]["mean"], sr["mrr"]["std"])
            cf1 = _fmt(sr["cve_f1"]["mean"], sr["cve_f1"]["std"]) if "cve_f1" in sr else "N/A"
            print(f"    {v:<25s} {h1:>13s} {h5:>13s} {h10:>13s} {mrr:>13s} {cf1:>13s}")

    # ── CWE recall table ─────────────────────────────────────────
    print(f"\n  ┌─ CWE RECALL (macro avg) {'─'*71}┐")
    for emb in embedders:
        print(f"\n  {emb}:")
        print(f"    {'Variant':<25s} {'CWE recall':>13s}")
        print(f"    {'─'*40}")
        for v in variants:
            row = next((r for r in agg if r["variant"] == v and r["embedder"] == emb), None)
            if not row:
                continue
            cw = row["cwe_recall"]["macro_avg"]
            val = _fmt(cw["mean"], cw["std"])
            print(f"    {v:<25s} {val:>13s}")

    # ── Summary: hit@1 across variants ───────────────────────────
    print(f"\n  ┌─ SUMMARY: hit@1  (mean ± std) {'─'*64}┐")
    header = f"    {'Embedder':<20s}" + "".join(f"{v:>22s}" for v in variants)
    print(header)
    print(f"    {'─'*(20 + 22*len(variants))}")
    for emb in embedders:
        cells = []
        for v in variants:
            row = next((r for r in agg if r["variant"] == v and r["embedder"] == emb), None)
            if row:
                sr = row["retrieval"]["hit@1"]
                cells.append(f"{_fmt(sr['mean'], sr['std']):>20s}")
            else:
                cells.append(f"{'N/A':>20s}")
        print(f"    {emb:<20s}" + "".join(f"{c:>22s}" for c in cells))

    # ── Summary: CWE recall across variants ──────────────────────
    print(f"\n  ┌─ SUMMARY: CWE recall  (mean ± std) {'─'*59}┐")
    header = f"    {'Embedder':<20s}" + "".join(f"{v:>22s}" for v in variants)
    print(header)
    print(f"    {'─'*(20 + 22*len(variants))}")
    for emb in embedders:
        cells = []
        for v in variants:
            row = next((r for r in agg if r["variant"] == v and r["embedder"] == emb), None)
            if row:
                cw = row["cwe_recall"]["macro_avg"]
                cells.append(f"{_fmt(cw['mean'], cw['std']):>20s}")
            else:
                cells.append(f"{'N/A':>20s}")
        print(f"    {emb:<20s}" + "".join(f"{c:>22s}" for c in cells))

    print(f"\n{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description="Repeated slicing comparison with stats")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--runs", type=int, default=5, help="Number of repeated runs")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting seed")
    parser.add_argument(
        "--query-variant",
        default=None,
        help="Fix query graphs to this variant (e.g. G_before, G_vuln, "
             "runner_compat).  Default: same as index variant.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_repeated(cfg, n_runs=args.runs, base_seed=args.base_seed,
                 query_variant=args.query_variant)


if __name__ == "__main__":
    main()
