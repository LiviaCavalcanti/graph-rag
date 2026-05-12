#!/usr/bin/env python3
"""
Repeated experiment runner with mean ± std aggregation.

Executes a *variant × embedder* grid multiple times with different
random splits (varying the seed) and aggregates all numeric results
with mean ± std.

The runner is **evaluation-function agnostic**: by default it performs
the slicing comparison (embed → HNSW index → retrieval + CWE recall),
but callers can supply a custom ``evaluate_fn`` and ``variant_defs``
to repeat *any* cell-level evaluation.

Default metrics (when using the built-in evaluator):
  • Self-retrieval: hit@1, hit@5, hit@10, MRR  (mean ± std)
  • CWE recall:     macro-average               (mean ± std)

Usage:
    python -m experiments.slicing_repeated [--runs 5] [--config config.yaml]

Programmatic:
    from experiments.slicing_repeated import run_repeated

    # Use defaults (slicing comparison with VARIANT_DEFS):
    run_repeated(cfg, n_runs=10)

    # Plug in a custom evaluator and variant grid:
    run_repeated(
        cfg,
        evaluate_fn=my_evaluate,
        variant_defs={"A": {...}, "B": {...}},
        tag="my_experiment",
    )
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import numpy as np
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import build_embedders
from experiments.common import (
    load_config, build_split, make_run_dir,
    evaluate_retrieval, evaluate_cwe_recall, save_json,
)
from src.data.autopatch import load_pairs
from src.rag.hnsw import HNSWIndex
from src.rag.utils import populate_index
from experiments.exp.slicing_comparison import (
    VARIANT_DEFS, _resolve_query_build_fn,
    _strip_diff_attrs, _add_labels_from_vuln,
)


# ── helpers ───────────────────────────────────────────────────────────

def _reset_pca(embedders: list) -> None:
    """Reset PCA state so each variant gets its own projection."""
    for emb in embedders:
        if hasattr(emb, '_fitted'):
            emb._fitted = False
            emb._pca = None


def _aggregate_runs(runs: list[dict],
                    skip_keys: frozenset[str] = frozenset(
                        {"variant", "embedder", "error", "n"}),
                    ) -> dict:
    """Recursively aggregate numeric leaf values with mean ± std."""
    if not runs:
        return {}
    agg: dict = {}
    sample = runs[0]
    for key in sample:
        if key in skip_keys:
            continue
        vals = [r.get(key) for r in runs]
        if all(isinstance(v, (int, float)) for v in vals):
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "values": [float(v) for v in vals],
            }
        elif all(isinstance(v, dict) for v in vals):
            agg[key] = _aggregate_runs(vals, skip_keys=frozenset())
    return agg


def _default_evaluate(
    index_pairs, query_pairs, embedder, variant_name, variant_def,
    run_dir, ks, *, query_variant=None, **_kw,
) -> dict:
    """Default evaluate_fn: embed → index → retrieve → metrics.

    Uses ``populate_index`` from ``src.rag.utils`` and the canonical
    ``evaluate_retrieval`` / ``evaluate_cwe_recall`` from
    ``src.metrics.retrieval_eval`` — the same primitives that
    ``RetrievalGridExperiment`` uses.
    """
    build_fn = variant_def["build"]
    q_build_fn, _ = _resolve_query_build_fn(query_variant, build_fn)
    q_fn = q_build_fn if q_build_fn is not None else build_fn

    # ── embed index ──────────────────────────────────────────────
    index_graphs = [build_fn(p) for p in index_pairs]
    index_embeddings = embedder.embed_many(index_graphs)

    norms = np.linalg.norm(index_embeddings, axis=1)
    if int(np.sum(norms < 1e-6)) == len(index_embeddings):
        return {
            "variant": variant_name,
            "embedder": embedder.name,
            "retrieval": {"hit@1": 0, "hit@5": 0, "hit@10": 0, "mrr": 0,
                          "cve_precision": 0, "cve_recall": 0, "cve_f1": 0, "n": 0},
            "cwe_recall": {"macro_avg": 0, "n_cwes": 0},
            "error": "all embeddings zero",
        }

    # ── build index (reuses src.rag.utils.populate_index) ────────
    max_k = max(ks)
    tag = f"{embedder.name}__{variant_name}"
    idx_dir = run_dir / "indices"
    idx_dir.mkdir(exist_ok=True)
    index = HNSWIndex(
        dim=embedder.dim,
        index_path=str(idx_dir / f"{tag}__hnsw.index"),
        metadata_path=str(idx_dir / f"{tag}__hnsw_meta.json"),
    )
    retriever = populate_index(index, index_pairs, index_embeddings,
                               embedder.name, top_k=max_k)

    # ── embed queries & evaluate ─────────────────────────────────
    query_graphs = [q_fn(p) for p in query_pairs]
    query_embeddings = embedder.embed_many(query_graphs)

    sr = evaluate_retrieval(
        query_pairs, query_embeddings, retriever, index_pairs, ks=ks,
    )
    cwe_result = evaluate_cwe_recall(
        query_pairs, query_embeddings, retriever, index.metadata, top_k=max_k,
    )

    return {
        "variant": variant_name,
        "embedder": embedder.name,
        "retrieval": sr,
        "cwe_recall": cwe_result,
    }


# ── main repeated experiment ─────────────────────────────────────────

def run_repeated(
    cfg: dict,
    n_runs: int = 5,
    base_seed: int = 42,
    *,
    evaluate_fn: Callable[..., dict] | None = None,
    variant_defs: dict[str, dict] | None = None,
    query_variant: str | None = None,
    tag: str | None = None,
    pre_variant_hook: Callable[[list], None] | None = _reset_pca,
    evaluate_kw: dict | None = None,
) -> dict:
    """Run a variant × embedder grid *n_runs* times and aggregate.

    Parameters
    ----------
    cfg : dict
        Project configuration (passed to ``load_pairs`` / ``build_embedders``).
    n_runs / base_seed
        How many repetitions and starting RNG seed.
    evaluate_fn : callable, optional
        ``(index_pairs, query_pairs, embedder, variant_name, variant_def,
        run_dir, ks, **evaluate_kw) -> dict``.  Must return a dict whose
        numeric leaf values will be aggregated across runs.
        Defaults to the slicing-comparison evaluator.
    variant_defs : dict, optional
        ``{name: {"build": callable, ...}, ...}``.  Defaults to
        ``VARIANT_DEFS`` from ``slicing_comparison``.
    query_variant : str, optional
        Forwarded to the default ``evaluate_fn`` (ignored by custom ones
        unless they read it from *evaluate_kw*).
    tag : str, optional
        Name prefix for the output directory.  Defaults to
        ``"repeated{n_runs}"``.
    pre_variant_hook : callable, optional
        ``(embedders) -> None`` called before each variant.
        Defaults to PCA-state reset.  Pass ``None`` to skip.
    evaluate_kw : dict, optional
        Extra keyword arguments forwarded to *evaluate_fn*.
    """
    if evaluate_fn is None:
        evaluate_fn = _default_evaluate
    if variant_defs is None:
        variant_defs = VARIANT_DEFS
    if tag is None:
        tag = f"repeated{n_runs}"
    extra_kw = dict(evaluate_kw or {})
    if query_variant is not None:
        extra_kw.setdefault("query_variant", query_variant)

    run_id, run_dir = make_run_dir(tag)

    # load data once
    pairs = load_pairs(cfg)
    print(f"Loaded {len(pairs)} pairs")
    if query_variant:
        print(f"Query variant override: {query_variant}")

    embedders = build_embedders(cfg)
    ks = [1, 5, 10]
    variant_names = list(variant_defs.keys())
    embedder_names = [e.name for e in embedders]

    # collector: (variant, embedder) → list of per-run dicts
    all_runs: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
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
            variant_def = variant_defs[variant_name]
            if pre_variant_hook is not None:
                pre_variant_hook(embedders)
            for embedder in embedders:
                print(f"  [{run_idx+1}/{n_runs}] {embedder.name} / {variant_name} ... ",
                      end="", flush=True)
                try:
                    result = evaluate_fn(
                        index_pairs, query_pairs, embedder,
                        variant_name, variant_def, sub_dir, ks,
                        **extra_kw,
                    )
                    print("  ".join(
                        f"{k}={v:.3f}" for k, v in result.items()
                        if isinstance(v, (int, float))
                    ) or "ok")
                except Exception as e:
                    print(f"ERROR: {e}")
                    result = {
                        "variant": variant_name,
                        "embedder": embedder.name,
                        "error": str(e),
                    }
                all_runs[(variant_name, embedder.name)].append(result)

    # ── aggregate ─────────────────────────────────────────────────
    aggregated = []
    for variant_name in variant_names:
        for emb_name in embedder_names:
            runs = all_runs[(variant_name, emb_name)]
            row = {
                "variant": variant_name,
                "embedder": emb_name,
                "n_runs": len(runs),
                **_aggregate_runs(runs),
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

    return report

