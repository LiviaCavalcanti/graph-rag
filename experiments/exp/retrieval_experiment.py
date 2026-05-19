"""
Retrieval grid experiment — embedder × backend × graph_variant.

Reimplements runner.py using the Experiment base class. Separates:
  - Core logic: embedding + index construction (run_cell)
  - Metrics: reuses src.metrics functions via declarative MetricSpecs
  - I/O: handled by base class + after_run hook for dashboard

Batchable: the embed step is shared across backends via ctx.cache,
and run_cell processes one (embedder, backend, graph_variant) cell.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data.split import build_split
from experiments.base import Axis, CellContext, Experiment, ExperimentOutput, MetricSpec
from src.embeddings import build_embedders
from src.metrics.metrics import embedding_space_stats, measure_latency
from src.metrics.retrieval_eval import (
    cve_retrieval_metrics,
    cwe_recall_metrics,
    retrieve_all,
)
from src.rag.hnsw import HNSWIndex
from src.rag.utils import populate_index

BACKEND_REGISTRY = {
    "hnsw": HNSWIndex,
}


# ── Metric functions (MetricSpec-compatible signatures) ──────────────


def _metric_space_stats(ctx: CellContext) -> dict:
    """Intrinsic embedding space quality (no labels needed)."""
    return ctx.artifacts["space_stats"]


def _metric_self_retrieval(ctx: CellContext) -> dict:
    """CVE-level hit@k / MRR via re-embedding queries."""
    qr = ctx.artifacts["query_results"]
    index_metadata = ctx.artifacts["index_metadata"]
    ks = ctx.cfg.get("experiment", {}).get("ks", [1, 5, 10])
    return cve_retrieval_metrics(qr, ks=ks, index_metadata=index_metadata)


def _metric_cwe_recall(ctx: CellContext) -> dict:
    """Cross-CWE recall from pre-retrieved results."""
    qr = ctx.artifacts["query_results"]
    index_metadata = ctx.artifacts["index_metadata"]
    ks = ctx.cfg.get("experiment", {}).get("ks", [1, 5, 10])
    return cwe_recall_metrics(qr, index_metadata, top_k=max(ks))


def _metric_latency(ctx: CellContext) -> dict:
    """Query latency percentiles (p50/p95/p99)."""
    retriever = ctx.artifacts["retriever"]
    embeddings = ctx.artifacts["index_embeddings"]
    return measure_latency(retriever, embeddings)


# ── Experiment class ─────────────────────────────────────────────────


class RetrievalGridExperiment(Experiment):
    """Embedder × backend × graph_variant grid search."""

    def __init__(
        self,
        *,
        run_leave_one_out: bool = False,
        run_self_retrieval: bool = True,
        ks: list[int] | None = None,
        preloaded_pairs: list | None = None,
    ):
        self._run_loo = run_leave_one_out
        self._run_self_retrieval = run_self_retrieval
        self._ks = ks
        self._preloaded_pairs = preloaded_pairs

    @property
    def name(self) -> str:
        return "experiment"

    def load_data(self, cfg: dict) -> dict[str, Any]:
        """Load pairs (or use preloaded) and split."""
        if self._preloaded_pairs is not None:
            pairs = self._preloaded_pairs
        else:
            from src.data import load_pairs
            pairs = load_pairs(cfg)
        index_pairs, query_pairs, split_info = build_split(pairs, cfg)
        return {
            "pairs": pairs,
            "index_pairs": index_pairs,
            "query_pairs": query_pairs,
            "split_info": split_info,
        }

    def axes(self, cfg: dict) -> list[Axis]:
        embedders = build_embedders(cfg)
        backends = list(BACKEND_REGISTRY.keys())
        graph_variants = cfg.get("experiment", {}).get(
            "graph_variants", ["G_vuln"]
        )
        return [
            Axis("embedder", embedders, description="Embedding model"),
            Axis("graph_variant", graph_variants, description="Which graph to embed"),
            Axis("backend", backends, description="Vector index backend"),
        ]

    def metrics(self) -> list[MetricSpec]:
        specs = [
            MetricSpec("space_stats", _metric_space_stats, requires=["space_stats"]),
            MetricSpec("latency", _metric_latency, requires=["retriever", "index_embeddings"]),
        ]
        if self._run_self_retrieval:
            specs.append(
                MetricSpec("self_retrieval", _metric_self_retrieval, requires=["query_results"])
            )
        specs.append(
            MetricSpec("cwe_recall", _metric_cwe_recall, requires=["query_results", "index_metadata"])
        )
        return specs

    def run_cell(self, ctx: CellContext) -> dict[str, Any]:
        embedder = ctx.coords["embedder"]
        backend_name = ctx.coords["backend"]
        graph_variant = ctx.coords["graph_variant"]
        index_pairs = ctx.data["index_pairs"]
        query_pairs = ctx.data["query_pairs"]
        ks = self._ks or ctx.cfg.get("experiment", {}).get("ks", [1, 5, 10])

        # ── embedding (cached across backends) ───────────────────────
        cache_key = f"{embedder.name}__{graph_variant}"
        if cache_key not in ctx.cache:
            t0 = time.perf_counter()
            graphs = [getattr(p, graph_variant) for p in index_pairs]
            embeddings = embedder.embed_many(graphs)
            embed_time = time.perf_counter() - t0
            print(f"    embedded {len(graphs)} graphs in {embed_time:.1f}s")

            stats = embedding_space_stats(embeddings)
            print(
                f"    eff_dim={stats['effective_dim']:.1f}  "
                f"mean_sim={stats['mean_pairwise_sim']:.3f}"
            )
            ctx.cache[cache_key] = {
                "embeddings": embeddings,
                "embed_time_s": embed_time,
                "space_stats": stats,
            }

        cached = ctx.cache[cache_key]
        index_embeddings = cached["embeddings"]

        # ── build index ──────────────────────────────────────────────
        t0 = time.perf_counter()
        index_dir = ctx.run_dir / "indices"
        index_dir.mkdir(exist_ok=True)
        stem = f"{embedder.name}__{graph_variant}"
        index = BACKEND_REGISTRY[backend_name](
            dim=embedder.dim,
            index_path=str(index_dir / f"{stem}__{backend_name}.index"),
            metadata_path=str(index_dir / f"{stem}__{backend_name}_meta.json"),
        )
        retriever = populate_index(
            index, index_pairs, index_embeddings, embedder.name, top_k=max(ks)
        )
        build_time = time.perf_counter() - t0

        # ── retrieve all queries (batch) ─────────────────────────────
        qr = retrieve_all(query_pairs, embedder, retriever, top_k=max(ks))

        # ── populate artifacts for MetricSpecs ───────────────────────
        ctx.artifacts["index_embeddings"] = index_embeddings
        ctx.artifacts["space_stats"] = cached["space_stats"]
        ctx.artifacts["retriever"] = retriever
        ctx.artifacts["index_metadata"] = index.metadata
        ctx.artifacts["query_results"] = qr

        # ── cell-level metadata (always returned) ────────────────────
        cell_meta = {
            "embedder": embedder.name,
            "backend": backend_name,
            "graph_variant": graph_variant,
            "n_index": len(index_pairs),
            "n_query": len(query_pairs),
            "embed_time_s": cached["embed_time_s"],
            "index_build_s": round(build_time, 3),
        }

        # ── optional: leave-one-out ──────────────────────────────────
        if self._run_loo and not ctx.data["split_info"].get("enabled") and len(index_pairs) <= 1000:
            from src.metrics.metrics import leave_one_out_metrics

            index_meta_list = [
                {"cve_id": p.cve_id, "cwe_id": p.cwe_id, "func_name": p.func_name,
                 "project": p.project, **p.meta}
                for p in index_pairs
            ]
            print(f"    LOO ({len(index_pairs)} iterations)...")
            loo = leave_one_out_metrics(
                embeddings=index_embeddings,
                metadata=index_meta_list,
                index_class=BACKEND_REGISTRY[backend_name],
                index_kwargs=dict(
                    dim=embedder.dim,
                    index_path=str(ctx.run_dir / "_loo_tmp.index"),
                    metadata_path=str(ctx.run_dir / "_loo_tmp_meta.json"),
                ),
                ks=ks,
            )
            cell_meta["leave_one_out"] = loo
            print(f"    LOO hit@1={loo.get('hit@1', 0):.3f}  mrr={loo.get('mrr', 0):.3f}")

        return cell_meta

    def after_run(self, output: ExperimentOutput) -> None:
        """Generate dashboard visualizations and flat summary."""
        import json
        from experiments.dashboard_scripts.visualization import generate_visualizations

        # Write legacy-format results.json (dashboard expects this structure)
        legacy = _to_legacy_format(output)
        results_path = output.run_dir / "results.json"
        results_path.write_text(json.dumps(legacy, indent=2, default=str))

        generate_visualizations(legacy, str(output.run_dir))

        # Flat summary for pandas
        _write_summary(output)


# ── I/O helpers ──────────────────────────────────────────────────────


def _to_legacy_format(output: ExperimentOutput) -> dict:
    """Convert ExperimentOutput to the legacy results.json format
    expected by the dashboard and downstream analysis scripts."""
    from datetime import datetime, timezone

    cells_legacy = []
    for cell in output.cells:
        m = cell.metrics
        cells_legacy.append({
            "embedder": m.get("embedder", ""),
            "backend": m.get("backend", ""),
            "graph_variant": m.get("graph_variant", ""),
            "n_samples": m.get("n_index", 0),
            "embed_time_s": m.get("embed_time_s", 0),
            "index_build_s": m.get("index_build_s", 0),
            "query_latency": m.get("latency", {}),
            "space_stats": m.get("space_stats", {}),
            "self_retrieval": m.get("self_retrieval", {}),
            "cwe_recall": m.get("cwe_recall", {}),
            "leave_one_out": m.get("leave_one_out", {}),
        })

    return {
        "run_id": output.run_id,
        "timestamp": output.metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "config": output.metadata.get("config", {}),
        "dataset_info": output.metadata.get("data_info", {}),
        "cells": cells_legacy,
    }


def _write_summary(output: ExperimentOutput) -> None:
    """Write a flat summary table (one row per cell) for easy comparison."""
    import json

    rows = []
    for cell in output.cells:
        m = cell.metrics
        rows.append({
            "run_id": output.run_id,
            **cell.coords_as_str(),
            "n_index": m.get("n_index", 0),
            "n_query": m.get("n_query", 0),
            "embed_time_s": m.get("embed_time_s", 0),
            "index_build_s": m.get("index_build_s", 0),
            "latency_p50_ms": m.get("latency", {}).get("p50_ms", 0),
            "latency_p99_ms": m.get("latency", {}).get("p99_ms", 0),
            "effective_dim": m.get("space_stats", {}).get("effective_dim", 0),
            "mean_pairwise_sim": m.get("space_stats", {}).get("mean_pairwise_sim", 0),
            "sr_hit@1": m.get("self_retrieval", {}).get("hit@1", 0),
            "sr_hit@5": m.get("self_retrieval", {}).get("hit@5", 0),
            "sr_hit@10": m.get("self_retrieval", {}).get("hit@10", 0),
            "sr_mrr": m.get("self_retrieval", {}).get("mrr", 0),
            "cwe_recall_macro": m.get("cwe_recall", {}).get("macro_avg", 0),
            "cwe_n_groups": m.get("cwe_recall", {}).get("n_cwes", 0),
            "loo_hit@1": m.get("leave_one_out", {}).get("hit@1", 0),
            "loo_mrr": m.get("leave_one_out", {}).get("mrr", 0),
        })

    summary_path = output.run_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))

    # Append to global registry
    registry_path = output.run_dir.parent / "all_runs.json"
    existing = json.loads(registry_path.read_text()) if registry_path.exists() else []
    existing.extend(rows)
    registry_path.write_text(json.dumps(existing, indent=2))
    print(f"Summary appended → {registry_path}")


# ── Entry point (backwards-compatible) ───────────────────────────────


def run_experiment(
    pairs: list,
    cfg: dict,
    run_leave_one_out: bool = False,
    run_self_retrieval: bool = True,
    ks: list[int] | None = None,
    output_dir: Path | None = None,
) -> ExperimentOutput:
    """Run the retrieval grid experiment.

    Drop-in replacement for the old runner.run_experiment().
    Accepts pre-loaded pairs to avoid re-loading from disk.
    """
    exp = RetrievalGridExperiment(
        run_leave_one_out=run_leave_one_out,
        run_self_retrieval=run_self_retrieval,
        ks=ks,
        preloaded_pairs=pairs,
    )
    if output_dir:
        return exp.run(cfg, output_dir=output_dir)
    return exp.run(cfg)
