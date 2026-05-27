"""
Combination Experiment A: Normalization vs PCA as Pre-processing.

Compares two approaches for equalizing sub-embedder contributions
before concatenation and final PCA:

  1. norm_concat_pca  — L2-norm each sub-embedder → concat → PCA to 128d
  2. pca_concat_pca   — PCA each sub-embedder (128→42d) → concat (126d) → PCA to 128d

Grid: strategy (single axis).

Usage:
    python -m experiments.exp.combining_norm_vs_pca [--config config.yaml]
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
torch.use_deterministic_algorithms(True)

from experiments.base import Axis, CellContext, Experiment, ExperimentOutput, MetricSpec
from experiments.common import build_flat_index, evaluate_retrieval, save_json
from experiments.exp.combining_strategies.scripts._combining_strategies import NormConcatPCA, PCAConcatPCA
from src.metrics.metrics import embedding_space_stats
from src.metrics.space_analysis import (
    alignment_uniformity,
    distance_concentration,
    hubness,
    intra_inter_ratio,
    isotropy,
)

STRATEGY_CLASSES = {
    "norm_concat_pca": NormConcatPCA,
    "pca_concat_pca": PCAConcatPCA,
}


# ── Helpers ──────────────────────────────────────────────────────────


def _encode_labels(pairs: list) -> np.ndarray:
    cwe_list = [p.cwe_id or "UNKNOWN" for p in pairs]
    unique = sorted(set(cwe_list))
    mapping = {c: i for i, c in enumerate(unique)}
    return np.array([mapping[c] for c in cwe_list])


def _metric_intrinsic(ctx: CellContext) -> dict:
    return ctx.artifacts["intrinsic"]


def _metric_retrieval(ctx: CellContext) -> dict:
    return ctx.artifacts["retrieval"]


def _metric_class_separation(ctx: CellContext) -> dict:
    return ctx.artifacts["class_separation"]


def _metric_alignment_uniformity(ctx: CellContext) -> dict:
    return ctx.artifacts["alignment_uniformity"]


# ── Experiment class ─────────────────────────────────────────────────


class NormVsPCAExperiment(Experiment):
    """Comparison A: L2-normalization vs PCA as pre-concatenation equalizer."""

    def __init__(self):
        self._cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "combining_norm_vs_pca"

    def load_data(self, cfg: dict) -> dict[str, Any]:
        from experiments.common import build_split
        from src.data import load_pairs

        pairs = load_pairs(cfg)
        index_pairs, query_pairs, split_info = build_split(pairs, cfg)
        labels = _encode_labels(index_pairs)
        return {
            "pairs": pairs,
            "index_pairs": index_pairs,
            "query_pairs": query_pairs,
            "split_info": split_info,
            "labels": labels,
        }

    def axes(self, cfg: dict) -> list[Axis]:
        return [
            Axis("strategy", list(STRATEGY_CLASSES.keys()), description="Pre-processing method"),
        ]

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec("intrinsic", _metric_intrinsic, requires=["intrinsic"]),
            MetricSpec("retrieval", _metric_retrieval, requires=["retrieval"]),
            MetricSpec("class_separation", _metric_class_separation, requires=["class_separation"]),
            MetricSpec("alignment_uniformity", _metric_alignment_uniformity, requires=["alignment_uniformity"]),
        ]

    def before_run(self, ctx: CellContext) -> None:
        self._cache = ctx.cache

    def run_cell(self, ctx: CellContext) -> dict[str, Any]:
        # Full RNG reset — isolates each strategy from execution order
        np.random.seed(42)
        torch.manual_seed(42)

        strategy_name = ctx.coords["strategy"]
        strategy_cls = STRATEGY_CLASSES[strategy_name]
        emb_cfg = ctx.cfg.get("embeddings", {})
        strategy = strategy_cls(emb_cfg)

        index_pairs = ctx.data["index_pairs"]
        query_pairs = ctx.data["query_pairs"]
        labels = ctx.data["labels"]
        ks = ctx.cfg.get("experiment", {}).get("ks", [1, 5, 10])

        # ── Embed index graphs ───────────────────────────────────────
        graphs = [p.G_vuln for p in index_pairs]
        t0 = time.perf_counter()
        index_embs = strategy.embed_many(graphs)
        embed_time = time.perf_counter() - t0
        print(f"    {strategy.name}: {index_embs.shape} in {embed_time:.1f}s")

        ctx.cache[f"emb__{strategy.name}"] = index_embs

        # ── Embed query graphs ───────────────────────────────────────
        query_graphs = [p.G_vuln for p in query_pairs]
        query_embs = strategy.embed_many(query_graphs)

        # ── Intrinsic space quality ──────────────────────────────────
        space_stats = embedding_space_stats(index_embs)
        iso = isotropy(index_embs)
        hub = hubness(index_embs, k=10)
        dc = distance_concentration(index_embs)
        intrinsic = {
            "dim": int(index_embs.shape[1]),
            "embed_time_s": embed_time,
            "space_stats": space_stats,
            "isotropy": iso,
            "hubness": hub,
            "distance_concentration": dc,
        }

        # ── Class separation ─────────────────────────────────────────
        class_sep = intra_inter_ratio(index_embs, labels)
        class_sep.pop("per_class", None)

        # ── Alignment / Uniformity ───────────────────────────────────
        align_unif = alignment_uniformity(index_embs, labels)

        # ── Retrieval evaluation ─────────────────────────────────────
        index, retriever = build_flat_index(
            index_pairs, index_embs, strategy.name, strategy.dim,
        )
        retrieval = evaluate_retrieval(
            query_pairs, query_embs, retriever, index_pairs, ks=ks,
        )
        retrieval.pop("raw_queries", None)

        # ── Store artifacts ──────────────────────────────────────────
        ctx.artifacts["intrinsic"] = intrinsic
        ctx.artifacts["retrieval"] = retrieval
        ctx.artifacts["class_separation"] = class_sep
        ctx.artifacts["alignment_uniformity"] = align_unif

        return {
            "strategy": strategy.name,
            "n_index": len(index_pairs),
            "n_query": len(query_pairs),
        }

    def after_run(self, output: ExperimentOutput) -> None:
        """Write comparison report."""
        rows = []
        for cell in output.cells:
            if cell.error:
                continue
            m = cell.metrics
            retrieval = m.get("retrieval") if isinstance(m.get("retrieval"), dict) else {}
            intrinsic = m.get("intrinsic") if isinstance(m.get("intrinsic"), dict) else {}
            class_sep = m.get("class_separation") if isinstance(m.get("class_separation"), dict) else {}
            rows.append({
                "strategy": cell.coords.get("strategy", "?"),
                "hit@1": retrieval.get("hit@1"),
                "hit@5": retrieval.get("hit@5"),
                "hit@10": retrieval.get("hit@10"),
                "mrr": retrieval.get("mrr"),
                "ndcg@10": retrieval.get("ndcg@10"),
                "map@10": retrieval.get("map@10"),
                "eff_dim": intrinsic.get("space_stats", {}).get("effective_dim"),
                "mean_sim": intrinsic.get("space_stats", {}).get("mean_pairwise_sim"),
                "isotropy": intrinsic.get("isotropy"),
                "hub_skew": intrinsic.get("hubness", {}).get("k_skewness"),
                "intra_inter": class_sep.get("ratio"),
                "embed_time_s": intrinsic.get("embed_time_s"),
            })

        for row in rows:
            s = row["strategy"]
            row["strategy"] = s.name if hasattr(s, "name") else str(s)

        report = {"comparison": rows}
        save_json(report, output.run_dir / "norm_vs_pca_report.json")

        print(f"\n{'='*70}")
        print("Comparison A: Normalization vs PCA (pre-concatenation equalizer)")
        print(f"{'='*70}")
        header = f"{'Strategy':<20} {'hit@1':>6} {'hit@5':>6} {'MRR':>6} {'nDCG@10':>8} {'eff_dim':>8} {'isotropy':>9}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['strategy']:<20} "
                f"{r['hit@1'] or 0:>6.3f} "
                f"{r['hit@5'] or 0:>6.3f} "
                f"{r['mrr'] or 0:>6.3f} "
                f"{r['ndcg@10'] or 0:>8.3f} "
                f"{r['eff_dim'] or 0:>8.1f} "
                f"{r['isotropy'] or 0:>9.4f}"
            )
        print(f"{'='*70}")
        print(f"\nReport → {output.run_dir}/norm_vs_pca_report.json")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from src.io import load_config

    parser = argparse.ArgumentParser(description="Comparison A: norm vs PCA pre-processing")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = NormVsPCAExperiment()
    experiment.run(cfg)
