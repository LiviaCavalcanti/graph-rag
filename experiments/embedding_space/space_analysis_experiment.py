"""
Embedding Space Analysis Experiment.

Evaluates intrinsic quality and cross-space comparison metrics for all
active embedders using the Experiment base class.

Grid: embedder (single axis — one cell per embedder).
Cross-space metrics (CKA, k-NN overlap, rank correlation, trustworthiness)
are computed in after_run once all embeddings are cached.

Outputs:
  <run_dir>/results.json         — base class output (per-cell intrinsic metrics)
  <run_dir>/space_analysis.json  — full report including pairwise matrices
  <run_dir>/pairwise_cka.json    — CKA matrix
  <run_dir>/pairwise_knn.json    — k-NN overlap matrix
  <run_dir>/pairwise_rank_corr.json — rank correlation matrix

Usage:
    python -m experiments.space_analysis_experiment [--config config.yaml]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.base import Axis, CellContext, Experiment, ExperimentOutput, MetricSpec
from experiments.common import save_json
from src.embeddings import build_embedders
from src.metrics.metrics import embedding_space_stats
from src.metrics.space_analysis import (
    alignment_uniformity,
    distance_concentration,
    hubness,
    intra_inter_ratio,
    isotropy,
    knn_overlap,
    linear_cka,
    rank_correlation,
    trustworthiness,
)


# ── helpers ──────────────────────────────────────────────────────────


def _encode_labels(pairs: list) -> np.ndarray:
    """Encode CWE strings as integer labels."""
    cwe_list = [p.cwe_id or "UNKNOWN" for p in pairs]
    unique = sorted(set(cwe_list))
    mapping = {c: i for i, c in enumerate(unique)}
    return np.array([mapping[c] for c in cwe_list])


# ── MetricSpec functions ─────────────────────────────────────────────


def _metric_intrinsic(ctx: CellContext) -> dict:
    """Per-embedder intrinsic space quality."""
    return ctx.artifacts["intrinsic"]


def _metric_class_separation(ctx: CellContext) -> dict:
    """Intra/inter CWE ratio."""
    return ctx.artifacts["class_separation"]


def _metric_alignment_uniformity(ctx: CellContext) -> dict:
    """Wang & Isola alignment + uniformity."""
    return ctx.artifacts["alignment_uniformity"]


# ── Experiment class ─────────────────────────────────────────────────


class SpaceAnalysisExperiment(Experiment):
    """Embedding space analysis — one cell per embedder."""

    def __init__(self, k_values: list[int] | None = None):
        self._k_values = k_values or [5, 10, 20]
        self._cache: dict[str, Any] = {}  # populated in before_run

    @property
    def name(self) -> str:
        return "space_analysis"

    def load_data(self, cfg: dict) -> dict[str, Any]:
        """Load pairs + split, precompute labels."""
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
            "n_cwe_classes": int(len(np.unique(labels))),
        }

    def axes(self, cfg: dict) -> list[Axis]:
        embedders = build_embedders(cfg)
        return [
            Axis("embedder", embedders, description="Embedding model"),
        ]

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec("intrinsic", _metric_intrinsic, requires=["intrinsic"]),
            MetricSpec("class_separation", _metric_class_separation, requires=["class_separation"]),
            MetricSpec("alignment_uniformity", _metric_alignment_uniformity, requires=["alignment_uniformity"]),
        ]

    def run_cell(self, ctx: CellContext) -> dict[str, Any]:
        embedder = ctx.coords["embedder"]
        index_pairs = ctx.data["index_pairs"]
        labels = ctx.data["labels"]
        k = self._k_values[1] if len(self._k_values) > 1 else self._k_values[0]

        # ── Embed (cached for after_run cross-space analysis) ────────
        cache_key = f"emb__{embedder.name}"
        if cache_key not in ctx.cache:
            graphs = [p.G_vuln for p in index_pairs]
            t0 = time.perf_counter()
            embs = embedder.embed_many(graphs)
            embed_time = time.perf_counter() - t0
            print(f"    {embedder.name}: {embs.shape} in {embed_time:.1f}s")

            ctx.cache[cache_key] = {
                "embeddings": embs,
                "embed_time_s": embed_time,
            }

            # Capture pre-PCA for Combined trustworthiness
            if embedder.name == "combined" and hasattr(embedder, "_pca") and embedder._pca is not None:
                raws = np.stack([embedder._raw_one(G) for G in graphs]).astype(np.float32)
                ctx.cache["combined_pre_pca"] = raws
                print(f"    combined pre-PCA: {raws.shape}")

        cached = ctx.cache[cache_key]
        embs = cached["embeddings"]

        # ── Intrinsic metrics ────────────────────────────────────────
        space_stats = embedding_space_stats(embs)
        intrinsic = {
            "dim": int(embs.shape[1]),
            "embed_time_s": cached["embed_time_s"],
            "space_stats": space_stats,
            "isotropy": isotropy(embs),
            "hubness": hubness(embs, k=k),
            "distance_concentration": distance_concentration(embs),
        }

        # Hubness sensitivity across k values
        intrinsic["hubness_by_k"] = {}
        for k_val in self._k_values:
            h = hubness(embs, k=k_val)
            intrinsic["hubness_by_k"][f"k={k_val}"] = {
                "skewness": round(h["k_skewness"], 4),
                "hub_fraction": round(h["hub_fraction"], 4),
            }

        # ── Class separation (requires labels) ──────────────────────
        class_sep = intra_inter_ratio(embs, labels)
        per_class = class_sep.pop("per_class", {})

        align_unif = alignment_uniformity(embs, labels)

        # ── Populate artifacts for MetricSpecs ───────────────────────
        ctx.artifacts["intrinsic"] = intrinsic
        ctx.artifacts["class_separation"] = class_sep
        ctx.artifacts["class_separation_per_cwe"] = per_class
        ctx.artifacts["alignment_uniformity"] = align_unif

        return {
            "embedder": embedder.name,
            "n_samples": len(embs),
        }

    def after_run(self, output: ExperimentOutput) -> None:
        """Compute pairwise cross-space metrics using cached embeddings."""
        print(f"\n{'─'*60}")
        print("Computing pairwise cross-space metrics...")
        print(f"{'─'*60}")

        # Retrieve all cached embeddings
        # The cache is shared — we need to get it from the last cell context
        # Instead, collect embedder names from cells and find cache
        # The cache lives in the run's shared state — we stored it there
        cache = {}
        # Reconstruct from output cells + re-read (cache not directly in output)
        # We stored embeddings in ctx.cache which persists across cells.
        # Access via a trick: after_run is called after all cells, and the
        # Experiment.run() method holds `cache` in scope, but the base class
        # doesn't pass it to after_run. We use the run_dir to find cached data.
        #
        # Better approach: store cross-space results in a dedicated attribute
        # that we populate in the last run_cell call.
        # For now, let's use the internal _cache set in before_run.
        cache = self._cache  # set in before_run

        embedder_names = [
            name.replace("emb__", "")
            for name in cache.keys()
            if name.startswith("emb__")
        ]
        if len(embedder_names) < 2:
            print("  < 2 embedders, skipping pairwise analysis")
            return

        embeddings = {name: cache[f"emb__{name}"]["embeddings"] for name in embedder_names}
        k = self._k_values[1] if len(self._k_values) > 1 else self._k_values[0]

        # ── Pairwise CKA matrix ──────────────────────────────────────
        cka_matrix: dict[str, dict[str, float]] = {}
        for i, name_a in enumerate(embedder_names):
            cka_matrix[name_a] = {}
            for j, name_b in enumerate(embedder_names):
                if i == j:
                    cka_matrix[name_a][name_b] = 1.0
                elif j < i:
                    cka_matrix[name_a][name_b] = cka_matrix[name_b][name_a]
                else:
                    val = linear_cka(embeddings[name_a], embeddings[name_b])
                    cka_matrix[name_a][name_b] = round(val, 4)
                    print(f"  CKA({name_a}, {name_b}) = {val:.4f}")

        # ── Pairwise k-NN overlap matrix ─────────────────────────────
        knn_matrix: dict[str, dict[str, float]] = {}
        for i, name_a in enumerate(embedder_names):
            knn_matrix[name_a] = {}
            for j, name_b in enumerate(embedder_names):
                if i == j:
                    knn_matrix[name_a][name_b] = 1.0
                elif j < i:
                    knn_matrix[name_a][name_b] = knn_matrix[name_b][name_a]
                else:
                    result = knn_overlap(embeddings[name_a], embeddings[name_b], k=k)
                    knn_matrix[name_a][name_b] = round(result["mean_overlap"], 4)
                    print(f"  kNN-overlap({name_a}, {name_b}) = {result['mean_overlap']:.4f}")

        # ── Pairwise rank correlation matrix ─────────────────────────
        rank_corr_matrix: dict[str, dict[str, float]] = {}
        for i, name_a in enumerate(embedder_names):
            rank_corr_matrix[name_a] = {}
            for j, name_b in enumerate(embedder_names):
                if i == j:
                    rank_corr_matrix[name_a][name_b] = 1.0
                elif j < i:
                    rank_corr_matrix[name_a][name_b] = rank_corr_matrix[name_b][name_a]
                else:
                    result = rank_correlation(embeddings[name_a], embeddings[name_b])
                    rank_corr_matrix[name_a][name_b] = round(result["mean_rho"], 4)
                    print(f"  rank_corr({name_a}, {name_b}) = {result['mean_rho']:.4f}")

        # ── Combined PCA trustworthiness ─────────────────────────────
        trust_results = {}
        pre_pca = cache.get("combined_pre_pca")
        if pre_pca is not None and "combined" in embeddings:
            print("\n  Computing Combined PCA trustworthiness...")
            for k_val in self._k_values:
                t_score = trustworthiness(pre_pca, embeddings["combined"], k=k_val)
                trust_results[f"k={k_val}"] = round(t_score, 4)
                print(f"    trustworthiness(k={k_val}) = {t_score:.4f}")

        # ── Save cross-space outputs ─────────────────────────────────
        run_dir = output.run_dir

        full_report = {
            "config": {
                "embedders": embedder_names,
                "k_values": self._k_values,
                "n_index": output.cells[0].metrics.get("n_samples", 0) if output.cells else 0,
                "n_cwe_classes": None,
                "dim": None,
            },
            "pairwise_cka": cka_matrix,
            "pairwise_knn_overlap": knn_matrix,
            "pairwise_rank_correlation": rank_corr_matrix,
            "combined_trustworthiness": trust_results,
            "hubness_sensitivity": {},
        }

        # Populate config from first cell
        if output.cells:
            first = output.cells[0].metrics
            intrinsic = first.get("intrinsic", {})
            full_report["config"]["n_index"] = first.get("n_samples", 0)
            full_report["config"]["dim"] = intrinsic.get("dim")

        # Collect hubness sensitivity from all cells
        for cell in output.cells:
            name = cell.coords.get("embedder")
            emb_name = name.name if hasattr(name, "name") else str(name)
            hub_by_k = cell.metrics.get("intrinsic", {}).get("hubness_by_k", {})
            if hub_by_k:
                full_report["hubness_sensitivity"][emb_name] = hub_by_k

        save_json(full_report, run_dir / "space_analysis.json")
        save_json(cka_matrix, run_dir / "pairwise_cka.json")
        save_json(knn_matrix, run_dir / "pairwise_knn.json")
        save_json(rank_corr_matrix, run_dir / "pairwise_rank_corr.json")

        print(f"\n  Cross-space results written → {run_dir}/space_analysis.json")

        # ── Generate HTML dashboard ──────────────────────────────────
        from experiments.dashboard_scripts.analyze_spaces import render_dashboard

        results_data = output.to_dict()
        html = render_dashboard(results_data, full_report)
        dashboard_path = run_dir / "space_dashboard.html"
        dashboard_path.write_text(html)
        print(f"  Dashboard written → {dashboard_path}")

    def before_run(self, ctx: CellContext) -> None:
        """Store reference to shared cache for after_run access."""
        self._cache = ctx.cache


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from src.io import load_config

    parser = argparse.ArgumentParser(description="Embedding space analysis experiment")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20],
                        help="k values for neighborhood metrics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = SpaceAnalysisExperiment(k_values=args.k)
    experiment.run(cfg)
