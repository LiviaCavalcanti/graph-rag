"""
Embedding Combination Strategy Experiment.

Compares three strategies for fusing NetLSD + WL + GIN embeddings:

  1. concat_pca       (baseline) — concatenate raw → PCA
  2. pca_concat_pca   — PCA each individual → concatenate → PCA
  3. norm_concat_pca  — L2-norm each individual → concatenate → PCA

Grid: strategy (single axis — one cell per combination strategy).

Metrics per cell:
  - Intrinsic space quality (effective dim, mean pairwise similarity, isotropy, hubness)
  - Self-retrieval hit@k / MRR / nDCG / MAP
  - Class separation (intra/inter CWE ratio)

Outputs:
  <run_dir>/results.json          — base class output
  <run_dir>/combining_report.json — summary comparison table

Usage:
    python -m experiments.combining_experiment [--config config.yaml]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from experiments.base import Axis, CellContext, Experiment, ExperimentOutput, MetricSpec
from experiments.common import build_hnsw, evaluate_retrieval, save_json
from src.embeddings.base import BaseEmbedder
from src.embeddings.gin import GINEmbedder
from src.embeddings.gin_codebert import GINCodeBERTEmbedder
from src.embeddings.gin_struct import GINStructEmbedder
from src.embeddings.netlsd import NetLSDEmbedder
from src.embeddings.wl import WLEmbedder
from src.embeddings.vuln_pattern import CodeBERTPatternEmbedder
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
)


# ── Combination strategy variants ───────────────────────────────────
#
# All three share the same sub-embedders (netlsd, wl, gin) and final
# PCA reduction to `dim`.  They differ only in what happens between
# the individual embed_one outputs and the final PCA.


class _BaseCombined(BaseEmbedder):
    """Shared scaffold for the three combination strategies."""

    _strategy: str = ""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._netlsd = NetLSDEmbedder(cfg, apply_norm=False)
        self._wl = WLEmbedder(cfg, apply_norm=False)
        self._gin = GINEmbedder(cfg, apply_norm=False)
        self._pca_final: PCA | None = None
        self._fitted = False

    @property
    def name(self) -> str:
        return self._strategy

    def _raw_parts(self, G: nx.MultiDiGraph) -> list[np.ndarray]:
        """Return individual embedding vectors (before any fusion)."""
        return [
            self._netlsd.embed_one(G),
            self._wl.embed_one(G),
            self._gin.embed_one(G),
        ]

    def _fuse(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        """Strategy-specific fusion. Returns (n_samples, concat_dim) matrix.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        parts = self._raw_parts(G)
        fused = self._fuse([parts])  # (1, concat_dim)
        proj = self._pca_final.transform(fused)[0].astype(np.float32)
        return self._norm_vec(proj)

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        parts_batch = [self._raw_parts(G) for G in graphs]
        fused = self._fuse(parts_batch)  # (n, concat_dim)

        if not self._fitted:
            n_components = min(self.dim, fused.shape[0], fused.shape[1])
            self._pca_final = PCA(n_components=n_components, random_state=42)
            self._pca_final.fit(fused)
            self._fitted = True
            self.dim = n_components
            explained = self._pca_final.explained_variance_ratio_.sum()
            print(f"    [{self._strategy}] final PCA fitted — dim={n_components}, explained variance: {explained:.2%}")

        projected = self._pca_final.transform(fused).astype(np.float32)
        return self._norm_mat(projected)


class ConcatPCA(_BaseCombined):
    """Baseline: concatenate raw embeddings → PCA."""

    _strategy = "concat_pca"

    def _fuse(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        return np.stack([np.concatenate(parts) for parts in parts_batch]).astype(np.float32)


class PCAConcatPCA(_BaseCombined):
    """PCA each individual embedding → concatenate → PCA."""

    _strategy = "pca_concat_pca"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._indiv_pcas: list[PCA] = [
            PCA(n_components=self.dim, random_state=42) for _ in range(3)
        ]
        self._indiv_fitted = False

    def _fuse(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        # Separate into per-embedder matrices
        n = len(parts_batch)
        mats = [
            np.stack([parts_batch[i][j] for i in range(n)]).astype(np.float32)
            for j in range(3)
        ]

        if not self._indiv_fitted:
            for pca, mat in zip(self._indiv_pcas, mats):
                pca.fit(mat)
                ev = pca.explained_variance_ratio_.sum()
                print(f"      individual PCA dim={self.dim} — explained variance: {ev:.2%}")
            self._indiv_fitted = True

        reduced = [pca.transform(mat).astype(np.float32) for pca, mat in zip(self._indiv_pcas, mats)]
        return np.hstack(reduced)


class PCAConcat(_BaseCombined):
    """PCA each individual embedding to dim//3 → concatenate (no second PCA)."""

    _strategy = "pca_concat"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._indiv_dim = self.dim // 3  # 128 // 3 = 42
        self._indiv_pcas: list[PCA] = [
            PCA(n_components=self._indiv_dim, random_state=42) for _ in range(3)
        ]
        self._indiv_fitted = False

    def _fuse(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        n = len(parts_batch)
        mats = [
            np.stack([parts_batch[i][j] for i in range(n)]).astype(np.float32)
            for j in range(3)
        ]

        if not self._indiv_fitted:
            for pca, mat in zip(self._indiv_pcas, mats):
                pca.fit(mat)
                ev = pca.explained_variance_ratio_.sum()
                print(f"      individual PCA dim={self._indiv_dim} — explained variance: {ev:.2%}")
            self._indiv_fitted = True

        reduced = [pca.transform(mat).astype(np.float32) for pca, mat in zip(self._indiv_pcas, mats)]
        return np.hstack(reduced)

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if not self._indiv_fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        parts = self._raw_parts(G)
        fused = self._fuse([parts])  # (1, indiv_dim*3)
        return self._norm_vec(fused[0])

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        parts_batch = [self._raw_parts(G) for G in graphs]
        fused = self._fuse(parts_batch)  # (n, indiv_dim*3)
        self.dim = fused.shape[1]
        self._fitted = True
        print(f"    [{self._strategy}] no final PCA — dim={self.dim}")
        return self._norm_mat(fused)


class NormConcatPCA(_BaseCombined):
    """L2-normalize each individual embedding → concatenate → PCA."""

    _strategy = "norm_concat_pca"

    def _fuse(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        n = len(parts_batch)
        mats = [
            np.stack([parts_batch[i][j] for i in range(n)]).astype(np.float32)
            for j in range(3)
        ]
        normed = [normalize(mat, norm="l2") for mat in mats]
        return np.hstack(normed)


class NormConcatPCA_GINStruct(_BaseCombined):
    """Like norm_concat_pca but with trained GIN-Struct instead of frozen GIN."""

    _strategy = "norm_concat_pca_gin_struct"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Replace frozen GIN with trained GIN-Struct
        self._gin = GINStructEmbedder(cfg, apply_norm=False)

    def _fuse(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        n = len(parts_batch)
        mats = [
            np.stack([parts_batch[i][j] for i in range(n)]).astype(np.float32)
            for j in range(3)
        ]
        normed = [normalize(mat, norm="l2") for mat in mats]
        return np.hstack(normed)


# ── 4-way variants (NetLSD + WL + GIN + CodeBERT-pattern) ───────────
#
# CodeBERT-pattern produces 802d raw vectors (34 pattern + 768 CodeBERT).
# Naively concatenating raw 802d with 3×128d = 1186d lets CodeBERT
# dominate PCA (68% of dimensions).  Fix: PCA CodeBERT-pattern to 128d
# first, so all 4 sub-embedders contribute equally (512d → PCA to 128d).


class _BaseCombined4(BaseEmbedder):
    """Scaffold for 4-way fusion including CodeBERT-pattern."""

    _strategy: str = ""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._netlsd = NetLSDEmbedder(cfg, apply_norm=False)  # let final PCA handle scaling
        self._wl = WLEmbedder(cfg, apply_norm=False)
        self._gin = GINEmbedder(cfg, apply_norm=False)
        self._cbpat = CodeBERTPatternEmbedder(cfg, apply_norm=False)
        self._pca_final: PCA | None = None
        self._pca_cb: PCA | None = None
        self._cb_pca_fitted = False
        self._fitted = False

    @property
    def name(self) -> str:
        return self._strategy

    def _raw_parts(self, G: nx.MultiDiGraph) -> list[np.ndarray]:
        return [
            self._netlsd.embed_one(G),
            self._wl.embed_one(G),
            self._gin.embed_one(G),
        ]

    def _raw_parts_batch(self, graphs: list[nx.MultiDiGraph]) -> list[list[np.ndarray]]:
        """Embed all graphs; PCA CodeBERT-pattern to 128d for balanced fusion."""
        # Structural: per-graph (each already 128d)
        struct_parts = [self._raw_parts(G) for G in graphs]

        # CodeBERT-pattern: get raw 802d, then PCA to 128d ourselves
        cbpat_raw, valid_idx = self._cbpat._build_raw(graphs)

        if not self._cb_pca_fitted:
            valid_raw = cbpat_raw[valid_idx] if valid_idx else cbpat_raw
            n_comp = min(self.dim, valid_raw.shape[0] - 1, valid_raw.shape[1])
            self._pca_cb = PCA(n_components=n_comp, random_state=42)
            self._pca_cb.fit(valid_raw)
            self._cb_pca_fitted = True
            expl = self._pca_cb.explained_variance_ratio_.sum()
            print(f"      [4way] CodeBERT-pattern PCA: 802d → {n_comp}d, explained variance: {expl:.2%}")

        cbpat_reduced = self._pca_cb.transform(cbpat_raw).astype(np.float32)
        # Pad if n_comp < self.dim
        if cbpat_reduced.shape[1] < self.dim:
            padded = np.zeros((cbpat_reduced.shape[0], self.dim), dtype=np.float32)
            padded[:, :cbpat_reduced.shape[1]] = cbpat_reduced
            cbpat_reduced = padded
        # L2 normalize each row
        cbpat_reduced = normalize(cbpat_reduced, norm="l2")

        # Append as 4th part
        for i, parts in enumerate(struct_parts):
            parts.append(cbpat_reduced[i])
        return struct_parts

    def _cb_embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        """PCA-reduced CodeBERT-pattern for a single graph."""
        if not self._cb_pca_fitted:
            raise RuntimeError("Call embed_many() first to fit CodeBERT PCA")
        raw, _ = self._cbpat._build_raw([G])
        reduced = self._pca_cb.transform(raw)[0].astype(np.float32)
        if reduced.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:reduced.shape[0]] = reduced
            reduced = padded
        norm = np.linalg.norm(reduced)
        return (reduced / (norm + 1e-8)).astype(np.float32)

    def _fuse(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        raise NotImplementedError

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        parts = self._raw_parts(G)
        parts.append(self._cb_embed_one(G))
        fused = self._fuse([parts])
        proj = self._pca_final.transform(fused)[0].astype(np.float32)
        return self._norm_vec(proj)

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        parts_batch = self._raw_parts_batch(graphs)
        fused = self._fuse(parts_batch)

        if not self._fitted:
            n_components = min(self.dim, fused.shape[0], fused.shape[1])
            self._pca_final = PCA(n_components=n_components, random_state=42)
            self._pca_final.fit(fused)
            self._fitted = True
            self.dim = n_components
            explained = self._pca_final.explained_variance_ratio_.sum()
            print(f"    [{self._strategy}] final PCA fitted — dim={n_components}, explained variance: {explained:.2%}")

        projected = self._pca_final.transform(fused).astype(np.float32)
        return self._norm_mat(projected)


class ConcatPCA4(_BaseCombined4):
    """4-way: concatenate raw NetLSD+WL+GIN+CodeBERT-pattern → PCA."""

    _strategy = "4way_concat_pca"

    def _fuse(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        return np.stack([np.concatenate(parts) for parts in parts_batch]).astype(np.float32)


class NormConcatPCA4(_BaseCombined4):
    """4-way: L2-norm each → concatenate → PCA."""

    _strategy = "4way_norm_concat_pca"

    def _fuse(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        n = len(parts_batch)
        mats = [
            np.stack([parts_batch[i][j] for i in range(n)]).astype(np.float32)
            for j in range(4)
        ]
        normed = [normalize(mat, norm="l2") for mat in mats]
        return np.hstack(normed)


STRATEGY_CLASSES = {
    "concat_pca": ConcatPCA,
    "pca_concat_pca": PCAConcatPCA,
    "pca_concat": PCAConcat,
    "norm_concat_pca": NormConcatPCA,
    "norm_concat_pca_gin_struct": NormConcatPCA_GINStruct,
    "4way_concat_pca": ConcatPCA4,
    "4way_norm_concat_pca": NormConcatPCA4,
    "gin_codebert": GINCodeBERTEmbedder,
}


# ── helpers ──────────────────────────────────────────────────────────


def _encode_labels(pairs: list) -> np.ndarray:
    cwe_list = [p.cwe_id or "UNKNOWN" for p in pairs]
    unique = sorted(set(cwe_list))
    mapping = {c: i for i, c in enumerate(unique)}
    return np.array([mapping[c] for c in cwe_list])


# ── MetricSpec functions ─────────────────────────────────────────────


def _metric_intrinsic(ctx: CellContext) -> dict:
    return ctx.artifacts["intrinsic"]


def _metric_retrieval(ctx: CellContext) -> dict:
    return ctx.artifacts["retrieval"]


def _metric_class_separation(ctx: CellContext) -> dict:
    return ctx.artifacts["class_separation"]


def _metric_alignment_uniformity(ctx: CellContext) -> dict:
    return ctx.artifacts["alignment_uniformity"]


# ── Experiment class ─────────────────────────────────────────────────


class CombiningExperiment(Experiment):
    """Compare embedding combination strategies."""

    def __init__(self):
        self._cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "combining"

    def load_data(self, cfg: dict) -> dict[str, Any]:
        from experiments.common import build_split
        from src.data.autopatch import load_pairs

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
        emb_cfg = cfg.get("embeddings", {})
        strategies = [cls(emb_cfg) for cls in STRATEGY_CLASSES.values()]
        return [
            Axis("strategy", strategies, description="Combination strategy"),
        ]

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec("intrinsic", _metric_intrinsic, requires=["intrinsic"]),
            MetricSpec("retrieval", _metric_retrieval, requires=["retrieval"]),
            MetricSpec("class_separation", _metric_class_separation, requires=["class_separation"]),
            MetricSpec("alignment_uniformity", _metric_alignment_uniformity, requires=["alignment_uniformity"]),
        ]

    def before_run(self, ctx: CellContext) -> None:
        """Store reference to shared cache for after_run access."""
        self._cache = ctx.cache

    def run_cell(self, ctx: CellContext) -> dict[str, Any]:
        strategy: _BaseCombined = ctx.coords["strategy"]
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

        # Cache embeddings for cross-space analysis in after_run
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
        index, retriever = build_hnsw(
            index_pairs, index_embs, strategy.name, strategy.dim, ctx.run_dir,
            tag=strategy.name,
        )
        retrieval = evaluate_retrieval(
            query_pairs, query_embs, retriever, index_pairs, ks=ks,
        )
        retrieval.pop("raw_queries", None)  # drop verbose per-query detail

        # ── Store artifacts for MetricSpecs ──────────────────────────
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
        """Write comparison table + space analysis dashboard."""
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

        # Stringify strategy names
        for row in rows:
            s = row["strategy"]
            row["strategy"] = s.name if hasattr(s, "name") else str(s)

        report = {"comparison": rows}
        save_json(report, output.run_dir / "combining_report.json")

        # Print summary table
        print(f"\n{'='*70}")
        print("Combination Strategy Comparison")
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
        print(f"\nFull report → {output.run_dir}/combining_report.json")

        # ── Generate space analysis dashboard ────────────────────────
        self._generate_dashboard(output, self._cache)

    @staticmethod
    def _generate_dashboard(output: ExperimentOutput, cache: dict[str, Any]) -> None:
        """Adapt output to the analyze_spaces dashboard format and render."""
        from experiments.dashboard_scripts.analyze_spaces import render_dashboard

        # The dashboard expects cell["coords"]["embedder"] — remap strategy → embedder
        strategy_names = []
        adapted_cells = []
        for cell in output.cells:
            s = cell.coords.get("strategy", "?")
            name = s.name if hasattr(s, "name") else str(s)
            strategy_names.append(name)
            adapted_cells.append({
                "coords": {"embedder": name},
                "metrics": cell.metrics,
                "timing_s": cell.timing_s,
            })

        results_data = {"cells": adapted_cells}

        # Build space_analysis dict with pairwise metrics
        first_intrinsic = (output.cells[0].metrics.get("intrinsic", {})
                           if output.cells else {})

        # Compute pairwise matrices from cached embeddings
        cka_matrix, knn_matrix, rank_matrix = {}, {}, {}
        embeddings = {}
        for name in strategy_names:
            key = f"emb__{name}"
            if key in cache:
                embeddings[name] = cache[key]

        if len(embeddings) >= 2:
            print("\n  Computing pairwise cross-space metrics...")
            for i, name_a in enumerate(strategy_names):
                cka_matrix.setdefault(name_a, {})
                knn_matrix.setdefault(name_a, {})
                rank_matrix.setdefault(name_a, {})
                for j, name_b in enumerate(strategy_names):
                    if name_a not in embeddings or name_b not in embeddings:
                        continue
                    if i == j:
                        cka_matrix[name_a][name_b] = 1.0
                        knn_matrix[name_a][name_b] = 1.0
                        rank_matrix[name_a][name_b] = 1.0
                    elif j < i:
                        cka_matrix[name_a][name_b] = cka_matrix[name_b][name_a]
                        knn_matrix[name_a][name_b] = knn_matrix[name_b][name_a]
                        rank_matrix[name_a][name_b] = rank_matrix[name_b][name_a]
                    else:
                        emb_a, emb_b = embeddings[name_a], embeddings[name_b]
                        cka_val = linear_cka(emb_a, emb_b)
                        cka_matrix[name_a][name_b] = round(cka_val, 4)
                        knn_val = knn_overlap(emb_a, emb_b, k=10)
                        knn_matrix[name_a][name_b] = round(knn_val["mean_overlap"], 4)
                        rc_val = rank_correlation(emb_a, emb_b)
                        rank_matrix[name_a][name_b] = round(rc_val["mean_rho"], 4)
                        print(f"    CKA({name_a}, {name_b})={cka_val:.4f}  "
                              f"kNN={knn_val['mean_overlap']:.4f}  "
                              f"rank_corr={rc_val['mean_rho']:.4f}")

        space_analysis = {
            "config": {
                "embedders": strategy_names,
                "n_index": output.cells[0].metrics.get("n_index")
                           if output.cells else 0,
                "n_cwe_classes": None,
                "dim": first_intrinsic.get("dim"),
            },
            "pairwise_cka": cka_matrix,
            "pairwise_knn_overlap": knn_matrix,
            "pairwise_rank_correlation": rank_matrix,
            "combined_trustworthiness": {},
            "hubness_sensitivity": {},
        }

        html = render_dashboard(results_data, space_analysis)
        dashboard_path = output.run_dir / "space_dashboard.html"
        dashboard_path.write_text(html)
        print(f"\n  Space dashboard → {dashboard_path}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from src.io import load_config

    parser = argparse.ArgumentParser(description="Embedding combination strategy experiment")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = CombiningExperiment()
    experiment.run(cfg)
