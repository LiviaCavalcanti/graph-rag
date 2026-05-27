"""
Shared embedding strategies for combining experiments.

All experiments import from here to guarantee identical code paths.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from src.embeddings.base import BaseEmbedder
from src.embeddings.gin import GINEmbedder
from src.embeddings.netlsd import NetLSDEmbedder
from src.embeddings.wl import WLEmbedder


class _BaseCombined(BaseEmbedder):
    """Shared scaffold: sub-embedders + final PCA."""

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
        return [
            self._netlsd.embed_one(G),
            self._wl.embed_one(G),
            self._gin.embed_one(G),
        ]

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        parts = self._raw_parts(G)
        fused = self._fuse_batch([parts])
        proj = self._pca_final.transform(fused)[0].astype(np.float32)
        return self._norm_vec(proj)

    def _fuse_batch(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        """Override in subclass to define fusion logic."""
        raise NotImplementedError

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        parts_batch = [self._raw_parts(G) for G in graphs]
        fused = self._fuse_batch(parts_batch)

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


class NormConcatPCA(_BaseCombined):
    """L2-normalize each sub-embedder matrix → concatenate → PCA."""

    _strategy = "norm_concat_pca"

    def _fuse_batch(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        n = len(parts_batch)
        mats = [
            np.stack([parts_batch[i][j] for i in range(n)]).astype(np.float32)
            for j in range(3)
        ]
        normed = [normalize(mat, norm="l2") for mat in mats]
        return np.hstack(normed)


class PCAConcatPCA(_BaseCombined):
    """PCA each sub-embedder to dim//3 → concatenate → final PCA.

    Reduces each sub-embedder to dim//3 (42d), genuinely discarding
    low-variance components before fusion.
    """

    _strategy = "pca_concat_pca"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._indiv_dim = self.dim // 3  # 128 // 3 = 42
        self._indiv_pcas: list[PCA] = [
            PCA(n_components=self._indiv_dim, random_state=42) for _ in range(3)
        ]
        self._indiv_fitted = False

    def _fuse_batch(self, parts_batch: list[list[np.ndarray]]) -> np.ndarray:
        n = len(parts_batch)
        mats = [
            np.stack([parts_batch[i][j] for i in range(n)]).astype(np.float32)
            for j in range(3)
        ]

        if not self._indiv_fitted:
            for idx, (pca, mat) in enumerate(zip(self._indiv_pcas, mats)):
                n_comp = min(self._indiv_dim, mat.shape[0] - 1, mat.shape[1])
                if n_comp < self._indiv_dim:
                    self._indiv_pcas[idx] = PCA(n_components=n_comp, random_state=42)
                    pca = self._indiv_pcas[idx]
                pca.fit(mat)
                ev = pca.explained_variance_ratio_.sum()
                print(f"      individual PCA {mat.shape[1]}d → {pca.n_components_}d — explained variance: {ev:.2%}")
            self._indiv_fitted = True

        reduced = [pca.transform(mat).astype(np.float32) for pca, mat in zip(self._indiv_pcas, mats)]
        return np.hstack(reduced)


class NormPCAConcat(_BaseCombined):
    """L2-normalize each → PCA each individually (128→42d) → concat.

    No final PCA — the concatenated 126d vectors are used directly.
    """

    _strategy = "norm_pca_concat"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._indiv_dim = self.dim // 3  # 42
        self._indiv_pcas: list[PCA] = [
            PCA(n_components=self._indiv_dim, random_state=42) for _ in range(3)
        ]
        self._indiv_fitted = False

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit individual PCAs")
        parts = self._raw_parts(G)
        reduced_parts = []
        for pca, part in zip(self._indiv_pcas, parts):
            normed = part / (np.linalg.norm(part) + 1e-8)
            reduced = pca.transform(normed.reshape(1, -1))[0].astype(np.float32)
            reduced_parts.append(reduced)
        fused = np.concatenate(reduced_parts)
        return self._norm_vec(fused)

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        parts_batch = [self._raw_parts(G) for G in graphs]
        n = len(parts_batch)
        mats = [
            np.stack([parts_batch[i][j] for i in range(n)]).astype(np.float32)
            for j in range(3)
        ]
        normed = [normalize(mat, norm="l2") for mat in mats]

        if not self._indiv_fitted:
            for idx, (pca, mat) in enumerate(zip(self._indiv_pcas, normed)):
                n_comp = min(self._indiv_dim, mat.shape[0] - 1, mat.shape[1])
                if n_comp < self._indiv_dim:
                    self._indiv_pcas[idx] = PCA(n_components=n_comp, random_state=42)
                    pca = self._indiv_pcas[idx]
                pca.fit(mat)
                ev = pca.explained_variance_ratio_.sum()
                print(f"      [{self._strategy}] individual PCA {mat.shape[1]}d → {pca.n_components_}d — explained variance: {ev:.2%}")
            self._indiv_fitted = True
            self._fitted = True
            self.dim = sum(pca.n_components_ for pca in self._indiv_pcas)
            print(f"    [{self._strategy}] final dim = {self.dim}")

        reduced = [pca.transform(mat).astype(np.float32) for pca, mat in zip(self._indiv_pcas, normed)]
        fused = np.hstack(reduced)
        return self._norm_mat(fused)
