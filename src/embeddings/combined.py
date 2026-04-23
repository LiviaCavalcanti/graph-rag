import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import networkx as nx
from .base import BaseEmbedder
from .netlsd import NetLSDEmbedder
from .wl import WLEmbedder
from .gin import GINEmbedder


class CombinedEmbedder(BaseEmbedder):
    """
    Concatenates NetLSD + WL + GIN then reduces with PCA.
    The PCA is fit on the first batch seen — refit if the
    data distribution changes significantly.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._netlsd = NetLSDEmbedder(cfg)
        self._wl     = WLEmbedder(cfg)
        self._gin    = GINEmbedder(cfg)
        self._pca    = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "combined"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        # used after fit — returns PCA-projected vector
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        raw = self._raw_one(G)
        return self._pca.transform(raw.reshape(1, -1))[0].astype(np.float32)

    def _raw_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        a = self._netlsd.embed_one(G)
        b = self._wl.embed_one(G)
        c = self._gin.embed_one(G)
        return np.concatenate([a, b, c])

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        raws = np.stack([self._raw_one(G) for G in graphs]).astype(np.float32)

        if not self._fitted:
            self._pca = PCA(n_components=self.dim, random_state=42)
            self._pca.fit(raws)
            self._fitted = True
            explained = self._pca.explained_variance_ratio_.sum()
            print(f"    [combined] PCA fitted — explained variance: {explained:.2%}")

        projected = self._pca.transform(raws).astype(np.float32)
        return normalize(projected, norm='l2')

    def save_pca(self, path: str | Path) -> None:
        """Persist the fitted PCA model to disk."""
        if not self._fitted:
            raise RuntimeError("PCA not fitted yet — call embed_many() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pca, f)
        print(f"    [combined] PCA saved → {path}")

    def load_pca(self, path: str | Path) -> None:
        """Load a previously fitted PCA model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            self._pca = pickle.load(f)
        self._fitted = True
        print(f"    [combined] PCA loaded ← {path}")