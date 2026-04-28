import networkx as nx
import numpy as np
from sklearn.decomposition import PCA

from .base import BaseEmbedder
from .gin import GINEmbedder
from .netlsd import NetLSDEmbedder
from .wl import WLEmbedder


class CombinedEmbedder(BaseEmbedder):
    """
    Concatenates NetLSD + WL + GIN then reduces with PCA.
    The PCA is fit on the first batch seen — refit if the
    data distribution changes significantly.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._netlsd = NetLSDEmbedder(cfg)
        self._wl = WLEmbedder(cfg)
        self._gin = GINEmbedder(cfg)
        self._pca = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "combined"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        raw = self._raw_one(G)
        if self.projection == "none":
            return self._norm_vec(raw)
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        proj = self._pca.transform(raw.reshape(1, -1))[0].astype(np.float32)
        return self._norm_vec(proj)

    def _raw_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        a = self._netlsd.embed_one(G)
        b = self._wl.embed_one(G)
        c = self._gin.embed_one(G)
        return np.concatenate([a, b, c])

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        raws = np.stack([self._raw_one(G) for G in graphs]).astype(np.float32)

        if self.projection == "none":
            self.dim = raws.shape[1]
            print(f"    [combined] no projection — dim={self.dim}")
            return self._norm_mat(raws)

        if not self._fitted:
            self._pca = PCA(n_components=self.dim, random_state=42)
            self._pca.fit(raws)
            self._fitted = True
            explained = self._pca.explained_variance_ratio_.sum()
            print(f"    [combined] PCA fitted — explained variance: {explained:.2%}")

        projected = self._pca.transform(raws).astype(np.float32)
        return self._norm_mat(projected)
