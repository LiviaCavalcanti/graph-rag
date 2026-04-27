from abc import ABC, abstractmethod

import networkx as nx
import numpy as np


class BaseEmbedder(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dim = cfg.get("dim", 128)
        self.projection = cfg.get("projection", "pca")  # "pca" or "none"
        self.l2_normalize = cfg.get("l2_normalize", True)

    def _norm_vec(self, v: np.ndarray) -> np.ndarray:
        """L2-normalize a single vector if l2_normalize is enabled."""
        if not self.l2_normalize:
            return v.astype(np.float32)
        norm = np.linalg.norm(v)
        return (v / (norm + 1e-8)).astype(np.float32)

    def _norm_mat(self, M: np.ndarray) -> np.ndarray:
        """L2-normalize each row of a matrix if l2_normalize is enabled."""
        if not self.l2_normalize:
            return M.astype(np.float32)
        from sklearn.preprocessing import normalize
        return normalize(M.astype(np.float32), norm='l2')

    @abstractmethod
    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray: ...

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        results = []
        for G in graphs:
            try:
                results.append(self.embed_one(G))
            except Exception:
                results.append(np.zeros(self.dim, dtype=np.float32))

        return np.stack(results).astype(np.float32)

    @property
    @abstractmethod
    def name(self) -> str: ...
