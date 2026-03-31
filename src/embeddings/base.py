from abc import ABC, abstractmethod
import numpy as np
import networkx as nx


class BaseEmbedder(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dim = cfg.get("dim", 128)

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
