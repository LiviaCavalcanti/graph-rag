import netlsd
import networkx as nx
import numpy as np

from .base import BaseEmbedder


class NetLSDEmbedder(BaseEmbedder):
    @property
    def name(self) -> str:
        return "netlsd"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if G.number_of_nodes() == 0:
            return np.zeros(self.dim, dtype=np.float32)

        H = nx.Graph(G)
        timescales = np.logspace(-2, 2, self.dim)
        desc = netlsd.heat(H, timescales=timescales)
        desc = desc.astype(np.float32)

        norm = np.linalg.norm(desc)
        return desc / (norm + 1e-8)
