import netlsd
import networkx as nx
import numpy as np

from .base import BaseEmbedder


class NetLSDEmbedder(BaseEmbedder):
    @property
    def name(self) -> str:
        return "netlsd"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if G.number_of_nodes() < 3:
            return np.zeros(self.dim, dtype=np.float32)

        H = nx.Graph(G)
        # remove isolates — they contribute nothing to the Laplacian spectrum
        # but inflate the zero eigenvalue and collapse the descriptor
        H.remove_nodes_from(list(nx.isolates(H)))
        if H.number_of_nodes() < 3:
            return np.zeros(self.dim, dtype=np.float32)

        timescales = np.logspace(-2, 2, self.dim)
        desc = netlsd.heat(H, timescales=timescales).astype(np.float32)

        norm = np.linalg.norm(desc)
        if norm < 1e-8:
            return np.zeros(self.dim, dtype=np.float32)
        return self._norm_vec(desc)
