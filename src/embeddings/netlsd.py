import netlsd
import networkx as nx
import numpy as np

from .base import BaseEmbedder


class NetLSDEmbedder(BaseEmbedder):
    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg, apply_norm)
        # pre-compute timescales once
        self.timescales = np.logspace(-2, 2, self.dim)

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

        desc = netlsd.heat(H, timescales=self.timescales).astype(np.float32)

        if self.apply_norm:
            norm = np.linalg.norm(desc)
            if norm < 1e-8:
                return np.zeros(self.dim, dtype=np.float32)
            return self._norm_vec(desc)
        return desc

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        """Process multiple graphs with progress indicators."""
        results = []
        print(
            f"[NetLSD] Processing {len(graphs)} graphs (CPU-bound, no batching available)..."
        )

        for i, G in enumerate(graphs):
            try:
                results.append(self.embed_one(G))
            except Exception:
                results.append(np.zeros(self.dim, dtype=np.float32))

            if (i + 1) % max(1, len(graphs) // 10) == 0:
                print(f"  [{i + 1}/{len(graphs)}] graphs processed")

        print(f"[NetLSD] Done! Embedded {len(results)}/{len(graphs)} graphs")
        return np.stack(results).astype(np.float32)
