"""
HNSW-backed index using faiss.IndexHNSWFlat.
Drop-in replacement for FAISSIndex — same interface.
"""

import json
from pathlib import Path

import faiss
import numpy as np


class HNSWIndex:
    """
    Hierarchical Navigable Small World index.
    Faster approximate search than flat IP at the cost of slight
    recall degradation. Good trade-off for >10k vectors.

    M:              number of neighbours per node (higher = better recall,
                    more memory). 32 is a good default.
    ef_construction: search depth during build (higher = better recall,
                    slower build). 200 is a good default.
    ef_search:      search depth at query time — can be tuned after build.
    """

    def __init__(
        self,
        dim: int,
        index_path: str,
        metadata_path: str,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ):
        self.dim = dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.ef_search = ef_search
        self.metadata: list[dict] = []

        # HNSW with inner product (cosine on L2-normed vectors)
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def add(self, pair, embedding: np.ndarray, variant: str):
        vec = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        self.metadata.append(
            {
                "cve_id": pair.cve_id,
                "cwe_id": pair.cwe_id,
                "func_name": pair.func_name,
                "project": pair.project,
                "variant": variant,
                "n_nodes": pair.G_vuln.number_of_nodes(),
                **pair.meta,
            }
        )

    def add_raw(self, embedding: np.ndarray, meta: dict):
        """Used by leave-one-out eval — bypasses FunctionPair."""
        vec = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        self.metadata.append(meta)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2))
        print(f"HNSW index saved: {self.index.ntotal} vectors → {self.index_path}")

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.index.hnsw.efSearch = self.ef_search
        self.metadata = json.loads(self.metadata_path.read_text())
        print(f"HNSW index loaded: {self.index.ntotal} vectors")
