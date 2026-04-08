import numpy as np
from .index import FAISSIndex

class Retriever:

    def __init__(self, index: FAISSIndex, top_k: int = 5):
        self.index = index
        self.top_k = top_k

    def query(self, embedding:np.ndarray, top_k:int | None = None) -> list[dict]:
        k = top_k or self.top_k
        vec = embedding.reshape(1, -1).astype(np.float32)

        distances, indices = self.index.index.search(vec, k)

        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            record = dict(self.index.metadata[idx])
            record['score'] = float(dist)
            results.append(record)

        return results
    
    def query_by_cve(self, cve_id: str) -> list[dict]:
        """
        Direct metadata lookup without embedding (for agents)
        """
        return [m for m in self.index.metadata if m['cve_id'] == cve_id]