from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import VectorIndex


class Retriever:

    def __init__(self, index: VectorIndex, top_k: int = 5):
        self.index = index
        self.top_k = top_k

    def query(self, embedding: np.ndarray, top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        vec = embedding.reshape(1, -1).astype(np.float32)

        distances, indices = self.index.index.search(vec, k)

        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            record = dict(self.index.metadata[idx])
            record["score"] = float(dist)
            results.append(record)

        return results

    def query_by_cve(self, cve_id: str) -> list[dict]:
        """
        Direct metadata lookup without embedding (for agents)
        """
        return [m for m in self.index.metadata if m["cve_id"] == cve_id]


class EmbeddingRetriever:
    """FAISS-based retriever with the same ``retrieve(query_pair)`` interface
    as :class:`~src.rag.oracle.OracleRetriever`.

    Wraps an embedder, a FAISS :class:`Retriever`, and a pair lookup table so
    that :func:`src.agents.batch_inference.run_batch_inference` can use it as a
    drop-in replacement for the oracle retriever.
    """

    def __init__(self, embedder, retriever: Retriever, index_pairs: list):
        self.embedder = embedder
        self.retriever = retriever

        # build (cve_id, variant) → FunctionPair lookup
        self._by_key: dict[tuple[str, str], object] = {}
        self._by_cve: dict[str, list] = defaultdict(list)
        for p in index_pairs:
            self._by_key[(p.cve_id, p.meta.get("variant", ""))] = p
            self._by_cve[p.cve_id].append(p)

    def _lookup_pair(self, meta: dict):
        """Resolve FAISS metadata back to its FunctionPair."""
        key = (meta.get("cve_id"), meta.get("variant", ""))
        pair = self._by_key.get(key)
        if pair is not None:
            return pair
        candidates = self._by_cve.get(meta.get("cve_id"), [])
        return candidates[0] if candidates else None

    def retrieve(self, query_pair) -> tuple:
        """Embed the query graph and return the best FAISS match.

        Returns ``(example_pair | None, retrieval_info)`` — same contract as
        :meth:`OracleRetriever.retrieve`.
        """
        try:
            q_emb = self.embedder.embed_one(query_pair.G_vuln)
        except Exception as e:
            return None, {"error": str(e), "cve_match": False, "cwe_match": False}

        results = self.retriever.query(q_emb)
        if not results:
            return None, {"cve_match": False, "cwe_match": False}

        top = results[0]
        example_pair = self._lookup_pair(top)
        if example_pair is None:
            return None, {"cve_match": False, "cwe_match": False, "top1": top}

        cve_match = example_pair.cve_id == query_pair.cve_id
        cwe_match = example_pair.cwe_id == query_pair.cwe_id

        return example_pair, {
            "cve_match": cve_match,
            "cwe_match": cwe_match,
            "retrieved_variant": example_pair.meta.get("variant", "unknown"),
            "score": top.get("score", 0.0),
        }
