"""
OracleRetriever — perfect retrieval (same-CVE lookup, no embeddings needed).
"""

from __future__ import annotations

from collections import defaultdict


class OracleRetriever:
    """Returns same-CVE pair from a list of index pairs (perfect retrieval).

    No embeddings, no index — just a dict lookup by cve_id.
    """

    def __init__(self, index_pairs: list):
        self._by_cve: dict[str, list] = defaultdict(list)
        for p in index_pairs:
            self._by_cve[p.cve_id].append(p)

    def retrieve(self, query_pair) -> tuple[object | None, dict]:
        """Return the best same-CVE match from the index."""
        candidates = self._by_cve.get(query_pair.cve_id, [])
        if not candidates:
            return None, {"cve_match": False, "cwe_match": False}
        # prefer original variant
        original = [p for p in candidates if p.meta.get("variant") == "original"]
        best = original[0] if original else candidates[0]
        return best, {
            "cve_match": True,
            "cwe_match": best.cwe_id == query_pair.cwe_id,
            "retrieved_variant": best.meta.get("variant", "unknown"),
        }
