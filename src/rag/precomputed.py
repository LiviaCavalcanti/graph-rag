"""
PrecomputedRetriever — reads pre-computed query results from ``--mode query``.

Provides the same ``.retrieve(query_pair)`` interface as OracleRetriever and
EmbeddingRetriever so it can be used as a drop-in for batch inference without
any embedding or FAISS overhead.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


class PrecomputedRetriever:
    """Retriever backed by a results.jsonl file from ``--mode query``."""

    def __init__(self, query_results_path: Path):
        # Key: (cve_id, variant, dir_name).  dir_name is included to avoid
        # collisions when a CVE has multiple functions that all share the same
        # (cve_id, variant) pair — without it only the last row wins.
        # Old JSONL files (pre-fix) omit query_dir; they use dir_name="" and
        # are handled by the fallback in retrieve().
        self._lookup: dict[tuple[str, str, str], dict] = {}
        with open(query_results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["query_cve"], rec.get("query_variant", ""), rec.get("query_dir", ""))
                self._lookup[key] = rec
        print(f"PrecomputedRetriever: loaded {len(self._lookup)} query results")

    def retrieve(self, query_pair) -> tuple:
        """Look up the pre-computed retrieval result for *query_pair*.

        Returns ``(example_pair | None, retrieval_info)`` — same contract as
        :meth:`OracleRetriever.retrieve`.
        """
        variant = query_pair.meta.get("variant", "")
        dir_name = query_pair.meta.get("dir_name", "")
        # Try the precise 3-part key first (new JSONL files that store query_dir).
        # Fall back to the legacy 2-part key (old files without query_dir) so
        # that existing results.jsonl files remain usable.
        rec = self._lookup.get((query_pair.cve_id, variant, dir_name))
        if rec is None and dir_name:
            rec = self._lookup.get((query_pair.cve_id, variant, ""))
        if rec is None or rec.get("status") not in ("retrieved", "success"):
            return None, {"cve_match": False, "cwe_match": False}

        example = SimpleNamespace(
            cve_id=rec.get("example_cve"),
            cwe_id=rec.get("example_cwe"),
            meta={
                "variant": rec.get("example_variant", ""),
                "dir_name": rec.get("example_dir", ""),
            },
        )

        retrieval = rec.get("retrieval", {})
        return example, retrieval
