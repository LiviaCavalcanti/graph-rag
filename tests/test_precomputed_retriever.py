"""Tests for PrecomputedRetriever.

Verifies that the retriever correctly loads query results from a JSONL file
and returns the right example pairs with dir_name metadata.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from rag.precomputed import PrecomputedRetriever


# ── helpers ───────────────────────────────────────────────────────────


def _write_results(path: Path, records: list[dict]):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_query_pair(cve_id="CVE-2025-0001", variant="original"):
    return SimpleNamespace(
        cve_id=cve_id,
        meta={"variant": variant},
    )


SAMPLE_RESULT = {
    "query_cve": "CVE-2025-0001",
    "query_variant": "original",
    "status": "retrieved",
    "example_cve": "CVE-2025-0099",
    "example_cwe": "CWE-119",
    "example_variant": "augmented",
    "example_dir": "CVE-2025-0099_1",
    "retrieval": {"cve_match": False, "cwe_match": True, "distance": 0.42},
}


# ── loading ───────────────────────────────────────────────────────────


class TestPrecomputedRetrieverLoading:

    def test_loads_all_records(self, tmp_path):
        path = tmp_path / "results.jsonl"
        records = [
            {**SAMPLE_RESULT, "query_cve": f"CVE-2025-{i:04d}"}
            for i in range(5)
        ]
        _write_results(path, records)
        retriever = PrecomputedRetriever(path)
        assert len(retriever._lookup) == 5

    def test_empty_file(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text("")
        retriever = PrecomputedRetriever(path)
        assert len(retriever._lookup) == 0

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text(json.dumps(SAMPLE_RESULT) + "\n\n\n")
        retriever = PrecomputedRetriever(path)
        assert len(retriever._lookup) == 1


# ── retrieve ──────────────────────────────────────────────────────────


class TestPrecomputedRetrieverRetrieve:

    def _build_retriever(self, tmp_path, records):
        path = tmp_path / "results.jsonl"
        _write_results(path, records)
        return PrecomputedRetriever(path)

    def test_returns_matching_example(self, tmp_path):
        retriever = self._build_retriever(tmp_path, [SAMPLE_RESULT])
        query = _make_query_pair("CVE-2025-0001", "original")
        example, info = retriever.retrieve(query)
        assert example is not None
        assert example.cve_id == "CVE-2025-0099"

    def test_returns_none_for_unknown_query(self, tmp_path):
        retriever = self._build_retriever(tmp_path, [SAMPLE_RESULT])
        query = _make_query_pair("CVE-UNKNOWN", "original")
        example, info = retriever.retrieve(query)
        assert example is None

    def test_matches_by_cve_and_variant(self, tmp_path):
        """Two records with same CVE but different variants must be distinguishable."""
        rec1 = {**SAMPLE_RESULT, "query_variant": "original", "example_dir": "DIR_A"}
        rec2 = {**SAMPLE_RESULT, "query_variant": "augmented", "example_dir": "DIR_B"}
        retriever = self._build_retriever(tmp_path, [rec1, rec2])

        ex_orig, _ = retriever.retrieve(_make_query_pair("CVE-2025-0001", "original"))
        ex_aug, _ = retriever.retrieve(_make_query_pair("CVE-2025-0001", "augmented"))
        assert ex_orig.meta["dir_name"] == "DIR_A"
        assert ex_aug.meta["dir_name"] == "DIR_B"

    def test_dir_name_propagated_in_meta(self, tmp_path):
        """The example's dir_name must come from the 'example_dir' field in results."""
        retriever = self._build_retriever(tmp_path, [SAMPLE_RESULT])
        query = _make_query_pair("CVE-2025-0001", "original")
        example, _ = retriever.retrieve(query)
        assert example.meta["dir_name"] == "CVE-2025-0099_1"

    def test_dir_name_empty_for_old_results_without_field(self, tmp_path):
        """Old results.jsonl files won't have 'example_dir'. dir_name should be empty string."""
        old_result = {k: v for k, v in SAMPLE_RESULT.items() if k != "example_dir"}
        retriever = self._build_retriever(tmp_path, [old_result])
        query = _make_query_pair("CVE-2025-0001", "original")
        example, _ = retriever.retrieve(query)
        assert example.meta["dir_name"] == ""

    def test_variant_propagated_in_meta(self, tmp_path):
        retriever = self._build_retriever(tmp_path, [SAMPLE_RESULT])
        query = _make_query_pair("CVE-2025-0001", "original")
        example, _ = retriever.retrieve(query)
        assert example.meta["variant"] == "augmented"

    def test_retrieval_info_returned(self, tmp_path):
        retriever = self._build_retriever(tmp_path, [SAMPLE_RESULT])
        query = _make_query_pair("CVE-2025-0001", "original")
        _, info = retriever.retrieve(query)
        assert info["cwe_match"] is True
        assert "distance" in info

    def test_skips_non_retrieved_status(self, tmp_path):
        """Records with status != 'retrieved'/'success' should not return an example."""
        failed = {**SAMPLE_RESULT, "status": "error"}
        retriever = self._build_retriever(tmp_path, [failed])
        query = _make_query_pair("CVE-2025-0001", "original")
        example, _ = retriever.retrieve(query)
        assert example is None

    def test_accepts_success_status(self, tmp_path):
        rec = {**SAMPLE_RESULT, "status": "success"}
        retriever = self._build_retriever(tmp_path, [rec])
        query = _make_query_pair("CVE-2025-0001", "original")
        example, _ = retriever.retrieve(query)
        assert example is not None

    def test_cwe_id_propagated(self, tmp_path):
        retriever = self._build_retriever(tmp_path, [SAMPLE_RESULT])
        query = _make_query_pair("CVE-2025-0001", "original")
        example, _ = retriever.retrieve(query)
        assert example.cwe_id == "CWE-119"


# ── db_cache integration scenario ────────────────────────────────────


class TestPrecomputedRetrieverDbCacheIntegration:
    """Verify that dir_name from PrecomputedRetriever can be used to look up db_cache."""

    def test_dir_name_matches_db_cache_key(self, tmp_path):
        """Simulate the batch_inference flow: retriever returns dir_name,
        which is used to look up db_cache."""
        retriever = PrecomputedRetriever.__new__(PrecomputedRetriever)
        retriever._lookup = {
            ("CVE-2025-0001", "original"): {
                "query_cve": "CVE-2025-0001",
                "status": "retrieved",
                "example_cve": "CVE-2025-0099",
                "example_cwe": "CWE-119",
                "example_variant": "original",
                "example_dir": "CVE-2025-0099_1",
                "retrieval": {},
            }
        }

        query = _make_query_pair("CVE-2025-0001", "original")
        example, _ = retriever.retrieve(query)

        # Simulate db_cache lookup
        db_cache = {
            "CVE-2025-0099_1": {"function_name": "target_func"},
            "CVE-2025-0099_2": {"function_name": "other_func"},
        }
        looked_up = db_cache.get(example.meta["dir_name"])
        assert looked_up is not None
        assert looked_up["function_name"] == "target_func"

    def test_empty_dir_name_causes_cache_miss(self, tmp_path):
        """Old results without example_dir → empty dir_name → db_cache miss."""
        path = tmp_path / "results.jsonl"
        old_result = {k: v for k, v in SAMPLE_RESULT.items() if k != "example_dir"}
        _write_results(path, [old_result])
        retriever = PrecomputedRetriever(path)

        query = _make_query_pair("CVE-2025-0001", "original")
        example, _ = retriever.retrieve(query)

        db_cache = {"CVE-2025-0099_1": {"function_name": "target_func"}}
        looked_up = db_cache.get(example.meta["dir_name"])
        assert looked_up is None  # cache miss because dir_name is ""
