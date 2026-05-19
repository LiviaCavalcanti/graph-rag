"""
Exhaustive unit tests for the individual IR metric functions in
src/metrics/retrieval_eval.py.

Each helper is tested in isolation with known inputs and expected outputs.
Edge cases (empty inputs, zero relevance, single documents) are covered.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pytest

from src.metrics.metrics import (
    _reciprocal_rank,
    _hit_rate_at_k,
    _precision_at_k,
    _recall_at_k,
    _ndcg_at_k,
    _average_precision_at_k,
    _compute_metrics,
    _cwe_recall_summary,
)
from src.metrics.retrieval_eval import (
    _doc_id,
    _build_cve_qrels_and_run,
)


# ===================================================================
# _reciprocal_rank
# ===================================================================


class TestReciprocalRank:
    """Tests for _reciprocal_rank(ranked_docs, qrels, k)."""

    def test_relevant_at_rank_1(self):
        ranked = [("d0", 1.0), ("d1", 0.5)]
        qrels = {"d0": 1, "d1": 0}
        assert _reciprocal_rank(ranked, qrels, k=10) == pytest.approx(1.0)

    def test_relevant_at_rank_2(self):
        ranked = [("d0", 1.0), ("d1", 0.5), ("d2", 0.3)]
        qrels = {"d1": 1}
        assert _reciprocal_rank(ranked, qrels, k=10) == pytest.approx(0.5)

    def test_relevant_at_rank_3(self):
        ranked = [("d0", 1.0), ("d1", 0.8), ("d2", 0.5)]
        qrels = {"d2": 1}
        assert _reciprocal_rank(ranked, qrels, k=10) == pytest.approx(1.0 / 3)

    def test_relevant_at_rank_5(self):
        ranked = [(f"d{i}", 10 - i) for i in range(10)]
        qrels = {"d4": 1}
        assert _reciprocal_rank(ranked, qrels, k=10) == pytest.approx(1.0 / 5)

    def test_no_relevant_doc(self):
        ranked = [("d0", 1.0), ("d1", 0.5)]
        qrels = {"d99": 1}
        assert _reciprocal_rank(ranked, qrels, k=10) == 0.0

    def test_empty_ranked_list(self):
        assert _reciprocal_rank([], {"d0": 1}, k=10) == 0.0

    def test_empty_qrels(self):
        ranked = [("d0", 1.0)]
        assert _reciprocal_rank(ranked, {}, k=10) == 0.0

    def test_k_cutoff_excludes_relevant(self):
        """Relevant at rank 3, but k=2 → should return 0."""
        ranked = [("d0", 1.0), ("d1", 0.8), ("d2", 0.5)]
        qrels = {"d2": 1}
        assert _reciprocal_rank(ranked, qrels, k=2) == 0.0

    def test_k_cutoff_includes_relevant(self):
        """Relevant at rank 2, k=2 → should return 0.5."""
        ranked = [("d0", 1.0), ("d1", 0.8), ("d2", 0.5)]
        qrels = {"d1": 1}
        assert _reciprocal_rank(ranked, qrels, k=2) == pytest.approx(0.5)

    def test_multiple_relevant_returns_first(self):
        """Multiple relevant docs: RR is 1/rank of the FIRST one."""
        ranked = [("d0", 1.0), ("d1", 0.8), ("d2", 0.5)]
        qrels = {"d1": 1, "d2": 1}
        assert _reciprocal_rank(ranked, qrels, k=10) == pytest.approx(0.5)

    def test_k_zero(self):
        ranked = [("d0", 1.0)]
        qrels = {"d0": 1}
        assert _reciprocal_rank(ranked, qrels, k=0) == 0.0


# ===================================================================
# _hit_rate_at_k
# ===================================================================


class TestHitRateAtK:
    """Tests for _hit_rate_at_k(relevance, k)."""

    def test_hit_at_position_1(self):
        assert _hit_rate_at_k([1, 0, 0], k=1) == 1.0

    def test_hit_at_position_k(self):
        """Relevant doc at exactly position k (last slot)."""
        assert _hit_rate_at_k([0, 0, 1], k=3) == 1.0

    def test_no_hit(self):
        assert _hit_rate_at_k([0, 0, 0], k=3) == 0.0

    def test_hit_beyond_k(self):
        """Relevant at position 4, but k=3."""
        assert _hit_rate_at_k([0, 0, 0, 1], k=3) == 0.0

    def test_all_relevant(self):
        assert _hit_rate_at_k([1, 1, 1], k=3) == 1.0

    def test_empty_relevance(self):
        assert _hit_rate_at_k([], k=5) == 0.0

    def test_k_larger_than_list(self):
        """k > len(relevance): should still find the hit."""
        assert _hit_rate_at_k([0, 1], k=10) == 1.0

    def test_k_zero(self):
        assert _hit_rate_at_k([1, 0], k=0) == 0.0

    def test_returns_float(self):
        result = _hit_rate_at_k([1], k=1)
        assert isinstance(result, float)


# ===================================================================
# _precision_at_k
# ===================================================================


class TestPrecisionAtK:
    """Tests for _precision_at_k(relevance, k)."""

    def test_all_relevant(self):
        assert _precision_at_k([1, 1, 1], k=3) == pytest.approx(1.0)

    def test_none_relevant(self):
        assert _precision_at_k([0, 0, 0], k=3) == pytest.approx(0.0)

    def test_half_relevant(self):
        assert _precision_at_k([1, 0, 1, 0], k=4) == pytest.approx(0.5)

    def test_one_of_three(self):
        assert _precision_at_k([1, 0, 0], k=3) == pytest.approx(1.0 / 3)

    def test_k_less_than_list(self):
        """Only first k elements count."""
        assert _precision_at_k([1, 1, 0, 0, 0], k=2) == pytest.approx(1.0)

    def test_k_one_relevant(self):
        assert _precision_at_k([1], k=1) == pytest.approx(1.0)

    def test_k_one_not_relevant(self):
        assert _precision_at_k([0], k=1) == pytest.approx(0.0)

    def test_k_larger_than_list(self):
        """k=5 but only 3 docs: precision = 2/5 (treats missing as non-relevant)."""
        assert _precision_at_k([1, 0, 1], k=5) == pytest.approx(2.0 / 5)


# ===================================================================
# _recall_at_k
# ===================================================================


class TestRecallAtK:
    """Tests for _recall_at_k(relevance, k, total_rel)."""

    def test_all_found(self):
        """All 2 relevant docs in top-3."""
        assert _recall_at_k([1, 0, 1], k=3, total_rel=2) == pytest.approx(1.0)

    def test_partial_found(self):
        """1 of 3 relevant docs found in top-2."""
        assert _recall_at_k([1, 0, 0], k=2, total_rel=3) == pytest.approx(1.0 / 3)

    def test_none_found(self):
        assert _recall_at_k([0, 0, 0], k=3, total_rel=5) == pytest.approx(0.0)

    def test_total_rel_zero(self):
        """No relevant docs exist → recall = 0.0 (avoid division by zero)."""
        assert _recall_at_k([0, 0], k=2, total_rel=0) == 0.0

    def test_k_cuts_off_relevant(self):
        """Relevant doc at position 3, but k=2."""
        assert _recall_at_k([0, 0, 1], k=2, total_rel=1) == pytest.approx(0.0)

    def test_total_rel_exceeds_k(self):
        """More relevant docs exist than k allows finding."""
        assert _recall_at_k([1, 1], k=2, total_rel=10) == pytest.approx(2.0 / 10)

    def test_empty_relevance(self):
        assert _recall_at_k([], k=5, total_rel=3) == pytest.approx(0.0)

    def test_k_zero(self):
        assert _recall_at_k([1, 1], k=0, total_rel=2) == pytest.approx(0.0)


# ===================================================================
# _ndcg_at_k
# ===================================================================


class TestNdcgAtK:
    """Tests for _ndcg_at_k(y_true, y_score, k)."""

    def test_perfect_ranking(self):
        """Scores perfectly correlate with relevance → NDCG = 1.0."""
        y_true = np.array([1.0, 1.0, 0.0, 0.0])
        y_score = np.array([4.0, 3.0, 2.0, 1.0])
        assert _ndcg_at_k(y_true, y_score, k=2) == pytest.approx(1.0)

    def test_reverse_ranking(self):
        """Scores inversely correlate with relevance → NDCG < 1."""
        y_true = np.array([1.0, 0.0, 0.0, 0.0])
        y_score = np.array([1.0, 2.0, 3.0, 4.0])  # relevant doc ranked last
        ndcg = _ndcg_at_k(y_true, y_score, k=4)
        assert ndcg < 1.0
        assert ndcg > 0.0  # still non-zero since the doc IS in the list

    def test_no_relevant_docs(self):
        """No relevant documents → NDCG = 0.0."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_score = np.array([3.0, 2.0, 1.0])
        assert _ndcg_at_k(y_true, y_score, k=3) == 0.0

    def test_single_document(self):
        """Single document → degenerate, should return 0.0."""
        y_true = np.array([1.0])
        y_score = np.array([1.0])
        assert _ndcg_at_k(y_true, y_score, k=1) == 0.0

    def test_k_less_than_total(self):
        """NDCG at k=1 with relevant doc at rank 1."""
        y_true = np.array([1.0, 0.0, 0.0])
        y_score = np.array([3.0, 2.0, 1.0])
        assert _ndcg_at_k(y_true, y_score, k=1) == pytest.approx(1.0)

    def test_graded_relevance(self):
        """Graded relevance: doc with rel=2 ranked above rel=1 → NDCG=1."""
        y_true = np.array([2.0, 1.0, 0.0])
        y_score = np.array([3.0, 2.0, 1.0])
        assert _ndcg_at_k(y_true, y_score, k=3) == pytest.approx(1.0)

    def test_returns_float(self):
        y_true = np.array([1.0, 0.0])
        y_score = np.array([2.0, 1.0])
        result = _ndcg_at_k(y_true, y_score, k=2)
        assert isinstance(result, float)


# ===================================================================
# _average_precision_at_k
# ===================================================================


class TestAveragePrecisionAtK:
    """Tests for _average_precision_at_k(relevance, k, total_rel)."""

    def test_perfect_ranking(self):
        """All relevant docs at top → AP = 1.0."""
        # 2 relevant out of 4, both at top
        assert _average_precision_at_k([1, 1, 0, 0], k=4, total_rel=2) == pytest.approx(1.0)

    def test_single_relevant_at_rank_1(self):
        """One relevant doc at rank 1 → AP = 1.0."""
        assert _average_precision_at_k([1, 0, 0], k=3, total_rel=1) == pytest.approx(1.0)

    def test_single_relevant_at_rank_2(self):
        """One relevant doc at rank 2.
        AP = (1/2) / min(1, 3) = 0.5
        """
        assert _average_precision_at_k([0, 1, 0], k=3, total_rel=1) == pytest.approx(0.5)

    def test_single_relevant_at_rank_3(self):
        """One relevant doc at rank 3.
        AP = (1/3) / min(1, 3) = 1/3
        """
        assert _average_precision_at_k([0, 0, 1], k=3, total_rel=1) == pytest.approx(1.0 / 3)

    def test_two_relevant_interleaved(self):
        """Relevant at ranks 1 and 3.
        P(1)=1/1=1, P(3)=2/3
        AP = (1 + 2/3) / min(2, 4) = (5/3) / 2 = 5/6
        """
        ap = _average_precision_at_k([1, 0, 1, 0], k=4, total_rel=2)
        assert ap == pytest.approx(5.0 / 6)

    def test_two_relevant_at_bottom(self):
        """Relevant at ranks 3 and 4.
        P(3)=1/3, P(4)=2/4=1/2
        AP = (1/3 + 1/2) / min(2, 4) = (5/6) / 2 = 5/12
        """
        ap = _average_precision_at_k([0, 0, 1, 1], k=4, total_rel=2)
        assert ap == pytest.approx(5.0 / 12)

    def test_no_relevant_found(self):
        """No relevant docs in top-k → AP = 0."""
        assert _average_precision_at_k([0, 0, 0], k=3, total_rel=2) == pytest.approx(0.0)

    def test_total_rel_zero(self):
        """No relevant docs exist → AP = 0.0."""
        assert _average_precision_at_k([0, 0], k=2, total_rel=0) == 0.0

    def test_k_truncation(self):
        """Relevant doc at rank 4, k=3 → not counted.
        AP = 0 / min(1, 3) = 0.0
        """
        assert _average_precision_at_k([0, 0, 0, 1], k=3, total_rel=1) == pytest.approx(0.0)

    def test_total_rel_greater_than_k(self):
        """total_rel=5 but k=2 and both top-2 are relevant.
        AP = (1 + 1) / min(5, 2) = 2/2 = 1.0
        """
        assert _average_precision_at_k([1, 1], k=2, total_rel=5) == pytest.approx(1.0)

    def test_total_rel_less_than_k(self):
        """total_rel=1, k=5, relevant at rank 1.
        AP = 1 / min(1, 5) = 1.0
        """
        assert _average_precision_at_k([1, 0, 0, 0, 0], k=5, total_rel=1) == pytest.approx(1.0)

    def test_empty_relevance(self):
        assert _average_precision_at_k([], k=3, total_rel=2) == pytest.approx(0.0)


# ===================================================================
# _compute_metrics (integration of all helpers)
# ===================================================================


class TestComputeMetrics:
    """Integration tests for _compute_metrics combining all helpers."""

    def _perfect_run(self):
        """2 queries, each with its relevant doc at rank 1."""
        qrels = {
            "q0": {"d0": 1, "d1": 0, "d2": 0},
            "q1": {"d3": 1, "d4": 0, "d5": 0},
        }
        run = {
            "q0": {"d0": 3.0, "d1": 2.0, "d2": 1.0},
            "q1": {"d3": 3.0, "d4": 2.0, "d5": 1.0},
        }
        return qrels, run

    def test_perfect_hit_rate(self):
        qrels, run = self._perfect_run()
        results = _compute_metrics(qrels, run, ks=[1, 3])
        assert results["hit_rate@1"] == pytest.approx(1.0)
        assert results["hit_rate@3"] == pytest.approx(1.0)

    def test_perfect_precision(self):
        qrels, run = self._perfect_run()
        results = _compute_metrics(qrels, run, ks=[1, 3])
        assert results["precision@1"] == pytest.approx(1.0)
        assert results["precision@3"] == pytest.approx(1.0 / 3)

    def test_perfect_recall(self):
        qrels, run = self._perfect_run()
        results = _compute_metrics(qrels, run, ks=[1, 3])
        assert results["recall@1"] == pytest.approx(1.0)
        assert results["recall@3"] == pytest.approx(1.0)

    def test_perfect_mrr(self):
        qrels, run = self._perfect_run()
        results = _compute_metrics(qrels, run, ks=[1, 3])
        assert results["mrr@3"] == pytest.approx(1.0)

    def test_perfect_ndcg(self):
        qrels, run = self._perfect_run()
        results = _compute_metrics(qrels, run, ks=[1, 3])
        assert results["ndcg@1"] == pytest.approx(1.0)
        assert results["ndcg@3"] == pytest.approx(1.0)

    def test_perfect_map(self):
        qrels, run = self._perfect_run()
        results = _compute_metrics(qrels, run, ks=[1, 3])
        assert results["map@1"] == pytest.approx(1.0)
        assert results["map@3"] == pytest.approx(1.0)

    def test_worst_case(self):
        """Relevant doc ranked last of 3."""
        qrels = {"q0": {"d0": 1, "d1": 0, "d2": 0}}
        run = {"q0": {"d0": 1.0, "d1": 2.0, "d2": 3.0}}  # d0 ranked last
        results = _compute_metrics(qrels, run, ks=[1, 3])
        assert results["hit_rate@1"] == pytest.approx(0.0)
        assert results["hit_rate@3"] == pytest.approx(1.0)
        assert results["precision@1"] == pytest.approx(0.0)
        assert results["precision@3"] == pytest.approx(1.0 / 3)
        assert results["recall@1"] == pytest.approx(0.0)
        assert results["recall@3"] == pytest.approx(1.0)
        assert results["mrr@3"] == pytest.approx(1.0 / 3)
        assert results["map@3"] == pytest.approx(1.0 / 3)

    def test_mixed_queries(self):
        """Query 0: relevant at rank 1. Query 1: relevant at rank 2."""
        qrels = {
            "q0": {"d0": 1, "d1": 0},
            "q1": {"d2": 0, "d3": 1},
        }
        run = {
            "q0": {"d0": 2.0, "d1": 1.0},
            "q1": {"d2": 2.0, "d3": 1.0},  # d3 (relevant) ranked 2nd
        }
        results = _compute_metrics(qrels, run, ks=[1, 2])
        # hit_rate@1: q0=1, q1=0 → mean=0.5
        assert results["hit_rate@1"] == pytest.approx(0.5)
        # hit_rate@2: both find it → 1.0
        assert results["hit_rate@2"] == pytest.approx(1.0)
        # mrr@2: q0=1.0, q1=0.5 → mean=0.75
        assert results["mrr@2"] == pytest.approx(0.75)
        # precision@1: q0=1.0, q1=0.0 → mean=0.5
        assert results["precision@1"] == pytest.approx(0.5)

    def test_no_relevant_docs_for_query(self):
        """Query with no relevant docs should get zeros for all metrics."""
        qrels = {"q0": {"__none__": 0}}
        run = {"q0": {"d0": 2.0, "d1": 1.0}}
        results = _compute_metrics(qrels, run, ks=[1, 2])
        assert results["hit_rate@1"] == 0.0
        assert results["precision@1"] == 0.0
        assert results["recall@1"] == 0.0
        assert results["ndcg@1"] == 0.0
        assert results["map@1"] == 0.0
        assert results["mrr@2"] == 0.0

    def test_empty_run(self):
        """No queries → all metrics should be 0.0."""
        results = _compute_metrics({}, {}, ks=[1, 5])
        assert results["hit_rate@1"] == 0.0
        assert results["mrr@5"] == 0.0

    def test_multiple_relevant_docs(self):
        """3 relevant docs out of 5, ranked at positions 1, 3, 5."""
        qrels = {"q0": {"d0": 1, "d1": 0, "d2": 1, "d3": 0, "d4": 1}}
        run = {"q0": {"d0": 5.0, "d1": 4.0, "d2": 3.0, "d3": 2.0, "d4": 1.0}}
        results = _compute_metrics(qrels, run, ks=[1, 3, 5])
        # recall@1: 1/3
        assert results["recall@1"] == pytest.approx(1.0 / 3)
        # recall@3: 2/3
        assert results["recall@3"] == pytest.approx(2.0 / 3)
        # recall@5: 3/3
        assert results["recall@5"] == pytest.approx(1.0)
        # precision@5: 3/5
        assert results["precision@5"] == pytest.approx(3.0 / 5)

    def test_returns_all_expected_keys(self):
        qrels, run = self._perfect_run()
        results = _compute_metrics(qrels, run, ks=[1, 5, 10])
        for k in [1, 5, 10]:
            assert f"hit_rate@{k}" in results
            assert f"precision@{k}" in results
            assert f"recall@{k}" in results
            assert f"ndcg@{k}" in results
            assert f"map@{k}" in results
        assert "mrr@10" in results


# ===================================================================
# _doc_id
# ===================================================================


class TestDocId:
    """Tests for _doc_id(result, rank, qid)."""

    def test_with_idx(self):
        result = {"_idx": 42, "cve_id": "CVE-1"}
        assert _doc_id(result, rank=0, qid="q0") == "d42"

    def test_with_idx_zero(self):
        """idx=0 is valid and should NOT fall through to fallback."""
        result = {"_idx": 0, "cve_id": "CVE-1"}
        assert _doc_id(result, rank=5, qid="q3") == "d0"

    def test_without_idx(self):
        """No _idx → fallback to rank-based id."""
        result = {"cve_id": "CVE-1"}
        assert _doc_id(result, rank=2, qid="q5") == "d_rank2_q5"

    def test_idx_none_explicit(self):
        """Explicitly set _idx=None → fallback."""
        result = {"_idx": None, "cve_id": "CVE-1"}
        assert _doc_id(result, rank=1, qid="q0") == "d_rank1_q0"


# ===================================================================
# _build_cve_qrels_and_run
# ===================================================================


class TestBuildCveQrelsAndRun:
    """Tests for _build_cve_qrels_and_run."""

    def test_with_index_metadata(self):
        """All index entries sharing query CVE are marked relevant."""
        index_metadata = [
            {"cve_id": "CVE-A"},
            {"cve_id": "CVE-A"},
            {"cve_id": "CVE-B"},
        ]
        results = [{"_idx": 2, "cve_id": "CVE-B", "score": 0.9}]
        query_results = [("q0", "CVE-A", results)]

        qrels, run = _build_cve_qrels_and_run(query_results, index_metadata)
        # Both d0 and d1 should be relevant (same CVE as query)
        assert qrels["q0"]["d0"] == 1
        assert qrels["q0"]["d1"] == 1
        assert "d2" not in qrels["q0"] or qrels["q0"].get("d2", 0) == 0

    def test_without_index_metadata(self):
        """Without metadata, judge relevance from retrieved results only."""
        results = [
            {"_idx": 0, "cve_id": "CVE-A", "score": 0.9},
            {"_idx": 1, "cve_id": "CVE-B", "score": 0.5},
        ]
        query_results = [("q0", "CVE-A", results)]

        qrels, run = _build_cve_qrels_and_run(query_results, None)
        assert qrels["q0"]["d0"] == 1
        assert "d1" not in qrels["q0"]

    def test_no_relevant_gets_placeholder(self):
        """Query with no matching CVE in results or metadata gets __none__."""
        index_metadata = [{"cve_id": "CVE-X"}]
        results = [{"_idx": 0, "cve_id": "CVE-X", "score": 0.5}]
        query_results = [("q0", "CVE-MISSING", results)]

        qrels, run = _build_cve_qrels_and_run(query_results, index_metadata)
        assert qrels["q0"] == {"__none__": 0}

    def test_empty_results_skipped(self):
        """Query with empty results → not added to run/qrels."""
        query_results = [("q0", "CVE-A", [])]
        qrels, run = _build_cve_qrels_and_run(query_results, None)
        assert "q0" not in run
        assert "q0" not in qrels

    def test_run_scores_preserved(self):
        """Scores in run dict match input scores."""
        results = [
            {"_idx": 0, "cve_id": "CVE-A", "score": 0.95},
            {"_idx": 1, "cve_id": "CVE-B", "score": 0.3},
        ]
        query_results = [("q0", "CVE-A", results)]
        _, run = _build_cve_qrels_and_run(query_results, None)
        assert run["q0"]["d0"] == pytest.approx(0.95)
        assert run["q0"]["d1"] == pytest.approx(0.3)


# ===================================================================
# _cwe_recall_summary
# ===================================================================


class TestCweRecallSummary:
    """Tests for _cwe_recall_summary — CWE-level recall computation."""

    def test_basic_recall(self):
        """2 queries for CWE-79, both retrieve 1 of 2 peers → recall 0.5."""
        index_metadata = [
            {"cwe_id": "CWE-79"},
            {"cwe_id": "CWE-79"},
            {"cwe_id": "CWE-89"},
        ]
        per_query = [
            ("CWE-79", "q0", [{"_idx": 0, "cwe_id": "CWE-79", "score": 0.9}], None),
            ("CWE-79", "q1", [{"_idx": 1, "cwe_id": "CWE-79", "score": 0.8}], None),
        ]
        result = _cwe_recall_summary(per_query, index_metadata, top_k=5)
        # macro_avg: min(5, 2)=2 → each gets 1/2 → mean=0.5
        assert result["macro_avg"] == pytest.approx(0.5)
        # ranx_recall: support=2, found=1 → 1/2=0.5
        assert result["ranx_recall"] == pytest.approx(0.5)

    def test_ranx_recall_works_without_idx(self):
        """Results without _idx should still produce non-zero ranx_recall.

        This was the bug: the old ID-based approach gave 0.0 here because
        doc IDs in qrels (d0, d1...) couldn't match run IDs (d_rank0_q0...).
        """
        index_metadata = [
            {"cwe_id": "CWE-79"},
            {"cwe_id": "CWE-79"},
            {"cwe_id": "CWE-89"},
        ]
        per_query = [
            ("CWE-79", "q0", [{"cwe_id": "CWE-79", "score": 0.9}], None),
        ]
        result = _cwe_recall_summary(per_query, index_metadata, top_k=5)
        # support=2, found=1 → ranx_recall = 0.5 (not 0.0!)
        assert result["ranx_recall"] == pytest.approx(0.5)
        assert result["macro_avg"] == pytest.approx(0.5)

    def test_self_idx_exclusion(self):
        """When self_idx is set, that position is excluded from support and matches."""
        index_metadata = [
            {"cwe_id": "CWE-79"},  # idx=0 — the query itself
            {"cwe_id": "CWE-79"},  # idx=1 — a peer
            {"cwe_id": "CWE-89"},
        ]
        per_query = [
            ("CWE-79", "q0", [
                {"_idx": 0, "cwe_id": "CWE-79", "score": 1.0},  # self — excluded
                {"_idx": 1, "cwe_id": "CWE-79", "score": 0.8},  # peer — counted
            ], 0),
        ]
        result = _cwe_recall_summary(per_query, index_metadata, top_k=5)
        # support=2-1=1, found=1 (only idx=1) → recall=1.0
        assert result["ranx_recall"] == pytest.approx(1.0)

    def test_singleton_cwe_skipped(self):
        """CWE with support <= 0 after self-exclusion → counted as singleton."""
        index_metadata = [
            {"cwe_id": "CWE-79"},
        ]
        per_query = [
            ("CWE-79", "q0", [{"_idx": 0, "cwe_id": "CWE-79", "score": 1.0}], 0),
        ]
        result = _cwe_recall_summary(per_query, index_metadata, top_k=5)
        assert result["n_singletons"] == 1
        assert result["macro_avg"] == 0.0
        assert result["ranx_recall"] == 0.0

    def test_top_k_cutoff(self):
        """Only first top_k results are considered for ranx_recall."""
        index_metadata = [
            {"cwe_id": "CWE-79"},
            {"cwe_id": "CWE-79"},
            {"cwe_id": "CWE-79"},
        ]
        per_query = [
            ("CWE-79", "q0", [
                {"_idx": 0, "cwe_id": "CWE-89", "score": 3.0},
                {"_idx": 1, "cwe_id": "CWE-89", "score": 2.0},
                {"_idx": 2, "cwe_id": "CWE-79", "score": 1.0},
            ], None),
        ]
        result = _cwe_recall_summary(per_query, index_metadata, top_k=2)
        # top_k=2 → only first 2 results considered → 0 matches
        assert result["ranx_recall"] == pytest.approx(0.0)

    def test_multiple_cwes(self):
        """Multiple CWE groups are independently tracked."""
        index_metadata = [
            {"cwe_id": "CWE-79"},
            {"cwe_id": "CWE-79"},
            {"cwe_id": "CWE-89"},
            {"cwe_id": "CWE-89"},
        ]
        per_query = [
            ("CWE-79", "q0", [{"_idx": 0, "cwe_id": "CWE-79", "score": 0.9}], None),
            ("CWE-89", "q1", [{"_idx": 2, "cwe_id": "CWE-89", "score": 0.8},
                              {"_idx": 3, "cwe_id": "CWE-89", "score": 0.7}], None),
        ]
        result = _cwe_recall_summary(per_query, index_metadata, top_k=5)
        assert result["n_cwes"] == 2
        # CWE-79: 1/2 = 0.5, CWE-89: 2/2 = 1.0 → macro = 0.75
        assert result["macro_avg"] == pytest.approx(0.75)
        # ranx_recall: q0: 1/2, q1: 2/2 → mean = 0.75
        assert result["ranx_recall"] == pytest.approx(0.75)

    def test_empty_per_query(self):
        """No queries → zeros everywhere."""
        index_metadata = [{"cwe_id": "CWE-79"}]
        result = _cwe_recall_summary([], index_metadata, top_k=5)
        assert result["macro_avg"] == 0.0
        assert result["ranx_recall"] == 0.0
        assert result["n_cwes"] == 0
