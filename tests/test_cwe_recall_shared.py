"""
Correctness verification for the shared _cwe_recall_summary helper.

Tests both self-retrieval (self_idx != None) and cross-split (self_idx = None)
modes, checking support calculation, singleton handling, ranx qrels exclusion,
and equivalence with hand-computed values.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pytest

from metrics.retrieval_eval import _cwe_recall_summary


# ── helpers ──────────────────────────────────────────────────────────

def _r(cwe_id: str, score: float, idx: int) -> dict:
    """Build a fake retrieval result dict."""
    return {"cwe_id": cwe_id, "score": score, "_idx": idx}


# ── cross-split mode (self_idx=None) ────────────────────────────────

class TestCrossSplit:
    """self_idx=None → query is NOT in the index."""

    def test_perfect_recall(self):
        """2 index docs with CWE-A, query retrieves both → recall 1.0."""
        metadata = [
            {"cwe_id": "CWE-A"},  # d0
            {"cwe_id": "CWE-A"},  # d1
            {"cwe_id": "CWE-B"},  # d2
        ]
        results = [_r("CWE-A", 0.9, 0), _r("CWE-A", 0.8, 1)]
        per_query = [("CWE-A", "q0", results, None)]
        out = _cwe_recall_summary(per_query, metadata, top_k=5)

        assert out["macro_avg"] == pytest.approx(1.0)
        assert out["per_cwe"]["CWE-A"]["support"] == 2  # full index count
        assert out["n_singletons"] == 0

    def test_no_support(self):
        """Query CWE not in index → skipped, empty result."""
        metadata = [{"cwe_id": "CWE-B"}]
        results = [_r("CWE-B", 0.5, 0)]
        per_query = [("CWE-X", "q0", results, None)]
        out = _cwe_recall_summary(per_query, metadata, top_k=5)

        assert out["n_cwes"] == 0
        assert out["macro_avg"] == 0.0

    def test_partial_recall(self):
        """3 CWE-A docs in index, only 1 retrieved → recall 1/3."""
        metadata = [
            {"cwe_id": "CWE-A"},
            {"cwe_id": "CWE-A"},
            {"cwe_id": "CWE-A"},
        ]
        results = [_r("CWE-A", 0.9, 0), _r("CWE-B", 0.5, 99)]
        per_query = [("CWE-A", "q0", results, None)]
        out = _cwe_recall_summary(per_query, metadata, top_k=5)

        # support=3, possible=min(5,3)=3, same=1 → 1/3
        assert out["per_cwe"]["CWE-A"]["recall"] == pytest.approx(1 / 3)

    def test_support_reports_full_count(self):
        """Support should be the total index count for that CWE."""
        metadata = [{"cwe_id": "CWE-A"}] * 5 + [{"cwe_id": "CWE-B"}] * 3
        results = [_r("CWE-A", 0.9, 0)]
        per_query = [("CWE-A", "q0", results, None)]
        out = _cwe_recall_summary(per_query, metadata, top_k=10)

        assert out["per_cwe"]["CWE-A"]["support"] == 5


# ── self-retrieval mode (self_idx != None) ───────────────────────────

class TestSelfRetrieval:
    """self_idx=i → query IS in the index, must subtract 1 from support."""

    def test_support_decremented(self):
        """With self_idx, support should be total-1 in denominator."""
        metadata = [
            {"cwe_id": "CWE-A"},  # d0 — self
            {"cwe_id": "CWE-A"},  # d1
            {"cwe_id": "CWE-B"},  # d2
        ]
        # Self already removed from results by caller
        results = [_r("CWE-A", 0.8, 1)]
        per_query = [("CWE-A", "q0", results, 0)]  # self_idx=0
        out = _cwe_recall_summary(per_query, metadata, top_k=5)

        # support in index = 2 for CWE-A. self_idx → possible = min(5, 2-1) = 1.
        # Retrieved 1 CWE-A → recall = 1/1 = 1.0
        assert out["macro_avg"] == pytest.approx(1.0)

    def test_singleton_detected(self):
        """CWE with only 1 entry → support-1=0 → singleton."""
        metadata = [
            {"cwe_id": "CWE-SOLO"},  # d0
            {"cwe_id": "CWE-A"},     # d1
            {"cwe_id": "CWE-A"},     # d2
        ]
        results_solo = [_r("CWE-A", 0.5, 1)]
        results_a = [_r("CWE-A", 0.9, 2)]
        per_query = [
            ("CWE-SOLO", "q0", results_solo, 0),  # singleton
            ("CWE-A", "q1", results_a, 1),
        ]
        out = _cwe_recall_summary(per_query, metadata, top_k=5)

        assert out["n_singletons"] == 1  # CWE-SOLO
        assert "CWE-SOLO" not in out["per_cwe"]
        assert "CWE-A" in out["per_cwe"]

    def test_self_excluded_from_qrels(self):
        """Self position should NOT appear in ranx qrels."""
        metadata = [
            {"cwe_id": "CWE-A"},  # d0 — self
            {"cwe_id": "CWE-A"},  # d1
            {"cwe_id": "CWE-A"},  # d2
        ]
        results = [_r("CWE-A", 0.8, 1), _r("CWE-A", 0.7, 2)]
        per_query = [("CWE-A", "q0", results, 0)]
        out = _cwe_recall_summary(per_query, metadata, top_k=5)

        # support=3, self→ possible = min(5, 3-1) = 2
        # Retrieved 2 CWE-A → recall = 2/2 = 1.0
        assert out["macro_avg"] == pytest.approx(1.0)
        # ranx_recall should also be 1.0 since qrels has d1,d2 and run has d1,d2
        assert out["ranx_recall"] == pytest.approx(1.0)

    def test_perfect_clustering_3_entries(self):
        """3 CWE-A entries, each retrieves the other 2 → recall 1.0."""
        metadata = [
            {"cwe_id": "CWE-A"},
            {"cwe_id": "CWE-A"},
            {"cwe_id": "CWE-A"},
        ]
        per_query = [
            ("CWE-A", "q0", [_r("CWE-A", 0.9, 1), _r("CWE-A", 0.8, 2)], 0),
            ("CWE-A", "q1", [_r("CWE-A", 0.9, 0), _r("CWE-A", 0.8, 2)], 1),
            ("CWE-A", "q2", [_r("CWE-A", 0.9, 0), _r("CWE-A", 0.8, 1)], 2),
        ]
        out = _cwe_recall_summary(per_query, metadata, top_k=5)

        assert out["macro_avg"] == pytest.approx(1.0)
        assert out["ranx_recall"] == pytest.approx(1.0)
        assert out["n_singletons"] == 0

    def test_mixed_cwe_partial(self):
        """2 CWEs, partial retrieval → correct per-CWE and macro."""
        metadata = [
            {"cwe_id": "CWE-A"},  # 0
            {"cwe_id": "CWE-A"},  # 1
            {"cwe_id": "CWE-A"},  # 2
            {"cwe_id": "CWE-B"},  # 3
            {"cwe_id": "CWE-B"},  # 4
        ]
        # CWE-A query from idx=0: support=3, possible=min(3, 3-1)=2
        # retrieves 1 CWE-A → 1/2 = 0.5
        per_query = [
            ("CWE-A", "q0", [_r("CWE-A", 0.9, 1), _r("CWE-B", 0.5, 3)], 0),
            # CWE-B query from idx=3: support=2, possible=min(3, 2-1)=1
            # retrieves 1 CWE-B → 1/1 = 1.0
            ("CWE-B", "q1", [_r("CWE-B", 0.8, 4), _r("CWE-A", 0.3, 0)], 3),
        ]
        out = _cwe_recall_summary(per_query, metadata, top_k=3)

        assert out["per_cwe"]["CWE-A"]["recall"] == pytest.approx(0.5)
        assert out["per_cwe"]["CWE-B"]["recall"] == pytest.approx(1.0)
        assert out["macro_avg"] == pytest.approx(0.75)  # (0.5+1.0)/2


# ── cwe_group_recall integration ────────────────────────────────────

class TestCweGroupRecallIntegration:
    """End-to-end test of the refactored cwe_group_recall function."""

    def _make_retriever(self, embs, meta):
        """Brute-force inner-product retriever with _idx."""
        class R:
            def __init__(self, e, m):
                self._e = e.astype(np.float32)
                self._m = m
            def query(self, vec, top_k=10):
                scores = self._e @ vec.ravel()
                order = np.argsort(-scores)[:top_k]
                return [
                    {**self._m[i], "score": float(scores[i]), "_idx": int(i)}
                    for i in order
                ]
        return R(embs, meta)

    def test_perfect_clustering(self):
        """Two CWEs, well-separated → macro 1.0."""
        from metrics.metrics import cwe_group_recall

        embs = np.array([
            [1.0, 0.0], [0.95, 0.05],  # CWE-A
            [0.0, 1.0], [0.05, 0.95],  # CWE-B
        ], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms

        meta = [
            {"cve_id": "CVE-1", "cwe_id": "CWE-A"},
            {"cve_id": "CVE-2", "cwe_id": "CWE-A"},
            {"cve_id": "CVE-3", "cwe_id": "CWE-B"},
            {"cve_id": "CVE-4", "cwe_id": "CWE-B"},
        ]
        retr = self._make_retriever(embs, meta)
        result = cwe_group_recall(embs, meta, retr, top_k=3)

        assert result["macro_avg"] == pytest.approx(1.0)
        assert "ranx_recall" in result  # new: ranx now available
        assert result["n_singletons"] == 0

    def test_singleton_skipped(self):
        """CWE with 1 entry → singleton, not in per_cwe."""
        from metrics.metrics import cwe_group_recall

        embs = np.eye(3, dtype=np.float32)
        meta = [
            {"cve_id": "CVE-1", "cwe_id": "CWE-SOLO"},
            {"cve_id": "CVE-2", "cwe_id": "CWE-DUO"},
            {"cve_id": "CVE-3", "cwe_id": "CWE-DUO"},
        ]
        retr = self._make_retriever(embs, meta)
        result = cwe_group_recall(embs, meta, retr, top_k=2)

        assert result["n_singletons"] == 1
        assert "CWE-SOLO" not in result["per_cwe"]

    def test_augmented_variants_not_over_excluded(self):
        """Same CVE ID but different augmented entries → only self removed."""
        from metrics.metrics import cwe_group_recall

        embs = np.array([
            [1.0, 0.1],  # CVE-1 variant A
            [1.0, 0.2],  # CVE-1 variant B (same CVE!)
            [0.9, 0.3],  # CVE-2
        ], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms

        meta = [
            {"cve_id": "CVE-1", "cwe_id": "CWE-A"},
            {"cve_id": "CVE-1", "cwe_id": "CWE-A"},  # same CVE!
            {"cve_id": "CVE-2", "cwe_id": "CWE-A"},
        ]
        retr = self._make_retriever(embs, meta)
        result = cwe_group_recall(embs, meta, retr, top_k=3)

        # Each query should see the other 2 CWE-A results → recall 1.0
        assert result["macro_avg"] == pytest.approx(1.0)
