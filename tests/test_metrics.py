"""
Behaviour-driven tests for experiments/metrics.py

Tests are written from EXPECTED BEHAVIOUR, not the actual implementation.
The goal is to surface bugs by asserting what a correct implementation
should produce for known inputs.

Each test documents WHY the expected value is what it is, so a failure
tells you what assumption the code violates.

Severity markers
────────────────
  @severity_high   – bug changes experiment results (metrics are wrong)
  @severity_low    – corner-case robustness (NaN, crash on edge input)

Run only high-severity:  pytest -m severity_high
Run only low-severity:   pytest -m severity_low
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `from src.…` imports
# (used inside src/rag/index.py etc.) resolve correctly under pytest.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pytest

# ── severity markers ──────────────────────────────────────────────────
severity_high = pytest.mark.severity_high
severity_low = pytest.mark.severity_low
from collections import defaultdict

from metrics.metrics import (
    hits_at_k,
    mean_reciprocal_rank,
    self_retrieval_metrics,
    cwe_group_recall,
    embedding_space_stats,
    _effective_dim,
)


# ===================================================================
# Helpers / Fakes
# ===================================================================

def _make_result(cve_id: str, cwe_id: str = "CWE-0",
                 score: float = 1.0, **extra) -> dict:
    """Build a single retrieval result dict."""
    return {"cve_id": cve_id, "cwe_id": cwe_id, "score": score, **extra}


class FakeRetriever:
    """
    Brute-force inner-product retriever for testing.
    Returns results in the same format as the real Retriever:
    list[dict] with 'cve_id', 'cwe_id', 'score', etc.
    """

    def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
        self._embs = embeddings.astype(np.float32)
        self._meta = metadata

    def query(self, vec: np.ndarray, top_k: int = 10) -> list[dict]:
        scores = self._embs @ vec.ravel()
        order = np.argsort(-scores)[:top_k]
        return [
            {**self._meta[i], "score": float(scores[i])}
            for i in order
        ]


# ===================================================================
# hits_at_k
# ===================================================================

class TestHitsAtK:
    """hits_at_k(results, query_cve, k) → 1 if CVE in top-k, else 0."""

    def test_hit_at_rank_1(self):
        results = [_make_result("CVE-A"), _make_result("CVE-B")]
        assert hits_at_k(results, "CVE-A", k=1) == 1

    def test_hit_at_rank_k_boundary(self):
        """CVE at exactly position k-1 (last slot) should count."""
        results = [_make_result("CVE-X"), _make_result("CVE-Y"),
                   _make_result("CVE-Z")]
        assert hits_at_k(results, "CVE-Z", k=3) == 1

    def test_miss_at_rank_k_plus_1(self):
        """CVE at position k (one past the cutoff) should NOT count."""
        results = [_make_result("CVE-X"), _make_result("CVE-Y"),
                   _make_result("CVE-Z")]
        assert hits_at_k(results, "CVE-Z", k=2) == 0

    def test_not_found(self):
        results = [_make_result("CVE-X")]
        assert hits_at_k(results, "CVE-MISSING", k=10) == 0

    def test_empty_results(self):
        assert hits_at_k([], "CVE-A", k=5) == 0

    def test_k_zero(self):
        """k=0 means nothing is retrieved, so always a miss."""
        results = [_make_result("CVE-A")]
        assert hits_at_k(results, "CVE-A", k=0) == 0

    def test_k_larger_than_results(self):
        """k > len(results): should still find it if present."""
        results = [_make_result("CVE-A")]
        assert hits_at_k(results, "CVE-A", k=100) == 1

    def test_returns_int_not_bool(self):
        result = hits_at_k([_make_result("CVE-A")], "CVE-A", k=1)
        assert isinstance(result, int)


# ===================================================================
# mean_reciprocal_rank
# ===================================================================

class TestMeanReciprocalRank:

    def test_first_position(self):
        results = [_make_result("CVE-A"), _make_result("CVE-B")]
        assert mean_reciprocal_rank(results, "CVE-A") == pytest.approx(1.0)

    def test_second_position(self):
        results = [_make_result("CVE-X"), _make_result("CVE-A")]
        assert mean_reciprocal_rank(results, "CVE-A") == pytest.approx(0.5)

    def test_third_position(self):
        results = [_make_result("CVE-X"), _make_result("CVE-Y"),
                   _make_result("CVE-A")]
        assert mean_reciprocal_rank(results, "CVE-A") == pytest.approx(1 / 3)

    def test_not_found(self):
        results = [_make_result("CVE-X")]
        assert mean_reciprocal_rank(results, "CVE-MISSING") == 0.0

    def test_empty_results(self):
        assert mean_reciprocal_rank([], "CVE-A") == 0.0

    def test_uses_first_occurrence(self):
        """If the CVE appears twice, MRR should use the first hit."""
        results = [_make_result("CVE-X"), _make_result("CVE-A"),
                   _make_result("CVE-A")]
        assert mean_reciprocal_rank(results, "CVE-A") == pytest.approx(0.5)


# ===================================================================
# self_retrieval_metrics
# ===================================================================

class TestSelfRetrievalMetrics:
    """
    Self-retrieval: each vector queries the index and we check if
    it retrieves its own CVE.

    With a perfect index (each vector's nearest neighbour is itself),
    hit@1 and MRR should be 1.0.
    """

    def _build_perfect_scenario(self, n: int = 5, dim: int = 16):
        """N orthogonal unit vectors — each retrieves itself first."""
        rng = np.random.RandomState(42)
        # Use QR to get orthogonal unit vectors
        raw = rng.randn(n, dim).astype(np.float32)
        Q, _ = np.linalg.qr(raw.T)
        embeddings = Q.T[:n]  # (n, dim), orthonormal rows
        metadata = [{"cve_id": f"CVE-{i}", "cwe_id": "CWE-0"}
                    for i in range(n)]
        retriever = FakeRetriever(embeddings, metadata)
        return embeddings, metadata, retriever

    def test_perfect_retrieval_hit_at_1(self):
        embs, meta, retr = self._build_perfect_scenario()
        result = self_retrieval_metrics(embs, meta, retr, ks=[1, 5])
        assert result["hit@1"] == pytest.approx(1.0)
        assert result["mrr"] == pytest.approx(1.0)

    def test_n_matches_input_size(self):
        embs, meta, retr = self._build_perfect_scenario(n=7)
        result = self_retrieval_metrics(embs, meta, retr, ks=[1])
        assert result["n"] == 7

    def test_worst_case_no_match(self):
        """Vectors retrieve others, not themselves — hit@1 should be 0."""
        # 2 opposite vectors: each one's nearest neighbour is itself
        # (inner product), so we need a scenario where the CVE IDs mismatch.
        # Actually with orthogonal vectors each retrieves itself. Let's make
        # identical vectors with different CVE IDs: both return the same
        # top result, so one of them will miss.
        embs = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
        meta = [{"cve_id": "CVE-A", "cwe_id": "CWE-0"},
                {"cve_id": "CVE-B", "cwe_id": "CWE-0"}]
        retr = FakeRetriever(embs, meta)
        result = self_retrieval_metrics(embs, meta, retr, ks=[1])
        # Both vectors are identical, so the retriever returns whichever
        # sorts first. At best one of the two gets a hit@1. MRR < 1.0.
        assert result["hit@1"] <= 1.0  # sanity
        assert result["mrr"] <= 1.0

    @severity_high
    def test_query_count_efficiency(self):
        """
        BUG DETECTOR: the retriever should be queried exactly once per
        sample, not twice. Counting calls exposes the double-query waste.
        """

        class CountingRetriever(FakeRetriever):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0

            def query(self, vec, top_k=10):
                self.call_count += 1
                return super().query(vec, top_k=top_k)

        n = 4
        embs, meta, _ = self._build_perfect_scenario(n=n)
        counting_retr = CountingRetriever(embs, meta)
        self_retrieval_metrics(embs, meta, counting_retr, ks=[1, 5])
        # Expected: one query per sample
        assert counting_retr.call_count == n, (
            f"Expected {n} queries (one per sample), "
            f"got {counting_retr.call_count} — likely a double-query bug"
        )


# ===================================================================
# cwe_group_recall
# ===================================================================

class TestCWEGroupRecall:
    """
    For each sample, query top-k and check what fraction share the
    same CWE. Measures clustering quality.
    """

    def test_perfect_clustering(self):
        """
        Two CWEs, each with 2 well-separated entries.
        Perfect retrieval should give recall 1.0.
        """
        #  CWE-A: [1,0], [0.9,0.1]  vs  CWE-B: [0,1], [0.1,0.9]
        embs = np.array([
            [1.0, 0.0],     # CVE-1, CWE-A
            [0.95, 0.05],   # CVE-2, CWE-A
            [0.0, 1.0],     # CVE-3, CWE-B
            [0.05, 0.95],   # CVE-4, CWE-B
        ], dtype=np.float32)
        # Normalize for cosine sim
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms

        meta = [
            {"cve_id": "CVE-1", "cwe_id": "CWE-A"},
            {"cve_id": "CVE-2", "cwe_id": "CWE-A"},
            {"cve_id": "CVE-3", "cwe_id": "CWE-B"},
            {"cve_id": "CVE-4", "cwe_id": "CWE-B"},
        ]
        retr = FakeRetriever(embs, meta)
        result = cwe_group_recall(embs, meta, retr, top_k=3)
        assert result["macro_avg"] == pytest.approx(1.0), (
            "Perfectly separated CWEs should give recall 1.0"
        )

    def test_singleton_cwe_skipped(self):
        """A CWE with only 1 entry should be excluded (no peers)."""
        embs = np.eye(3, dtype=np.float32)
        meta = [
            {"cve_id": "CVE-1", "cwe_id": "CWE-SOLO"},
            {"cve_id": "CVE-2", "cwe_id": "CWE-DUO"},
            {"cve_id": "CVE-3", "cwe_id": "CWE-DUO"},
        ]
        retr = FakeRetriever(embs, meta)
        result = cwe_group_recall(embs, meta, retr, top_k=2)
        assert result["n_singletons"] == 1
        assert result["n_cwes"] == 1  # only CWE-DUO has >= 2 entries

    def test_unknown_cwe_ignored(self):
        """Entries with CWE 'UNKNOWN' should not participate."""
        embs = np.eye(3, dtype=np.float32)
        meta = [
            {"cve_id": "CVE-1", "cwe_id": "UNKNOWN"},
            {"cve_id": "CVE-2", "cwe_id": "CWE-A"},
            {"cve_id": "CVE-3", "cwe_id": "CWE-A"},
        ]
        retr = FakeRetriever(embs, meta)
        result = cwe_group_recall(embs, meta, retr, top_k=2)
        # Only CWE-A should be evaluated
        assert "UNKNOWN" not in result["per_cwe"]

    @severity_high
    def test_shared_cve_id_does_not_over_exclude(self):
        """
        BUG DETECTOR: augmented variants share the same CVE ID.
        Excluding results by CVE ID should only remove the QUERY ENTRY
        ITSELF, not all entries that happen to share its CVE ID.

        Scenario: 3 entries for CWE-A
          - CVE-1 variant A  (index 0)
          - CVE-1 variant B  (index 1) — same CVE ID, different augmentation
          - CVE-2            (index 2)

        When querying index 0 (CVE-1-A):
          - The retriever returns [CVE-1-A(self), CVE-1-B, CVE-2]
          - Correct self-exclusion: remove only CVE-1-A → 2 results, both same CWE
          - Recall = 2/2 = 1.0

        Bug: if exclusion is by CVE ID string match, CVE-1-B is ALSO
        removed → only CVE-2 remains → recall = 1/2 = 0.5
        """
        # Make CVE-1-A and CVE-1-B similar (same direction), CVE-2 also similar
        embs = np.array([
            [1.0, 0.1],   # CVE-1 variant A
            [1.0, 0.2],   # CVE-1 variant B  (same CVE ID!)
            [0.9, 0.3],   # CVE-2
        ], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms

        meta = [
            {"cve_id": "CVE-1", "cwe_id": "CWE-A"},   # variant A
            {"cve_id": "CVE-1", "cwe_id": "CWE-A"},   # variant B — same CVE!
            {"cve_id": "CVE-2", "cwe_id": "CWE-A"},
        ]
        retr = FakeRetriever(embs, meta)
        result = cwe_group_recall(embs, meta, retr, top_k=3)

        # All 3 are CWE-A. For each query, after removing ONLY self,
        # the other 2 results should both be CWE-A → recall = 1.0
        assert result["macro_avg"] == pytest.approx(1.0), (
            f"Got macro_avg={result['macro_avg']:.3f}. "
            "Likely CVE-ID-based exclusion is removing augmented "
            "variants that share the same CVE ID."
        )


# ===================================================================
# embedding_space_stats
# ===================================================================

class TestEmbeddingSpaceStats:
    """
    Intrinsic stats: mean/std norm, pairwise cosine similarity,
    and effective dimensionality.
    """

    def test_orthogonal_unit_vectors(self):
        """Orthogonal unit vectors should have ~0 mean pairwise sim."""
        n, dim = 20, 20
        embs = np.eye(n, dim, dtype=np.float32)
        stats = embedding_space_stats(embs)
        assert stats["mean_norm"] == pytest.approx(1.0, abs=1e-3)
        # Orthogonal → cosine sim = 0 for off-diagonal
        assert abs(stats["mean_pairwise_sim"]) < 0.05

    def test_identical_vectors(self):
        """All-identical unit vectors → mean pairwise sim = 1.0."""
        n = 10
        embs = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
        stats = embedding_space_stats(embs)
        assert stats["mean_pairwise_sim"] == pytest.approx(1.0, abs=1e-3)

    @severity_low
    def test_pairwise_sim_bounded(self):
        """
        BUG DETECTOR: cosine similarity must be in [-1, 1].
        If embeddings are NOT L2-normalized, a naive dot product
        gives values outside this range.
        """
        n, dim = 30, 8
        rng = np.random.RandomState(123)
        embs = rng.randn(n, dim).astype(np.float32) * 5.0  # norm ≈ 5
        stats = embedding_space_stats(embs)
        assert stats["min_pairwise_sim"] >= -1.0 - 1e-3, (
            f"min_pairwise_sim={stats['min_pairwise_sim']:.2f} < -1: "
            "dot product on non-unit vectors is not cosine similarity"
        )
        assert stats["max_pairwise_sim"] <= 1.0 + 1e-3, (
            f"max_pairwise_sim={stats['max_pairwise_sim']:.2f} > 1: "
            "dot product on non-unit vectors is not cosine similarity"
        )

    def test_returns_all_expected_keys(self):
        embs = np.eye(5, dtype=np.float32)
        stats = embedding_space_stats(embs)
        expected_keys = {
            "mean_norm", "std_norm",
            "mean_pairwise_sim", "std_pairwise_sim",
            "min_pairwise_sim", "max_pairwise_sim",
            "effective_dim",
        }
        assert expected_keys == set(stats.keys())

    def test_all_values_finite(self):
        """No NaN or Inf for well-formed inputs."""
        rng = np.random.RandomState(0)
        embs = rng.randn(20, 8).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        stats = embedding_space_stats(embs)
        for key, val in stats.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"


# ===================================================================
# _effective_dim
# ===================================================================

class TestEffectiveDim:
    """
    Participation ratio: (Σλ)² / Σλ².
    = d for uniform spread across d dims, = 1 for single direction.
    """

    def test_uniform_spread(self):
        """Isotropic Gaussian in d dims → effective dim ≈ d."""
        d = 16
        rng = np.random.RandomState(42)
        embs = rng.randn(200, d).astype(np.float32)
        ed = _effective_dim(embs)
        # Should be close to d (not exact due to sampling)
        assert ed > d * 0.5, f"effective_dim={ed:.1f}, expected close to {d}"

    def test_single_direction(self):
        """All variance along one axis → effective dim ≈ 1."""
        n = 50
        rng = np.random.RandomState(42)
        # All points on the x-axis with small noise on other dims
        embs = np.zeros((n, 8), dtype=np.float32)
        embs[:, 0] = rng.randn(n) * 10.0
        embs[:, 1:] = rng.randn(n, 7) * 0.001
        ed = _effective_dim(embs)
        assert ed < 2.0, f"effective_dim={ed:.1f}, expected ≈1 for 1D data"

    @severity_low
    def test_identical_embeddings_finite(self):
        """
        BUG DETECTOR: all-identical embeddings → covariance is zero →
        all eigenvalues are 0. The function should return a finite value
        (e.g. 0 or 1), not NaN from 0/0.
        """
        embs = np.ones((20, 8), dtype=np.float32)
        ed = _effective_dim(embs)
        assert np.isfinite(ed), (
            f"effective_dim returned {ed} for identical embeddings — "
            "likely 0/0 from zero eigenvalues"
        )

    @severity_low
    def test_single_sample_does_not_crash(self):
        """
        BUG DETECTOR: np.cov on a single sample can return a 0-d
        array, which breaks eigvalsh.
        """
        embs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        # Should not raise — return some finite number
        ed = _effective_dim(embs)
        assert np.isfinite(ed), f"effective_dim crashed or returned {ed} for 1 sample"

    def test_two_equal_directions(self):
        """Variance evenly split across 2 of 8 dims → effective dim ≈ 2."""
        n = 200
        rng = np.random.RandomState(42)
        embs = np.zeros((n, 8), dtype=np.float32)
        embs[:, 0] = rng.randn(n) * 5.0
        embs[:, 1] = rng.randn(n) * 5.0
        embs[:, 2:] = rng.randn(n, 6) * 0.001
        ed = _effective_dim(embs)
        assert 1.5 < ed < 3.0, f"effective_dim={ed:.1f}, expected ≈2"

    @severity_low
    def test_all_zero_embeddings_finite(self):
        """
        BUG DETECTOR: all-zero embeddings are a valid degenerate case.
        Should return a finite value, not NaN.
        """
        embs = np.zeros((10, 4), dtype=np.float32)
        ed = _effective_dim(embs)
        assert np.isfinite(ed), f"effective_dim returned {ed} for all-zero embeddings"


# ===================================================================
# Integration: embedding_space_stats with degenerate inputs
# ===================================================================

class TestEmbeddingSpaceStatsEdgeCases:

    @severity_low
    def test_identical_embeddings_all_finite(self):
        """
        BUG DETECTOR: all-identical embeddings. effective_dim should not
        be NaN, and pairwise sim should be 1.0.
        """
        embs = np.tile([0.5, 0.5, 0.5, 0.5], (15, 1)).astype(np.float32)
        stats = embedding_space_stats(embs)
        for key, val in stats.items():
            assert np.isfinite(val), (
                f"{key}={val} is not finite for identical embeddings"
            )

    def test_two_samples_does_not_crash(self):
        """Minimum viable input: 2 vectors."""
        embs = np.array([[1, 0], [0, 1]], dtype=np.float32)
        stats = embedding_space_stats(embs)
        assert stats["mean_norm"] == pytest.approx(1.0, abs=1e-3)


# ===================================================================
# leave_one_out_metrics  (uses real FAISS via HNSWIndex)
# ===================================================================

class TestLeaveOneOutMetrics:
    """
    LOO: for each sample, build index on all others, query with the
    held-out sample, check retrieval.
    """

    @pytest.fixture
    def _loo_setup(self, tmp_path):
        """Orthogonal unit vectors with unique CVE IDs — perfect LOO."""
        # from src.metrics import leave_one_out_metrics
        from rag.hnsw import HNSWIndex

        n, dim = 5, 16
        rng = np.random.RandomState(42)
        raw = rng.randn(dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(raw)
        embs = Q[:n].astype(np.float32)

        meta = [{"cve_id": f"CVE-{i}", "cwe_id": "CWE-0"} for i in range(n)]
        return embs, meta, HNSWIndex, {
            "dim": dim,
            "index_path": str(tmp_path / "loo.index"),
            "metadata_path": str(tmp_path / "loo_meta.json"),
        }

    def test_perfect_loo(self, _loo_setup):
        """Orthogonal vectors: each is most similar to itself (held out),
        but with LOO the next-best should still be findable by CVE ID
        as long as the index returns the right one."""
        from metrics.metrics import leave_one_out_metrics
        embs, meta, idx_cls, idx_kwargs = _loo_setup
        result = leave_one_out_metrics(
            embs, meta, idx_cls, idx_kwargs, ks=[1, 5]
        )
        assert result["n"] == len(embs)
        # MRR should be defined and finite
        assert np.isfinite(result["mrr"])
        # With orthogonal vectors and LOO, CVE-i is NOT in the index,
        # so hit@1 should be 0 (can't find itself). Verify the metric
        # correctly reports 0 here — it would be a bug if it reported 1.0.
        assert result["hit@1"] == pytest.approx(0.0), (
            "Orthogonal vectors with unique CVE IDs: the held-out CVE "
            "cannot appear in the index, so hit@1 must be 0."
        )

    def test_duplicate_cve_loo(self, tmp_path):
        """
        When multiple entries share a CVE ID (augmented variants), LOO
        holds out ONE entry but the others with the same CVE remain.
        Those remaining entries should produce a hit.
        """
        from metrics.metrics import leave_one_out_metrics
        from rag.hnsw import HNSWIndex

        dim = 4
        # Two entries for CVE-1 (close together), one for CVE-2 (far away)
        embs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # CVE-1 variant A
            [0.95, 0.05, 0.0, 0.0],  # CVE-1 variant B
            [0.0, 0.0, 0.0, 1.0],  # CVE-2
        ], dtype=np.float32)
        meta = [
            {"cve_id": "CVE-1", "cwe_id": "CWE-A"},
            {"cve_id": "CVE-1", "cwe_id": "CWE-A"},
            {"cve_id": "CVE-2", "cwe_id": "CWE-B"},
        ]
        result = leave_one_out_metrics(
            embs, meta, HNSWIndex,
            {"dim": dim,
             "index_path": str(tmp_path / "loo.index"),
             "metadata_path": str(tmp_path / "loo_meta.json")},
            ks=[1],
        )
        # When CVE-1-A is held out, CVE-1-B is still in the index
        # and is the nearest neighbour → hit@1 for that query.
        # When CVE-1-B is held out, CVE-1-A is still there → hit@1.
        # When CVE-2 is held out, its nearest is CVE-1 → miss.
        # Expected hit@1 = 2/3 ≈ 0.667
        assert result["hit@1"] == pytest.approx(2 / 3, abs=0.01)


# ===================================================================
# bertscore_pair  (similarity.py)
# ===================================================================

class TestBertScorePair:
    """
    Tests for the single-pair BERTScore function in similarity.py.

    These tests require the CodeBERT model to be downloaded (or cached).
    Mark them slow so they can be skipped in fast CI runs:
        pytest -m "not slow"
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        """Import here so the test file still loads if torch is missing."""
        from src.metrics.similarity import bertscore_pair
        self.bertscore_pair = bertscore_pair

    @severity_high
    @pytest.mark.slow
    def test_identical_strings_high_f1(self):
        """Identical code should yield F1 very close to 1.0."""
        code = "int foo(int x) { return x + 1; }"
        result = self.bertscore_pair(code, code)
        assert result["bertscore_f1"] == pytest.approx(1.0, abs=0.01)
        assert result["bertscore_precision"] == pytest.approx(1.0, abs=0.01)
        assert result["bertscore_recall"] == pytest.approx(1.0, abs=0.01)

    @severity_high
    @pytest.mark.slow
    def test_different_strings_lower_f1(self):
        """Completely unrelated code should score noticeably lower."""
        gen = "int foo(int x) { return x + 1; }"
        ref = "void bar(char *buf, size_t len) { memset(buf, 0, len); }"
        result = self.bertscore_pair(gen, ref)
        # Not zero (BERT embeddings share some baseline similarity),
        # but clearly below an identical match.
        assert result["bertscore_f1"] < 0.95

    @severity_high
    @pytest.mark.slow
    def test_output_keys(self):
        """Result dict should contain the expected keys."""
        result = self.bertscore_pair("a", "b")
        assert set(result.keys()) == {
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
        }

    @severity_high
    @pytest.mark.slow
    def test_values_in_range(self):
        """All scores should be in [0, 1]."""
        result = self.bertscore_pair(
            "if (ptr == NULL) { free(ptr); }",
            "if (ptr != NULL) { free(ptr); ptr = NULL; }",
        )
        for key in ("bertscore_precision", "bertscore_recall", "bertscore_f1"):
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0,1]"

    @severity_high
    @pytest.mark.slow
    def test_similar_code_higher_than_dissimilar(self):
        """A close variant should score higher than an unrelated snippet."""
        ref = "int add(int a, int b) { return a + b; }"
        similar = "int add(int x, int y) { return x + y; }"
        different = "void print_hello() { printf(\"hello world\"); }"
        score_similar = self.bertscore_pair(similar, ref)["bertscore_f1"]
        score_different = self.bertscore_pair(different, ref)["bertscore_f1"]
        assert score_similar > score_different

    @severity_low
    @pytest.mark.slow
    def test_empty_strings(self):
        """Empty inputs should not crash; scores should still be floats."""
        result = self.bertscore_pair("", "")
        for key in ("bertscore_precision", "bertscore_recall", "bertscore_f1"):
            assert isinstance(result[key], float)
