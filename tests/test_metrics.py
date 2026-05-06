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
            {**self._meta[i], "score": float(scores[i]), "_idx": int(i)}
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


# ===================================================================
# code_similarity  (similarity.py)
# ===================================================================

from src.metrics.similarity import code_similarity


class TestCodeSimilarity:
    """Tests for the line-level code_similarity function."""

    @severity_high
    def test_both_empty(self):
        """Two empty strings → 0.0 (early return guard)."""
        assert code_similarity("", "") == 0.0

    @severity_high
    def test_one_empty_gen(self):
        """Empty generated, non-empty reference → 0.0."""
        assert code_similarity("", "int x = 1;") == 0.0

    @severity_high
    def test_one_empty_ref(self):
        """Non-empty generated, empty reference → 0.0."""
        assert code_similarity("int x = 1;", "") == 0.0

    @severity_high
    def test_identical_code(self):
        """Identical multi-line code → 1.0."""
        code = "int foo(int x) {\n    return x + 1;\n}\n"
        assert code_similarity(code, code) == pytest.approx(1.0)

    @severity_high
    def test_whitespace_only_diffs(self):
        """Code differing only in leading/trailing whitespace → 1.0 (lines are stripped)."""
        gen = "  int foo(int x) {  \n    return x + 1;\n  }\n"
        ref = "int foo(int x) {\n  return x + 1;\n}\n"
        assert code_similarity(gen, ref) == pytest.approx(1.0)

    @severity_high
    def test_partial_overlap(self):
        """Some shared lines → 0 < ratio < 1."""
        gen = "int foo(int x) {\n    return x + 1;\n}\n"
        ref = "int foo(int x) {\n    return x * 2;\n}\n"
        ratio = code_similarity(gen, ref)
        assert 0.0 < ratio < 1.0

    @severity_low
    def test_blank_lines_ignored(self):
        """Blank-only lines are stripped out, so extra blank lines don't affect ratio."""
        code = "int x = 1;\nint y = 2;\n"
        code_with_blanks = "int x = 1;\n\n\nint y = 2;\n\n"
        assert code_similarity(code, code_with_blanks) == pytest.approx(1.0)

    @severity_high
    def test_completely_different(self):
        """Totally different code → low similarity."""
        gen = "void foo() { printf(\"hello\"); }"
        ref = "struct Bar { int x; int y; int z; };"
        assert code_similarity(gen, ref) < 0.5
