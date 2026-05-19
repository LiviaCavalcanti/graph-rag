"""Tests for experiments/common.py and src/metrics/retrieval_eval."""

import random

import numpy as np
import networkx as nx
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.base import FunctionPair
from src.data.split import (
    _is_original,
    _split_by_variant,
    _stratified_split,
    _sample_pairs,
    build_split,
)
from experiments.common import (
    softmax,
    is_uncertain,
)
from src.metrics.retrieval_eval import evaluate_retrieval, evaluate_cwe_recall


# ── helpers ──────────────────────────────────────────────────────────


def _make_pair(cve_id="CVE-2025-0001", cwe_id="CWE-119", variant="original", **extra):
    G = nx.MultiDiGraph()
    meta = {"variant": variant, "dir_name": cve_id, **extra}
    return FunctionPair(
        cve_id=cve_id,
        cwe_id=cwe_id,
        func_name="fn",
        project="test",
        G_before=G,
        G_after=G,
        G_vuln=G,
        meta=meta,
    )


def _make_pairs(specs: list[tuple[str, str, str]]) -> list[FunctionPair]:
    """Create pairs from (cve_id, cwe_id, variant) tuples."""
    return [_make_pair(cve_id=s[0], cwe_id=s[1], variant=s[2]) for s in specs]


# ── _is_original ─────────────────────────────────────────────────────


class TestIsOriginal:
    def test_original(self):
        assert _is_original(_make_pair(variant="original")) is True

    def test_augmented(self):
        assert _is_original(_make_pair(variant="augmented_v1")) is False

    def test_no_variant(self):
        p = _make_pair()
        p.meta.pop("variant")
        assert _is_original(p) is False


# ── _split_by_variant ────────────────────────────────────────────────


class TestSplitByVariant:
    def test_basic_split(self):
        pairs = _make_pairs([
            ("CVE-1", "CWE-1", "original"),
            ("CVE-2", "CWE-1", "augmented_v1"),
            ("CVE-3", "CWE-2", "original"),
        ])
        orig, aug = _split_by_variant(pairs)
        assert len(orig) == 2
        assert len(aug) == 1
        assert all(_is_original(p) for p in orig)

    def test_empty(self):
        orig, aug = _split_by_variant([])
        assert orig == [] and aug == []

    def test_all_original(self):
        pairs = _make_pairs([("CVE-1", "CWE-1", "original")])
        orig, aug = _split_by_variant(pairs)
        assert len(orig) == 1 and len(aug) == 0

    def test_all_augmented(self):
        pairs = _make_pairs([("CVE-1", "CWE-1", "aug_v1")])
        orig, aug = _split_by_variant(pairs)
        assert len(orig) == 0 and len(aug) == 1


# ── _stratified_split ────────────────────────────────────────────────


class TestStratifiedSplit:
    def test_empty(self):
        train, test = _stratified_split([], 0.2, 42)
        assert train == [] and test == []

    def test_single_pair(self):
        pairs = [_make_pair()]
        train, test = _stratified_split(pairs, 0.2, 42)
        assert len(train) == 1 and len(test) == 0

    def test_deterministic(self):
        pairs = _make_pairs([
            (f"CVE-{i}", "CWE-119", "aug") for i in range(20)
        ])
        t1, q1 = _stratified_split(pairs, 0.2, 42)
        t2, q2 = _stratified_split(pairs, 0.2, 42)
        assert [p.cve_id for p in t1] == [p.cve_id for p in t2]
        assert [p.cve_id for p in q1] == [p.cve_id for p in q2]

    def test_different_seeds(self):
        pairs = _make_pairs([
            (f"CVE-{i}", "CWE-119", "aug") for i in range(20)
        ])
        t1, q1 = _stratified_split(pairs, 0.2, 42)
        t2, q2 = _stratified_split(pairs, 0.2, 99)
        # different seeds should (very likely) produce different orderings
        assert [p.cve_id for p in t1] != [p.cve_id for p in t2]

    def test_preserves_all_pairs(self):
        pairs = _make_pairs([
            (f"CVE-{i}", f"CWE-{i % 3}", "aug") for i in range(30)
        ])
        train, test = _stratified_split(pairs, 0.3, 42)
        all_ids = sorted(p.cve_id for p in train + test)
        expected = sorted(p.cve_id for p in pairs)
        assert all_ids == expected

    def test_both_sets_nonempty(self):
        pairs = _make_pairs([
            (f"CVE-{i}", "CWE-1", "aug") for i in range(5)
        ])
        train, test = _stratified_split(pairs, 0.2, 42)
        assert len(train) > 0 and len(test) > 0

    def test_clamped_ratio(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(10)])
        train, test = _stratified_split(pairs, 1.5, 42)  # clamped to 0.9
        assert len(train) > 0  # at least 1 in train
        assert len(test) > 0

    def test_unknown_cwe_grouped(self):
        pairs = _make_pairs([
            ("CVE-1", "UNKNOWN", "aug"),
            ("CVE-2", "UNKNOWN", "aug"),
            ("CVE-3", "", "aug"),
            ("CVE-4", "CWE-119", "aug"),
        ])
        train, test = _stratified_split(pairs, 0.5, 42)
        assert len(train) + len(test) == 4


# ── _sample_pairs ────────────────────────────────────────────────────


class TestSamplePairs:
    def test_full_ratio(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(10)])
        sampled = _sample_pairs(pairs, 1.0, 42)
        assert len(sampled) == 10

    def test_zero_ratio(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(10)])
        sampled = _sample_pairs(pairs, 0.0, 42)
        assert sampled == []

    def test_half_ratio(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(10)])
        sampled = _sample_pairs(pairs, 0.5, 42)
        assert len(sampled) == 5

    def test_empty(self):
        assert _sample_pairs([], 0.5, 42) == []

    def test_deterministic(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(10)])
        s1 = _sample_pairs(pairs, 0.3, 42)
        s2 = _sample_pairs(pairs, 0.3, 42)
        assert [p.cve_id for p in s1] == [p.cve_id for p in s2]

    def test_at_least_one(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(10)])
        sampled = _sample_pairs(pairs, 0.01, 42)
        assert len(sampled) >= 1


# ── build_split ──────────────────────────────────────────────────────


class TestBuildSplit:
    def _cfg(self, enabled=False, **overrides):
        split = {"enabled": enabled, "seed": 42, "test_ratio": 0.2, **overrides}
        return {"experiment": {"split": split}}

    def test_disabled_returns_all(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(5)])
        idx, qry, info = build_split(pairs, self._cfg(enabled=False))
        assert len(idx) == 5 and len(qry) == 5
        assert info["enabled"] is False

    def test_enabled_splits(self):
        # need some augmented pairs for the split to be meaningful
        pairs = _make_pairs(
            [("CVE-R1", "CWE-1", "original")]
            + [(f"CVE-A{i}", "CWE-1", "augmented_v1") for i in range(10)]
        )
        idx, qry, info = build_split(pairs, self._cfg(enabled=True))
        assert info["enabled"] is True
        assert len(idx) > 0 and len(qry) > 0
        # index + query should cover all augmented pairs
        assert info["counts"]["aug_train_total"] + info["counts"]["aug_test_total"] == 10

    def test_seed_override(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(20)])
        _, q1, _ = build_split(pairs, self._cfg(enabled=True), seed_override=1)
        _, q2, _ = build_split(pairs, self._cfg(enabled=True), seed_override=2)
        ids1 = sorted(p.cve_id for p in q1)
        ids2 = sorted(p.cve_id for p in q2)
        # different seeds should produce different query sets (very high probability)
        assert ids1 != ids2

    def test_aug_train_ratio(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(20)])
        _, _, info_full = build_split(
            pairs, self._cfg(enabled=True, augmented_train_ratio=1.0)
        )
        _, _, info_half = build_split(
            pairs, self._cfg(enabled=True, augmented_train_ratio=0.5)
        )
        assert info_half["counts"]["aug_train_used"] < info_full["counts"]["aug_train_used"]

    def test_include_real_in_index(self):
        pairs = _make_pairs(
            [("CVE-R1", "CWE-1", "original"), ("CVE-R2", "CWE-1", "original")]
            + [(f"CVE-A{i}", "CWE-1", "augmented") for i in range(10)]
        )
        idx_yes, _, _ = build_split(
            pairs, self._cfg(enabled=True, include_real_in_index=True)
        )
        idx_no, _, _ = build_split(
            pairs, self._cfg(enabled=True, include_real_in_index=False)
        )
        assert len(idx_yes) > len(idx_no)

    def test_none_cfg_is_safe(self):
        pairs = _make_pairs([("CVE-1", "CWE-1", "aug")])
        idx, qry, info = build_split(pairs, None)
        assert info["enabled"] is False

    def test_empty_pairs(self):
        idx, qry, info = build_split([], self._cfg(enabled=True))
        assert idx == [] or qry == [] or not info["enabled"]


# ── softmax ──────────────────────────────────────────────────────────


class TestSoftmax:
    def test_sums_to_one(self):
        result = softmax([1.0, 2.0, 3.0])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_empty(self):
        assert softmax([]) == []

    def test_single_element(self):
        assert softmax([5.0]) == [1.0]

    def test_large_values_stable(self):
        result = softmax([1000.0, 1001.0, 1002.0])
        assert abs(sum(result) - 1.0) < 1e-9
        # last should be largest
        assert result[2] > result[1] > result[0]

    def test_temperature(self):
        sharp = softmax([1.0, 2.0, 3.0], temperature=0.1)
        flat = softmax([1.0, 2.0, 3.0], temperature=10.0)
        # lower temperature → more peaked distribution
        assert max(sharp) > max(flat)


# ── is_uncertain ─────────────────────────────────────────────────────


class TestIsUncertain:
    def test_confident(self):
        assert is_uncertain(0.9, 0.3) is False

    def test_low_prob(self):
        assert is_uncertain(0.05, 0.3) is True

    def test_low_margin(self):
        assert is_uncertain(0.9, 0.001) is True

    def test_both_low(self):
        assert is_uncertain(0.01, 0.001) is True

    def test_at_threshold(self):
        assert is_uncertain(0.12, 0.005) is False


# ── load_pairs / load_pairs_lightweight ──────────────────────────────


class TestLoadPairs:
    """Test load_pairs and load_pairs_lightweight with mocked AutoPatchDataset."""

    def _cfg(self):
        return {"data": {"active": ["autopatch"], "autopatch": {"root": "/fake/path"}}}

    def test_load_pairs_delegates(self):
        sentinel = [MagicMock(), MagicMock()]
        with patch("src.data.autopatch.AutoPatchDataset") as MockDS:
            MockDS.return_value.load_all.return_value = sentinel
            from src.data import load_pairs
            result = load_pairs(self._cfg())
        MockDS.assert_called_once_with({"root": "/fake/path"})
        MockDS.return_value.load_all.assert_called_once()
        assert result is sentinel

    def test_load_pairs_lightweight_delegates(self):
        sentinel = [MagicMock()]
        with patch("src.data.autopatch.AutoPatchDataset") as MockDS:
            MockDS.return_value.load_lightweight.return_value = sentinel
            from src.data import load_pairs_lightweight
            result = load_pairs_lightweight(self._cfg())
        MockDS.assert_called_once_with({"root": "/fake/path"})
        MockDS.return_value.load_lightweight.assert_called_once()
        assert result is sentinel


# ── evaluate_retrieval (ranx-backed) ────────────────────────────────


class _FakeRetriever:
    """Brute-force inner-product retriever for testing."""

    def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
        self._embs = embeddings.astype(np.float32)
        self._meta = metadata

    def query(self, vec: np.ndarray, top_k: int = 10) -> list[dict]:
        scores = self._embs @ vec.ravel()
        order = np.argsort(-scores)[:top_k]
        return [{**self._meta[i], "score": float(scores[i]), "_idx": int(i)} for i in order]


class TestEvaluateRetrieval:
    """Tests for the ranx-backed evaluate_retrieval function."""

    def _setup(self):
        """3 pairs, orthogonal embeddings → perfect self-retrieval."""
        dim = 3
        embs = np.eye(dim, dtype=np.float32)
        pairs = [
            _make_pair(cve_id=f"CVE-{i}", cwe_id=f"CWE-{i}", variant="original")
            for i in range(dim)
        ]
        meta = [
            {"cve_id": p.cve_id, "cwe_id": p.cwe_id, "func_name": p.func_name, "variant": "original"}
            for p in pairs
        ]
        retriever = _FakeRetriever(embs, meta)
        return pairs, embs, retriever

    def test_perfect_retrieval(self):
        pairs, embs, retriever = self._setup()
        result = evaluate_retrieval(pairs, embs, retriever, pairs, ks=[1, 3])
        assert result["hit@1"] == pytest.approx(1.0)
        assert result["mrr"] == pytest.approx(1.0)
        assert result["n"] == 3
        assert "ndcg@1" in result
        assert "map@1" in result

    def test_no_hits(self):
        """Queries retrieve the wrong CVE at rank 1."""
        dim = 3
        # Index has 3 orthogonal vectors
        index_embs = np.eye(dim, dtype=np.float32)
        pairs = [
            _make_pair(cve_id=f"CVE-{i}", cwe_id=f"CWE-{i}", variant="original")
            for i in range(dim)
        ]
        meta = [
            {"cve_id": p.cve_id, "cwe_id": p.cwe_id, "func_name": p.func_name, "variant": "original"}
            for p in pairs
        ]
        retriever = _FakeRetriever(index_embs, meta)
        # Query embeddings: each points toward the NEXT pair's direction
        query_embs = np.roll(index_embs, shift=1, axis=0)
        result = evaluate_retrieval(pairs, query_embs, retriever, pairs, ks=[1])
        assert result["hit@1"] == pytest.approx(0.0)

    def test_zero_vectors_skipped(self):
        pairs, embs, retriever = self._setup()
        embs[0] = 0.0  # zero out first
        result = evaluate_retrieval(pairs, embs, retriever, pairs, ks=[1])
        assert result["n"] == 2

    def test_empty_returns_n_zero(self):
        result = evaluate_retrieval([], np.array([]).reshape(0, 3), _FakeRetriever(np.eye(3), []), [], ks=[1])
        assert result["n"] == 0


class TestEvaluateCweRecall:
    """Tests for the ranx-backed evaluate_cwe_recall function."""

    def test_perfect_cwe_recall(self):
        """All share CWE-A, so recall should be 1.0."""
        dim = 3
        embs = np.eye(dim, dtype=np.float32)
        pairs = [
            _make_pair(cve_id=f"CVE-{i}", cwe_id="CWE-A", variant="original")
            for i in range(dim)
        ]
        meta = [{"cve_id": p.cve_id, "cwe_id": "CWE-A", "func_name": "fn", "variant": "original"} for p in pairs]
        retriever = _FakeRetriever(embs, meta)
        result = evaluate_cwe_recall(pairs, embs, retriever, meta, top_k=3)
        assert result["macro_avg"] == pytest.approx(1.0)
        assert result["n_cwes"] == 1

    def test_unknown_cwe_skipped(self):
        dim = 2
        embs = np.eye(dim, dtype=np.float32)
        pairs = [
            _make_pair(cve_id="CVE-0", cwe_id="UNKNOWN"),
            _make_pair(cve_id="CVE-1", cwe_id="CWE-A"),
        ]
        meta = [{"cve_id": "CVE-0", "cwe_id": "UNKNOWN"}, {"cve_id": "CVE-1", "cwe_id": "CWE-A"}]
        retriever = _FakeRetriever(embs, meta)
        result = evaluate_cwe_recall(pairs, embs, retriever, meta, top_k=2)
        # UNKNOWN should be skipped; CWE-A has support 1 so possible=1
        assert "UNKNOWN" not in result["per_cwe"]
