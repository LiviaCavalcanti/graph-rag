"""Tests for AutoPatch-specific split logic (src/data/autopatch.py).

Covers the real vs. LLM-augmented-variant splitting that only makes sense
for the AutoPatch dataset, plus the transparent dispatch from the generic
``src.data.split.build_split`` whenever pairs are AutoPatch pairs
(``pair.project == "autopatch"``).
"""

import networkx as nx

from src.data.base import FunctionPair
from src.data.autopatch import AutoPatchDataset, _is_original, _split_by_variant
from src.data.split import build_split


# ── helpers ──────────────────────────────────────────────────────────


def _make_pair(cve_id="CVE-2025-0001", cwe_id="CWE-119", variant="original", **extra):
    G = nx.MultiDiGraph()
    meta = {"variant": variant, "dir_name": cve_id, **extra}
    return FunctionPair(
        cve_id=cve_id,
        cwe_id=cwe_id,
        func_name="fn",
        project="autopatch",
        G_before=G,
        G_after=G,
        G_vuln=G,
        meta=meta,
    )


def _make_pairs(specs: list[tuple[str, str, str]]) -> list[FunctionPair]:
    """Create pairs from (cve_id, cwe_id, variant) tuples."""
    return [_make_pair(cve_id=s[0], cwe_id=s[1], variant=s[2]) for s in specs]


def _cfg(enabled=False, **overrides):
    split = {"enabled": enabled, "seed": 42, "test_ratio": 0.2, **overrides}
    return {"experiment": {"split": split}}


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


# ── AutoPatchDataset.build_split ──────────────────────────────────────


class TestAutoPatchBuildSplit:
    def test_disabled_returns_all(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(5)])
        idx, qry, info = AutoPatchDataset.build_split(pairs, _cfg(enabled=False))
        assert len(idx) == 5 and len(qry) == 5
        assert info["enabled"] is False

    def test_enabled_splits(self):
        # need some augmented pairs for the split to be meaningful
        pairs = _make_pairs(
            [("CVE-R1", "CWE-1", "original")]
            + [(f"CVE-A{i}", "CWE-1", "augmented_v1") for i in range(10)]
        )
        idx, qry, info = AutoPatchDataset.build_split(pairs, _cfg(enabled=True))
        assert info["enabled"] is True
        assert len(idx) > 0 and len(qry) > 0
        # index + query should cover all augmented pairs
        assert info["counts"]["aug_train_total"] + info["counts"]["aug_test_total"] == 10

    def test_seed_override(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(20)])
        _, q1, _ = AutoPatchDataset.build_split(pairs, _cfg(enabled=True), seed_override=1)
        _, q2, _ = AutoPatchDataset.build_split(pairs, _cfg(enabled=True), seed_override=2)
        ids1 = sorted(p.cve_id for p in q1)
        ids2 = sorted(p.cve_id for p in q2)
        # different seeds should produce different query sets (very high probability)
        assert ids1 != ids2

    def test_aug_train_ratio(self):
        pairs = _make_pairs([(f"CVE-{i}", "CWE-1", "aug") for i in range(20)])
        _, _, info_full = AutoPatchDataset.build_split(
            pairs, _cfg(enabled=True, augmented_train_ratio=1.0)
        )
        _, _, info_half = AutoPatchDataset.build_split(
            pairs, _cfg(enabled=True, augmented_train_ratio=0.5)
        )
        assert info_half["counts"]["aug_train_used"] < info_full["counts"]["aug_train_used"]

    def test_include_real_in_index(self):
        pairs = _make_pairs(
            [("CVE-R1", "CWE-1", "original"), ("CVE-R2", "CWE-1", "original")]
            + [(f"CVE-A{i}", "CWE-1", "augmented") for i in range(10)]
        )
        idx_yes, _, _ = AutoPatchDataset.build_split(
            pairs, _cfg(enabled=True, include_real_in_index=True)
        )
        idx_no, _, _ = AutoPatchDataset.build_split(
            pairs, _cfg(enabled=True, include_real_in_index=False)
        )
        assert len(idx_yes) > len(idx_no)

    def test_none_cfg_is_safe(self):
        pairs = _make_pairs([("CVE-1", "CWE-1", "aug")])
        idx, qry, info = AutoPatchDataset.build_split(pairs, None)
        assert info["enabled"] is False

    def test_empty_pairs(self):
        idx, qry, info = AutoPatchDataset.build_split([], _cfg(enabled=True))
        assert idx == [] or qry == [] or not info["enabled"]


# ── generic build_split dispatch ──────────────────────────────────────


class TestBuildSplitDispatch:
    def test_dispatches_to_autopatch_for_autopatch_pairs(self):
        pairs = _make_pairs(
            [("CVE-R1", "CWE-1", "original")]
            + [(f"CVE-A{i}", "CWE-1", "augmented_v1") for i in range(10)]
        )
        idx, qry, info = build_split(pairs, _cfg(enabled=True))
        # AutoPatch-specific keys prove delegation occurred.
        assert "real_total" in info["counts"]
        assert info["counts"]["real_total"] == 1

    def test_does_not_dispatch_for_non_autopatch_pairs(self):
        pairs = [
            FunctionPair(
                cve_id=f"CVE-{i}",
                cwe_id="CWE-1",
                func_name="fn",
                project="cvefixes",
                G_before=nx.MultiDiGraph(),
                G_after=nx.MultiDiGraph(),
                G_vuln=nx.MultiDiGraph(),
                meta={},
            )
            for i in range(10)
        ]
        idx, qry, info = build_split(pairs, _cfg(enabled=True))
        assert "real_total" not in info["counts"]
        assert info["counts"]["index_total"] + info["counts"]["query_total"] == 10
