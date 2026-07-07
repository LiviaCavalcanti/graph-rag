"""
Unit tests for src.metrics.threshold_analysis.

The fixtures use hand-constructed ``raw_queries`` with known scores and labels
so every confusion count and derived metric can be verified by hand.  All
tests are offline (no model / no retrieval).
"""

from __future__ import annotations

import pytest

from metrics.threshold_analysis import (
    AXES,
    analyze_cell,
    build_analysis,
    classify_at_threshold,
    default_grid,
    error_analysis_at,
    f1_argmax,
    normalize_records,
    sweep_thresholds,
)

# ── fixture builders ─────────────────────────────────────────────────


def _q(query_cve, query_cwe, retrieved, variant="v0"):
    return {
        "query_cve": query_cve,
        "query_cwe": query_cwe,
        "query_variant": variant,
        "retrieved": retrieved,
    }


def _r(rank, cve_id, cwe_id, score):
    return {"rank": rank, "cve_id": cve_id, "cwe_id": cwe_id, "score": score}


# Controlled query set (see per-test hand computations):
#   Q1 score .9  cwe A pred A (ok)   cve C1 pred C1 (ok)
#   Q2 score .8  cwe A pred B (bad)  cve C2 pred C9 (bad)
#   Q3 score .4  cwe A pred A (ok)   cve C3 pred C3 (ok)
#   Q4 score .3  cwe A pred B (bad)  cve C4 pred C8 (bad); rank-2 has cwe A
#   Q5 score .7  cwe UNKNOWN         cve C5 pred C5 (ok)   → excluded from CWE
#   Q6 no results (abstain always)   cwe A / cve C6
BASE_QUERIES = [
    _q("C1", "A", [_r(1, "C1", "A", 0.9), _r(2, "Cx", "B", 0.2)]),
    _q("C2", "A", [_r(1, "C9", "B", 0.8), _r(2, "Cy", "B", 0.1)]),
    _q("C3", "A", [_r(1, "C3", "A", 0.4), _r(2, "Cz", "B", 0.1)]),
    _q("C4", "A", [_r(1, "C8", "B", 0.3), _r(2, "Cw", "A", 0.25)]),
    _q("C5", "UNKNOWN", [_r(1, "C5", "Z", 0.7)]),
    _q("C6", "A", []),
]


def _cell(queries):
    return {
        "embedder": "test",
        "backend": "flat",
        "graph_variant": "G_vuln",
        "self_retrieval": {"raw_queries": queries},
    }


def _row_at(sweep_rows, t):
    for row in sweep_rows:
        if abs(row["threshold"] - t) < 1e-9:
            return row
    raise AssertionError(f"threshold {t} not in grid")


# ── normalization ────────────────────────────────────────────────────


def test_normalize_flags_correctness_and_unknown():
    recs = normalize_records(BASE_QUERIES)
    assert len(recs) == 6
    q1, q2, q3, q4, q5, q6 = recs

    assert q1["cwe_correct"] and q1["cve_correct"]
    assert not q2["cwe_correct"] and not q2["cve_correct"]
    assert q3["cwe_correct"] and q3["cve_correct"]

    # Q4 top-1 is wrong, but a correct-CWE item sits at rank 2
    assert not q4["cwe_correct"]
    assert q4["first_correct_rank_cwe"] == 2

    # Q5 has an unknown CWE → excluded from CWE axis, still valid for CVE
    assert not q5["cwe_known"]
    assert q5["cve_correct"]

    # Q6 has no results → abstained at any threshold
    assert q6["top1_score"] is None
    assert not q6["cwe_correct"] and not q6["cve_correct"]


# ── sweep confusion counts (hand-verified at t=0.5) ─────────────────


def test_sweep_cwe_counts_at_half():
    rows = sweep_thresholds(normalize_records(BASE_QUERIES), "cwe", default_grid())
    row = _row_at(rows, 0.5)
    # CWE axis excludes Q5 (unknown): TP=Q1, FP=Q2, FN=Q3, TN={Q4,Q6}
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 1, 2)
    assert row["n"] == 5
    assert row["coverage"] == pytest.approx(0.4)
    assert row["selective_accuracy"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["f1"] == pytest.approx(0.5)
    assert row["accuracy"] == pytest.approx(0.6)


def test_sweep_cve_counts_at_half():
    rows = sweep_thresholds(normalize_records(BASE_QUERIES), "cve", default_grid())
    row = _row_at(rows, 0.5)
    # CVE axis includes all: TP={Q1,Q5}, FP=Q2, FN=Q3, TN={Q4,Q6}
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2, 1, 1, 2)
    assert row["n"] == 6
    assert row["coverage"] == pytest.approx(0.5)
    assert row["selective_accuracy"] == pytest.approx(2 / 3)
    assert row["accuracy"] == pytest.approx(2 / 3)


def test_classify_matches_sweep_row():
    recs = normalize_records(BASE_QUERIES)
    counts = classify_at_threshold([r for r in recs if r["cwe_known"]], "cwe", 0.5)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}


# ── properties ───────────────────────────────────────────────────────


@pytest.mark.parametrize("axis", AXES)
def test_coverage_non_increasing(axis):
    rows = sweep_thresholds(normalize_records(BASE_QUERIES), axis, default_grid())
    covs = [r["coverage"] for r in rows]
    assert all(b <= a + 1e-9 for a, b in zip(covs, covs[1:]))


@pytest.mark.parametrize("axis", AXES)
def test_low_threshold_is_full_coverage(axis):
    rows = sweep_thresholds(normalize_records(BASE_QUERIES), axis, default_grid())
    row0 = _row_at(rows, 0.0)
    # Everything with a score >= 0 is predicted; only no-result queries abstain.
    assert row0["coverage"] == pytest.approx(row0["predicted"] / row0["n"])
    assert row0["fn"] == 0  # nothing correct is left un-predicted at t=0


def test_unknown_cwe_excluded_from_cwe_axis_only():
    cell = analyze_cell(_cell(BASE_QUERIES))
    assert cell["n_queries"] == 6
    assert cell["n_queries_cwe"] == 5
    assert cell["n_unknown_cwe"] == 1
    # CVE sweep sees all 6, CWE sweep sees 5
    assert cell["sweep"]["cve"][0]["n"] == 6
    assert cell["sweep"]["cwe"][0]["n"] == 5


def test_default_grid_endpoints():
    grid = default_grid(0.05)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert len(grid) == 21


# ── error analysis groups ────────────────────────────────────────────


def test_error_analysis_cwe_groups_at_half():
    recs = normalize_records(BASE_QUERIES)
    ea = error_analysis_at(recs, "cwe", 0.5)
    has, no = ea["has_class"], ea["no_class"]

    assert has["n"] == 2 and has["correct"] == 1 and has["wrong"] == 1
    # the single wrong prediction is A → B
    assert has["wrong_cwe_confusion"] == {"A → B": 1}

    assert no["n"] == 3 and no["fn"] == 1 and no["tn"] == 2
    # Q4 (TN) had a correct CWE at rank 2 → recoverable-deeper
    assert no["tn_recoverable_in_topk"] == 1
    assert no["tn_recoverable_rank_hist"] == {"2": 1}
    # Q3 (FN) sat 0.1 below the threshold
    assert no["fn_score_deficit_stats"]["mean"] == pytest.approx(0.1)


def test_error_analysis_cve_soft_errors():
    # wrong CVE but right CWE = a "soft" error
    soft = [
        _q("C1", "A", [_r(1, "C1", "A", 0.9)]),  # TP
        _q("C2", "A", [_r(1, "C99", "A", 0.8)]),  # FP but CWE right
        _q("C3", "B", [_r(1, "C88", "Z", 0.7)]),  # FP, CWE also wrong
    ]
    ea = error_analysis_at(normalize_records(soft), "cve", 0.5)
    assert ea["has_class"]["soft_errors_wrong_cve_right_cwe"] == 1


# ── assembly ─────────────────────────────────────────────────────────


def test_build_analysis_structure():
    analysis = build_analysis({"run_id": "r1", "cells": [_cell(BASE_QUERIES)]})
    assert analysis["run_id"] == "r1"
    assert analysis["axes"] == ["cwe", "cve"]
    assert analysis["global"]["n_cells"] == 1
    assert analysis["global"]["n_queries_total"] == 6
    cell = analysis["cells"][0]
    for axis in AXES:
        assert axis in cell["sweep"]
        for label in ("p50", "p75", "p90"):
            assert label in cell["error_analysis"][axis]


def test_f1_argmax_returns_grid_threshold():
    rows = sweep_thresholds(normalize_records(BASE_QUERIES), "cwe", default_grid())
    thr = f1_argmax(rows)
    assert thr in {r["threshold"] for r in rows}
