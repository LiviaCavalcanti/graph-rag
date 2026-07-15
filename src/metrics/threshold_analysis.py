#!/usr/bin/env python3
"""
Similarity-threshold study and abstention error analysis for retrieval results.

Given the per-query retrieval records stored in a run's ``results.json``
(``cells[i].cve_retrieval.raw_queries``), this module answers two questions:

1. **Threshold sweep** — if we only *accept* a retrieval when the top-1
   similarity score is ``>= t``, how do retrieval-quality metrics change as
   ``t`` varies?  We sweep an absolute grid of thresholds and, for each,
   compute a confusion matrix and derived metrics for two correctness axes:

     * ``cwe`` — the retrieved top-1 CWE class matches the query CWE.
     * ``cve`` — the retrieved top-1 CVE instance matches the query CVE.

2. **Abstention error analysis** — at a given threshold, every query is either
   *predicted* (``top1_score >= t`` → we assign the top-1 CWE/CVE, "has class")
   or *abstained* (``top1_score < t`` → "no class").  We report the error
   distribution within each group.

Confusion-matrix convention (per threshold ``t``, per axis):

    predicted (has class, score >= t):  TP = correct        , FP = wrong
    abstained (no class,  score <  t):  FN = would-be-correct, TN = would-be-wrong

Derived metrics:

    coverage           = predicted / N
    selective_accuracy = TP / (TP + FP)          (a.k.a. precision)
    recall             = TP / (TP + FN)
    f1                 = harmonic mean of the two
    accuracy           = (TP + TN) / N
    abstention_rate    = 1 - coverage
    risk               = 1 - selective_accuracy

This is a *post-hoc* study over already-computed retrieval results — it never
re-embeds or re-runs retrieval.  Queries whose ground-truth CWE is ``UNKNOWN``
are excluded from the ``cwe`` axis (and counted separately) but kept for the
``cve`` axis.

CLI:
    uv run python -m src.metrics.threshold_analysis <results.json>
        [--axis cwe|cve] [--grid-step 0.05]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

UNKNOWN_CWE = "UNKNOWN"
DEFAULT_GRID_STEP = 0.05
DEFAULT_MARKER_PERCENTILES = (50.0, 75.0, 90.0)
AXES = ("cwe", "cve")


# ── small numeric helpers (kept local; src/ must not import experiments/) ──


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    vals = sorted(values)
    pos = (len(vals) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _is_known_cwe(cwe: Any) -> bool:
    return bool(cwe) and str(cwe).strip().upper() != UNKNOWN_CWE


def default_grid(step: float = DEFAULT_GRID_STEP) -> list[float]:
    """Absolute similarity-score grid from 0.0 to 1.0 (inclusive)."""
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(round(1.0 / step))
    return [round(i * step, 4) for i in range(n + 1)]


# ── record normalization ────────────────────────────────────────────


def normalize_records(raw_queries: list[dict]) -> list[dict]:
    """Flatten each raw query into the flat fields threshold analysis needs.

    Robust to both ``raw_queries`` schemas found in the codebase: the
    ``self_retrieval`` variant (with ``query_variant`` and per-item ``variant``)
    and the leaner pre-computed-embedding variant.  Missing fields default
    gracefully.  A query with no retrieved results gets ``top1_score = None``
    so it is always treated as abstained.
    """
    records: list[dict] = []
    for q in raw_queries or []:
        retrieved = q.get("retrieved") or []
        query_cve = q.get("query_cve")
        query_cwe = q.get("query_cwe")
        variant = q.get("query_variant") or ""

        if retrieved:
            top1 = retrieved[0]
            top1_score = _safe_float(top1.get("score"), 0.0)
            last_score = _safe_float(retrieved[-1].get("score"), 0.0)
            pred_cwe = top1.get("cwe_id")
            pred_cve = top1.get("cve_id")
            has_results = True
        else:
            top1_score = None
            last_score = None
            pred_cwe = None
            pred_cve = None
            has_results = False

        cwe_known = _is_known_cwe(query_cwe)
        cwe_correct = bool(has_results and cwe_known and pred_cwe == query_cwe)
        cve_correct = bool(has_results and pred_cve == query_cve)

        # 1-based rank of the first retrieved item that would count as correct
        first_correct_rank_cwe: int | None = None
        first_correct_rank_cve: int | None = None
        for idx, r in enumerate(retrieved, start=1):
            rank = r.get("rank", idx)
            if first_correct_rank_cve is None and r.get("cve_id") == query_cve:
                first_correct_rank_cve = rank
            if (
                first_correct_rank_cwe is None
                and cwe_known
                and r.get("cwe_id") == query_cwe
            ):
                first_correct_rank_cwe = rank
            if first_correct_rank_cve is not None and (
                first_correct_rank_cwe is not None or not cwe_known
            ):
                break

        records.append(
            {
                "query_cve": query_cve,
                "query_cwe": query_cwe if cwe_known else UNKNOWN_CWE,
                "query_variant": variant,
                "has_results": has_results,
                "top1_score": top1_score,
                "last_score": last_score,
                "pred_cwe": pred_cwe,
                "pred_cve": pred_cve,
                "cwe_known": cwe_known,
                "cwe_correct": cwe_correct,
                "cve_correct": cve_correct,
                "first_correct_rank_cwe": first_correct_rank_cwe,
                "first_correct_rank_cve": first_correct_rank_cve,
            }
        )
    return records


def _records_for_axis(records: list[dict], axis: str) -> list[dict]:
    """CWE axis excludes queries with an unknown ground-truth CWE."""
    if axis == "cwe":
        return [r for r in records if r["cwe_known"]]
    return list(records)


# ── threshold sweep ─────────────────────────────────────────────────


def classify_at_threshold(records: list[dict], axis: str, t: float) -> dict[str, int]:
    """Confusion counts for *axis-filtered* records at threshold ``t``."""
    correct_key = "cwe_correct" if axis == "cwe" else "cve_correct"
    tp = fp = fn = tn = 0
    for r in records:
        score = r["top1_score"]
        confident = score is not None and score >= t
        correct = r[correct_key]
        if confident and correct:
            tp += 1
        elif confident and not correct:
            fp += 1
        elif (not confident) and correct:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    n = tp + fp + fn + tn
    predicted = tp + fp
    sel_acc = tp / predicted if predicted else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * sel_acc * recall / (sel_acc + recall) if (sel_acc + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": n,
        "predicted": predicted,
        "abstained": fn + tn,
        "coverage": predicted / n if n else 0.0,
        "selective_accuracy": sel_acc,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / n if n else 0.0,
        "risk": (1.0 - sel_acc) if predicted else 0.0,
        "abstention_rate": (n - predicted) / n if n else 0.0,
    }


def _round_row(row: dict[str, Any], ndigits: int = 6) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        out[k] = round(v, ndigits) if isinstance(v, float) else v
    return out


def sweep_thresholds(records: list[dict], axis: str, grid: list[float]) -> list[dict]:
    """Full sweep: one metrics row per threshold in ``grid`` for ``axis``."""
    axis_records = _records_for_axis(records, axis)
    rows = []
    for t in grid:
        counts = classify_at_threshold(axis_records, axis, t)
        row = {"threshold": round(t, 4), **_metrics_from_counts(counts)}
        rows.append(_round_row(row))
    return rows


def f1_argmax(sweep_rows: list[dict]) -> float | None:
    """Threshold that maximizes F1 (first-wins on ties). Informational only."""
    best = None
    for row in sweep_rows:
        if best is None or row["f1"] > best["f1"]:
            best = row
    return best["threshold"] if best else None


# ── score distribution / reference markers ──────────────────────────


def score_stats(records: list[dict]) -> dict[str, Any]:
    scores = [r["top1_score"] for r in records if r["top1_score"] is not None]
    stats = _summarize(scores)
    stats["n_no_results"] = sum(1 for r in records if not r["has_results"])
    return stats


def percentile_markers(
    records: list[dict], percentiles: tuple[float, ...] = DEFAULT_MARKER_PERCENTILES
) -> dict[str, float | None]:
    scores = [r["top1_score"] for r in records if r["top1_score"] is not None]
    return {f"p{int(p)}": _percentile(scores, p) for p in percentiles}


def random_baseline(records: list[dict]) -> float | None:
    """'Better than random' floor: mean of the weakest (last-rank) score.

    Mirrors ``src/evaluate/confidence_eval.compute_random_baseline``.
    """
    tail = [r["last_score"] for r in records if r["last_score"] is not None]
    return float(mean(tail)) if tail else None


# ── abstention error analysis at a single threshold ─────────────────


def _bump(bucket: dict, key: str, field: str) -> None:
    slot = bucket.setdefault(key, {})
    slot[field] = slot.get(field, 0) + 1


def error_analysis_at(records: list[dict], axis: str, t: float) -> dict[str, Any]:
    """Two-group ("has class" vs "no class") error distribution at ``t``."""
    axis_records = _records_for_axis(records, axis)
    correct_key = "cwe_correct" if axis == "cwe" else "cve_correct"
    rank_key = "first_correct_rank_cwe" if axis == "cwe" else "first_correct_rank_cve"

    has_by_cwe: dict[str, dict] = {}
    has_by_variant: dict[str, dict] = {}
    no_by_cwe: dict[str, dict] = {}
    no_by_variant: dict[str, dict] = {}
    wrong_cwe_confusion: dict[str, int] = {}

    has_n = has_correct = has_wrong = 0
    no_n = no_fn = no_tn = 0
    soft_errors = 0  # cve axis: wrong CVE but right CWE
    score_correct: list[float] = []
    score_wrong: list[float] = []
    fn_deficits: list[float] = []
    tn_recoverable = 0
    tn_rank_hist: dict[str, int] = {}

    for r in axis_records:
        score = r["top1_score"]
        confident = score is not None and score >= t
        correct = r[correct_key]
        cwe_bucket = r["query_cwe"]
        variant = r["query_variant"] or "(none)"

        if confident:
            has_n += 1
            if correct:
                has_correct += 1
                _bump(has_by_cwe, cwe_bucket, "correct")
                _bump(has_by_variant, variant, "correct")
                if score is not None:
                    score_correct.append(score)
            else:
                has_wrong += 1
                _bump(has_by_cwe, cwe_bucket, "wrong")
                _bump(has_by_variant, variant, "wrong")
                if score is not None:
                    score_wrong.append(score)
                if axis == "cwe" and r["pred_cwe"] is not None:
                    key = f"{cwe_bucket} → {r['pred_cwe']}"
                    wrong_cwe_confusion[key] = wrong_cwe_confusion.get(key, 0) + 1
                if axis == "cve" and r["cwe_correct"]:
                    soft_errors += 1
        else:
            no_n += 1
            if correct:
                no_fn += 1
                _bump(no_by_cwe, cwe_bucket, "fn")
                _bump(no_by_variant, variant, "fn")
                if score is not None:
                    fn_deficits.append(t - score)
            else:
                no_tn += 1
                _bump(no_by_cwe, cwe_bucket, "tn")
                _bump(no_by_variant, variant, "tn")
                rank = r[rank_key]
                if rank is not None:
                    tn_recoverable += 1
                    key = str(int(rank))
                    tn_rank_hist[key] = tn_rank_hist.get(key, 0) + 1

    has_class = {
        "n": has_n,
        "correct": has_correct,
        "wrong": has_wrong,
        "accuracy": (has_correct / has_n) if has_n else 0.0,
        "by_true_cwe": has_by_cwe,
        "by_variant": has_by_variant,
        "score_correct": _summarize(score_correct),
        "score_wrong": _summarize(score_wrong),
    }
    if axis == "cwe":
        has_class["wrong_cwe_confusion"] = dict(
            sorted(wrong_cwe_confusion.items(), key=lambda kv: kv[1], reverse=True)
        )
    else:
        has_class["soft_errors_wrong_cve_right_cwe"] = soft_errors

    no_class = {
        "n": no_n,
        "fn": no_fn,
        "tn": no_tn,
        "by_true_cwe": no_by_cwe,
        "by_variant": no_by_variant,
        "fn_score_deficit_stats": _summarize(fn_deficits),
        "tn_recoverable_in_topk": tn_recoverable,
        "tn_recoverable_rank_hist": dict(
            sorted(tn_rank_hist.items(), key=lambda kv: int(kv[0]))
        ),
    }

    return {
        "threshold": round(t, 6),
        "n": len(axis_records),
        "has_class": has_class,
        "no_class": no_class,
    }


# ── per-cell + full-run assembly ────────────────────────────────────


def analyze_cell(
    cell: dict,
    grid: list[float] | None = None,
    marker_percentiles: tuple[float, ...] = DEFAULT_MARKER_PERCENTILES,
    source_key: str = "cve_retrieval",
) -> dict[str, Any]:
    """Threshold sweep + error analysis for one results.json cell."""
    grid = grid or default_grid()
    raw_queries = (cell.get(source_key) or cell.get("self_retrieval") or {}).get(
        "raw_queries", []
    ) or []
    records = normalize_records(raw_queries)

    markers = percentile_markers(records, marker_percentiles)
    markers["random"] = random_baseline(records)

    sweep = {axis: sweep_thresholds(records, axis, grid) for axis in AXES}
    for axis in AXES:
        markers[f"f1max_{axis}"] = f1_argmax(sweep[axis])

    # Error analysis at each percentile marker (keeps the JSON compact — the
    # full grid lives in `sweep`, detailed group breakdowns only at markers).
    error_analysis: dict[str, dict] = {axis: {} for axis in AXES}
    for label in (f"p{int(p)}" for p in marker_percentiles):
        t = markers.get(label)
        if t is None:
            continue
        for axis in AXES:
            error_analysis[axis][label] = error_analysis_at(records, axis, t)

    return {
        "embedder": cell.get("embedder"),
        "backend": cell.get("backend"),
        "graph_variant": cell.get("graph_variant"),
        "n_queries": len(records),
        "n_queries_cwe": sum(1 for r in records if r["cwe_known"]),
        "n_unknown_cwe": sum(1 for r in records if not r["cwe_known"]),
        "score_stats": score_stats(records),
        "markers": markers,
        "sweep": sweep,
        "error_analysis": error_analysis,
    }


def build_analysis(
    results: dict,
    grid: list[float] | None = None,
    marker_percentiles: tuple[float, ...] = DEFAULT_MARKER_PERCENTILES,
    source_key: str = "cve_retrieval",
) -> dict[str, Any]:
    """Build the full ``threshold_analysis`` structure from a results dict."""
    grid = grid or default_grid()
    cells = results.get("cells", []) or []
    cells_out = [
        analyze_cell(c, grid, marker_percentiles, source_key=source_key) for c in cells
    ]
    return {
        "run_id": results.get("run_id"),
        "axes": list(AXES),
        "grid": grid,
        "marker_percentiles": list(marker_percentiles),
        "global": {
            "n_cells": len(cells_out),
            "n_queries_total": sum(c["n_queries"] for c in cells_out),
        },
        "cells": cells_out,
    }


# ── CLI (mirrors src/evaluate/confidence_eval terminal UX) ──────────


def _fmt_pct(x: float | None) -> str:
    return "  -  " if x is None else f"{x * 100:6.2f}%"


def _print_sweep_table(cell: dict, axis: str) -> None:
    name = f"{cell['embedder']} · {cell['backend']} · {cell['graph_variant']}"
    print(f"\n{'=' * 92}")
    print(f"  THRESHOLD SWEEP — axis={axis.upper()}  |  {name}")
    n_axis = cell["n_queries_cwe"] if axis == "cwe" else cell["n_queries"]
    print(
        f"  queries={n_axis}   markers: "
        + "  ".join(
            f"{k}={_num(v)}" for k, v in cell["markers"].items() if v is not None
        )
    )
    print(f"{'=' * 92}")
    print(
        f"  {'thr':>5s}  {'cov':>7s}  {'sel.acc':>7s}  {'recall':>7s}  "
        f"{'f1':>7s}  {'acc':>7s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'TN':>4s}"
    )
    print(f"  {'-' * 86}")
    for row in cell["sweep"][axis]:
        print(
            f"  {row['threshold']:5.2f}  {_fmt_pct(row['coverage'])}  "
            f"{_fmt_pct(row['selective_accuracy'])}  {_fmt_pct(row['recall'])}  "
            f"{_fmt_pct(row['f1'])}  {_fmt_pct(row['accuracy'])}  "
            f"{row['tp']:4d}  {row['fp']:4d}  {row['fn']:4d}  {row['tn']:4d}"
        )
    print(f"{'=' * 92}")


def _num(x: float | None, d: int = 3) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Similarity-threshold study over a run's results.json."
    )
    parser.add_argument(
        "results_json", help="Path to experiments/output/<run>/results.json"
    )
    parser.add_argument(
        "--axis",
        choices=["cwe", "cve", "both"],
        default="both",
        help="Correctness axis to print (default: both).",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=DEFAULT_GRID_STEP,
        help="Absolute threshold grid step (default: 0.05).",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to also write the full threshold_analysis.json.",
    )
    args = parser.parse_args()

    results = json.loads(Path(args.results_json).read_text())
    grid = default_grid(args.grid_step)
    analysis = build_analysis(results, grid=grid)

    axes = AXES if args.axis == "both" else (args.axis,)
    for cell in analysis["cells"]:
        for axis in axes:
            _print_sweep_table(cell, axis)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(analysis, indent=2))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
