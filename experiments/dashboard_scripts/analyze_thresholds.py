#!/usr/bin/env python3
"""
Derive ``threshold_analysis.json`` from an experiment run's ``results.json``.

This is a thin, post-hoc wrapper around :mod:`src.metrics.threshold_analysis`.
It never re-embeds or re-runs retrieval — it only reads the per-query records
already stored in ``results.json`` (``cells[i].cve_retrieval.raw_queries``).

The rendered HTML lives in the unified dashboard's "Threshold Study" tab
(:func:`experiments.dashboard_scripts.dashboard._tab_threshold`); this wrapper
therefore produces JSON only, mirroring the ``verify_crossing`` →
``crossing_analysis.json`` flow.

Usage:
    uv run python -m experiments.dashboard_scripts.analyze_thresholds \\
        --results experiments/output/<run_id>/results.json [--grid-step 0.05]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Allow direct execution: put the repo root on sys.path so that
    # ``import src.metrics...`` resolves.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.metrics.threshold_analysis import (
    DEFAULT_GRID_STEP,
    DEFAULT_MARKER_PERCENTILES,
    build_analysis,
    default_grid,
)


def analyze_results(
    results_path: Path,
    out_json: Path | None = None,
    grid: list[float] | None = None,
    marker_percentiles: tuple[float, ...] = DEFAULT_MARKER_PERCENTILES,
    regenerate_unified: bool = False,
) -> dict[str, Any]:
    """Compute and persist the threshold analysis for one run.

    Args:
        results_path: Path to ``experiments/output/<run>/results.json``.
        out_json: Where to write ``threshold_analysis.json``
            (default: alongside ``results.json``).
        grid: Absolute threshold grid (default: 0.0..1.0 step 0.05).
        marker_percentiles: Score percentiles used as reference markers and as
            the thresholds at which the detailed error analysis is computed.
        regenerate_unified: When True, rebuild the unified ``dashboard.html``
            afterward.  Kept False when called from the dashboard itself to
            avoid infinite recursion.
    """
    results_path = Path(results_path)
    if out_json is None:
        out_json = _default_output_path(results_path)

    results = json.loads(results_path.read_text())
    analysis = build_analysis(
        results, grid=grid, marker_percentiles=marker_percentiles
    )
    analysis["source_results"] = str(results_path)

    out_json.write_text(json.dumps(analysis, indent=2))

    if regenerate_unified:
        try:
            from experiments.dashboard_scripts.dashboard import generate_html_dashboard

            run_dir = out_json.parent
            if (run_dir / "results.json").exists():
                generate_html_dashboard(run_dir)
        except Exception:
            pass  # non-fatal; unified dashboard is best-effort

    return analysis


def _default_output_path(results_path: Path) -> Path:
    if results_path.name == "results.json":
        return results_path.parent / "threshold_analysis.json"
    return results_path.with_suffix(".threshold_analysis.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive threshold_analysis.json from a run's results.json."
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to experiments/output/<run_id>/results.json",
    )
    parser.add_argument(
        "--out-json", default=None,
        help="Output JSON path (default: run_dir/threshold_analysis.json)",
    )
    parser.add_argument(
        "--grid-step", type=float, default=DEFAULT_GRID_STEP,
        help="Absolute threshold grid step (default: 0.05)",
    )
    parser.add_argument(
        "--regenerate-unified", action="store_true",
        help="Rebuild the unified dashboard.html after writing the JSON.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    out_json = Path(args.out_json) if args.out_json else None
    analysis = analyze_results(
        results_path=results_path,
        out_json=out_json,
        grid=default_grid(args.grid_step),
        regenerate_unified=args.regenerate_unified,
    )
    n_cells = analysis["global"]["n_cells"]
    n_queries = analysis["global"]["n_queries_total"]
    dest = out_json or _default_output_path(results_path)
    print(f"Threshold analysis written → {dest}  ({n_cells} cells, {n_queries} queries)")


if __name__ == "__main__":
    main()
