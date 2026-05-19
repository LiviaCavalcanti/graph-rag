#!/usr/bin/env python3
"""
Full pipeline: retrieval → miss analysis → crossing analysis → dashboard.

Usage:
    python -m experiments.run_full_pipeline [--config config.yaml] [--loo]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import load_config
from src.data import load_pairs


def main():
    parser = argparse.ArgumentParser(description="Run full retrieval + analysis pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--loo", action="store_true", help="Run leave-one-out (slow)")
    parser.add_argument("--split", action="store_true", help="Force split mode on")
    parser.add_argument("--no-split", action="store_true", help="Force split mode off")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("experiment", {}).setdefault("split", {})
    if args.split:
        cfg["experiment"]["split"]["enabled"] = True
    if args.no_split:
        cfg["experiment"]["split"]["enabled"] = False

    pairs = load_pairs(cfg)
    print(f"Loaded {len(pairs)} pairs")

    # ── Step 1: Retrieval experiment ─────────────────────────────────
    print(f"\n{'='*50}\n  Step 1: Retrieval experiment\n{'='*50}")
    from experiments.exp.retrieval_experiment import RetrievalGridExperiment

    output = RetrievalGridExperiment(
        run_leave_one_out=args.loo,
        preloaded_pairs=pairs,
    ).run(cfg)
    run_dir = output.run_dir
    results_path = run_dir / "results.json"

    # ── Step 2: Miss analysis ────────────────────────────────────────
    print(f"\n{'='*50}\n  Step 2: Miss analysis\n{'='*50}")
    try:
        from experiments.dashboard_scripts.analyze_misses import analyze_results
        analyze_results(
            results_path=results_path,
            out_json=run_dir / "miss_analysis.json",
            out_html=run_dir / "miss_dashboard.html",
            uncertainty_quantile=25.0,
            uncertainty_prob_floor=0.12,
            uncertainty_margin_floor=0.005,
            temperature=1.0,
            max_examples=20,
        )
    except Exception as e:
        print(f"  Warning: miss analysis failed: {e}")

    # ── Step 3: Crossing analysis ────────────────────────────────────
    print(f"\n{'='*50}\n  Step 3: Crossing analysis\n{'='*50}")
    try:
        from experiments.put_aside.verify_crossing import analyze_crossing
        analyze_crossing(str(results_path))
    except Exception as e:
        print(f"  Warning: crossing analysis failed: {e}")

    # ── Step 4: Dashboard ────────────────────────────────────────────
    print(f"\n{'='*50}\n  Step 4: Dashboard\n{'='*50}")
    try:
        from experiments.dashboard_scripts.dashboard import generate_html_dashboard
        generate_html_dashboard(str(run_dir))
    except Exception as e:
        print(f"  Warning: dashboard generation failed: {e}")

    print(f"\nFull pipeline complete → {run_dir}")


if __name__ == "__main__":
    main()
