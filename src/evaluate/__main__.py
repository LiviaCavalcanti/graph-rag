#!/usr/bin/env python3
"""
Run the full evaluation pipeline and generate the HTML dashboard.

Chains:
  1. evaluate_patches  → evaluation.jsonl + evaluation_summary.json
  2. retrieval_eval    → retrieval_eval.jsonl + retrieval_eval_summary.json
  3. confidence_eval   → confidence_eval_*.json/jsonl
  4. dashboard         → evaluation_dashboard.html

Usage:
    python -m src.evaluate <results.jsonl> [--config config.yaml]
    python -m src.evaluate <run_dir>/        # auto-finds results.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_all(
    results_path: Path,
    config_path: str = "config.yaml",
    base_dir: Path | None = None,
    top_k: int = 5,
    strip_comments: bool = False,
) -> Path:
    """Execute the full evaluation pipeline and return the dashboard path."""
    from src.evaluate.confidence_eval import run_confidence_eval
    from src.evaluate.dashboard import generate_dashboard
    from src.evaluate.evaluate_patches import main as _patches_main

    run_dir = results_path.parent
    base = base_dir or Path.cwd()

    # ── 1. Patch evaluation ──────────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  STEP 1/3 — Patch Evaluation")
    print(f"{'━'*60}")
    _run_patch_eval(results_path, base, strip_comments=strip_comments)

    # ── 2. Confidence evaluation ─────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  STEP 2/3 — Confidence Evaluation")
    print(f"{'━'*60}")
    retrieval_eval_path = run_dir / "retrieval_eval.jsonl"
    if retrieval_eval_path.exists():
        run_confidence_eval(retrieval_eval_path)
    else:
        # compute basic retrieval stats from the results.jsonl records
        _run_retrieval_summary(results_path)

    # ── 3. Dashboard generation ──────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  STEP 3/3 — Dashboard Generation")
    print(f"{'━'*60}")
    dashboard_path = generate_dashboard(run_dir)
    print(f"\n  Evaluation dashboard: {dashboard_path}")

    # ── Patch analysis dashboard (code triples + detailed scores) ─
    from experiments.dashboard_scripts.analyze_patches import analyze as _analyze_patches
    from experiments.dashboard_scripts.analyze_patches import _render_html as _render_patch_html

    eval_jsonl = run_dir / "evaluation.jsonl"
    if eval_jsonl.exists():
        analysis = _analyze_patches(results_path, eval_jsonl, base)
        patch_html = run_dir / "patch_analysis.html"
        patch_html.write_text(_render_patch_html(analysis))
        print(f"  Patch analysis:      {patch_html}")
    else:
        print("  WARNING: evaluation.jsonl not found, skipping patch analysis dashboard")

    return dashboard_path


def _run_retrieval_summary(results_path: Path) -> None:
    """Compute retrieval metrics from results.jsonl (compatible with dashboard)."""
    import json

    from src.metrics.retrieval_eval import evaluate_retrieval_from_records

    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    summary = evaluate_retrieval_from_records(records)

    # Attach split_info from run_meta.json if available
    meta_path = results_path.parent / "retrieval_meta.json"
    if not meta_path.exists():
        meta_path = results_path.parent / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        summary["split_info"] = meta.get("split_info", {})

    summary_path = results_path.parent / "retrieval_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Retrieval summary: {summary_path}")
    print(
        f"  Hit@1: {summary['hit_rate_at_1']:.1%}  |  "
        f"Hit@{summary['top_k']}: {summary['hit_rate_at_k']:.1%}  |  "
        f"MRR: {summary['mrr']:.3f}  |  "
        f"CWE@{summary['top_k']}: {summary['cwe_hit_rate_at_k']:.1%}"
    )


def _run_patch_eval(results_path: Path, base_dir: Path, *, strip_comments: bool = False) -> None:
    """Run patch evaluation programmatically (avoid argparse)."""
    import json

    from src.evaluate.evaluate_patches import aggregate, evaluate_one

    out_path = results_path.parent / "evaluation.jsonl"
    summary_path = out_path.with_name("evaluation_summary.json")

    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {results_path}")
    if strip_comments:
        print("  (stripping C/C++ comments before comparison)")

    evaluations = []
    for i, rec in enumerate(records):
        label = f"{rec.get('query_cve', '?')}/{rec.get('query_variant', '?')}"
        try:
            ev = evaluate_one(rec, base_dir, strip_comments=strip_comments)
            status = ev.get("eval_status")
            if status == "evaluated":
                bleu = ev["metrics_vs_function_body"]["bleu_4"]
                jaccard = ev["metrics_vs_function_body"]["token_jaccard"]
                print(
                    f"  [{i+1}/{len(records)}] {label}  BLEU-4={bleu:.4f}  Jaccard={jaccard:.4f}"
                )
            else:
                print(
                    f"  [{i+1}/{len(records)}] {label}  {status}: {ev.get('reason', '')}"
                )
        except Exception as e:
            ev = {
                "query_cve": rec.get("query_cve"),
                "query_variant": rec.get("query_variant"),
                "eval_status": "error",
                "reason": str(e),
            }
            print(f"  [{i+1}/{len(records)}] {label}  ERROR: {e}")
        evaluations.append(ev)

    with open(out_path, "w") as f:
        for ev in evaluations:
            f.write(json.dumps(ev, default=str) + "\n")
    print(f"Per-record evaluations: {out_path}")

    agg = aggregate(evaluations)
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"Summary: {summary_path}")

    print(
        f"  Evaluated: {agg['evaluated']}  |  "
        f"BLEU-4: {agg.get('avg_bleu_4', 'N/A')}  |  "
        f"Jaccard: {agg.get('avg_token_jaccard', 'N/A')}  |  "
        f"Exact: {agg.get('exact_matches', 0)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the full evaluation pipeline (patches + retrieval + confidence + dashboard)."
    )
    parser.add_argument(
        "path",
        help="Path to results.jsonl or to the run directory containing it",
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument(
        "--base-dir", default=None, help="Base dir for ground truth paths"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Retrieval top-k (default: 5)"
    )
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        results_path = path / "results.jsonl"
    else:
        results_path = path

    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    base_dir = Path(args.base_dir) if args.base_dir else None
    run_all(results_path, config_path=args.config, base_dir=base_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
