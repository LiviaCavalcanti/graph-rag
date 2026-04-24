#!/usr/bin/env python3
"""
Post-process batch inference results.

Reads the JSONL output from batch_inference.py and produces a summary
JSON file with aggregate metrics identical to agent_experiment.py output.

Can be re-run at any time (on partial or complete results).

Usage:
    python -m experiments.postprocess experiments/output/<run_dir>
    python -m experiments.postprocess experiments/output/<run_dir> --format table
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS_FILENAME = "results.jsonl"
META_FILENAME = "run_meta.json"
SUMMARY_FILENAME = "results.json"


def load_results(run_dir: Path) -> tuple[dict, list[dict]]:
    """Load run metadata and per-query results from a batch run directory."""
    meta_path = run_dir / META_FILENAME
    jsonl_path = run_dir / RESULTS_FILENAME

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found")
        sys.exit(1)

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    records = []
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: skipping malformed line {lineno}")

    return meta, records


def compute_summary(meta: dict, records: list[dict]) -> dict:
    """Compute aggregate metrics from per-query results."""
    total = len(records)
    evaluated = [r for r in records if r.get("status") in ("success", "parse_error")]
    successful = [r for r in records if r.get("status") == "success"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    errors = [r for r in records if r.get("status") == "error"]
    parse_errors = [r for r in records if r.get("status") == "parse_error"]

    n_evaluated = len(evaluated)
    n_success = len(successful)

    cve_hits = sum(1 for r in evaluated if r.get("cve_match"))
    cwe_hits = sum(1 for r in evaluated if r.get("cwe_match"))

    similarities = [r["similarity"] for r in successful if r.get("similarity") is not None]
    exact_matches = sum(1 for r in successful if r.get("exact_match"))

    elapsed_times = [r["elapsed_s"] for r in records if r.get("elapsed_s") is not None]

    summary = {
        "run_id": meta.get("run_id", "unknown"),
        "mode": meta.get("mode", "unknown"),
        "model": meta.get("model", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split_info": meta.get("split_info", {}),
        "total_queries": total,
        "evaluated": n_evaluated,
        "successful_parses": n_success,
        "skipped": len(skipped),
        "errors": len(errors),
        "completeness": total / meta.get("total_queries", total) if meta.get("total_queries") else 1.0,
        "retrieval": {
            "cve_recall": cve_hits / n_evaluated if n_evaluated > 0 else 0.0,
            "cwe_recall": cwe_hits / n_evaluated if n_evaluated > 0 else 0.0,
            "cve_hits": cve_hits,
            "cwe_hits": cwe_hits,
        },
        "patching": {
            "mean_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "median_similarity": float(np.median(similarities)) if similarities else 0.0,
            "std_similarity": float(np.std(similarities)) if similarities else 0.0,
            "min_similarity": float(np.min(similarities)) if similarities else 0.0,
            "max_similarity": float(np.max(similarities)) if similarities else 0.0,
            "exact_matches": exact_matches,
            "exact_match_rate": exact_matches / n_success if n_success > 0 else 0.0,
            "parse_errors": len(parse_errors),
        },
        "timing": {
            "mean_elapsed_s": float(np.mean(elapsed_times)) if elapsed_times else 0.0,
            "median_elapsed_s": float(np.median(elapsed_times)) if elapsed_times else 0.0,
            "total_elapsed_s": float(np.sum(elapsed_times)) if elapsed_times else 0.0,
        },
    }

    # ── CWE-level breakdown ──────────────────────────────────────────
    cwe_stats = defaultdict(lambda: {
        "total": 0, "cve_hits": 0, "cwe_hits": 0,
        "similarities": [], "exact_matches": 0,
    })
    for r in evaluated:
        cwe = r.get("query_cwe", "Unknown")
        cwe_stats[cwe]["total"] += 1
        if r.get("cve_match"):
            cwe_stats[cwe]["cve_hits"] += 1
        if r.get("cwe_match"):
            cwe_stats[cwe]["cwe_hits"] += 1
        if r.get("similarity", 0) > 0:
            cwe_stats[cwe]["similarities"].append(r["similarity"])
        if r.get("exact_match"):
            cwe_stats[cwe]["exact_matches"] += 1

    cwe_breakdown = {}
    for cwe, stats in sorted(cwe_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        cwe_breakdown[cwe] = {
            "total": stats["total"],
            "cve_recall": stats["cve_hits"] / stats["total"] if stats["total"] > 0 else 0,
            "cwe_recall": stats["cwe_hits"] / stats["total"] if stats["total"] > 0 else 0,
            "mean_similarity": float(np.mean(stats["similarities"])) if stats["similarities"] else 0,
            "exact_matches": stats["exact_matches"],
        }
    summary["cwe_breakdown"] = cwe_breakdown

    # ── miss analysis ────────────────────────────────────────────────
    misses = []
    for r in records:
        if r.get("status") == "skipped":
            misses.append({
                "type": "skip",
                "query_cve": r["query_cve"],
                "query_cwe": r.get("query_cwe", ""),
                "reason": r.get("reason", "unknown"),
            })
        elif r.get("status") == "error":
            misses.append({
                "type": "error",
                "query_cve": r["query_cve"],
                "query_cwe": r.get("query_cwe", ""),
                "error": r.get("error", ""),
            })
        elif r.get("status") == "parse_error":
            misses.append({
                "type": "parse_error",
                "query_cve": r["query_cve"],
                "query_cwe": r.get("query_cwe", ""),
                "example_cve": r.get("example_cve", ""),
            })
        elif r.get("status") == "success" and r.get("similarity", 0) < 0.5:
            misses.append({
                "type": "low_similarity",
                "query_cve": r["query_cve"],
                "query_cwe": r.get("query_cwe", ""),
                "example_cve": r.get("example_cve", ""),
                "cve_match": r.get("cve_match", False),
                "cwe_match": r.get("cwe_match", False),
                "similarity": r.get("similarity", 0),
            })
    summary["misses"] = misses

    # ── status distribution ──────────────────────────────────────────
    status_counts = defaultdict(int)
    for r in records:
        status_counts[r.get("status", "unknown")] += 1
    summary["status_distribution"] = dict(status_counts)

    # ── error breakdown ──────────────────────────────────────────────
    if errors:
        error_types = defaultdict(int)
        for r in errors:
            err = r.get("error", "unknown")
            # group by first line / exception type
            first_line = err.split("\n")[0][:120]
            error_types[first_line] += 1
        summary["error_breakdown"] = dict(error_types)

    return summary


def print_summary(summary: dict):
    """Print a human-readable summary to stdout."""
    mode = summary.get("mode", "?").upper()
    model = summary.get("model", "?")
    completeness = summary.get("completeness", 1.0)

    print(f"\n{'='*60}")
    print(f"  Batch Inference Summary — {mode}")
    print(f"{'='*60}")
    print(f"  Model:            {model}")
    print(f"  Completeness:     {completeness:.0%}")
    print(f"  Total results:    {summary['total_queries']}")
    print(f"  Evaluated:        {summary['evaluated']}")
    print(f"  Successful:       {summary['successful_parses']}")
    print(f"  Skipped:          {summary['skipped']}")
    print(f"  Errors:           {summary['errors']}")
    print()

    ret = summary.get("retrieval", {})
    print(f"  CVE recall:       {ret.get('cve_recall', 0):.3f}")
    print(f"  CWE recall:       {ret.get('cwe_recall', 0):.3f}")
    print()

    pat = summary.get("patching", {})
    print(f"  Mean similarity:  {pat.get('mean_similarity', 0):.3f}")
    print(f"  Median similarity:{pat.get('median_similarity', 0):.3f}")
    print(f"  Std similarity:   {pat.get('std_similarity', 0):.3f}")
    print(f"  Min/Max:          {pat.get('min_similarity', 0):.3f} / {pat.get('max_similarity', 0):.3f}")
    print(f"  Exact matches:    {pat.get('exact_matches', 0)}/{summary['successful_parses']} "
          f"({pat.get('exact_match_rate', 0):.1%})")
    print(f"  Parse errors:     {pat.get('parse_errors', 0)}")
    print()

    timing = summary.get("timing", {})
    print(f"  Mean latency:     {timing.get('mean_elapsed_s', 0):.1f}s")
    print(f"  Total time:       {timing.get('total_elapsed_s', 0):.0f}s")
    print()

    # status distribution
    status_dist = summary.get("status_distribution", {})
    if status_dist:
        print(f"  Status breakdown:")
        for status, count in sorted(status_dist.items(), key=lambda x: -x[1]):
            print(f"    {status:20s} {count}")
        print()

    # CWE breakdown
    cwe_breakdown = summary.get("cwe_breakdown", {})
    if cwe_breakdown:
        print(f"  CWE Breakdown:")
        for cwe, s in cwe_breakdown.items():
            print(f"    {cwe:30s}  n={s['total']:3d}  CVE_R={s['cve_recall']:.2f}  "
                  f"CWE_R={s['cwe_recall']:.2f}  sim={s['mean_similarity']:.3f}  "
                  f"exact={s.get('exact_matches', 0)}")
        print()

    # error breakdown
    error_breakdown = summary.get("error_breakdown", {})
    if error_breakdown:
        print(f"  Error types:")
        for err, count in sorted(error_breakdown.items(), key=lambda x: -x[1]):
            print(f"    [{count}x] {err}")
        print()

    # top misses
    misses = summary.get("misses", [])
    if misses:
        print(f"  Misses ({len(misses)}):")
        for m in misses[:20]:
            line = f"    [{m['type']}] {m['query_cve']} ({m.get('query_cwe', '?')})"
            if m.get("similarity") is not None:
                line += f" — sim={m['similarity']}"
            if m.get("reason"):
                line += f" — {m['reason']}"
            print(line)
        if len(misses) > 20:
            print(f"    ... and {len(misses) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process batch inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", type=str,
                        help="Path to the batch run directory")
    parser.add_argument("--format", choices=["json", "table"], default="table",
                        help="Output format (default: table)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results.json (just print)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    meta, records = load_results(run_dir)

    if not records:
        print("No results found.")
        sys.exit(1)

    summary = compute_summary(meta, records)

    if args.format == "table":
        print_summary(summary)
    else:
        print(json.dumps(summary, indent=2, default=str))

    if not args.no_save:
        out_path = run_dir / SUMMARY_FILENAME
        output = {"summary": summary, "per_query": records}
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
