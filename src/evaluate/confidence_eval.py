#!/usr/bin/env python3
"""
Evaluate retrieval confidence at different score thresholds.

Reads a retrieval_eval.jsonl and determines, for each threshold, whether
the retriever "confidently" identifies the query as a known vulnerability.

A prediction is considered correct if the top-1 retrieved CVE matches the
query CVE.  Confidence = top-1 score >= threshold.

Usage:
    python -m src.evaluate.confidence_eval <retrieval_eval.jsonl> [--thresholds 0.1 0.15 0.2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── loading ──────────────────────────────────────────────────────────

def _load_entries(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ── per-entry annotation ─────────────────────────────────────────────

def annotate_entries(entries: list[dict]) -> list[dict]:
    """Add derived fields to each entry for downstream analysis."""
    annotated = []
    for e in entries:
        if e.get("status") != "evaluated" or not e.get("retrieved"):
            annotated.append({**e, "top1_score": 0.0, "top1_correct": False})
            continue

        top1 = e["retrieved"][0]
        top1_score = top1.get("score", 0.0)
        top1_correct = top1.get("cve_id") == e.get("query_cve")

        # score gap: how much top-1 stands out above top-2
        top2_score = e["retrieved"][1]["score"] if len(e["retrieved"]) > 1 else 0.0
        score_gap = top1_score - top2_score

        annotated.append({
            **e,
            "top1_score": top1_score,
            "top1_correct": top1_correct,
            "top2_score": top2_score,
            "score_gap": round(score_gap, 6),
        })
    return annotated


# ── threshold evaluation ─────────────────────────────────────────────

def evaluate_threshold(entries: list[dict], threshold: float) -> dict:
    """Classify entries at a given threshold and compute metrics.

    Confident = top1_score >= threshold.
    Correct   = top1 CVE matches query CVE.
    """
    tp = fp = fn = tn = 0
    per_entry = []

    for e in entries:
        score = e.get("top1_score", 0.0)
        correct = e.get("top1_correct", False)
        confident = score >= threshold

        if confident and correct:
            label = "TP"
            tp += 1
        elif confident and not correct:
            label = "FP"
            fp += 1
        elif not confident and correct:
            label = "FN"
            fn += 1
        else:
            label = "TN"
            tn += 1

        per_entry.append({
            "query_cve": e.get("query_cve"),
            "query_variant": e.get("query_variant"),
            "top1_score": score,
            "top1_correct": correct,
            "confident": confident,
            "label": label,
        })

    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "confident_total": tp + fp,
        "confident_correct": tp,
        "per_entry": per_entry,
    }


def compute_random_baseline(entries: list[dict]) -> float:
    """Compute a 'better than random' threshold.

    Uses the mean score of rank-5 results as the background noise level,
    then sets the threshold at noise + 1 stddev of top-1 scores.
    """
    tail_scores = []
    top1_scores = []
    for e in entries:
        if not e.get("retrieved"):
            continue
        top1_scores.append(e["retrieved"][-1]["score"])  # last rank = weakest
        tail_scores.append(e["retrieved"][-1]["score"])

    if not tail_scores:
        return 0.0
    noise = float(np.mean(tail_scores))
    return round(noise, 4)


# ── high confidence cases ────────────────────────────────────────────

def extract_high_confidence(entries: list[dict], threshold: float) -> list[dict]:
    """Return entries above threshold where top-1 matches the query CVE.

    A same-CVE match at high confidence means the retriever still recognises
    the vulnerability pattern — the patch was likely not successful.
    """
    confident = [
        {
            "query_cve": e.get("query_cve"),
            "query_cwe": e.get("query_cwe"),
            "query_variant": e.get("query_variant"),
            "top1_score": e["top1_score"],
            "top1_correct": e["top1_correct"],
            "score_gap": e.get("score_gap", 0.0),
            "top1_cve": e["retrieved"][0]["cve_id"] if e.get("retrieved") else None,
            "top1_func": e["retrieved"][0].get("func_name") if e.get("retrieved") else None,
            "top1_variant": e["retrieved"][0].get("variant") if e.get("retrieved") else None,
            "ground_truth_patch": e.get("ground_truth_patch", ""),
        }
        for e in entries
        if e.get("top1_score", 0.0) >= threshold and e.get("top1_correct", False)
    ]
    confident.sort(key=lambda x: x["top1_score"], reverse=True)
    return confident


# ── output ───────────────────────────────────────────────────────────

def _write_jsonl(data: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, default=str) + "\n")


def _print_table(results: list[dict]) -> None:
    print(f"\n{'═'*80}")
    print(f"  CONFIDENCE THRESHOLD EVALUATION")
    print(f"{'═'*80}")
    print(f"  {'Threshold':>10s}  {'Conf':>5s}  {'TP':>4s}  {'FP':>4s}  "
          f"{'FN':>4s}  {'TN':>4s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Acc':>6s}")
    print(f"  {'-'*74}")
    for r in results:
        t = r["threshold"]
        label = r.get("label", f"{t:.4f}")
        print(f"  {label:>10s}  {r['confident_total']:5d}  {r['tp']:4d}  {r['fp']:4d}  "
              f"{r['fn']:4d}  {r['tn']:4d}  {r['precision']:6.2%}  {r['recall']:6.2%}  "
              f"{r['f1']:6.2%}  {r['accuracy']:6.2%}")
    print(f"{'═'*80}")


# ── main orchestrator ────────────────────────────────────────────────

def run_confidence_eval(
    eval_path: Path,
    thresholds: list[float] | None = None,
    out_dir: Path | None = None,
) -> dict:
    # 1. load and annotate
    raw = _load_entries(eval_path)
    entries = annotate_entries(raw)
    evaluated = [e for e in entries if e.get("status") == "evaluated"]
    print(f"Loaded {len(evaluated)} evaluated entries from {eval_path}")

    if not evaluated:
        print("ERROR: no evaluated entries found. "
              "Make sure you pass a retrieval_eval.jsonl, not results.jsonl.")
        return {"error": "no_evaluated_entries"}

    # 2. compute score distribution
    top1_scores = np.array([e["top1_score"] for e in evaluated])
    print(f"Top-1 score stats: min={top1_scores.min():.4f}  max={top1_scores.max():.4f}  "
          f"mean={top1_scores.mean():.4f}  median={np.median(top1_scores):.4f}")

    # 3. build thresholds
    random_baseline = compute_random_baseline(raw)
    p50 = float(np.percentile(top1_scores, 50))
    p75 = float(np.percentile(top1_scores, 75))
    p90 = float(np.percentile(top1_scores, 90))

    default_thresholds = [
        random_baseline,
        round(p50, 4),
        round(p75, 4),
        round(p90, 4),
    ]
    if thresholds:
        all_thresholds = sorted(set(default_thresholds + thresholds))
    else:
        all_thresholds = sorted(set(default_thresholds))

    threshold_labels = {
        random_baseline: "random",
        round(p50, 4): "p50",
        round(p75, 4): "p75",
        round(p90, 4): "p90",
    }

    # 4. evaluate each threshold
    results = []
    for t in all_thresholds:
        r = evaluate_threshold(evaluated, t)
        r["label"] = threshold_labels.get(t, f"{t:.4f}")
        results.append(r)

    _print_table(results)

    # 5. extract high confidence cases (above p75)
    high_conf_threshold = round(p75, 4)
    high_conf = extract_high_confidence(evaluated, high_conf_threshold)
    print(f"\nHigh-confidence cases (score >= {high_conf_threshold}): {len(high_conf)}")
    for c in high_conf:
        match = "✓" if c["top1_correct"] else "✗"
        print(f"  {match} {c['query_cve']}/{c['query_variant']:30s}  "
              f"score={c['top1_score']:.4f}  gap={c['score_gap']:.4f}  "
              f"→ {c['top1_cve']}/{c['top1_func']}")

    # 6. write outputs
    out = out_dir or eval_path.parent
    summary_path = out / "confidence_eval_summary.json"
    per_entry_path = out / "confidence_eval_per_entry.jsonl"
    high_conf_path = out / "confidence_eval_high_conf.jsonl"

    summary = {
        "score_stats": {
            "min": round(float(top1_scores.min()), 4),
            "max": round(float(top1_scores.max()), 4),
            "mean": round(float(top1_scores.mean()), 4),
            "median": round(float(np.median(top1_scores)), 4),
            "p75": round(p75, 4),
            "p90": round(p90, 4),
            "random_baseline": random_baseline,
        },
        "thresholds": [
            {k: v for k, v in r.items() if k != "per_entry"}
            for r in results
        ],
        "high_confidence_threshold": high_conf_threshold,
        "high_confidence_count": len(high_conf),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary:          {summary_path}")

    # per-entry: use the p75 threshold classification
    p75_result = next(r for r in results if r["threshold"] == high_conf_threshold)
    _write_jsonl(p75_result["per_entry"], per_entry_path)
    print(f"Per-entry (p75):  {per_entry_path}")

    _write_jsonl(high_conf, high_conf_path)
    print(f"High confidence:  {high_conf_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval confidence at different score thresholds."
    )
    parser.add_argument("retrieval_eval_jsonl", help="Path to retrieval_eval.jsonl")
    parser.add_argument(
        "--thresholds", type=float, nargs="*", default=None,
        help="Additional thresholds to evaluate (e.g. 0.15 0.2 0.3)",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()

    path = Path(args.retrieval_eval_jsonl)
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else None
    run_confidence_eval(path, thresholds=args.thresholds, out_dir=out_dir)


if __name__ == "__main__":
    main()
