#!/usr/bin/env python3
"""
Evaluate generated patches against ground-truth files.

Reads a results.jsonl produced by batch inference, loads each ground-truth
file referenced in ``ground_truth_patch``, and compares it to the
``generated_patch`` field using multiple similarity metrics.

Usage:
    python -m src.evaluate.evaluate_patches <results.jsonl> [--out evaluation.jsonl]

Output: a JSONL file (one object per query) with detailed metrics, diff
positions, and identifiers for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.evaluate.preprocessing import extract_function_body
from src.metrics.similarity import (bertscore_pair, bleu_score,
                                    codebleu_weighted,
                                    compute_diff_details, exact_match,
                                    line_level_ratio, normalised_edit_distance,
                                    normalised_exact_match,
                                    sequence_matcher_ratio, token_jaccard,
                                    token_jaccard_multiset, tokenize)

# ── main evaluation ──────────────────────────────────────────────────


def _find_cve_dir(cve_id: str, base_dir: Path) -> Path | None:
    """Find the CVE directory, handling suffixed names like CVE-2024-53142_1."""
    exact = base_dir / "CVE-list" / cve_id
    if exact.is_dir():
        return exact
    cve_list = base_dir / "CVE-list"
    if cve_list.is_dir():
        candidates = sorted(
            d for d in cve_list.iterdir()
            if d.is_dir() and d.name.startswith(cve_id + "_")
        )
        if candidates:
            return candidates[0]
    return None


def strip_c_comments(code: str) -> str:
    """Remove C/C++ comments (block and line) from source code."""
    # Remove block comments /* ... */ (non-greedy, handles multiline)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # Remove line comments // ...
    code = re.sub(r"//[^\n]*", "", code)
    # Collapse multiple blank lines into one
    code = re.sub(r"\n\s*\n", "\n\n", code)
    return code


def evaluate_one(record: dict, base_dir: Path, strip_comments: bool = False) -> dict:
    """Evaluate a single result record. Returns an evaluation dict."""

    ident = {
        "query_cve": record.get("query_cve"),
        "query_cwe": record.get("query_cwe"),
        "query_variant": record.get("query_variant"),
        "example_cve": record.get("example_cve"),
        "example_variant": record.get("example_variant"),
        "status": record.get("status"),
        "elapsed_s": record.get("elapsed_s"),
    }

    generated = (record.get("generated_patch") or "").strip()
    gt_path_str = record.get("ground_truth_patch", "")

    if not generated:
        return {**ident, "eval_status": "skipped", "reason": "missing_patch"}

    # Try to load ground truth from the canonical file on disk first
    # (inline code in ground_truth_patch may be truncated)
    gt_full = None
    cve_id = record.get("query_cve", "")
    variant = record.get("query_variant", "")
    if cve_id and variant:
        cve_dir = _find_cve_dir(cve_id, base_dir)
        if cve_dir is not None:
            gt_file = cve_dir / "out_v2" / "code" / f"{variant}_fixed.c"
            if gt_file.exists():
                gt_full = gt_file.read_text(errors="replace")

    # Fall back to ground_truth_patch field (may be a path or inline code)
    if gt_full is None and gt_path_str:
        gt_candidate = base_dir / gt_path_str if len(gt_path_str) < 260 else None
        if gt_candidate is not None and gt_candidate.exists():
            gt_full = gt_candidate.read_text(errors="replace")
        elif "\n" in gt_path_str or len(gt_path_str) > 260:
            gt_full = gt_path_str

    if gt_full is None:
        return {
            **ident,
            "eval_status": "skipped",
            "reason": f"file_not_found: {cve_id}/{variant}",
        }

    # Extract just the function body from the ground-truth file
    # (files contain stubs + the actual function)
    gt_body = extract_function_body(gt_full).strip()

    # Also keep the full file for a secondary comparison
    gt_full_stripped = gt_full.strip()

    # ── optionally strip comments before comparison ──────────
    if strip_comments:
        generated = strip_c_comments(generated).strip()
        gt_body = strip_c_comments(gt_body).strip()
        gt_full_stripped = strip_c_comments(gt_full_stripped).strip()

    # ── compute metrics against extracted function body ──────────
    metrics_body = {
        "exact_match": exact_match(generated, gt_body),
        "normalised_exact_match": normalised_exact_match(generated, gt_body),
        "char_sequence_ratio": round(sequence_matcher_ratio(generated, gt_body), 4),
        "line_sequence_ratio": round(line_level_ratio(generated, gt_body), 4),
        "normalised_edit_distance": round(
            normalised_edit_distance(generated, gt_body), 4
        ),
        "token_jaccard": round(token_jaccard(generated, gt_body), 4),
        "token_jaccard_multiset": round(token_jaccard_multiset(generated, gt_body), 4),
        "bleu_1": round(bleu_score(generated, gt_body, max_n=1), 4),
        "bleu_2": round(bleu_score(generated, gt_body, max_n=2), 4),
        "bleu_4": round(bleu_score(generated, gt_body, max_n=4), 4),
        "codebleu_proxy": round(codebleu_weighted(generated, gt_body), 4),
        **bertscore_pair(generated, gt_body),
    }

    # ── compute metrics against full file (secondary) ────────────
    metrics_full = {
        "full_file_char_ratio": round(
            sequence_matcher_ratio(generated, gt_full_stripped), 4
        ),
        "full_file_token_jaccard": round(token_jaccard(generated, gt_full_stripped), 4),
        "full_file_bleu_4": round(bleu_score(generated, gt_full_stripped, max_n=4), 4),
    }

    # ── diff details ─────────────────────────────────────────────
    diff = compute_diff_details(generated, gt_body)

    # ── size info ────────────────────────────────────────────────
    gen_lines = generated.count("\n") + 1
    ref_lines = gt_body.count("\n") + 1
    gen_tokens = len(tokenize(generated))
    ref_tokens = len(tokenize(gt_body))

    size_info = {
        "generated_lines": gen_lines,
        "ground_truth_lines": ref_lines,
        "generated_tokens": gen_tokens,
        "ground_truth_tokens": ref_tokens,
        "line_count_diff": gen_lines - ref_lines,
        "token_count_diff": gen_tokens - ref_tokens,
    }

    return {
        **ident,
        "eval_status": "evaluated",
        "ground_truth_file": gt_path_str,
        "ground_truth_extracted_preview": gt_body[:300],
        "metrics_vs_function_body": metrics_body,
        "metrics_vs_full_file": metrics_full,
        "size_info": size_info,
        "diff_details": diff,
    }


def aggregate(results: list[dict]) -> dict:
    """Compute aggregate statistics across all evaluated records."""
    evaluated = [r for r in results if r.get("eval_status") == "evaluated"]
    if not evaluated:
        return {"total_records": len(results), "evaluated": 0, "skipped": len(results)}

    def _avg(key):
        vals = [
            r["metrics_vs_function_body"][key]
            for r in evaluated
            if key in r.get("metrics_vs_function_body", {})
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    n = len(evaluated)
    return {
        "total_records": len(results),
        "evaluated": n,
        "skipped": len(results) - n,
        "exact_matches": sum(
            1 for r in evaluated if r["metrics_vs_function_body"]["exact_match"]
        ),
        "normalised_exact_matches": sum(
            1
            for r in evaluated
            if r["metrics_vs_function_body"]["normalised_exact_match"]
        ),
        "avg_char_sequence_ratio": _avg("char_sequence_ratio"),
        "avg_line_sequence_ratio": _avg("line_sequence_ratio"),
        "avg_normalised_edit_distance": _avg("normalised_edit_distance"),
        "avg_token_jaccard": _avg("token_jaccard"),
        "avg_token_jaccard_multiset": _avg("token_jaccard_multiset"),
        "avg_bleu_1": _avg("bleu_1"),
        "avg_bleu_2": _avg("bleu_2"),
        "avg_bleu_4": _avg("bleu_4"),
        "avg_codebleu_proxy": _avg("codebleu_proxy"),
        "avg_bertscore_precision": _avg("bertscore_precision"),
        "avg_bertscore_recall": _avg("bertscore_recall"),
        "avg_bertscore_f1": _avg("bertscore_f1"),
        "by_cwe": _aggregate_by_field(evaluated, "query_cwe"),
        "by_variant": _aggregate_by_field(evaluated, "query_variant"),
    }


def _aggregate_by_field(evaluated: list[dict], field: str) -> dict:
    groups: dict[str, list] = {}
    for r in evaluated:
        key = r.get(field, "unknown")
        groups.setdefault(key, []).append(r)

    out = {}
    for key, recs in sorted(groups.items()):
        n = len(recs)
        out[key] = {
            "count": n,
            "avg_bleu_4": round(
                sum(r["metrics_vs_function_body"]["bleu_4"] for r in recs) / n, 4
            ),
            "avg_token_jaccard": round(
                sum(r["metrics_vs_function_body"]["token_jaccard"] for r in recs) / n, 4
            ),
            "avg_char_ratio": round(
                sum(r["metrics_vs_function_body"]["char_sequence_ratio"] for r in recs)
                / n,
                4,
            ),
            "avg_codebleu_proxy": round(
                sum(r["metrics_vs_function_body"]["codebleu_proxy"] for r in recs) / n,
                4,
            ),
            "avg_bertscore_f1": round(
                sum(r["metrics_vs_function_body"].get("bertscore_f1", 0) for r in recs) / n,
                4,
            ),
            "exact_matches": sum(
                1 for r in recs if r["metrics_vs_function_body"]["exact_match"]
            ),
        }
    return out


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated patches vs ground truth."
    )
    parser.add_argument(
        "results_jsonl", help="Path to results.jsonl from batch inference"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output evaluation JSONL path (default: <input_dir>/evaluation.jsonl)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for resolving ground_truth_patch paths (default: repo root)",
    )
    parser.add_argument(
        "--strip-comments",
        action="store_true",
        default=False,
        help="Remove C/C++ comments from generated and ground-truth code before comparison",
    )
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()
    out_path = Path(args.out) if args.out else results_path.parent / "evaluation.jsonl"
    summary_path = out_path.with_name("evaluation_summary.json")

    # load records
    records = []
    with open(results_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping line {line_num}: {e}")

    print(f"Loaded {len(records)} records from {results_path}")
    print(f"Base directory: {base_dir}")

    # evaluate
    evaluations = []
    for i, rec in enumerate(records):
        label = f"{rec.get('query_cve', '?')}/{rec.get('query_variant', '?')}"
        try:
            ev = evaluate_one(rec, base_dir, strip_comments=args.strip_comments)
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

    # write per-record evaluations
    with open(out_path, "w") as f:
        for ev in evaluations:
            f.write(json.dumps(ev, default=str) + "\n")
    print(f"\nPer-record evaluations written to: {out_path}")

    # write aggregate summary
    agg = aggregate(evaluations)
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"Aggregate summary written to:      {summary_path}")

    # print summary
    print(f"\n{'═'*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total records:     {agg['total_records']}")
    print(f"  Evaluated:         {agg['evaluated']}")
    print(f"  Skipped:           {agg['skipped']}")
    print(f"  Exact matches:     {agg.get('exact_matches', 0)}")
    print(f"  Norm exact match:  {agg.get('normalised_exact_matches', 0)}")
    print(f"  Avg BLEU-4:        {agg.get('avg_bleu_4', 'N/A')}")
    print(f"  Avg Token Jaccard: {agg.get('avg_token_jaccard', 'N/A')}")
    print(f"  Avg Char Ratio:    {agg.get('avg_char_sequence_ratio', 'N/A')}")
    print(f"  Avg CodeBLEU*:     {agg.get('avg_codebleu_proxy', 'N/A')}")
    print(f"  Avg BERTScore F1:  {agg.get('avg_bertscore_f1', 'N/A')}")
    print(f"  Avg Edit Dist:     {agg.get('avg_normalised_edit_distance', 'N/A')}")
    print(f"{'═'*60}")

    if "by_cwe" in agg:
        print(f"\n  By CWE type:")
        for cwe, stats in agg["by_cwe"].items():
            print(
                f"    {cwe:40s}  n={stats['count']:3d}  BLEU-4={stats['avg_bleu_4']:.4f}  Jaccard={stats['avg_token_jaccard']:.4f}"
            )

    if "by_variant" in agg:
        print(f"\n  By variant:")
        for var, stats in agg["by_variant"].items():
            print(
                f"    {var:40s}  n={stats['count']:3d}  BLEU-4={stats['avg_bleu_4']:.4f}  Jaccard={stats['avg_token_jaccard']:.4f}"
            )


if __name__ == "__main__":
    main()
