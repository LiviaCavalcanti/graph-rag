#!/usr/bin/env python3
"""
Analyze patch evaluation results: for each triple (input vulnerable code,
ground-truth fix, agent-generated patch), display retrieval metadata and
all evaluation scores side-by-side.

Inputs:
  - results.jsonl   (from batch inference)
  - evaluation.jsonl (from evaluate_patches)

Outputs:
  - patch_analysis.json   : structured per-record analysis + aggregates
  - patch_analysis.html   : human-friendly HTML dashboard

Usage:
    python -m experiments.dashboard_scripts.analyze_patches \
        --results experiments/output/<run>/results.jsonl \
        --evaluation experiments/output/<run>/evaluation.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from statistics import mean, median

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.dashboard_scripts._theme import THEME_CSS, score_color as _score_color
from src.evaluate.preprocessing import extract_function_body


# ── helpers ──────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


from src.data.autopatch import AutoPatchDataset

_find_cve_dir = AutoPatchDataset.find_cve_dir


def _load_input_code(cve_id: str, base_dir: Path) -> str | None:
    """Load the original vulnerable code for a CVE."""
    cve_dir = _find_cve_dir(cve_id, base_dir)
    if cve_dir is None:
        return None
    original = cve_dir / "original_code.txt"
    if original.exists():
        return original.read_text(errors="replace").strip()
    return None


def _load_ground_truth(cve_id: str, variant: str, base_dir: Path) -> str | None:
    """Load the ground-truth fixed code for a CVE+variant."""
    cve_dir = _find_cve_dir(cve_id, base_dir)
    if cve_dir is None:
        return None
    gt_file = cve_dir / "out_v2" / "code" / f"{variant}_fixed.c"
    if gt_file.exists():
        raw = gt_file.read_text(errors="replace")
        return extract_function_body(raw).strip()
    return None


def _fmt(val, ndigits: int = 4) -> str:
    if val is None:
        return "-"
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, float):
        return f"{val:.{ndigits}f}"
    return str(val)


def _summarize_metric(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


# ── core analysis ────────────────────────────────────────────────────

def _build_record(result: dict, evaluation: dict, base_dir: Path) -> dict:
    """Merge a results.jsonl record with its evaluation.jsonl counterpart."""
    cve_id = result.get("query_cve", "")
    variant = result.get("query_variant", "")

    input_code = _load_input_code(cve_id, base_dir)
    ground_truth = _load_ground_truth(cve_id, variant, base_dir)
    generated = (result.get("generated_patch") or "").strip()

    retrieval = result.get("retrieval", {})
    metrics = evaluation.get("metrics_vs_function_body", {})
    size = evaluation.get("size_info", {})

    return {
        "query_cve": cve_id,
        "query_cwe": result.get("query_cwe"),
        "query_variant": variant,
        "example_cve": result.get("example_cve"),
        "example_variant": result.get("example_variant"),
        # retrieval
        "retrieval": {
            "cve_match": result.get("cve_match"),
            "cwe_match": result.get("cwe_match"),
            "similarity": result.get("similarity"),
            "retrieved_variant": retrieval.get("retrieved_variant"),
        },
        # code triple
        "input_code": input_code,
        "ground_truth": ground_truth,
        "generated_patch": generated,
        # evaluation scores
        "scores": metrics,
        "size_info": size,
        "status": result.get("status"),
        "elapsed_s": result.get("elapsed_s"),
    }


def _load_llm_eval(run_dir: Path) -> dict[tuple[str, str], dict]:
    """Load LLM vulnerability evaluation results if available."""
    llm_path = run_dir / "llm_vulnerability_eval.jsonl"
    if not llm_path.exists():
        return {}
    index: dict[tuple[str, str], dict] = {}
    for entry in _load_jsonl(llm_path):
        key = (entry.get("query_cve", ""), entry.get("query_variant", ""))
        index[key] = entry
    return index


def analyze(
    results_path: Path,
    evaluation_path: Path,
    base_dir: Path,
) -> dict:
    results = _load_jsonl(results_path)
    evaluations = _load_jsonl(evaluation_path)
    llm_eval_index = _load_llm_eval(results_path.parent)

    # Index evaluations by (query_cve, query_variant)
    eval_index: dict[tuple[str, str], dict] = {}
    for ev in evaluations:
        key = (ev.get("query_cve", ""), ev.get("query_variant", ""))
        eval_index[key] = ev

    records = []
    for r in results:
        key = (r.get("query_cve", ""), r.get("query_variant", ""))
        ev = eval_index.get(key, {})
        rec = _build_record(r, ev, base_dir)
        # Attach LLM evaluation if available
        llm = llm_eval_index.get(key)
        if llm:
            rec["llm_eval"] = {
                "verdict": llm.get("verdict", ""),
                "confidence": llm.get("confidence", 0.0),
                "reasoning": llm.get("reasoning", ""),
                "fix_description": llm.get("fix_description", ""),
                "issues": llm.get("issues", []),
            }
        else:
            rec["llm_eval"] = None
        records.append(rec)

    # Aggregate scores
    metric_keys = [
        "exact_match", "normalised_exact_match",
        "char_sequence_ratio", "line_sequence_ratio",
        "normalised_edit_distance",
        "token_jaccard", "token_jaccard_multiset",
        "bleu_1", "bleu_2", "bleu_4",
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
        "rouge1_f1", "rouge2_f1", "rougeL_f1", "rougeL_precision", "rougeL_recall",
    ]
    aggregates = {}
    for mk in metric_keys:
        vals = [
            rec["scores"][mk]
            for rec in records
            if mk in rec.get("scores", {}) and isinstance(rec["scores"][mk], (int, float))
        ]
        aggregates[mk] = _summarize_metric([float(v) for v in vals])

    # Per-CWE breakdown of key metrics
    by_cwe: dict[str, list[dict]] = {}
    for rec in records:
        cwe = rec.get("query_cwe", "unknown")
        by_cwe.setdefault(cwe, []).append(rec)

    cwe_summary = {}
    for cwe, recs in sorted(by_cwe.items()):
        n = len(recs)
        def _cwe_avg(key):
            vals = [r["scores"][key] for r in recs if key in r.get("scores", {}) and isinstance(r["scores"].get(key), (int, float))]
            return round(mean(vals), 4) if vals else None
        cwe_summary[cwe] = {
            "count": n,
            "avg_bleu_4": _cwe_avg("bleu_4"),
            "avg_bertscore_f1": _cwe_avg("bertscore_f1"),
            "avg_token_jaccard": _cwe_avg("token_jaccard"),
            "avg_rouge1_f1": _cwe_avg("rouge1_f1"),
            "avg_rouge2_f1": _cwe_avg("rouge2_f1"),
            "avg_rougeL_f1": _cwe_avg("rougeL_f1"),
        }

    # Per-variant breakdown
    by_variant: dict[str, list[dict]] = {}
    for rec in records:
        var = rec.get("query_variant", "unknown")
        by_variant.setdefault(var, []).append(rec)

    variant_summary = {}
    for var, recs in sorted(by_variant.items()):
        n = len(recs)
        def _var_avg(key):
            vals = [r["scores"][key] for r in recs if key in r.get("scores", {}) and isinstance(r["scores"].get(key), (int, float))]
            return round(mean(vals), 4) if vals else None
        variant_summary[var] = {
            "count": n,
            "avg_bleu_4": _var_avg("bleu_4"),
            "avg_bertscore_f1": _var_avg("bertscore_f1"),
            "avg_token_jaccard": _var_avg("token_jaccard"),
            "avg_rouge1_f1": _var_avg("rouge1_f1"),
            "avg_rouge2_f1": _var_avg("rouge2_f1"),
            "avg_rougeL_f1": _var_avg("rougeL_f1"),
        }

    # LLM evaluation summary
    llm_summary = None
    llm_records = [r for r in records if r.get("llm_eval")]
    if llm_records:
        from collections import Counter
        verdict_counts = Counter(r["llm_eval"]["verdict"] for r in llm_records)
        total_llm = len(llm_records)
        llm_summary = {
            "total": total_llm,
            "verdicts": dict(verdict_counts),
            "fix_rate": round(
                (verdict_counts.get("FIXED", 0) + verdict_counts.get("PARTIAL", 0))
                / total_llm * 100, 1
            ) if total_llm else 0,
            "avg_confidence": round(
                mean(r["llm_eval"]["confidence"] for r in llm_records), 3
            ),
        }

    return {
        "source": {
            "results": str(results_path),
            "evaluation": str(evaluation_path),
        },
        "total_records": len(records),
        "aggregates": aggregates,
        "by_cwe": cwe_summary,
        "by_variant": variant_summary,
        "llm_evaluation": llm_summary,
        "records": records,
    }


# ── HTML rendering ───────────────────────────────────────────────────


def _render_html(analysis: dict) -> str:
    records = analysis["records"]
    agg = analysis["aggregates"]

    # ── summary table ────────────────────────────────────────────
    summary_rows = ""
    key_metrics = [
        ("BLEU-4", "bleu_4"),
        ("BERTScore F1", "bertscore_f1"),
        ("BERTScore P", "bertscore_precision"),
        ("BERTScore R", "bertscore_recall"),
        ("ROUGE-1 F1", "rouge1_f1"),
        ("ROUGE-2 F1", "rouge2_f1"),
        ("ROUGE-L F1", "rougeL_f1"),
        ("ROUGE-L P", "rougeL_precision"),
        ("ROUGE-L R", "rougeL_recall"),
        ("Token Jaccard", "token_jaccard"),
        ("Char Seq Ratio", "char_sequence_ratio"),
        ("Line Seq Ratio", "line_sequence_ratio"),
        ("Edit Dist (norm)", "normalised_edit_distance"),
    ]
    for label, key in key_metrics:
        s = agg.get(key, {})
        summary_rows += (
            f"<tr><td>{escape(label)}</td>"
            f"<td>{_fmt(s.get('mean'))}</td>"
            f"<td>{_fmt(s.get('median'))}</td>"
            f"<td>{_fmt(s.get('min'))}</td>"
            f"<td>{_fmt(s.get('max'))}</td>"
            f"<td>{s.get('n', 0)}</td></tr>\n"
        )

    # ── per-CWE table ────────────────────────────────────────────
    cwe_rows = ""
    for cwe, stats in sorted(analysis.get("by_cwe", {}).items()):
        cwe_rows += (
            f"<tr><td>{escape(cwe)}</td>"
            f"<td>{stats['count']}</td>"
            f"<td style='color:{_score_color(stats.get('avg_bleu_4'))}'>{_fmt(stats.get('avg_bleu_4'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_bertscore_f1'))}'>{_fmt(stats.get('avg_bertscore_f1'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_token_jaccard'))}'>{_fmt(stats.get('avg_token_jaccard'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_rouge1_f1'))}'>{_fmt(stats.get('avg_rouge1_f1'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_rouge2_f1'))}'>{_fmt(stats.get('avg_rouge2_f1'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_rougeL_f1'))}'>{_fmt(stats.get('avg_rougeL_f1'))}</td>"
            f"</tr>\n"
        )

    # ── per-variant table ────────────────────────────────────────
    variant_rows = ""
    for var, stats in sorted(analysis.get("by_variant", {}).items()):
        variant_rows += (
            f"<tr><td>{escape(var)}</td>"
            f"<td>{stats['count']}</td>"
            f"<td style='color:{_score_color(stats.get('avg_bleu_4'))}'>{_fmt(stats.get('avg_bleu_4'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_bertscore_f1'))}'>{_fmt(stats.get('avg_bertscore_f1'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_token_jaccard'))}'>{_fmt(stats.get('avg_token_jaccard'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_rouge1_f1'))}'>{_fmt(stats.get('avg_rouge1_f1'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_rouge2_f1'))}'>{_fmt(stats.get('avg_rouge2_f1'))}</td>"
            f"<td style='color:{_score_color(stats.get('avg_rougeL_f1'))}'>{_fmt(stats.get('avg_rougeL_f1'))}</td>"
            f"</tr>\n"
        )

    # ── per-record cards ─────────────────────────────────────────
    record_cards = ""
    for i, rec in enumerate(records, 1):
        scores = rec.get("scores", {})
        retrieval = rec.get("retrieval", {})
        llm_eval = rec.get("llm_eval")

        score_rows = ""
        for label, key in key_metrics:
            v = scores.get(key)
            score_rows += (
                f"<tr><td>{escape(label)}</td>"
                f"<td style='color:{_score_color(v) if isinstance(v, (int, float)) else '#888'}'>"
                f"<strong>{_fmt(v)}</strong></td></tr>\n"
            )

        retrieval_info = (
            f"CVE match: <strong>{_fmt(retrieval.get('cve_match'))}</strong> | "
            f"CWE match: <strong>{_fmt(retrieval.get('cwe_match'))}</strong> | "
            f"Similarity: <strong>{_fmt(retrieval.get('similarity'))}</strong> | "
            f"Retrieved variant: <strong>{escape(str(retrieval.get('retrieved_variant', '-')))}</strong>"
        )

        # LLM evaluation box (per-record)
        llm_box = ""
        if llm_eval:
            verdict = llm_eval.get("verdict", "")
            verdict_color = {"FIXED": "#2e7d32", "PARTIAL": "#f57c00", "NOT_FIXED": "#c62828", "ERROR": "#888"}.get(verdict, "#888")
            issues_html = ""
            if llm_eval.get("issues"):
                issues_html = "<ul class='llm-issues'>" + "".join(
                    f"<li>{escape(issue)}</li>" for issue in llm_eval["issues"]
                ) + "</ul>"
            llm_box = f"""
            <div class="llm-eval-box">
              <h4>LLM Vulnerability Assessment</h4>
              <div class="llm-verdict" style="color:{verdict_color}">
                <strong>{escape(verdict)}</strong>
                <span class="llm-confidence">(confidence: {llm_eval.get('confidence', 0):.2f})</span>
              </div>
              <div class="llm-reasoning">{escape(llm_eval.get('reasoning', ''))}</div>
              <div class="llm-fix-desc"><em>{escape(llm_eval.get('fix_description', ''))}</em></div>
              {issues_html}
            </div>
            """

        # Verdict badge for summary line
        verdict_badge = ""
        if llm_eval:
            v = llm_eval.get("verdict", "")
            vc = {"FIXED": "#2e7d32", "PARTIAL": "#f57c00", "NOT_FIXED": "#c62828", "ERROR": "#888"}.get(v, "#888")
            verdict_badge = f"<span class='verdict-badge' style='background:{vc}'>{escape(v)}</span>"

        record_cards += f"""
        <details class="card" {'open' if i <= 3 else ''}>
          <summary>
            <span class="idx">#{i}</span>
            <strong>{escape(rec.get('query_cve', ''))}</strong> /
            {escape(rec.get('query_variant', ''))}
            &mdash;
            <span style="color:{_score_color(scores.get('bertscore_f1'))}">
              BERTScore F1={_fmt(scores.get('bertscore_f1'))}
            </span>
            &nbsp;
            <span style="color:{_score_color(scores.get('bleu_4'))}">
              BLEU-4={_fmt(scores.get('bleu_4'))}
            </span>
            &nbsp; CWE: {escape(str(rec.get('query_cwe', '')))}
          </summary>
          <div class="card-body">
            <div class="retrieval-info">{retrieval_info}</div>
            <div class="meta">
              Example: {escape(str(rec.get('example_cve', '')))} / {escape(str(rec.get('example_variant', '')))}
              &nbsp;|&nbsp; Status: {escape(str(rec.get('status', '')))}
              &nbsp;|&nbsp; Elapsed: {_fmt(rec.get('elapsed_s'), 2)}s
            </div>
            <table class="score-table">
              <tr><th>Metric</th><th>Value</th></tr>
              {score_rows}
            </table>
            <div class="code-triple">
              <div class="code-block-light">
                <h4>Input (Vulnerable Code)</h4>
                <pre>{escape(rec.get('input_code') or '')}</pre>
              </div>
              <div class="code-block-light">
                <h4>Ground Truth (Fixed)</h4>
                <pre>{escape(rec.get('ground_truth') or '')}</pre>
              </div>
              <div class="code-block-light">
                <h4>Agent Patch (Generated)</h4>
                <pre>{escape(rec.get('generated_patch') or '')}</pre>
              </div>
            </div>
          </div>
        </details>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Patch Evaluation Analysis</title>
<style>{THEME_CSS}
  .score-table {{ width: auto; max-width: 400px; }}
  .idx {{ color: var(--muted); font-weight: normal; }}
</style>
</head>
<body>
<div class="content">
<div class="page-header" style="padding:0">
  <h1>Patch Evaluation Analysis</h1>
  <p class="meta">{analysis['total_records']} records from <code>{escape(analysis['source']['results'])}</code></p>
</div>

<h2>Aggregate Scores</h2>
<table>
  <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>N</th></tr>
  {summary_rows}
</table>

<h2>By CWE Type</h2>
<table>
  <tr><th>CWE</th><th>Count</th><th>BLEU-4</th><th>BERTScore F1</th><th>Jaccard</th><th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th></tr>
  {cwe_rows}
</table>

<h2>By Variant</h2>
<table>
  <tr><th>Variant</th><th>Count</th><th>BLEU-4</th><th>BERTScore F1</th><th>Jaccard</th><th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th></tr>
  {variant_rows}
</table>

<h2>Per-Record Details</h2>
<p class="sub">Click to expand each record. First 3 are open by default.</p>
{record_cards}
</div>
</body>
</html>"""


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze patch evaluation results with code triples and scores."
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to results.jsonl from batch inference",
    )
    parser.add_argument(
        "--evaluation", default=None,
        help="Path to evaluation.jsonl (default: sibling of results.jsonl)",
    )
    parser.add_argument(
        "--base-dir", default=None,
        help="Base directory for CVE-list (default: repo root / cwd)",
    )
    parser.add_argument(
        "--out-json", default=None,
        help="Output JSON path (default: <run_dir>/patch_analysis.json)",
    )
    parser.add_argument(
        "--out-html", default=None,
        help="Output HTML path (default: <run_dir>/patch_analysis.html)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    eval_path = (
        Path(args.evaluation) if args.evaluation
        else results_path.parent / "evaluation.jsonl"
    )
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found. Run evaluate_patches first.")
        sys.exit(1)

    base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()
    out_json = Path(args.out_json) if args.out_json else results_path.parent / "patch_analysis.json"
    out_html = Path(args.out_html) if args.out_html else results_path.parent / "patch_analysis.html"

    print(f"Results:    {results_path}")
    print(f"Evaluation: {eval_path}")
    print(f"Base dir:   {base_dir}")

    analysis = analyze(results_path, eval_path, base_dir)

    # Write JSON (full code included for downstream analysis)
    json_analysis = {k: v for k, v in analysis.items() if k != "records"}
    json_analysis["records"] = []
    for rec in analysis["records"]:
        slim = dict(rec)
        json_analysis["records"].append(slim)

    out_json.write_text(json.dumps(json_analysis, indent=2, default=str))
    print(f"JSON:       {out_json}")

    # Write HTML (full code in expandable cards)
    out_html.write_text(_render_html(analysis))
    print(f"HTML:       {out_html}")

    # Print summary
    agg = analysis["aggregates"]
    print(f"\n{'═'*60}")
    print(f"  PATCH ANALYSIS  ({analysis['total_records']} records)")
    print(f"{'═'*60}")
    for label, key in [
        ("BLEU-4", "bleu_4"),
        ("BERTScore F1", "bertscore_f1"),
        ("ROUGE-1 F1", "rouge1_f1"),
        ("ROUGE-2 F1", "rouge2_f1"),
        ("ROUGE-L F1", "rougeL_f1"),
        ("Token Jaccard", "token_jaccard"),
        ("Edit Dist", "normalised_edit_distance"),
    ]:
        s = agg.get(key, {})
        print(f"  {label:20s}  mean={_fmt(s.get('mean'))}  median={_fmt(s.get('median'))}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
