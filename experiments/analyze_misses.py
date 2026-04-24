#!/usr/bin/env python3
"""
Analyze experiment retrieval misses and score uncertainty.

Inputs:
  - experiments/output/<run_id>/results.json

Outputs:
  - miss_analysis.json: structured miss/uncertainty statistics per cell
  - miss_dashboard.html: human-friendly dashboard

Focus:
  1) How often top-1 CVE is wrong but CWE is correct
  2) How far away the true CVE appears in ranking when missed
  3) How often wrong predictions are low-confidence ("not sure")
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _safe_float(value: Any, default: float = 0.0) -> float:
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
        return values[0]
    vals = sorted(values)
    pos = (len(vals) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p25": _percentile(values, 25.0),
        "p75": _percentile(values, 75.0),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    if temperature <= 0:
        temperature = 1.0
    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps)
    if z <= 0:
        return [0.0 for _ in exps]
    return [e / z for e in exps]


def _analyze_query(
    query: dict[str, Any],
    uncertainty_prob_threshold: float,
    uncertainty_margin_threshold: float,
    temperature: float,
) -> dict[str, Any]:
    retrieved = query.get("retrieved", []) or []
    query_cve = query.get("query_cve")
    query_cwe = query.get("query_cwe")

    if not retrieved:
        return {
            "query_cve": query_cve,
            "query_cwe": query_cwe,
            "has_results": False,
            "top1_cve": None,
            "top1_cwe": None,
            "top1_score": None,
            "top2_score": None,
            "score_margin": None,
            "top1_prob": None,
            "norm_entropy": None,
            "actual_cve_rank": None,
            "actual_cve_distance": None,
            "top1_cve_hit": False,
            "top1_cwe_hit": False,
            "any_cwe_hit": False,
            "uncertain_prob": True,
            "uncertain_margin": True,
            "uncertain_either": True,
        }

    scores = [_safe_float(r.get("score"), 0.0) for r in retrieved]
    probs = _softmax(scores, temperature=temperature)

    top1 = retrieved[0]
    top1_cve = top1.get("cve_id")
    top1_cwe = top1.get("cwe_id")
    top1_score = _safe_float(top1.get("score"), 0.0)
    top2_score = _safe_float(retrieved[1].get("score"), 0.0) if len(retrieved) >= 2 else None
    margin = (top1_score - top2_score) if top2_score is not None else None

    top1_prob = probs[0] if probs else None
    if probs and len(probs) > 1:
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(len(probs))
        norm_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        norm_entropy = 0.0

    actual_rank = None
    for i, r in enumerate(retrieved, start=1):
        if r.get("cve_id") == query_cve:
            actual_rank = i
            break

    top1_cve_hit = bool(top1_cve == query_cve)
    top1_cwe_hit = bool(top1_cwe == query_cwe)
    any_cwe_hit = any(r.get("cwe_id") == query_cwe for r in retrieved)

    uncertain_prob = (top1_prob is None) or (top1_prob < uncertainty_prob_threshold)
    uncertain_margin = (margin is None) or (margin < uncertainty_margin_threshold)

    return {
        "query_cve": query_cve,
        "query_cwe": query_cwe,
        "has_results": True,
        "top1_cve": top1_cve,
        "top1_cwe": top1_cwe,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "score_margin": margin,
        "top1_prob": top1_prob,
        "norm_entropy": norm_entropy,
        "actual_cve_rank": actual_rank,
        "actual_cve_distance": (actual_rank - 1) if actual_rank is not None else None,
        "top1_cve_hit": top1_cve_hit,
        "top1_cwe_hit": top1_cwe_hit,
        "any_cwe_hit": any_cwe_hit,
        "uncertain_prob": uncertain_prob,
        "uncertain_margin": uncertain_margin,
        "uncertain_either": (uncertain_prob or uncertain_margin),
    }


def _analyze_cell(
    cell: dict[str, Any],
    uncertainty_quantile: float,
    uncertainty_prob_floor: float,
    uncertainty_margin_floor: float,
    temperature: float,
    max_examples: int,
) -> dict[str, Any]:
    sr = cell.get("self_retrieval", {})
    raw_queries = sr.get("raw_queries", []) or []

    first_pass = []
    top1_probs = []
    margins = []

    for q in raw_queries:
        analyzed = _analyze_query(
            q,
            uncertainty_prob_threshold=0.0,
            uncertainty_margin_threshold=0.0,
            temperature=temperature,
        )
        first_pass.append(analyzed)
        if analyzed["top1_prob"] is not None:
            top1_probs.append(analyzed["top1_prob"])
        if analyzed["score_margin"] is not None:
            margins.append(analyzed["score_margin"])

    q_prob = _percentile(top1_probs, uncertainty_quantile)
    q_margin = _percentile(margins, uncertainty_quantile)
    prob_thr = max(uncertainty_prob_floor, q_prob if q_prob is not None else uncertainty_prob_floor)
    margin_thr = max(uncertainty_margin_floor, q_margin if q_margin is not None else uncertainty_margin_floor)

    analyzed_queries = []
    for q in raw_queries:
        analyzed_queries.append(
            _analyze_query(
                q,
                uncertainty_prob_threshold=prob_thr,
                uncertainty_margin_threshold=margin_thr,
                temperature=temperature,
            )
        )

    n_queries = len(analyzed_queries)
    top1_hits = [a for a in analyzed_queries if a["top1_cve_hit"]]
    top1_misses = [a for a in analyzed_queries if not a["top1_cve_hit"]]

    miss_top1_cwe_correct = [a for a in top1_misses if a["top1_cwe_hit"]]
    miss_any_cwe_correct = [a for a in top1_misses if a["any_cwe_hit"]]

    miss_rank_found = [a["actual_cve_rank"] for a in top1_misses if a["actual_cve_rank"] is not None]
    miss_distance_found = [a["actual_cve_distance"] for a in top1_misses if a["actual_cve_distance"] is not None]
    miss_rank_not_found = sum(1 for a in top1_misses if a["actual_cve_rank"] is None)

    rank_hist = {}
    for r in miss_rank_found:
        key = str(int(r))
        rank_hist[key] = rank_hist.get(key, 0) + 1

    wrong_uncertain = [a for a in top1_misses if a["uncertain_either"]]
    wrong_confident = [a for a in top1_misses if not a["uncertain_either"]]

    correct_scores = [_safe_float(a["top1_score"]) for a in top1_hits if a["top1_score"] is not None]
    wrong_scores = [_safe_float(a["top1_score"]) for a in top1_misses if a["top1_score"] is not None]
    correct_margin = [_safe_float(a["score_margin"]) for a in top1_hits if a["score_margin"] is not None]
    wrong_margin = [_safe_float(a["score_margin"]) for a in top1_misses if a["score_margin"] is not None]

    wrong_uncertain_sorted = sorted(
        wrong_uncertain,
        key=lambda x: (
            x["top1_prob"] if x["top1_prob"] is not None else 1.0,
            x["score_margin"] if x["score_margin"] is not None else 1.0,
        ),
    )
    wrong_confident_sorted = sorted(
        wrong_confident,
        key=lambda x: (
            -(x["top1_prob"] if x["top1_prob"] is not None else 0.0),
            -(x["score_margin"] if x["score_margin"] is not None else 0.0),
        ),
    )

    n_misses = len(top1_misses)

    return {
        "cell": {
            "embedder": cell.get("embedder"),
            "backend": cell.get("backend"),
            "graph_variant": cell.get("graph_variant"),
        },
        "counts": {
            "n_queries": n_queries,
            "n_top1_cve_hit": len(top1_hits),
            "n_top1_cve_miss": n_misses,
            "top1_cve_hit_rate": (len(top1_hits) / n_queries) if n_queries else 0.0,
            "n_miss_but_top1_cwe_correct": len(miss_top1_cwe_correct),
            "rate_miss_but_top1_cwe_correct_over_misses": (len(miss_top1_cwe_correct) / n_misses) if n_misses else 0.0,
            "n_miss_but_any_topk_cwe_correct": len(miss_any_cwe_correct),
            "rate_miss_but_any_topk_cwe_correct_over_misses": (len(miss_any_cwe_correct) / n_misses) if n_misses else 0.0,
        },
        "actual_cve_rank_when_missed": {
            "n_misses": n_misses,
            "n_true_cve_found_in_topk": len(miss_rank_found),
            "n_true_cve_not_found_in_topk": miss_rank_not_found,
            "found_rate": (len(miss_rank_found) / n_misses) if n_misses else 0.0,
            "rank_stats": _summarize([float(x) for x in miss_rank_found]),
            "distance_stats": _summarize([float(x) for x in miss_distance_found]),
            "rank_histogram": rank_hist,
        },
        "uncertainty": {
            "temperature": temperature,
            "thresholds": {
                "quantile_percent": uncertainty_quantile,
                "prob_threshold": prob_thr,
                "margin_threshold": margin_thr,
            },
            "wrong_and_uncertain": {
                "count": len(wrong_uncertain),
                "rate_over_wrong": (len(wrong_uncertain) / n_misses) if n_misses else 0.0,
                "rate_over_all": (len(wrong_uncertain) / n_queries) if n_queries else 0.0,
            },
            "wrong_and_confident": {
                "count": len(wrong_confident),
                "rate_over_wrong": (len(wrong_confident) / n_misses) if n_misses else 0.0,
            },
            "score_stats": {
                "top1_score_correct": _summarize(correct_scores),
                "top1_score_wrong": _summarize(wrong_scores),
                "margin_correct": _summarize(correct_margin),
                "margin_wrong": _summarize(wrong_margin),
            },
        },
        "examples": {
            "wrong_uncertain": wrong_uncertain_sorted[:max_examples],
            "wrong_confident": wrong_confident_sorted[:max_examples],
            "miss_but_top1_cwe_correct": miss_top1_cwe_correct[:max_examples],
        },
    }


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x * 100.0:.1f}%"


def _fmt_num(x: float | None, ndigits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{ndigits}f}"


def _render_dashboard(analysis: dict[str, Any]) -> str:
    rows = []
    for c in analysis.get("cells", []):
        cell = c.get("cell", {})
        cnt = c.get("counts", {})
        unc = c.get("uncertainty", {})
        acr = c.get("actual_cve_rank_when_missed", {})
        rows.append(
            """
            <tr>
              <td>{embedder}</td>
              <td>{backend}</td>
              <td>{variant}</td>
              <td>{nq}</td>
              <td>{top1}</td>
              <td>{miss_top1_cwe}</td>
              <td>{miss_any_cwe}</td>
              <td>{found_rate}</td>
              <td>{rank_med}</td>
              <td>{wu}</td>
            </tr>
            """.format(
                embedder=escape(str(cell.get("embedder", ""))),
                backend=escape(str(cell.get("backend", ""))),
                variant=escape(str(cell.get("graph_variant", ""))),
                nq=cnt.get("n_queries", 0),
                top1=_fmt_pct(cnt.get("top1_cve_hit_rate")),
                miss_top1_cwe=f"{cnt.get('n_miss_but_top1_cwe_correct', 0)} ({_fmt_pct(cnt.get('rate_miss_but_top1_cwe_correct_over_misses'))})",
                miss_any_cwe=f"{cnt.get('n_miss_but_any_topk_cwe_correct', 0)} ({_fmt_pct(cnt.get('rate_miss_but_any_topk_cwe_correct_over_misses'))})",
                found_rate=_fmt_pct(acr.get("found_rate")),
                rank_med=_fmt_num((acr.get("rank_stats") or {}).get("median"), 2),
                wu=f"{(unc.get('wrong_and_uncertain') or {}).get('count', 0)} ({_fmt_pct((unc.get('wrong_and_uncertain') or {}).get('rate_over_wrong'))})",
            )
        )

    detail_blocks = []
    for c in analysis.get("cells", []):
        cell = c.get("cell", {})
        cnt = c.get("counts", {})
        unc = c.get("uncertainty", {})
        acr = c.get("actual_cve_rank_when_missed", {})
        ex = c.get("examples", {})

        ident = f"{cell.get('embedder')} | {cell.get('backend')} | {cell.get('graph_variant')}"

        rank_hist_items = "".join(
            f"<li>rank {escape(k)}: {v}</li>"
            for k, v in sorted((acr.get("rank_histogram") or {}).items(), key=lambda t: int(t[0]))
        ) or "<li>No missed queries recovered within top-k.</li>"

        wrong_uncertain_rows = []
        for q in ex.get("wrong_uncertain", []):
            wrong_uncertain_rows.append(
                "<tr>"
                f"<td>{escape(str(q.get('query_cve')))}</td>"
                f"<td>{escape(str(q.get('query_cwe')))}</td>"
                f"<td>{escape(str(q.get('top1_cve')))}</td>"
                f"<td>{escape(str(q.get('top1_cwe')))}</td>"
                f"<td>{_fmt_num(q.get('top1_score'), 4)}</td>"
                f"<td>{_fmt_num(q.get('score_margin'), 4)}</td>"
                f"<td>{_fmt_num(q.get('top1_prob'), 4)}</td>"
                f"<td>{q.get('actual_cve_rank') if q.get('actual_cve_rank') is not None else '-'}</td>"
                "</tr>"
            )

        detail_blocks.append(
            """
            <details class="card">
              <summary>{ident}</summary>
              <div class="grid">
                <div>
                  <h4>Miss Profile</h4>
                  <ul>
                    <li>Queries: {n_queries}</li>
                    <li>Top-1 CVE hit: {top1_hit}</li>
                    <li>Miss but top-1 CWE correct: {miss_top1_cwe}</li>
                    <li>Miss but any top-k CWE correct: {miss_any_cwe}</li>
                    <li>True CVE found in top-k (when missed): {found_rate}</li>
                    <li>Median true CVE rank (miss subset): {rank_median}</li>
                  </ul>
                </div>
                <div>
                  <h4>Uncertainty</h4>
                  <ul>
                    <li>Wrong and uncertain: {wu}</li>
                    <li>Wrong and confident: {wc}</li>
                    <li>Uncertainty prob threshold: {pth}</li>
                    <li>Uncertainty margin threshold: {mth}</li>
                  </ul>
                </div>
              </div>
              <h4>True CVE Rank Histogram (Miss Subset)</h4>
              <ul>{rank_hist}</ul>
              <h4>Wrong + Uncertain Examples</h4>
              <table>
                <thead>
                  <tr>
                    <th>query_cve</th><th>query_cwe</th><th>top1_cve</th><th>top1_cwe</th>
                    <th>top1_score</th><th>margin</th><th>top1_prob</th><th>true_rank</th>
                  </tr>
                </thead>
                <tbody>
                  {wrong_uncertain_rows}
                </tbody>
              </table>
            </details>
            """.format(
                ident=escape(ident),
                n_queries=cnt.get("n_queries", 0),
                top1_hit=_fmt_pct(cnt.get("top1_cve_hit_rate")),
                miss_top1_cwe=f"{cnt.get('n_miss_but_top1_cwe_correct', 0)} ({_fmt_pct(cnt.get('rate_miss_but_top1_cwe_correct_over_misses'))})",
                miss_any_cwe=f"{cnt.get('n_miss_but_any_topk_cwe_correct', 0)} ({_fmt_pct(cnt.get('rate_miss_but_any_topk_cwe_correct_over_misses'))})",
                found_rate=_fmt_pct(acr.get("found_rate")),
                rank_median=_fmt_num((acr.get("rank_stats") or {}).get("median"), 2),
                wu=f"{(unc.get('wrong_and_uncertain') or {}).get('count', 0)} ({_fmt_pct((unc.get('wrong_and_uncertain') or {}).get('rate_over_wrong'))})",
                wc=f"{(unc.get('wrong_and_confident') or {}).get('count', 0)} ({_fmt_pct((unc.get('wrong_and_confident') or {}).get('rate_over_wrong'))})",
                pth=_fmt_num(((unc.get("thresholds") or {}).get("prob_threshold")), 4),
                mth=_fmt_num(((unc.get("thresholds") or {}).get("margin_threshold")), 4),
                rank_hist=rank_hist_items,
                wrong_uncertain_rows="\n".join(wrong_uncertain_rows) or "<tr><td colspan='8'>No examples.</td></tr>",
            )
        )

    global_section = analysis.get("global", {})
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Miss Analysis Dashboard</title>
  <style>
    :root {{
      --bg: #f7f4ef;
      --ink: #1b1b1b;
      --muted: #5c5c5c;
      --accent: #0e6b6e;
      --card: #ffffff;
      --line: #ddd4c8;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 10% 0%, #fff8ec, var(--bg));
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 24px auto 56px;
      padding: 0 16px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 2rem; }}
    p.meta {{ margin: 0 0 18px; color: var(--muted); }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      margin: 14px 0;
      box-shadow: 0 8px 28px rgba(10, 20, 30, 0.07);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.93rem; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; }}
    th {{ background: #fff6e8; position: sticky; top: 0; }}
    details > summary {{ cursor: pointer; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }}
    ul {{ margin-top: 6px; }}
    .pill {{ display: inline-block; padding: 4px 8px; border-radius: 999px; background: #e6f4f4; color: #0e6b6e; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Miss And Uncertainty Dashboard</h1>
    <p class="meta">Run <span class="pill">{escape(str(analysis.get('run_id', '-')))}</span> | Generated {escape(str(analysis.get('generated_at', '-')))}</p>

    <div class="card">
      <h3>Global Summary</h3>
      <ul>
        <li>Cells analyzed: {global_section.get('n_cells', 0)}</li>
        <li>Total queries: {global_section.get('n_queries', 0)}</li>
        <li>Total top-1 CVE misses: {global_section.get('n_top1_misses', 0)}</li>
        <li>Total miss but top-1 CWE correct: {global_section.get('n_miss_top1_cwe_correct', 0)} ({_fmt_pct(global_section.get('rate_miss_top1_cwe_correct_over_misses'))})</li>
        <li>Total miss but any top-k CWE correct: {global_section.get('n_miss_any_cwe_correct', 0)} ({_fmt_pct(global_section.get('rate_miss_any_cwe_correct_over_misses'))})</li>
        <li>Wrong and uncertain: {global_section.get('n_wrong_uncertain', 0)} ({_fmt_pct(global_section.get('rate_wrong_uncertain_over_wrong'))})</li>
      </ul>
    </div>

    <div class="card">
      <h3>Cell Comparison</h3>
      <table>
        <thead>
          <tr>
            <th>embedder</th>
            <th>backend</th>
            <th>variant</th>
            <th>queries</th>
            <th>top1_cve_hit</th>
            <th>miss + top1_cwe_ok</th>
            <th>miss + any_cwe_topk</th>
            <th>true_cve_found_on_miss</th>
            <th>true_rank_median</th>
            <th>wrong_uncertain</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>

    <h3>Per-Cell Details</h3>
    {''.join(detail_blocks)}
  </div>
</body>
</html>
"""
    return html


def analyze_results(
    results_path: Path,
    out_json: Path,
    out_html: Path,
    uncertainty_quantile: float,
    uncertainty_prob_floor: float,
    uncertainty_margin_floor: float,
    temperature: float,
    max_examples: int,
) -> dict[str, Any]:
    raw = json.loads(results_path.read_text())
    cells = raw.get("cells", []) or []

    analyzed_cells = []
    total_queries = 0
    total_misses = 0
    total_miss_top1_cwe = 0
    total_miss_any_cwe = 0
    total_wrong_uncertain = 0

    for cell in cells:
        analyzed = _analyze_cell(
            cell=cell,
            uncertainty_quantile=uncertainty_quantile,
            uncertainty_prob_floor=uncertainty_prob_floor,
            uncertainty_margin_floor=uncertainty_margin_floor,
            temperature=temperature,
            max_examples=max_examples,
        )
        analyzed_cells.append(analyzed)

        cnt = analyzed["counts"]
        unc = analyzed["uncertainty"]
        total_queries += cnt["n_queries"]
        total_misses += cnt["n_top1_cve_miss"]
        total_miss_top1_cwe += cnt["n_miss_but_top1_cwe_correct"]
        total_miss_any_cwe += cnt["n_miss_but_any_topk_cwe_correct"]
        total_wrong_uncertain += unc["wrong_and_uncertain"]["count"]

    analysis = {
        "run_id": raw.get("run_id"),
        "source_results": str(results_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "uncertainty_quantile_percent": uncertainty_quantile,
            "uncertainty_prob_floor": uncertainty_prob_floor,
            "uncertainty_margin_floor": uncertainty_margin_floor,
            "softmax_temperature": temperature,
            "max_examples_per_bucket": max_examples,
        },
        "global": {
            "n_cells": len(analyzed_cells),
            "n_queries": total_queries,
            "n_top1_misses": total_misses,
            "n_miss_top1_cwe_correct": total_miss_top1_cwe,
            "rate_miss_top1_cwe_correct_over_misses": (total_miss_top1_cwe / total_misses) if total_misses else 0.0,
            "n_miss_any_cwe_correct": total_miss_any_cwe,
            "rate_miss_any_cwe_correct_over_misses": (total_miss_any_cwe / total_misses) if total_misses else 0.0,
            "n_wrong_uncertain": total_wrong_uncertain,
            "rate_wrong_uncertain_over_wrong": (total_wrong_uncertain / total_misses) if total_misses else 0.0,
        },
        "cells": analyzed_cells,
    }

    out_json.write_text(json.dumps(analysis, indent=2))
    out_html.write_text(_render_dashboard(analysis))

    # Regenerate unified dashboard if it's a standard run dir
    try:
        from experiments.dashboard import generate_html_dashboard
        run_dir = out_json.parent
        if (run_dir / "results.json").exists():
            generate_html_dashboard(run_dir)
    except Exception as e:
        pass  # non-fatal; unified dashboard is best-effort

    return analysis


def _default_output_paths(results_path: Path) -> tuple[Path, Path]:
    if results_path.name == "results.json":
        run_dir = results_path.parent
        return run_dir / "miss_analysis.json", run_dir / "miss_dashboard.html"
    return results_path.with_suffix(".miss_analysis.json"), results_path.with_suffix(".miss_dashboard.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze miss patterns and uncertainty from experiment results.")
    parser.add_argument("--results", required=True, help="Path to experiments/output/<run_id>/results.json")
    parser.add_argument("--out-json", help="Output JSON path (default: run_dir/miss_analysis.json)")
    parser.add_argument("--out-html", help="Output HTML path (default: run_dir/miss_dashboard.html)")
    parser.add_argument("--uncertainty-quantile", type=float, default=25.0,
                        help="Per-cell quantile for uncertainty thresholds (default: 25)")
    parser.add_argument("--uncertainty-prob-floor", type=float, default=0.12,
                        help="Minimum threshold for top1 softmax probability uncertainty (default: 0.12)")
    parser.add_argument("--uncertainty-margin-floor", type=float, default=0.005,
                        help="Minimum threshold for top1-top2 score margin uncertainty (default: 0.005)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for score->probability mapping (default: 1.0)")
    parser.add_argument("--max-examples", type=int, default=20,
                        help="Max examples per bucket in output (default: 20)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"results file not found: {results_path}")

    default_json, default_html = _default_output_paths(results_path)
    out_json = Path(args.out_json) if args.out_json else default_json
    out_html = Path(args.out_html) if args.out_html else default_html

    analysis = analyze_results(
        results_path=results_path,
        out_json=out_json,
        out_html=out_html,
        uncertainty_quantile=args.uncertainty_quantile,
        uncertainty_prob_floor=args.uncertainty_prob_floor,
        uncertainty_margin_floor=args.uncertainty_margin_floor,
        temperature=args.temperature,
        max_examples=args.max_examples,
    )

    g = analysis["global"]
    print("Miss analysis completed")
    print(f"  Source: {results_path}")
    print(f"  JSON:   {out_json}")
    print(f"  HTML:   {out_html}")
    print(f"  Cells:  {g['n_cells']}")
    print(f"  Queries:{g['n_queries']}")
    print(f"  Misses: {g['n_top1_misses']}")
    print(
        "  Miss but top1 CWE correct: "
        f"{g['n_miss_top1_cwe_correct']} ({g['rate_miss_top1_cwe_correct_over_misses']:.1%})"
    )
    print(
        "  Wrong and uncertain: "
        f"{g['n_wrong_uncertain']} ({g['rate_wrong_uncertain_over_wrong']:.1%})"
    )


if __name__ == "__main__":
    main()
