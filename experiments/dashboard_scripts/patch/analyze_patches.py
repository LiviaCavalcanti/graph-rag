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
  - patch_analysis.html   : human-friendly HTML dashboard (two tabs)

Tabs:
  - Test Data Evaluation : metric-based evaluation (BLEU, ROUGE, BERTScore, etc.)
  - Data Evaluation      : LLM vulnerability assessment + human labeling

Usage:
    python -m experiments.dashboard_scripts.patch.analyze_patches \
        --results experiments/output/<run>/results.jsonl \
        --evaluation experiments/output/<run>/evaluation.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from helpers import load_jsonl, load_input_code, score_color, summarize_metric, fmt
from eval import build_record, _load_llm_eval, analyze
from helpers import load_ground_truth, find_cve_dir
from experiments.dashboard_scripts._theme import THEME_CSS, THEME_JS
from src.evaluate.preprocessing import extract_function_body


# ── HTML rendering ───────────────────────────────────────────────────


def _render_llm_summary(llm_summary: dict | None) -> str:
    """Render the LLM evaluation summary box, or empty string if unavailable."""
    if not llm_summary:
        return ""

    verdicts = llm_summary.get("verdicts", {})
    total = llm_summary.get("total", 0)
    fix_rate = llm_summary.get("fix_rate", 0)
    avg_conf = llm_summary.get("avg_confidence", 0)

    verdict_items = ""
    colors = {"FIXED": "#2e7d32", "PARTIAL": "#f57c00", "NOT_FIXED": "#c62828", "ERROR": "#888"}
    for v in ["FIXED", "PARTIAL", "NOT_FIXED", "ERROR"]:
        count = verdicts.get(v, 0)
        if count > 0:
            pct = count / total * 100 if total else 0
            color = colors.get(v, "#888")
            verdict_items += (
                f'<div class="llm-stat">'
                f'<div class="value" style="color:{color}">{count}</div>'
                f'<div class="label">{v} ({pct:.0f}%)</div>'
                f'</div>'
            )

    return f"""
<div class="llm-summary-box">
  <h3>LLM Vulnerability Assessment</h3>
  <div class="llm-summary-stats">
    <div class="llm-stat">
      <div class="value">{total}</div>
      <div class="label">Evaluated</div>
    </div>
    <div class="llm-stat">
      <div class="value" style="color:#1976d2">{fix_rate:.1f}%</div>
      <div class="label">Fix Rate (FIXED+PARTIAL)</div>
    </div>
    <div class="llm-stat">
      <div class="value">{avg_conf:.2f}</div>
      <div class="label">Avg Confidence</div>
    </div>
    {verdict_items}
  </div>
</div>
"""


def _render_human_summary(human_summary: dict | None) -> str:
    """Render the human labeling summary box."""
    if not human_summary:
        return ""

    verdicts = human_summary.get("verdicts", {})
    total = human_summary.get("total", 0)
    fix_rate = human_summary.get("fix_rate", 0)

    verdict_items = ""
    colors = {"FIXED": "#2e7d32", "PARTIAL": "#f57c00", "NOT_FIXED": "#c62828"}
    for v in ["FIXED", "PARTIAL", "NOT_FIXED"]:
        count = verdicts.get(v, 0)
        if count > 0:
            pct = count / total * 100 if total else 0
            color = colors.get(v, "#888")
            verdict_items += (
                f'<div class="llm-stat">'
                f'<div class="value" style="color:{color}">{count}</div>'
                f'<div class="label">{v} ({pct:.0f}%)</div>'
                f'</div>'
            )

    return f"""
<div class="human-summary-box">
  <h3>Human Labeling</h3>
  <div class="llm-summary-stats">
    <div class="llm-stat">
      <div class="value">{total}</div>
      <div class="label">Labeled</div>
    </div>
    <div class="llm-stat">
      <div class="value" style="color:#2e7d32">{fix_rate:.1f}%</div>
      <div class="label">Fix Rate (FIXED+PARTIAL)</div>
    </div>
    {verdict_items}
  </div>
</div>
"""


def _render_agreement_table(records: list[dict]) -> str:
    """Render LLM vs Human agreement confusion matrix."""
    both = [r for r in records if r.get("llm_eval") and r.get("human_label")]
    if not both:
        return ""

    verdicts = ["FIXED", "PARTIAL", "NOT_FIXED"]
    matrix: dict[str, dict[str, int]] = {v: {h: 0 for h in verdicts} for v in verdicts}
    agree = 0
    for r in both:
        lv = r["llm_eval"]["verdict"]
        hv = r["human_label"]["verdict"]
        if lv in matrix and hv in matrix[lv]:
            matrix[lv][hv] += 1
        if lv == hv:
            agree += 1

    total = len(both)
    agreement_pct = round(agree / total * 100, 1) if total else 0

    rows = ""
    for lv in verdicts:
        cells = ""
        for hv in verdicts:
            count = matrix[lv][hv]
            bg = "#e8faf3" if lv == hv and count > 0 else ""
            style = f' style="background:{bg}"' if bg else ""
            cells += f"<td{style}><strong>{count}</strong></td>"
        rows += f"<tr><td>{escape(lv)}</td>{cells}</tr>\n"

    return f"""
<h3>LLM vs Human Agreement</h3>
<p class="sub">Agreement: <strong>{agreement_pct}%</strong> ({agree}/{total})</p>
<table class="agreement-table">
  <tr><th>LLM \\ Human</th><th>FIXED</th><th>PARTIAL</th><th>NOT_FIXED</th></tr>
  {rows}
</table>
"""


def _render_data_eval_tab(analysis: dict) -> str:
    """Render the Data Evaluation tab: augmented data pairs (vulnerable code + ground truth)."""
    base_dir = Path(analysis["source"].get("base_dir", "."))
    cve_list_dir = base_dir / "CVE-list"

    if not cve_list_dir.is_dir():
        return '<p class="muted">CVE-list directory not found. Pass --base-dir to locate it.</p>'

    # Collect all augmented data pairs
    data_pairs: list[dict] = []
    for cve_dir in sorted(cve_list_dir.iterdir()):
        if not cve_dir.is_dir():
            continue
        code_dir = cve_dir / "out_v2" / "code"
        if not code_dir.is_dir():
            continue

        # Load CVE metadata
        info_path = cve_dir / "info.json"
        cve_id = cve_dir.name
        cwe = ""
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
                cve_id = info.get("cve_id", cve_dir.name)
                cwe = info.get("cwe_id", "")
            except (json.JSONDecodeError, OSError):
                pass

        # Load vulnerable code
        orig_path = cve_dir / "original_code.txt"
        if not orig_path.exists():
            continue
        vuln_code = orig_path.read_text(errors="replace").strip()

        # Find all augmented variant files
        for fixed_file in sorted(code_dir.glob("*_fixed.c")):
            variant = fixed_file.name.replace("_fixed.c", "")
            # Skip "original" variant — that's the real fix, not augmented
            if variant == "original":
                continue
            raw = fixed_file.read_text(errors="replace")
            gt_code = extract_function_body(raw).strip()
            data_pairs.append({
                "cve_id": cve_id,
                "cve_dir_name": cve_dir.name,
                "cwe": cwe,
                "variant": variant,
                "vuln_code": vuln_code,
                "gt_code": gt_code,
            })

    # Summary stats
    n_total = len(data_pairs)
    n_cves = len(set(p["cve_dir_name"] for p in data_pairs))
    variants = Counter(p["variant"] for p in data_pairs)
    cwes = Counter(p["cwe"] for p in data_pairs)

    content = f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-label">Total Pairs</div>
    <div class="stat-value">{n_total}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">CVEs</div>
    <div class="stat-value">{n_cves}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Variants</div>
    <div class="stat-value">{len(variants)}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">CWE Types</div>
    <div class="stat-value">{len(cwes)}</div>
  </div>
</div>
"""

    # Per-variant count table
    variant_rows = ""
    for var, count in sorted(variants.items(), key=lambda x: -x[1]):
        variant_rows += f"<tr><td>{escape(var)}</td><td>{count}</td></tr>\n"
    content += f"""
<h2>Variants Distribution</h2>
<table>
  <tr><th>Variant</th><th>Count</th></tr>
  {variant_rows}
</table>
"""

    # Per-CWE count table
    cwe_rows = ""
    for cwe_name, count in sorted(cwes.items(), key=lambda x: -x[1]):
        cwe_rows += f"<tr><td>{escape(cwe_name or 'unknown')}</td><td>{count}</td></tr>\n"
    content += f"""
<h2>CWE Distribution</h2>
<table>
  <tr><th>CWE</th><th>Count</th></tr>
  {cwe_rows}
</table>
"""

    # Annotation toolbar
    content += """
<h2>Augmented Data Pairs</h2>
<div class="annotation-toolbar">
  <button class="btn-save" onclick="saveAnnotations()">Save Annotations (JSON)</button>
  <label class="btn-load">Load Annotations
    <input type="file" accept=".json" onchange="loadAnnotations(event)" style="display:none">
  </label>
  <span class="annotation-status" id="annotation-status"></span>
</div>
"""
    content += f'<p class="sub">{n_total} pairs. Click to expand. First 5 open by default.</p>\n'

    for i, pair in enumerate(data_pairs, 1):
        pair_id = f"{pair['cve_dir_name']}__{pair['variant']}"
        content += f"""
        <details class="card" {'open' if i <= 5 else ''}>
          <summary>
            <span class="idx">#{i}</span>
            <strong>{escape(pair['cve_id'])}</strong> /
            {escape(pair['variant'])}
            &nbsp; <span class="pill">{escape(pair['cwe'])}</span>
            <span class="annotation-badge" id="badge-{escape(pair_id)}"></span>
          </summary>
          <div class="card-body">
            <div class="annotation-row" data-pair-id="{escape(pair_id)}">
              <label>Status:
                <select class="annotation-select" data-pair-id="{escape(pair_id)}" onchange="onAnnotationChange(this)">
                  <option value="">Not Reviewed</option>
                  <option value="correct">Correct</option>
                  <option value="wrong">Wrong</option>
                </select>
              </label>
              <label class="annotation-comment-label">Comment:
                <textarea class="annotation-comment" data-pair-id="{escape(pair_id)}"
                  placeholder="Observations about this example..." rows="2"
                  onchange="onAnnotationChange(this)"></textarea>
              </label>
            </div>
            <div class="code-triple" style="grid-template-columns: 1fr 1fr;">
              <div class="code-block-light">
                <h4>Vulnerable Code</h4>
                <pre>{escape(pair['vuln_code'])}</pre>
              </div>
              <div class="code-block-light">
                <h4>Ground Truth ({escape(pair['variant'])})</h4>
                <pre>{escape(pair['gt_code'])}</pre>
              </div>
            </div>
          </div>
        </details>
        """

    return content


def _render_html(analysis: dict) -> str:
    """Render the full two-tab HTML dashboard."""
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
            f"<td>{fmt(s.get('mean'))}</td>"
            f"<td>{fmt(s.get('median'))}</td>"
            f"<td>{fmt(s.get('min'))}</td>"
            f"<td>{fmt(s.get('max'))}</td>"
            f"<td>{s.get('n', 0)}</td></tr>\n"
        )

    # ── per-CWE table ────────────────────────────────────────────
    cwe_rows = ""
    for cwe, stats in sorted(analysis.get("by_cwe", {}).items()):
        cwe_rows += (
            f"<tr><td>{escape(cwe)}</td>"
            f"<td>{stats['count']}</td>"
            f"<td style='color:{score_color(stats.get('avg_bleu_4'))}'>{fmt(stats.get('avg_bleu_4'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_bertscore_f1'))}'>{fmt(stats.get('avg_bertscore_f1'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_token_jaccard'))}'>{fmt(stats.get('avg_token_jaccard'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_rouge1_f1'))}'>{fmt(stats.get('avg_rouge1_f1'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_rouge2_f1'))}'>{fmt(stats.get('avg_rouge2_f1'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_rougeL_f1'))}'>{fmt(stats.get('avg_rougeL_f1'))}</td>"
            f"</tr>\n"
        )

    # ── per-variant table ────────────────────────────────────────
    variant_rows = ""
    for var, stats in sorted(analysis.get("by_variant", {}).items()):
        variant_rows += (
            f"<tr><td>{escape(var)}</td>"
            f"<td>{stats['count']}</td>"
            f"<td style='color:{score_color(stats.get('avg_bleu_4'))}'>{fmt(stats.get('avg_bleu_4'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_bertscore_f1'))}'>{fmt(stats.get('avg_bertscore_f1'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_token_jaccard'))}'>{fmt(stats.get('avg_token_jaccard'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_rouge1_f1'))}'>{fmt(stats.get('avg_rouge1_f1'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_rouge2_f1'))}'>{fmt(stats.get('avg_rouge2_f1'))}</td>"
            f"<td style='color:{score_color(stats.get('avg_rougeL_f1'))}'>{fmt(stats.get('avg_rougeL_f1'))}</td>"
            f"</tr>\n"
        )

    # ── per-record cards (test data) ─────────────────────────────
    record_cards = ""
    for i, rec in enumerate(records, 1):
        scores = rec.get("scores", {})
        retrieval = rec.get("retrieval", {})
        llm_eval = rec.get("llm_eval")
        human_label = rec.get("human_label")

        score_rows = ""
        for label, key in key_metrics:
            v = scores.get(key)
            score_rows += (
                f"<tr><td>{escape(label)}</td>"
                f"<td style='color:{score_color(v) if isinstance(v, (int, float)) else '#888'}'>"
                f"<strong>{fmt(v)}</strong></td></tr>\n"
            )

        retrieval_info = (
            f"CVE match: <strong>{fmt(retrieval.get('cve_match'))}</strong> | "
            f"CWE match: <strong>{fmt(retrieval.get('cwe_match'))}</strong> | "
            f"Similarity: <strong>{fmt(retrieval.get('similarity'))}</strong> | "
            f"Retrieved variant: <strong>{escape(str(retrieval.get('retrieved_variant', '-')))}</strong>"
        )

        # Verdict badge for summary line
        verdict_badge = ""
        if llm_eval:
            v = llm_eval.get("verdict", "")
            vc = {"FIXED": "#2e7d32", "PARTIAL": "#f57c00", "NOT_FIXED": "#c62828", "ERROR": "#888"}.get(v, "#888")
            verdict_badge = f"<span class='verdict-badge' style='background:{vc}'>{escape(v)}</span>"

        # LLM eval details box
        llm_detail = ""
        if llm_eval:
            verdict = llm_eval.get("verdict", "")
            verdict_color = {"FIXED": "#2e7d32", "PARTIAL": "#f57c00", "NOT_FIXED": "#c62828", "ERROR": "#888"}.get(verdict, "#888")
            issues_html = ""
            if llm_eval.get("issues"):
                issues_html = "<ul class='llm-issues'>" + "".join(
                    f"<li>{escape(issue)}</li>" for issue in llm_eval["issues"]
                ) + "</ul>"
            llm_detail = f"""
            <div class="llm-eval-box">
              <h4>LLM Assessment</h4>
              <div class="llm-verdict" style="color:{verdict_color}">
                <strong>{escape(verdict)}</strong>
                <span class="llm-confidence">(confidence: {llm_eval.get('confidence', 0):.2f})</span>
              </div>
              <div class="llm-reasoning">{escape(llm_eval.get('reasoning', ''))}</div>
              <div class="llm-fix-desc"><em>{escape(llm_eval.get('fix_description', ''))}</em></div>
              {issues_html}
            </div>
            """

        # Human label box
        human_detail = ""
        if human_label:
            notes = human_label.get("notes", "")
            labeler = human_label.get("labeler", "")
            labeler_info = f" &mdash; <em>{escape(labeler)}</em>" if labeler else ""
            human_detail = f"""
            <div class="human-note">
              <strong>Human Label:</strong> {escape(human_label.get('verdict', ''))}{labeler_info}
              {"<br>" + escape(notes) if notes else ""}
            </div>
            """

        record_cards += f"""
        <details class="card" {'open' if i <= 3 else ''}>
          <summary>
            <span class="idx">#{i}</span>
            <strong>{escape(rec.get('query_cve', ''))}</strong> /
            {escape(rec.get('query_variant', ''))}
            &mdash;
            <span style="color:{score_color(scores.get('bertscore_f1'))}">
              BERTScore F1={fmt(scores.get('bertscore_f1'))}
            </span>
            &nbsp;
            <span style="color:{score_color(scores.get('bleu_4'))}">
              BLEU-4={fmt(scores.get('bleu_4'))}
            </span>
            &nbsp; CWE: {escape(str(rec.get('query_cwe', '')))}
            &nbsp; {verdict_badge}
          </summary>
          <div class="card-body">
            <div class="retrieval-info">{retrieval_info}</div>
            <div class="meta">
              Example: {escape(str(rec.get('example_cve', '')))} / {escape(str(rec.get('example_variant', '')))}
              &nbsp;|&nbsp; Status: {escape(str(rec.get('status', '')))}
              &nbsp;|&nbsp; Elapsed: {fmt(rec.get('elapsed_s'), 2)}s
            </div>
            <table class="score-table">
              <tr><th>Metric</th><th>Value</th></tr>
              {score_rows}
            </table>
            {llm_detail}
            {human_detail}
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

    # ── LLM/Human evaluation summaries (test data tab) ──────────
    llm_human_section = ""
    llm_summary = analysis.get("llm_evaluation")
    human_summary = analysis.get("human_evaluation")
    if llm_summary or human_summary:
        llm_human_section += _render_llm_summary(llm_summary)
        llm_human_section += _render_human_summary(human_summary)
        llm_human_section += _render_agreement_table(records)

    # ── Data Evaluation tab content ──────────────────────────────
    data_eval_content = _render_data_eval_tab(analysis)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Patch Evaluation Analysis</title>
<style>{THEME_CSS}
  .score-table {{ width: auto; max-width: 400px; }}
  .idx {{ color: var(--muted); font-weight: normal; }}
  .llm-eval-box {{
    margin: 0.8rem 0;
    padding: 0.8rem 1rem;
    border-left: 4px solid #1976d2;
    background: #f5f8ff;
    border-radius: 4px;
  }}
  .llm-eval-box h4 {{ margin: 0 0 0.4rem 0; color: #1976d2; font-size: 0.95rem; }}
  .llm-verdict {{ font-size: 1.1rem; margin-bottom: 0.4rem; }}
  .llm-confidence {{ font-size: 0.85rem; color: #666; }}
  .llm-reasoning {{ margin: 0.4rem 0; font-size: 0.9rem; line-height: 1.4; }}
  .llm-fix-desc {{ margin: 0.3rem 0; font-size: 0.85rem; color: #555; }}
  .llm-issues {{ margin: 0.3rem 0 0 1.2rem; font-size: 0.85rem; color: #c62828; }}
  .verdict-badge {{
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 3px;
    color: #fff;
    font-size: 0.75rem;
    font-weight: bold;
    vertical-align: middle;
  }}
  .llm-summary-box {{
    margin: 1rem 0;
    padding: 1rem 1.5rem;
    border: 2px solid #1976d2;
    border-radius: 8px;
    background: #f5f8ff;
  }}
  .llm-summary-box h3 {{ margin: 0 0 0.6rem 0; color: #1976d2; }}
  .llm-summary-stats {{
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    align-items: center;
  }}
  .llm-stat {{
    text-align: center;
  }}
  .llm-stat .value {{
    font-size: 1.5rem;
    font-weight: bold;
  }}
  .llm-stat .label {{
    font-size: 0.8rem;
    color: #666;
  }}
  .human-summary-box {{
    margin: 1rem 0;
    padding: 1rem 1.5rem;
    border: 2px solid #2e7d32;
    border-radius: 8px;
    background: #f0faf4;
  }}
  .human-summary-box h3 {{ margin: 0 0 0.6rem 0; color: #2e7d32; }}
  .agreement-table {{ max-width: 600px; }}
  .agreement-table td {{ text-align: center; }}
  .eval-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
  }}
  .eval-card-header {{
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    margin-bottom: 8px;
  }}
  .eval-card-header .cve {{ font-weight: 700; }}
  .eval-card-body {{ font-size: 0.9rem; line-height: 1.5; }}
  .human-note {{
    background: #f0faf4; padding: 6px 10px; border-radius: 4px;
    margin-top: 6px; border-left: 3px solid #2e7d32;
  }}
  .pill {{
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 3px;
    background: #e3f2fd;
    color: #1565c0;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .stat-row {{
    display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0;
  }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1rem 1.5rem;
    text-align: center; box-shadow: var(--shadow);
  }}
  .stat-label {{ font-size: 0.8rem; color: var(--muted); }}
  .stat-value {{ font-size: 1.5rem; font-weight: bold; }}
  .annotation-toolbar {{
    display: flex; align-items: center; gap: 1rem; margin: 1rem 0;
    padding: 0.8rem 1rem; background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: var(--shadow);
  }}
  .btn-save, .btn-load {{
    padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer;
    font-weight: 600; font-size: 0.85rem; border: none;
  }}
  .btn-save {{ background: #1976d2; color: #fff; }}
  .btn-save:hover {{ background: #1565c0; }}
  .btn-load {{
    background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9;
    display: inline-block;
  }}
  .btn-load:hover {{ background: #bbdefb; }}
  .annotation-status {{ font-size: 0.8rem; color: #2e7d32; }}
  .annotation-row {{
    display: flex; align-items: flex-start; gap: 1.5rem; margin-bottom: 1rem;
    padding: 0.6rem 0.8rem; background: #fffde7; border: 1px solid #fff9c4;
    border-radius: 4px;
  }}
  .annotation-row label {{ font-weight: 600; font-size: 0.85rem; color: #555; }}
  .annotation-select {{
    margin-left: 0.5rem; padding: 0.3rem 0.5rem; border-radius: 4px;
    border: 1px solid #ccc; font-size: 0.85rem;
  }}
  .annotation-comment-label {{ flex: 1; display: flex; flex-direction: column; }}
  .annotation-comment {{
    margin-top: 0.3rem; padding: 0.4rem; border-radius: 4px;
    border: 1px solid #ccc; font-size: 0.85rem; resize: vertical; width: 100%;
  }}
  .annotation-badge {{
    display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px;
    font-size: 0.7rem; font-weight: bold; margin-left: 0.5rem; vertical-align: middle;
  }}
  .annotation-badge.correct {{ background: #c8e6c9; color: #2e7d32; }}
  .annotation-badge.wrong {{ background: #ffcdd2; color: #c62828; }}
</style>
</head>
<body>
<div class="page-header" style="padding: 18px 32px 0;">
  <h1>Patch Evaluation Analysis</h1>
  <p class="meta">{analysis['total_records']} records from <code>{escape(analysis['source']['results'])}</code></p>
</div>

<nav class="tab-nav">
  <button class="tab-btn active" data-tab="tab-test-eval">Test Data Evaluation</button>
  <button class="tab-btn" data-tab="tab-data-eval">Data Evaluation</button>
</nav>

<div class="content">

<div id="tab-test-eval" class="tab-panel active">
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

{llm_human_section}

<h2>Per-Record Details</h2>
<p class="sub">Click to expand each record. First 3 are open by default.</p>
{record_cards}
</div>

<div id="tab-data-eval" class="tab-panel">
{data_eval_content}
</div>

</div>
<script>{THEME_JS}</script>
<script>
// Annotation persistence
function getAnnotations() {{
  const annots = {{}};
  document.querySelectorAll('.annotation-select').forEach(sel => {{
    const id = sel.dataset.pairId;
    const comment = document.querySelector('.annotation-comment[data-pair-id="' + id + '"]');
    if (sel.value || (comment && comment.value.trim())) {{
      annots[id] = {{
        status: sel.value || "not_reviewed",
        comment: comment ? comment.value.trim() : ""
      }};
    }}
  }});
  return annots;
}}

function updateBadge(pairId, status) {{
  const badge = document.getElementById('badge-' + pairId);
  if (!badge) return;
  badge.className = 'annotation-badge';
  if (status === 'correct') {{
    badge.textContent = 'CORRECT';
    badge.classList.add('correct');
  }} else if (status === 'wrong') {{
    badge.textContent = 'WRONG';
    badge.classList.add('wrong');
  }} else {{
    badge.textContent = '';
  }}
}}

function onAnnotationChange(el) {{
  const pairId = el.dataset.pairId;
  const sel = document.querySelector('.annotation-select[data-pair-id="' + pairId + '"]');
  updateBadge(pairId, sel.value);
  // Auto-save to localStorage
  localStorage.setItem('patch_annotations', JSON.stringify(getAnnotations()));
}}

function saveAnnotations() {{
  const annots = getAnnotations();
  const blob = new Blob([JSON.stringify(annots, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'data_annotations.json';
  a.click();
  URL.revokeObjectURL(url);
  document.getElementById('annotation-status').textContent = 'Saved ' + Object.keys(annots).length + ' annotations';
}}

function loadAnnotations(event) {{
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {{
    try {{
      const annots = JSON.parse(e.target.result);
      applyAnnotations(annots);
      localStorage.setItem('patch_annotations', JSON.stringify(annots));
      document.getElementById('annotation-status').textContent = 'Loaded ' + Object.keys(annots).length + ' annotations';
    }} catch(err) {{
      alert('Invalid JSON file');
    }}
  }};
  reader.readAsText(file);
}}

function applyAnnotations(annots) {{
  for (const [pairId, data] of Object.entries(annots)) {{
    const sel = document.querySelector('.annotation-select[data-pair-id="' + pairId + '"]');
    const comment = document.querySelector('.annotation-comment[data-pair-id="' + pairId + '"]');
    if (sel) sel.value = data.status || '';
    if (comment) comment.value = data.comment || '';
    updateBadge(pairId, data.status || '');
  }}
}}

// Restore from localStorage on load
(function() {{
  const saved = localStorage.getItem('patch_annotations');
  if (saved) {{
    try {{
      applyAnnotations(JSON.parse(saved));
    }} catch(e) {{}}
  }}
}})();
</script>
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
        print(f"  {label:20s}  mean={fmt(s.get('mean'))}  median={fmt(s.get('median'))}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
