#!/usr/bin/env python3
"""
Standalone dashboard for Joern CPG-based patch verification results.

Displays:
  - Summary stat cards (accuracy @1/@5, avg fix-overlap, no-diff count)
  - By-CWE and by-variant breakdown tables
  - Per-record expandable cards with:
    - Verification status pill (green / red / grey)
    - Side-by-side coloured G_diff code lines (patch vs ground-truth)
    - Full code triple (input vulnerable / ground-truth fix / agent patch)

Inputs:
  - patch_verification.jsonl  (from src.evaluate.patch_verification)
  - results.jsonl             (optional, for generated-patch code)

Outputs:
  - patch_verification.html   (self-contained HTML dashboard)

Usage:
    python -m experiments.dashboard_scripts.patch_verification_dashboard \
        --verification experiments/output/<run>/patch_verification.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from statistics import mean

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.dashboard_scripts._theme import THEME_CSS, score_color as _score_color
from src.data.autopatch import AutoPatchDataset
from src.evaluate.preprocessing import extract_function_body

_find_cve_dir = AutoPatchDataset.find_cve_dir

# ── diff-category colours ────────────────────────────────────────────

_DIFF_COLORS: dict[str, tuple[str, str]] = {
    "removed":      ("#c62828", "#fff0f3"),
    "fix_adjacent": ("#e65100", "#fff8e1"),
    "edge_changed": ("#f9a825", "#fffde7"),
    "context":      ("#6b7280", "#f9fafb"),
}

# ── small helpers ────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _truncate(text: str | None, max_len: int = 800) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n… [truncated]"


def _fmt(val, ndigits: int = 4) -> str:
    if val is None:
        return "-"
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, float):
        return f"{val:.{ndigits}f}"
    return str(val)


def _load_input_code(cve_id: str, base_dir: Path) -> str | None:
    cve_dir = _find_cve_dir(cve_id, base_dir)
    if cve_dir is None:
        return None
    original = cve_dir / "original_code.txt"
    if original.exists():
        return original.read_text(errors="replace").strip()
    return None


def _load_ground_truth(cve_id: str, variant: str, base_dir: Path) -> str | None:
    cve_dir = _find_cve_dir(cve_id, base_dir)
    if cve_dir is None:
        return None
    gt_file = cve_dir / "out_v2" / "code" / f"{variant}_fixed.c"
    if gt_file.exists():
        raw = gt_file.read_text(errors="replace")
        return extract_function_body(raw).strip()
    return None


# ── verification-specific helpers ────────────────────────────────────


def _render_diff_block(summary: dict, title: str) -> str:
    """Render a diff summary block with coloured code lines."""
    if not summary:
        return (
            f'<div class="code-block-light"><h4>{escape(title)}</h4>'
            f'<pre style="color:var(--muted)"><em>No diff data</em></pre></div>'
        )
    lines_html = ""
    for cat in ("removed", "fix_adjacent", "edge_changed", "context"):
        code_lines = summary.get(cat, [])
        if not code_lines:
            continue
        color, bg = _DIFF_COLORS.get(cat, ("#6b7280", "#f9fafb"))
        meaningful = [
            l for l in code_lines
            if l.strip() and l not in ("<empty>", "<global>")
        ]
        if not meaningful:
            continue
        lines_html += (
            f'<div style="margin-bottom:4px;">'
            f'<span class="pill" style="background:{bg};color:{color};'
            f'font-size:11px;">{cat} ({len(meaningful)})</span></div>\n'
        )
        for line in meaningful[:12]:
            esc = escape(line[:200])
            lines_html += (
                f'<div style="border-left:3px solid {color};padding:1px 6px;'
                f'margin:1px 0;background:{bg};font-family:var(--font-mono);'
                f'font-size:12px;white-space:pre-wrap;overflow-x:auto;">'
                f'{esc}</div>\n'
            )
        if len(meaningful) > 12:
            lines_html += (
                f'<div style="color:var(--muted);font-size:11px;'
                f'padding-left:10px;">… +{len(meaningful) - 12} more</div>\n'
            )
    return (
        f'<div class="code-block-light"><h4>{escape(title)}</h4>'
        f'<div style="max-height:400px;overflow-y:auto;padding:8px;">'
        f'{lines_html}</div></div>'
    )


def _compute_overlap(patch_summary: dict, gt_summary: dict) -> float:
    """Fraction of GT removed code lines also present in patch diff."""
    gt_removed = {
        l for l in gt_summary.get("removed", [])
        if l.strip() and l not in ("<empty>", "<global>")
    }
    if not gt_removed:
        return 0.0
    patch_all: set[str] = set()
    for lines in patch_summary.values():
        patch_all.update(lines)
    return len(gt_removed & patch_all) / len(gt_removed)


def _verification_pill(status: str | None, same_cve: bool | None) -> str:
    if not status:
        return ""
    if status == "no_diff":
        return (
            '<span class="pill" style="background:#f3f4f6;'
            'color:#6b7280;">NO DIFF (identical to vuln)</span>'
        )
    if status not in ("verified",):
        return (
            f'<span class="pill" style="background:#f3f4f6;'
            f'color:#6b7280;">{escape(status)}</span>'
        )
    if same_cve:
        return '<span class="pill ok">&check; SAME CVE (fix verified)</span>'
    return '<span class="pill warn">&cross; DIFFERENT (vuln persists)</span>'


def _sparkbar(value: float, width: int = 80) -> str:
    pct = max(0.0, min(1.0, value))
    fill_w = int(pct * width)
    color = "#146c43" if pct >= 0.5 else "#f9a825" if pct >= 0.2 else "#c62828"
    return (
        f'<span style="display:inline-block;width:{width}px;height:10px;'
        f'background:#e5e7eb;border-radius:5px;vertical-align:middle;">'
        f'<span style="display:block;width:{fill_w}px;height:10px;'
        f'background:{color};border-radius:5px;"></span></span>'
        f' <span style="font-size:11px;color:var(--muted);">{pct:.0%}</span>'
    )


# ── data loading & aggregation ───────────────────────────────────────


def _load_data(
    verification_path: Path,
    results_path: Path | None,
    base_dir: Path,
) -> list[dict]:
    """Load verification records and enrich with code triple."""
    verif_records = _load_jsonl(verification_path)

    # Optional: load generated patches from results.jsonl
    gen_index: dict[tuple[str, str], str] = {}
    if results_path and results_path.exists():
        for r in _load_jsonl(results_path):
            key = (r.get("query_cve", ""), r.get("query_variant", ""))
            gen_index[key] = (r.get("generated_patch") or "").strip()

    merged: list[dict] = []
    for vr in verif_records:
        cve = vr.get("query_cve", "")
        variant = vr.get("query_variant", "")
        key = (cve, variant)

        rec = dict(vr)  # keep all verification fields
        rec["input_code"] = _load_input_code(cve, base_dir)
        rec["ground_truth"] = _load_ground_truth(cve, variant, base_dir)
        rec["generated_patch"] = gen_index.get(key, "")
        rec["overlap"] = _compute_overlap(
            vr.get("patch_diff_summary", {}),
            vr.get("gt_vuln_summary", {}),
        )
        merged.append(rec)

    return merged


def _compute_summary(records: list[dict]) -> dict:
    n_total = len(records)
    by_status: dict[str, int] = {}
    for r in records:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    verified = [r for r in records if r.get("status") == "verified"]
    n_v = len(verified)
    n_top1 = sum(1 for r in verified if r.get("same_cve_at_top1"))
    n_topk = sum(1 for r in verified if r.get("same_cve_in_topk"))
    overlaps = [r["overlap"] for r in verified]

    # By-CWE
    by_cwe: dict[str, list[dict]] = {}
    for r in records:
        by_cwe.setdefault(r.get("query_cwe", "unknown"), []).append(r)
    cwe_breakdown: dict[str, dict] = {}
    for cwe, recs in sorted(by_cwe.items()):
        vf = [r for r in recs if r.get("status") == "verified"]
        n_cwe_top1 = sum(1 for r in vf if r.get("same_cve_at_top1"))
        cwe_breakdown[cwe] = {
            "count": len(recs),
            "n_verified": len(vf),
            "accuracy_top1": n_cwe_top1 / len(vf) if vf else 0.0,
            "avg_overlap": mean([r["overlap"] for r in vf]) if vf else 0.0,
        }

    # By-variant
    by_var: dict[str, list[dict]] = {}
    for r in records:
        by_var.setdefault(r.get("query_variant", "unknown"), []).append(r)
    variant_breakdown: dict[str, dict] = {}
    for var, recs in sorted(by_var.items()):
        vf = [r for r in recs if r.get("status") == "verified"]
        n_var_top1 = sum(1 for r in vf if r.get("same_cve_at_top1"))
        variant_breakdown[var] = {
            "count": len(recs),
            "n_verified": len(vf),
            "accuracy_top1": n_var_top1 / len(vf) if vf else 0.0,
            "avg_overlap": mean([r["overlap"] for r in vf]) if vf else 0.0,
        }

    return {
        "n_total": n_total,
        "n_verified": n_v,
        "n_no_diff": by_status.get("no_diff", 0),
        "n_no_patch": by_status.get("no_patch", 0),
        "n_joern_failed": by_status.get("joern_failed", 0),
        "n_error": by_status.get("error", 0) + by_status.get("embedding_error", 0),
        "accuracy_top1": n_top1 / n_v if n_v else 0.0,
        "accuracy_topk": n_topk / n_v if n_v else 0.0,
        "n_top1": n_top1,
        "n_topk": n_topk,
        "avg_overlap": mean(overlaps) if overlaps else 0.0,
        "by_cwe": cwe_breakdown,
        "by_variant": variant_breakdown,
    }


# ── HTML rendering ───────────────────────────────────────────────────


def _render_html(
    records: list[dict],
    summary: dict,
    source_path: str,
) -> str:
    s = summary

    # ── stat cards ───────────────────────────────────────────────
    acc1_cls = "warn" if s["accuracy_top1"] < 0.1 else ""
    stat_cards = f"""
    <div class="stat-row">
      <div class="stat-card {acc1_cls}">
        <div class="stat-value">{s['accuracy_top1']:.1%}</div>
        <div class="stat-label">Patch Accuracy @1</div>
        <div class="stat-sub">{s['n_top1']}/{s['n_verified']} verified</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{s['accuracy_topk']:.1%}</div>
        <div class="stat-label">Patch Accuracy @5</div>
        <div class="stat-sub">{s['n_topk']}/{s['n_verified']} verified</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{s['avg_overlap']:.1%}</div>
        <div class="stat-label">Avg Fix Overlap</div>
        <div class="stat-sub">GT removed lines in patch diff</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{s['n_no_diff']}</div>
        <div class="stat-label">No Structural Diff</div>
        <div class="stat-sub">Patch identical to vuln code</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{s['n_total']}</div>
        <div class="stat-label">Total Records</div>
        <div class="stat-sub">{s['n_verified']} verified, {s['n_no_patch']} no-patch, {s['n_joern_failed']} joern-failed</div>
      </div>
    </div>
    """

    # ── by-CWE table ─────────────────────────────────────────────
    cwe_rows = ""
    for cwe, cs in sorted(s["by_cwe"].items()):
        cwe_rows += (
            f"<tr><td>{escape(str(cwe))}</td>"
            f"<td>{cs['count']}</td>"
            f"<td>{cs['n_verified']}</td>"
            f"<td style='color:{_score_color(cs['accuracy_top1'])}'>"
            f"<strong>{cs['accuracy_top1']:.1%}</strong></td>"
            f"<td>{cs['avg_overlap']:.1%}</td></tr>\n"
        )

    # ── by-variant table ─────────────────────────────────────────
    var_rows = ""
    for var, vs in sorted(s["by_variant"].items()):
        var_rows += (
            f"<tr><td>{escape(var)}</td>"
            f"<td>{vs['count']}</td>"
            f"<td>{vs['n_verified']}</td>"
            f"<td style='color:{_score_color(vs['accuracy_top1'])}'>"
            f"<strong>{vs['accuracy_top1']:.1%}</strong></td>"
            f"<td>{vs['avg_overlap']:.1%}</td></tr>\n"
        )

    # ── per-record cards ─────────────────────────────────────────
    record_cards = ""
    for i, rec in enumerate(records, 1):
        status = rec.get("status", "unknown")
        same_top1 = rec.get("same_cve_at_top1")
        pill = _verification_pill(status, same_top1)
        overlap = rec.get("overlap", 0.0)

        # Verification metadata line
        verif_meta = (
            f"Status: <strong>{escape(status)}</strong> | "
            f"Top-1: <strong>{escape(str(rec.get('top1_cve', '-')))}</strong> "
            f"(score={_fmt(rec.get('top1_score'))}) | "
            f"G_diff: {rec.get('g_diff_nodes', '-')} nodes | "
            f"G_gen: {rec.get('g_generated_nodes', '-')} nodes | "
            f"Fix overlap: {_sparkbar(overlap)}"
        )

        # Top-k retrieval table
        retrieved = rec.get("retrieved", [])
        topk_rows = ""
        for hit in retrieved[:5]:
            is_match = hit.get("cve_id", "") == rec.get("query_cve", "")
            match_style = "color:var(--ok);font-weight:600;" if is_match else ""
            topk_rows += (
                f"<tr style='{match_style}'>"
                f"<td>#{hit.get('rank', '-')}</td>"
                f"<td>{escape(str(hit.get('cve_id', '-')))}</td>"
                f"<td>{escape(str(hit.get('variant', '-')))}</td>"
                f"<td>{_fmt(hit.get('score'))}</td></tr>\n"
            )
        topk_html = ""
        if topk_rows:
            topk_html = (
                '<table style="width:auto;max-width:500px;margin:8px 0;">'
                '<tr><th>Rank</th><th>CVE</th><th>Variant</th><th>Score</th></tr>'
                f'{topk_rows}</table>'
            )

        # Side-by-side diff blocks
        patch_ds = rec.get("patch_diff_summary", {})
        gt_ds = rec.get("gt_vuln_summary", {})
        diff_section = ""
        if status in ("verified", "no_diff"):
            diff_section = (
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;">'
                f'{_render_diff_block(patch_ds, "Patch Changed (G_diff of generated)")}'
                f'{_render_diff_block(gt_ds, "GT Fix Changed (G_vuln)")}'
                '</div>'
            )

        # Code triple
        code_triple = f"""
        <details style="margin-top:12px;">
          <summary style="cursor:pointer;font-weight:600;color:var(--accent);">
            Code Triple (Input / Ground Truth / Generated)
          </summary>
          <div class="code-triple" style="margin-top:8px;">
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
        </details>
        """

        record_cards += f"""
        <details class="card" {'open' if i <= 3 else ''}>
          <summary>
            <span style="color:var(--muted);font-weight:normal;">#{i}</span>
            <strong>{escape(rec.get('query_cve', ''))}</strong> /
            {escape(rec.get('query_variant', ''))}
            &nbsp; {pill}
            &nbsp; <span style="color:var(--muted);">CWE: {escape(str(rec.get('query_cwe', '')))}</span>
          </summary>
          <div class="card-body">
            <div class="retrieval-info">{verif_meta}</div>
            {topk_html}
            {diff_section}
            {code_triple}
          </div>
        </details>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Patch Verification Dashboard (Joern CPG Diff)</title>
<style>{THEME_CSS}
</style>
</head>
<body>
<div class="content">
<div class="page-header" style="padding:0">
  <h1>Patch Verification Dashboard</h1>
  <p class="meta">Joern CPG structural diff analysis &mdash; {s['n_total']} records from
    <code>{escape(source_path)}</code></p>
</div>

<h2>Summary</h2>
{stat_cards}

<h2>By CWE Type</h2>
<table>
  <tr><th>CWE</th><th>Count</th><th>Verified</th><th>Accuracy @1</th><th>Avg Overlap</th></tr>
  {cwe_rows}
</table>

<h2>By Variant</h2>
<table>
  <tr><th>Variant</th><th>Count</th><th>Verified</th><th>Accuracy @1</th><th>Avg Overlap</th></tr>
  {var_rows}
</table>

<h2>Per-Record Details</h2>
<p class="sub">Click to expand. First 3 open by default.</p>
{record_cards}
</div>
</body>
</html>"""


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Standalone patch verification dashboard (Joern CPG diff).",
    )
    parser.add_argument(
        "--verification", required=True,
        help="Path to patch_verification.jsonl",
    )
    parser.add_argument(
        "--results", default=None,
        help="Path to results.jsonl for generated-patch code (auto-detect in same dir)",
    )
    parser.add_argument(
        "--base-dir", default=None,
        help="Base directory for CVE-list (default: cwd)",
    )
    parser.add_argument(
        "--out-html", default=None,
        help="Output HTML path (default: <run_dir>/patch_verification.html)",
    )
    args = parser.parse_args()

    verif_path = Path(args.verification)
    if not verif_path.exists():
        print(f"ERROR: {verif_path} not found")
        sys.exit(1)

    results_path = (
        Path(args.results) if args.results
        else verif_path.parent / "results.jsonl"
    )
    if results_path.exists():
        print(f"Results:      {results_path}")
    else:
        results_path = None
        print("Results:      not found (code triple will be incomplete)")

    base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()
    out_html = (
        Path(args.out_html) if args.out_html
        else verif_path.parent / "patch_verification.html"
    )

    print(f"Verification: {verif_path}")
    print(f"Base dir:     {base_dir}")

    records = _load_data(verif_path, results_path, base_dir)
    summary = _compute_summary(records)

    out_html.write_text(_render_html(records, summary, str(verif_path)))
    print(f"HTML:         {out_html}")

    # CLI summary
    s = summary
    print(f"\n{'═' * 60}")
    print(f"  PATCH VERIFICATION  ({s['n_total']} records)")
    print(f"{'═' * 60}")
    print(f"  Verified:         {s['n_verified']}")
    print(f"  No diff:          {s['n_no_diff']}")
    print(f"  No patch:         {s['n_no_patch']}")
    print(f"  Joern failed:     {s['n_joern_failed']}")
    print(f"  Errors:           {s['n_error']}")
    print(f"  ─────────────────────────────────")
    print(f"  Accuracy @1:      {s['accuracy_top1']:.1%}  ({s['n_top1']}/{s['n_verified']})")
    print(f"  Accuracy @5:      {s['accuracy_topk']:.1%}  ({s['n_topk']}/{s['n_verified']})")
    print(f"  Avg Fix Overlap:  {s['avg_overlap']:.1%}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
