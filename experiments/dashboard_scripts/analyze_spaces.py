#!/usr/bin/env python3
"""
Dashboard for the Embedding Space Analysis experiment.

Reads:
  <run_dir>/results.json         — per-cell intrinsic metrics (from base class)
  <run_dir>/space_analysis.json  — pairwise matrices + trustworthiness

Writes:
  <run_dir>/space_dashboard.html — self-contained HTML dashboard

Tabs:
  1  Overview         — stat cards, isotropy/hubness summary
  2  Class Separation — intra/inter ratio, alignment/uniformity per embedder
  3  Pairwise CKA     — heatmap matrix
  4  k-NN Overlap     — heatmap matrix
  5  Rank Correlation  — heatmap matrix
  6  Trustworthiness  — PCA quality across k values

Usage:
    python -m experiments.dashboard_scripts.analyze_spaces \
        --run-dir experiments/output/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.dashboard_scripts._theme import (
    PALETTE,
    THEME_CSS,
    THEME_JS,
    heatmap_color,
    score_color,
)


# ── helpers ──────────────────────────────────────────────────────────


def _fmt(val, ndigits: int = 4) -> str:
    if val is None:
        return "–"
    if isinstance(val, float):
        return f"{val:.{ndigits}f}"
    return str(val)


def _pct(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.1f}%"


def _embedder_color(names: list[str], name: str) -> str:
    try:
        idx = names.index(name) % len(PALETTE)
    except ValueError:
        idx = 0
    return PALETTE[idx]


def _dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def _stat_card(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    cls = "stat-card accent" if accent else "stat-card"
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="{cls}">'
        f'<div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-value">{value}</div>'
        f'{sub_html}</div>'
    )


def _heatmap_table(matrix: dict[str, dict[str, float]], embedder_names: list[str],
                   low: float = 0.0, high: float = 1.0) -> str:
    """Render a symmetric matrix as a colored HTML table."""
    header = "<tr><th></th>" + "".join(
        f"<th>{escape(n)}</th>" for n in embedder_names
    ) + "</tr>\n"

    rows = ""
    for name_a in embedder_names:
        row_data = matrix.get(name_a, {})
        cells = ""
        for name_b in embedder_names:
            val = row_data.get(name_b)
            if val is None:
                cells += "<td>–</td>"
            elif name_a == name_b:
                cells += '<td style="background:#f0f0f0; color:#999">1.0</td>'
            else:
                bg = heatmap_color(val, low=low, high=high)
                cells += f'<td style="background:{bg}"><strong>{_fmt(val)}</strong></td>'
        rows += f"<tr><td><strong>{escape(name_a)}</strong></td>{cells}</tr>\n"

    return f'<div class="scroll-x"><table class="matrix">{header}{rows}</table></div>'


# ── tab renderers ────────────────────────────────────────────────────


def _tab_overview(cells: list[dict], config: dict, embedder_names: list[str]) -> str:
    n_embedders = len(embedder_names)
    n_samples = config.get("n_index", "?")
    n_cwe = config.get("n_cwe_classes", "?")
    dim = config.get("dim", 128)

    # stat cards row
    cards = (
        _stat_card("Embedders", str(n_embedders), accent=True)
        + _stat_card("Index Samples", str(n_samples))
        + _stat_card("CWE Classes", str(n_cwe))
        + _stat_card("Embedding Dim", str(dim))
    )

    # Per-embedder intrinsic summary table
    rows = ""
    for cell in cells:
        name = cell.get("coords", {}).get("embedder", "?")
        m = cell.get("metrics", {})
        intrinsic = m.get("intrinsic", {})
        stats = intrinsic.get("space_stats", {})
        iso = intrinsic.get("isotropy")
        hub = intrinsic.get("hubness", {})
        dc = intrinsic.get("distance_concentration", {})

        color = _embedder_color(embedder_names, name)
        rows += (
            f"<tr>"
            f"<td>{_dot(color)}<strong>{escape(name)}</strong></td>"
            f"<td>{intrinsic.get('dim', '?')}</td>"
            f"<td style='color:{score_color(iso or 0, high=0.3)}'>{_fmt(iso)}</td>"
            f"<td>{_fmt(hub.get('k_skewness'), 2)}</td>"
            f"<td>{_pct(hub.get('hub_fraction'))}</td>"
            f"<td>{_fmt(stats.get('mean_pairwise_sim'))}</td>"
            f"<td>{_fmt(stats.get('effective_dim'), 1)}</td>"
            f"<td>{_fmt(dc.get('relative_contrast_mean'), 3)}</td>"
            f"<td>{_fmt(intrinsic.get('embed_time_s'), 1)}s</td>"
            f"</tr>\n"
        )

    return f"""
    <div class="stat-row">{cards}</div>

    <h3>Per-Embedder Intrinsic Quality</h3>
    <p class="sub">Higher isotropy = more uniform. Low hubness skew = fewer false-positive hubs.
    Higher effective dim = richer space. Higher contrast = better discriminability.</p>
    <div class="scroll-x">
    <table>
      <tr>
        <th>Embedder</th><th>Dim</th><th>Isotropy</th>
        <th>Hub Skew</th><th>Hub Frac</th>
        <th>Mean Sim</th><th>Eff. Dim</th><th>Contrast</th><th>Time</th>
      </tr>
      {rows}
    </table>
    </div>
    """


def _tab_class_separation(cells: list[dict], embedder_names: list[str]) -> str:
    rows = ""
    for cell in cells:
        name = cell.get("coords", {}).get("embedder", "?")
        m = cell.get("metrics", {})
        sep = m.get("class_separation", {})
        au = m.get("alignment_uniformity", {})

        color = _embedder_color(embedder_names, name)
        ratio = sep.get("ratio")
        ratio_color = score_color(min(ratio, 2.0) / 2.0 if ratio and ratio > 0 else 0) if ratio else "#888"

        rows += (
            f"<tr>"
            f"<td>{_dot(color)}<strong>{escape(name)}</strong></td>"
            f"<td>{_fmt(sep.get('intra_mean'))}</td>"
            f"<td>{_fmt(sep.get('inter_mean'))}</td>"
            f"<td style='color:{ratio_color}'><strong>{_fmt(ratio, 3)}</strong></td>"
            f"<td>{_fmt(au.get('alignment'))}</td>"
            f"<td>{_fmt(au.get('uniformity'))}</td>"
            f"</tr>\n"
        )

    return f"""
    <h3>CWE Class Separation</h3>
    <p class="sub">
      <strong>Intra/Inter ratio &gt; 1</strong>: same-CWE embeddings are more similar than cross-CWE → good separation.<br>
      <strong>Alignment</strong> (lower = better): same-class pairs are close on the hypersphere.<br>
      <strong>Uniformity</strong> (lower = better): embeddings spread uniformly (no clustering collapse).
    </p>
    <div class="scroll-x">
    <table>
      <tr>
        <th>Embedder</th><th>Intra Sim</th><th>Inter Sim</th>
        <th>Ratio</th><th>Alignment ↓</th><th>Uniformity ↓</th>
      </tr>
      {rows}
    </table>
    </div>
    """


def _tab_matrix(title: str, description: str, matrix: dict, embedder_names: list[str],
                low: float = 0.0, high: float = 1.0) -> str:
    table = _heatmap_table(matrix, embedder_names, low=low, high=high)
    return f"""
    <h3>{escape(title)}</h3>
    <p class="sub">{description}</p>
    {table}
    """


def _tab_trustworthiness(trust: dict, hubness_sens: dict, embedder_names: list[str]) -> str:
    # Trustworthiness table
    if trust:
        trust_rows = ""
        for k_label, val in sorted(trust.items()):
            color = score_color(val, low=0.7, high=1.0)
            trust_rows += f"<tr><td>{escape(k_label)}</td><td style='color:{color}'><strong>{_fmt(val)}</strong></td></tr>\n"
        trust_html = f"""
        <h3>Combined PCA Trustworthiness</h3>
        <p class="sub">
          Measures whether PCA (384d → 128d) preserved local neighborhoods.
          1.0 = perfect preservation. Values below 0.9 indicate meaningful distortion.
        </p>
        <table style="max-width:300px">
          <tr><th>Neighborhood k</th><th>Score</th></tr>
          {trust_rows}
        </table>
        """
    else:
        trust_html = '<p class="muted">No trustworthiness data (Combined embedder not active or PCA not used).</p>'

    # Hubness sensitivity table
    if hubness_sens:
        hub_header = "<tr><th>Embedder</th>"
        k_labels = []
        first_name = next(iter(hubness_sens))
        for k_label in sorted(hubness_sens[first_name].keys()):
            k_labels.append(k_label)
            hub_header += f"<th>{escape(k_label)}<br><small>skew / hub%</small></th>"
        hub_header += "</tr>\n"

        hub_rows = ""
        for name in embedder_names:
            if name not in hubness_sens:
                continue
            color = _embedder_color(embedder_names, name)
            hub_rows += f"<tr><td>{_dot(color)}<strong>{escape(name)}</strong></td>"
            for k_label in k_labels:
                data = hubness_sens[name].get(k_label, {})
                skew = data.get("skewness", 0)
                frac = data.get("hub_fraction", 0)
                hub_rows += f"<td>{_fmt(skew, 2)} / {_pct(frac)}</td>"
            hub_rows += "</tr>\n"

        hub_html = f"""
        <h3 style="margin-top:28px">Hubness Sensitivity (across k)</h3>
        <p class="sub">
          Positive skewness = hub points exist. Hub fraction = % of points appearing as neighbor &gt; 2k times.
          Stable across k → structural property; increasing with k → mild concern.
        </p>
        <div class="scroll-x">
        <table>{hub_header}{hub_rows}</table>
        </div>
        """
    else:
        hub_html = ""

    return trust_html + hub_html


# ── main renderer ────────────────────────────────────────────────────


def render_dashboard(results: dict, space_analysis: dict) -> str:
    """Generate the full HTML dashboard string."""
    cells = results.get("cells", [])
    config = space_analysis.get("config", {})
    embedder_names = config.get("embedders", [])

    if not embedder_names:
        # Fallback: extract from results cells
        embedder_names = [c.get("coords", {}).get("embedder", "?") for c in cells]

    cka_matrix = space_analysis.get("pairwise_cka", {})
    knn_matrix = space_analysis.get("pairwise_knn_overlap", {})
    rank_matrix = space_analysis.get("pairwise_rank_correlation", {})
    trust = space_analysis.get("combined_trustworthiness", {})
    hubness_sens = space_analysis.get("hubness_sensitivity", {})

    # Render tabs
    tab_overview = _tab_overview(cells, config, embedder_names)
    tab_class = _tab_class_separation(cells, embedder_names)
    tab_cka = _tab_matrix(
        "Linear CKA Matrix",
        "Structural similarity between embedding spaces. "
        "High (→1) = spaces encode the same geometry. Low = different representations. "
        "Invariant to rotation and isotropic scaling.",
        cka_matrix, embedder_names, low=0.0, high=1.0,
    )
    tab_knn = _tab_matrix(
        "k-NN Overlap Matrix",
        "Fraction of shared k-nearest neighbors. "
        "High = same retrieval behavior (redundant). Low = complementary signal. "
        "Directly predicts whether fusion adds value.",
        knn_matrix, embedder_names, low=0.0, high=0.6,
    )
    tab_rank = _tab_matrix(
        "Rank Correlation Matrix (Spearman ρ)",
        "Correlation of pairwise distance rankings. "
        "High = one embedder's distance structure dominates the other. "
        "Low = fundamentally different similarity judgments.",
        rank_matrix, embedder_names, low=0.0, high=1.0,
    )
    tab_trust = _tab_trustworthiness(trust, hubness_sens, embedder_names)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Embedding Space Analysis</title>
<style>{THEME_CSS}
  .matrix td {{ text-align: center; padding: 10px 14px; font-family: var(--font-mono); font-size: .82rem; }}
  .matrix th {{ text-align: center; font-size: .72rem; padding: 8px 10px; }}
  .matrix td strong {{ font-weight: 600; }}
</style>
</head>
<body>

<div class="topbar">
  <h1>Embedding Space Analysis</h1>
  <span class="run-id">{len(embedder_names)} embedders · {config.get('n_index', '?')} samples</span>
</div>

<nav class="tab-nav">
  <button class="tab-btn active" data-tab="tab-overview">Overview</button>
  <button class="tab-btn" data-tab="tab-class">Class Separation</button>
  <button class="tab-btn" data-tab="tab-cka">CKA</button>
  <button class="tab-btn" data-tab="tab-knn">k-NN Overlap</button>
  <button class="tab-btn" data-tab="tab-rank">Rank Corr.</button>
  <button class="tab-btn" data-tab="tab-trust">Trustworthiness</button>
</nav>

<div class="content">
  <div id="tab-overview" class="tab-panel active">{tab_overview}</div>
  <div id="tab-class" class="tab-panel">{tab_class}</div>
  <div id="tab-cka" class="tab-panel">{tab_cka}</div>
  <div id="tab-knn" class="tab-panel">{tab_knn}</div>
  <div id="tab-rank" class="tab-panel">{tab_rank}</div>
  <div id="tab-trust" class="tab-panel">{tab_trust}</div>
</div>

<script>{THEME_JS}</script>
</body>
</html>"""


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML dashboard for embedding space analysis."
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to the experiment run directory",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output HTML path (default: <run_dir>/space_dashboard.html)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.json"
    space_path = run_dir / "space_analysis.json"

    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)
    if not space_path.exists():
        print(f"ERROR: {space_path} not found")
        sys.exit(1)

    results = json.loads(results_path.read_text())
    space_analysis = json.loads(space_path.read_text())

    out_path = Path(args.out) if args.out else run_dir / "space_dashboard.html"
    html = render_dashboard(results, space_analysis)
    out_path.write_text(html)
    print(f"Dashboard written → {out_path}")


if __name__ == "__main__":
    main()
