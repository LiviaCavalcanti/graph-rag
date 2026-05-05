#!/usr/bin/env python3
"""
Embedder comparison dashboard.

Generates a self-contained HTML dashboard comparing embedders across:
  Tab 1: Hit@1, Hit@5, MRR by CWE, variance by CWE
  Tab 2: Code examples with per-embedder performance

Usage:
    python -m experiments.dashboard_scripts.comparison_dashboard [run_dir]
"""

from __future__ import annotations

import json
import math
import statistics
from html import escape
from pathlib import Path
from typing import Any

from experiments.dashboard_scripts._theme import PALETTE, THEME_CSS, THEME_JS


def _approach_color(approach_names: list[str], name: str) -> str:
    try:
        idx = approach_names.index(name) % len(PALETTE)
    except ValueError:
        idx = 0
    return PALETTE[idx]


def _esc(s: Any) -> str:
    return escape(str(s)) if s is not None else ""


def _pct(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.1f}%"


def _num(v: float | None, d: int = 3) -> str:
    if v is None:
        return "–"
    return f"{v:.{d}f}"


# ─────────────────────────────────────────────────────────────────────
#  Data extraction
# ─────────────────────────────────────────────────────────────────────

def _extract_per_cwe_metrics(cells: list[dict]) -> dict:
    """Extract per-CWE hit@1, hit@5, mrr for each embedder from raw_queries."""
    approach_names = [c["embedder"] for c in cells]
    # {cwe: {embedder: {hit1: [], hit5: [], mrr: []}}}
    cwe_metrics: dict[str, dict[str, dict[str, list[float]]]] = {}

    for cell in cells:
        emb = cell["embedder"]
        raw = cell["self_retrieval"].get("raw_queries", [])
        for q in raw:
            cwe = q.get("query_cwe", "Unknown")
            if cwe not in cwe_metrics:
                cwe_metrics[cwe] = {}
            if emb not in cwe_metrics[cwe]:
                cwe_metrics[cwe][emb] = {"hit1": [], "hit5": [], "mrr": []}
            # hit@1: mrr==1.0 means rank 1
            hit1 = 1.0 if q.get("hit", False) and q.get("mrr", 0) == 1.0 else 0.0
            # hit@5: check if any retrieved in top 5 matches
            retrieved = q.get("retrieved", [])
            query_cve = q.get("query_cve", "")
            hit5 = 1.0 if any(
                r.get("cve_id") == query_cve for r in retrieved[:5]
            ) else 0.0
            mrr_val = q.get("mrr", 0.0)
            cwe_metrics[cwe][emb]["hit1"].append(hit1)
            cwe_metrics[cwe][emb]["hit5"].append(hit5)
            cwe_metrics[cwe][emb]["mrr"].append(mrr_val)

    return cwe_metrics


def _compute_cwe_summary(cwe_metrics: dict) -> list[dict]:
    """Compute mean and variance per CWE per embedder."""
    rows = []
    for cwe, emb_data in sorted(cwe_metrics.items()):
        row: dict[str, Any] = {"cwe": cwe}
        for emb, metrics in emb_data.items():
            n = len(metrics["hit1"])
            row[f"{emb}_hit1"] = statistics.mean(metrics["hit1"]) if n else 0
            row[f"{emb}_hit5"] = statistics.mean(metrics["hit5"]) if n else 0
            row[f"{emb}_mrr"] = statistics.mean(metrics["mrr"]) if n else 0
            row[f"{emb}_n"] = n
            # Variance of MRR across queries within this CWE
            row[f"{emb}_mrr_var"] = statistics.variance(metrics["mrr"]) if n > 1 else 0
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────
#  Tab 1: Embedder Comparison
# ─────────────────────────────────────────────────────────────────────

def _tab_comparison(results: dict) -> str:
    cells = results.get("cells", [])
    approach_names = [c["embedder"] for c in cells]
    colors = {nm: _approach_color(approach_names, nm) for nm in approach_names}

    cwe_metrics = _extract_per_cwe_metrics(cells)
    cwe_summary = _compute_cwe_summary(cwe_metrics)

    # Overall metrics table
    overall_rows = ""
    for cell in sorted(cells, key=lambda c: c["self_retrieval"].get("mrr", 0), reverse=True):
        emb = cell["embedder"]
        sr = cell["self_retrieval"]
        col = colors[emb]
        overall_rows += f"""<tr>
          <td><span class="dot" style="background:{col}"></span>{_esc(emb)}</td>
          <td>{_pct(sr.get('hit@1'))}</td>
          <td>{_pct(sr.get('hit@5'))}</td>
          <td>{_pct(sr.get('hit@10'))}</td>
          <td>{_num(sr.get('mrr'))}</td>
          <td>{_pct(cell['cwe_recall'].get('macro_avg'))}</td>
        </tr>"""

    # Chart.js data: grouped bar per CWE
    cwe_labels = [r["cwe"] for r in cwe_summary]
    # Truncate long CWE names for chart
    cwe_labels_short = [c[:30] + "…" if len(c) > 30 else c for c in cwe_labels]

    # Datasets for hit@1 chart
    hit1_datasets = []
    for emb in approach_names:
        data = [r.get(f"{emb}_hit1", 0) for r in cwe_summary]
        hit1_datasets.append({
            "label": emb,
            "data": data,
            "backgroundColor": colors[emb] + "99",
            "borderColor": colors[emb],
            "borderWidth": 1,
        })

    hit5_datasets = []
    for emb in approach_names:
        data = [r.get(f"{emb}_hit5", 0) for r in cwe_summary]
        hit5_datasets.append({
            "label": emb,
            "data": data,
            "backgroundColor": colors[emb] + "99",
            "borderColor": colors[emb],
            "borderWidth": 1,
        })

    mrr_datasets = []
    for emb in approach_names:
        data = [r.get(f"{emb}_mrr", 0) for r in cwe_summary]
        mrr_datasets.append({
            "label": emb,
            "data": data,
            "backgroundColor": colors[emb] + "99",
            "borderColor": colors[emb],
            "borderWidth": 1,
        })

    # Variance chart (MRR variance by CWE)
    var_datasets = []
    for emb in approach_names:
        data = [r.get(f"{emb}_mrr_var", 0) for r in cwe_summary]
        var_datasets.append({
            "label": emb,
            "data": data,
            "backgroundColor": colors[emb] + "66",
            "borderColor": colors[emb],
            "borderWidth": 1,
        })

    # Per-CWE detail table
    per_cwe_rows = ""
    for r in cwe_summary:
        per_cwe_rows += f"<tr><td><strong>{_esc(r['cwe'])}</strong></td>"
        for emb in approach_names:
            h1 = r.get(f"{emb}_hit1", 0)
            h5 = r.get(f"{emb}_hit5", 0)
            mrr = r.get(f"{emb}_mrr", 0)
            var = r.get(f"{emb}_mrr_var", 0)
            n = r.get(f"{emb}_n", 0)
            per_cwe_rows += (
                f'<td style="text-align:center">'
                f'<span style="font-weight:600">{_pct(h1)}</span><br>'
                f'<span class="muted" style="font-size:.75rem">{_pct(h5)} | {_num(mrr,3)}</span><br>'
                f'<span class="muted" style="font-size:.7rem">σ²={_num(var,3)} n={n}</span>'
                f'</td>'
            )
        per_cwe_rows += "</tr>"

    header_cols = "".join(
        f'<th style="text-align:center"><span class="dot" style="background:{colors[e]}"></span>{_esc(e)}</th>'
        for e in approach_names
    )

    return f"""
<section id="tab-comparison" class="tab-panel active">
  <h2>Embedder Comparison</h2>
  <p class="sub">Hit@1, Hit@5, MRR across all embedders. Grouped by CWE category with variance analysis.</p>

  <div class="card">
    <h3>Overall Performance</h3>
    <table>
      <thead><tr><th>Embedder</th><th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th><th>CWE Recall</th></tr></thead>
      <tbody>{overall_rows}</tbody>
    </table>
  </div>

  <div class="card chart-card" style="min-height:350px">
    <h3>Hit@1 by CWE</h3>
    <canvas id="hit1ByCweChart" height="300"></canvas>
  </div>

  <div class="card chart-card" style="min-height:350px">
    <h3>Hit@5 by CWE</h3>
    <canvas id="hit5ByCweChart" height="300"></canvas>
  </div>

  <div class="card chart-card" style="min-height:350px">
    <h3>MRR by CWE</h3>
    <canvas id="mrrByCweChart" height="300"></canvas>
  </div>

  <div class="card chart-card" style="min-height:350px">
    <h3>MRR Variance by CWE (lower = more consistent)</h3>
    <canvas id="varByCweChart" height="300"></canvas>
  </div>

  <div class="card">
    <h3>Detailed Per-CWE Metrics</h3>
    <p class="sub">Each cell: <strong>Hit@1</strong> / Hit@5 | MRR / σ²(MRR) n=samples</p>
    <div class="scroll-x">
    <table>
      <thead><tr><th>CWE</th>{header_cols}</tr></thead>
      <tbody>{per_cwe_rows}</tbody>
    </table>
    </div>
  </div>

  <script>
  (function(){{
    const cweLabels = {json.dumps(cwe_labels_short)};
    const chartOpts = {{
      responsive: true,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }} }},
      scales: {{
        x: {{ ticks: {{ maxRotation: 90, minRotation: 45, font: {{ size: 10 }} }} }},
        y: {{ min: 0, max: 1 }}
      }}
    }};
    const varOpts = {{
      responsive: true,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }} }},
      scales: {{
        x: {{ ticks: {{ maxRotation: 90, minRotation: 45, font: {{ size: 10 }} }} }},
        y: {{ min: 0 }}
      }}
    }};

    new Chart(document.getElementById('hit1ByCweChart'), {{
      type: 'bar',
      data: {{ labels: cweLabels, datasets: {json.dumps(hit1_datasets)} }},
      options: chartOpts
    }});
    new Chart(document.getElementById('hit5ByCweChart'), {{
      type: 'bar',
      data: {{ labels: cweLabels, datasets: {json.dumps(hit5_datasets)} }},
      options: chartOpts
    }});
    new Chart(document.getElementById('mrrByCweChart'), {{
      type: 'bar',
      data: {{ labels: cweLabels, datasets: {json.dumps(mrr_datasets)} }},
      options: chartOpts
    }});
    new Chart(document.getElementById('varByCweChart'), {{
      type: 'bar',
      data: {{ labels: cweLabels, datasets: {json.dumps(var_datasets)} }},
      options: varOpts
    }});
  }})();
  </script>
</section>"""


# ─────────────────────────────────────────────────────────────────────
#  Tab 2: Code Examples
# ─────────────────────────────────────────────────────────────────────

def _tab_code_examples(results: dict) -> str:
    cells = results.get("cells", [])
    if not cells:
        return '<section id="tab-examples" class="tab-panel"><h2>Code Examples</h2><p class="muted">No data.</p></section>'

    approach_names = [c["embedder"] for c in cells]
    colors = {nm: _approach_color(approach_names, nm) for nm in approach_names}

    first_queries = cells[0]["self_retrieval"].get("raw_queries", [])

    # Build per-query map: {idx: {embedder: query_dict}}
    query_map: dict[int, dict[str, dict]] = {}
    for cell in cells:
        qs = cell["self_retrieval"].get("raw_queries", [])
        for i, q in enumerate(qs):
            if i not in query_map:
                query_map[i] = {}
            query_map[i][cell["embedder"]] = q

    # Build query cards as JSON
    query_cards: list[dict] = []
    for i, q_ref in enumerate(first_queries):
        card: dict[str, Any] = {
            "idx": i,
            "cve": q_ref.get("query_cve", ""),
            "cwe": q_ref.get("query_cwe", ""),
            "func": q_ref.get("query_func", ""),
            "variant": q_ref.get("query_variant", ""),
            "query_code": q_ref.get("query_code", ""),
            "approaches": {},
        }
        for nm in approach_names:
            q = query_map[i].get(nm, {})
            query_cve = q.get("query_cve", card["cve"])
            retrieved = q.get("retrieved", [])
            hit1 = 1.0 if q.get("mrr", 0) == 1.0 else 0.0
            hit5 = 1.0 if any(r.get("cve_id") == query_cve for r in retrieved[:5]) else 0.0
            card["approaches"][nm] = {
                "hit1": hit1 == 1.0,
                "hit5": hit5 == 1.0,
                "mrr": q.get("mrr", 0),
                "color": colors[nm],
                "retrieved": [
                    {
                        "rank": r.get("rank"),
                        "cve_id": r.get("cve_id", ""),
                        "cwe_id": r.get("cwe_id", ""),
                        "func_name": r.get("func_name", ""),
                        "score": r.get("score", 0),
                        "code": r.get("code", ""),
                        "is_correct": r.get("cve_id") == card["cve"],
                    }
                    for r in retrieved[:5]
                ],
            }
        query_cards.append(card)

    # Truncate code
    for card in query_cards:
        if len(card.get("query_code") or "") > 2500:
            card["query_code"] = card["query_code"][:2500] + "\n... [truncated]"
        for ap in card["approaches"].values():
            for r in ap.get("retrieved", []):
                if len(r.get("code") or "") > 1500:
                    r["code"] = r["code"][:1500] + "\n... [truncated]"

    # Build options
    options_html = ""
    for i, q in enumerate(first_queries):
        # Check status across all embedders
        statuses = []
        for nm in approach_names:
            m = query_map[i].get(nm, {})
            statuses.append(m.get("mrr", 0) == 1.0)
        all_hit = all(statuses)
        any_miss = not all_hit
        flag = " ⚠️" if any_miss else " ✓"
        options_html += f'<option value="{i}">{_esc(q["query_cve"])} — {_esc(q.get("query_func",""))} [{_esc(q.get("query_cwe",""))}]{flag}</option>\n'

    # Filter options by CWE
    all_cwes = sorted(set(q.get("query_cwe", "") for q in first_queries))
    cwe_filter_options = "".join(
        f'<option value="{_esc(c)}">{_esc(c)}</option>' for c in all_cwes
    )

    return f"""
<section id="tab-examples" class="tab-panel">
  <h2>Code Examples &amp; Per-Embedder Performance</h2>
  <p class="sub">Browse each query with source code and see how each embedder performed. ⚠️ = at least one embedder missed.</p>

  <div class="card">
    <label><strong>CWE Filter:</strong></label>
    <select id="cweFilter" class="query-select">
      <option value="">(all CWEs)</option>
      {cwe_filter_options}
    </select>
    <label style="margin-left:1rem"><strong>Query:</strong></label>
    <select id="querySelect2" class="query-select" style="max-width:500px">
      {options_html}
    </select>
    <label style="margin-left:1rem"><strong>Show only misses:</strong></label>
    <input type="checkbox" id="missOnly" style="margin-left:4px">
    <label style="margin-left:1rem"><strong>Top-k:</strong></label>
    <select id="topK2" class="query-select" style="width:70px">
      <option value="1">1</option>
      <option value="3">3</option>
      <option value="5" selected>5</option>
    </select>
  </div>

  <div id="codeExplorer2"></div>

  <script>
  (function(){{
    const CARDS   = {json.dumps(query_cards, ensure_ascii=False)};
    const COLORS  = {json.dumps(colors)};
    const NAMES   = {json.dumps(approach_names)};
    const sel     = document.getElementById('querySelect2');
    const cweF    = document.getElementById('cweFilter');
    const missF   = document.getElementById('missOnly');
    const topKSel = document.getElementById('topK2');
    const wrap    = document.getElementById('codeExplorer2');

    function esc(s) {{
      return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}

    function renderCard(idx) {{
      const card = CARDS[idx];
      if (!card) return;
      const topK = parseInt(topKSel.value)||5;

      // Performance summary bar
      let perfHtml = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">';
      for (const nm of NAMES) {{
        const ap = card.approaches[nm]||{{}};
        const col = COLORS[nm];
        const h1 = ap.hit1 ? '✓' : '✗';
        const h5 = ap.hit5 ? '✓' : '✗';
        const bg = ap.hit1 ? '#e8faf3' : '#fff0f3';
        const border = ap.hit1 ? '#06D6A0' : '#EF476F';
        perfHtml += `<div style="border:2px solid ${{border}};background:${{bg}};border-radius:8px;padding:6px 12px;font-size:.82rem">
          <span class="dot" style="background:${{col}}"></span>
          <strong>${{esc(nm)}}</strong>
          &nbsp; Hit@1:${{h1}} &nbsp; Hit@5:${{h5}} &nbsp; MRR:${{ap.mrr?.toFixed(3)||'–'}}
        </div>`;
      }}
      perfHtml += '</div>';

      let html = `<div class="code-card">
        <div class="code-card-header">
          <span class="pill-big">${{esc(card.cve)}}</span>
          <span class="pill">${{esc(card.cwe)}}</span>
          <span class="muted" style="margin-left:.5rem">func: ${{esc(card.func)}} &nbsp; variant: ${{esc(card.variant)}}</span>
        </div>
        ${{perfHtml}}
        <div class="code-split">
          <div class="code-pane">
            <div class="pane-label">📥 Query — vulnerable code</div>
            <pre class="code-block"><code>${{esc(card.query_code||'(no code available)')}}</code></pre>
          </div>
          <div class="code-results">`;

      for (const nm of NAMES) {{
        const ap = card.approaches[nm]||{{}};
        const col = COLORS[nm];
        const hitBadge = ap.hit1
          ? `<span class="badge" style="background:#06D6A0">✓ hit@1</span>`
          : ap.hit5
            ? `<span class="badge" style="background:#FB8500">✓ hit@5</span>`
            : `<span class="badge" style="background:#EF476F">✗ miss</span>`;
        html += `<div class="approach-block">
          <div class="approach-header" style="border-left:4px solid ${{col}}">
            <span class="dot" style="background:${{col}}"></span>
            <strong>${{esc(nm)}}</strong> &nbsp; ${{hitBadge}}
            &nbsp; <span class="muted">MRR=${{ap.mrr?.toFixed(3)||'–'}}</span>
          </div>`;
        for (const r of (ap.retrieved||[]).slice(0,topK)) {{
          const isCorrect = r.is_correct;
          const border = isCorrect ? '2px solid #06D6A0' : '1px solid #ddd';
          html += `<div class="retrieved-item" style="border:${{border}}">
            <div class="retrieved-meta">
              <span class="rank">#${{r.rank}}</span>
              <span class="${{isCorrect?'pill ok':'pill warn'}}">${{esc(r.cve_id)}}</span>
              <span class="pill">${{esc(r.cwe_id)}}</span>
              <span class="muted">${{esc(r.func_name||'')}}</span>
              <span class="muted">score ${{r.score?.toFixed(4)||'–'}}</span>
            </div>
            <pre class="code-block small"><code>${{esc(r.code||'(no code)')}}</code></pre>
          </div>`;
        }}
        html += `</div>`;
      }}
      html += `</div></div></div>`;
      wrap.innerHTML = html;
    }}

    function filterOptions() {{
      const cweVal = cweF.value;
      const missOnly = missF.checked;
      let anyVisible = false;
      for (let opt of sel.options) {{
        const idx = parseInt(opt.value);
        const card = CARDS[idx];
        let show = true;
        if (cweVal && card.cwe !== cweVal) show = false;
        if (missOnly) {{
          const allHit = NAMES.every(nm => (card.approaches[nm]||{{}}).hit1);
          if (allHit) show = false;
        }}
        opt.hidden = !show;
        if (show) anyVisible = true;
      }}
      const curOpt = sel.options[sel.selectedIndex];
      if (curOpt && curOpt.hidden) {{
        for (let opt of sel.options) {{
          if (!opt.hidden) {{ sel.value = opt.value; break; }}
        }}
      }}
      if (!anyVisible) {{
        wrap.innerHTML = '<div class="card"><p class="muted">No queries match the current filter.</p></div>';
        return;
      }}
      renderCard(parseInt(sel.value));
    }}

    sel.addEventListener('change', () => renderCard(parseInt(sel.value)));
    topKSel.addEventListener('change', () => renderCard(parseInt(sel.value)));
    cweF.addEventListener('change', filterOptions);
    missF.addEventListener('change', filterOptions);
    renderCard(0);
  }})();
  </script>
</section>"""


# ─────────────────────────────────────────────────────────────────────
#  HTML shell
# ─────────────────────────────────────────────────────────────────────

_CSS = THEME_CSS + """
.chart-card canvas { max-height: 400px; }
@media (max-width: 800px) { .code-split { grid-template-columns: 1fr; } }
"""


def generate_comparison_dashboard(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    results_path = run_dir / "results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")

    results = json.loads(results_path.read_text())
    run_id = results.get("run_id", run_dir.name)

    tabs = [
        ("tab-comparison", "📊 Embedder Comparison", _tab_comparison(results)),
        ("tab-examples", "💻 Code Examples", _tab_code_examples(results)),
    ]

    nav_html = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-tab="{tid}">{label}</button>'
        for i, (tid, label, _) in enumerate(tabs)
    )
    content_html = "\n".join(body for _, _, body in tabs)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Embedder Comparison — {_esc(run_id)}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>{_CSS}</style>
</head>
<body>

<div class="topbar">
  <h1>Embedder Comparison Dashboard</h1>
  <span class="run-id">{_esc(run_id)}</span>
</div>

<nav class="tab-nav">{nav_html}</nav>

<main class="content">
{content_html}
</main>

<script>
{THEME_JS}
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
}});
</script>
</body>
</html>"""

    out = run_dir / "comparison_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Comparison dashboard written → {out}")
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate embedder comparison dashboard.")
    parser.add_argument("run_dir", nargs="?", help="Path to run directory (default: latest)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        output_dir = Path("experiments/output")
        runs = sorted(
            [r for r in output_dir.iterdir() if (r / "results.json").exists()],
            key=lambda p: p.name,
        )
        if not runs:
            raise SystemExit("No runs found in experiments/output/")
        run_dir = runs[-1]

    generate_comparison_dashboard(run_dir)


if __name__ == "__main__":
    main()
