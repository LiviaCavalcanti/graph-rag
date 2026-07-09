#!/usr/bin/env python3
"""
Unified multi-tab HTML dashboard for graph-RAG experiments.

Reads:
  <run_dir>/results.json          — raw retrieval results
  <run_dir>/miss_analysis.json    — uncertainty / miss breakdown  (optional)
  <run_dir>/crossing_analysis.json— fusion strategy comparison   (optional)

Writes:
  <run_dir>/dashboard.html        — single self-contained HTML file

Layout (5 tabs):
  1  Overview          — leaderboard, dataset info, key numbers
  2  Retrieval         — Hit@k, MRR, per-CWE recall, latency
  3  Miss Analysis     — uncertainty, rank histograms, per-cell details
  4  Crossing Strategies — fusion strategy comparison, pairwise complementarity
  5  Code Explorer     — per-query code input + top-k retrieved code
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────
#  Colour palette  (single source of truth)
# ─────────────────────────────────────────────────────────────────────
# We use a fixed assignment so that "combined" is always the same
# colour across all tabs and charts.

_PALETTE = [
    "#4361EE",   # 0  blue        → combined
    "#F72585",   # 1  pink        → codebert_seq
    "#7209B7",   # 2  violet      → codebert_pattern
    "#FB8500",   # 3  orange      → vuln_pattern
    "#06D6A0",   # 4  teal        → rgcn
    "#118AB2",   # 5  steel-blue  → extra
    "#FFD166",   # 6  yellow
    "#EF476F",   # 7  coral
]
_CSS_PALETTE_VARS = "\n".join(
    f"  --c{i}: {v};" for i, v in enumerate(_PALETTE)
)


def _approach_color(approach_names: list[str], name: str) -> str:
    try:
        idx = approach_names.index(name) % len(_PALETTE)
    except ValueError:
        idx = 0
    return _PALETTE[idx]


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.1f}%"


def _num(v: float | None, d: int = 3) -> str:
    if v is None:
        return "–"
    return f"{v:.{d}f}"


def _esc(s: Any) -> str:
    return escape(str(s)) if s is not None else ""


def _badge(text: str, color: str = "var(--accent)") -> str:
    return (
        f'<span class="badge" style="background:{color};color:#fff">'
        f"{_esc(text)}</span>"
    )


def _sparkbar(value: float, color: str = "var(--accent)", max_w: int = 120) -> str:
    w = max(2, int(value * max_w))
    return (
        f'<span class="sparkbar-wrap">'
        f'<span class="sparkbar" style="width:{w}px;background:{color}"></span>'
        f'<span class="sparkbar-label">{_pct(value)}</span>'
        f'</span>'
    )


def _code_block(code: str | None, lang: str = "c") -> str:
    if not code or not code.strip():
        return '<pre class="code-block muted">(no source code available)</pre>'
    return f'<pre class="code-block"><code class="lang-{lang}">{_esc(code)}</code></pre>'


# ─────────────────────────────────────────────────────────────────────
#  Tab 1: Overview
# ─────────────────────────────────────────────────────────────────────

def _cve_block(cell: dict) -> dict:
    """Same-CVE retrieval metrics; prefers new 'cve_retrieval', falls back to legacy 'self_retrieval'."""
    return cell.get("cve_retrieval") or cell.get("self_retrieval") or {}


def _tab_overview(results: dict, miss: dict | None, crossing: dict | None) -> str:
    cells = results.get("cells", [])
    approach_names = list(dict.fromkeys(c["embedder"] for c in cells))
    dataset = results.get("dataset_info", {})
    g = (miss or {}).get("global", {})

    # Leaderboard rows sorted by hit@1
    leaderboard = sorted(
        [
            {
                "name": c["embedder"],
                "hit1": _cve_block(c).get("hit@1", 0),
                "hit5": _cve_block(c).get("hit@5", 0),
                "hit10": _cve_block(c).get("hit@10", 0),
                "mrr": _cve_block(c).get("mrr", 0),
                "cwe_rec": c["cwe_recall"].get("macro_avg", 0),
                "lat": c.get("query_latency", {}).get("p50_ms", 0),
                "n": _cve_block(c).get("n", 0),
            }
            for c in cells
        ],
        key=lambda r: r["hit1"],
        reverse=True,
    )

    rows_html = ""
    for rank, row in enumerate(leaderboard, 1):
        col = _approach_color(approach_names, row["name"])
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
        rows_html += f"""
        <tr>
          <td>{medal} {rank}</td>
          <td><span class="dot" style="background:{col}"></span>{_esc(row['name'])}</td>
          <td>{_sparkbar(row['hit1'], col)}</td>
          <td>{_sparkbar(row['hit5'], col)}</td>
          <td>{_sparkbar(row['hit10'], col)}</td>
          <td>{_num(row['mrr'])}</td>
          <td>{_pct(row['cwe_rec'])}</td>
          <td>{_num(row['lat'], 2)} ms</td>
          <td>{row['n']}</td>
        </tr>"""

    # Stat cards
    best = leaderboard[0] if leaderboard else {}
    oracle_rate = ""
    if crossing:
        for s in crossing.get("strategy_summary", []):
            if s["strategy"] == "oracle":
                oracle_rate = _pct(s["rate"])
    cascade_rate = ""
    if crossing:
        for s in crossing.get("strategy_summary", []):
            if s["strategy"] == "fallback_cascade":
                cascade_rate = _pct(s["rate"])

    split = dataset.get("split", {})
    split_html = ""
    if split.get("enabled"):
        counts = split.get("counts", {})
        split_html = f"""
        <div class="stat-card">
          <div class="stat-label">Dataset split</div>
          <div class="stat-value" style="font-size:1rem">
            index&nbsp;<strong>{counts.get('index_total','?')}</strong> &nbsp;|&nbsp;
            query&nbsp;<strong>{counts.get('query_total','?')}</strong>
          </div>
          <div class="stat-sub">
            real {counts.get('real_total','?')} &nbsp;+&nbsp;
            aug_train {counts.get('aug_train_used','?')} &nbsp;/&nbsp;
            aug_test {counts.get('aug_test_total','?')}
          </div>
        </div>"""

    return f"""
<section id="tab-overview" class="tab-panel active">
  <h2>Experiment Overview</h2>
  <p class="sub">Run <code>{_esc(results.get('run_id','?'))}</code>
     &nbsp;·&nbsp; {_esc(results.get('timestamp','')[:19].replace('T',' '))} UTC
  </p>

  <div class="stat-row">
    <div class="stat-card accent">
      <div class="stat-label">Best Hit@1</div>
      <div class="stat-value">{_pct(best.get('hit1'))}</div>
      <div class="stat-sub">{_esc(best.get('name',''))}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Best MRR</div>
      <div class="stat-value">{_num(best.get('mrr'))}</div>
      <div class="stat-sub">{_esc(best.get('name',''))}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Oracle ceiling</div>
      <div class="stat-value">{oracle_rate or '–'}</div>
      <div class="stat-sub">any approach hits</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Fallback cascade</div>
      <div class="stat-value">{cascade_rate or '–'}</div>
      <div class="stat-sub">best practical fusion</div>
    </div>
    {split_html}
  </div>

  <div class="card">
    <h3>Leaderboard</h3>
    <table>
      <thead><tr>
        <th>#</th><th>Approach</th>
        <th>Hit@1</th><th>Hit@5</th><th>Hit@10</th>
        <th>MRR</th><th>CWE Rec@10</th>
        <th>Latency p50</th><th>Queries</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <div class="card">
    <h3>Dataset</h3>
    <ul>
      <li>Total pairs: <strong>{dataset.get('n_pairs','?')}</strong></li>
      <li>Index pairs: <strong>{dataset.get('n_index_pairs','?')}</strong></li>
      <li>Query pairs: <strong>{dataset.get('n_query_pairs','?')}</strong></li>
      <li>CVE families: <strong>{len(dataset.get('cwe_ids',[]))}</strong> CWE types</li>
    </ul>
  </div>
</section>"""


# ─────────────────────────────────────────────────────────────────────
#  Tab 2: Retrieval Performance
# ─────────────────────────────────────────────────────────────────────

def _tab_retrieval(results: dict) -> str:
    cells = results.get("cells", [])
    approach_names = list(dict.fromkeys(c["embedder"] for c in cells))

    # Build data for Chart.js grouped bar chart (hit@1/5/10/MRR)
    labels_js = json.dumps([c["embedder"] for c in cells])
    hit1_js   = json.dumps([_cve_block(c).get("hit@1", 0) for c in cells])
    hit5_js   = json.dumps([_cve_block(c).get("hit@5", 0) for c in cells])
    hit10_js  = json.dumps([_cve_block(c).get("hit@10", 0) for c in cells])
    mrr_js    = json.dumps([_cve_block(c).get("mrr", 0) for c in cells])
    colors_js = json.dumps([_approach_color(approach_names, c["embedder"]) for c in cells])

    # CWE recall macro
    cwe_mac_js = json.dumps([c["cwe_recall"].get("macro_avg", 0) for c in cells])

    # Per-CWE recall table for best cell
    best_cell = max(cells, key=lambda c: _cve_block(c).get("mrr", 0)) if cells else None
    per_cwe_html = ""
    if best_cell:
        per_cwe = best_cell.get("cwe_recall", {}).get("per_cwe", {})
        rows = sorted(per_cwe.items(), key=lambda kv: kv[1]["recall"])
        per_cwe_html = "".join(
            f'<tr>'
            f'<td>{_esc(cwe)}</td>'
            f'<td>{_sparkbar(v["recall"], _approach_color(approach_names, best_cell["embedder"]))}</td>'
            f'<td>{v["support"]}</td>'
            f'</tr>'
            for cwe, v in rows
        )

    # Latency table
    lat_rows = "".join(
        f'<tr>'
        f'<td><span class="dot" style="background:{_approach_color(approach_names, c["embedder"])}"></span>'
        f'{_esc(c["embedder"])}</td>'
        f'<td>{_num(c.get("query_latency", {}).get("p50_ms"), 2)}</td>'
        f'<td>{_num(c.get("query_latency", {}).get("p95_ms"), 2)}</td>'
        f'<td>{_num(c.get("query_latency", {}).get("p99_ms"), 2)}</td>'
        f'<td>{_num(c.get("embed_time_s", {}), 1)} s</td>'
        f'</tr>'
        for c in cells
    )

    return f"""
<section id="tab-retrieval" class="tab-panel">
  <h2>Retrieval Performance</h2>

  <div class="chart-grid">
    <div class="card chart-card">
      <h3>Hit@k &amp; MRR</h3>
      <canvas id="hitChart" height="180"></canvas>
    </div>
    <div class="card chart-card">
      <h3>CWE Group Recall@10</h3>
      <canvas id="cweChart" height="180"></canvas>
    </div>
  </div>

  <div class="card">
    <h3>Per-CWE Recall — best approach: <em>{_esc(best_cell['embedder'] if best_cell else '')}</em></h3>
    <table>
      <thead><tr><th>CWE type</th><th>Recall@10</th><th>Support</th></tr></thead>
      <tbody>{per_cwe_html}</tbody>
    </table>
  </div>

  <div class="card">
    <h3>Query Latency</h3>
    <table>
      <thead><tr><th>Approach</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th><th>Embed time</th></tr></thead>
      <tbody>{lat_rows}</tbody>
    </table>
  </div>

  <script>
  (function(){{
    const labels  = {labels_js};
    const colors  = {colors_js};
    const alpha   = (hex) => hex + "99";

    new Chart(document.getElementById('hitChart'), {{
      type: 'bar',
      data: {{
        labels,
        datasets: [
          {{ label:'Hit@1',  data:{hit1_js},  backgroundColor: colors }},
          {{ label:'Hit@5',  data:{hit5_js},  backgroundColor: colors.map(alpha) }},
          {{ label:'Hit@10', data:{hit10_js}, backgroundColor: colors.map(c=>c+'44') }},
          {{ label:'MRR',    data:{mrr_js},   type:'line', borderColor: '#333', pointBackgroundColor: colors, fill: false, tension:.3 }},
        ]
      }},
      options: {{
        responsive:true,
        scales:{{ y:{{ min:0, max:1, ticks:{{ format:'~p' }} }} }},
        plugins:{{ legend:{{ position:'bottom' }} }}
      }}
    }});

    new Chart(document.getElementById('cweChart'), {{
      type:'bar',
      data:{{ labels, datasets:[{{ label:'CWE Macro Recall@10', data:{cwe_mac_js}, backgroundColor:colors }}] }},
      options:{{ responsive:true, scales:{{ y:{{ min:0, max:1 }} }}, plugins:{{ legend:{{ display:false }} }} }}
    }});
  }})();
  </script>
</section>"""


# ─────────────────────────────────────────────────────────────────────
#  Tab: Threshold Study & Abstention Error Analysis
# ─────────────────────────────────────────────────────────────────────
#  Consumes threshold_analysis.json (built by src.metrics.threshold_analysis
#  via experiments.dashboard_scripts.analyze_thresholds).  Framing: at a
#  similarity threshold t, a query is "has class" (predicted) when the top-1
#  score >= t, else "no class" (abstained).  See that module for the full
#  confusion-matrix convention.

def _thr_metrics_from_ea(ea: dict) -> dict:
    """Derive coverage / selective-accuracy / recall / F1 from an error blob."""
    tp = ea["has_class"]["correct"]
    fp = ea["has_class"]["wrong"]
    fn = ea["no_class"]["fn"]
    tn = ea["no_class"]["tn"]
    n = ea.get("n", tp + fp + fn + tn)
    predicted = tp + fp
    return {
        "threshold": ea.get("threshold"),
        "coverage": predicted / n if n else 0.0,
        "sel_acc": tp / predicted if predicted else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "accuracy": (tp + tn) / n if n else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "predicted": predicted,
    }


def _thr_line_chart_js(canvas_id: str, rows: list[dict], col: str) -> str:
    thr = json.dumps([r["threshold"] for r in rows])
    cov = json.dumps([r["coverage"] for r in rows])
    sel = json.dumps([r["selective_accuracy"] for r in rows])
    rec = json.dumps([r["recall"] for r in rows])
    f1v = json.dumps([r["f1"] for r in rows])
    tpl = """
    new Chart(document.getElementById('__CANVAS__'), {
      type:'line',
      data:{ labels:__THR__, datasets:[
        { label:'Coverage',      data:__COV__, borderColor:'__COL__', backgroundColor:'__COL__', tension:.2, pointRadius:2, borderWidth:2 },
        { label:'Selective Acc', data:__SEL__, borderColor:'#146c43', tension:.2, pointRadius:2, borderWidth:2 },
        { label:'Recall',        data:__REC__, borderColor:'#c62828', borderDash:[5,4], tension:.2, pointRadius:1, borderWidth:1.5 },
        { label:'F1',            data:__F1__,  borderColor:'#1a1a2e', tension:.2, pointRadius:1, borderWidth:1.5 }
      ]},
      options:{ responsive:true, interaction:{mode:'index',intersect:false},
        scales:{ y:{min:0,max:1,ticks:{format:'~p'}},
                 x:{title:{display:true,text:'similarity threshold'}} },
        plugins:{ legend:{position:'bottom'} } }
    });"""
    return (tpl.replace("__CANVAS__", canvas_id).replace("__THR__", thr)
            .replace("__COV__", cov).replace("__SEL__", sel)
            .replace("__REC__", rec).replace("__F1__", f1v).replace("__COL__", col))


def _thr_cr_chart_js(canvas_id: str, cwe_rows: list[dict], cve_rows: list[dict], col: str) -> str:
    def _pts(rows):
        return json.dumps(
            [{"x": r["coverage"], "y": r["selective_accuracy"]} for r in rows]
        )
    tpl = """
    new Chart(document.getElementById('__CANVAS__'), {
      type:'scatter',
      data:{ datasets:[
        { label:'CWE', data:__CWE__, showLine:true, borderColor:'__COL__', backgroundColor:'__COL__', pointRadius:2 },
        { label:'CVE', data:__CVE__, showLine:true, borderColor:'#7209B7', backgroundColor:'#7209B7', borderDash:[5,4], pointRadius:2 }
      ]},
      options:{ responsive:true,
        scales:{ x:{min:0,max:1,title:{display:true,text:'coverage'},ticks:{format:'~p'}},
                 y:{min:0,max:1,title:{display:true,text:'selective accuracy'},ticks:{format:'~p'}} },
        plugins:{ legend:{position:'bottom'} } }
    });"""
    return (tpl.replace("__CANVAS__", canvas_id)
            .replace("__CWE__", _pts(cwe_rows)).replace("__CVE__", _pts(cve_rows))
            .replace("__COL__", col))


def _thr_markers_table(cell: dict) -> str:
    body = ""
    for label in ("p50", "p75", "p90"):
        t = cell["markers"].get(label)
        if t is None:
            continue
        ea_cwe = cell["error_analysis"]["cwe"].get(label)
        ea_cve = cell["error_analysis"]["cve"].get(label)
        body += f'<tr><td><span class="pill">{label}</span> {_num(t, 3)}</td>'
        for ea in (ea_cwe, ea_cve):
            if ea:
                m = _thr_metrics_from_ea(ea)
                body += (f"<td>{_pct(m['coverage'])}</td><td>{_pct(m['sel_acc'])}</td>"
                         f"<td>{_pct(m['recall'])}</td><td>{_pct(m['f1'])}</td>")
            else:
                body += "<td>–</td><td>–</td><td>–</td><td>–</td>"
        body += "</tr>"
    return f"""<table>
      <thead>
        <tr><th rowspan="2">Marker</th><th colspan="4">CWE axis</th><th colspan="4">CVE axis</th></tr>
        <tr><th>Cov</th><th>Sel.Acc</th><th>Rec</th><th>F1</th><th>Cov</th><th>Sel.Acc</th><th>Rec</th><th>F1</th></tr>
      </thead>
      <tbody>{body}</tbody></table>"""


def _thr_sweep_table(rows: list[dict], axis_label: str, marker_ts: set) -> str:
    body = ""
    for r in rows:
        hl = ' class="thr-marker"' if round(r["threshold"], 4) in marker_ts else ""
        body += (
            f"<tr{hl}><td>{r['threshold']:.2f}</td>"
            f"<td>{_pct(r['coverage'])}</td><td>{_pct(r['selective_accuracy'])}</td>"
            f"<td>{_pct(r['recall'])}</td><td>{_pct(r['f1'])}</td><td>{_pct(r['accuracy'])}</td>"
            f"<td>{r['tp']}</td><td>{r['fp']}</td><td>{r['fn']}</td><td>{r['tn']}</td></tr>"
        )
    return f"""<h4>{axis_label} axis</h4>
      <div class="scroll-x"><table><thead><tr>
        <th>Thr</th><th>Cov</th><th>Sel.Acc</th><th>Rec</th><th>F1</th><th>Acc</th>
        <th>TP</th><th>FP</th><th>FN</th><th>TN</th>
      </tr></thead><tbody>{body}</tbody></table></div>"""


def _thr_group_cards(m: dict, ea: dict) -> str:
    has = ea["has_class"]
    no = ea["no_class"]
    recoverable = no.get("tn_recoverable_in_topk", 0)
    return f"""
    <div class="stat-row">
      <div class="stat-card accent">
        <div class="stat-label">Has class · predicted</div>
        <div class="stat-value">{has['n']}</div>
        <div class="stat-sub">{has['correct']} correct · {has['wrong']} wrong ({_pct(m['sel_acc'])} sel-acc)</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">No class · abstained</div>
        <div class="stat-value">{no['n']}</div>
        <div class="stat-sub">{no['fn']} missed (FN) · {no['tn']} correct-abstain (TN)</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Coverage</div>
        <div class="stat-value">{_pct(m['coverage'])}</div>
        <div class="stat-sub">threshold {_num(m['threshold'], 3)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Recoverable abstains</div>
        <div class="stat-value">{recoverable}</div>
        <div class="stat-sub">TN with correct CWE deeper in top-k</div>
      </div>
    </div>"""


def _thr_by_cwe_table(ea: dict) -> str:
    has = ea["has_class"].get("by_true_cwe", {})
    no = ea["no_class"].get("by_true_cwe", {})
    cwes = set(has) | set(no)
    rows = []
    for cwe in cwes:
        h = has.get(cwe, {})
        nn = no.get(cwe, {})
        correct = h.get("correct", 0)
        wrong = h.get("wrong", 0)
        fn = nn.get("fn", 0)
        tn = nn.get("tn", 0)
        rows.append((correct + wrong + fn + tn, cwe, correct, wrong, fn, tn))
    rows.sort(reverse=True)
    body = "".join(
        f"<tr><td>{_esc(cwe)}</td><td>{correct}</td><td>{wrong}</td>"
        f"<td>{fn}</td><td>{tn}</td><td>{tot}</td></tr>"
        for tot, cwe, correct, wrong, fn, tn in rows
    )
    if not body:
        return "<p class='muted'>No queries in this group.</p>"
    return f"""<div class="scroll-x"><table><thead><tr>
        <th>CWE (true)</th><th>Pred ✓</th><th>Pred ✗</th><th>Abstain FN</th><th>Abstain TN</th><th>Total</th>
      </tr></thead><tbody>{body}</tbody></table></div>"""


def _thr_confusion(ea: dict) -> str:
    conf = ea["has_class"].get("wrong_cwe_confusion", {})
    if not conf:
        return "<p class='muted'>No mispredicted CWE pairs at this threshold.</p>"
    body = "".join(
        f"<tr><td>{_esc(k)}</td><td>{v}</td></tr>" for k, v in conf.items()
    )
    return f"""<div class="scroll-x"><table><thead><tr>
        <th>true → predicted CWE</th><th>count</th>
      </tr></thead><tbody>{body}</tbody></table></div>"""


def _thr_recoverable(ea: dict) -> str:
    hist = ea["no_class"].get("tn_recoverable_rank_hist", {})
    if not hist:
        return ""
    parts = " ".join(
        f'<span class="pill">rank {_esc(k)}: {v}</span>' for k, v in hist.items()
    )
    return (
        '<p style="margin-top:8px"><strong>Correctly-abstained (TN) with a right-CWE '
        f"neighbour deeper in top-k:</strong> {parts}</p>"
    )


def _thr_cve_summary(ea_cve: dict, m_cve: dict) -> str:
    has = ea_cve["has_class"]
    no = ea_cve["no_class"]
    soft = has.get("soft_errors_wrong_cve_right_cwe", 0)
    return f"""<p>
      <span class="pill ok">has class {has['n']}</span>
      <span class="pill">correct {has['correct']}</span>
      <span class="pill warn">wrong {has['wrong']}</span>
      <span class="pill">soft errors (wrong CVE / right CWE) {soft}</span>
      &nbsp;·&nbsp;
      <span class="pill">no class {no['n']}</span>
      <span class="pill warn">FN {no['fn']}</span>
      <span class="pill ok">TN {no['tn']}</span>
      &nbsp;·&nbsp; coverage {_pct(m_cve['coverage'])}, sel-acc {_pct(m_cve['sel_acc'])}
    </p>"""


def _nearest_grid(grid: list[float], value: float | None) -> float | None:
    if not grid or value is None:
        return None
    return round(min(grid, key=lambda g: abs(g - value)), 4)


def _tab_threshold(threshold: dict | None, results: dict) -> str:
    if not threshold:
        return """<section id="tab-threshold" class="tab-panel">
          <h2>Threshold Study</h2>
          <p class="muted">Run <code>uv run python -m experiments.dashboard_scripts.analyze_thresholds
          --results &lt;results.json&gt;</code> to generate threshold_analysis.json first.</p></section>"""

    cells = threshold.get("cells", [])
    grid = threshold.get("grid", [])
    approach_names = list(dict.fromkeys(c["embedder"] for c in results.get("cells", [])))

    intro = (
        "Only <em>accept</em> a retrieval when the top-1 similarity is at or above a "
        "threshold <code>t</code>. A query is then <strong>has class</strong> (we assign the "
        "top-1 CWE/CVE) or <strong>no class</strong> (we abstain). Sweeping <code>t</code> "
        "trades coverage for selective accuracy. The <strong>CWE</strong> axis scores a "
        "class-match; the <strong>CVE</strong> axis scores an exact-instance match. Percentile "
        "markers are per-embedder (raw scores are not comparable across embedders), and no "
        "single threshold is recommended — inspect the curves and pick per your precision/coverage target."
    )

    card_blocks = []
    chart_js = []
    for i, cell in enumerate(cells):
        name = cell["embedder"]
        col = _approach_color(approach_names, name)
        cwe_rows = cell["sweep"]["cwe"]
        cve_rows = cell["sweep"]["cve"]
        c1, c2, c3 = f"thrCwe{i}", f"thrCve{i}", f"thrCR{i}"
        chart_js.append(_thr_line_chart_js(c1, cwe_rows, col))
        chart_js.append(_thr_line_chart_js(c2, cve_rows, col))
        chart_js.append(_thr_cr_chart_js(c3, cwe_rows, cve_rows, col))

        marker_ts = {_nearest_grid(grid, cell["markers"].get(l)) for l in ("p50", "p75", "p90")}
        marker_ts = {t for t in marker_ts if t is not None}

        # Primary error analysis at p75 (fall back to first available marker)
        cwe_ea_map = cell["error_analysis"]["cwe"]
        prim = "p75" if "p75" in cwe_ea_map else (next(iter(cwe_ea_map), None))
        err_html = ""
        if prim:
            ea_cwe = cwe_ea_map[prim]
            m_cwe = _thr_metrics_from_ea(ea_cwe)
            ea_cve = cell["error_analysis"]["cve"].get(prim)
            m_cve = _thr_metrics_from_ea(ea_cve) if ea_cve else None
            cve_block = _thr_cve_summary(ea_cve, m_cve) if ea_cve else ""
            err_html = f"""
          <h4>Error analysis @ {prim} (threshold {_num(cell['markers'].get(prim), 3)}) — CWE axis</h4>
          {_thr_group_cards(m_cwe, ea_cwe)}
          <div class="chart-grid">
            <div><h4>Group breakdown by true CWE</h4>{_thr_by_cwe_table(ea_cwe)}</div>
            <div><h4>Mispredicted CWE pairs (predicted &amp; wrong)</h4>{_thr_confusion(ea_cwe)}</div>
          </div>
          {_thr_recoverable(ea_cwe)}
          <h4>CVE axis summary @ {prim}</h4>
          {cve_block}"""

        other_html = ""
        for label in ("p50", "p90"):
            ea = cwe_ea_map.get(label)
            if not ea:
                continue
            m = _thr_metrics_from_ea(ea)
            other_html += (
                f"<details><summary>Error analysis @ {label} "
                f"(threshold {_num(cell['markers'].get(label), 3)}) — CWE axis</summary>"
                f"{_thr_group_cards(m, ea)}{_thr_by_cwe_table(ea)}{_thr_confusion(ea)}</details>"
            )

        sweep_details = (
            "<details><summary>Full threshold sweep table (all grid points)</summary>"
            f"{_thr_sweep_table(cwe_rows, 'CWE', marker_ts)}"
            f"{_thr_sweep_table(cve_rows, 'CVE', marker_ts)}</details>"
        )

        ss = cell.get("score_stats", {})
        card_blocks.append(f"""
        <div class="card">
          <div class="cell-header">
            <span class="dot" style="background:{col}"></span>
            <strong>{_esc(name)}</strong> · {_esc(cell.get('backend'))} · {_esc(cell.get('graph_variant'))}
            &nbsp; <span class="pill">{cell.get('n_queries', 0)} queries</span>
            <span class="pill">{cell.get('n_queries_cwe', 0)} with CWE</span>
          </div>
          <p class="muted">Top-1 score range {_num(ss.get('min'), 3)}–{_num(ss.get('max'), 3)},
             median {_num(ss.get('median'), 3)}.</p>
          <div class="chart-grid">
            <div class="chart-card"><h4>CWE axis — metrics vs threshold</h4><canvas id="{c1}" height="200"></canvas></div>
            <div class="chart-card"><h4>CVE axis — metrics vs threshold</h4><canvas id="{c2}" height="200"></canvas></div>
          </div>
          <div class="chart-grid">
            <div class="chart-card"><h4>Coverage vs selective accuracy</h4><canvas id="{c3}" height="200"></canvas></div>
            <div>
              <h4>Reference markers (non-prescriptive)</h4>
              {_thr_markers_table(cell)}
              <p class="muted" style="margin-top:8px">F1-max grid threshold:
                 CWE {_num(cell['markers'].get('f1max_cwe'), 2)},
                 CVE {_num(cell['markers'].get('f1max_cve'), 2)};
                 random-baseline {_num(cell['markers'].get('random'), 3)}.</p>
            </div>
          </div>
          {err_html}
          {other_html}
          {sweep_details}
        </div>""")

    style_block = "<style>.thr-marker td{background:#fff8e1;}</style>"
    script_block = "<script>(function(){" + "".join(chart_js) + "})();</script>"
    return (
        '<section id="tab-threshold" class="tab-panel">'
        "<h2>Threshold Study &amp; Abstention Error Analysis</h2>"
        f'<p class="sub">{intro}</p>'
        + style_block
        + "".join(card_blocks)
        + script_block
        + "</section>"
    )


# ─────────────────────────────────────────────────────────────────────
#  Tab 3: Miss Analysis
# ─────────────────────────────────────────────────────────────────────

def _tab_miss(miss: dict | None, results: dict) -> str:
    if not miss:
        return """<section id="tab-miss" class="tab-panel">
          <h2>Miss Analysis</h2>
          <p class="muted">Run <code>uv run python -m experiments.analyze_misses --results &lt;results.json&gt;</code>
          to generate miss_analysis.json first.</p></section>"""

    cells = miss.get("cells", [])
    approach_names = list(dict.fromkeys(c["cell"]["embedder"] for c in cells))
    g = miss.get("global", {})

    # Global stat cards
    global_html = f"""
    <div class="stat-row">
      <div class="stat-card accent">
        <div class="stat-label">Total misses</div>
        <div class="stat-value">{g.get('n_top1_misses','?')}</div>
        <div class="stat-sub">/ {g.get('n_queries','?')} queries</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Uncertain misses</div>
        <div class="stat-value">{_pct(g.get('rate_wrong_uncertain_over_wrong'))}</div>
        <div class="stat-sub">system knows it's unsure</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Miss → CWE correct (top-1)</div>
        <div class="stat-value">{_pct(g.get('rate_miss_top1_cwe_correct_over_misses'))}</div>
        <div class="stat-sub">wrong CVE, right vulnerability class</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Miss → CWE in top-k</div>
        <div class="stat-value">{_pct(g.get('rate_miss_any_cwe_correct_over_misses'))}</div>
        <div class="stat-sub">correct class somewhere in results</div>
      </div>
    </div>"""

    # Per-cell rank histogram data for Chart.js
    hist_charts = ""
    for idx, cell in enumerate(cells):
        name = cell["cell"]["embedder"]
        col  = _approach_color(approach_names, name)
        acr  = cell.get("actual_cve_rank_when_missed", {})
        hist = acr.get("rank_histogram", {})
        # fill ranks 2-10
        hist_labels = list(range(2, 11))
        hist_data   = [hist.get(str(r), 0) for r in hist_labels]

        cnt = cell.get("counts", {})
        unc = cell.get("uncertainty", {})

        detail_rows = ""
        for ex in cell.get("examples", {}).get("wrong_uncertain", [])[:10]:
            detail_rows += (
                f'<tr>'
                f'<td>{_esc(ex.get("query_cve"))}</td>'
                f'<td><span class="pill">{_esc(ex.get("query_cwe"))}</span></td>'
                f'<td>{_esc(ex.get("top1_cve"))}</td>'
                f'<td><span class="pill warn">{_esc(ex.get("top1_cwe"))}</span></td>'
                f'<td>{_num(ex.get("top1_score"),4)}</td>'
                f'<td>{_num(ex.get("score_margin"),4)}</td>'
                f'<td>{_num(ex.get("top1_prob"),4)}</td>'
                f'<td>{ex.get("actual_cve_rank") or "–"}</td>'
                f'</tr>\n'
            )

        hist_charts += f"""
        <div class="card">
          <div class="cell-header">
            <span class="dot" style="background:{col}"></span>
            <strong>{_esc(name)}</strong>
            &nbsp;·&nbsp;{_esc(cell['cell'].get('backend'))}
            &nbsp;·&nbsp;{_esc(cell['cell'].get('graph_variant'))}
          </div>
          <div class="miss-grid">
            <div>
              <h4>Miss Profile</h4>
              <ul>
                <li>Hit@1: <strong>{_pct(cnt.get('top1_cve_hit_rate'))}</strong></li>
                <li>Misses: <strong>{cnt.get('n_top1_cve_miss','?')}</strong> / {cnt.get('n_queries','?')}</li>
                <li>Miss + top1 CWE correct: {cnt.get('n_miss_but_top1_cwe_correct','?')} ({_pct(cnt.get('rate_miss_but_top1_cwe_correct_over_misses'))})</li>
                <li>Miss + any CWE in top-k: {cnt.get('n_miss_but_any_topk_cwe_correct','?')} ({_pct(cnt.get('rate_miss_but_any_topk_cwe_correct_over_misses'))})</li>
                <li>True CVE found (when missed): {_pct(acr.get('found_rate'))} (median rank {_num((acr.get('rank_stats') or {{}}).get('median'), 1)})</li>
              </ul>
              <h4>Uncertainty</h4>
              <ul>
                <li>Wrong &amp; uncertain: {(unc.get('wrong_and_uncertain') or {{}}).get('count','?')} ({_pct((unc.get('wrong_and_uncertain') or {{}}).get('rate_over_wrong'))})</li>
                <li>Wrong &amp; confident: {(unc.get('wrong_and_confident') or {{}}).get('count','?')} ({_pct((unc.get('wrong_and_confident') or {{}}).get('rate_over_wrong'))})</li>
              </ul>
            </div>
            <div>
              <canvas id="hist_{idx}" height="150"></canvas>
            </div>
          </div>

          <details>
            <summary>Wrong + Uncertain examples ({len((cell.get('examples') or {{}}).get('wrong_uncertain', [])[:10])})</summary>
            <div class="scroll-x">
            <table>
              <thead><tr>
                <th>Query CVE</th><th>Query CWE</th>
                <th>Top-1 CVE</th><th>Top-1 CWE</th>
                <th>Score</th><th>Margin</th><th>Prob</th><th>True rank</th>
              </tr></thead>
              <tbody>{detail_rows or '<tr><td colspan=8 class="muted">No examples</td></tr>'}</tbody>
            </table>
            </div>
          </details>
        </div>
        <script>
        (function(){{
          new Chart(document.getElementById('hist_{idx}'), {{
            type:'bar',
            data:{{
              labels: {json.dumps(hist_labels)},
              datasets:[{{ label:'True CVE rank (misses)', data:{json.dumps(hist_data)},
                backgroundColor:'{col}88', borderColor:'{col}', borderWidth:1 }}]
            }},
            options:{{ responsive:true, plugins:{{ legend:{{ display:false }},
              title:{{ display:true, text:'True CVE rank distribution (when missed)' }} }},
              scales:{{ y:{{ title:{{ display:true, text:'count' }} }},
                        x:{{ title:{{ display:true, text:'rank' }} }} }} }}
          }});
        }})();
        </script>"""

    return f"""
<section id="tab-miss" class="tab-panel">
  <h2>Miss &amp; Uncertainty Analysis</h2>
  <p class="sub">When the system gets it wrong, does it know? Lower confidence misses = system is honest.</p>
  {global_html}
  {hist_charts}
</section>"""


# ─────────────────────────────────────────────────────────────────────
#  Tab 4: Crossing Strategies
# ─────────────────────────────────────────────────────────────────────

def _tab_crossing(crossing: dict | None, results: dict) -> str:
    if not crossing:
        return """<section id="tab-crossing" class="tab-panel">
          <h2>Crossing Strategies</h2>
          <p class="muted">Run <code>uv run python experiments/verify_crossing.py</code>
          to generate crossing_analysis.json first.</p></section>"""

    approach_names = crossing.get("approaches", [])
    strategy_rows = crossing.get("strategy_summary", [])
    approach_rows = crossing.get("approach_summary", [])

    # Find best individual for delta colouring
    best_ind = max((r["rate"] for r in approach_rows), default=0)

    strat_names_js = json.dumps([r["strategy"] for r in strategy_rows])
    strat_rates_js = json.dumps([r["rate"] for r in strategy_rows])
    strat_colors_js = json.dumps([
        "#4361EE" if r["strategy"] == "individual_best"
        else "#06D6A0" if r["strategy"] == "fallback_cascade"
        else "#F72585" if r["strategy"] == "oracle"
        else "#FB8500"
        for r in strategy_rows
    ])

    # Strategy rows HTML
    strat_html = ""
    for r in sorted(strategy_rows, key=lambda x: x["rate"], reverse=True):
        delta = r["rate"] - best_ind
        delta_str = (
            f'<span class="delta up">+{delta:.1%}</span>'
            if delta > 0.001 else
            f'<span class="delta down">{delta:.1%}</span>'
            if delta < -0.001 else
            '<span class="delta">baseline</span>'
        )
        strat_html += (
            f'<tr><td><strong>{_esc(r["strategy"])}</strong></td>'
            f'<td>{r["hit_at_1"]}/{r["total"]}</td>'
            f'<td>{_sparkbar(r["rate"])}</td>'
            f'<td>{delta_str}</td></tr>\n'
        )

    # Per-approach rows
    app_html = "".join(
        f'<tr>'
        f'<td><span class="dot" style="background:{_approach_color(approach_names, r["approach"])}"></span>'
        f'{_esc(r["approach"])}</td>'
        f'<td>{r["hit_at_1"]}/{r["total"]}</td>'
        f'<td>{_sparkbar(r["rate"], _approach_color(approach_names, r["approach"]))}</td>'
        f'</tr>\n'
        for r in sorted(approach_rows, key=lambda x: x["rate"], reverse=True)
    )

    # Combined miss deep-dive
    dd = crossing.get("combined_miss_deep_dive", {})
    miss_rows = ""
    for m in dd.get("misses", []):
        rescuers = ", ".join(m.get("rescued_by_approaches") or []) or "–"
        strats = ", ".join(s for s, hit in m.get("strategy_hits", {}).items() if hit and s != "individual_best") or "–"
        rank = m.get("combined_true_rank")
        rank_str = f"rank {rank}" if rank else "not in top-k"
        miss_rows += (
            f'<tr>'
            f'<td>{_esc(m["query_cve"])}</td>'
            f'<td><span class="pill">{_esc(m.get("query_cwe",""))}</span></td>'
            f'<td>{rank_str}</td>'
            f'<td>{"✓" if m.get("combined_uncertain") else "✗"}</td>'
            f'<td>{_esc(rescuers)}</td>'
            f'<td class="small">{_esc(strats)}</td>'
            f'</tr>\n'
        )

    # Pairwise complementarity matrix
    pair = crossing.get("pairwise_complementarity", {})
    pw_header = "".join(f'<th>{_esc(b)}</th>' for b in approach_names)
    pw_rows = ""
    for a in approach_names:
        col_a = _approach_color(approach_names, a)
        pw_rows += f'<tr><td><span class="dot" style="background:{col_a}"></span>{_esc(a)}</td>'
        for b in approach_names:
            info = pair.get(f"{a}+{b}", {})
            rate = info.get("union_rate", 0)
            only_b = info.get("only_b", 0)
            extra = f'<span class="delta up">+{only_b}</span>' if only_b > 0 else ""
            bg = f"rgba(67,97,238,{rate*0.5:.2f})"
            pw_rows += f'<td style="background:{bg}">{_pct(rate)} {extra}</td>'
        pw_rows += "</tr>\n"

    cascade_order_str = " → ".join(_esc(n) for n in (crossing.get("settings") or {}).get("cascade_order", []))

    return f"""
<section id="tab-crossing" class="tab-panel">
  <h2>Crossing Strategies</h2>
  <p class="sub">Can combining multiple approaches rescue the uncertainty-driven misses?
    Cascade order: <code>{cascade_order_str}</code></p>

  <div class="chart-grid">
    <div class="card chart-card">
      <h3>Fusion Strategy Hit@1</h3>
      <canvas id="stratChart" height="200"></canvas>
    </div>
    <div class="card">
      <h3>Individual Approaches</h3>
      <table><thead><tr><th>Approach</th><th>Hits</th><th>Hit@1</th></tr></thead>
        <tbody>{app_html}</tbody></table>
    </div>
  </div>

  <div class="card">
    <h3>Strategy Comparison</h3>
    <table>
      <thead><tr><th>Strategy</th><th>Hits</th><th>Hit@1</th><th>vs best individual</th></tr></thead>
      <tbody>{strat_html}</tbody>
    </table>
  </div>

  <div class="card">
    <h3>Pairwise Complementarity — Union Hit@1</h3>
    <p class="sub">Row = query approach, Column = added approach. +N = extra queries rescued.</p>
    <div class="scroll-x">
    <table class="matrix">
      <thead><tr><th></th>{pw_header}</tr></thead>
      <tbody>{pw_rows}</tbody>
    </table>
    </div>
  </div>

  <div class="card">
    <h3>Combined-Approach Miss Deep-Dive</h3>
    <div class="stat-row compact">
      <div class="stat-card"><div class="stat-label">Total misses</div>
        <div class="stat-value">{dd.get('n_misses','?')}</div></div>
      <div class="stat-card"><div class="stat-label">All uncertain</div>
        <div class="stat-value">{dd.get('n_all_uncertain','?')}</div></div>
      <div class="stat-card"><div class="stat-label">Rescued (any)</div>
        <div class="stat-value">{dd.get('n_rescued_by_any_approach','?')}</div></div>
      <div class="stat-card accent"><div class="stat-label">Rescued (cascade)</div>
        <div class="stat-value">{dd.get('n_rescued_by_cascade','?')}</div></div>
    </div>
    <div class="scroll-x">
    <table>
      <thead><tr>
        <th>Query CVE</th><th>CWE</th><th>True rank</th>
        <th>Uncertain?</th><th>Rescued by</th><th>Strategies that fix it</th>
      </tr></thead>
      <tbody>{miss_rows or '<tr><td colspan=6 class="muted">No misses.</td></tr>'}</tbody>
    </table>
    </div>
  </div>

  <script>
  (function(){{
    new Chart(document.getElementById('stratChart'), {{
      type:'bar',
      data:{{
        labels: {strat_names_js},
        datasets:[{{ data:{strat_rates_js}, backgroundColor:{strat_colors_js} }}]
      }},
      options:{{
        indexAxis:'y', responsive:true,
        plugins:{{ legend:{{ display:false }} }},
        scales:{{ x:{{ min:0, max:1 }} }}
      }}
    }});
  }})();
  </script>
</section>"""


# ─────────────────────────────────────────────────────────────────────
#  Tab 5: Code Explorer
# ─────────────────────────────────────────────────────────────────────

def _tab_code(results: dict, crossing: dict | None = None) -> str:
    cells = results.get("cells", [])
    approach_names = list(dict.fromkeys(c["embedder"] for c in cells))
    # per_query_detail from crossing_analysis: keyed by query_idx for O(1) lookup
    crossing_detail_by_idx: dict[int, dict] = {
        row["query_idx"]: row
        for row in (crossing or {}).get("per_query_detail", [])
        if "query_idx" in row
    }
    strategy_names: list[str] = [
        r["strategy"]
        for r in (crossing or {}).get("strategy_summary", [])
        if r.get("strategy")
    ]

    # Collect all unique queries (same order across cells)
    # We use the first cell's query list as the canonical order
    if not cells:
        return '<section id="tab-code" class="tab-panel"><h2>Code Explorer</h2><p class="muted">No data.</p></section>'

    first_queries = _cve_block(cells[0]).get("raw_queries", [])

    # Build per-query index: {query_idx: {approach_name: query_dict}}
    query_map: dict[int, dict[str, dict]] = {}
    for cell in cells:
        qs = _cve_block(cell).get("raw_queries", [])
        for i, q in enumerate(qs):
            if i not in query_map:
                query_map[i] = {}
            query_map[i][cell["embedder"]] = q

    # Build options HTML for query selector
    options_html = ""
    for i, q in enumerate(first_queries):
        all_hit  = all(query_map[i].get(nm, {}).get("mrr", 0) == 1.0 for nm in approach_names)
        any_miss = any(query_map[i].get(nm, {}).get("mrr", 0) < 1.0 for nm in approach_names)
        flag = " ⚠️" if any_miss else ""
        options_html += f'<option value="{i}">{_esc(q["query_cve"])} — {_esc(q.get("query_cwe",""))}{flag}</option>\n'

    # Build query cards as JSON for JS rendering
    query_cards: list[dict] = []
    for i, q_ref in enumerate(first_queries):
        crossing_row = crossing_detail_by_idx.get(i, {})
        card: dict[str, Any] = {
            "idx": i,
            "cve": q_ref.get("query_cve", ""),
            "cwe": q_ref.get("query_cwe", ""),
            "func": q_ref.get("query_func", ""),
            "variant": q_ref.get("query_variant", ""),
            "query_code": q_ref.get("query_code", ""),
            "approaches": {},
            # {strategy_name: {hit: bool, prediction: str}} – empty if no crossing data
            "strategies": crossing_row.get("strategies", {}),
        }
        for nm in approach_names:
            q = query_map[i].get(nm, {})
            card["approaches"][nm] = {
                "hit": q.get("mrr", 0) == 1.0,
                "mrr": q.get("mrr", 0),
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
                    for r in (q.get("retrieved") or [])[:5]
                ],
            }
        query_cards.append(card)

    # Limit payload size: truncate code to 2000 chars for JS
    for card in query_cards:
        if len(card.get("query_code") or "") > 2000:
            card["query_code"] = card["query_code"][:2000] + "\n... [truncated]"
        for ap in card["approaches"].values():
            for r in ap.get("retrieved", []):
                if len(r.get("code") or "") > 2000:
                    r["code"] = r["code"][:2000] + "\n... [truncated]"

    colors_map = {nm: _approach_color(approach_names, nm) for nm in approach_names}

    # Build strategy filter options – only available when crossing data is present
    strategy_opts_html = "".join(
        f'<option value="{_esc(s)}">{_esc(s)}</option>'
        for s in strategy_names
    )
    strategy_filter_html = (
        f'''
    <label style="margin-left:1rem"><strong>Strategy failed:</strong></label>
    <select id="strategyFailSel" class="query-select">
      <option value="">(all queries)</option>
      {strategy_opts_html}
    </select>'''
        if strategy_names else
        '<span class="muted" style="margin-left:1rem">Strategy filter: run verify_crossing.py first</span>'
    )

    return f"""
<section id="tab-code" class="tab-panel">
  <h2>Code Explorer</h2>
  <p class="sub">Select a query to see the vulnerable function and what each approach retrieved.
     ⚠️ marks queries where at least one approach missed.</p>

  <div class="card">
    <label for="querySelect"><strong>Query:</strong></label>
    <select id="querySelect" class="query-select">
      {options_html}
    </select>
    <label style="margin-left:1rem"><strong>Show top-k:</strong></label>
    <select id="topKSelect" class="query-select" style="width:80px">
      <option value="1">1</option>
      <option value="3">3</option>
      <option value="5" selected>5</option>
    </select>
    {strategy_filter_html}
  </div>

  <div id="codeExplorer"></div>

  <script>
  (function(){{
    const CARDS   = {json.dumps(query_cards, ensure_ascii=False)};
    const COLORS  = {json.dumps(colors_map)};
    const names   = {json.dumps(approach_names)};
    const sel             = document.getElementById('querySelect');
    const topKSel         = document.getElementById('topKSelect');
    const strategyFailSel = document.getElementById('strategyFailSel');
    const wrap            = document.getElementById('codeExplorer');

    function esc(s) {{
      return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}
    function renderCard(idx) {{
      const card = CARDS[idx];
      if (!card) return;
      const topK = parseInt(topKSel.value)||5;
      // Show strategy badge for active filter
      const activeStrat = strategyFailSel ? strategyFailSel.value : '';
      let stratBadge = '';
      if (activeStrat && card.strategies && card.strategies[activeStrat] !== undefined) {{
        const s = card.strategies[activeStrat];
        const hit = s.hit;
        stratBadge = hit
          ? ` <span class="badge" style="background:#06D6A0">${{esc(activeStrat)}} ✓ hit</span>`
          : ` <span class="badge" style="background:#EF476F">${{esc(activeStrat)}} ✗ miss &nbsp; predicted: ${{esc(s.prediction||'–')}}</span>`;
      }}
      let html = `<div class="code-card">
        <div class="code-card-header">
          <span class="pill-big">${{esc(card.cve)}}</span>
          <span class="pill">${{esc(card.cwe)}}</span>
          ${{stratBadge}}
          <span class="muted" style="margin-left:.5rem">func: ${{esc(card.func)}} &nbsp; variant: ${{esc(card.variant)}}</span>
        </div>
        <div class="code-split">
          <div class="code-pane">
            <div class="pane-label">📥 Query — vulnerable code</div>
            <pre class="code-block"><code>${{esc(card.query_code||'(no code available)')}}</code></pre>
          </div>
          <div class="code-results">`;

      for (const nm of names) {{
        const ap = card.approaches[nm]||{{}};
        const col = COLORS[nm]||'#888';
        const hit = ap.hit;
        const hitBadge = hit
          ? `<span class="badge" style="background:#06D6A0">✓ hit@1</span>`
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
      const stratFail = strategyFailSel ? strategyFailSel.value : '';
      let anyVisible = false;
      for (let opt of sel.options) {{
        const idx = parseInt(opt.value);
        const card = CARDS[idx];
        let show = true;
        if (stratFail) {{
          // Show only queries where this strategy failed (missed)
          const s = (card.strategies || {{}})[stratFail];
          show = s !== undefined && !s.hit;
        }}
        opt.hidden = !show;
        if (show) anyVisible = true;
      }}
      // If current selection is now hidden, advance to first visible
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
    if (strategyFailSel) strategyFailSel.addEventListener('change', filterOptions);
    renderCard(parseInt(sel.value)||0);
  }})();
  </script>
</section>"""


# ─────────────────────────────────────────────────────────────────────
#  Tab 6: Embedding Space
# ─────────────────────────────────────────────────────────────────────

def _collect_embedding_space_artifacts(results: dict, run_dir: Path) -> dict[str, dict[str, str]]:
  """Collect per-embedder embedding-space artifacts from results and file fallback."""
  by_embedder: dict[str, dict[str, str]] = {}

  for cell in results.get("cells", []):
    emb = cell.get("embedder")
    if not emb:
      continue
    plots = cell.get("embedding_space_plots") or {}
    if plots:
      by_embedder[emb] = dict(plots)

  space_dir = run_dir / "embedding_space"
  if space_dir.exists():
    for emb, d in by_embedder.items():
      d.setdefault("raw_vs_pca", f"embedding_space/{emb}_raw_vs_pca.png")
      d.setdefault("umap_2d", f"embedding_space/{emb}_umap.png")
      d.setdefault("umap_3d_html", f"embedding_space/{emb}_3d_interactive.html")
      d.setdefault("pca_scree", f"embedding_space/{emb}_pca_scree.png")

    # Fallback for runs where results.json lacks explicit plot metadata.
    for p in sorted(space_dir.glob("*_raw_vs_pca.png")):
      emb = p.name[: -len("_raw_vs_pca.png")]
      by_embedder.setdefault(emb, {})
      by_embedder[emb].setdefault("raw_vs_pca", f"embedding_space/{p.name}")
      by_embedder[emb].setdefault("umap_2d", f"embedding_space/{emb}_umap.png")
      by_embedder[emb].setdefault("umap_3d_html", f"embedding_space/{emb}_3d_interactive.html")
      by_embedder[emb].setdefault("pca_scree", f"embedding_space/{emb}_pca_scree.png")

  return by_embedder


def _tab_embedding_space(results: dict, run_dir: Path) -> str:
  artifacts = _collect_embedding_space_artifacts(results, run_dir)
  if not artifacts:
    # Reuse the full embedding-space analysis dashboard (analyze_spaces),
    # embedded inline, rather than maintaining a parallel visualisation here.
    if (run_dir / "space_dashboard.html").exists():
      return """<section id="tab-embedding-space" class="tab-panel">
  <h2>Embedding Space</h2>
  <p class="sub">Intrinsic quality, CWE class separation and cross-space comparison,
     computed from the persisted vector indices.</p>
  <div class="card" style="padding:0; overflow:hidden">
    <iframe src="space_dashboard.html" title="Embedding space analysis"
            style="width:100%; height:1400px; border:0; display:block"></iframe>
  </div>
  <p class="sub"><a href="space_dashboard.html" target="_blank" rel="noopener">Open full screen ↗</a></p>
</section>"""
    return """<section id="tab-embedding-space" class="tab-panel">
  <h2>Embedding Space</h2>
  <p class="muted">Embedding-space analysis was not found for this run.</p>
</section>"""

  cards_html = ""
  for emb_name in sorted(artifacts.keys()):
    p = artifacts[emb_name]

    raw_vs_pca = p.get("raw_vs_pca", "")
    umap_2d = p.get("umap_2d", "")
    pca_scree = p.get("pca_scree", "")
    umap_3d_html = p.get("umap_3d_html", "")

    raw_block = (
      f'<img src="{_esc(raw_vs_pca)}" alt="{_esc(emb_name)} raw vs pca" loading="lazy" />'
      if raw_vs_pca and (run_dir / raw_vs_pca).exists()
      else '<p class="muted">Missing raw vs PCA plot.</p>'
    )
    umap_block = (
      f'<img src="{_esc(umap_2d)}" alt="{_esc(emb_name)} umap" loading="lazy" />'
      if umap_2d and (run_dir / umap_2d).exists()
      else '<p class="muted">Missing UMAP 2D plot.</p>'
    )
    scree_block = (
      f'<img src="{_esc(pca_scree)}" alt="{_esc(emb_name)} pca scree" loading="lazy" />'
      if pca_scree and (run_dir / pca_scree).exists()
      else '<p class="muted">Missing PCA scree plot.</p>'
    )
    umap3d_link = (
      f'<a href="{_esc(umap_3d_html)}" target="_blank" rel="noopener">Open interactive 3D plot</a>'
      if umap_3d_html and (run_dir / umap_3d_html).exists()
      else '<span class="muted">Interactive 3D plot not available.</span>'
    )

    cards_html += f"""
  <div class="card">
  <h3>{_esc(emb_name)}</h3>
  <div class="emb-grid">
    <div class="emb-plot"><h4>t-SNE Raw vs PCA</h4>{raw_block}</div>
    <div class="emb-plot"><h4>UMAP Raw vs PCA</h4>{umap_block}</div>
    <div class="emb-plot"><h4>PCA Scree Curve</h4>{scree_block}</div>
    <div class="emb-plot emb-plot-link"><h4>3D UMAP</h4><p>{umap3d_link}</p></div>
  </div>
  </div>"""

  return f"""
<section id="tab-embedding-space" class="tab-panel">
  <h2>Embedding Space</h2>
  <p class="sub">Diagnostic projections and PCA variance curves for each embedder.</p>
  {cards_html}
</section>"""


# ─────────────────────────────────────────────────────────────────────
#  CSS + JS shell
# ─────────────────────────────────────────────────────────────────────

_CSS = f"""
:root {{
{_CSS_PALETTE_VARS}
  --bg: #f5f4f0;
  --surface: #ffffff;
  --ink: #1a1a2e;
  --muted: #6b7280;
  --accent: var(--c0);
  --accent-light: #eef0fd;
  --border: #e2e0db;
  --radius: 12px;
  --shadow: 0 4px 24px rgba(10,10,40,.07);
}}
*, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
body {{
  font-family: "Inter", "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  color: var(--ink);
  font-size: 14px;
  line-height: 1.6;
}}
a {{ color: var(--accent); }}

/* ── Top bar ── */
.topbar {{
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #fff;
  padding: 18px 32px;
  display: flex; align-items: center; gap: 16px;
}}
.topbar h1 {{ font-size: 1.4rem; font-weight: 700; letter-spacing: -.5px; }}
.topbar .run-id {{ font-size:.8rem; background:#ffffff22; padding:3px 10px; border-radius:99px; }}

/* ── Tab nav ── */
.tab-nav {{
  background: var(--surface);
  border-bottom: 2px solid var(--border);
  display: flex; padding: 0 32px; gap: 4px; position: sticky; top:0; z-index:10;
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
}}
.tab-btn {{
  padding: 12px 20px; border: none; background: none;
  cursor: pointer; font-size: .95rem; font-weight: 500; color: var(--muted);
  border-bottom: 3px solid transparent; margin-bottom: -2px;
  transition: color .2s, border-color .2s;
}}
.tab-btn:hover {{ color: var(--ink); }}
.tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}

/* ── Content ── */
.content {{ max-width: 1300px; margin: 0 auto; padding: 28px 24px 64px; }}
.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}

h2 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; }}
h3 {{ font-size: 1.05rem; font-weight: 600; margin-bottom: 12px; }}
h4 {{ font-size: .9rem; font-weight: 600; margin: 10px 0 6px; }}
p.sub {{ color: var(--muted); margin-bottom: 20px; font-size:.92rem; }}
p.muted {{ color: var(--muted); }}
li {{ margin-bottom: 4px; }}

/* ── Cards ── */
.card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 20px 24px;
  margin-bottom: 18px; box-shadow: var(--shadow);
}}

/* ── Stat cards ── */
.stat-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 20px; }}
.stat-row.compact .stat-card {{ min-width: 110px; }}
.stat-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px 20px;
  min-width: 160px; box-shadow: var(--shadow); flex: 1;
}}
.stat-card.accent {{ background: var(--accent-light); border-color: var(--accent); }}
.stat-label {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }}
.stat-value {{ font-size: 1.9rem; font-weight: 700; color: var(--ink); line-height:1.2; }}
.stat-sub {{ font-size:.8rem; color:var(--muted); margin-top:4px; }}

/* ── Table ── */
table {{ width: 100%; border-collapse: collapse; font-size:.9rem; }}
th, td {{ padding: 9px 12px; text-align:left; border-bottom: 1px solid var(--border); }}
th {{ background: var(--bg); font-weight:600; font-size:.8rem;
  text-transform:uppercase; letter-spacing:.05em; }}
tr:hover td {{ background: #f9f8f5; }}
.scroll-x {{ overflow-x: auto; }}
.matrix td {{ text-align:center; font-size:.82rem; }}

/* ── Sparkbar ── */
.sparkbar-wrap {{ display:inline-flex; align-items:center; gap:6px; }}
.sparkbar {{ display:inline-block; height:10px; border-radius:3px; }}
.sparkbar-label {{ font-size:.82rem; color:var(--ink); }}

/* ── Dot / badge / pill ── */
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:99px; font-size:.75rem; font-weight:600; }}
.pill {{ display:inline-block; padding:2px 8px; border-radius:99px; font-size:.78rem;
         background:var(--accent-light); color:var(--accent); }}
.pill.warn {{ background:#fff0f3; color:#c62828; }}
.pill.ok   {{ background:#e8faf3; color:#146c43; }}
.pill-big  {{ display:inline-block; padding:4px 12px; border-radius:99px; font-size:.9rem; font-weight:700;
              background:#1a1a2e; color:#fff; }}
.delta {{ font-size:.8rem; font-weight:600; padding:1px 6px; border-radius:4px; }}
.delta.up {{ background:#e8faf3; color:#146c43; }}
.delta.down {{ background:#fff0f3; color:#c62828; }}

/* ── Chart layout ── */
.chart-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(440px,1fr)); gap:18px; margin-bottom:18px; }}
.chart-card {{ min-height: 280px; }}

/* ── Miss ── */
.cell-header {{ display:flex; align-items:center; gap:6px; margin-bottom:14px; font-size:.95rem; }}
.miss-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:12px; }}

/* ── Code explorer ── */
.query-select {{ margin-left:.5rem; padding:6px 10px; border-radius:8px; border:1px solid var(--border); font-size:.9rem; }}
.code-card {{ margin-top:18px; }}
.code-card-header {{ display:flex; align-items:center; gap:8px; margin-bottom:14px; flex-wrap:wrap; }}
.code-split {{ display:grid; grid-template-columns:1fr 2fr; gap:16px; align-items:start; }}
.code-pane {{ background:#0d1117; border-radius:var(--radius); overflow:hidden; }}
.pane-label {{ background:#161b22; color:#8b949e; font-size:.75rem; padding:8px 14px; }}
.code-block {{ background:#0d1117; color:#c9d1d9; padding:14px; font-size:.8rem;
               font-family:"JetBrains Mono","Fira Code",monospace; overflow-x:auto;
               max-height:400px; overflow-y:auto; white-space:pre; border-radius:0 0 var(--radius) var(--radius); }}
.code-block.small {{ max-height:200px; font-size:.75rem; }}
.code-block.muted {{ color: #555; font-style:italic; }}
.code-results {{ display:flex; flex-direction:column; gap:12px; }}
.approach-block {{ border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
.approach-header {{ background:#fafafa; padding:10px 14px; display:flex; align-items:center; gap:6px; flex-wrap:wrap; }}
.retrieved-item {{ padding:10px 14px; border-top:1px solid var(--border); }}
.retrieved-meta {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap:wrap; }}
.rank {{ font-size:.8rem; font-weight:700; color:var(--muted); min-width:24px; }}

/* ── Embedding-space tab ── */
.emb-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:14px; }}
.emb-plot {{ border:1px solid var(--border); border-radius:10px; padding:10px; background:#fafafa; }}
.emb-plot img {{ width:100%; height:auto; border-radius:8px; display:block; }}
.emb-plot-link {{ display:flex; align-items:center; justify-content:center; min-height:180px; }}

details > summary {{ cursor:pointer; font-weight:600; user-select:none; }}
details > summary:hover {{ color:var(--accent); }}
code {{ font-family:"JetBrains Mono","Fira Code",monospace; }}
"""

_JS_TABS = """
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});
"""


# ─────────────────────────────────────────────────────────────────────
#  Main entry
# ─────────────────────────────────────────────────────────────────────

def _ensure_analyses(run_dir: Path) -> None:
    """Derive optional analysis artifacts in-place when they are missing.

    Keeps the dashboard self-contained: the Miss Analysis, Crossing Strategies
    and Embedding Space tabs are computed on demand from results.json and the
    persisted vector indices, so no separate scripts need to be run first.
    Each step is best-effort — on failure the corresponding tab falls back to
    its guidance message.
    """
    results_path = run_dir / "results.json"

    # ── Miss / uncertainty analysis (miss_analysis.json) ─────────────
    miss_path = run_dir / "miss_analysis.json"
    if not miss_path.exists():
        try:
            from experiments.dashboard_scripts.analyze_misses import analyze_results
            analyze_results(
                results_path=results_path,
                out_json=miss_path,
                out_html=run_dir / "miss_dashboard.html",
                uncertainty_quantile=25.0,
                uncertainty_prob_floor=0.12,
                uncertainty_margin_floor=0.005,
                temperature=1.0,
                max_examples=20,
                regenerate_unified=False,  # prevent recursion back into this dashboard
            )
        except Exception as e:
            print(f"  [dashboard] miss analysis skipped: {e}")

    # ── Crossing / fusion strategy analysis (crossing_analysis.json) ─
    crossing_path = run_dir / "crossing_analysis.json"
    if not crossing_path.exists():
        try:
            from experiments.put_aside.verify_crossing import analyze_crossing
            report = analyze_crossing(results_path)
            crossing_path.write_text(json.dumps(report, indent=2))
        except Exception as e:
            print(f"  [dashboard] crossing analysis skipped: {e}")

    # ── Threshold study / abstention error analysis (threshold_analysis.json) ─
    threshold_path = run_dir / "threshold_analysis.json"
    if not threshold_path.exists():
        try:
            from experiments.dashboard_scripts.analyze_thresholds import analyze_results as _analyze_thresholds
            _analyze_thresholds(
                results_path=results_path,
                out_json=threshold_path,
                regenerate_unified=False,  # prevent recursion back into this dashboard
            )
        except Exception as e:
            print(f"  [dashboard] threshold analysis skipped: {e}")

    # ── Embedding-space analysis (space_dashboard.html) ──────────────
    # Reuse analyze_spaces + src.metrics.space_analysis, sourcing embeddings
    # from the persisted vector indices instead of re-embedding.
    if not (run_dir / "space_dashboard.html").exists():
        try:
            from experiments.dashboard_scripts.analyze_spaces import ensure_space_dashboard
            ensure_space_dashboard(run_dir)
        except Exception as e:
            print(f"  [dashboard] embedding-space analysis skipped: {e}")


def generate_html_dashboard(run_dir: str | Path, compute_missing: bool = True) -> Path:
    """
    Read all JSON files from run_dir, generate a single dashboard.html.
    Returns the path to the written file.

    When ``compute_missing`` is True (default), any missing miss/crossing
    analyses are derived from results.json and the Embedding Space tab is built
    from the persisted vector indices first, so every tab is populated without
    running separate scripts.
    """
    run_dir = Path(run_dir)

    results_path  = run_dir / "results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")

    if compute_missing:
        _ensure_analyses(run_dir)

    miss_path     = run_dir / "miss_analysis.json"
    crossing_path = run_dir / "crossing_analysis.json"
    threshold_path = run_dir / "threshold_analysis.json"

    results  = json.loads(results_path.read_text())
    miss     = json.loads(miss_path.read_text())     if miss_path.exists()     else None
    crossing = json.loads(crossing_path.read_text()) if crossing_path.exists() else None
    threshold = json.loads(threshold_path.read_text()) if threshold_path.exists() else None

    run_id = results.get("run_id", run_dir.name)

    tabs = [
        ("tab-overview",  "🏠 Overview",              _tab_overview(results, miss, crossing)),
        ("tab-retrieval", "📊 Retrieval Performance",  _tab_retrieval(results)),
        ("tab-threshold", "🎯 Threshold Study",        _tab_threshold(threshold, results)),
        ("tab-miss",      "🔍 Miss Analysis",          _tab_miss(miss, results)),
        ("tab-crossing",  "🔀 Crossing Strategies",    _tab_crossing(crossing, results)),
      ("tab-embedding-space", "🧭 Embedding Space", _tab_embedding_space(results, run_dir)),
        ("tab-code",      "💻 Code Explorer",          _tab_code(results, crossing)),
    ]

    nav_html = "".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" data-tab="{tid}">{label}</button>'
        for i, (tid, label, _) in enumerate(tabs)
    )
    content_html = "\n".join(body for _, _, body in tabs)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Graph-RAG Dashboard — {_esc(run_id)}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>{_CSS}</style>
</head>
<body>

<div class="topbar">
  <h1>Graph-RAG Experiment Dashboard</h1>
  <span class="run-id">{_esc(run_id)}</span>
</div>

<nav class="tab-nav">{nav_html}</nav>

<main class="content">
{content_html}
</main>

<script>
{_JS_TABS}
</script>
</body>
</html>"""

    out = run_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {out}")
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate unified HTML dashboard for an experiment run.")
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

    generate_html_dashboard(run_dir)


if __name__ == "__main__":
    main()
