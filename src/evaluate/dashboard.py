"""Generate an HTML evaluation dashboard from evaluation output files.

Reads evaluation_summary.json, retrieval_eval_summary.json,
confidence_eval_summary.json, and confidence_eval_high_conf.jsonl
from a run directory and produces a self-contained evaluation_dashboard.html.

Usage (standalone):
    python -m src.evaluate.dashboard <run_dir>

Programmatic:
    from src.evaluate.dashboard import generate_dashboard
    generate_dashboard(Path("experiments/output/some_run/"))
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

# ── CSS (shared with experiments dashboard) ──────────────────────────

_CSS = """\
:root {
  --c0: #4361EE; --c1: #F72585; --c2: #7209B7; --c3: #FB8500;
  --c4: #06D6A0; --c5: #118AB2; --c6: #FFD166; --c7: #EF476F;
  --bg: #f5f4f0; --surface: #ffffff; --ink: #1a1a2e; --muted: #6b7280;
  --accent: var(--c0); --accent-light: #eef0fd; --border: #e2e0db;
  --radius: 12px; --shadow: 0 4px 24px rgba(10,10,40,.07);
}
*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }
body { font-family: "Inter","Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--ink); font-size:14px; line-height:1.6; }
a { color:var(--accent); }
.topbar { background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%); color:#fff; padding:18px 32px; display:flex; align-items:center; gap:16px; }
.topbar h1 { font-size:1.4rem; font-weight:700; letter-spacing:-.5px; }
.topbar .run-id { font-size:.8rem; background:#ffffff22; padding:3px 10px; border-radius:99px; }
.tab-nav { background:var(--surface); border-bottom:2px solid var(--border); display:flex; padding:0 32px; gap:4px; position:sticky; top:0; z-index:10; box-shadow:0 2px 8px rgba(0,0,0,.06); }
.tab-btn { padding:12px 20px; border:none; background:none; cursor:pointer; font-size:.95rem; font-weight:500; color:var(--muted); border-bottom:3px solid transparent; margin-bottom:-2px; transition:color .2s,border-color .2s; }
.tab-btn:hover { color:var(--ink); }
.tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
.content { max-width:1300px; margin:0 auto; padding:28px 24px 64px; }
.tab-panel { display:none; }
.tab-panel.active { display:block; }
h2 { font-size:1.5rem; font-weight:700; margin-bottom:6px; }
h3 { font-size:1.05rem; font-weight:600; margin-bottom:12px; }
p.sub { color:var(--muted); margin-bottom:20px; font-size:.92rem; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:20px 24px; margin-bottom:18px; box-shadow:var(--shadow); }
.stat-row { display:flex; flex-wrap:wrap; gap:14px; margin-bottom:20px; }
.stat-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:16px 20px; min-width:160px; box-shadow:var(--shadow); flex:1; }
.stat-card.accent { background:var(--accent-light); border-color:var(--accent); }
.stat-card.warn { background:#fff0f3; border-color:#c62828; }
.stat-card.ok { background:#e8faf3; border-color:#146c43; }
.stat-label { font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.stat-value { font-size:1.9rem; font-weight:700; color:var(--ink); line-height:1.2; }
.stat-sub { font-size:.8rem; color:var(--muted); margin-top:4px; }
table { width:100%; border-collapse:collapse; font-size:.9rem; }
th, td { padding:9px 12px; text-align:left; border-bottom:1px solid var(--border); }
th { background:var(--bg); font-weight:600; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }
tr:hover td { background:#f9f8f5; }
.scroll-x { overflow-x:auto; }
.sparkbar-wrap { display:inline-flex; align-items:center; gap:6px; }
.sparkbar { display:inline-block; height:10px; border-radius:3px; }
.sparkbar-label { font-size:.82rem; color:var(--ink); }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.pill { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.78rem; background:var(--accent-light); color:var(--accent); }
.pill.warn { background:#fff0f3; color:#c62828; }
.pill.ok { background:#e8faf3; color:#146c43; }
.chart-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:18px; margin-bottom:18px; }
.chart-card { min-height:280px; }
.gauge-row { display:flex; flex-wrap:wrap; gap:18px; margin-bottom:20px; }
.gauge-card { flex:1; min-width:180px; text-align:center; }
.gauge-ring { position:relative; width:120px; height:120px; margin:0 auto 8px; }
.gauge-ring svg { transform:rotate(-90deg); }
.gauge-ring .value { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:1.4rem; font-weight:700; }
.high-conf-table td.cve { font-family:"JetBrains Mono","Fira Code",monospace; font-size:.82rem; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
"""

_VARIANT_COLORS = [
    ("#4361EE", "re_implemented_deepseek-r1"),
    ("#F72585", "re_implemented_llama"),
    ("#7209B7", "augmented"),
    ("#FB8500", "re_implemented_deepseek"),
    ("#06D6A0", "re_implemented_o3-mini"),
    ("#118AB2", "re_implemented_gpt-4o"),
]

_CHART_COLORS = ["#4361EE", "#F72585", "#7209B7", "#FB8500", "#06D6A0",
                 "#118AB2", "#FFD166", "#EF476F", "#333"]


# ── helpers ──────────────────────────────────────────────────────────

def _esc(s) -> str:
    return html.escape(str(s))


def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _gauge_svg(value: float, color: str) -> str:
    circ = 2 * 3.14159 * 50
    filled = round(value * circ, 1)
    empty = round(circ - filled, 1)
    return (
        f'<svg width="120" height="120" viewBox="0 0 120 120">'
        f'<circle cx="60" cy="60" r="50" fill="none" stroke="#e2e0db" stroke-width="10"/>'
        f'<circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="10"'
        f' stroke-dasharray="{filled} {empty}" stroke-linecap="round"/>'
        f'</svg>'
    )


def _js_obj(d: dict) -> str:
    return json.dumps(d)


def _pct(v: float) -> str:
    return f"{v:.1%}"


# ── section builders ─────────────────────────────────────────────────

def _build_overview(patch: dict, retr: dict, conf: dict, high_conf: list, run_id: str) -> str:
    n = patch.get("total_records", 0)
    hit1 = retr.get("hit_at_1", 0)
    hit1r = retr.get("hit_rate_at_1", 0)
    mrr = retr.get("mrr", 0)
    bleu4 = patch.get("avg_bleu_4", 0)
    exact = patch.get("exact_matches", 0)
    hc = len(high_conf)

    split_info = retr.get("split_info", {})
    counts = split_info.get("counts", {})

    return f"""
<section id="tab-overview" class="tab-panel active">
  <h2>Pipeline Overview</h2>
  <p class="sub">End-to-end evaluation: Retrieval &rarr; Patch Generation &rarr; Confidence Analysis</p>
  <div class="stat-row">
    <div class="stat-card accent"><div class="stat-label">Total Queries</div><div class="stat-value">{n}</div><div class="stat-sub">all evaluated</div></div>
    <div class="stat-card ok"><div class="stat-label">Retrieval Hit@1</div><div class="stat-value">{_pct(hit1r)}</div><div class="stat-sub">{hit1} / {n} correct</div></div>
    <div class="stat-card"><div class="stat-label">Retrieval MRR</div><div class="stat-value">{mrr:.3f}</div></div>
    <div class="stat-card"><div class="stat-label">Avg BLEU-4</div><div class="stat-value">{bleu4:.3f}</div><div class="stat-sub">patch vs ground truth</div></div>
    <div class="stat-card warn"><div class="stat-label">Exact Matches</div><div class="stat-value">{exact}</div><div class="stat-sub">out of {n}</div></div>
    <div class="stat-card"><div class="stat-label">High-Conf Failures</div><div class="stat-value">{hc}</div><div class="stat-sub">retriever re-identifies vuln</div></div>
  </div>
  <div class="card"><h3>Dataset Split</h3><table>
    <thead><tr><th>Split</th><th>Count</th></tr></thead><tbody>
    <tr><td>Total pairs</td><td><strong>{counts.get('total', '–')}</strong></td></tr>
    <tr><td>Real (original)</td><td><strong>{counts.get('real_total', '–')}</strong></td></tr>
    <tr><td>Augmented total</td><td><strong>{counts.get('aug_total', '–')}</strong></td></tr>
    <tr><td>Index (train)</td><td><strong>{counts.get('index_total', '–')}</strong></td></tr>
    <tr><td>Query (test)</td><td><strong>{counts.get('query_total', '–')}</strong></td></tr>
  </tbody></table></div>
</section>"""


def _build_patch_tab(patch: dict) -> str:
    bleu4 = patch.get("avg_bleu_4", 0)
    jaccard = patch.get("avg_token_jaccard", 0)
    codebleu = patch.get("avg_codebleu_proxy", 0)
    edit = patch.get("avg_normalised_edit_distance", 0)
    n = patch.get("evaluated", 0)
    exact = patch.get("exact_matches", 0)

    by_cwe = patch.get("by_cwe", {})
    by_var = patch.get("by_variant", {})

    return f"""
<section id="tab-patch" class="tab-panel">
  <h2>Patch Quality</h2>
  <p class="sub">Generated patches vs ground-truth fixed functions &mdash; {n} records, {exact} exact matches</p>
  <div class="gauge-row">
    <div class="card gauge-card"><h3>BLEU-4</h3><div class="gauge-ring">{_gauge_svg(bleu4, '#4361EE')}<span class="value">{bleu4:.3f}</span></div></div>
    <div class="card gauge-card"><h3>Token Jaccard</h3><div class="gauge-ring">{_gauge_svg(jaccard, '#06D6A0')}<span class="value">{jaccard:.3f}</span></div></div>
    <div class="card gauge-card"><h3>CodeBLEU</h3><div class="gauge-ring">{_gauge_svg(codebleu, '#7209B7')}<span class="value">{codebleu:.3f}</span></div></div>
    <div class="card gauge-card"><h3>Edit Distance</h3><div class="gauge-ring">{_gauge_svg(edit, '#F72585')}<span class="value">{edit:.3f}</span></div><div class="stat-sub">normalised (lower = better)</div></div>
  </div>
  <div class="chart-grid">
    <div class="card chart-card"><h3>Patch Quality by CWE</h3><canvas id="cwePatchChart" height="260"></canvas></div>
    <div class="card chart-card"><h3>Patch Quality by Variant</h3><canvas id="variantPatchChart" height="260"></canvas></div>
  </div>
  <div class="card"><h3>By CWE Type</h3><div class="scroll-x"><table>
    <thead><tr><th>CWE</th><th>Count</th><th>BLEU-4</th><th>Jaccard</th><th>CodeBLEU</th><th>Char Ratio</th></tr></thead>
    <tbody id="cwePatchTable"></tbody></table></div></div>
  <div class="card"><h3>By Variant</h3><table>
    <thead><tr><th>Variant</th><th>Count</th><th>BLEU-4</th><th>Jaccard</th><th>CodeBLEU</th><th>Char Ratio</th></tr></thead>
    <tbody id="variantPatchTable"></tbody></table></div>
</section>"""


def _build_retrieval_tab(retr: dict) -> str:
    n = retr.get("matched", 0)
    top_k = retr.get("top_k", 5)
    hit1 = retr.get("hit_at_1", 0)
    hit1r = retr.get("hit_rate_at_1", 0)
    hitk = retr.get("hit_cve_at_k", 0)
    hitkr = retr.get("hit_rate_at_k", 0)
    mrr = retr.get("mrr", 0)
    cwek = retr.get("cwe_hit_at_k", 0)
    cwekr = retr.get("cwe_hit_rate_at_k", 0)

    return f"""
<section id="tab-retrieval" class="tab-panel">
  <h2>Retrieval Performance</h2>
  <p class="sub">FAISS index with combined embedder (NetLSD+WL+GIN &rarr; PCA &rarr; 128d)</p>
  <div class="stat-row">
    <div class="stat-card accent"><div class="stat-label">Hit@1</div><div class="stat-value">{_pct(hit1r)}</div><div class="stat-sub">{hit1} / {n}</div></div>
    <div class="stat-card"><div class="stat-label">Hit@{top_k}</div><div class="stat-value">{_pct(hitkr)}</div><div class="stat-sub">{hitk} / {n}</div></div>
    <div class="stat-card"><div class="stat-label">MRR</div><div class="stat-value">{mrr:.3f}</div></div>
    <div class="stat-card"><div class="stat-label">CWE Recall@{top_k}</div><div class="stat-value">{_pct(cwekr)}</div><div class="stat-sub">{cwek} / {n}</div></div>
  </div>
  <div class="chart-grid">
    <div class="card chart-card"><h3>Hit Rate Breakdown</h3><canvas id="hitRateChart" height="200"></canvas></div>
    <div class="card chart-card"><h3>Score Distribution</h3><canvas id="scoreDistChart" height="200"></canvas></div>
  </div>
</section>"""


def _build_confidence_tab(conf: dict) -> str:
    thresholds = conf.get("thresholds", [])
    rows = ""
    for t in thresholds:
        label = t.get("label", f"{t['threshold']:.4f}")
        pill_cls = "ok" if label == "p75" else ("warn" if label == "p90" else "")
        style = ' style="background:#eef0fd"' if label == "p75" else ""
        rows += (
            f'<tr{style}><td>{t["threshold"]:.4f}</td>'
            f'<td><span class="pill {pill_cls}">{_esc(label)}</span></td>'
            f'<td>{t.get("confident_total", 0)}</td><td>{t["tp"]}</td><td>{t["fp"]}</td>'
            f'<td><strong>{_pct(t["precision"])}</strong></td>'
            f'<td>{_pct(t["recall"])}</td><td>{_pct(t["f1"])}</td><td>{_pct(t["accuracy"])}</td></tr>\n'
        )
    return f"""
<section id="tab-confidence" class="tab-panel">
  <h2>Confidence Threshold Evaluation</h2>
  <p class="sub">Evaluating retrieval confidence as a predictor of correct vulnerability identification</p>
  <div class="chart-grid">
    <div class="card chart-card"><h3>Precision / Recall / F1 by Threshold</h3><canvas id="prfChart" height="240"></canvas></div>
    <div class="card chart-card"><h3>Confident Predictions</h3><canvas id="confCountChart" height="240"></canvas></div>
  </div>
  <div class="card"><h3>Threshold Results</h3><table>
    <thead><tr><th>Threshold</th><th>Label</th><th>Confident</th><th>TP</th><th>FP</th><th>Precision</th><th>Recall</th><th>F1</th><th>Accuracy</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
</section>"""


def _build_highconf_tab(high_conf: list, conf: dict) -> str:
    hc_threshold = conf.get("high_confidence_threshold", 0)
    n = len(high_conf)
    unique_cves = len({e["query_cve"] for e in high_conf})
    max_score = max((e["top1_score"] for e in high_conf), default=0)
    min_score = min((e["top1_score"] for e in high_conf), default=0)
    max_cve = next((e["query_cve"] for e in high_conf if e["top1_score"] == max_score), "–")
    min_cve = next((e["query_cve"] for e in high_conf if e["top1_score"] == min_score), "–")

    rows = ""
    for i, e in enumerate(high_conf, 1):
        rows += (
            f'<tr><td>{i}</td><td class="cve">{_esc(e.get("query_cve", ""))}</td>'
            f'<td>{_esc(e.get("query_cwe", ""))}</td><td>{_esc(e.get("query_variant", ""))}</td>'
            f'<td class="num"><strong>{e["top1_score"]:.4f}</strong></td>'
            f'<td class="num">{e.get("score_gap", 0):.4f}</td>'
            f'<td class="cve">{_esc(e.get("top1_cve", ""))}</td>'
            f'<td>{_esc(e.get("top1_func", ""))}</td></tr>\n'
        )

    return f"""
<section id="tab-highconf" class="tab-panel">
  <h2>High-Confidence Patch Failures</h2>
  <p class="sub">{n} entries where the retriever confidently re-identifies the same CVE after patching.
     Threshold: p75 ({hc_threshold}). All entries have <code>top1_correct = true</code>.</p>
  <div class="stat-row">
    <div class="stat-card warn"><div class="stat-label">Unsuccessful Patches</div><div class="stat-value">{n}</div><div class="stat-sub">same CVE re-identified</div></div>
    <div class="stat-card"><div class="stat-label">Unique CVEs</div><div class="stat-value">{unique_cves}</div><div class="stat-sub">distinct vulnerabilities</div></div>
    <div class="stat-card"><div class="stat-label">Max Score</div><div class="stat-value">{max_score:.3f}</div><div class="stat-sub">{_esc(max_cve)}</div></div>
    <div class="stat-card"><div class="stat-label">Min Score</div><div class="stat-value">{min_score:.3f}</div><div class="stat-sub">{_esc(min_cve)}</div></div>
  </div>
  <div class="card"><h3>Score Distribution of Failures</h3><canvas id="highConfScoreChart" height="200" style="max-height:260px"></canvas></div>
  <div class="card"><h3>All High-Confidence Failures</h3><div class="scroll-x"><table class="high-conf-table">
    <thead><tr><th>#</th><th>Query CVE</th><th>CWE</th><th>Variant</th><th>Score</th><th>Gap</th><th>Top-1 Match</th><th>Top-1 Func</th></tr></thead>
    <tbody>{rows}</tbody></table></div></div>
  <div class="card"><h3>CWE Breakdown of Failures</h3><canvas id="highConfCweChart" height="180"></canvas></div>
  <div class="card"><h3>Variant Breakdown of Failures</h3><canvas id="highConfVariantChart" height="160"></canvas></div>
</section>"""


# ── JS chart builder ─────────────────────────────────────────────────

def _build_js(patch: dict, retr: dict, conf: dict, high_conf: list) -> str:
    by_cwe = patch.get("by_cwe", {})
    by_var = patch.get("by_variant", {})
    score_stats = conf.get("score_stats", {})
    thresholds = conf.get("thresholds", [])

    # variant color map
    var_color_map = {}
    for color, name in _VARIANT_COLORS:
        var_color_map[name] = color
    # assign colors to any new variants
    idx = 0
    for v in by_var:
        if v not in var_color_map:
            var_color_map[v] = _CHART_COLORS[idx % len(_CHART_COLORS)]
            idx += 1

    # build by_cwe JS data
    cwe_js = {}
    for cwe, stats in sorted(by_cwe.items(), key=lambda x: -x[1].get("avg_bleu_4", 0)):
        cwe_js[cwe] = {
            "count": stats["count"],
            "bleu4": stats.get("avg_bleu_4", 0),
            "jaccard": stats.get("avg_token_jaccard", 0),
            "codebleu": stats.get("avg_codebleu_proxy", 0),
            "charRatio": stats.get("avg_char_ratio", 0),
        }

    var_js = {}
    for var, stats in sorted(by_var.items(), key=lambda x: -x[1].get("avg_bleu_4", 0)):
        var_js[var] = {
            "count": stats["count"],
            "bleu4": stats.get("avg_bleu_4", 0),
            "jaccard": stats.get("avg_token_jaccard", 0),
            "codebleu": stats.get("avg_codebleu_proxy", 0),
            "charRatio": stats.get("avg_char_ratio", 0),
        }

    # high-conf entries for chart
    hc_entries = [
        {"cve": f"{e['query_cve']}\\n({e.get('query_variant','')[:12]})", "score": round(e["top1_score"], 4)}
        for e in high_conf
    ]
    hc_cwes = [e.get("query_cwe", "") for e in high_conf]
    hc_variants = [e.get("query_variant", "") for e in high_conf]

    # threshold data
    hit1r = retr.get("hit_rate_at_1", 0)
    hitkr = retr.get("hit_rate_at_k", 0)
    cwekr = retr.get("cwe_hit_rate_at_k", 0)

    return f"""
<script>
// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  }});
}});

const byCwe = {json.dumps(cwe_js)};
const byVariant = {json.dumps(var_js)};
const variantColors = {json.dumps(var_color_map)};

function sparkbar(val, max, color) {{
  const w = Math.round((val / max) * 120);
  return `<span class="sparkbar-wrap"><span class="sparkbar" style="width:${{w}}px;background:${{color}}"></span><span class="sparkbar-label">${{val.toFixed(4)}}</span></span>`;
}}

// CWE table
(function(){{
  const tbody = document.getElementById('cwePatchTable');
  if (!tbody) return;
  Object.entries(byCwe).sort((a,b) => b[1].bleu4 - a[1].bleu4).forEach(([cwe, d]) => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{cwe}}</td><td class="num">${{d.count}}</td>
      <td>${{sparkbar(d.bleu4, 1, '#4361EE')}}</td><td>${{sparkbar(d.jaccard, 1, '#06D6A0')}}</td>
      <td>${{sparkbar(d.codebleu, 1, '#7209B7')}}</td><td>${{sparkbar(d.charRatio, 1, '#FB8500')}}</td>`;
    tbody.appendChild(tr);
  }});
}})();

// Variant table
(function(){{
  const tbody = document.getElementById('variantPatchTable');
  if (!tbody) return;
  Object.entries(byVariant).sort((a,b) => b[1].bleu4 - a[1].bleu4).forEach(([v, d]) => {{
    const c = variantColors[v] || '#999';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><span class="dot" style="background:${{c}}"></span>${{v}}</td><td class="num">${{d.count}}</td>
      <td>${{sparkbar(d.bleu4, 1, c)}}</td><td>${{sparkbar(d.jaccard, 1, c)}}</td>
      <td>${{sparkbar(d.codebleu, 1, c)}}</td><td>${{sparkbar(d.charRatio, 1, c)}}</td>`;
    tbody.appendChild(tr);
  }});
}})();

// CWE Patch chart
(function(){{
  const el = document.getElementById('cwePatchChart'); if (!el) return;
  const labels = Object.keys(byCwe);
  new Chart(el, {{ type:'bar', data:{{ labels, datasets:[
    {{ label:'BLEU-4', data:labels.map(l=>byCwe[l].bleu4), backgroundColor:'#4361EE99' }},
    {{ label:'Jaccard', data:labels.map(l=>byCwe[l].jaccard), backgroundColor:'#06D6A099' }},
    {{ label:'CodeBLEU', data:labels.map(l=>byCwe[l].codebleu), backgroundColor:'#7209B799' }},
  ]}}, options:{{ indexAxis:'y', responsive:true, scales:{{x:{{min:0,max:1}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});
}})();

// Variant Patch chart
(function(){{
  const el = document.getElementById('variantPatchChart'); if (!el) return;
  const labels = Object.keys(byVariant);
  const colors = labels.map(l => variantColors[l] || '#999');
  new Chart(el, {{ type:'bar', data:{{ labels, datasets:[
    {{ label:'BLEU-4', data:labels.map(l=>byVariant[l].bleu4), backgroundColor:colors.map(c=>c+'99') }},
    {{ label:'Jaccard', data:labels.map(l=>byVariant[l].jaccard), backgroundColor:colors.map(c=>c+'66') }},
    {{ label:'CodeBLEU', data:labels.map(l=>byVariant[l].codebleu), backgroundColor:colors.map(c=>c+'33') }},
  ]}}, options:{{ responsive:true, scales:{{y:{{min:0,max:1}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});
}})();

// Hit rate chart
(function(){{
  const el = document.getElementById('hitRateChart'); if (!el) return;
  new Chart(el, {{ type:'bar', data:{{ labels:['Hit@1','Hit@{retr.get("top_k",5)}','CWE Recall@{retr.get("top_k",5)}'],
    datasets:[{{ data:[{hit1r},{hitkr},{cwekr}], backgroundColor:['#4361EE','#06D6A0','#7209B7'] }}] }},
    options:{{ responsive:true, scales:{{y:{{min:0,max:1,ticks:{{callback:v=>(v*100)+'%'}}}}}}, plugins:{{legend:{{display:false}}}} }} }});
}})();

// Score distribution
(function(){{
  const el = document.getElementById('scoreDistChart'); if (!el) return;
  const scores = [{score_stats.get('min',0)},{score_stats.get('random_baseline',0)},{score_stats.get('median',0)},{score_stats.get('p75',0)},{score_stats.get('p90',0)},{score_stats.get('max',0)}];
  new Chart(el, {{ type:'line', data:{{ labels:['Min','Random BL','p50','p75','p90','Max'],
    datasets:[{{ label:'Top-1 Score', data:scores, borderColor:'#4361EE', backgroundColor:'#4361EE22', fill:true, tension:0.3, pointRadius:5 }}] }},
    options:{{ responsive:true, plugins:{{legend:{{display:false}}}} }} }});
}})();

// PRF chart
(function(){{
  const el = document.getElementById('prfChart'); if (!el) return;
  const thr = {json.dumps(thresholds)};
  const labels = thr.map(t => (t.label||t.threshold.toFixed(4))+'\\n'+t.threshold.toFixed(3));
  new Chart(el, {{ type:'line', data:{{ labels,
    datasets:[
      {{ label:'Precision', data:thr.map(t=>t.precision), borderColor:'#4361EE', tension:0.3 }},
      {{ label:'Recall', data:thr.map(t=>t.recall), borderColor:'#F72585', tension:0.3 }},
      {{ label:'F1', data:thr.map(t=>t.f1), borderColor:'#06D6A0', tension:0.3 }},
    ] }}, options:{{ responsive:true, scales:{{y:{{min:0,max:1}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});
}})();

// Confident count chart
(function(){{
  const el = document.getElementById('confCountChart'); if (!el) return;
  const thr = {json.dumps(thresholds)};
  new Chart(el, {{ type:'bar', data:{{ labels:thr.map(t=>t.label||t.threshold.toFixed(4)),
    datasets:[
      {{ label:'TP', data:thr.map(t=>t.tp), backgroundColor:'#06D6A0' }},
      {{ label:'FP', data:thr.map(t=>t.fp), backgroundColor:'#F72585' }},
    ] }}, options:{{ responsive:true, scales:{{x:{{stacked:true}},y:{{stacked:true}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});
}})();

// High-conf score chart
(function(){{
  const el = document.getElementById('highConfScoreChart'); if (!el) return;
  const entries = {json.dumps(hc_entries)};
  new Chart(el, {{ type:'bar', data:{{ labels:entries.map(e=>e.cve),
    datasets:[{{ label:'Top-1 Similarity Score', data:entries.map(e=>e.score),
      backgroundColor:entries.map(e=>e.score>0.5?'#c6282899':'#F7258599'),
      borderColor:entries.map(e=>e.score>0.5?'#c62828':'#F72585'), borderWidth:1 }}] }},
    options:{{ responsive:true, scales:{{y:{{min:0,max:1,title:{{display:true,text:'Similarity Score'}}}}}}, plugins:{{legend:{{display:false}}}} }} }});
}})();

// High-conf CWE doughnut
(function(){{
  const el = document.getElementById('highConfCweChart'); if (!el) return;
  const cweList = {json.dumps(hc_cwes)};
  const cweCounts = {{}};
  cweList.forEach(c => cweCounts[c] = (cweCounts[c]||0) + 1);
  const sorted = Object.entries(cweCounts).sort((a,b) => b[1]-a[1]);
  const colors = {json.dumps(_CHART_COLORS)};
  new Chart(el, {{ type:'doughnut', data:{{ labels:sorted.map(s=>s[0]), datasets:[{{data:sorted.map(s=>s[1]),backgroundColor:colors}}] }},
    options:{{ responsive:true, plugins:{{legend:{{position:'right'}}}} }} }});
}})();

// High-conf variant bar
(function(){{
  const el = document.getElementById('highConfVariantChart'); if (!el) return;
  const varList = {json.dumps(hc_variants)};
  const varCounts = {{}};
  varList.forEach(v => varCounts[v] = (varCounts[v]||0) + 1);
  const sorted = Object.entries(varCounts).sort((a,b) => b[1]-a[1]);
  new Chart(el, {{ type:'bar', data:{{ labels:sorted.map(s=>s[0]),
    datasets:[{{ label:'Failures', data:sorted.map(s=>s[1]), backgroundColor:sorted.map(s=>variantColors[s[0]]||'#999') }}] }},
    options:{{ responsive:true, scales:{{y:{{beginAtZero:true,ticks:{{stepSize:1}}}}}}, plugins:{{legend:{{display:false}}}} }} }});
}})();
</script>"""


# ── public API ───────────────────────────────────────────────────────

def generate_dashboard(run_dir: Path) -> Path:
    """Read evaluation outputs from *run_dir* and write evaluation_dashboard.html."""
    patch = _load_json(run_dir / "evaluation_summary.json") or {}
    retr = _load_json(run_dir / "retrieval_eval_summary.json") or {}
    conf = _load_json(run_dir / "confidence_eval_summary.json") or {}
    high_conf = _load_jsonl(run_dir / "confidence_eval_high_conf.jsonl")

    run_id = run_dir.name

    sections = [
        _build_overview(patch, retr, conf, high_conf, run_id),
        _build_patch_tab(patch),
        _build_retrieval_tab(retr),
        _build_confidence_tab(conf),
        _build_highconf_tab(high_conf, conf),
    ]

    tabs = [
        ("tab-overview", "Overview"),
        ("tab-patch", "Patch Quality"),
        ("tab-retrieval", "Retrieval"),
        ("tab-confidence", "Confidence"),
        ("tab-highconf", "High-Confidence Failures"),
    ]
    tab_btns = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-tab="{tid}">{label}</button>'
        for i, (tid, label) in enumerate(tabs)
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Evaluation Dashboard &mdash; {_esc(run_id)}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>{_CSS}</style>
</head>
<body>
<div class="topbar"><h1>Evaluation Dashboard</h1><span class="run-id">{_esc(run_id)}</span></div>
<nav class="tab-nav">{tab_btns}</nav>
<main class="content">
{"".join(sections)}
</main>
{_build_js(patch, retr, conf, high_conf)}
</body>
</html>"""

    out_path = run_dir / "evaluation_dashboard.html"
    out_path.write_text(page)
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation dashboard HTML.")
    parser.add_argument("run_dir", help="Path to run output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        raise SystemExit(1)

    out = generate_dashboard(run_dir)
    print(f"Dashboard written to: {out}")


if __name__ == "__main__":
    main()
