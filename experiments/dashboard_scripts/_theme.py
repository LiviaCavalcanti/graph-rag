"""
Shared visual identity for all graph-RAG HTML dashboards.

Usage:
    from experiments.dashboard_scripts._theme import THEME_CSS, THEME_JS, score_color

    html = f"<style>{THEME_CSS}</style> ... <script>{THEME_JS}</script>"
"""
from __future__ import annotations

# ── Palette ──────────────────────────────────────────────────────────────
PALETTE = [
    "#4361EE",  # 0  blue        → combined / primary accent
    "#F72585",  # 1  pink        → codebert_seq
    "#7209B7",  # 2  violet      → codebert_pattern
    "#FB8500",  # 3  orange      → vuln_pattern
    "#06D6A0",  # 4  teal        → rgcn
    "#118AB2",  # 5  steel-blue  → extra
    "#FFD166",  # 6  yellow
    "#EF476F",  # 7  coral
]

# ── Shared CSS ───────────────────────────────────────────────────────────
THEME_CSS = """\
/* ─── graph-RAG dashboard theme ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --c0: #4361EE; --c1: #F72585; --c2: #7209B7; --c3: #FB8500;
  --c4: #06D6A0; --c5: #118AB2; --c6: #FFD166; --c7: #EF476F;

  --bg:           #f5f4f0;
  --surface:      #ffffff;
  --ink:          #1a1a2e;
  --muted:        #6b7280;
  --accent:       var(--c0);
  --accent-light: #eef0fd;
  --border:       #e2e0db;
  --ok:           #146c43;
  --ok-bg:        #e8faf3;
  --warn:         #c62828;
  --warn-bg:      #fff0f3;

  --radius:       10px;
  --radius-sm:    6px;
  --radius-pill:  999px;
  --shadow:       0 2px 12px rgba(10, 10, 40, .06);
  --shadow-lg:    0 4px 24px rgba(10, 10, 40, .08);

  --font:         "Inter", "Segoe UI", system-ui, sans-serif;
  --font-mono:    "JetBrains Mono", "Fira Code", ui-monospace, monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--ink);
  font-size: 14px;
  line-height: 1.6;
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre { font-family: var(--font-mono); }

/* ─── Topbar ─── */
.topbar {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #fff;
  padding: 18px 32px;
  display: flex; align-items: center; gap: 16px;
}
.topbar h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: -.4px; }
.topbar .run-id {
  font-size: .78rem; background: rgba(255,255,255,.13);
  padding: 3px 10px; border-radius: var(--radius-pill);
}

/* ─── Page header (non-topbar variant) ─── */
.page-header {
  max-width: 1300px; margin: 28px auto 0; padding: 0 24px;
}
.page-header h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }
.page-header .meta { color: var(--muted); font-size: .88rem; margin-bottom: 18px; }

/* ─── Tab navigation ─── */
.tab-nav {
  background: var(--surface);
  border-bottom: 2px solid var(--border);
  display: flex; padding: 0 32px; gap: 4px;
  position: sticky; top: 0; z-index: 10;
  box-shadow: 0 2px 8px rgba(0,0,0,.04);
}
.tab-btn {
  padding: 11px 18px; border: none; background: none;
  cursor: pointer; font-size: .9rem; font-weight: 500; color: var(--muted);
  border-bottom: 3px solid transparent; margin-bottom: -2px;
  transition: color .15s, border-color .15s;
}
.tab-btn:hover { color: var(--ink); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

/* ─── Content wrapper ─── */
.content { max-width: 1300px; margin: 0 auto; padding: 24px 24px 64px; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ─── Typography ─── */
h2 { font-size: 1.35rem; font-weight: 700; margin-bottom: 6px; }
h3 { font-size: 1rem; font-weight: 600; margin-bottom: 10px; }
h4 { font-size: .88rem; font-weight: 600; margin: 8px 0 6px; }
p.sub { color: var(--muted); margin-bottom: 18px; font-size: .9rem; }
p.muted { color: var(--muted); }

/* ─── Card ─── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 22px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}

/* ─── Stat cards ─── */
.stat-row { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }
.stat-row.compact .stat-card { min-width: 100px; }
.stat-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 18px;
  min-width: 150px; box-shadow: var(--shadow); flex: 1;
}
.stat-card.accent { background: var(--accent-light); border-color: var(--accent); }
.stat-label {
  font-size: .72rem; text-transform: uppercase;
  letter-spacing: .06em; color: var(--muted); margin-bottom: 2px;
}
.stat-value { font-size: 1.75rem; font-weight: 700; color: var(--ink); line-height: 1.2; }
.stat-sub { font-size: .78rem; color: var(--muted); margin-top: 2px; }

/* ─── Tables ─── */
table { width: 100%; border-collapse: collapse; font-size: .88rem; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
th {
  background: var(--bg); font-weight: 600;
  font-size: .76rem; text-transform: uppercase; letter-spacing: .05em;
  position: sticky; top: 0;
}
tr:hover td { background: #faf9f6; }
.scroll-x { overflow-x: auto; }
.matrix td { text-align: center; font-size: .8rem; }

/* ─── Sparkbar ─── */
.sparkbar-wrap { display: inline-flex; align-items: center; gap: 6px; }
.sparkbar { display: inline-block; height: 10px; border-radius: 3px; }
.sparkbar-label { font-size: .8rem; color: var(--ink); }

/* ─── Badges / pills / dots ─── */
.dot {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; margin-right: 5px; vertical-align: middle;
}
.badge {
  display: inline-block; padding: 2px 8px;
  border-radius: var(--radius-pill); font-size: .73rem; font-weight: 600;
}
.pill {
  display: inline-block; padding: 2px 8px;
  border-radius: var(--radius-pill); font-size: .78rem;
  background: var(--accent-light); color: var(--accent);
}
.pill.ok   { background: var(--ok-bg);   color: var(--ok); }
.pill.warn { background: var(--warn-bg); color: var(--warn); }
.pill-big {
  display: inline-block; padding: 4px 12px;
  border-radius: var(--radius-pill); font-size: .88rem; font-weight: 700;
  background: #1a1a2e; color: #fff;
}
.delta { font-size: .78rem; font-weight: 600; padding: 1px 6px; border-radius: 4px; }
.delta.up   { background: var(--ok-bg);   color: var(--ok); }
.delta.down { background: var(--warn-bg); color: var(--warn); }

/* ─── Grid layouts ─── */
.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 16px; margin-bottom: 16px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.code-triple {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px; margin-top: 12px;
}
@media (max-width: 1000px) { .code-triple { grid-template-columns: 1fr; } }

/* ─── Code blocks (dark) ─── */
.code-pane { background: #0d1117; border-radius: var(--radius); overflow: hidden; }
.pane-label {
  background: #161b22; color: #8b949e;
  font-size: .73rem; padding: 7px 14px;
}
.code-block {
  background: #0d1117; color: #c9d1d9;
  padding: 14px; font-size: .8rem;
  font-family: var(--font-mono);
  overflow-x: auto; max-height: 400px; overflow-y: auto;
  white-space: pre;
  border-radius: 0 0 var(--radius) var(--radius);
}
.code-block.small { max-height: 200px; font-size: .75rem; }
.code-block.muted { color: #555; font-style: italic; }

/* ─── Code blocks (light — for diffs / patches) ─── */
.code-block-light {
  background: #fafaf8; border: 1px solid var(--border);
  border-radius: var(--radius-sm); overflow: auto;
}
.code-block-light h4 {
  margin: 0; padding: 6px 10px;
  background: var(--bg); font-size: .8rem; border-bottom: 1px solid var(--border);
}
.code-block-light pre {
  margin: 0; padding: 10px 12px; font-size: .8rem;
  font-family: var(--font-mono);
  white-space: pre-wrap; word-wrap: break-word;
  max-height: 400px; overflow-y: auto;
}

/* ─── Retrieval info block ─── */
.retrieval-info {
  background: var(--accent-light); padding: 8px 12px;
  border-radius: var(--radius-sm); margin-bottom: 10px;
  font-size: .88rem;
}

/* ─── Code explorer ─── */
.query-select {
  margin-left: .5rem; padding: 6px 10px;
  border-radius: var(--radius-sm); border: 1px solid var(--border);
  font-size: .88rem; font-family: var(--font);
}
.code-card { margin-top: 16px; }
.code-card-header {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 12px; flex-wrap: wrap;
}
.code-split {
  display: grid; grid-template-columns: 1fr 2fr;
  gap: 14px; align-items: start;
}
.code-results { display: flex; flex-direction: column; gap: 10px; }
.approach-block {
  border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden;
}
.approach-header {
  background: #fafaf8; padding: 9px 14px;
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}
.retrieved-item { padding: 9px 14px; border-top: 1px solid var(--border); }
.retrieved-meta {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 4px; flex-wrap: wrap;
}
.rank { font-size: .78rem; font-weight: 700; color: var(--muted); min-width: 24px; }

/* ─── Miss analysis ─── */
.cell-header {
  display: flex; align-items: center; gap: 6px;
  margin-bottom: 12px; font-size: .92rem;
}
.miss-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 16px; margin-bottom: 12px;
}

/* ─── Expandable details ─── */
details > summary {
  cursor: pointer; font-weight: 600; user-select: none;
  padding: 4px 0;
}
details > summary:hover { color: var(--accent); }

/* ─── Utility ─── */
ul { margin-top: 6px; }
li { margin-bottom: 4px; }
.text-center { text-align: center; }
.text-mono { font-family: var(--font-mono); font-size: .85rem; }
"""

# ── Tab-switching JS ─────────────────────────────────────────────────────
THEME_JS = """\
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector('[data-tab="' + tabId + '"]').classList.add('active');
  document.getElementById(tabId).classList.add('active');
}
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.tab-btn').forEach(function(btn) {
    btn.addEventListener('click', function() { switchTab(btn.getAttribute('data-tab')); });
  });
});
"""


def score_color(value: float, *, low: float = 0.0, high: float = 1.0) -> str:
    """Return an ``rgb(...)`` string on a red → yellow → green ramp."""
    if high == low:
        norm = 0.5
    else:
        norm = max(0.0, min(1.0, (value - low) / (high - low)))
    r = int(220 * (1 - norm))
    g = int(60 + 160 * norm)
    return f"rgb({r},{g},50)"


def heatmap_color(value: float, *, low: float = 0.0, high: float = 1.0) -> str:
    """Return a background color for heatmap cells (white → accent-tinted)."""
    if high == low:
        norm = 0.0
    else:
        norm = max(0.0, min(1.0, (value - low) / (high - low)))
    r = int(255 - 100 * norm)
    g = int(255 - 25 * norm)
    b = int(255 - 100 * norm)
    return f"rgb({r},{g},{b})"
