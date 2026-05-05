---
name: eval-dashboard
description: 'Modify the patch evaluation pipeline and HTML dashboards. USE FOR: adding new metrics, updating dashboard tables/cards, changing evaluation scoring, updating CWE/variant breakdowns, modifying HTML rendering, extending CLI summaries. Keywords: metric, score, evaluate, dashboard, BLEU, ROUGE, BERTScore, CodeBLEU, Jaccard, similarity, patches, CWE breakdown, variant breakdown.'
argument-hint: 'Describe the metric or dashboard change you need'
---

# Evaluation Pipeline & Dashboard Modification

## When to Use

- Adding a new similarity metric to the evaluation pipeline
- Displaying new or existing metrics in the HTML dashboard
- Updating aggregate breakdowns (by CWE, by variant)
- Modifying per-record detail cards
- Changing the CLI summary output

## Architecture Overview

The evaluation system has 3 layers that must stay in sync:

```
src/metrics/similarity.py          ← Define metric functions
        ↓
src/evaluate/evaluate_patches.py   ← Compute metrics per patch
        ↓
experiments/dashboard_scripts/analyze_patches.py  ← Display in dashboard
```

Supporting files:
- `experiments/dashboard_scripts/_theme.py` — CSS/JS theme, `score_color()` for coloring, `heatmap_color()`
- `src/evaluate/preprocessing.py` — `extract_function_body()` for ground-truth extraction
- `src/data/autopatch.py` — `AutoPatchDataset` for loading CVE data

See [architecture reference](./references/architecture.md) for full details.

## Procedure: Adding a New Metric

Follow the [metric addition checklist](./references/add-metric-checklist.md) step by step. Summary:

### Step 1 — Define the metric (`src/metrics/similarity.py`)

Create a function returning either a single `float` or a `dict[str, float]` for multi-value metrics (precision/recall/F1). Use the existing `tokenize()` helper for code-aware tokenization.

### Step 2 — Wire into evaluation (`src/evaluate/evaluate_patches.py`)

1. Add import
2. Add to `metrics_body` dict in `evaluate_one()` — round to 4 decimal places
3. For dict-return metrics, use spread: `**{k: round(v, 4) for k, v in my_metric(gen, ref).items()}`
4. For single-value metrics: `"my_metric": round(my_metric(gen, ref), 4)`

### Step 3 — Update the dashboard (`experiments/dashboard_scripts/analyze_patches.py`)

There are **6 locations** that must be updated — see the checklist for exact details:

1. `metric_keys` list in `analyze()` — enables aggregation
2. `cwe_summary` dict in `analyze()` — adds to CWE breakdown data
3. `variant_summary` dict in `analyze()` — adds to variant breakdown data
4. `key_metrics` list in `_render_html()` — controls summary table AND per-record cards
5. CWE/Variant HTML table headers (`<th>`) AND data rows (`<td>`) in `_render_html()`
6. CLI summary `print` block in `main()`

### Step 4 — Verify

- Run `python -c "from src.metrics.similarity import <new_func>"` to verify import
- Check for lint errors in both files
- Old evaluation.jsonl files will show `-` for new metrics (handled gracefully)

## Critical Patterns

### Metric return conventions
- Single value: `def my_metric(gen: str, ref: str) -> float`
- Multi-value: `def my_metric(gen: str, ref: str) -> dict[str, float]` with flat keys like `{metric_precision, metric_recall, metric_f1}`

### Dashboard coloring
All numeric metric values use `_score_color(value)` from `_theme.py` — maps 0→red, 0.5→yellow, 1→green. Apply via inline `style='color:{_score_color(val)}'`.

### Formatting
Use `_fmt(val, ndigits=4)` for all displayed values — handles None→`"-"`, bool→`"yes"/"no"`, float→formatted string.

### CWE/Variant breakdown pattern
```python
# In analyze() — follow this pattern for each new metric:
"avg_my_metric": _cwe_avg("my_metric_key"),  # add to cwe_summary dict
"avg_my_metric": _var_avg("my_metric_key"),  # add to variant_summary dict
```

### HTML table pattern
Both header and data rows must be updated together:
```python
# Header (one location):
<th>MyMetric</th>

# Data row (in the loop):
f"<td style='color:{_score_color(stats.get('avg_my_metric'))}'>{_fmt(stats.get('avg_my_metric'))}</td>"
```

## Common Mistakes

1. **Forgetting the HTML `<th>` header** — columns shift and data appears under wrong headers
2. **Not rounding metric values** — always `round(value, 4)` in evaluate_patches
3. **Missing one of the 6 dashboard locations** — metric appears in some views but not others
4. **Using backticks in f-strings** — use regular quotes for HTML attributes
5. **Not handling missing keys** — old data won't have new metrics; `.get()` with `_fmt()` handles this
