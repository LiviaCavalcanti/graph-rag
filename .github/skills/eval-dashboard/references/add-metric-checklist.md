# Checklist: Adding a New Metric

Use this as a step-by-step guide. Each checkbox is one edit location.

## Layer 1: Metric Definition (`src/metrics/similarity.py`)

- [ ] **Define the function** at the bottom of the metrics section
  - Single-value: `def my_metric(gen: str, ref: str) -> float`
  - Multi-value: `def my_metric(gen: str, ref: str) -> dict[str, float]`
  - Use `tokenize()` for code-aware tokenization (identifiers, operators, literals)
  - Use `ngrams(tokens, n)` if you need n-gram overlaps

## Layer 2: Evaluation Pipeline (`src/evaluate/evaluate_patches.py`)

- [ ] **Import** the function in the import block at the top
- [ ] **Add to `metrics_body`** dict in `evaluate_one()` (~line 92-108)
  - Single value: `"my_metric": round(my_metric(generated, gt_body), 4),`
  - Dict return: `**{k: round(v, 4) for k, v in my_metric(generated, gt_body).items()},`

## Layer 3: Dashboard (`experiments/dashboard_scripts/analyze_patches.py`)

There are **6 exact locations** to update:

### 3a. Aggregation in `analyze()`

- [ ] **`metric_keys` list** (~line 155) — add all new key names
  ```python
  metric_keys = [
      ...existing...,
      "my_metric",  # or "my_metric_f1", "my_metric_precision", etc.
  ]
  ```

- [ ] **`cwe_summary` dict** (~line 175) — add averages
  ```python
  cwe_summary[cwe] = {
      ...existing...,
      "avg_my_metric": _cwe_avg("my_metric"),
  }
  ```

- [ ] **`variant_summary` dict** (~line 190) — add averages
  ```python
  variant_summary[var] = {
      ...existing...,
      "avg_my_metric": _var_avg("my_metric"),
  }
  ```

### 3b. HTML rendering in `_render_html()`

- [ ] **`key_metrics` list** (~line 210) — controls summary table AND per-record cards
  ```python
  key_metrics = [
      ...existing...,
      ("My Metric", "my_metric"),
  ]
  ```

- [ ] **CWE table** — update BOTH header and data row
  ```python
  # Header:
  <tr><th>CWE</th>...<th>MyMetric</th></tr>

  # Data row (in the loop):
  f"<td style='color:{_score_color(stats.get('avg_my_metric'))}'>{_fmt(stats.get('avg_my_metric'))}</td>"
  ```

- [ ] **Variant table** — update BOTH header and data row (same pattern as CWE)

### 3c. CLI summary in `main()`

- [ ] **Print block** (~line 440) — add to the summary output
  ```python
  for label, key in [
      ...existing...,
      ("My Metric", "my_metric"),
  ]:
  ```

## Verification

- [ ] `python -c "from src.metrics.similarity import my_metric"` — import works
- [ ] Check for lint/type errors in both files
- [ ] Old evaluation.jsonl files gracefully show `-` for missing metrics
- [ ] New evaluations include the metric in `metrics_vs_function_body`

## Quick Reference: Existing Metric Keys

```
exact_match, normalised_exact_match,
char_sequence_ratio, line_sequence_ratio, normalised_edit_distance,
token_jaccard, token_jaccard_multiset,
bleu_1, bleu_2, bleu_4, codebleu_proxy,
bertscore_precision, bertscore_recall, bertscore_f1,
rouge1_f1, rouge2_f1, rougeL_f1, rougeL_precision, rougeL_recall
```
