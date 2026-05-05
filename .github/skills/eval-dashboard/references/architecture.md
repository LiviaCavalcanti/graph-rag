# Architecture Reference

## File Map

| File | Role | Key exports |
|------|------|-------------|
| `src/metrics/similarity.py` | All metric functions | `tokenize`, `exact_match`, `normalised_exact_match`, `sequence_matcher_ratio`, `line_level_ratio`, `normalised_edit_distance`, `token_jaccard`, `token_jaccard_multiset`, `bleu_score`, `rouge_scores`, `rouge_n`, `rouge_l`, `codebleu_weighted`, `bertscore_pair`, `compute_diff_details` |
| `src/evaluate/evaluate_patches.py` | Per-patch evaluation | `evaluate_one()`, `aggregate()`, `strip_c_comments()`, CLI `main()` |
| `src/evaluate/preprocessing.py` | Code extraction | `extract_function_body()` |
| `src/data/autopatch.py` | CVE data loading | `AutoPatchDataset` (`.stream()`, `.load_lightweight()`, `.find_cve_dir()`, `.load_ground_truth()`, `.load_db_cache()`) |
| `experiments/dashboard_scripts/analyze_patches.py` | Patch dashboard | `analyze()`, `_render_html()`, `_build_record()`, CLI `main()` |
| `experiments/dashboard_scripts/_theme.py` | Shared theme | `THEME_CSS`, `THEME_JS`, `PALETTE`, `score_color()`, `heatmap_color()` |

## Data Flow

```
CVE-list/<cve>/original_code.txt          → input vulnerable code
CVE-list/<cve>/out_v2/code/*_fixed.c      → ground-truth patches (per variant)

results.jsonl                              → batch inference output
  {query_cve, query_cwe, query_variant, example_cve, example_variant,
   generated_patch, retrieval{}, status, elapsed_s, similarity, cve_match, cwe_match}

evaluation.jsonl                           → evaluate_patches output
  {query_cve, query_variant, eval_status, metrics_vs_function_body{...}, size_info{...}}

analyze_patches merges both → patch_analysis.html + patch_analysis.json
```

## evaluate_one() Return Structure

```python
{
    # identifiers
    "query_cve", "query_cwe", "query_variant",
    "example_cve", "example_variant",
    "status", "elapsed_s", "eval_status",

    # primary metrics (all in metrics_vs_function_body)
    "metrics_vs_function_body": {
        "exact_match": bool,
        "normalised_exact_match": bool,
        "char_sequence_ratio": float,      # SequenceMatcher on chars
        "line_sequence_ratio": float,      # SequenceMatcher on lines
        "normalised_edit_distance": float, # Levenshtein / max(len)
        "token_jaccard": float,            # set intersection / union
        "token_jaccard_multiset": float,   # multiset intersection / union
        "bleu_1": float, "bleu_2": float, "bleu_4": float,
        "codebleu_proxy": float,           # 0.5*BLEU + 0.25*Jaccard + 0.25*LineRatio
        "bertscore_precision": float,      # CodeBERT-based
        "bertscore_recall": float,
        "bertscore_f1": float,
        "rouge1_f1": float,               # unigram overlap F1
        "rouge2_f1": float,               # bigram overlap F1
        "rougeL_f1": float,               # LCS-based F1
        "rougeL_precision": float,
        "rougeL_recall": float,
    },

    # secondary metrics (against full file, not function body)
    "metrics_vs_full_file": {
        "full_file_char_ratio", "full_file_token_jaccard", "full_file_bleu_4"
    },

    "size_info": {
        "generated_lines", "ground_truth_lines",
        "generated_tokens", "ground_truth_tokens",
        "line_count_diff", "token_count_diff"
    },

    "diff_details": { "num_hunks", "hunks", ... }
}
```

## analyze_patches.py Structure

### analyze() returns:
```python
{
    "source": {"results": path, "evaluation": path},
    "total_records": int,
    "aggregates": {metric_key → {n, mean, median, min, max}},
    "by_cwe": {cwe → {count, avg_bleu_4, avg_bertscore_f1, avg_token_jaccard, avg_codebleu_proxy, avg_rouge1_f1, avg_rouge2_f1, avg_rougeL_f1}},
    "by_variant": {variant → same structure as by_cwe},
    "records": [merged record dicts with input_code, ground_truth, generated_patch]
}
```

### _render_html() sections:
1. **Aggregate Scores table** — driven by `key_metrics` list of `(label, metric_key)` tuples
2. **By CWE table** — headers + data rows from `by_cwe`
3. **By Variant table** — headers + data rows from `by_variant`
4. **Per-Record cards** — `<details>` elements with score table (reuses `key_metrics`) + code triple

### _theme.py helpers:
- `score_color(value)` → `rgb(r,g,b)` string: 0=red(220,60,60), 0.5=yellow(110,140,60), 1=green(0,220,60)
- `heatmap_color(value)` → background color for cells
- `_fmt(val, ndigits=4)` is in analyze_patches.py itself (not _theme.py)

## Other Dashboard Scripts

| Script | Input | Output |
|--------|-------|--------|
| `dashboard.py` | retrieval_results.jsonl | 5-tab retrieval dashboard (overview, per-CWE, similarity, retrieval details, rank analysis) |
| `analyze_misses.py` | retrieval_results.jsonl | Miss analysis: confidence, rank, CWE match for wrong retrievals |
| `analyze_spaces.py` | embedding vectors | Space quality: trustworthiness, isotropy, class separation, CKA heatmaps |
| `comparison_dashboard.py` | multiple runs | Cross-embedder comparison by CWE |
| `visualization.py` | experiment results | Matplotlib performance plots |
| `visualize_diagnostics.py` | graph data | Node/edge distribution, t-SNE plots |
