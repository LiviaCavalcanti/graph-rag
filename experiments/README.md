# Experiments

This directory contains the evaluation pipeline for the graph-RAG retrieval system. The goal is to answer a concrete question: *given a vulnerable function, can the system retrieve the correct CVE from a corpus?* The pipeline runs multiple embedding approaches against multiple retrieval backends, measures how well each one does, and then asks whether combining them can recover the cases any single approach misses.

---

## Quick start

All scripts are meant to be run from the **repo root**, not from inside `experiments/`.

```bash
# 1. Run a full experiment (embeds corpus, builds indices, evaluates all cells)
uv run python -m experiments.runner

# 2. Analyse miss patterns and build the miss dashboard
uv run python experiments/analyze_misses.py experiments/output/<run_id>/results.json

# 3. Run the crossing / fusion strategy analysis and build the unified dashboard
uv run python experiments/verify_crossing.py experiments/output/<run_id>/results.json
```

Step 3 produces the main `dashboard.html` that you probably want to open in a browser. Steps 1–2 are prerequisites — step 3 will complain if `results.json` isn't there.

If you want to regenerate the dashboard without re-running the analysis (e.g., after changing `dashboard.py`):

```python
from experiments.dashboard import generate_html_dashboard
generate_html_dashboard("experiments/output/<run_id>")
```

---

## Scripts

### `runner.py`

The main experiment entry point. It sweeps a grid of `(embedder × backend)` combinations, embeds the full CVE corpus once per embedder, builds a retrieval index for each backend, and then runs self-retrieval evaluation on every cell. Results land in `experiments/output/<run_id>/results.json`.

A "cell" is one `(embedder, backend)` pair. Each cell records:

- **Embedding cost** — how long it took to embed the corpus
- **Index build time** — how long the ANN index took to build
- **Query latency** — p50, p95, p99 in milliseconds
- **Space stats** — mean pairwise cosine similarity, effective dimensionality
- **Self-retrieval** — Hit@1, Hit@5, Hit@10, MRR
- **CWE recall** — how well the embedding clusters by vulnerability class
- **Leave-one-out** — a stricter retrieval test (can be slow; skipped for large corpora)
- **Raw queries** — the full retrieved list for every query, used by downstream analysis

The run ID is a timestamp + short hash, e.g. `20260416_100627_bba620`.

---

### `analyze_misses.py`

Takes `results.json` and digs into the failure cases. For every approach it asks:

- When the top-1 CVE is wrong, is the **CWE still correct**? (Wrong CVE, right vulnerability class — a "soft hit")
- How far down the ranking does the **true CVE** appear?
- When the system is wrong, is it at least **uncertain about it**? (Low confidence → wrong is more forgivable than high confidence → wrong)

Produces:
- `miss_analysis.json` — structured per-cell statistics
- `miss_dashboard.html` — standalone HTML with charts and tables

---

### `verify_crossing.py`

The most interesting analysis. The hypothesis is that different embedding approaches make *different* mistakes — so if you could combine them intelligently, you might do better than any single approach. This script tests several combination strategies:

| Strategy | How it works |
|---|---|
| **individual_best** | Just the best single approach (baseline) |
| **oracle** | Hit if *any* approach gets it right — the theoretical ceiling |
| **majority_vote** | Pick whichever CVE most approaches rank first |
| **confidence_weighted_vote** | Like majority vote, but uncertain approaches get down-weighted (0.25×) |
| **reciprocal_rank_fusion** | Classic RRF: score = Σ 1/(60 + rank) across all approaches |
| **fallback_cascade** | Try the best approach first; if it's uncertain, fall back to the next confident one |

"Uncertain" means the top score is below a probability floor *or* the margin between rank-1 and rank-2 is too small (uses a softmax + thresholds that match the miss analysis defaults).

The cascade order is computed automatically — approaches are sorted by their Hit@1 rate, so the most reliable one gets tried first.

Produces:
- `crossing_analysis.json` — full results including per-query strategy outcomes
- `dashboard.html` (via `dashboard.py`) — the unified dashboard

---

### `dashboard.py`

Reads `results.json`, `miss_analysis.json`, and `crossing_analysis.json` from a run directory and produces a single self-contained `dashboard.html` with four tabs:

- **Overview** — Hit@1, MRR, and a leaderboard across all approaches
- **Deep Dive** — per-approach charts (latency, space stats, miss breakdown)
- **Crossing Strategies** — how the fusion strategies compare, pairwise complementarity heatmap, combined-approach miss deep-dive
- **Code Explorer** — query-by-query inspection: the vulnerable function on the left, what each approach retrieved on the right

The Code Explorer is particularly useful for manual analysis — you can filter the query list by which fusion strategy *failed* on it, then step through those queries one by one to see what each individual approach retrieved and why the combination went wrong.

---

### `metrics.py`

Utility functions used by `runner.py`. Self-retrieval metrics (Hit@k, MRR), embedding space statistics (mean pairwise similarity, intrinsic dimensionality), and leave-one-out evaluation.

---

### `visualization.py`

Generates static matplotlib/seaborn figures for a run:

- Performance dashboard (embedding cost, latency)
- Retrieval quality dashboard (Hit@k bars, MRR)
- Embedding space dashboard (UMAP projection, similarity distribution)
- Combined comparison across all cells

These land in `output/<run_id>/visualizations/`.

---

### `visualize_diagnostics.py`

Reads the per-query diagnostic data and produces charts specifically about failure modes — where the true CVE lands when missed, confidence distributions, CWE-level patterns.

---

## Output directory structure

```
experiments/output/<run_id>/
    results.json            — raw experiment results (all cells, all queries)
    summary.json            — high-level stats (generated by runner)
    miss_analysis.json      — miss/uncertainty breakdown (from analyze_misses.py)
    crossing_analysis.json  — fusion strategy results (from verify_crossing.py)
    dashboard.html          — unified interactive dashboard
    miss_dashboard.html     — miss-focused dashboard
    indices/                — saved FAISS/HNSW index files
    visualizations/         — static PNG figures
    diagnostics/            — per-approach diagnostic JSON files
```

---

## Reading the results

### Hit@1 and MRR

Hit@1 is the fraction of queries where the correct CVE is the very first result. It's the strictest and most useful metric — in a real workflow, you'd look at the top result and decide whether it matches. MRR (Mean Reciprocal Rank) is softer: if the correct CVE is at rank 3, it still gets a score of 1/3.

A random baseline on a corpus of N items would score roughly 1/N on Hit@1. The numbers here should be dramatically better than that.

### Fusion strategies — what the numbers mean

The oracle rate tells you the *ceiling* for this set of approaches — it answers "if we had a perfect oracle to pick the right approach for each query, what's the best we could do?" The gap between oracle and individual_best is the potential gain from better combination.

The other strategies are attempts to close that gap without cheating. RRF tends to be robust and is a good default — it doesn't need any uncertainty estimates, just the ranked lists. The fallback cascade is more aggressive: it bets on the best single approach and only falls back when that approach looks shaky.

If majority_vote beats individual_best, that's a signal the approaches are genuinely diverse and a consensus is valuable. If it doesn't, the approaches are probably making the same mistakes.

### Pairwise complementarity

The table in the Crossing Strategies tab shows, for every pair of approaches, the union Hit@1 — i.e., what fraction of queries at least one of the two gets right. Numbers in parentheses (e.g. `6+`) are queries that approach B rescues from approach A's misses.

High union rates with high individual rates means the approaches agree. High union rates with *lower* individual rates means they're complementary — each catches different cases. That's where fusion is most valuable.

### Combined miss deep-dive

"Combined" refers to the best-performing individual approach. The deep-dive table lists every query that approach got wrong and shows:

- How far down the true CVE was (if it appeared at all)
- Whether the approach was uncertain on that query
- Which other approaches got it right
- Which fusion strategies recovered it

If most misses have `rescued_by=[]`, the whole ensemble is struggling on those queries — likely hard cases where the code is ambiguous or the embedding space just doesn't separate them well.

---

## Adding a new embedder

1. Implement the embedder in `src/embeddings/` following the existing interface
2. Register it in `src/embeddings/__init__.py` via `build_embedders()`
3. Re-run `experiments/runner.py` — it'll pick it up automatically

---

## Adding a new fusion strategy

1. Write a function in `verify_crossing.py` that takes `per_approach: dict[str, dict]` and returns the predicted CVE string (or `None`)
2. Add its name to the `strategies` list and call it in the per-query loop, following the existing pattern
3. Re-run `verify_crossing.py` — the dashboard will include it automatically
