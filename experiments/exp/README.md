# experiments/exp — Grid Experiments

This directory contains experiment scripts built on the `Experiment` base class
(see [`experiments/base.py`](../base.py)).  Each experiment defines a grid of
axes (independent variables), runs one pipeline cell per axis combination, and
collects declarative metrics into a canonical `results.json`.

All scripts are invoked through the unified CLI coordinator:

```bash
python -m experiments.experiment <command> [options]
```

---

## Architecture recap

```
Experiment (base class — base.py)
│
├── axes(cfg)         → list[Axis]          # grid dimensions
├── load_data(cfg)    → dict                # shared data (pairs, split, …)
├── run_cell(ctx)     → dict                # execute one grid cell
├── metrics()         → list[MetricSpec]    # declarative post-cell metrics
├── before_run(ctx)   → None                # setup hook
└── after_run(output) → None                # dashboard / summary hook
```

The orchestrator calls `load_data` once, then iterates over
`product(*axis_values)`, calling `run_cell` and evaluating `MetricSpec`s for
each combination.  Results land in `experiments/output/<run_id>/`.

---

## Experiments at a glance

| File | CLI command | Grid axes | What it measures |
|---|---|---|---|
| `retrieval_experiment.py` | `retrieval` | embedder × backend × graph\_variant | Embedding quality, retrieval hit\@k, CWE recall |
| `slicing_comparison.py` | `slicing` | graph\_variant × embedder | Slicing & labelling ablation (2×2 factorial) |
| `combining_experiment.py` | `combining` | strategy | Fusion of multiple embedders (concat, PCA, norm) |
| `train_gin_codebert.py` | *(standalone)* | — | Train GIN-CodeBERT with CWE-level triplet loss |
| `train_gin_struct.py` | *(standalone)* | — | Train GIN-Struct with CVE-level triplet loss |
| `test_gin_struct_fusion.py` | *(standalone)* | — | Quick A/B: frozen GIN vs trained GIN-Struct fusion |
| `prompt/patching_experiment.py` | `agent --agent-mode patch` | retriever\_mode × model\_name | LLM patch generation (CodeBLEU, ROUGE, exact match) |
| `prompt/test_default_v2.py` | *(standalone)* | — | Re-run failed CVEs with `default_v2` prompt |
| `prompt/test_graph_v2.py` | *(standalone)* | — | Re-run failed CVEs with `graph_v2` prompt |

---

## Detailed descriptions

### `retrieval_experiment.py` — Retrieval grid

Reimplements the legacy `runner.py` on top of the `Experiment` base class.

**Grid axes:**

| Axis | Values (from config) | Description |
|---|---|---|
| `embedder` | codebert, netlsd, wl, gin, … | Embedding approach |
| `backend` | hnsw | Vector index implementation |
| `graph_variant` | G\_vuln (default) | Which graph representation to embed |

**Metrics (declarative):**

| Metric | What it measures |
|---|---|
| `space_stats` | Effective dimensionality, mean pairwise similarity |
| `latency` | Query latency p50 / p95 / p99 |
| `self_retrieval` | CVE-level hit\@k, MRR, nDCG, MAP |
| `cwe_recall` | Macro-average recall grouped by CWE class |
| `leave_one_out` | *Optional, slow* — most honest evaluation |

**Implementation notes:**

- Embeddings are cached per `(embedder, graph_variant)` so switching backends
  doesn't re-embed.
- PCA is fitted on index embeddings only; queries are projected through the
  same transform.
- Produces both the new base-class `results.json` and a legacy-compatible flat
  summary for downstream analysis scripts.

**CLI:**

```bash
python -m experiments.experiment retrieval [--config config.yaml] [--loo]
```

---

### `slicing_comparison.py` — Graph variant ablation

Compares four graph representations in a 2×2 factorial design:

|  | No diff labels | Diff labels |
|---|---|---|
| **Full graph** | `G_before` | `G_before_labeled` |
| **Fingerprint-sliced** | `G_vuln_no_labels` | `G_vuln` |

Each variant is embedded with every configured embedder.  The experiment
answers: *does slicing help? do diff labels help? do they interact?*

**Query variant modes:**

| Mode | Behaviour |
|---|---|
| *(default)* | Queries use the same variant as the index |
| `--query-variant runner_compat` | Queries always use `G_vuln` (reproduces checkpoint-2 protocol) |
| `--query-variant G_before` | All queries fixed to `G_before` |

**Metrics:** CVE hit\@k, MRR, CVE F1, CWE recall, space quality (effective
dimension, mean similarity).

**Implementation notes:**

- Variant factories build graph variants on the fly (strip diff attrs,
  transfer labels from G\_vuln → G\_before).
- PCA state is reset per variant to prevent cross-contamination.
- Degenerate embeddings (all-zero) are caught and reported as errors without
  aborting the grid.

**CLI:**

```bash
python -m experiments.experiment slicing [--repeat N] [--query-variant runner_compat]
```

---

### `combining_experiment.py` — Embedding fusion strategies

Evaluates how to combine multiple embedders (NetLSD + WL + GIN ± CodeBERT)
into a single vector.

**Strategies (single grid axis):**

| Strategy | Description |
|---|---|
| `concat_pca` | Concatenate raw → PCA to `dim` |
| `pca_concat_pca` | PCA each → concatenate → PCA |
| `pca_concat` | PCA each to `dim/3` → concatenate (no second PCA) |
| `norm_concat_pca` | L2-norm each → concatenate → PCA |
| `norm_concat_pca_gin_struct` | Same, but with trained GIN-Struct weights |
| `4way_concat_pca` | 4-way (+ CodeBERT-pattern) → concatenate → PCA |
| `4way_norm_concat_pca` | 4-way L2-norm → concatenate → PCA |
| `gin_codebert` | GIN-CodeBERT embedder (standalone) |

**Metrics:**

| Metric | What it measures |
|---|---|
| `intrinsic` | Effective dim, mean similarity, isotropy, hubness, distance concentration |
| `retrieval` | CVE hit\@k, MRR, nDCG\@10, MAP\@10 |
| `class_separation` | Intra/inter CWE distance ratio |
| `alignment_uniformity` | Alignment vs. uniformity tradeoff |

**Implementation notes:**

- Sub-embedders (NetLSD, WL, GIN) are shared and frozen across strategies.
- 4-way strategies PCA CodeBERT-pattern down to `dim` first to prevent the
  802-d CodeBERT from dominating the fused vector.
- Pairwise comparison (CKA, kNN overlap, rank correlation) is computed between
  all strategy pairs in `after_run`.

**CLI:**

```bash
python -m experiments.experiment combining [--config config.yaml]
```

---

### `train_gin_codebert.py` — GIN-CodeBERT training

Trains a GIN model whose node features come from CodeBERT (768-d per node).
The training objective is **triplet loss** with CWE-ID as class label:
same-CWE pairs are positives, different-CWE pairs are negatives.

**Pipeline:**

1. Load pairs → split into index / query
2. Encode node features with `NodeCodeBERTEncoder` (cached)
3. Build `TripletDataset` grouped by CWE
4. Train `GINCodeBERTModel` (configurable hidden dim, layers, output dim)
5. Evaluate on query set: CWE recall + CVE hit\@k
6. Save `gin_codebert_checkpoint.pt`

**Key config (`gin_codebert.training`):** `epochs`, `lr`, `margin`,
`hidden_dim`, `output_dim`, `num_layers`.

**CLI:**

```bash
python -m experiments.exp.train_gin_codebert [--config config.yaml] [--epochs 50] [--output dir]
```

---

### `train_gin_struct.py` — GIN-Struct training

Trains a structural GIN model (11-d node-type features) with **triplet loss**.
Groups positives by **CVE-ID** — variant pairs of the same vulnerability are
pulled together.

**Key differences from GIN-CodeBERT:**

| Property | GIN-CodeBERT | GIN-Struct |
|---|---|---|
| Node features | 768-d CodeBERT | 11-d node type |
| Positive grouping | CWE-ID | CVE-ID (or `dir_name`) |
| Warm-start | No | Optional (from frozen GIN) |

**Warm-start:** Copies weights from the frozen GIN checkpoint into the
trainable model.  Preserves learned graph geometry while allowing fine-tuning;
reduces training time.

**Label modes:**

- `cve` *(default)* — Variants of the same CVE are positives.
- `dir_name` — All variants of the same function (directory) are positives.

**CLI:**

```bash
python -m experiments.exp.train_gin_struct [--config config.yaml] [--epochs 100] \
    [--label-mode cve|dir_name] [--no-warm-start]
```

---

### `test_gin_struct_fusion.py` — Quick fusion comparison

A lightweight test script that embeds the same data with two fusion strategies
side by side:

1. **`norm_concat_pca`** — uses the frozen (untrained) GIN
2. **`norm_concat_pca_gin_struct`** — uses the trained GIN-Struct checkpoint

Prints a comparison table (hit\@k, MRR) and saves HNSW indices to
`experiments/output/gin_struct_fusion_test/`.

Not part of the automated pipeline — used for manual validation after
training.

---

### `prompt/patching_experiment.py` — LLM patch generation

Runs retrieval-augmented patch generation via Azure OpenAI.

**Grid axes:**

| Axis | Values | Description |
|---|---|---|
| `retriever_mode` | oracle, precomputed | How relevant context is retrieved |
| `model_name` | gpt-4o, … | LLM deployment |

**Run-cell flow:**

1. Build retriever (`oracle` = all same-CVE pairs; `precomputed` = read from
   prior retrieval run via `--query-run`)
2. Batch inference: for each query, prompt LLM with retrieved context →
   extract generated patch
3. Compare against ground-truth patch → CodeBLEU, ROUGE, exact-match

**Key parameters:** `batch_size`, `prompt_variant` (default / default\_v2 /
graph\_v2), `resume` (continue from partial run).

**CLI:**

```bash
python -m experiments.experiment agent --agent-mode patch [--model gpt-4o]
```

---

### `prompt/test_default_v2.py` / `prompt/test_graph_v2.py`

Targeted re-runs of the patching experiment on a hardcoded set of 23 CVEs that
were `NOT_FIXED` in checkpoint-5.  Each script sets a specific `prompt_variant`
(`default_v2` or `graph_v2`) and uses oracle retrieval.

```bash
python -m experiments.exp.prompt.test_default_v2
python -m experiments.exp.prompt.test_graph_v2
```

---

## End-to-end pipeline (`full` command)

The `full` command chains three stages:

```
retrieval → patching → evaluation → dashboard
```

```bash
python -m experiments.experiment full [--config config.yaml]
```

This runs `RetrievalGridExperiment`, feeds the results into
`PatchingExperiment` (precomputed mode), evaluates patches, and produces a
unified dashboard.

---

## Output

All experiments write to `experiments/output/<run_id>/`:

```
experiments/output/<run_id>/
├── results.json           # canonical grid results (all cells + metrics)
├── summary.json           # flat high-level stats
├── indices/               # saved HNSW index files
│   ├── <tag>.index
│   └── <tag>_meta.json
├── visualizations/        # static PNG figures
└── *.html                 # dashboards (space, combining, etc.)
```

Training scripts write checkpoints alongside their results:

```
<output_dir>/
├── gin_codebert_checkpoint.pt   # or gin_struct_checkpoint.pt
└── training_results.json
```
