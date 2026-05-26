# CVEfixes Experiments

All experiments using the **CVEfixes** dataset are consolidated here. These experiments evaluate how well our graph-based vulnerability representations generalize beyond the hand-curated AutoPatch dataset to thousands of real-world vulnerability fixes.

---

## Directory Structure

```
cvefixes_experiments/
├── README.md                  ← You are here
├── scripts/                   ← Experiment scripts (run from repo root)
│   ├── extract_cvefixes_code.py
│   ├── preliminary_study.py
│   ├── preliminary_study_lines.py
│   ├── plot_clusters.py
│   ├── exp_joern_sink_verification.py
│   ├── exp_pattern_matching.py
│   └── slicing_depth_study.py
├── data/                      ← Input data (JSON extractions + symlinks)
│   ├── CVEfixes.db            → ../data/cvefixes/CVEfixes.db (symlink)
│   ├── graphml_cvefixes       → ../graphml_cvefixes (symlink)
│   ├── graphml_cvefixes_fixed → ../graphml_cvefixes_fixed (symlink)
│   ├── cvefixes_code_extraction.json
│   ├── cvefixes_filtered_by_cwe.json
│   └── cvefixes_filtered_by_cve.json
└── output/                    ← Experiment results
    ├── preliminary_study/     ← CPG diffs for 10 diverse examples
    ├── diagnostics/           ← Visualization PNGs (heatmaps, t-SNE)
    ├── slicing_depth_runs/    ← Timestamped run directories
    ├── slicing_depth_study_work/
    ├── joern_sink_results.json
    ├── joern_sink_subset_100.json
    ├── joern_sink_experiment.log
    ├── joern_pattern_matching_results.json
    ├── joern_kb_ranking_results.json
    ├── kb_ranking_codebert.log
    ├── pattern_matching.log
    └── token_guided_codebert_pattern_clusters.png
```

---

## Scripts Reference

### 1. `extract_cvefixes_code.py` — Data Extraction & Filtering

**Purpose:** Extracts before/after code pairs from the CVEfixes SQLite database for all entries that have pre-built GraphML CPGs. Then filters the dataset to subsets matching AutoPatch CWEs/CVEs.

**Produces:**
- `data/cvefixes_code_extraction.json` — Full extraction (4607 entries, 1777 CVEs, 92 CWEs)
- `data/cvefixes_filtered_by_cwe.json` — Subset matching AutoPatch CWEs (1618 entries)
- `data/cvefixes_filtered_by_cve.json` — Subset matching AutoPatch CVEs (0 entries — no overlap)

**How to run:**
```bash
# From repo root:
python cvefixes_experiments/scripts/extract_cvefixes_code.py
```

**Prerequisites:**
- `data/cvefixes/CVEfixes.db` must exist (the v1.0.8 database)
- `graphml_cvefixes/` must contain Joern-exported CPGs

**Expected output:** Prints statistics about CWE/CVE overlap, saves JSON files.

---

### 2. `preliminary_study.py` — CPG Generation Feasibility

**Purpose:** Smoke test — takes 10 diverse CVEfixes entries, generates CPGs via Joern, computes graph diffs, and compares with ground truth changed lines.

**Produces:** `output/preliminary_study/` with per-example before/after CPG directories and `study_results.json`.

**How to run:**
```bash
python cvefixes_experiments/scripts/preliminary_study.py
```

**Prerequisites:**
- Joern must be installed at `/usr/local/bin`
- `data/cvefixes_code_extraction.json` must exist

**Expected results:** ~70-80% of changed lines captured in sliced graph; validates that the pipeline works on CVEfixes code.

---

### 3. `preliminary_study_lines.py` — Line-Level Metrics

**Purpose:** Post-processing script that extracts line-based metrics from the preliminary study CPGs for quantitative comparison with ground truth.

**How to run:**
```bash
python cvefixes_experiments/scripts/preliminary_study_lines.py
```

**Prerequisites:** `output/preliminary_study/` must exist from step 2.

---

### 4. `exp_joern_sink_verification.py` — Sink Detection Experiment

**Purpose:** Tests whether Joern queries using CWE-specific vulnerability patterns can locate the actual vulnerable code (sinks) in CVEfixes methods. Creates a stratified-by-CWE subset of 100 entries, builds CPGs, runs pattern-based sink queries, then measures precision/recall against the ground-truth diff.

**Produces:**
- `output/joern_sink_subset_100.json` — The stratified 100-entry subset
- `output/joern_sink_results.json` — Per-entry results (sink nodes found, overlap with GT)
- `output/joern_sink_experiment.log` — Detailed log

**How to run:**
```bash
python cvefixes_experiments/scripts/exp_joern_sink_verification.py
```

**Prerequisites:**
- Joern installed
- `data/cvefixes_filtered_by_cwe.json`

**Expected results:** Recall ~50-60% (many sinks found match changed lines), Precision ~10-15% (many false positives — most pattern matches are not the vulnerability).

---

### 5. `exp_pattern_matching.py` — KB-Guided Sink Ranking

**Purpose:** Addresses the false-positive problem from experiment 4. Uses a knowledge base of confirmed vulnerability subgraphs to **rank** candidate sinks by similarity. If a candidate sink's local graph neighbourhood resembles known vulnerabilities, it's more likely to be a true positive.

**Protocol:** Leave-one-out evaluation over 87 verified entries. For each held-out entry, builds a KB from all other entries' true-positive subgraphs, then ranks the held-out entry's candidates by max cosine similarity to any KB subgraph.

**Produces:**
- `output/joern_pattern_matching_results.json` — Pattern matching precision/recall
- `output/joern_kb_ranking_results.json` — KB ranking AUC-ROC and Precision@K
- `output/kb_ranking_codebert.log` — CodeBERT embedding variant log

**How to run:**
```bash
python cvefixes_experiments/scripts/exp_pattern_matching.py
```

**Expected results:** AUC-ROC ~0.65-0.75 (KB similarity can partially separate true from false sinks); Precision@5 improves 2-3x over random baseline.

---

### 6. `slicing_depth_study.py` — Retrieval on CVEfixes Data

**Purpose:** The main retrieval experiment on CVEfixes. Compares different vulnerability slice configurations (1-hop, 2-hop, 3-hop, changed-only) across multiple embedders (GIN, CodeBERT, combined). Evaluates whether the graph-RAG retrieval system generalizes to the larger CVEfixes corpus.

**Grid:** `slice_config × embedder` (4×3 = 12 cells minimum)

**Produces:** Timestamped run directories under `output/slicing_depth_runs/` with `results.json` per run.

**How to run:**
```bash
python -m experiments.exp.slicing_depth_study --config config.yaml --n-index 100 --n-query 20
```

> **Note:** This script uses imports from `experiments.exp` and must be invoked as a module from the repo root. It was moved here for organizational clarity but still relies on the experiment infrastructure in `experiments/`.

**Prerequisites:**
- `data/cvefixes_code_extraction.json`
- Embedder models available (CodeBERT checkpoint, trained GIN)
- `config.yaml` with `datasets.cvefixes` section

**Expected results:** Hit@1 ~0.3-0.5, Hit@5 ~0.6-0.8 depending on embedder; 2-hop and 3-hop slicing outperform changed-only by 10-20%.

---

### 7. `plot_clusters.py` — Embedding Visualization

**Purpose:** Generates a t-SNE plot of CodeBERT-pattern embeddings on token-guided slices, colored by CWE vulnerability family and marked by K-means cluster assignment. Also computes retrieval metrics (P@k, MAP, MRR).

**Produces:** `output/token_guided_codebert_pattern_clusters.png`

**How to run:**
```bash
python cvefixes_experiments/scripts/plot_clusters.py
```

**Prerequisites:**
- `config.yaml`
- Trained CodeBERT-pattern embedder
- `data/cvefixes_code_extraction.json`

---

## Data Flow

```
CVEfixes.db ──► extract_cvefixes_code.py ──► cvefixes_code_extraction.json
                                                      │
                    ┌─────────────────────────────────┼──────────────────────┐
                    │                                  │                      │
                    ▼                                  ▼                      ▼
          preliminary_study.py            slicing_depth_study.py    exp_joern_sink_verification.py
                    │                           │                             │
                    ▼                           ▼                             ▼
       preliminary_study_output/     slicing_depth_runs/         joern_sink_results.json
                                                                             │
                                                                             ▼
                                                              exp_pattern_matching.py
                                                                             │
                                                                             ▼
                                                              joern_kb_ranking_results.json
```

---

## Configuration

The main project `config.yaml` (repo root) contains the CVEfixes dataset section:

```yaml
datasets:
  active: [cvefixes]
  cvefixes:
    db_path: data/cvefixes/CVEfixes.db
    graphml_root: graphml_cvefixes_fixed
```

Scripts reference paths relative to the **repo root**, not this directory. Always run from repo root.

---

## Key JSON Data Formats

### `cvefixes_code_extraction.json`

```json
{
  "total_entries": 4607,
  "total_cves": 1777,
  "entries": [
    {
      "method_change_id": "m36336806924475",
      "cve_id": "CVE-2016-9535",
      "cwe": [{"cwe_id": "CWE-787", "cwe_name": "Out-of-bounds Write"}],
      "code_before": "...",
      "code_after": "...",
      "lines_before": 45,
      "lines_after": 52,
      "diff_stats": {"added": 7, "removed": 0}
    }
  ]
}
```

### `cvefixes_filtered_by_cwe.json`

Same structure, filtered to entries whose CWE matches one of the AutoPatch CWEs (1618 entries). This is the primary dataset used for experiments.

---

## Instructions for Future Agents

1. **Always run scripts from the repo root** — paths are relative to it.
2. **The canonical dataset is `cvefixes_filtered_by_cwe.json`** (1618 entries) — use this for experiments unless you need the full 4607.
3. **Joern is required** for sink verification and preliminary study experiments. Check `which joern` before running.
4. **The slicing depth study** is the most computationally expensive — it embeds hundreds of graphs. Use `--n-index` and `--n-query` to control sample size.
5. **Do not regenerate `cvefixes_code_extraction.json`** unless the database or GraphML dirs change — extraction takes several minutes.
6. **Results are append-only** — each run creates a timestamped directory. Never delete old runs without explicit user request.
7. **The CVEfixes database is ~800MB** — it lives at `data/cvefixes/CVEfixes.db`. See `CVEfixes_v1.0.8/` for installation docs.
8. **Imports:** Scripts use `src.*` and `experiments.*` imports. The virtual environment must be activated and the repo root on `PYTHONPATH`.

---

## Relationship to Main Experiments

The **main retrieval experiments** (`experiments/exp/retrieval_experiment.py`, `experiments/exp/combining_experiment.py`, etc.) run on the AutoPatch dataset by default. CVEfixes experiments here serve as a **generalization test** — validating that approaches developed on AutoPatch transfer to a larger, more diverse corpus.

| Aspect | AutoPatch (main) | CVEfixes (this folder) |
|--------|------------------|------------------------|
| Size | 75 CVEs, ~225 method pairs | 1777 CVEs, 4607 method pairs |
| Languages | C/C++ | C/C++ (primarily) |
| Source | Manual curation | Automated mining |
| GraphML | `graphml_original/`, `graphml_augmented/` | `graphml_cvefixes/`, `graphml_cvefixes_fixed/` |
| Use | Primary dev/eval | Generalization/scale test |
