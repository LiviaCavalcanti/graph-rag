# graph-rag

Structure-aware RAG system for automated vulnerability detection using Code Property Graph (CPG) embeddings. Parses C/C++ functions with Joern, computes graph diffs between vulnerable and patched versions, embeds the vulnerability-relevant subgraph, and indexes it for retrieval.

---

### Pipeline stages

```
raw source code
      │
      ▼
[export]  Joern parses .c files → exports GraphML per function
      │
      ▼
[index]   load graphs → compute diff → embed G_vuln → store in FAISS
      │
      ▼
[query]   embed query graph → retrieve top-k similar CVEs
```

---

## Datasets

| Dataset | Source | Format |
|---|---|---|
| BigVul | CSV with `func_before` / `func_after` columns | `data/raw/bigvul.csv` |
| CVEFixes | SQLite database | `data/raw/CVEfixes.db` |
| AutoPatch | Folder per CVE with `.txt` / `.c` source files | `data/raw/CVE-list/` |

AutoPatch folder structure expected:
```
CVE-list/CVE-XXXX-YYYY/
    db_entry.json
    original_code.txt        ← vulnerable function
    original_code_fixed.c    ← patched function
    supplementary_code.txt
    out_v2/code/             ← augmented variants (optional)
        augmented.json
        augmented_fixed.c
        re_implemented_*.json
        re_implemented_*_fixed.c
```

---

## Embedders

| Name | Method | Training |
|---|---|---|
| `netlsd` | Network Laplacian Spectral Descriptor — heat trace of graph Laplacian | none |
| `wl` | Weisfeiler-Lehman colour refinement → pooled → linear projection | none |
| `gin` | Neural equivalent of the WL test that learns a differentiable function to aggregate neighbor information | none |
Both output L2-normalised vectors of configurable dimension (default 128). The active embedder for FAISS indexing is set in `config.yaml` under `rag.embedding_variant`.

---

## Setup

```bash
# create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install networkx pandas netlsd torch torch_geometric faiss-cpu scikit-learn pyyaml

# install Joern (required for export step only)
# https://docs.joern.io/installation
# set bin_dir in config.yaml to your joern-cli folder
```

---

## Configuration

Edit `config.yaml` before running. Key fields:

```yaml
joern:
  bin_dir: /opt/joern/joern-cli   # path to joern-cli folder
  workers: 8                       # parallel export jobs

data:
  bigvul:
    csv_path: data/raw/bigvul.csv
    graphml_root: data/graphs/bigvul
  cvefixes:
    db_path: data/raw/CVEfixes.db
    graphml_root: data/graphs/cvefixes
  autopatch:
    root: data/raw/CVE-list
    graphml_root: data/graphs/autopatch
    include_variants: false          # set true to include LLM-generated variants

embeddings:
  active: [netlsd, wl, gin, combined]
  dim: 128

rag:
  index_path: data/rag/faiss.index
  metadata_path: data/rag/metadata.json
  top_k: 5
  embedding_variant: netlsd         # which embedder feeds the FAISS index
```

---

## Running the pipeline

### Step 1 — export graphs

Runs Joern on all source files and writes GraphML exports. Safe to re-run: already-exported functions are skipped.

```bash
# export all configured datasets
python main.py --config config.yaml --mode export

# export a single dataset
python main.py --config config.yaml --mode export --dataset autopatch
```

Exported graphs are written to:
```
data/graphs/<dataset>/<CVE-ID>/<variant>/before/
data/graphs/<dataset>/<CVE-ID>/<variant>/after/
```

### Step 2 — build FAISS index

Loads exported graphs, computes vulnerability-relevant subgraph (diff of before/after), embeds with the configured embedder, and stores in FAISS.

```bash
python main.py --config config.yaml --mode index
```

### Step 3 — query
```
# TODO
````
---

## Using from ADK agents

```python
from agents import load_retriever
import numpy as np

# load index once at agent startup
retriever = load_retriever('config.yaml')

# query with a pre-computed embedding
embedding = np.zeros(128, dtype=np.float32)   # replace with real embedding
results = retriever.query(embedding, top_k=5)

for r in results:
    print(r['cve_id'], r['func_name'], r['score'])

# or look up by CVE ID directly
results = retriever.query_by_cve('CVE-2025-22017')
```
---

## Experiment evaluation

Run the full evaluation grid (all embedders × all graph variants × all backends):

```bash
python main.py --config config.yaml --mode experiment
```

Results are written to `experiments/output/<run_id>/`:
- `results.json` — full structured output including per-query raw data
- `summary.json` — flat table suitable for pandas
- `all_runs.json` — cumulative log across all runs
- `visualizations/dashboard_performance.png`
- `visualizations/dashboard_quality.png`
- `visualizations/dashboard_metrics.png`

---

### Metrics

The experiment runner evaluates each cell in the grid (one embedder × one graph variant × one backend) across three complementary metric families.

---

#### 1. Code-query retrieval (self_retrieval)

**What it measures:** given the vulnerability subgraph of a CVE as the query, does the index return the same CVE in the top-k results?

**How it is computed:**

For each `FunctionPair` in the dataset:
1. The query graph is selected: `G_vuln` (the vulnerability-relevant subgraph) for original samples, `G_before` for LLM-generated variants.
2. The query graph is embedded with the same embedder used to build the index.
3. The index is queried with `top_k` results.
4. A result is a **hit** at rank `k` if the correct CVE ID appears anywhere in the top-k list.

$$\text{Hit@k} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\text{cve\_id}_i \in \text{top-k results}_i]$$

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank of first correct result}_i}$$

Samples whose embedding has near-zero norm (degenerate graphs) are skipped and not counted in N.

**Output fields** (`self_retrieval`):

| Field | Description |
|---|---|
| `hit@1`, `hit@5`, `hit@10` | Fraction of queries with correct CVE in top-k |
| `mrr` | Mean reciprocal rank |
| `n` | Number of valid (non-degenerate) queries |
| `raw_queries` | Per-query list: `query_cve`, `query_cwe`, `hit`, `mrr`, and ranked `retrieved` items with `cve_id`, `cwe_id`, `score` |

---

#### 2. CWE-group recall (cwe_recall)

**What it measures:** for each sample, do the top-k retrieved results share the same vulnerability type (CWE)? This evaluates clustering quality without needing exact CVE matches.

**How it is computed:**

Samples are grouped by CWE. A CWE with only one sample is a **singleton** and is excluded from scoring (there are no same-type peers to retrieve).

For each sample $i$ with $\text{CWE} = c$, let $S_c$ be the set of all other samples with the same CWE. The index is queried and self is excluded by CVE ID:

$$\text{Recall}_i = \frac{|\{r \in \text{top-k}\setminus\{i\} : \text{cwe}(r) = c\}|}{\min(k,\, |S_c|)}$$

$$\text{CWE Recall}(c) = \frac{1}{|S_c|} \sum_{i \in S_c} \text{Recall}_i$$

$$\text{Macro Avg} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{CWE Recall}(c)$$

where $\mathcal{C}$ is the set of CWEs with at least 2 samples.

**Output fields** (`cwe_recall`):

| Field | Description |
|---|---|
| `per_cwe` | Dict of `cwe → {recall, support}` for each qualifying CWE |
| `macro_avg` | Unweighted mean of per-CWE recall |
| `n_cwes` | Number of CWEs with ≥ 2 samples |
| `n_singletons` | Number of CWEs with only 1 sample (excluded) |
| `raw_queries` | Per-query list: `query_cve`, `query_cwe`, `recall`, and ranked `retrieved` items |

---

#### 3. Embedding space statistics (space_stats)

**What it measures:** intrinsic quality of the embedding space, independent of any retrieval task. Useful for detecting collapsed embeddings before running retrieval.

| Field | Description |
|---|---|
| `mean_norm`, `std_norm` | Mean and std of L2 norms (should be ≈1.0 after normalisation) |
| `mean_pairwise_sim` | Mean cosine similarity over a random subset of 500 pairs |
| `std_pairwise_sim` | Std of pairwise similarities (low std → collapsed space) |
| `min_pairwise_sim`, `max_pairwise_sim` | Range of similarities |
| `effective_dim` | Participation ratio of PCA eigenvalues: $(\sum \lambda_i)^2 / \sum \lambda_i^2$. Value of 1 means the space has collapsed to one direction; value of $d$ means all dimensions are used equally. |

---

#### 4. Leave-one-out retrieval (leave_one_out, optional)

**What it measures:** same as code-query retrieval but more honest — for each sample, the index is rebuilt on all *other* samples, then queried. Avoids the trivial self-match.

Enabled only when `run_leave_one_out=True` and dataset size ≤ 1000. Reports the same `hit@k` and `mrr` fields.

---

#### Computing precision and recall yourself

Each cell in `results.json` contains a `raw_queries` array under both `self_retrieval` and `cwe_recall`. Each entry includes the full ranked list of retrieved items. You can recompute any metric directly:

```python
import json

with open("experiments/output/<run_id>/results.json") as f:
    data = json.load(f)

cell = data['cells'][0]  # pick a cell

# precision@5 for code-query retrieval
queries = cell['self_retrieval']['raw_queries']
p5 = sum(
    any(r['cve_id'] == q['query_cve'] for r in q['retrieved'][:5])
    for q in queries
) / len(queries)

# per-CWE recall from raw data
from collections import defaultdict
cwe_recalls = defaultdict(list)
for q in cell['cwe_recall']['raw_queries']:
    cwe_recalls[q['query_cwe']].append(q['recall'])
per_cwe = {cwe: sum(v)/len(v) for cwe, v in cwe_recalls.items()}
```