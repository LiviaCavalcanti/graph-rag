# graph-rag

Structure-aware RAG system for automated vulnerability detection using Code Property Graph (CPG) embeddings. Parses C/C++ functions with Joern, computes graph diffs between vulnerable and patched versions, embeds the vulnerability-relevant subgraph, and indexes it for retrieval.

---

## Architecture

```
graph-rag/
├── config.yaml                  # all settings live here
├── main.py                      # CLI entrypoint
├── data/
│   ├── base.py                  # FunctionPair + ExportJob dataclasses, BaseDataset interface
│   ├── pipeline.py              # graph loading, Joern export, graph diff
│   ├── bigvul.py                # BigVul dataset
│   ├── cvefixes.py              # CVEFixes dataset
│   └── autopatch.py             # AutoPatch dataset
├── embeddings/
│   ├── base.py                  # BaseEmbedder interface
│   ├── netlsd.py                # NetLSD (spectral descriptor, no training)
│   ├── wl.py                    # WL baseline (Weisfeiler-Lehman, no training)
│   └── __init__.py              # embedder registry
├── rag/
│   ├── index.py                 # FAISS index build + save/load
│   └── retriever.py             # query interface for agents
└── agents/
    └── __init__.py              # load_retriever() — single call for ADK agents
```

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
  active: [netlsd, wl]
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
python main.py --config config.yaml --mode export --dataset bigvul
python main.py --config config.yaml --mode export --dataset cvefixes
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

Look up stored metadata by CVE ID (no embedding needed):

```bash
python main.py --config config.yaml --mode query --cve CVE-2025-22017
```

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

## Adding a new dataset

1. Create `data/mydata.py` subclassing `BaseDataset`
2. Implement `name()`, `stream()` → yields `FunctionPair`, and `export_jobs()` → yields `ExportJob`
3. Register in `main.py`:
```python
from data.mydata import MyDataset
DATASETS = { ..., 'mydata': MyDataset }
```
4. Add config block in `config.yaml` under `data.mydata`

## Adding a new embedder

1. Create `embeddings/myembedder.py` subclassing `BaseEmbedder`
2. Implement `name` property and `embed_one(G) -> np.ndarray`
3. Register in `embeddings/__init__.py`:
```python
from .myembedder import MyEmbedder
REGISTRY = { ..., 'myembedder': MyEmbedder }
```
4. Add to `embeddings.active` list in `config.yaml`