# graph-rag — Agent Guide

Structure-aware RAG for automated **C/C++ vulnerability detection & patching**.
Joern parses functions into Code Property Graphs (CPGs) → diff vulnerable vs patched →
embed the vulnerability subgraph → index in FAISS → retrieve similar CVEs → (optionally)
LLM generates a patch → evaluate.

```
raw source ──[export]──▶ Joern CPGs (GraphML) ──[index]──▶ embed G_vuln ──▶ FAISS
                                                                              │
                                          [query] embed query graph ──▶ top-k CVEs
                                                                              │
                                          [batch] LLM patch ──▶ [evaluate] metrics + dashboards
```

---

## Golden rules (read first)

- **Package manager is `uv`.** Always run Python via `uv run …`. Never use bare `pip`,
  `python -m venv`, or a bare `python`/`python3`.
- **Local model only.** The LLM/embedding backbone is the **local CodeBERT** at
  `models/codebert-base/`. Do **not** download or swap models without asking the user.
- **New experiments MUST subclass `Experiment`** from [experiments/base.py](experiments/base.py).
  Never add standalone experiment scripts that bypass it.
- **This repo is large — searches time out.** Always scope `grep`/`file_search` with an
  `includePattern` such as `src/**`, `experiments/exp/**`, or `tests/**`. Never run
  workspace-wide regex over everything (the `graphml_*/`, `experiments/output/`, and
  `CVE-list/` trees contain thousands of files).
- **Don't edit or commit generated artifacts:** `experiments/output/`, `graphml_*/`,
  `rag/**/*.index`, `data/cvefixes/CVEfixes.db`, `workspace/`. They are large and rebuildable.
- **Secrets live in `.env`.** Never print, commit, or read `.env` contents into context.

---

## Environment & secrets

`.env` at repo root is loaded via `python-dotenv` (in `src/agents/utils.py` and `patcher.py`):

| Var | Purpose | Default |
|---|---|---|
| `MODEL_NAME` | Azure deployment name for patching | `deepseek-R1` |
| `AZURE_API_KEY` | Azure/litellm auth | — |
| `AZURE_API_BASEURL` | Azure endpoint | — |

Only `--mode batch`/`full` (LLM patching) and the `integration`-marked tests need these.
`export`, `index`, `query`, and unit tests run fully offline.

---

## Common commands (always prefix with `uv run`)

```bash
# ── setup ─────────────────────────────────────────────
uv sync                                   # install/refresh the environment

# ── tests ─────────────────────────────────────────────
uv run pytest -m "not integration" -q     # fast unit suite (no live API)
uv run pytest tests/test_embedders.py -q  # a single test module
uv run pytest -m integration              # live-API tests (needs AZURE_* keys)

# ── formatting ────────────────────────────────────────
uv run isort . && uv run black .          # canonical formatters (both are deps)

# ── main pipeline (see main.py docstring) ─────────────
uv run python main.py --mode export       # run Joern → GraphML CPGs (needs Joern installed)
uv run python main.py --mode index        # embed CPGs → build FAISS index
uv run python main.py --mode query --cve CVE-2024-XXXXX
uv run python main.py --mode batch --model gpt-4o --max-queries 10   # LLM patching
uv run python main.py --mode full         # retrieval → patching → evaluation

# ── evaluation of a batch run ─────────────────────────
uv run python -m src.evaluate experiments/output/<run_id>/

# ── experiments (each script has its own argparse/__main__) ──
uv run python experiments/exp/retrieval_experiment.py --help
```

---

## Repository map

| Path | What lives here |
|---|---|
| [main.py](main.py) | Unified CLI: `export` / `index` / `query` / `batch` / `full`. Start here. |
| [config.yaml](config.yaml) | Primary config (datasets, embedders, RAG, experiment split). `config_nosplit.yaml` is a variant. |
| `src/data/` | Datasets & graph build. `base.py` (`FunctionPair`, `ExportJob`), `pipeline.py` (`load_cpg_dir`, `compute_graph_diff`, `run_joern_export`), `cvefixes.py`, `autopatch.py`. |
| `src/embeddings/` | Embedder registry (`__init__.py` → `build_embedders`). Variants: `netlsd`, `wl`, `gin`, `combined`, `codebert_seq`, `codebert_pattern`, `rgcn`, `gin_codebert`, `gin_struct`. |
| `src/rag/` | Retrieval: `faiss_index.py`, `hnsw.py`, `oracle.py` (perfect same-CVE), `precomputed.py`, `retriever.py`. |
| `src/agents/` | LLM patching: `batch_inference.py`, `patcher.py`, `graph_context.py`, `prompts/`, `utils.py` (`MODEL_NAME`). Uses `litellm` Azure backend. |
| `src/evaluate/` | Patch/retrieval scoring + dashboards. Entry: `python -m src.evaluate`. `patch_verification.py`, `llm_evaluation.py`, `confidence_eval.py`. |
| `src/metrics/` | `metrics.py` → `embedding_space_stats`, retrieval metrics. |
| `src/graph/` | `transform_graph.py` (node-type vocab), `joern_graph.py` (`export_graph_json`). |
| `src/schema_config.py` | Typed config dataclasses (`AppConfig`, `EmbeddingConfig`, …). |
| `experiments/base.py` | `Experiment` base class + `Axis`, `CellContext`, `MetricSpec`. **All experiments subclass this.** |
| `experiments/common.py` | Shared primitives: `load_config`, `load_pairs`, `build_split`, `build_hnsw`, `evaluate_retrieval`, `evaluate_cwe_recall`, `save_json`. |
| `experiments/exp/` | Concrete experiment scripts (run directly with `uv run python`). |
| `experiments/dashboard_scripts/` | HTML dashboards (`patch/`, `_theme.py`, comparison/diagnostics). |
| `experiments/output/` | **Generated** run dirs `<timestamp>_<name>_<hash>/`. Do not edit/commit. |
| `tests/` | `pytest`. `pythonpath=src`. Markers: `severity_high`, `severity_low`, `integration`. |
| `data/cvefixes/CVEfixes.db` | CVEfixes SQLite DB (download from Zenodo; **generated**). |
| `CVE-list/` | AutoPatch dataset — one folder per CVE. |
| `graphml_*/`, `rag/` | **Generated** Joern exports and FAISS indices. |
| `models/codebert-base/` | Local CodeBERT weights (referenced absolutely in `config.yaml`). |
| `.github/skills/` | Domain skills (see below). `.github/AGENTS.md` holds the short canonical rules. |

---

## Data & graph model

A vulnerability fix is a `FunctionPair` ([src/data/base.py](src/data/base.py)) with three
NetworkX `MultiDiGraph` CPGs:

- **`G_before`** — vulnerable code, **`G_after`** — patched code,
  **`G_vuln`** — semantic diff + bounded program slice (`compute_graph_diff`).

Node attrs: `labelV` (type), `CODE`, `LINE_NUMBER`, plus `diff` /`diff_weight`
(`removed`=1.0, `fix_adjacent`=0.8, `edge_changed`=0.6, `context`=0.2) added by the diff.
Edge attr `labelE`: `AST`, `CFG`, `CDG`, `REACHING_DEF`, `REF`, `ARGUMENT`, `RECEIVER`, …
Full schema is in repo memory (`graph-representation-schema.md`).

---

## Config knobs (`config.yaml`)

- `data.active` → `[cvefixes]` (primary) or `[autopatch]`.
- `data.cvefixes.{db_path, graphml_root, target_cwes, level}`.
- `embeddings.active` → which embedders are built; `embeddings.dim` (128), `projection: pca`, `l2_normalize: true`.
- `embeddings.codebert.model_path` → **absolute** path to local CodeBERT; fix if the repo moves.
- `rag.{embedding_variant, prompt_variant, top_k, index_path}` → active retrieval config.
- `experiment.split.*` → stratified CVE-aware split (seed 42, `test_ratio` 0.2).

---

## Conventions & gotchas

- **PCA embedders need fitting first.** `CombinedEmbedder` and `VulnPatternEmbedder` must see a
  batch via `embed_many(...)` before `embed_one(...)`.
- **CVE-aware split.** Keep same-CVE support in the index for every query to avoid leakage /
  impossible positives (CVEfixes "original" variant is prone to this).
- **Joern is only needed for `--mode export`.** Set `joern.bin_dir` in config; it may be empty by default.
- **Run scripts from repo root** so `from src… import …` resolves (root is on `sys.path`).
- After edits, validate with `uv run pytest -m "not integration" -q` and format with `black`+`isort`.
- Prefer editing existing modules over adding new top-level scripts; the root already has many analysis scripts.

---

## Skills (`.github/skills/`)

Read the skill's `SKILL.md` before tasks in its domain:

- **cvefixes-dataset** — CVEfixes schema, sampling, diffs from `code_before`/`code_after`.
- **eval-dashboard** — add metrics / update patch-evaluation HTML dashboards.
- **checkpoint-presentation** — generate LaTeX Beamer checkpoint decks.
- **review-tex** — review/lint Beamer meeting `.tex` files.
