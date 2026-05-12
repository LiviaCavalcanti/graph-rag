# Embedding Space Analysis

Scripts for visualizing and comparing embedding spaces produced by the project's embedders.

## Files

### `visualize_embeddings.py`

Main visualization script for **registry-based embedders** (combined, gin, wl, codebert_seq, codebert_pattern). Generates per-embedder:

- **t-SNE 2D** side-by-side: raw (high-dim, L2-normed) vs PCA-reduced
- **UMAP 2D** side-by-side: same comparison
- **3D interactive HTML** (UMAP on PCA space) via Plotly
- **PCA scree curve** (cumulative + per-component variance)

Also exports **reusable functions** for other scripts:

| Function | Purpose |
|---|---|
| `tsne_project(X, perplexity, seed)` | t-SNE → 2D projection |
| `umap_project(X, n_components, seed)` | UMAP projection (cosine metric) |
| `plot_2d_comparison(left_2d, right_2d, ...)` | Side-by-side scatter with query annotations, legend, and save |
| `plot_space(ax, coords_2d, labels, ...)` | Single scatter plot colored by CWE class |
| `plot_3d_interactive(coords_3d, labels, ...)` | Interactive 3D HTML scatter via Plotly |
| `plot_pca_scree(raw_embs, emb_name, out_path)` | PCA cumulative variance scree curve |
| `get_raw_embeddings(embedder, graphs)` | Get pre-PCA/pre-L2 embeddings from any embedder |
| `get_pca_l2_embeddings(embedder, graphs)` | Get final PCA+L2 embeddings (calls `embed_many`) |

```bash
python -m experiments.embedding_space.visualize_embeddings [--output-dir DIR] [--embedders NAME ...] [--perplexity N] [--seed N]
```

### `visualize_gin_struct.py`

Visualization for the **trained GIN-Struct** model (loaded from checkpoint). Generates:

- 3D interactive HTML (UMAP)
- t-SNE vs UMAP 2D comparison
- Raw vs PCA→64d comparison (t-SNE)

Uses `plot_2d_comparison`, `tsne_project`, and `umap_project` from `visualize_embeddings.py`.

```bash
python -m experiments.embedding_space.visualize_gin_struct
```

Output: `experiments/output/checkpoint5/`

### `compare_spaces.py`

Quantitative comparison of **GIN-Struct (trained) vs Combined (norm_concat_pca)**. Computes intrinsic quality metrics:

- Silhouette score (CVE-level and CWE-level)
- Intra-class / inter-class distance ratio
- k-NN purity (fraction of k nearest neighbors sharing same CVE/CWE)
- Alignment score (mean cosine similarity within same-CVE pairs)

```bash
python -m experiments.embedding_space.compare_spaces
```

### `space_analysis_experiment.py`

Full experiment using the `Experiment` base class. Evaluates intrinsic quality and **cross-space pairwise metrics** (CKA, k-NN overlap, rank correlation, trustworthiness) for all active embedders.

Outputs structured JSON results for downstream analysis.

```bash
python -m experiments.embedding_space.space_analysis_experiment [--config config.yaml]
```

## Common patterns

All scripts load data via `load_config` → `load_pairs` → `build_split`, producing `index_pairs` and `query_pairs`. Query samples are highlighted with black outlined markers in plots.

To add a new embedder visualization, reuse `plot_2d_comparison` from `visualize_embeddings.py`:

```python
from experiments.embedding_space.visualize_embeddings import (
    plot_2d_comparison, tsne_project, umap_project,
)
```
