# Index Evaluation Experiment

## Overview

The `exp_evaluate_index.py` script evaluates a pre-built FAISS index without needing to rebuild CPGs or recompute embeddings. This enables fast iteration on evaluation metrics and embedding space analysis.

## What It Does

1. **Loads the index** from `src/rag/faiss.index` and metadata from `src/rag/metadata.json`
2. **Splits the index** into evaluation and query sets, stratified by CWE (Weakness Type)
3. **Analyzes embedding space** - computes effective dimensionality and similarity statistics
4. **Runs retrieval evaluation**:
   - hit@1, hit@5, hit@10 (same CVE in top-k)
   - MRR (Mean Reciprocal Rank)
   - CWE recall (cross-vulnerability-class retrieval)
   - nDCG and MAP scores
5. **Generates visualizations**:
   - PCA projection of embedding space (color-coded by CWE)
   - t-SNE projection (if sklearn with newer API)
   - UMAP projection (if installed)

## Quick Start

```bash
# Evaluate with default settings (netlsd embedder)
uv run python -m cvefixes_experiments.scripts.exp_evaluate_index

# Specify custom settings
uv run python -m cvefixes_experiments.scripts.exp_evaluate_index \
  --index-dir src/rag \
  --embedder netlsd \
  --output-dir cvefixes_experiments/output/my_eval \
  --test-ratio 0.3 \
  --seed 42

# Filter to specific CWEs
uv run python -m cvefixes_experiments.scripts.exp_evaluate_index \
  --target-cwes CWE-20 CWE-476 CWE-787
```

## Output

The experiment creates:

```
cvefixes_experiments/output/evaluate_index_test/
├── results.json                 # Comprehensive results (metrics + raw queries)
├── metadata_summary.json        # Index statistics
├── split_info.json             # Reproducible split info (index/query entries)
└── plots/
    ├── netlsd_pca.png          # PCA projection
    ├── netlsd_tsne.png         # t-SNE projection
    └── netlsd_umap.png         # UMAP projection
```

### results.json

Key fields:
- **index_statistics**: Dataset breakdown (CWEs, CVEs, n_nodes)
- **split_info**: Train/test split details
- **embedding_space**: Effective dimensionality and similarity
- **retrieval_metrics.cve**: CVE-level retrieval metrics
  - `hit@1`, `hit@5`, `hit@10`: Binary hit rate
  - `mrr`: Mean reciprocal rank of first hit
  - `ndcg@k`, `map@k`: Ranking quality metrics
- **retrieval_metrics.cwe**: CWE-level recall metrics
  - Per-CWE breakdown
  - Macro-averaged recall

## Interpretation

### Good Results
- **hit@5 > 0.5**: At least 50% of queries find same-CVE in top-5
- **MRR > 0.3**: First hits typically within top-3
- **CWE recall > 0.4**: Cross-vulnerability retrieval working
- **Effective dim ~ 10-50**: Embeddings are not degenerate

### Red Flags
- **hit@1 << MRR**: Embeddings not capturing exact matches
- **CWE recall < 0.1**: Vulnerability classes not separating
- **Effective dim < 2**: Embedding space is degenerate
- **Mean pairwise sim ~ 1.0**: All vectors too similar

## Parameters

```
--index-dir DIR              Path to index (default: src/rag)
--embedder NAME              Embedder used (default: netlsd)
--output-dir DIR             Output path (default: cvefixes_experiments/output/evaluate_index)
--test-ratio FLOAT           Query ratio 0.0-1.0 (default: 0.2)
--seed INT                   Random seed (default: 42)
--target-cwes CWE_ID...      Filter to specific CWEs (optional)
--load-embeddings DIR        Load pre-computed embeddings (optional, WIP)
--save-embeddings DIR        Save computed embeddings (optional, WIP)
```

## Implementation Notes

- Uses pre-computed embeddings from the FAISS index
- Queries are embedded using the same vectors as their index entry
- Leave-one-out evaluation: queries are separate from index
- Metrics computed using NDCG, MAP, precision@k, recall@k
- CWE recall: binary "is any same-CWE entry in top-k?"

## Future Enhancements

- [ ] Load/compute graph embeddings on-the-fly
- [ ] Add graph embedding caching
- [ ] Support multiple embedder comparison
- [ ] Generate HTML dashboard
- [ ] Add retrieval examples (show actual retrieved results)
- [ ] Compute per-CWE metrics breakdown

## References

- **NDCG**: Normalized Discounted Cumulative Gain (ranking quality)
- **MAP**: Mean Average Precision (ranking quality)
- **MRR**: Mean Reciprocal Rank (first-hit position)
- **Effective Dimension**: Number of principal components explaining 90% variance
