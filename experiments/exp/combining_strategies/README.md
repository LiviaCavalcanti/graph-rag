# Combining Strategy Experiments

This folder contains all experiments for evaluating different strategies to combine multiple embedding methods into a single unified embedding space.

## Overview

The combining strategy experiments evaluate how to best fuse embeddings from multiple sources (NetLSD, WL, GIN, CodeBERT) into a single vector representation. Three main strategies are explored:

1. **norm_concat_pca** — L2-normalize each embedder separately, concatenate, then apply PCA
2. **pca_concat_pca** — Apply PCA to each embedder individually, concatenate, then apply final PCA
3. **norm_pca_concat** — L2-normalize each embedder, apply PCA individually to each, then concatenate (no final PCA)

## Structure

```
combining_strategies/
├── README.md                    (this file)
├── scripts/                     (all experiment scripts)
│   ├── combining_experiment.py          (base experiment class)
│   ├── _combining_strategies.py         (strategy implementations: NormConcatPCA, PCAConcatPCA, NormPCAConcat)
│   ├── combining_norm_vs_pca.py         (Experiment A: norm vs pca pre-processing)
│   ├── combining_joint_vs_indiv_pca.py  (Experiment B: joint vs individual PCA)
│   ├── combining_all_strategies.py      (runs all 3 strategies in one go for direct comparison)
│   └── combining_repeated.py            (repeated runner for robust estimates)
└── output/                      (experiment results organized by type)
    ├── norm_vs_pca/            (runs comparing norm_concat_pca vs pca_concat_pca)
    ├── joint_vs_indiv_pca/     (runs comparing norm_concat_pca vs norm_pca_concat)
    ├── all_strategies/         (runs of combining_all_strategies.py)
    ├── repeated_runs/          (repeated runs with multiple random seeds)
    └── legacy/                 (early pilot runs)
```

## Script Reference

### Base Experiment & Strategies

**`combining_experiment.py`**
- Base class for all combining experiments
- Defines common grid axes, metric computation, and reporting
- Metrics: self-retrieval (hit@k, MRR, nDCG, MAP), space quality (effective dimension, isotropy, hubness), class separation (intra/inter-CWE ratio)

**`_combining_strategies.py`**
- Strategy implementations with different fusion approaches
- Classes:
  - `NormConcatPCA` — L2-norm each → concat → PCA
  - `PCAConcatPCA` — PCA each → concat → PCA
  - `NormPCAConcat` — L2-norm → PCA each → concat (no final PCA)

### Main Experiments

**`combining_norm_vs_pca.py` — Experiment A**
- Compares pre-processing normalization strategies
- Evaluates whether individual PCA or global L2-normalization better controls embedder contribution
- Strategies: `norm_concat_pca` vs `pca_concat_pca`
- Grid: strategy (2 cells)
- Output: `norm_vs_pca_report.json` with comparison metrics

**`combining_joint_vs_indiv_pca.py` — Experiment B**
- Compares dimensionality reduction placement
- Evaluates whether to apply PCA before or after concatenation
- Strategies: `norm_concat_pca` (joint) vs `norm_pca_concat` (individual)
- Grid: strategy (2 cells)
- Output: `joint_vs_indiv_pca_report.json` with comparison metrics

**`combining_all_strategies.py`**
- Runs all three strategies (norm_concat_pca, pca_concat_pca, norm_pca_concat) in one execution
- Single data load → directly comparable results across all three
- Useful for three-way comparison on same train/test split
- Output: `combining_report.json` with all three strategies

**`combining_repeated.py`**
- Repeated runner for robust statistics across multiple random seeds (default: 10 runs)
- Supports running Experiment A (norm vs pca), Experiment B (joint vs indiv), or both
- Usage: `python -m experiments.exp.combining_repeated [--runs N] [--experiment A|B|both]`
- Outputs aggregated statistics (mean, std, quartiles) per strategy

## How to Run

### Single Experiment
```bash
# Experiment A: norm vs pca pre-processing
python -m experiments.exp.combining_norm_vs_pca --config config.yaml

# Experiment B: joint vs individual PCA
python -m experiments.exp.combining_joint_vs_indiv_pca --config config.yaml

# All strategies at once
python -m experiments.exp.combining_all_strategies --config config.yaml
```

### Repeated Runs
```bash
# Run Experiment A 10 times with different random seeds
python -m experiments.exp.combining_repeated --config config.yaml --runs 10 --experiment A

# Run both A and B, 5 runs each
python -m experiments.exp.combining_repeated --config config.yaml --runs 5 --experiment both
```

## Output Format

Each run produces:
- `results.json` — Standard experiment output with all metrics per cell
- `{experiment}_report.json` — Comparison table and summary statistics
- `indices/` — HNSW retrieval indices for each strategy variant
- `space_dashboard.html` — Interactive visualization of embedding space

Example `results.json` structure:
```json
{
  "experiment": "combining_norm_vs_pca",
  "grid": {"strategy": ["norm_concat_pca", "pca_concat_pca"]},
  "cells": {
    "norm_concat_pca": {
      "metrics": {
        "hit@1": 0.85,
        "hit@10": 0.95,
        "effective_dim": 92.3,
        "isotropy": 0.42,
        ...
      }
    },
    ...
  }
}
```

## Key Files in `output/`

### norm_vs_pca/ folder
- **Purpose**: Extensive ablation of L2-normalization vs individual PCA as pre-processing step
- **Count**: 18 separate runs
- **Pattern**: `YYYYMMDD_HHMMSS_combining_norm_vs_pca_XXXXXX`
- **Analysis**: Determines whether norm or pca better equalizes embedder contributions

### joint_vs_indiv_pca/ folder
- **Purpose**: Tests when to apply PCA (before concat vs after concat)
- **Count**: 13 separate runs
- **Pattern**: `YYYYMMDD_HHMMSS_combining_joint_vs_indiv_pca_XXXXXX`
- **Key Finding**: Individual PCA per embedder (lower-dim concat) vs joint PCA on concatenated (higher-dim reduction)

### repeated_runs/ folder
- **Purpose**: Multiple runs with different random seeds for robust statistics
- **Count**: 2 repeated run experiments
- **Pattern**: `YYYYMMDD_HHMMSS_combining_both_repeated10_XXXXXX`
- **Method**: Each contains 10 (or N) independent runs aggregated

### legacy/ folder
- **Purpose**: Early pilot runs from initial development
- **Count**: 3 runs
- **Note**: May use different configurations or incomplete metric set

## Configuration

These experiments use standard config.yaml with the following relevant sections:
```yaml
dataset:
  autopatch:
    train_split: 0.8
    test_split: 0.2
embeddings:
  - gin
  - netlsd
  - wl
  # CodeBERT optionally combined
```

## Metrics Computed

**Retrieval Performance:**
- hit@1, hit@5, hit@10, hit@20
- MRR (Mean Reciprocal Rank)
- nDCG (normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)

**Embedding Space Quality:**
- Effective Dimension (intrinsic dimensionality)
- Isotropy (alignment of variance across dimensions)
- Hubness (concentration of similarities)
- Distance Concentration
- Alignment & Uniformity (USLE metrics)

**Classification Quality:**
- Intra/Inter CWE Ratio (cluster separation by vulnerability class)

## Results Summary

From analysis of completed runs:
- **norm_concat_pca** typically achieves best retrieval performance
- **pca_concat_pca** offers moderate trade-off in space quality vs performance
- **norm_pca_concat** provides most interpretable individual subspaces but lower final retrieval
- Joint PCA (norm_concat_pca) generally superior to individual PCA (norm_pca_concat)

## Import Compatibility

For backward compatibility, stub modules exist at `experiments/exp/`:
- `combining_experiment.py`
- `_combining_strategies.py`
- `combining_norm_vs_pca.py`
- `combining_joint_vs_indiv_pca.py`
- `combining_all_strategies.py`
- `combining_repeated.py`

These stubs re-export from `combining_strategies/scripts/` to maintain existing import paths.

## Future Work

1. **Dimensionality Settings**: Sweep different final PCA dimensions (64d, 128d, 256d)
2. **Embedder Subsets**: Test combining different subsets of embedders
3. **Weighting Strategies**: Learnable or heuristic weights for different embedders
4. **Online Combination**: Combine embedders at query time vs index time
## Contact & Questions

For questions about these experiments:
- Check the script docstrings for detailed methodology
- Review `_combining_strategies.py` for fusion algorithm details
- See `combining_experiment.py` base class for metric definitions
