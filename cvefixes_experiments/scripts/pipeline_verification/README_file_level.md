# File-Level Retrieval Experiment

## Overview

This experiment evaluates the **same CVEfixes sample** as the pipeline verification experiment (`pipeline_verification_structure`), but at **file granularity** instead of function granularity.

**Goal:** Compare retrieval performance when switching from function-level to file-level aggregation.

## Motivation

The original pipeline-verification experiment evaluates each function independently:
- Each function gets its own CPG (code property graph)
- Embeddings are computed per-function
- Retrieval metrics are function-level hits

This file-level experiment investigates:
1. **Does file-level aggregation improve retrieval?** (More context might help matching)
2. **Does it degrade performance?** (Noise from unrelated functions in the same file)
3. **How does embedding space change?** (Larger graphs, potentially different structure)

## Experiment Design

### Data Preparation

```
Function-level pairs from pipeline_verification_structure
    ↓
Group by (CVE_id, filename)
    ↓
Merge graphs: G_vuln(func1) ∪ G_vuln(func2) ∪ ... ∪ G_vuln(funcN)
    ↓
Create FilePair objects with aggregated graphs
    ↓
Run retrieval experiment (same metrics, same split strategy)
```

### Key Features

- **Same split**: Uses `pipeline_verification_structure/split_info.json` as the authoritative split
  - Index: 109 functions → ~X files (depends on co-location)
  - Query: 27 functions → ~Y files
  
- **Same embedders**: Compares `gin`, `combined`, `codebert_seq`, `codebert_pattern`

- **Same metrics**: CVE hit@k, MRR, CWE recall, embedding space stats

### Graph Aggregation Method

Functions from the same file and CVE are combined using NetworkX union:

```python
G_vuln_file = nx.union(G_vuln_func1, G_vuln_func2, ...)
```

This preserves:
- All nodes from all functions
- All edges from all functions
- Node/edge attributes

Result: A single "supergraph" representing all vulnerability-relevant code changes in that file for that CVE.

## Running the Experiment

```bash
# From workspace root, using reference split from pipeline_verification_structure
python -m cvefixes_experiments.scripts.pipeline_verification.exp_cvefixes_retrieval_grid_file_level \
  --config config.yaml \
  --reference-split cvefixes_experiments/output/pipeline_verification_structure/split_info.json \
  --output-dir cvefixes_experiments/output/cvefixes_retrieval_grid_file_level \
  --embedders codebert_seq codebert_pattern combined gin
```

## Expected Outputs

```
cvefixes_experiments/output/cvefixes_retrieval_grid_file_level/
├── results.json              # Aggregated metrics per embedder/backend
├── split_info.json           # File-level split metadata + aggregation stats
├── dashboard.html            # Interactive HTML dashboard
├── indices/                  # HNSW indices (one per embedder)
└── cpg_cache/                # Cached file-level CPGs
```

## Interpreting Results

### Split Reduction Example

If 109 functions are distributed across ~80 files:
- **Reduction**: 109 → 80 (compression ratio: 0.73)
- **Implication**: On average, 1.36 functions per file for these CVEs

If reduction is low (e.g., 0.9), then most functions appear alone in their file, and results will be similar to function-level.

If reduction is high (e.g., 0.5), then significant aggregation occurs, and differences will be pronounced.

### Metrics Interpretation

| Metric | Function-level | File-level (expected) |
|--------|---|---|
| **CVE hit@1** | Higher (smaller search space) | May decrease (larger, noisier graphs) |
| **CVE hit@5/10** | Baseline | May improve (richer context) or decrease (noise) |
| **Mean pairwise sim** | Baseline | Likely increase (more redundancy from merging) |
| **Effective dim** | Baseline | May decrease (aggregation compresses structure) |

## Comparison

To compare function-level vs file-level side-by-side:

```bash
# Both results in same benchmark format
results_func = load_json("pipeline_verification_structure/results.json")
results_file = load_json("cvefixes_retrieval_grid_file_level/results.json")

# Compare per embedder
for embedder in ["gin", "combined", "codebert_seq", "codebert_pattern"]:
    func_hit = results_func["cells"][embedder]["self_retrieval"]["hit@5"]
    file_hit = results_file["cells"][embedder]["self_retrieval"]["hit@5"]
    delta = file_hit - func_hit
    print(f"{embedder}: {func_hit:.3f} → {file_hit:.3f} (Δ={delta:+.3f})")
```

## Implementation Notes

### FilePair Class

- Extends the `FunctionPair` interface with additional file-level context
- `func_name` property returns `filename` for compatibility with embedder pipeline
- `meta` includes `original_funcs` list and `num_functions` count

### Graph Merging Semantics

- **Preservation of structure**: Only merges nodes/edges, no re-annotulation
- **No deduplication**: If two functions have identical code sequences, both appear in the merged graph
- **Attribute coexistence**: If a node has the same attributes in both functions, attributes are preserved; conflicts are handled by NetworkX (typically keeps both with annotations)

## Limitations

1. **Sample composition**: Only applies to the 136 pairs in the reference split
2. **File grouping**: Assumes filename uniqueness; if two CVEs modify the same file, they're treated separately
3. **No filtering**: All functions in a file are included; doesn't attempt to isolate vulnerability-specific subsets
4. **Single aggregation method**: Only tests union; could explore other strategies (intersection, weighted sum, etc.)

## Future Directions

- Test alternative aggregation: intersection, attention-weighted, context windows
- Vary aggregation scope: per-file, per-module, per-project
- Analyze impact of file size (number of functions) on degradation
- Hierarchical retrieval: file-level screening → function-level ranking
