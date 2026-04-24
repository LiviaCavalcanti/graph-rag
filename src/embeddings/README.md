# Embedders

Graph embedding pipeline for vulnerability retrieval. Each embedder converts a CPG (Code Property Graph) into a fixed-dimensional vector suitable for ANN search.

---

## Architecture overview

```mermaid
flowchart TD
    subgraph Input
        G["G_vuln<br/>(nx.MultiDiGraph)<br/>CPG with diff labels"]
    end

    G --> S["Structural embedders"]
    G --> T["Semantic embedders"]
    G --> F["Fusion embedders"]

    subgraph S[" "]
        direction TB
        S1["GIN<br/>128d"]
        S2["WL<br/>128d"]
        S3["NetLSD<br/>128d"]
    end

    subgraph T[" "]
        direction TB
        T1["CodeBERT seq<br/>768d → PCA → 128d"]
        T2["RGCN<br/>128d"]
    end

    subgraph F[" "]
        direction TB
        F1["Combined<br/>(NetLSD + WL + GIN) → PCA → 128d"]
        F2["CodeBERT Pattern<br/>(VulnPattern 34d + CodeBERT 768d) → PCA → 128d"]
    end

    S1 --> V["L2-normalised vector<br/>(128d)"]
    S2 --> V
    S3 --> V
    T1 --> V
    T2 --> V
    F1 --> V
    F2 --> V

    V --> IDX["HNSW Index"]
```

---

## Embedding pipeline per graph

Every embedder follows the same contract defined in `base.py`:

```mermaid
flowchart LR
    A["nx.MultiDiGraph"] -->|embed_one| B["np.ndarray<br/>(dim,)"]
    A2["list[MultiDiGraph]"] -->|embed_many| C["np.ndarray<br/>(N, dim)"]
    C -->|PCA fit on first batch| C
```

- `embed_many()` is called first on index graphs — this fits PCA if the embedder uses it
- `embed_one()` projects through the already-fitted PCA (for query-time embedding)
- All outputs are L2-normalised so cosine similarity = dot product

---

## Structural embedders

These operate on graph topology and node types. They do **not** use pretrained language models.

### GIN (Graph Isomorphism Network)

```mermaid
flowchart LR
    A["Node types<br/>(one-hot, 11d)"] --> B["Linear proj<br/>→ 128d"]
    B --> C["GIN layer 1<br/>(MLP + sum)"]
    C --> D["GIN layer 2"]
    D --> E["GIN layer 3"]
    E --> F["Global pool<br/>(add + mean)"]
    F --> G["Linear readout<br/>→ 128d"]
    G --> H["L2 norm"]
```

- **Input:** one-hot node type (METHOD, CALL, IDENTIFIER, etc.)
- **Random weights** — frozen, no training. Random MLP weights in GIN still produce structurally discriminative embeddings
- **Why it works:** GIN is provably as powerful as the WL test for graph isomorphism. Two graphs with different local structure produce different embeddings

### WL (Weisfeiler-Lehman)

```mermaid
flowchart LR
    A["Node type<br/>colours"] --> B["WL iteration 1<br/>(colour refine)"]
    B --> C["Embed + pool"]
    C --> D["WL iteration 2"]
    D --> E["Embed + pool"]
    E --> F["... × 4 iters"]
    F --> G["Concat all pools"]
    G --> H["Linear proj<br/>→ 128d"]
    H --> I["L2 norm"]
```

- **Input:** integer node colour from type index
- **4 iterations** of colour refinement → embedding lookup → sum pooling per iteration
- Concatenates all iteration outputs → linear projection

### NetLSD (Network Laplacian Spectral Descriptor)

```mermaid
flowchart LR
    A["nx.MultiDiGraph"] --> B["Convert to<br/>undirected Graph"]
    B --> C["Remove isolates"]
    C --> D["Laplacian<br/>eigenvalues"]
    D --> E["Heat kernel<br/>trace at 128<br/>log-spaced scales"]
    E --> F["L2 norm"]
```

- Purely spectral — encodes global graph shape, not node content
- Timescales from $10^{-2}$ to $10^{2}$
- Fast but least discriminative (no node features, no diff labels)

---

## Semantic embedders

### CodeBERT seq

```mermaid
flowchart TD
    A["G_vuln nodes"] --> B{"diff_weight<br/>> 0.3?"}
    B -- yes --> C["Collect CODE text<br/>sorted by weight"]
    B -- no --> D["Fallback: all CODE"]
    C --> E["Concatenate<br/>(max 400 tokens)"]
    D --> E
    E --> F["CodeBERT<br/>[CLS] token → 768d"]
    F --> G["PCA → 128d"]
    G --> H["L2 norm"]
```

- **Text-only** baseline — no graph structure used
- Filters nodes by `diff_weight > 0.3` (keeps `removed`, `fix_adjacent`, `edge_changed`; drops `context`)
- Sorts by importance (weight descending) then line number
- PCA fitted on first `embed_many()` call

### RGCN (Relational Graph Convolutional Network)

```mermaid
flowchart TD
    subgraph NodeFeatures["Node feature construction (per node)"]
        direction LR
        NF1["type_onehot<br/>(15d)"]
        NF2["diff_onehot<br/>(8d)"]
        NF3["diff_weight<br/>(1d)"]
        NF4["CodeBERT [CLS]<br/>(768d)"]
        NF5["semantic_flags<br/>(6d)"]
        NF1 --- NF2 --- NF3 --- NF4 --- NF5
    end

    NodeFeatures --> A["Concat → 798d"]
    A --> B["R-GCN layers<br/>(per-relation weights:<br/>AST, CFG, CDG, REACHING_DEF,<br/>REF, ARGUMENT, RECEIVER, CALL)"]
    B --> C["Weighted global pool<br/>(weights = diff_weight)"]
    C --> D["MLP projection<br/>→ 128d"]
    D --> E["L2 norm"]
```

- **Heterogeneous edges** — separate weight matrix per edge type (8 relation types)
- **Rich node features** — combines structural type, diff annotations, CodeBERT code embedding, and semantic flags (pointer ops, alloc/free, lock/unlock, checks, code length)
- **Weighted pooling** — nodes with higher diff_weight contribute more to the global embedding
- Frozen random weights (no training)
- Graphs > 200 nodes are trimmed (keep highest diff_weight + degree)

Semantic flags extracted from CODE text per node:
| Flag | Pattern |
|---|---|
| Pointer op | `*var`, `->` |
| Allocation | `malloc`, `alloc`, `new` |
| Free | `free`, `kfree`, `delete` |
| Lock | `mutex_lock`, `spin_lock` |
| Check | `if`, `assert`, `BUG_ON` |
| Code length | `len(words) / 20` |

---

## Fusion embedders

### Combined (NetLSD + WL + GIN → PCA)

```mermaid
flowchart LR
    G["G_vuln"] --> A["NetLSD<br/>(128d)"]
    G --> B["WL<br/>(128d)"]
    G --> C["GIN<br/>(128d)"]
    A --> D["Concat<br/>(384d)"]
    B --> D
    C --> D
    D --> E["PCA → 128d"]
    E --> F["L2 norm"]
```

- Concatenates three structural embedders, then PCA-reduces
- PCA fitted on first `embed_many()` batch
- **Best structural-only performer** — benefits from each sub-embedder capturing different graph properties (spectrum, colour refinement, message passing)

### CodeBERT Pattern (VulnPattern + CodeBERT → PCA)

```mermaid
flowchart TD
    G["G_vuln"] --> A["VulnPattern<br/>(34d)"]
    G --> B["CodeBERT seq<br/>(768d)"]
    A --> C["Concat<br/>(802d)"]
    B --> C
    C --> D["PCA → 128d"]
    D --> E["L2 norm"]
```

**VulnPattern** extracts 34 vulnerability-specific structural features in 5 groups:

| Group | Dims | What it captures |
|---|---|---|
| A. Vulnerability flow patterns | 8 | UAF (free→use via REACHING_DEF), NPD (deref without CDG guard), unchecked return, lock imbalance, arithmetic without bounds, use-without-def, alloc/free imbalance, cast connected to changes |
| B. Diff edge composition | 8 | Fraction of each edge type touching changed nodes |
| C. Boundary flow direction | 6 | CFG/CDG/REACHING_DEF edges crossing changed↔context boundary |
| D. Diff topology | 6 | Change density, connected components, degree distribution, internal vs crossing edges |
| E. Changed node roles | 6 | Distribution of CALL, CONTROL_STRUCTURE, IDENTIFIER, LITERAL, BLOCK among changed nodes |

```mermaid
flowchart LR
    subgraph "Group A: Flow Patterns (8d)"
        A1["free→use<br/>(UAF)"]
        A2["deref no guard<br/>(NPD)"]
        A3["unchecked<br/>return"]
        A4["lock/unlock<br/>imbalance"]
    end
    subgraph "Group B: Edge Composition (8d)"
        B1["AST frac"]
        B2["CFG frac"]
        B3["CDG frac"]
        B4["REACHING_DEF<br/>frac"]
    end
    subgraph "Group C: Boundary (6d)"
        C1["changed→ctx<br/>vs ctx→changed<br/>per edge type"]
    end
```

**Key ablation:** If CodeBERT Pattern beats CodeBERT seq → graph structure adds value. If it beats VulnPattern alone → CodeBERT adds value.

---

## Node feature pipeline (used by GIN, WL, RGCN)

```mermaid
flowchart TD
    A["CPG node"] --> B["labelV attribute"]
    B --> C{"Known type?"}
    C -- yes --> D["One-hot index<br/>(METHOD=0, CALL=6, ...)"]
    C -- no --> E["UNKNOWN index"]
    D --> F["Node colour / feature vector"]
    E --> F
```

Node types used across embedders:
```
METHOD, METHOD_PARAMETER_IN, BLOCK, LOCAL, CALL,
IDENTIFIER, LITERAL, RETURN, CONTROL_STRUCTURE,
FIELD_IDENTIFIER, UNKNOWN
```

RGCN extends this with diff labels and CodeBERT features. GIN and WL use only the type one-hot.

---

## Diff annotation features

Embedders that use diff annotations get node-level labels from `compute_graph_diff()`:

| Label | Weight | Meaning | Used by |
|---|---|---|---|
| `removed` | 1.0 | Code deleted by patch | All (as feature) |
| `fix_adjacent` | 0.8 | Neighbour of inserted fix | RGCN (weight), CodeBERT seq (filter) |
| `edge_changed` | 0.6 | Endpoint of changed edge | RGCN (weight), CodeBERT seq (filter) |
| `context` | 0.2 | Unchanged, reached by BFS | All (as feature or ignored) |

How each embedder uses diff info:
- **RGCN** — diff_onehot as node feature + diff_weight for weighted global pooling
- **CodeBERT seq** — filters nodes with `diff_weight > 0.3`, sorts by weight
- **GIN, WL** — diff label used as categorical node feature (when present on G_vuln)
- **NetLSD** — ignores diff labels entirely (spectral only)

---

## PCA behaviour

Several embedders use PCA for dimensionality reduction. The PCA lifecycle:

```mermaid
stateDiagram-v2
    [*] --> Unfitted: __init__()
    Unfitted --> Fitted: embed_many()<br/>fits PCA on batch
    Fitted --> Fitted: embed_many()<br/>transforms only
    Fitted --> Fitted: embed_one()<br/>transforms only
    Fitted --> Unfitted: manual reset<br/>(_fitted = False)
```

**Correct protocol for evaluation:**
1. Call `embed_many(index_graphs)` — fits PCA on index distribution
2. Call `embed_one(query_graph)` for each query — projects through index PCA
3. Never refit PCA on query data (information leakage)

Embedders with PCA: `combined`, `codebert_seq`, `codebert_pattern`, `vuln_pattern`, `rgcn`
