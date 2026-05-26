"""
Plot t-SNE of codebert_pattern embeddings on token_guided slices,
colored by true vulnerability family, markers by K-means cluster.
Also computes retrieval metrics (P@k, MAP, MRR).
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import yaml
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvefixes_experiments.scripts.slicing_depth_study import (
    CWE_FAMILIES,
    N_FAMILIES,
    SAMPLES_PER_FAMILY,
    SEED,
    compute_token_guided_slice_all,
    prepare_dataset,
)
from src.embeddings import REGISTRY

# ── Config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
emb_cfg = cfg["embeddings"]

# ── Prepare data
n_total = N_FAMILIES * SAMPLES_PER_FAMILY
prepared = prepare_dataset(n_total, seed=SEED)

# ── Build token-guided slices
slices = []
for item in prepared:
    G_before = item["G_before"]
    family = item["entry"].get("cwe_family")
    if not family:
        continue
    G_token = compute_token_guided_slice_all(G_before, slice_depth=2)
    if G_token.number_of_nodes() >= 3:
        slices.append((G_token, family))

print(f"Got {len(slices)} valid slices")

graphs = [g for g, _ in slices]
true_labels = [lbl for _, lbl in slices]

label_set = sorted(set(true_labels))
label_to_int = {l: i for i, l in enumerate(label_set)}
y_true = np.array([label_to_int[l] for l in true_labels])

# ── Embed with codebert_pattern
embedder = REGISTRY["codebert_pattern"](emb_cfg)
embeddings = embedder.embed_many(graphs)
print(f"Embeddings shape: {embeddings.shape}")

# ── K-means
km = KMeans(n_clusters=N_FAMILIES, random_state=SEED, n_init=10)
y_pred = km.fit_predict(embeddings)

# ── Retrieval metrics ──────────────────────────────────────────────────
# For each sample as query, rank all others by cosine similarity,
# check if retrieved neighbours share the same family.
sim_matrix = cosine_similarity(embeddings)
np.fill_diagonal(sim_matrix, -1)  # exclude self

n = len(y_true)
K_VALUES = [1, 3, 5, 10]

precisions = {k: [] for k in K_VALUES}
avg_precisions = []
reciprocal_ranks = []

for i in range(n):
    ranked = np.argsort(-sim_matrix[i])  # descending similarity
    relevant = (y_true[ranked] == y_true[i])

    # P@k
    for k in K_VALUES:
        precisions[k].append(relevant[:k].sum() / k)

    # MRR — rank of first relevant
    first_rel = np.where(relevant)[0]
    if len(first_rel) > 0:
        reciprocal_ranks.append(1.0 / (first_rel[0] + 1))
    else:
        reciprocal_ranks.append(0.0)

    # AP — average precision over all relevant
    n_relevant = relevant.sum()
    if n_relevant > 0:
        cum_rel = np.cumsum(relevant)
        prec_at_rank = cum_rel / np.arange(1, len(relevant) + 1)
        avg_precisions.append((prec_at_rank * relevant).sum() / n_relevant)
    else:
        avg_precisions.append(0.0)

print("\n" + "=" * 50)
print("  RETRIEVAL METRICS (codebert_pattern, token_guided)")
print("=" * 50)
for k in K_VALUES:
    print(f"  P@{k:<2d} = {np.mean(precisions[k]):.3f}")
print(f"  MAP  = {np.mean(avg_precisions):.3f}")
print(f"  MRR  = {np.mean(reciprocal_ranks):.3f}")
print("=" * 50)

# Per-family retrieval breakdown
print(f"\n  {'Family':<22s} {'P@1':>5s} {'P@3':>5s} {'P@5':>5s} {'MAP':>6s} {'MRR':>6s}")
print(f"  {'─' * 52}")
for fi, fam in enumerate(label_set):
    mask = y_true == fi
    idx = np.where(mask)[0]
    fam_p1 = np.mean([precisions[1][i] for i in idx])
    fam_p3 = np.mean([precisions[3][i] for i in idx])
    fam_p5 = np.mean([precisions[5][i] for i in idx])
    fam_map = np.mean([avg_precisions[i] for i in idx])
    fam_mrr = np.mean([reciprocal_ranks[i] for i in idx])
    print(f"  {fam:<22s} {fam_p1:>5.3f} {fam_p3:>5.3f} {fam_p5:>5.3f} {fam_map:>6.3f} {fam_mrr:>6.3f}")

# ── t-SNE to 2D
tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(graphs) - 1))
X_2d = tsne.fit_transform(embeddings)

# ── Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Color palette
colors = plt.cm.Set1(np.linspace(0, 1, N_FAMILIES))
markers = ["o", "s", "^", "D", "v"]

# Left: colored by TRUE family
ax = axes[0]
for i, fam in enumerate(label_set):
    mask = y_true == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[colors[i]], label=fam, alpha=0.7, s=50, edgecolors="k", linewidths=0.3)
ax.set_title("True Vulnerability Families", fontsize=13)
ax.legend(fontsize=8, loc="best")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")

# Right: colored by K-means cluster
ax = axes[1]
cluster_colors = plt.cm.tab10(np.linspace(0, 1, N_FAMILIES))
for i in range(N_FAMILIES):
    mask = y_pred == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[cluster_colors[i]], label=f"Cluster {i}", alpha=0.7, s=50,
               edgecolors="k", linewidths=0.3)
ax.set_title("K-means Clusters (k=5)", fontsize=13)
ax.legend(fontsize=8, loc="best")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")

plt.suptitle("codebert_pattern on token_guided slices\n"
             f"(n={len(graphs)}, ARI={float(np.round(km.inertia_, 0)):.0f} inertia)",
             fontsize=14)
plt.tight_layout()

out_path = Path("cvefixes_experiments/output/token_guided_codebert_pattern_clusters.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to: {out_path}")
