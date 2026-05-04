"""
Embedding space comparison and quality metrics.

Provides tools to:
- Assess intrinsic quality of a single embedding space (isotropy, hubness, alignment-uniformity)
- Compare two embedding spaces (CKA, k-NN overlap, rank correlation)
- Evaluate class separation using CWE labels (intra/inter ratio)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────────────────────────────
#  Single-space intrinsic metrics
# ─────────────────────────────────────────────────────────────────────


def isotropy(embeddings: np.ndarray) -> float:
    """
    Measure how uniformly embeddings use the available dimensions.

    Computes the partition function ratio:
        I(W) = min_c exp(c^T w) / max_c exp(c^T w)
    approximated via eigenvalues of the covariance matrix.

    Returns a value in (0, 1]:
        1.0 = perfectly isotropic (uniform use of all directions)
        ~0  = anisotropic (a few directions dominate)
    """
    if len(embeddings) < 2:
        return 0.0
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.ndim == 0:
        return 1.0
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 0.0
    return float(eigvals.min() / eigvals.max())


def hubness(embeddings: np.ndarray, k: int = 10) -> dict:
    """
    Measure hubness: how often each point appears as a k-NN of others.

    Hub points are retrieved too often → false positives in retrieval.

    Returns:
        k_occurrence_mean: average times a point is someone's neighbor
        k_occurrence_std: spread (high = hubs exist)
        k_skewness: skewness of k-occurrence distribution (>0 = hubs)
        hub_fraction: fraction of points appearing as neighbor > 2*k times
    """
    n = len(embeddings)
    if n < k + 1:
        return {"k_occurrence_mean": 0, "k_occurrence_std": 0,
                "k_skewness": 0, "hub_fraction": 0}

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    neighbors = nn.kneighbors(embeddings, return_distance=False)[:, 1:]  # exclude self

    # Count how often each point is a neighbor of someone else
    k_occ = np.zeros(n, dtype=int)
    for row in neighbors:
        for idx in row:
            k_occ[idx] += 1

    mean = float(k_occ.mean())
    std = float(k_occ.std())
    # Skewness
    if std > 0:
        skew = float(((k_occ - mean) ** 3).mean() / std**3)
    else:
        skew = 0.0

    hub_threshold = 2 * k
    hub_frac = float((k_occ > hub_threshold).sum() / n)

    return {
        "k_occurrence_mean": mean,
        "k_occurrence_std": std,
        "k_skewness": skew,
        "hub_fraction": hub_frac,
    }


def alignment_uniformity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    t_align: float = 2.0,
    t_uniform: float = 2.0,
    max_pairs: int = 50_000,
) -> dict:
    """
    Wang & Isola (2020) alignment and uniformity on the hypersphere.

    - Alignment: expected distance between same-class pairs (lower = better)
    - Uniformity: log of expected exp(-t * ||x-y||^2) over all pairs (lower = better)

    Args:
        embeddings: (n, d) array, will be L2-normalized internally
        labels: (n,) class labels (e.g., CWE ids encoded as integers)
        t_align: temperature for alignment
        t_uniform: temperature for uniformity
        max_pairs: subsample cap for uniformity computation
    """
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb = embeddings / norms
    n = len(emb)

    # Alignment: same-class pairs
    pos_dists = []
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        group = emb[mask]
        if len(group) < 2:
            continue
        # pairwise distances within class
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pos_dists.append(np.sum((group[i] - group[j]) ** 2))
    if pos_dists:
        pos_dists = np.array(pos_dists)
        alignment = float(np.mean(pos_dists**((t_align / 2))))
    else:
        alignment = 0.0

    # Uniformity: all pairs (subsampled)
    rng = np.random.default_rng(42)
    if n * (n - 1) // 2 > max_pairs:
        idx_i = rng.integers(0, n, size=max_pairs)
        idx_j = rng.integers(0, n, size=max_pairs)
        # Avoid same index
        same = idx_i == idx_j
        idx_j[same] = (idx_j[same] + 1) % n
    else:
        triu = np.triu_indices(n, k=1)
        idx_i, idx_j = triu[0], triu[1]

    diffs = emb[idx_i] - emb[idx_j]
    sq_dists = np.sum(diffs**2, axis=1)
    uniformity = float(np.log(np.mean(np.exp(-t_uniform * sq_dists))))

    return {
        "alignment": alignment,
        "uniformity": uniformity,
    }


# ─────────────────────────────────────────────────────────────────────
#  Class separation (using CWE labels)
# ─────────────────────────────────────────────────────────────────────


def intra_inter_ratio(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_samples_per_class: int = 100,
) -> dict:
    """
    Ratio of intra-class to inter-class cosine similarity.

    Values < 1 mean same-class points are less similar than cross-class → bad.
    Values > 1 mean classes are separated → good retrieval.

    Also returns per-class intra similarity for diagnostics.
    """
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb = embeddings / norms

    unique_labels = np.unique(labels)
    rng = np.random.default_rng(42)

    intra_sims = []
    class_centroids = []
    per_class = {}

    for lab in unique_labels:
        mask = labels == lab
        group = emb[mask]
        if len(group) < 2:
            class_centroids.append(group.mean(axis=0))
            continue
        # Subsample if needed
        if len(group) > max_samples_per_class:
            idx = rng.choice(len(group), max_samples_per_class, replace=False)
            group = group[idx]
        sim_matrix = group @ group.T
        mask_triu = np.triu_indices(len(group), k=1)
        class_sims = sim_matrix[mask_triu]
        mean_sim = float(class_sims.mean())
        intra_sims.append(mean_sim)
        per_class[str(lab)] = mean_sim
        class_centroids.append(group.mean(axis=0))

    # Inter-class: similarity between class centroids
    if len(class_centroids) < 2:
        return {"intra_mean": 0, "inter_mean": 0, "ratio": 0, "per_class": per_class}

    centroids = np.array(class_centroids)
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    c_norms = np.where(c_norms == 0, 1.0, c_norms)
    centroids = centroids / c_norms
    inter_matrix = centroids @ centroids.T
    mask_triu = np.triu_indices(len(centroids), k=1)
    inter_sims = inter_matrix[mask_triu]

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(inter_sims.mean())

    ratio = intra_mean / inter_mean if inter_mean != 0 else float("inf")

    return {
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "ratio": ratio,
        "per_class": per_class,
    }


# ─────────────────────────────────────────────────────────────────────
#  Cross-space comparison metrics
# ─────────────────────────────────────────────────────────────────────


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment (Kornblith et al., 2019).

    Measures structural similarity between two representation matrices.
    Invariant to orthogonal transformations and isotropic scaling.

    Args:
        X: (n, d1) embeddings from space A
        Y: (n, d2) embeddings from space B (same n samples)

    Returns:
        CKA score in [0, 1]. 1 = identical structure.
    """
    assert len(X) == len(Y), "Same number of samples required"
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom == 0:
        return 0.0
    return float(hsic_xy / denom)


def knn_overlap(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    k: int = 10,
) -> dict:
    """
    Fraction of shared k-nearest neighbors between two embedding spaces.

    High overlap → the two spaces retrieve the same items.
    Low overlap → they capture different structure (complementary).

    Args:
        emb_a: (n, d1) first space
        emb_b: (n, d2) second space (same n, same sample ordering)
        k: number of neighbors to compare

    Returns:
        mean_overlap: average Jaccard of neighbor sets
        per_sample_overlap: (n,) array of per-sample overlaps
    """
    assert len(emb_a) == len(emb_b), "Same number of samples required"
    n = len(emb_a)

    nn_a = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb_a)
    nn_b = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb_b)

    neigh_a = nn_a.kneighbors(emb_a, return_distance=False)[:, 1:]
    neigh_b = nn_b.kneighbors(emb_b, return_distance=False)[:, 1:]

    overlaps = np.zeros(n)
    for i in range(n):
        set_a = set(neigh_a[i])
        set_b = set(neigh_b[i])
        overlaps[i] = len(set_a & set_b) / k

    return {
        "mean_overlap": float(overlaps.mean()),
        "std_overlap": float(overlaps.std()),
        "min_overlap": float(overlaps.min()),
        "max_overlap": float(overlaps.max()),
        "per_sample_overlap": overlaps,
    }


def rank_correlation(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    n_queries: int = 200,
) -> dict:
    """
    Spearman rank correlation of pairwise distances between two spaces.

    High correlation → one embedder dominates the combined space.
    Low correlation → spaces encode fundamentally different relationships.

    Args:
        emb_a, emb_b: (n, d) embeddings (same samples, same order)
        n_queries: number of query points to compute distances from
    """
    assert len(emb_a) == len(emb_b)
    n = len(emb_a)
    rng = np.random.default_rng(42)
    query_idx = rng.choice(n, min(n_queries, n), replace=False)

    correlations = []
    for qi in query_idx:
        dist_a = np.linalg.norm(emb_a - emb_a[qi], axis=1)
        dist_b = np.linalg.norm(emb_b - emb_b[qi], axis=1)
        rho, _ = spearmanr(dist_a, dist_b)
        if not np.isnan(rho):
            correlations.append(rho)

    correlations = np.array(correlations) if correlations else np.array([0.0])
    return {
        "mean_rho": float(correlations.mean()),
        "std_rho": float(correlations.std()),
        "min_rho": float(correlations.min()),
        "max_rho": float(correlations.max()),
    }


def trustworthiness(
    emb_high: np.ndarray,
    emb_low: np.ndarray,
    k: int = 10,
) -> float:
    """
    Trustworthiness (Venna & Kaski, 2006).

    Measures whether the k-nearest neighbors in the low-dimensional
    (reduced) space are also neighbors in the high-dimensional (original)
    space. Penalizes "intrusions" — points that are close in the reduced
    space but far in the original.

    Use case: validates that CombinedEmbedder's PCA (384d→128d) does not
    distort local neighborhoods.

    Args:
        emb_high: (n, D) original high-dimensional embeddings (e.g. 384d concat)
        emb_low: (n, d) reduced embeddings (e.g. 128d after PCA)
        k: neighborhood size

    Returns:
        Score in (0, 1]. 1 = perfect preservation of neighborhoods.
    """
    assert len(emb_high) == len(emb_low), "Same number of samples required"
    n = len(emb_high)
    if n < k + 1:
        return 1.0

    # Neighbors in original space
    nn_high = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb_high)
    neigh_high = nn_high.kneighbors(emb_high, return_distance=False)[:, 1:]

    # Neighbors in reduced space
    nn_low = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb_low)
    neigh_low = nn_low.kneighbors(emb_low, return_distance=False)[:, 1:]

    # Ranks in the original space (for penalty computation)
    dist_high = nn_high.kneighbors_graph(emb_high, n_neighbors=n - 1, mode="distance")

    penalty = 0.0
    for i in range(n):
        set_high = set(neigh_high[i])
        set_low = set(neigh_low[i])
        # Intrusions: in low-d neighbors but NOT in high-d neighbors
        intrusions = set_low - set_high
        for j in intrusions:
            # rank of j w.r.t. i in the original space
            row = dist_high[i].toarray().ravel()
            # rank = number of points closer than j in original space
            rank_j = int((row[row > 0] < row[j]).sum()) + 1 if row[j] > 0 else k
            penalty += max(0, rank_j - k)

    # Normalization factor
    norm = n * k * (2 * n - 3 * k - 1)
    if norm == 0:
        return 1.0
    return float(1.0 - (2.0 / norm) * penalty)


def distance_concentration(embeddings: np.ndarray, n_queries: int = 200) -> dict:
    """
    Measure distance concentration (curse of dimensionality).

    In high-dimensional spaces, distances converge:
        (max_dist - min_dist) / min_dist → 0

    When concentration is high, cosine similarity becomes unreliable
    for distinguishing neighbors from non-neighbors.

    Returns:
        relative_contrast_mean: mean of (max-min)/min per query (higher = better)
        relative_contrast_std: spread across queries
        cv_distances: mean coefficient of variation of distances (lower = more concentrated)
    """
    n = len(embeddings)
    rng = np.random.default_rng(42)
    query_idx = rng.choice(n, min(n_queries, n), replace=False)

    # L2 normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb = embeddings / norms

    contrasts = []
    cvs = []

    for qi in query_idx:
        # Cosine distance = 1 - cosine_similarity
        sims = emb @ emb[qi]
        dists = 1.0 - sims
        # Exclude self
        dists = np.delete(dists, qi)
        dists = dists[dists > 0]

        if len(dists) < 2:
            continue

        d_min = dists.min()
        d_max = dists.max()
        if d_min > 0:
            contrasts.append((d_max - d_min) / d_min)
        cvs.append(dists.std() / dists.mean() if dists.mean() > 0 else 0)

    contrasts = np.array(contrasts) if contrasts else np.array([0.0])
    cvs = np.array(cvs) if cvs else np.array([0.0])

    return {
        "relative_contrast_mean": float(contrasts.mean()),
        "relative_contrast_std": float(contrasts.std()),
        "cv_distances": float(cvs.mean()),
    }


# ─────────────────────────────────────────────────────────────────────
#  Convenience: full comparison report
# ─────────────────────────────────────────────────────────────────────


def compare_spaces(
    emb_combined: np.ndarray,
    emb_individual: np.ndarray,
    labels: np.ndarray | None = None,
    emb_high: np.ndarray | None = None,
    k: int = 10,
) -> dict:
    """
    Full comparison between a combined space and an individual embedder space.

    Args:
        emb_combined: (n, d) combined embeddings (post-PCA)
        emb_individual: (n, d) individual embedder embeddings
        labels: (n,) class labels (CWE) for separation metrics. Optional.
        emb_high: (n, D) pre-PCA concatenated embeddings for trustworthiness. Optional.
        k: number of neighbors for overlap/hubness

    Returns dict with all comparison metrics.
    """
    report = {
        "cka": linear_cka(emb_combined, emb_individual),
        "knn_overlap": knn_overlap(emb_combined, emb_individual, k=k),
        "rank_correlation": rank_correlation(emb_combined, emb_individual),
        "combined_isotropy": isotropy(emb_combined),
        "individual_isotropy": isotropy(emb_individual),
        "combined_hubness": hubness(emb_combined, k=k),
        "individual_hubness": hubness(emb_individual, k=k),
        "combined_distance_concentration": distance_concentration(emb_combined),
        "individual_distance_concentration": distance_concentration(emb_individual),
    }

    # Drop per_sample_overlap from summary (keep mean only)
    if "per_sample_overlap" in report["knn_overlap"]:
        report["knn_overlap"] = {
            k_: v for k_, v in report["knn_overlap"].items()
            if k_ != "per_sample_overlap"
        }

    if labels is not None:
        report["combined_class_separation"] = intra_inter_ratio(emb_combined, labels)
        report["individual_class_separation"] = intra_inter_ratio(emb_individual, labels)
        report["combined_alignment_uniformity"] = alignment_uniformity(emb_combined, labels)
        report["individual_alignment_uniformity"] = alignment_uniformity(emb_individual, labels)

    if emb_high is not None:
        report["trustworthiness"] = trustworthiness(emb_high, emb_combined, k=k)

    return report
