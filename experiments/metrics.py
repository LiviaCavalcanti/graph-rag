"""
Evaluation metrics for unsupervised RAG experiments.
All metrics work without ground-truth query→CVE labels.
"""
import numpy as np
from collections import defaultdict


def hits_at_k(results: list[dict], query_cve: str, k: int) -> int:
    """1 if query_cve appears in top-k results, else 0."""
    return int(any(r['cve_id'] == query_cve for r in results[:k]))


def mean_reciprocal_rank(results: list[dict], query_cve: str) -> float:
    """1/rank of first hit, 0 if not found."""
    for i, r in enumerate(results):
        if r['cve_id'] == query_cve:
            return 1.0 / (i + 1)
    return 0.0


def self_retrieval_metrics(
    embeddings: np.ndarray,
    metadata: list[dict],
    retriever,
    ks: list[int] = [1, 5, 10],
) -> dict:
    """
    For each vector, query the index with itself and check if it
    retrieves itself at rank 1. This is the minimum sanity check —
    a perfect embedder + index should score 1.0 at k=1.

    Note: leave-one-out would be stronger but requires rebuilding
    the index N times. Self-retrieval is O(N) and catches
    degenerate embeddings (constant vectors, collapsed dimensions).
    """
    hits  = defaultdict(int)
    mrrs  = []
    n     = len(embeddings)

    for i, (vec, meta) in enumerate(zip(embeddings, metadata)):
        results = retriever.query(vec, top_k=max(ks) + 1)
        # exclude self from results (same index position)
        results_no_self = [r for r in results if r.get('_idx') != i]

        # self-retrieval: the result AT rank 0 should be itself
        # we check by cve_id match since _idx may not be stored
        all_results = retriever.query(vec, top_k=max(ks))
        for k in ks:
            hits[k] += hits_at_k(all_results, meta['cve_id'], k)
        mrrs.append(mean_reciprocal_rank(all_results, meta['cve_id']))

    return {
        **{f'hit@{k}': hits[k] / n for k in ks},
        'mrr': float(np.mean(mrrs)),
        'n':   n,
    }


def leave_one_out_metrics(
    embeddings:  np.ndarray,
    metadata:    list[dict],
    index_class,
    index_kwargs: dict,
    ks: list[int] = [1, 5, 10],
) -> dict:
    """
    For each sample i: build index on all OTHER samples, query with i,
    check if i's CVE appears in top-k.

    Slower (O(N) index builds) but more honest than self-retrieval.
    Uses a lightweight rebuild so it stays tractable for <5k samples.
    """
    hits = defaultdict(int)
    mrrs = []
    n    = len(embeddings)

    for i in range(n):
        mask    = np.ones(n, dtype=bool)
        mask[i] = False

        sub_embs = embeddings[mask]
        sub_meta = [m for j, m in enumerate(metadata) if j != i]

        # build a temporary index on the subset
        tmp_index = index_class(**index_kwargs)
        for vec, meta in zip(sub_embs, sub_meta):
            # wrap meta in a minimal FunctionPair-like object
            tmp_index.add_raw(vec, meta)

        from src.rag.retriever import Retriever
        tmp_retriever = Retriever(tmp_index, top_k=max(ks))
        results = tmp_retriever.query(embeddings[i], top_k=max(ks))

        for k in ks:
            hits[k] += hits_at_k(results, metadata[i]['cve_id'], k)
        mrrs.append(mean_reciprocal_rank(results, metadata[i]['cve_id']))

    return {
        **{f'hit@{k}': hits[k] / n for k in ks},
        'mrr': float(np.mean(mrrs)),
        'n':   n,
    }


def cwe_group_recall(
    embeddings: np.ndarray,
    metadata:   list[dict],
    retriever,
    top_k: int = 10,
) -> dict:
    """
    For each sample, query top-k and measure what fraction share the
    same CWE. Measures whether the embedding space clusters by
    vulnerability type — useful signal even without CVE-level labels.

    Returns per-CWE recall and macro average.
    """
    by_cwe = defaultdict(list)
    for i, m in enumerate(metadata):
        cwe = m.get('cwe_id', 'UNKNOWN')
        if cwe and cwe != 'UNKNOWN':
            by_cwe[cwe].append(i)

    cwe_recalls = {}
    for cwe, indices in by_cwe.items():
        if len(indices) < 2:
            continue  # need at least 2 samples to measure recall

        recalls = []
        for i in indices:
            results  = retriever.query(embeddings[i], top_k=top_k + 1)
            # exclude self
            results  = [r for r in results if r.get('cve_id') != metadata[i]['cve_id']][:top_k]
            same_cwe = sum(1 for r in results if r.get('cwe_id') == cwe)
            possible = min(top_k, len(indices) - 1)
            recalls.append(same_cwe / possible if possible > 0 else 0.0)

        cwe_recalls[cwe] = float(np.mean(recalls))

    macro_avg = float(np.mean(list(cwe_recalls.values()))) if cwe_recalls else 0.0
    return {
        'per_cwe':   cwe_recalls,
        'macro_avg': macro_avg,
        'n_cwes':    len(cwe_recalls),
    }


def embedding_space_stats(embeddings: np.ndarray) -> dict:
    """
    Intrinsic embedding quality metrics — no labels needed.
    Catches degenerate embeddings before running retrieval.
    """
    norms     = np.linalg.norm(embeddings, axis=1)
    # pairwise cosine sim on a random subset (expensive for large N)
    n         = min(len(embeddings), 500)
    idx       = np.random.choice(len(embeddings), n, replace=False)
    sub       = embeddings[idx]
    sim_matrix = sub @ sub.T  # already L2-normed → cosine sim
    # exclude diagonal
    mask      = ~np.eye(n, dtype=bool)
    sims      = sim_matrix[mask]

    return {
        'mean_norm':         float(np.mean(norms)),
        'std_norm':          float(np.std(norms)),
        'mean_pairwise_sim': float(np.mean(sims)),
        'std_pairwise_sim':  float(np.std(sims)),
        'min_pairwise_sim':  float(np.min(sims)),
        'max_pairwise_sim':  float(np.max(sims)),
        # effective dimensionality via participation ratio of PCA eigenvalues
        'effective_dim':     _effective_dim(embeddings),
    }


def _effective_dim(embeddings: np.ndarray) -> float:
    """
    Participation ratio: (sum eigenvalues)^2 / sum(eigenvalues^2).
    = 1 means one dominant direction (collapsed), = d means uniform.
    """
    cov  = np.cov(embeddings.T)
    eigv = np.linalg.eigvalsh(cov)
    eigv = eigv[eigv > 0]
    pr   = (eigv.sum() ** 2) / (eigv ** 2).sum()
    return float(pr)