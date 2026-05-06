"""
Evaluation metrics for unsupervised RAG experiments.
All metrics work without ground-truth query→CVE labels.
"""

import time
from collections import defaultdict


import numpy as np
from sklearn.metrics import ndcg_score


def hits_at_k(results: list[dict], query_cve: str, k: int) -> int:
    """1 if query_cve appears in top-k results, else 0."""
    return int(any(r["cve_id"] == query_cve for r in results[:k]))


def mean_reciprocal_rank(results: list[dict], query_cve: str) -> float:
    """1/rank of first hit, 0 if not found."""
    for i, r in enumerate(results):
        if r["cve_id"] == query_cve:
            return 1.0 / (i + 1)
    return 0.0


# ── IR metric helpers ────────────────────────────────────────────────


def _reciprocal_rank(ranked_docs: list[tuple[str, float]], qrels: dict[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top-k.

    Scans the ranked list from position 1 to k and returns 1/rank for the
    first document marked relevant in *qrels*. Returns 0.0 if no relevant
    document appears within the cutoff.

    Args:
        ranked_docs: List of (doc_id, score) tuples sorted by score descending.
        qrels: Mapping of doc_id → relevance (>0 means relevant).
        k: Rank cutoff.

    Returns:
        Float in (0, 1] or 0.0.
    """
    for i, (doc_id, _) in enumerate(ranked_docs[:k]):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def _hit_rate_at_k(relevance: list[int], k: int) -> float:
    """Binary hit indicator for the top-k positions.

    Returns 1.0 if at least one relevant document exists in the first k
    positions of the ranked list, 0.0 otherwise. Equivalent to
    ``min(1, recall@k * total_rel)`` but avoids the denominator.

    Args:
        relevance: Binary relevance labels in rank order.
        k: Rank cutoff.
    """
    return 1.0 if any(r > 0 for r in relevance[:k]) else 0.0


def _precision_at_k(relevance: list[int], k: int) -> float:
    """Precision at rank k: fraction of retrieved documents that are relevant.

    P@k = |{relevant docs in top-k}| / k

    Args:
        relevance: Binary relevance labels in rank order.
        k: Rank cutoff (must be > 0).
    """
    return sum(relevance[:k]) / k


def _recall_at_k(relevance: list[int], k: int, total_rel: int) -> float:
    """Recall at rank k: fraction of all relevant documents found in top-k.

    R@k = |{relevant docs in top-k}| / |{all relevant docs}|

    Returns 0.0 when total_rel is 0 (no relevant documents exist).

    Args:
        relevance: Binary relevance labels in rank order.
        k: Rank cutoff.
        total_rel: Total number of relevant documents in the collection.
    """
    if total_rel == 0:
        return 0.0
    return sum(relevance[:k]) / total_rel


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k.

    Uses ``sklearn.metrics.ndcg_score`` which computes:
        DCG@k  = Σ_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k

    Returns 0.0 when there are no relevant documents or fewer than 2
    documents in the ranking (sklearn requires at least 2).

    Args:
        y_true: Ground-truth relevance scores for all ranked documents.
        y_score: Predicted scores for all ranked documents (same order).
        k: Rank cutoff.
    """
    if y_true.sum() == 0 or len(y_true) < 2:
        return 0.0
    try:
        return float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k))
    except ValueError:
        return 0.0


def _average_precision_at_k(relevance: list[int], k: int, total_rel: int) -> float:
    """Average Precision at rank k (AP@k).

    AP@k = (1 / min(total_rel, k)) * Σ_{i=1}^{k} P(i) * rel(i)

    where P(i) is precision at position i and rel(i) is the binary
    relevance of the document at position i. The denominator is
    ``min(total_rel, k)`` so that AP is 1.0 when all available relevant
    documents are ranked at the top.

    Returns 0.0 when total_rel is 0.

    Args:
        relevance: Binary relevance labels in rank order.
        k: Rank cutoff.
        total_rel: Total number of relevant documents in the collection.
    """
    if total_rel == 0:
        return 0.0
    ap = 0.0
    n_rel = 0
    for rank_i, rel in enumerate(relevance[:k]):
        if rel > 0:
            n_rel += 1
            ap += n_rel / (rank_i + 1)
    return ap / min(total_rel, k)


def _compute_metrics(
    qrels_dict: dict[str, dict[str, int]],
    run_dict: dict[str, dict[str, float]],
    ks: list[int],
) -> dict:
    """Compute standard IR metrics using sklearn/numpy.

    Computes hit_rate@k, precision@k, recall@k, ndcg@k, map@k for each k,
    and mrr@max(ks). Averaged over all queries in run_dict.
    """
    max_k = max(ks)

    per_k: dict[int, dict[str, list[float]]] = {
        k: {"hit_rate": [], "precision": [], "recall": [], "ndcg": [], "map": []}
        for k in ks
    }
    mrrs = []

    for qid in run_dict:
        q_qrels = qrels_dict.get(qid, {})
        q_run = run_dict[qid]

        ranked_docs = sorted(q_run.items(), key=lambda x: x[1], reverse=True)
        relevance = [q_qrels.get(doc_id, 0) for doc_id, _ in ranked_docs]
        total_rel = sum(1 for v in q_qrels.values() if v > 0)

        mrrs.append(_reciprocal_rank(ranked_docs, q_qrels, max_k))

        y_true = np.array(relevance, dtype=float)
        y_score = np.array([s for _, s in ranked_docs], dtype=float)

        for k in ks:
            per_k[k]["hit_rate"].append(_hit_rate_at_k(relevance, k))
            per_k[k]["precision"].append(_precision_at_k(relevance, k))
            per_k[k]["recall"].append(_recall_at_k(relevance, k, total_rel))
            per_k[k]["ndcg"].append(_ndcg_at_k(y_true, y_score, k))
            per_k[k]["map"].append(_average_precision_at_k(relevance, k, total_rel))

    # Aggregate across queries
    results = {}
    for k in ks:
        for metric_name, vals in per_k[k].items():
            results[f"{metric_name}@{k}"] = float(np.mean(vals)) if vals else 0.0
    results[f"mrr@{max_k}"] = float(np.mean(mrrs)) if mrrs else 0.0

    return results


def _cwe_standard_recall(
    per_query: list[tuple[str, str, list[dict], int | None]],
    cwe_support: dict[str, int],
    top_k: int,
) -> float:
    """Standard recall@k for CWE retrieval using CWE string matching.

    Unlike the capped ``macro_avg`` (denominator = min(top_k, support)),
    this uses the full support as the denominator so the score reflects
    how much of the CWE neighbourhood was retrieved.

    Excludes the query itself (by ``_idx``) when *self_idx* is set.

    Args:
        per_query: List of (query_cwe, qid, results, self_idx) tuples.
        cwe_support: Mapping of CWE → total number of index entries.
        top_k: Rank cutoff applied to results.

    Returns:
        Mean recall across all eligible queries, or 0.0 if none.
    """
    recall_values: list[float] = []
    for query_cwe, qid, results, self_idx in per_query:
        support = cwe_support.get(query_cwe, 0)
        if self_idx is not None:
            support -= 1
        if support <= 0:
            continue
        found = 0
        for r in results[:top_k]:
            if r.get("cwe_id") == query_cwe:
                if self_idx is None or r.get("_idx") != self_idx:
                    found += 1
        recall_values.append(found / support)
    return float(np.mean(recall_values)) if recall_values else 0.0


def _cwe_recall_summary(
    per_query: list[tuple[str, str, list[dict], int | None]],
    index_metadata: list[dict],
    top_k: int,
) -> dict:
    """Shared CWE-recall computation for both cross-split and self-retrieval.

    *per_query* is a list of ``(query_cwe, qid, results, self_idx)`` tuples.
    *self_idx* is the FAISS index position of the query itself (for
    self-retrieval where the query is *inside* the index) or ``None``
    (for cross-split where query and index are disjoint).

    When *self_idx* is not ``None``:
    - CWE support is decremented by 1 (the query cannot retrieve itself).
    - That position is excluded when counting matches.

    Returns a dict with per_cwe breakdown, macro_avg (capped recall,
    denominator = min(top_k, support)), ranx_recall (standard recall@k,
    denominator = support), n_cwes, and n_singletons.
    """
    cwe_support = defaultdict(int)
    for m in index_metadata:
        cwe = m.get("cwe_id")
        if cwe and cwe != "UNKNOWN":
            cwe_support[cwe] += 1

    per_cwe_scores: dict[str, list[float]] = defaultdict(list)
    n_singletons_seen: set[str] = set()

    for query_cwe, qid, results, self_idx in per_query:
        support = cwe_support.get(query_cwe, 0)
        if self_idx is not None:
            support -= 1  # exclude self from available peers
        possible = min(top_k, support)
        if possible <= 0:
            n_singletons_seen.add(query_cwe)
            continue

        same = sum(1 for r in results if r.get("cwe_id") == query_cwe)
        per_cwe_scores[query_cwe].append(same / possible)

    per_cwe = {
        cwe: {
            "recall": float(np.mean(vals)),
            "support": int(cwe_support.get(cwe, 0)),
        }
        for cwe, vals in per_cwe_scores.items()
    }
    macro = float(np.mean([v["recall"] for v in per_cwe.values()])) if per_cwe else 0.0

    std_recall = _cwe_standard_recall(per_query, cwe_support, top_k)

    return {
        "per_cwe": per_cwe,
        "macro_avg": macro,
        "ranx_recall": std_recall,
        "n_cwes": len(per_cwe),
        "n_singletons": len(n_singletons_seen - set(per_cwe)),
    }


# not currently used, kept for reference for previous work
# def self_retrieval_metrics(
#     embeddings: np.ndarray,
#     metadata: list[dict],
#     retriever,
#     ks: list[int] = [1, 5, 10],
# ) -> dict:
#     """
#     For each vector, query the index with itself and check if it
#     retrieves itself at rank 1. This is the minimum sanity check —
#     a perfect embedder + index should score 1.0 at k=1.

#     Note: leave-one-out would be stronger but requires rebuilding
#     the index N times. Self-retrieval is O(N) and catches
#     degenerate embeddings (constant vectors, collapsed dimensions).
#     """
#     hits = defaultdict(int)
#     mrrs = []
#     n = len(embeddings)

#     for i, (vec, meta) in enumerate(zip(embeddings, metadata)):
#         results = retriever.query(vec, top_k=max(ks))
#         for k in ks:
#             hits[k] += hits_at_k(results, meta["cve_id"], k)
#         mrrs.append(mean_reciprocal_rank(results, meta["cve_id"]))

#     return {
#         **{f"hit@{k}": hits[k] / n for k in ks},
#         "mrr": float(np.mean(mrrs)),
#         "n": n,
#     }

# DEPRECATED
def leave_one_out_metrics(
    embeddings: np.ndarray,
    metadata: list[dict],
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
    n = len(embeddings)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        sub_embs = embeddings[mask]
        sub_meta = [m for j, m in enumerate(metadata) if j != i]

        # build a temporary index on the subset
        tmp_index = index_class(**index_kwargs)
        for vec, meta in zip(sub_embs, sub_meta):
            # wrap meta in a minimal FunctionPair-like object
            tmp_index.add_raw(vec, meta)

        from rag.retriever import Retriever

        tmp_retriever = Retriever(tmp_index, top_k=max(ks))
        results = tmp_retriever.query(embeddings[i], top_k=max(ks))

        for k in ks:
            hits[k] += hits_at_k(results, metadata[i]["cve_id"], k)
        mrrs.append(mean_reciprocal_rank(results, metadata[i]["cve_id"]))

    return {
        **{f"hit@{k}": hits[k] / n for k in ks},
        "mrr": float(np.mean(mrrs)),
        "n": n,
    }


def embedding_space_stats(embeddings: np.ndarray) -> dict:
    """
    Intrinsic embedding quality metrics — no labels needed.
    Catches degenerate embeddings before running retrieval.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    # pairwise cosine sim on a random subset (expensive for large N)
    n = min(len(embeddings), 500)
    idx = np.random.choice(len(embeddings), n, replace=False)
    sub = embeddings[idx]
    # L2-normalize before computing cosine similarity
    sub_norms = np.linalg.norm(sub, axis=1, keepdims=True)
    sub_norms = np.where(sub_norms == 0, 1.0, sub_norms)
    sub = sub / sub_norms
    sim_matrix = sub @ sub.T
    # exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    sims = sim_matrix[mask]

    return {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "mean_pairwise_sim": float(np.mean(sims)),
        "std_pairwise_sim": float(np.std(sims)),
        "min_pairwise_sim": float(np.min(sims)),
        "max_pairwise_sim": float(np.max(sims)),
        # effective dimensionality via participation ratio of PCA eigenvalues
        "effective_dim": _effective_dim(embeddings),
    }


def _effective_dim(embeddings: np.ndarray) -> float:
    """
    Participation ratio: (sum eigenvalues)^2 / sum(eigenvalues^2).
    = 1 means one dominant direction (collapsed), = d means uniform.
    
    Ansuini, A., Laio, A., Macke, J.H., & Zoccolan, D. (2019). 
    "Intrinsic dimension of data representations in deep neural networks."
    NeurIPS 2019.
    """
    if len(embeddings) < 2:
        return 0.0
    cov = np.cov(embeddings.T)
    if cov.ndim == 0:
        # Single feature dimension: cov is a scalar
        return 1.0 if float(cov) > 0 else 0.0
    eigv = np.linalg.eigvalsh(cov)
    eigv = eigv[eigv > 0]
    if len(eigv) == 0:
        return 0.0
    pr = (eigv.sum() ** 2) / (eigv**2).sum()
    return float(pr)


def measure_latency(retriever, embeddings: np.ndarray, n_queries: int = 200) -> dict:
    """Sample query latencies in milliseconds."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(embeddings), min(n_queries, len(embeddings)), replace=False)
    samples = embeddings[idx]
    times = []
    for vec in samples:
        t0 = time.perf_counter()
        retriever.query(vec, top_k=10)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "mean_ms": float(np.mean(times)),
    }