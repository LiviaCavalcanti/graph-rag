"""
Evaluation metrics for unsupervised RAG experiments.
All metrics work without ground-truth query→CVE labels.
"""

import time
from collections import defaultdict


import numpy as np

from src.metrics.retrieval_eval import _cwe_recall_summary


def hits_at_k(results: list[dict], query_cve: str, k: int) -> int:
    """1 if query_cve appears in top-k results, else 0."""
    return int(any(r["cve_id"] == query_cve for r in results[:k]))


def mean_reciprocal_rank(results: list[dict], query_cve: str) -> float:
    """1/rank of first hit, 0 if not found."""
    for i, r in enumerate(results):
        if r["cve_id"] == query_cve:
            return 1.0 / (i + 1)
    return 0.0

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


def cwe_group_recall(
    embeddings: np.ndarray,
    metadata: list[dict],
    retriever,
    top_k: int = 10,
) -> dict:
    """
    **Self-retrieval** CWE recall: query and index are the *same* set.

    For each sample, query top-k+1 (to account for the self-hit),
    remove the query itself via ``_idx``, then measure what fraction
    of the remaining top-k share the same CWE.

    Unlike ``cross_cwe_recall`` / ``evaluate_cwe_recall`` (where
    query and index are disjoint), this function must:
    - fetch one extra result and filter the self-hit by FAISS position,
    - use ``support - 1`` as the recall denominator (self is not a peer).

    Returns per-CWE recall, macro average, ranx recall, and singletons.
    """
    by_cwe = defaultdict(list)
    unknown_cwe_count = 0
    for i, m in enumerate(metadata):
        cwe = m.get("cwe_id", "UNKNOWN")
        if cwe and cwe != "UNKNOWN":
            by_cwe[cwe].append(i)
        else:
            unknown_cwe_count += 1
    print(
        f"Metadata contains {len(by_cwe)} unique CWEs, plus {unknown_cwe_count} with unknown CWE."
    )

    per_query: list[tuple[str, str, list[dict], int | None]] = []
    raw_queries = []
    n = 0

    for cwe, indices in by_cwe.items():
        for i in indices:
            results = retriever.query(embeddings[i], top_k=top_k + 1)
            # Remove self-hit by FAISS index position, keep augmented
            # variants that share the same cve_id.
            results = [r for r in results if r.get("_idx") != i][:top_k]

            qid = f"q{n}"
            per_query.append((cwe, qid, results, i))

            support = len(indices) - 1
            possible = min(top_k, support)
            same_cwe = sum(1 for r in results if r.get("cwe_id") == cwe)
            raw_queries.append(
                {
                    "query_cve": metadata[i]["cve_id"],
                    "query_cwe": cwe,
                    "recall": same_cwe / possible if possible > 0 else 0.0,
                    "retrieved": [
                        {
                            "rank": j + 1,
                            "cve_id": r.get("cve_id"),
                            "cwe_id": r.get("cwe_id"),
                            "score": r.get("score"),
                        }
                        for j, r in enumerate(results)
                    ],
                }
            )
            n += 1

    summary = _cwe_recall_summary(per_query, metadata, top_k)
    return {
        **summary,
        "raw_queries": raw_queries,
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