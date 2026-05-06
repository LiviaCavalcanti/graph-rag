"""
Retrieval evaluation metrics — code-query hit@k / MRR and cross-CWE recall.

These operate on (pairs, embedder, retriever) and are independent of
the experiment runner's orchestration logic.
"""

import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import ndcg_score

from src.io import read_code_file


# ── ranx helpers ─────────────────────────────────────────────────────


def _doc_id(result: dict, rank: int, qid: str) -> str:
    """Return a stable global document id from a retriever result.

    Two cases:
    - Live retrieval (via ``Retriever.query()``): ``_idx`` is always
      present — it's the FAISS vector position injected by the Retriever.
    - Offline evaluation (via ``evaluate_retrieval_from_records()``):
      results are deserialized from JSON and lack ``_idx``, so we
      synthesize a per-query rank-based id instead.
    """
    idx = result.get("_idx")
    if idx is not None:
        return f"d{idx}"
    return f"d_rank{rank}_{qid}"


def _build_cve_qrels_and_run(
    query_results: list[tuple[str, str, list[dict]]],
    index_metadata: list[dict] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """Build qrels/run dicts for CVE-level retrieval.

    *query_results* is a list of ``(query_id, query_cve, results)`` triples
    where *results* are the ranked dicts returned by the retriever.

    If *index_metadata* is provided, **all** index entries sharing the
    query CVE are marked relevant (needed for recall).  Otherwise only
    retrieved entries are judged.

    Returns plain dicts (not ranx objects):
    - qrels_dict: ``{qid: {doc_id: relevance}}``
    - run_dict: ``{qid: {doc_id: score}}``
    """
    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}

    # Pre-compute per-CVE doc ids from the full index when available
    cve_to_docs: dict[str, list[str]] = defaultdict(list)
    if index_metadata is not None:
        for idx, m in enumerate(index_metadata):
            cve_to_docs[m.get("cve_id", "")].append(f"d{idx}")

    for qid, query_cve, results in query_results:
        q_qrels: dict[str, int] = {}
        q_run: dict[str, float] = {}

        # Mark all index docs with the same CVE as relevant
        if index_metadata is not None:
            for doc_id in cve_to_docs.get(query_cve, []):
                q_qrels[doc_id] = 1

        for j, r in enumerate(results):
            doc_id = _doc_id(r, j, qid)
            q_run[doc_id] = float(r.get("score", 0.0))
            # Without full index metadata, judge from retrieved results
            if index_metadata is None and r.get("cve_id") == query_cve:
                q_qrels[doc_id] = 1

        if q_run:
            qrels_dict[qid] = q_qrels if q_qrels else {"__none__": 0}
            run_dict[qid] = q_run

    return qrels_dict, run_dict


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
    - That position is excluded from ranx qrels.

    Returns a dict with per_cwe breakdown, macro_avg (capped recall),
    ranx_recall (standard recall@k), n_cwes, and n_singletons.
    """
    cwe_support = defaultdict(int)
    for m in index_metadata:
        cwe = m.get("cwe_id")
        if cwe and cwe != "UNKNOWN":
            cwe_support[cwe] += 1

    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}
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

        # qrels: all index docs with same CWE (excluding self when applicable)
        q_qrels: dict[str, int] = {}
        for idx, m in enumerate(index_metadata):
            if m.get("cwe_id") == query_cwe and idx != self_idx:
                q_qrels[f"d{idx}"] = 1
        q_run: dict[str, float] = {}
        for j, r in enumerate(results):
            q_run[_doc_id(r, j, qid)] = float(r.get("score", 0.0))

        if q_qrels and q_run:
            qrels_dict[qid] = q_qrels
            run_dict[qid] = q_run

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

    sklearn_recall = 0.0
    if qrels_dict and run_dict:
        recall_scores = _compute_metrics(qrels_dict, run_dict, [top_k])
        sklearn_recall = recall_scores.get(f"recall@{top_k}", 0.0)

    return {
        "per_cwe": per_cwe,
        "macro_avg": macro,
        "ranx_recall": sklearn_recall,
        "n_cwes": len(per_cwe),
        "n_singletons": len(n_singletons_seen - set(per_cwe)),
    }


def _query_graph_for(pair):
    """Pick the right graph to embed as a query for this pair."""
    return pair.G_vuln


def _retrieve_for(pair, embedder, retriever, top_k: int) -> list[dict] | None:
    """Embed the appropriate graph for *pair* and retrieve top_k results.

    Returns None if the embedding has near-zero norm.
    """
    query_graph = _query_graph_for(pair)
    query_vec = embedder.embed_one(query_graph)
    if np.linalg.norm(query_vec) < 1e-6:
        return None
    return retriever.query(query_vec, top_k=top_k)


def retrieve_all(
    pairs: list,
    embedder,
    retriever,
    top_k: int,
) -> list[tuple]:
    """Run retrieval for all pairs, returning (pair, results) tuples.

    Pairs whose embedding has near-zero norm are silently skipped.
    """
    out = []
    for pair in pairs:
        results = _retrieve_for(pair, embedder, retriever, top_k)
        if results is not None:
            out.append((pair, results))
    return out


# ── Modular metric functions ─────────────────────────────────────────


def cve_retrieval_metrics(
    query_results: list[tuple],
    ks: list[int],
    index_metadata: list[dict],
) -> dict:
    """Compute CVE-level IR metrics from pre-retrieved (pair, results) tuples.

    Args:
        query_results: List of (pair, results) tuples from retrieval.
        ks: List of k values for hit@k, nDCG@k, MAP@k.
        index_metadata: Full index metadata (list of dicts with at least
            'cve_id') so that recall/nDCG/MAP are computed against the
            complete set of relevant documents, not just retrieved ones.

    Returns hit@k, MRR, nDCG, MAP, per-query details, and n.
    """
    raw_queries = []
    ranx_input = []  # (qid, cve_id, results) for ranx
    n = 0

    for pair, results in query_results:
        qid = f"q{n}"
        ranx_input.append((qid, pair.cve_id, results))

        mrr = next(
            (
                1.0 / (j + 1)
                for j, r in enumerate(results)
                if r["cve_id"] == pair.cve_id
            ),
            0.0,
        )

        query_code = pair.meta.get("source_before")

        raw_queries.append(
            {
                "query_cve": pair.cve_id,
                "query_cwe": pair.cwe_id,
                "query_func": pair.func_name,
                "query_variant": pair.meta.get("variant", ""),
                "query_code": query_code,
                "hit": mrr > 0,
                "mrr": mrr,
                "retrieved": [
                    {
                        "rank": j + 1,
                        "cve_id": r.get("cve_id"),
                        "cwe_id": r.get("cwe_id"),
                        "func_name": r.get("func_name"),
                        "variant": r.get("variant"),
                        "score": r.get("score"),
                        "code": (
                            read_code_file(r.get("source_before"))
                            if r.get("source_before")
                            and os.path.exists(r.get("source_before"))
                            else r.get("source_before")
                        ),
                    }
                    for j, r in enumerate(results)
                ],
            }
        )
        n += 1

    if n == 0:
        return {"n": 0, "raw_queries": []}

    qrels_dict, run_dict = _build_cve_qrels_and_run(ranx_input, index_metadata)
    scores = _compute_metrics(qrels_dict, run_dict, ks)

    max_k = max(ks)
    return {
        **{f"hit@{k}": scores.get(f"hit_rate@{k}", 0.0) for k in ks},
        "mrr": scores.get(f"mrr@{max_k}", 0.0),
        **{f"ndcg@{k}": scores.get(f"ndcg@{k}", 0.0) for k in ks},
        **{f"map@{k}": scores.get(f"map@{k}", 0.0) for k in ks},
        "n": n,
        "raw_queries": raw_queries,
    }


def cwe_recall_metrics(
    query_results: list[tuple],
    index_metadata: list[dict],
    top_k: int,
) -> dict:
    """Compute CWE recall from pre-retrieved (pair, results) tuples.

    Pairs with unknown CWE or no index support are skipped.
    Returns per-CWE breakdown, macro_avg, ranx_recall, and raw_queries.
    """
    support_by_cwe = defaultdict(int)
    for m in index_metadata:
        cwe = m.get("cwe_id")
        if cwe and cwe != "UNKNOWN":
            support_by_cwe[cwe] += 1

    per_query = []
    raw_queries = []
    skipped_no_support = 0
    n = 0

    for pair, results in query_results:
        cwe = pair.cwe_id
        if not cwe or cwe == "UNKNOWN":
            continue
        if support_by_cwe.get(cwe, 0) <= 0:
            skipped_no_support += 1
            continue

        qid = f"q{n}"
        per_query.append((cwe, qid, results, None))

        possible = min(top_k, support_by_cwe.get(cwe, 0))
        same_cwe = sum(1 for r in results if r.get("cwe_id") == cwe)
        raw_queries.append(
            {
                "query_cve": pair.cve_id,
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

    summary = _cwe_recall_summary(per_query, index_metadata, top_k)
    return {
        **summary,
        "n_singletons": 0,
        "n_queries": len(raw_queries),
        "n_skipped_no_support": skipped_no_support,
        "raw_queries": raw_queries,
    }


# ── Convenience wrappers (retrieve + compute in one call) ────────────


def code_query_eval(
    pairs: list,
    retriever,
    embedder,
    ks: list[int],
    index_metadata: list[dict],
) -> dict:
    """Self-retrieval via re-embedding + CVE metrics.

    Convenience wrapper: calls ``retrieve_all`` then ``cve_retrieval_metrics``.
    """
    qr = retrieve_all(pairs, embedder, retriever, max(ks))
    return cve_retrieval_metrics(qr, ks, index_metadata)


def cross_cwe_recall(
    query_pairs: list,
    retriever,
    embedder,
    index_metadata: list[dict],
    top_k: int,
) -> dict:
    """CWE recall where queries and index can come from different splits.

    Convenience wrapper: calls ``retrieve_all`` then ``cwe_recall_metrics``.
    """
    qr = retrieve_all(query_pairs, embedder, retriever, top_k)
    return cwe_recall_metrics(qr, index_metadata, top_k)


# ── Pre-computed embedding evaluation ────────────────────────────────
# These variants accept pre-computed embedding arrays instead of an
# embedder object, useful when embeddings are already cached.


def evaluate_retrieval(
    query_pairs: list,
    query_embeddings: np.ndarray,
    retriever,
    index_pairs: list,
    ks: list[int] = (1, 5, 10),
) -> dict:
    """
    Core retrieval evaluation — hit@k, MRR, nDCG, MAP, CVE P/R/F1.

    Returns a dict with all metrics + raw_queries list.
    """
    raw_queries = []
    query_results = []  # (qid, cve_id, results) for ranx
    n = 0
    max_k = max(ks)

    # CVE support in index
    cve_support = defaultdict(int)
    for p in index_pairs:
        cve_support[p.cve_id] += 1

    # per-query results grouped by CVE for macro-averaging
    per_cve_hits = defaultdict(list)
    per_cve_recalls = defaultdict(list)

    # Build index metadata for full-index qrels
    index_metadata = [
        {"cve_id": p.cve_id, "cwe_id": p.cwe_id, "func_name": p.func_name,
         "variant": p.meta.get("variant", ""), **p.meta}
        for p in index_pairs
    ]

    for pair, qvec in zip(query_pairs, query_embeddings):
        if np.linalg.norm(qvec) < 1e-6:
            continue
        res = retriever.query(qvec, top_k=max_k)

        qid = f"q{n}"
        query_results.append((qid, pair.cve_id, res))

        # Per-query MRR for raw_queries detail
        mrr = next(
            (1.0 / (j + 1) for j, r in enumerate(res) if r["cve_id"] == pair.cve_id),
            0.0,
        )

        # Per-query CVE metrics for macro-averaging
        top_k_res = res[:max_k]
        tp = sum(1 for r in top_k_res if r["cve_id"] == pair.cve_id)
        binary_hit = 1 if tp > 0 else 0
        support = cve_support.get(pair.cve_id, 0)
        recall = tp / support if support > 0 else 0.0

        per_cve_hits[pair.cve_id].append(binary_hit)
        per_cve_recalls[pair.cve_id].append(recall)

        raw_queries.append({
            "query_cve": pair.cve_id,
            "query_cwe": pair.cwe_id,
            "hit": mrr > 0,
            "mrr": mrr,
            "cve_binary_hit": binary_hit,
            "cve_recall": recall,
            "retrieved": [
                {"rank": j + 1, "cve_id": r.get("cve_id"), "cwe_id": r.get("cwe_id"), "score": r.get("score")}
                for j, r in enumerate(res)
            ],
        })
        n += 1

    if n == 0:
        return {"n": 0, "raw_queries": []}

    # Compute aggregate IR metrics via sklearn
    qrels_dict, run_dict = _build_cve_qrels_and_run(query_results, index_metadata)
    scores = _compute_metrics(qrels_dict, run_dict, ks)

    # Macro-averaged CVE precision / recall / F1 (per-class, then averaged)
    class_precisions = []
    class_recalls = []
    class_f1s = []
    for cve_id in per_cve_hits:
        p = float(np.mean(per_cve_hits[cve_id]))
        r = float(np.mean(per_cve_recalls[cve_id]))
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_precisions.append(p)
        class_recalls.append(r)
        class_f1s.append(f1)

    return {
        **{f"hit@{k}": scores.get(f"hit_rate@{k}", 0.0) for k in ks},
        "mrr": scores.get(f"mrr@{max_k}", 0.0),
        **{f"ndcg@{k}": scores.get(f"ndcg@{k}", 0.0) for k in ks},
        **{f"map@{k}": scores.get(f"map@{k}", 0.0) for k in ks},
        **{f"precision@{k}": scores.get(f"precision@{k}", 0.0) for k in ks},
        **{f"recall@{k}": scores.get(f"recall@{k}", 0.0) for k in ks},
        "cve_precision": float(np.mean(class_precisions)) if class_precisions else 0.0,
        "cve_recall": float(np.mean(class_recalls)) if class_recalls else 0.0,
        "cve_f1": float(np.mean(class_f1s)) if class_f1s else 0.0,
        "n": n,
        "n_cve_classes": len(per_cve_hits),
        "raw_queries": raw_queries,
    }


def evaluate_cwe_recall(
    query_pairs: list,
    query_embeddings: np.ndarray,
    retriever,
    index_metadata: list[dict],
    top_k: int = 10,
) -> dict:
    """CWE-level recall evaluation from pre-computed embeddings."""
    cwe_support = defaultdict(int)
    for m in index_metadata:
        cwe = m.get("cwe_id")
        if cwe and cwe != "UNKNOWN":
            cwe_support[cwe] += 1

    per_query = []  # (cwe, qid, results) for _cwe_recall_summary
    n = 0

    for pair, qvec in zip(query_pairs, query_embeddings):
        cwe = pair.cwe_id
        if not cwe or cwe == "UNKNOWN":
            continue
        if cwe_support.get(cwe, 0) <= 0:
            continue
        if np.linalg.norm(qvec) < 1e-6:
            continue
        res = retriever.query(qvec, top_k=top_k)
        per_query.append((cwe, f"q{n}", res, None))
        n += 1

    return _cwe_recall_summary(per_query, index_metadata, top_k)


def evaluate_retrieval_from_records(records: list[dict]) -> dict:
    """Compute retrieval metrics from serialized results.jsonl records.

    Each record must have ``query_cve``, ``query_cwe``, and
    ``retrieval.top_k`` (list of {rank, cve_id, cwe_id, score, ...}).

    Returns a dict compatible with the evaluation dashboard schema:
    matched, top_k, hit_at_1, hit_rate_at_1, hit_cve_at_k, hit_rate_at_k,
    mrr, cwe_hit_at_k, cwe_hit_rate_at_k.
    """
    retrieved = [r for r in records if r.get("retrieval", {}).get("top_k")]
    total = len(retrieved)

    if total == 0:
        return {"matched": 0, "top_k": 5, "hit_at_1": 0, "hit_rate_at_1": 0,
                "hit_cve_at_k": 0, "hit_rate_at_k": 0, "mrr": 0,
                "cwe_hit_at_k": 0, "cwe_hit_rate_at_k": 0}

    # Determine k from first record
    sample_topk = retrieved[0].get("retrieval", {}).get("top_k", [])
    k = len(sample_topk) if sample_topk else 5

    # Build query_results for ranx
    query_results = []
    for i, r in enumerate(retrieved):
        qid = f"q{i}"
        query_cve = r.get("query_cve", "")
        # Convert top_k entries to the format expected by _build_cve_qrels_and_run
        results = []
        for entry in r.get("retrieval", {}).get("top_k", []):
            results.append({
                "cve_id": entry.get("cve_id"),
                "cwe_id": entry.get("cwe_id"),
                "score": entry.get("score", 0.0),
            })
        query_results.append((qid, query_cve, results))

    # Use existing infrastructure for CVE hit metrics
    qrels_dict, run_dict = _build_cve_qrels_and_run(query_results)
    scores = _compute_metrics(qrels_dict, run_dict, [1, k])

    # CWE recall@k (any result in top-k shares CWE)
    cwe_hit_at_k = 0
    for r in retrieved:
        query_cwe = r.get("query_cwe", "")
        top_k_list = r.get("retrieval", {}).get("top_k", [])
        if any(entry.get("cwe_id") == query_cwe for entry in top_k_list):
            cwe_hit_at_k += 1

    return {
        "matched": total,
        "top_k": k,
        "hit_at_1": int(round(scores.get("hit_rate@1", 0.0) * total)),
        "hit_rate_at_1": round(scores.get("hit_rate@1", 0.0), 4),
        "hit_cve_at_k": int(round(scores.get(f"hit_rate@{k}", 0.0) * total)),
        "hit_rate_at_k": round(scores.get(f"hit_rate@{k}", 0.0), 4),
        "mrr": round(scores.get(f"mrr@{k}", 0.0), 4),
        "cwe_hit_at_k": cwe_hit_at_k,
        "cwe_hit_rate_at_k": round(cwe_hit_at_k / total, 4),
    }
