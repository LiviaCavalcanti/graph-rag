"""
Retrieval evaluation metrics — code-query hit@k / MRR and cross-CWE recall.

These operate on (pairs, embedder, retriever) and are independent of
the experiment runner's orchestration logic.
"""

import os
from collections import defaultdict
import numpy as np
from ranx import Qrels, Run, evaluate as ranx_evaluate

from src.io import read_code_file


# ── ranx helpers ─────────────────────────────────────────────────────


def _doc_id(result: dict, rank: int, qid: str) -> str:
    """Return a stable global document id from a retriever result.

    The ``Retriever`` always injects ``_idx`` (the FAISS vector position)
    into every result dict, so each document has a canonical id.
    Falls back to a per-query rank-based id for non-FAISS retrievers.
    """
    idx = result.get("_idx")
    if idx is not None:
        return f"d{idx}"
    return f"d_rank{rank}_{qid}"


def _build_cve_qrels_and_run(
    query_results: list[tuple[str, str, list[dict]]],
    index_metadata: list[dict] | None = None,
) -> tuple[Qrels, Run]:
    """Build ranx Qrels/Run for CVE-level retrieval.

    *query_results* is a list of ``(query_id, query_cve, results)`` triples
    where *results* are the ranked dicts returned by the retriever.

    If *index_metadata* is provided, **all** index entries sharing the
    query CVE are marked relevant (needed for recall).  Otherwise only
    retrieved entries are judged.
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

    return Qrels(qrels_dict), Run(run_dict)


def _ranx_metrics(qrels: Qrels, run: Run, ks: list[int]) -> dict:
    """Compute standard IR metrics via ranx."""
    metric_strings = []
    for k in ks:
        metric_strings.extend([
            f"hit_rate@{k}",
            f"precision@{k}",
            f"recall@{k}",
            f"ndcg@{k}",
            f"map@{k}",
        ])
    metric_strings.append(f"mrr@{max(ks)}")

    scores = ranx_evaluate(qrels, run, metric_strings)
    return {str(k): float(v) for k, v in scores.items()}


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

    ranx_recall = 0.0
    if qrels_dict and run_dict:
        qrels = Qrels(qrels_dict)
        run = Run(run_dict)
        ranx_recall = float(ranx_evaluate(qrels, run, f"recall@{top_k}"))

    return {
        "per_cwe": per_cwe,
        "macro_avg": macro,
        "ranx_recall": ranx_recall,
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
) -> dict:
    """Compute CVE-level IR metrics from pre-retrieved (pair, results) tuples.

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

    qrels, run = _build_cve_qrels_and_run(ranx_input)
    ranx_scores = _ranx_metrics(qrels, run, ks)

    max_k = max(ks)
    return {
        **{f"hit@{k}": ranx_scores.get(f"hit_rate@{k}", 0.0) for k in ks},
        "mrr": ranx_scores.get(f"mrr@{max_k}", 0.0),
        **{f"ndcg@{k}": ranx_scores.get(f"ndcg@{k}", 0.0) for k in ks},
        **{f"map@{k}": ranx_scores.get(f"map@{k}", 0.0) for k in ks},
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
) -> dict:
    """Self-retrieval via re-embedding + CVE metrics.

    Convenience wrapper: calls ``retrieve_all`` then ``cve_retrieval_metrics``.
    """
    qr = retrieve_all(pairs, embedder, retriever, max(ks))
    return cve_retrieval_metrics(qr, ks)


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

    # Compute aggregate IR metrics via ranx
    qrels, run = _build_cve_qrels_and_run(query_results, index_metadata)
    ranx_scores = _ranx_metrics(qrels, run, ks)

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
        **{f"hit@{k}": ranx_scores.get(f"hit_rate@{k}", 0.0) for k in ks},
        "mrr": ranx_scores.get(f"mrr@{max_k}", 0.0),
        **{f"ndcg@{k}": ranx_scores.get(f"ndcg@{k}", 0.0) for k in ks},
        **{f"map@{k}": ranx_scores.get(f"map@{k}", 0.0) for k in ks},
        **{f"precision@{k}": ranx_scores.get(f"precision@{k}", 0.0) for k in ks},
        **{f"recall@{k}": ranx_scores.get(f"recall@{k}", 0.0) for k in ks},
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

    # Use existing ranx infrastructure for CVE hit metrics
    qrels, run = _build_cve_qrels_and_run(query_results)
    ranx_scores = _ranx_metrics(qrels, run, [1, k])

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
        "hit_at_1": int(round(ranx_scores.get("hit_rate@1", 0.0) * total)),
        "hit_rate_at_1": round(ranx_scores.get("hit_rate@1", 0.0), 4),
        "hit_cve_at_k": int(round(ranx_scores.get(f"hit_rate@{k}", 0.0) * total)),
        "hit_rate_at_k": round(ranx_scores.get(f"hit_rate@{k}", 0.0), 4),
        "mrr": round(ranx_scores.get(f"mrr@{k}", 0.0), 4),
        "cwe_hit_at_k": cwe_hit_at_k,
        "cwe_hit_rate_at_k": round(cwe_hit_at_k / total, 4),
    }
