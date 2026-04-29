"""
Retrieval evaluation metrics — code-query hit@k / MRR and cross-CWE recall.

These operate on (pairs, embedder, retriever) and are independent of
the experiment runner's orchestration logic.
"""

import os
from collections import defaultdict
from pathlib import Path

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


def _query_graph_for(pair):
    """Pick the right graph to embed as a query for this pair."""
    if (
        pair.meta.get("dataset") == "autopatch"
        and pair.meta.get("variant") != "original"
    ):
        return pair.G_before
    return pair.G_vuln


def code_query_eval(
    pairs: list,
    retriever,
    embedder,
    ks: list[int],
) -> dict:
    """
    Self-retrieval via re-embedding: for each query pair, embed its graph
    with ``embedder.embed_one`` and retrieve from the index.

    Returns hit@k for each k, MRR, nDCG, MAP, and per-query details.
    """
    raw_queries = []
    query_results = []  # (qid, cve_id, results) for ranx
    n = 0

    for pair in pairs:
        query_graph = _query_graph_for(pair)
        query_vec = embedder.embed_one(query_graph)
        if np.linalg.norm(query_vec) < 1e-6:
            continue
        results = retriever.query(query_vec, top_k=max(ks))

        qid = f"q{n}"
        query_results.append((qid, pair.cve_id, results))

        # Per-query MRR for raw_queries detail
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

    # Compute aggregate metrics via ranx
    qrels, run = _build_cve_qrels_and_run(query_results)
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


def cross_cwe_recall(
    query_pairs: list,
    retriever,
    embedder,
    index_metadata: list[dict],
    top_k: int,
) -> dict:
    """
    CWE recall where queries and index can come from different splits.

    For each query pair with a known CWE, retrieve top_k results and
    measure how many share the same CWE type.

    Aggregate recall is computed via ranx; per-CWE breakdown is kept for
    diagnostic purposes.
    """
    support_by_cwe = defaultdict(int)
    for m in index_metadata:
        cwe = m.get("cwe_id")
        if cwe and cwe != "UNKNOWN":
            support_by_cwe[cwe] += 1

    # Build ranx structures for CWE-level recall
    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}
    per_cwe_scores = defaultdict(list)
    raw_queries = []
    skipped_no_support = 0
    n = 0

    for pair in query_pairs:
        cwe = pair.cwe_id
        if not cwe or cwe == "UNKNOWN":
            continue
        possible = min(top_k, support_by_cwe.get(cwe, 0))
        if possible <= 0:
            skipped_no_support += 1
            continue

        query_graph = _query_graph_for(pair)
        query_vec = embedder.embed_one(query_graph)
        if np.linalg.norm(query_vec) < 1e-6:
            continue

        results = retriever.query(query_vec, top_k=top_k)

        # Build per-query qrels: all index docs with same CWE are relevant
        qid = f"q{n}"
        q_qrels: dict[str, int] = {}
        for idx, m in enumerate(index_metadata):
            if m.get("cwe_id") == cwe:
                q_qrels[f"d{idx}"] = 1
        q_run: dict[str, float] = {}
        for j, r in enumerate(results):
            q_run[_doc_id(r, j, qid)] = float(r.get("score", 0.0))

        if q_qrels and q_run:
            qrels_dict[qid] = q_qrels
            run_dict[qid] = q_run

        same_cwe = sum(1 for r in results if r.get("cwe_id") == cwe)
        recall = same_cwe / possible
        per_cwe_scores[cwe].append(recall)
        raw_queries.append(
            {
                "query_cve": pair.cve_id,
                "query_cwe": cwe,
                "recall": recall,
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

    per_cwe = {
        cwe: {
            "recall": float(np.mean(vals)),
            "support": int(support_by_cwe.get(cwe, 0)),
        }
        for cwe, vals in per_cwe_scores.items()
    }

    # Compute aggregate recall via ranx
    ranx_recall = 0.0
    if qrels_dict and run_dict:
        qrels = Qrels(qrels_dict)
        run = Run(run_dict)
        ranx_recall = float(ranx_evaluate(qrels, run, f"recall@{top_k}"))

    macro = float(np.mean([v["recall"] for v in per_cwe.values()])) if per_cwe else 0.0
    return {
        "per_cwe": per_cwe,
        "macro_avg": macro,
        "ranx_recall": ranx_recall,
        "n_cwes": len(per_cwe),
        "n_singletons": 0,
        "n_queries": len(raw_queries),
        "n_skipped_no_support": skipped_no_support,
        "raw_queries": raw_queries,
    }


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

    # Build ranx structures
    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}
    per_cwe_scores = defaultdict(list)
    n = 0

    for pair, qvec in zip(query_pairs, query_embeddings):
        cwe = pair.cwe_id
        if not cwe or cwe == "UNKNOWN":
            continue
        possible = min(top_k, cwe_support.get(cwe, 0))
        if possible <= 0:
            continue
        if np.linalg.norm(qvec) < 1e-6:
            continue
        res = retriever.query(qvec, top_k=top_k)

        # Build qrels: all index docs with same CWE are relevant
        qid = f"q{n}"
        q_qrels: dict[str, int] = {}
        for idx, m in enumerate(index_metadata):
            if m.get("cwe_id") == cwe:
                q_qrels[f"d{idx}"] = 1
        q_run: dict[str, float] = {}
        for j, r in enumerate(res):
            q_run[_doc_id(r, j, qid)] = float(r.get("score", 0.0))

        if q_qrels and q_run:
            qrels_dict[qid] = q_qrels
            run_dict[qid] = q_run

        same = sum(1 for r in res if r.get("cwe_id") == cwe)
        per_cwe_scores[cwe].append(same / possible)
        n += 1

    per_cwe = {
        cwe: {"recall": float(np.mean(vals)), "support": int(cwe_support.get(cwe, 0))}
        for cwe, vals in per_cwe_scores.items()
    }
    macro = float(np.mean([v["recall"] for v in per_cwe.values()])) if per_cwe else 0.0

    # Compute aggregate recall via ranx
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
    }
