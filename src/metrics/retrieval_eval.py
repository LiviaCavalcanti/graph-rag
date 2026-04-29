"""
Retrieval evaluation metrics — code-query hit@k / MRR and cross-CWE recall.

These operate on (pairs, embedder, retriever) and are independent of
the experiment runner's orchestration logic.
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.io import read_code_file


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

    Returns hit@k for each k, MRR, and per-query details.
    """
    hits = defaultdict(int)
    mrrs = []
    raw_queries = []
    n = 0

    for pair in pairs:
        query_graph = _query_graph_for(pair)
        query_vec = embedder.embed_one(query_graph)
        if np.linalg.norm(query_vec) < 1e-6:
            continue
        results = retriever.query(query_vec, top_k=max(ks))
        for k in ks:
            hits[k] += int(any(r["cve_id"] == pair.cve_id for r in results[:k]))
        mrr = next(
            (
                1.0 / (j + 1)
                for j, r in enumerate(results)
                if r["cve_id"] == pair.cve_id
            ),
            0.0,
        )
        mrrs.append(mrr)

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

    return {
        **{f"hit@{k}": hits[k] / n for k in ks},
        "mrr": float(np.mean(mrrs)),
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
    """
    support_by_cwe = defaultdict(int)
    for m in index_metadata:
        cwe = m.get("cwe_id")
        if cwe and cwe != "UNKNOWN":
            support_by_cwe[cwe] += 1

    per_cwe_scores = defaultdict(list)
    raw_queries = []
    skipped_no_support = 0

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

    per_cwe = {
        cwe: {
            "recall": float(np.mean(vals)),
            "support": int(support_by_cwe.get(cwe, 0)),
        }
        for cwe, vals in per_cwe_scores.items()
    }
    macro = float(np.mean([v["recall"] for v in per_cwe.values()])) if per_cwe else 0.0
    return {
        "per_cwe": per_cwe,
        "macro_avg": macro,
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
    Core retrieval evaluation — hit@k, MRR, CVE P/R/F1, CWE recall.

    Returns a dict with all metrics + raw_queries list.
    """
    hits = defaultdict(int)
    mrrs = []
    raw_queries = []
    n = 0
    max_k = max(ks)

    # CVE support in index
    cve_support = defaultdict(int)
    for p in index_pairs:
        cve_support[p.cve_id] += 1

    # per-query results grouped by CVE for macro-averaging
    per_cve_hits = defaultdict(list)
    per_cve_recalls = defaultdict(list)

    for pair, qvec in zip(query_pairs, query_embeddings):
        if np.linalg.norm(qvec) < 1e-6:
            continue
        res = retriever.query(qvec, top_k=max_k)

        # hit@k
        for k in ks:
            hits[k] += int(any(r["cve_id"] == pair.cve_id for r in res[:k]))

        # MRR
        mrr = next(
            (1.0 / (j + 1) for j, r in enumerate(res) if r["cve_id"] == pair.cve_id),
            0.0,
        )
        mrrs.append(mrr)

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
        **{f"hit@{k}": hits[k] / n for k in ks},
        "mrr": float(np.mean(mrrs)),
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

    per_cwe_scores = defaultdict(list)
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
        same = sum(1 for r in res if r.get("cwe_id") == cwe)
        per_cwe_scores[cwe].append(same / possible)

    per_cwe = {
        cwe: {"recall": float(np.mean(vals)), "support": int(cwe_support.get(cwe, 0))}
        for cwe, vals in per_cwe_scores.items()
    }
    macro = float(np.mean([v["recall"] for v in per_cwe.values()])) if per_cwe else 0.0
    return {
        "per_cwe": per_cwe,
        "macro_avg": macro,
        "n_cwes": len(per_cwe),
    }
