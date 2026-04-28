"""
Retrieval evaluation metrics — code-query hit@k / MRR and cross-CWE recall.

These operate on (pairs, embedder, retriever) and are independent of
the experiment runner's orchestration logic.
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def _query_graph_for(pair):
    """Pick the right graph to embed as a query for this pair."""
    if (
        pair.meta.get("dataset") == "autopatch"
        and pair.meta.get("variant") != "original"
    ):
        return pair.G_before
    return pair.G_vuln


def _read_code_file(path: str | None, max_chars: int = 4000) -> str:
    """Read source code from a file path or inline string, truncating if needed."""
    if not path:
        return ""
    p = Path(path)
    if p.exists():
        try:
            text = p.read_text(errors="replace")
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
            return text
        except Exception:
            return ""
    # path may be inline code (augmented variants store the code directly in source_before)
    if len(path) > 20:
        return path[:max_chars] + ("\n... [truncated]" if len(path) > max_chars else "")
    return ""


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
                            _read_code_file(r.get("source_before"))
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
