"""Query/retrieval CLI commands: single-CVE lookup and batch retrieval.

Extracted from ``main.run_query`` / ``main.run_batch_query`` and helpers —
the CLI's ``--mode query`` now just calls into this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.embeddings import build_embedders
from src.io.read_write import make_run_dir, save_json
from src.rag.faiss_index import FAISSIndex
from src.schema_config import RetrievalResult


def run_query(cfg: dict, cve_id: str):
    """Direct metadata lookup for a single CVE against the built global index."""
    from src.rag.retriever import Retriever

    rag_cfg = cfg["rag"]
    index = FAISSIndex(
        dim=cfg["embeddings"]["dim"],
        index_path=rag_cfg["index_path"],
        metadata_path=rag_cfg["metadata_path"],
    )
    index.load()
    retriever = Retriever(index, top_k=rag_cfg["top_k"])
    raw_results = retriever.query_by_cve(cve_id)
    for r in raw_results:
        print(r)

    retrieval_contract = RetrievalResult(
        run_id="query",
        query_id=cve_id,
        query_cve=cve_id,
        retriever_name="metadata_lookup",
        top_k=len(raw_results),
        hit_ids=[str(r.get("_idx", i)) for i, r in enumerate(raw_results)],
        hit_scores=[float(r.get("score", 1.0)) for r in raw_results],
        hit_metadata=raw_results,
        metadata={"result_count": len(raw_results)},
    )

    query_contract_path = Path(rag_cfg["metadata_path"]).with_name(
        f"query_{cve_id}_retrieval_contract.json"
    )
    query_contract_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(retrieval_contract.__dict__, query_contract_path)
    print(f"Retrieval contract: {query_contract_path}")


def _precomputed_row(query_meta: dict, example_meta: dict | None, score: float) -> dict:
    """Build one ``PrecomputedRetriever``-compatible JSONL row.

    ``query_meta``/``example_meta`` follow the FAISS/HNSW metadata shape
    (cve_id, cwe_id, variant, dir_name, ...) so this works whether the hit
    came from a live query or a reconstructed persisted index.
    """
    query_row = {
        "query_cve": query_meta.get("cve_id"),
        "query_cwe": query_meta.get("cwe_id"),
        "query_variant": query_meta.get("variant", ""),
        # dir_name disambiguates multiple functions from the same CVE that share
        # the same (cve_id, variant) — without it PrecomputedRetriever silently
        # overwrites earlier functions with the last one in the JSONL.
        "query_dir": query_meta.get("dir_name", ""),
    }
    if example_meta is None:
        return {**query_row, "status": "no_match"}
    return {
        **query_row,
        "status": "success",
        "example_cve": example_meta.get("cve_id"),
        "example_cwe": example_meta.get("cwe_id"),
        "example_variant": example_meta.get("variant", ""),
        "example_dir": example_meta.get("dir_name", ""),
        "retrieval": {
            "cve_match": example_meta.get("cve_id") == query_meta.get("cve_id"),
            "cwe_match": example_meta.get("cwe_id") == query_meta.get("cwe_id"),
            "score": float(score),
        },
    }


def _write_query_results(
    rows: list[dict], run_tag: str, output_dir: Path | None = None
) -> Path:
    """Write query-retrieval rows as results.jsonl in a new run dir.

    The output is consumable by ``--mode batch --query-run <run_dir>``
    (src.rag.precomputed.PrecomputedRetriever). ``output_dir`` overrides the
    default ``experiments/output/`` base (e.g. to consolidate a multi-variant
    sweep under one parent folder).
    """
    _run_id, run_dir = make_run_dir(run_tag, output_dir=output_dir)
    out_path = run_dir / "results.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"Wrote {len(rows)} query results → {out_path}")
    return run_dir


def _leave_one_out_from_index(
    index_dir: Path, embedding_variant: str, max_queries: int | None
) -> list[dict]:
    """Reuse a PERSISTED retrieval-experiment index with NO re-embedding and
    NO CPGs needed: every indexed item's own vector is reconstructed straight
    from the FAISS/HNSW index (``index.reconstruct_n``) and queried against
    the index excluding itself; the best OTHER match becomes "the retrieved
    example" for that item (leave-one-out).

    Looks for ``<index_dir>/<embedding_variant>__hnsw.index`` + ``_meta.json``
    — the naming convention written by RetrievalGridExperiment /
    EvaluateIndexExperiment (e.g. a prior ``--mode experiment`` run's
    ``indices/`` directory).
    """
    import faiss

    index_path = index_dir / f"{embedding_variant}__hnsw.index"
    meta_path = index_dir / f"{embedding_variant}__hnsw_meta.json"
    if not index_path.exists() or not meta_path.exists():
        available = sorted(
            p.name[: -len("__hnsw.index")] for p in index_dir.glob("*__hnsw.index")
        )
        raise FileNotFoundError(
            f"No persisted index for embedder {embedding_variant!r} in {index_dir}. "
            f"Available: {available or 'none found'}"
        )

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text())
    n = index.ntotal
    if n != len(metadata):
        print(
            f"WARNING: index has {n} vectors but metadata has {len(metadata)} entries"
        )

    vectors = index.reconstruct_n(0, n)
    n_queries = n if not max_queries else min(max_queries, n)
    search_k = min(n, 10)

    rows = []
    for i in range(n_queries):
        distances, indices = index.search(vectors[i : i + 1], search_k)
        best = None
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx == i:
                continue
            best = (idx, dist)
            break
        query_meta = metadata[i]
        example_meta = metadata[best[0]] if best else None
        score = float(best[1]) if best else 0.0
        rows.append(_precomputed_row(query_meta, example_meta, score))
    return rows


def run_batch_query(cfg: dict, args):
    """Batch retrieval: produce a per-query results.jsonl reusable by
    ``--mode batch --query-run <this run dir>``.

    Two modes:
      --index-dir <dir>: reuse a PERSISTED retrieval-experiment index (fast,
        offline, no CPGs) via leave-one-out reconstruction. Use this to reuse
        e.g. a prior ``--mode experiment`` run's ``indices/`` directory
        (looks for ``<embedding-variant>__hnsw.index`` + ``_meta.json``).
      (default): live retrieval — embeds all pairs from cfg["data"]["active"]
        (needs CPGs) against the global index built by ``--mode index``.
    """
    output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
    if args.index_dir:
        rows = _leave_one_out_from_index(
            Path(args.index_dir),
            args.embedding_variant or cfg["rag"]["embedding_variant"],
            args.max_queries,
        )
        return _write_query_results(rows, "query_indexed", output_dir=output_dir)

    from src.data import load_pairs
    from src.metrics.retrieval_eval import retrieve_all
    from src.rag.retriever import Retriever

    pairs = load_pairs(cfg)

    # Apply train/test split when configured so we only query the held-out
    # set (pairs NOT in the index).  Without this every query would find
    # itself in the index — a guaranteed perfect hit that measures nothing.
    split_cfg = cfg.get("experiment", {}).get("split", {})
    if split_cfg.get("enabled") or split_cfg.get("precomputed_split_dir"):
        from src.data.split import build_split

        _, pairs, split_info = build_split(pairs, cfg)
        print(
            f"  [split] Querying {len(pairs)} held-out pairs "
            f"({split_info.get('mode', 'split')} mode — index pairs excluded to prevent leakage)"
        )

    if args.max_queries:
        pairs = pairs[: args.max_queries]

    rag_cfg = cfg["rag"]
    variant = args.embedding_variant or rag_cfg["embedding_variant"]
    embedders = build_embedders(cfg)
    embedder = next((e for e in embedders if e.name == variant), None)
    if embedder is None:
        raise ValueError(
            f"Embedding variant {variant!r} not found. "
            f"Available: {[e.name for e in embedders]}"
        )

    index = FAISSIndex(
        dim=cfg["embeddings"]["dim"],
        index_path=rag_cfg["index_path"],
        metadata_path=rag_cfg["metadata_path"],
    )
    index.load()

    # Load the PCA state that was fitted (and persisted) at index build time.
    # Without this, PCA-based embedders re-fit on the query corpus — a
    # different basis from the index — making cosine similarity meaningless.
    if embedder.requires_fitting:
        state = index.load_embedder_state(variant)
        if state is not None:
            embedder.load_pca_state(state)
        else:
            raise RuntimeError(
                f"Embedder '{variant}' requires a pre-fitted PCA but no state "
                f"file was found at '{index.embedder_state_path(variant)}'. "
                f"Rebuild the FAISS index with '--mode index' to generate it."
            )

    retriever = Retriever(index, top_k=rag_cfg["top_k"])
    query_results = retrieve_all(pairs, embedder, retriever, top_k=rag_cfg["top_k"])

    rows = [
        _precomputed_row(
            {
                "cve_id": pair.cve_id,
                "cwe_id": pair.cwe_id,
                "variant": pair.meta.get("variant", ""),
                "dir_name": pair.meta.get("dir_name", ""),
            },
            hits[0] if hits else None,
            hits[0].get("score", 0.0) if hits else 0.0,
        )
        for pair, hits in query_results
    ]
    return _write_query_results(rows, "query", output_dir=output_dir)
