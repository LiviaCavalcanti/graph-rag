"""Agent experiment — batch retrieval over the CVE dataset.

Extracted from main.py run_batch_query(). Performs embedding-based retrieval
only (no LLM calls), following the same patterns as batch_inference
(BackgroundWriter, resumability, batch flushing).

Usage (via experiment coordinator):
    python -m experiments.experiment agent --agent-mode retriever --model M --split

Direct usage:
    from experiments.agent_experiment import run_experiment
    run_experiment(cfg, mode="retriever", model_name="gemini-2.0-flash")
"""

from __future__ import annotations

import sys
from pathlib import Path

from experiments.common import build_split
from src.agents.utils import get_ground_truth_patch
from src.data.autopatch import load_pairs
from src.evaluate.patch_verification import _build_index_and_retriever
from src.io import make_run_dir, run_batched
from src.metrics.retrieval_eval import _retrieve_for


def run_experiment(
    cfg: dict,
    *,
    mode: str = "retriever",
    model_name: str | None = None,
    max_queries: int | None = None,
    batch_size: int = 10,
    resume: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Batch query: embed each query pair, retrieve top-k from FAISS, write results.

    Parameters
    ----------
    cfg : dict
        Full config (split overrides should already be applied by the caller).
    mode : str
        Retrieval mode label (stored in metadata).
    model_name : str | None
        Model name label (stored in metadata, not used for retrieval).
    max_queries : int | None
        Limit the number of query pairs processed.
    batch_size : int
        Number of queries per batch flush.
    resume : str | None
        Path to an existing run directory to resume.
    output_dir : Path | None
        If provided, write results into this directory instead of creating a new one.
        The results file will be named ``retrieval_results.jsonl``.

    Returns
    -------
    Path
        The run directory containing results.jsonl and run_meta.json.
    """
    # ── load pairs and split ─────────────────────────────────────────
    pairs = load_pairs(cfg)
    index_pairs, query_pairs, split_info = build_split(pairs, cfg)

    if max_queries:
        query_pairs = query_pairs[:max_queries]

    # ── build embedder + reuse existing FAISS index ──────────────────
    rag_cfg = cfg["rag"]
    top_k = rag_cfg.get("top_k", 5)
    embedder, retriever = _build_index_and_retriever(index_pairs, cfg, top_k)

    # ── per-query callback ───────────────────────────────────────────
    def process_one(qp, current: int, total: int) -> dict:
        cve_id = qp.cve_id
        cwe_id = qp.cwe_id
        variant = qp.meta.get("variant", "")

        base = {
            "query_cve": cve_id,
            "query_cwe": cwe_id,
            "query_variant": variant,
            "query_dir": qp.meta.get("dir_name", ""),
        }

        try:
            results = _retrieve_for(qp, embedder, retriever, top_k)
            if results is None:
                print(f"  [{current}/{total}] {cve_id}/{variant}  zero-norm embedding")
                return {**base, "status": "zero_norm"}
        except Exception as e:
            print(f"  [{current}/{total}] {cve_id}/{variant}  ERROR: {e}")
            return {**base, "status": "error", "error": str(e)}

        if not results:
            print(f"  [{current}/{total}] {cve_id}/{variant}  no results")
            return {**base, "status": "no_results"}

        top = results[0]
        top_cve = top.get("cve_id", "?")
        cve_match = top_cve == cve_id
        cwe_match = top.get("cwe_id") == cwe_id
        ground_truth = get_ground_truth_patch(qp)

        print(
            f"  [{current}/{total}] {cve_id}/{variant}  "
            f"→ {top_cve}/{top.get('variant','?')}  "
            f"score={top.get('score',0):.4f}  "
            f"cve_match={cve_match}"
        )

        return {
            **base,
            "example_cve": top_cve,
            "example_cwe": top.get("cwe_id"),
            "example_variant": top.get("variant", ""),
            "example_dir": top.get("dir_name", ""),
            "status": "retrieved",
            "cve_match": cve_match,
            "cwe_match": cwe_match,
            "retrieval": {
                "cve_match": cve_match,
                "cwe_match": cwe_match,
                "retrieved_variant": top.get("variant", ""),
                "score": round(top.get("score", 0.0), 6),
                "top_k": [
                    {
                        "rank": j + 1,
                        "cve_id": r.get("cve_id"),
                        "cwe_id": r.get("cwe_id"),
                        "variant": r.get("variant"),
                        "func_name": r.get("func_name"),
                        "score": round(r.get("score", 0), 6),
                    }
                    for j, r in enumerate(results)
                ],
            },
            "ground_truth_patch": ground_truth[:500],
        }

    # ── run batched ──────────────────────────────────────────────────
    results_filename = "retrieval_results.jsonl" if output_dir else "results.jsonl"
    meta_filename = "retrieval_meta.json" if output_dir else "run_meta.json"

    return run_batched(
        query_pairs,
        process_one=process_one,
        run_tag="batch_query",
        batch_size=batch_size,
        resume=resume,
        output_dir=output_dir,
        results_filename=results_filename,
        meta_filename=meta_filename,
        meta={
            "mode": mode,
            "model_name": model_name,
            "top_k": top_k,
            "split_info": split_info,
        },
    )


# ── LLM patching pipeline ────────────────────────────────────────────


def run_patching_experiment(
    cfg: dict,
    *,
    retriever_mode: str = "oracle",
    model_name: str | None = None,
    query_run: str | None = None,
    max_queries: int | None = None,
    batch_size: int = 10,
    resume: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run the LLM patching experiment end-to-end.

    Orchestrates: load pairs → split → build retriever → load db_cache →
    call batch_inference.

    Parameters
    ----------
    cfg : dict
        Full config (split overrides should already be applied by caller).
    retriever_mode : str
        "oracle" for perfect same-CVE retrieval, "precomputed" for results
        from a prior retrieval run (requires *query_run*).
    model_name : str | None
        LLM model/deployment name (default from .env MODEL_NAME).
    query_run : str | None
        Path to a retrieval run directory containing results.jsonl.
        Required when retriever_mode == "precomputed".
    max_queries : int | None
        Limit query pairs for testing.
    batch_size : int
        Queries per disk flush.
    resume : str | None
        Path to a previous run directory to resume.

    Returns
    -------
    Path
        The run directory containing results.jsonl and run_meta.json.
    """
    import os

    from dotenv import load_dotenv

    from src.agents.batch_inference import run_batch_inference
    from src.data.autopatch import AutoPatchDataset, load_pairs_lightweight
    from src.rag.oracle import OracleRetriever
    from src.rag.precomputed import PrecomputedRetriever

    load_dotenv()

    if not os.getenv("AZURE_API_KEY") or not os.getenv("AZURE_API_BASEURL"):
        print("ERROR: Set AZURE_API_KEY and AZURE_API_BASEURL in .env")
        sys.exit(1)

    # ── load pairs and split ─────────────────────────────────────────
    pairs = load_pairs_lightweight(cfg)
    print(f"Loaded {len(pairs)} lightweight pairs (no CPGs)")
    index_pairs, query_pairs, split_info = build_split(pairs, cfg)

    if max_queries:
        query_pairs = query_pairs[:max_queries]

    # ── build retriever ──────────────────────────────────────────────
    if retriever_mode == "oracle":
        retriever = OracleRetriever(index_pairs)
        print(f"Oracle retriever built from {len(index_pairs)} index pairs")
    elif retriever_mode == "precomputed":
        if not query_run:
            print("ERROR: --query-run <run_dir> required for precomputed mode")
            print("Run retrieval first, then pass its output dir.")
            sys.exit(1)

        # Support both unified (retrieval_results.jsonl) and legacy (results.jsonl)
        query_results = Path(query_run) / "retrieval_results.jsonl"
        if not query_results.exists():
            query_results = Path(query_run) / "results.jsonl"
        if not query_results.exists():
            print(f"ERROR: {query_results} not found")
            sys.exit(1)

        retriever = PrecomputedRetriever(query_results)
    else:
        print(f"ERROR: unknown retriever_mode: {retriever_mode!r}")
        sys.exit(1)

    # ── load db_cache ────────────────────────────────────────────────
    cve_root = Path(cfg["data"]["autopatch"]["root"])
    db_cache = AutoPatchDataset.load_db_cache(cve_root)
    print(f"Cached {len(db_cache)} db_entries")

    # ── run batch inference ──────────────────────────────────────────
    return run_batch_inference(
        query_pairs=query_pairs,
        retriever=retriever,
        db_cache=db_cache,
        model_name=model_name,
        batch_size=batch_size,
        run_tag=f"batch_{retriever_mode}",
        resume_dir=resume,
        meta_extra={"mode": retriever_mode, "split_info": split_info},
        output_dir=output_dir,
    )
