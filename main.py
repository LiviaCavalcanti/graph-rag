"""
Unified CVE graph-RAG pipeline supporting both AutoPatch and CVEfixes datasets.

PRIMARY WORKFLOW (CVEfixes — Recommended):
────────────────────────────────────────
  1. Export CPGs:   python main.py --mode export [--dataset cvefixes]
  2. Build index:   python main.py --mode index
  3. Query:         python main.py --mode query --cve CVE-2024-XXXXX
  4. Full pipeline: python main.py --mode full

CONFIGURATION:
──────────────
- config.yaml: Dataset selection (data.active), embedders, Joern paths
  • data.active: [cvefixes]  ← CVEfixes is now the default
  • data.cvefixes.db_path: data/cvefixes/CVEfixes.db (download from Zenodo)
  • data.cvefixes.graphml_root: graphml_cvefixes_fixed (Joern outputs)

CVEFIXES SETUP:
───────────────
  1. Download: https://zenodo.org/record/4476563 (DOI: 10.5281/zenodo.4476563)
  2. Inflate:  gzcat CVEfixes.sql.gz | sqlite3 data/cvefixes/CVEfixes.db
  3. Export:   python main.py --mode export
     → Generates CPGs for C/C++ code in graphml_cvefixes_fixed/

MODES:
──────
  export:       Run Joern to generate CPGs (before/after code)
  index:        Embed CPGs and build FAISS retrieval index
  query:        Retrieve similar vulnerabilities from index
  batch:        End-to-end inference with LLM patching
  full:         Complete pipeline: retrieval → patching → evaluation
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.data import entries_cache
from src.data.export import run_export
from src.embeddings import build_embedders
from src.rag.faiss_index import FAISSIndex
from src.rag.indexing import build_index
from src.rag.query import run_batch_query, run_query
from src.schema_config import AppConfig, RetrievalResult


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON with safe fallback for non-serializable objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def apply_split_overrides(cfg: dict, args) -> None:
    """Apply CLI split/augmentation overrides to the experiment config."""
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("split", {})
    split_cfg = cfg["experiment"]["split"]
    if args.split:
        split_cfg["enabled"] = True
    if args.no_split:
        split_cfg["enabled"] = False
    if args.split_test_ratio is not None:
        split_cfg["test_ratio"] = args.split_test_ratio
    if args.aug_train_ratio is not None:
        split_cfg["augmented_train_ratio"] = args.aug_train_ratio


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
    _write_json(query_contract_path, retrieval_contract.__dict__)
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


def _write_query_results(rows: list[dict], run_tag: str) -> Path:
    """Write query-retrieval rows as results.jsonl in a new run dir.

    The output is consumable by ``--mode batch --query-run <run_dir>``
    (src.rag.precomputed.PrecomputedRetriever).
    """
    from src.io.read_write import make_run_dir

    _run_id, run_dir = make_run_dir(run_tag)
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


def _resolve_query_pairs(cfg: dict, args):
    """Load the pairs to embed/query for batch retrieval (query/full modes).

    By default, CVEfixes runs are restricted to a pinned JSON entries
    subset (``data.cvefixes.input_file``, default
    ``experiments_cves/selected_entries.json`` — the same sample used by
    ``cvefixes_experiments/scripts/performance/exp_method_vs_file_level.py``)
    instead of streaming the entire DB. Pass ``--input-file <path>`` to use a
    different subset, or ``--full-dataset`` to bypass the subset entirely.
    """
    from src.data import load_pairs, load_pairs_from_file

    if args.full_dataset:
        return load_pairs(cfg)

    input_file = args.input_file
    if input_file is None and "cvefixes" in cfg.get("data", {}).get("active", []):
        input_file = cfg.get("data", {}).get("cvefixes", {}).get(
            "input_file", entries_cache.DEFAULT_ENTRIES_FILE
        )

    if input_file:
        print(f"Loading pinned CVEfixes subset from {input_file}")
        return load_pairs_from_file(input_file, cfg)

    return load_pairs(cfg)


def run_batch_query(cfg: dict, args):
    """Batch retrieval: produce a per-query results.jsonl reusable by
    ``--mode batch --query-run <this run dir>``.

    Three modes:
      --index-dir <dir>: reuse a PERSISTED retrieval-experiment index (fast,
        offline, no CPGs) via leave-one-out reconstruction. Use this to reuse
        e.g. a prior ``--mode experiment`` run's ``indices/`` directory
        (looks for ``<embedding-variant>__hnsw.index`` + ``_meta.json``).
      --input-file / default: embeds a pinned CVEfixes subset (see
        ``_resolve_query_pairs``) against the global index built by
        ``--mode index``.
      --full-dataset: embeds ALL pairs from cfg["data"]["active"] (needs
        CPGs for the whole dataset).
    """
    if args.index_dir:
        rows = _leave_one_out_from_index(
            Path(args.index_dir),
            args.embedding_variant or cfg["rag"]["embedding_variant"],
            args.max_queries,
        )
        return _write_query_results(rows, "query_indexed")

    from src.metrics.retrieval_eval import retrieve_all
    from src.rag.retriever import Retriever

    pairs = _resolve_query_pairs(cfg, args)
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

    if hasattr(embedder, "fit") and not getattr(embedder, "_fitted", True):
        print(f"Fitting PCA embedder {variant!r} on {len(pairs)} query graphs...")
        embedder.fit([p.G_vuln for p in pairs])

    retriever = Retriever(index, top_k=rag_cfg["top_k"])
    query_results = retrieve_all(pairs, embedder, retriever, top_k=rag_cfg["top_k"])

    rows = [
        _precomputed_row(
            {
                "cve_id": pair.cve_id,
                "cwe_id": pair.cwe_id,
                "variant": pair.meta.get("variant", ""),
            },
            hits[0] if hits else None,
            hits[0].get("score", 0.0) if hits else 0.0,
        )
        for pair, hits in query_results
    ]
    return _write_query_results(rows, "query")


def run_full_pipeline(cfg: dict, args):
    """End-to-end: retrieval → LLM patching → evaluation."""
    from experiments.exp.prompt.patching_experiment import run_patching_experiment
    from experiments.exp.retrieval_experiment import run_experiment as run_retrieval_exp
    from src.io.read_write import make_run_dir

    run_id, run_dir = make_run_dir("full")
    print(f"\n{'━'*60}")
    print(f"  FULL PIPELINE — unified output: {run_dir}")
    print(f"{'━'*60}")

    # Step 1: Retrieval
    print(f"\n{'━'*60}")
    print(f"  STEP 1/3 — Retrieval (embed + FAISS top-k)")
    print(f"{'━'*60}")
    full_pairs = _resolve_query_pairs(cfg, args)
    if args.max_queries:
        full_pairs = full_pairs[: args.max_queries]
    run_retrieval_exp(full_pairs, cfg, output_dir=run_dir)
    print(f"\n  ✓ Retrieval complete: {run_dir / 'retrieval_results.jsonl'}")

    # Step 2: LLM Patching
    print(f"\n{'━'*60}")
    print(f"  STEP 2/3 — LLM Patching (using retrieval results)")
    print(f"{'━'*60}")
    run_patching_experiment(
        cfg,
        retriever_mode="precomputed",
        model_name=args.model,
        query_run=str(run_dir),
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        output_dir=run_dir,
        architecture=getattr(args, "architecture", None)
        or cfg.get("agents", {}).get("architecture", "single_turn"),
        prompt_variant=getattr(args, "prompt_variant", None)
        or cfg.get("agents", {}).get("prompt_variant")
        or cfg.get("rag", {}).get("prompt_variant", "default"),
        cve_root=getattr(args, "cve_root", None),
    )
    print(f"\n  ✓ Patching complete: {run_dir / 'results.jsonl'}")

    # Step 3: Evaluation + Dashboards
    print(f"\n{'━'*60}")
    print(f"  STEP 3/3 — Evaluation & Dashboards")
    print(f"{'━'*60}")
    from src.evaluate.__main__ import run_all

    results_jsonl = run_dir / "results.jsonl"
    run_all(
        results_path=results_jsonl,
        config_path=args.config,
        strip_comments=args.strip_comments,
    )

    print(f"\n{'━'*60}")
    print(f"  ALL DONE")
    print(f"{'━'*60}")
    print(f"  Output folder:  {run_dir}")
    print(f"  Patch analysis: {run_dir / 'patch_analysis.html'}")
    print(f"{'━'*60}")


def _print_cvefixes_info(cfg: dict):
    """Print CVEfixes setup and workflow status."""
    print(f"\n{'╭'+'─'*68+'╮'}")
    print(f"{'│':1}{'CVEfixes Graph-RAG Workflow':^68}{'│':1}")
    print(f"{'├'+'─'*68+'┤'}")

    # Check if CVEfixes is in active datasets
    active = cfg.get("data", {}).get("active", [])
    cvefixes_active = "cvefixes" in active
    status = "✓ ACTIVE" if cvefixes_active else "○ Configured"
    print(f"{'│':1} Dataset Status: {status:50} {'│':1}")

    # Check database
    db_path = Path(cfg.get("data", {}).get("cvefixes", {}).get("db_path", ""))
    db_exists = db_path.exists()
    db_status = f"✓ {db_path}" if db_exists else f"✗ Not found: {db_path}"
    print(f"{'│':1} Database: {db_status:60} {'│':1}")

    # Check graphml_root
    graphml_root = Path(cfg.get("data", {}).get("cvefixes", {}).get("graphml_root", ""))
    print(f"{'│':1} CPG Output: {str(graphml_root):60} {'│':1}")

    print(f"{'├'+'─'*68+'┤'}")
    print(f"{'│':1}{'Quick Start Commands':^68}{'│':1}")
    print(f"{'├'+'─'*68+'┤'}")
    print(
        f"{'│':1} 1. Generate CPGs:  python main.py --mode export              {'│':1}"
    )
    print(
        f"{'│':1} 2. Build index:    python main.py --mode index               {'│':1}"
    )
    print(
        f"{'│':1} 3. Query:          python main.py --mode query --cve CVE-... {'│':1}"
    )
    print(f"{'╰'+'─'*68+'╯'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=[
            "index",
            "query",
            "export",
            "experiment",
            "diagnostics",
            "batch",
            "full",
        ],
        default="export",
        help="Operation mode: export (CPG generation), index (embedding+FAISS), query (retrieval), etc.",
    )
    parser.add_argument(
        "--dataset",
        choices=["autopatch", "cvefixes"],
        default=None,
        help="Dataset to process ('cvefixes' for CVEfixes SQLite, 'autopatch' for CVE-list folder). "
        "If not specified, uses config.data.active",
    )
    parser.add_argument(
        "--level",
        choices=["method", "file"],
        default="method",
        help="Export granularity: 'method' (per-function CPG, legacy) or 'file' (one CPG per file, recommended)",
    )
    parser.add_argument("--cve")
    parser.add_argument(
        "--loo",
        action="store_true",
        help="run leave-one-out eval (slow, max 1000 samples)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="enable experiment split mode (overrides config)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="disable experiment split mode (overrides config)",
    )
    parser.add_argument(
        "--split-test-ratio", type=float, help="test ratio for split mode, e.g. 0.2"
    )
    parser.add_argument(
        "--aug-train-ratio",
        type=float,
        help="fraction of augmented train pairs to keep in index, e.g. 0.5",
    )
    # batch-mode arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="queries per batch flush (batch mode, default: 10)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="limit total queries for testing (batch mode)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Azure model/deployment name (batch mode, default: MODEL_NAME from .env)",
    )
    parser.add_argument(
        "--resume", default=None, help="path to run dir to resume (batch mode)"
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="use oracle retriever (perfect same-CVE lookup) instead of FAISS embedding retriever (batch mode)",
    )
    parser.add_argument(
        "--query-run",
        default=None,
        help="path to a --mode query run dir whose results.jsonl provides pre-computed retrieval (batch mode, replaces FAISS)",
    )
    parser.add_argument(
        "--strip-comments",
        action="store_true",
        default=True,
        help="remove C/C++ comments before patch comparison (full/batch mode)",
    )
    parser.add_argument(
        "--architecture",
        default=None,
        help="agent architecture (batch/full mode): single_turn (default). "
        "Overrides config agents.architecture.",
    )
    parser.add_argument(
        "--prompt-variant",
        default=None,
        help="prompt variant from src/agents/prompts/registry.yaml "
        "(batch/full mode). Overrides config agents.prompt_variant.",
    )
    parser.add_argument(
        "--cve-root",
        default=None,
        help="path to the AutoPatch CVE-list directory (batch/full mode). "
        "Overrides config data.autopatch.root.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="reuse a persisted retrieval-experiment index directory (query mode), "
        "e.g. a prior '--mode experiment' run's indices/ folder. Looks for "
        "<embedding-variant>__hnsw.index + _meta.json and does leave-one-out "
        "retrieval reconstruction — no CPGs or re-embedding needed.",
    )
    parser.add_argument(
        "--embedding-variant",
        default=None,
        help="embedder name to use for query mode (default: config rag.embedding_variant).",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="path to a pinned JSON entries file (e.g. "
        "experiments_cves/selected_entries.json) restricting batch query mode "
        "to a fixed CVEfixes subset, instead of the full DB. Defaults to "
        f"'{entries_cache.DEFAULT_ENTRIES_FILE}' for cvefixes runs unless "
        "--full-dataset is passed (query/full modes).",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="bypass the default pinned entries subset and load the full "
        "CVEfixes dataset for batch query mode (query/full modes).",
    )

    args = parser.parse_args()
    app_cfg = AppConfig.from_yaml(args.config)
    cfg = app_cfg.raw
    cfg.setdefault("paths", {})
    cfg["paths"].update(
        {
            "joern_bin_dir": str(app_cfg.paths.joern_bin_dir),
            "output_dir": str(app_cfg.paths.output_dir),
            "models_cache_dir": str(app_cfg.paths.models_cache_dir),
            "index_dir": str(app_cfg.paths.index_dir),
        }
    )

    # Print CVEfixes workflow info for relevant modes
    if args.mode in ("export", "index") and "cvefixes" in cfg.get("data", {}).get(
        "active", []
    ):
        _print_cvefixes_info(cfg)

    # Apply split overrides for modes that use them
    if args.mode in ("query", "experiment", "batch", "full"):
        apply_split_overrides(cfg, args)

    if args.mode == "export":
        run_export(app_cfg, args.dataset, level=getattr(args, "level", "method"))
    elif args.mode == "index":
        build_index(cfg)
    elif args.mode == "query":
        if args.cve:
            run_query(cfg, args.cve)
        else:
            run_batch_query(cfg, args)
    elif args.mode == "experiment":
        from experiments.exp.retrieval_experiment import RetrievalGridExperiment
        from src.data import load_pairs

        all_pairs = load_pairs(cfg)
        exp = RetrievalGridExperiment(
            run_leave_one_out=args.loo,
            preloaded_pairs=all_pairs,
        )
        exp.run(cfg)
    elif args.mode == "diagnostics":
        from src.data import load_pairs
        from src.diagnostics import run_diagnostics

        run_diagnostics(load_pairs(cfg))

    elif args.mode == "batch":
        from experiments.exp.prompt.patching_experiment import run_patching_experiment

        agent_cfg = cfg.get("agents", {})
        run_patching_experiment(
            cfg,
            retriever_mode="oracle" if args.oracle else "precomputed",
            model_name=args.model,
            query_run=args.query_run,
            max_queries=args.max_queries,
            batch_size=args.batch_size,
            resume=args.resume,
            architecture=args.architecture
            or agent_cfg.get("architecture", "single_turn"),
            prompt_variant=args.prompt_variant
            or agent_cfg.get("prompt_variant")
            or cfg.get("rag", {}).get("prompt_variant", "default"),
            cve_root=args.cve_root,
            dataset=args.dataset or "autopatch",
        )

    elif args.mode == "full":
        run_full_pipeline(cfg, args)
