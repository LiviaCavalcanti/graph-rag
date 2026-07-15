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
from pathlib import Path

from src.data.export import run_export
from src.rag.indexing import build_index
from src.rag.query import run_batch_query, run_query
from src.schema_config import AppConfig


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


def run_full_pipeline(cfg: dict, args):
    """End-to-end: retrieval → LLM patching → evaluation.

    Retrieval pairs come from ``src.data.load_pairs(cfg)`` — i.e. whichever
    dataset(s) are in ``data.active``. Point ``data.active`` at
    ``cvefixes_file`` (with ``data.cvefixes_file.input_file``/``sample_mode``)
    for a pinned, CWE-sampled CVEfixes subset instead of the full DB.
    """
    from experiments.exp.prompt.patching_experiment import run_patching_experiment
    from experiments.exp.retrieval_experiment import run_experiment as run_retrieval_exp
    from src.data import load_pairs
    from src.io.read_write import make_run_dir

    run_id, run_dir = make_run_dir("full")
    print(f"\n{'━'*60}")
    print(f"  FULL PIPELINE — unified output: {run_dir}")
    print(f"{'━'*60}")

    # Step 1: Retrieval
    print(f"\n{'━'*60}")
    print(f"  STEP 1/3 — Retrieval (embed + FAISS top-k)")
    print(f"{'━'*60}")
    full_pairs = load_pairs(cfg)
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
