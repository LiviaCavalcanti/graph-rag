#!/usr/bin/env python3
"""
Experiment coordinator — unified CLI for running and chaining experiments.

Replaces the scattered entry points (python -m experiments.slicing_comparison,
python -m experiments.runner, etc.) with a single pipeline-aware command.

Usage:
    # Run the full retrieval grid (embedder × backend)
    python -m experiments.experiment retrieval

    # Slicing ablation (single run)
    python -m experiments.experiment slicing

    # Slicing ablation (repeated with error bars)
    python -m experiments.experiment slicing --repeat 5

    # Post-hoc analysis on an existing run
    python -m experiments.experiment analyze --run-dir experiments/output/<run_id>

    # Agent patching experiment
    python -m experiments.experiment agent --agent-mode oracle

    # Knowledge analysis (CWE ontology + callgraph)
    python -m experiments.experiment knowledge --run-dir experiments/output/<run_id>

    # Full pipeline: retrieval → miss analysis → crossing → dashboard
    python -m experiments.experiment full

Common flags:
    --config config.yaml         Config file (default: config.yaml)
    --split / --no-split         Override split mode
    --max-queries N              Limit queries (for testing)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import load_config, build_split, make_run_dir, save_json
from src.data.autopatch import load_pairs


def _apply_split_overrides(cfg: dict, args) -> dict:
    """Apply CLI split overrides to config."""
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("split", {})
    sc = cfg["experiment"]["split"]
    if getattr(args, "split", False):
        sc["enabled"] = True
    if getattr(args, "no_split", False):
        sc["enabled"] = False
    if getattr(args, "split_test_ratio", None) is not None:
        sc["test_ratio"] = args.split_test_ratio
    if getattr(args, "aug_train_ratio", None) is not None:
        sc["augmented_train_ratio"] = args.aug_train_ratio
    return cfg


# ── subcommands ──────────────────────────────────────────────────────

def cmd_retrieval(args):
    """Full retrieval grid experiment (embedder × backend)."""
    from experiments.exp.retrieval_experiment import RetrievalGridExperiment

    cfg = _apply_split_overrides(load_config(args.config), args)
    pairs = load_pairs(cfg)
    print(f"Loaded {len(pairs)} pairs")
    exp = RetrievalGridExperiment(
        run_leave_one_out=args.loo,
        preloaded_pairs=pairs,
    )
    exp.run(cfg)


def cmd_slicing(args):
    """Slicing ablation: 2×2 factorial (slicing × labelling)."""
    cfg = _apply_split_overrides(load_config(args.config), args)
    qv = getattr(args, 'query_variant', None)

    if args.repeat and args.repeat > 1:
        from experiments.repeated_experience_runner import run_repeated
        run_repeated(cfg, n_runs=args.repeat, query_variant=qv)
    else:
        from experiments.exp.slicing_comparison import run_comparison
        run_comparison(cfg, query_variant=qv)


def cmd_agent(args):
    """Agent patching experiment (oracle or combined retriever)."""
    from experiments.exp.retrieval_experiment import run_experiment as run_retrieval
    from experiments.exp.prompt.patching_experiment import run_patching_experiment

    cfg = _apply_split_overrides(load_config(args.config), args)

    mode = args.agent_mode
    prompt_variant = cfg.get("rag", {}).get("prompt_variant", "default")

    if mode == "retriever":
        pairs = load_pairs(cfg)
        if args.max_queries:
            pairs = pairs[: args.max_queries]
        run_retrieval(pairs, cfg)
    elif mode == "patch":
        run_patching_experiment(
            cfg,
            retriever_mode="oracle" if args.oracle else "precomputed",
            model_name=args.model,
            query_run=args.query_run,
            max_queries=args.max_queries,
            batch_size=args.batch_size,
            resume=args.resume,
            prompt_variant=prompt_variant,
        )
    elif mode == "full":
        # ── Step 1: Retrieval ──────────────────────────────────────
        print(f"\n{'━'*60}")
        print(f"  STEP 1/3 — Retrieval (embed + FAISS top-k)")
        print(f"{'━'*60}")
        pairs = load_pairs(cfg)
        if args.max_queries:
            pairs = pairs[: args.max_queries]
        retrieval_output = run_retrieval(pairs, cfg)
        retrieval_dir = retrieval_output.run_dir
        print(f"\n  ✓ Retrieval complete: {retrieval_dir}")

        # ── Step 2: LLM Patching ──────────────────────────────────
        print(f"\n{'━'*60}")
        print(f"  STEP 2/3 — LLM Patching (using retrieval results)")
        print(f"{'━'*60}")
        patch_dir = run_patching_experiment(
            cfg,
            retriever_mode="precomputed",
            model_name=args.model,
            query_run=str(retrieval_dir),
            max_queries=args.max_queries,
            batch_size=args.batch_size,
            prompt_variant=prompt_variant,
        )
        print(f"\n  ✓ Patching complete: {patch_dir}")

        # ── Step 3: Evaluation + Dashboard ─────────────────────────
        print(f"\n{'━'*60}")
        print(f"  STEP 3/3 — Evaluation & Dashboard")
        print(f"{'━'*60}")
        from src.evaluate.__main__ import run_all

        results_jsonl = patch_dir / "results.jsonl"
        dashboard_path = run_all(
            results_path=results_jsonl,
            config_path=args.config,
        )

        print(f"\n{'━'*60}")
        print(f"  ALL DONE")
        print(f"{'━'*60}")
        print(f"  Retrieval run:  {retrieval_dir}")
        print(f"  Patching run:   {patch_dir}")
        print(f"  Dashboard:      {dashboard_path}")
        print(f"{'━'*60}")


def cmd_analyze(args):
    """Post-hoc analysis pipeline on an existing run directory."""
    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    steps = args.steps or ["misses", "crossing", "dashboard"]

    if "misses" in steps:
        print(f"\n{'='*50}\n  Miss analysis\n{'='*50}")
        from experiments.dashboard_scripts.analyze_misses import analyze_results
        analyze_results(
            results_path=results_path,
            out_json=run_dir / "miss_analysis.json",
            out_html=run_dir / "miss_dashboard.html",
            uncertainty_quantile=25.0,
            uncertainty_prob_floor=0.12,
            uncertainty_margin_floor=0.005,
            temperature=1.0,
            max_examples=20,
        )
        print(f"  → {run_dir / 'miss_analysis.json'}")

    if "crossing" in steps:
        print(f"\n{'='*50}\n  Crossing / fusion analysis\n{'='*50}")
        from experiments.verify_crossing import analyze_crossing
        analyze_crossing(str(results_path))

    if "dashboard" in steps:
        print(f"\n{'='*50}\n  Dashboard generation\n{'='*50}")
        from experiments.dashboard_scripts.dashboard import generate_html_dashboard
        generate_html_dashboard(str(run_dir))

    print(f"\nAll analysis written to {run_dir}")


def cmd_knowledge(args):
    """CWE ontology + callgraph analysis."""
    from experiments.put_aside.knowledge_experiment import run_knowledge_analysis

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else None

    if run_dir and not (run_dir / "results.json").exists():
        print(f"ERROR: {run_dir / 'results.json'} not found")
        sys.exit(1)

    run_knowledge_analysis(
        config_path=args.config,
        run_dir=str(run_dir) if run_dir else None,
    )


def cmd_diagnostics(args):
    """Per-graph diagnostic figures (vuln size, diff ratio, etc.)."""
    from src.diagnostics import run_diagnostics

    cfg = load_config(args.config)
    pairs = load_pairs(cfg)
    run_diagnostics(pairs)


def cmd_full(args):
    """Full pipeline: retrieval → analyze → dashboard."""
    from experiments.exp.retrieval_experiment import RetrievalGridExperiment

    cfg = _apply_split_overrides(load_config(args.config), args)
    pairs = load_pairs(cfg)
    print(f"Loaded {len(pairs)} pairs")

    # 1. Run retrieval experiment
    print(f"\n{'='*50}\n  Step 1: Retrieval experiment\n{'='*50}")
    exp = RetrievalGridExperiment(
        run_leave_one_out=args.loo,
        preloaded_pairs=pairs,
    )
    output = exp.run(cfg)
    run_dir = output.run_dir

    if run_dir and run_dir.exists():
        results_path = run_dir / "results.json"

        # 2. Miss analysis
        print(f"\n{'='*50}\n  Step 2: Miss analysis\n{'='*50}")
        try:
            from experiments.dashboard_scripts.analyze_misses import analyze_results
            analyze_results(
                results_path=results_path,
                out_json=run_dir / "miss_analysis.json",
                out_html=run_dir / "miss_dashboard.html",
                uncertainty_quantile=25.0,
                uncertainty_prob_floor=0.12,
                uncertainty_margin_floor=0.005,
                temperature=1.0,
                max_examples=20,
            )
        except Exception as e:
            print(f"  Warning: miss analysis failed: {e}")

        # 3. Crossing analysis
        print(f"\n{'='*50}\n  Step 3: Crossing analysis\n{'='*50}")
        try:
            from experiments.verify_crossing import analyze_crossing
            analyze_crossing(str(results_path))
        except Exception as e:
            print(f"  Warning: crossing analysis failed: {e}")

        # 4. Dashboard
        print(f"\n{'='*50}\n  Step 4: Dashboard\n{'='*50}")
        try:
            from experiments.dashboard_scripts.dashboard import generate_html_dashboard
            generate_html_dashboard(str(run_dir))
        except Exception as e:
            print(f"  Warning: dashboard generation failed: {e}") 

        print(f"\nFull pipeline complete → {run_dir}")
    else:
        print("Warning: could not locate run directory for post-processing")


# ── CLI ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.experiment",
        description="Experiment coordinator — run, analyze, and visualize experiments.",
    )
    # global flags
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", action="store_true", help="Force split mode on")
    parser.add_argument("--no-split", action="store_true", help="Force split mode off")
    parser.add_argument("--split-test-ratio", type=float)
    parser.add_argument("--aug-train-ratio", type=float)
    parser.add_argument("--max-queries", type=int)

    sub = parser.add_subparsers(dest="command", required=True)

    # retrieval
    p = sub.add_parser("retrieval", help="Full embedder × backend grid")
    p.add_argument("--loo", action="store_true", help="Run leave-one-out (slow)")

    # slicing
    p = sub.add_parser("slicing", help="Slicing ablation (2×2 factorial)")
    p.add_argument("--repeat", type=int, default=1, help="Number of repeated runs (default: 1)")
    p.add_argument(
        "--query-variant",
        default=None,
        help="Fix query graphs to a variant (G_before, G_vuln, runner_compat). "
             "Default: same as index variant.",
    )

    # agent
    p = sub.add_parser("agent", help="Agent patching experiment")
    p.add_argument("--agent-mode", choices=["oracle", "retriever", "patch", "full"], required=True)
    p.add_argument("--model", default="gpt-4o", help="LLM model/deployment name")
    p.add_argument("--oracle", action="store_true", help="Use oracle retriever for patch mode")
    p.add_argument("--query-run", default=None, help="Path to retrieval run dir (for patch mode)")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--resume", default=None, help="Path to run dir to resume")

    # analyze (post-hoc)
    p = sub.add_parser("analyze", help="Post-hoc analysis on existing run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--steps", nargs="+", choices=["misses", "crossing", "dashboard"],
                   help="Which analysis steps (default: all)")

    # knowledge
    p = sub.add_parser("knowledge", help="CWE ontology + callgraph analysis")
    p.add_argument("--run-dir")

    # diagnostics
    p = sub.add_parser("diagnostics", help="Per-graph diagnostic figures")

    # full pipeline
    p = sub.add_parser("full", help="Full pipeline: retrieval → analyze → dashboard")
    p.add_argument("--loo", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "retrieval": cmd_retrieval,
        "slicing": cmd_slicing,
        "agent": cmd_agent,
        "analyze": cmd_analyze,
        "knowledge": cmd_knowledge,
        "diagnostics": cmd_diagnostics,
        "full": cmd_full,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
