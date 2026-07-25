#!/usr/bin/env python3
"""
Run the retrieval → patching → evaluation pipeline (the manual
``main.py --mode index`` / ``--mode query`` / ``--mode batch`` /
``python -m src.evaluate`` sequence from cvefixes_experiments/scripts/run.sh)
back-to-back for several embedding variants, consolidated under ONE output
folder, and print a combined results table at the end.

Each variant gets its own FAISS index/metadata (so runs never clobber each
other) but shares the same dataset/config (``config.yaml``'s
``data.active`` / ``experiment.split`` / ``agents.*`` settings) so results
are directly comparable.

Use ``--precomputed-split-dir``/``--precomputed-split-variant`` to pin ALL
variants to the exact same index/query partition (the format written by
``utils/build_balanced_split.py`` and consumed by
``exp_file_method_interface.py``) instead of each variant resampling its own
split — see ``src.data.split.build_split`` / ``_precomputed_split``.

Output layout (one parent folder for everything):

    cvefixes_experiments/output/embedding_sweep_<ts>_<tag>/
        <variant>/
            config.yaml                         (resolved per-variant config, for reproducibility)
            indices/<variant>__G_vuln__hnsw.index
            indices/<variant>__G_vuln__hnsw_meta.json
            query/<ts>_query_.../results.jsonl   (retrieval results, reusable via --query-run)
            patching/<ts>_patching_.../<cell>/results.jsonl
                                                  evaluation.jsonl
                                                  evaluation_summary.json
                                                  patch_analysis.html
        summary.json                             (combined retrieval + patch metrics)

Usage:
    uv run python -m cvefixes_experiments.scripts.run_embedding_sweep
    uv run python -m cvefixes_experiments.scripts.run_embedding_sweep \\
        --variants combined codebert_pattern \\
        --architecture tool_calling --max-queries 5 \\
        --precomputed-split-dir cvefixes_experiments/output/method_vs_file_level/ \\
        --precomputed-split-variant balanced
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from src.io.read_write import make_run_dir
from src.rag.indexing import build_index
from src.rag.query import run_batch_query
from src.schema_config import AppConfig

DEFAULT_VARIANTS = ["combined", "codebert_pattern"]
DEFAULT_OUTPUT_ROOT = Path("cvefixes_experiments/output")


def build_base_cfg(config_path: str) -> dict:
    """Load config.yaml the same way main.py does (typed validation + paths)."""
    app_cfg = AppConfig.from_yaml(config_path)
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
    return cfg


def _make_direct_cfg(cfg: dict, entries_file: Path) -> dict:
    """Return a deep copy of cfg that loads pairs directly from *entries_file*
    with the split machinery disabled.  Avoids the (cve_id, func_name)
    re-matching in build_split which can silently drop unresolved entries."""
    out = copy.deepcopy(cfg)
    out.setdefault("data", {}).setdefault("cvefixes_file", {})["input_file"] = str(entries_file)
    split_cfg = out.setdefault("experiment", {}).setdefault("split", {})
    split_cfg["enabled"] = False
    split_cfg.pop("precomputed_split_dir", None)
    split_cfg.pop("precomputed_split_variant", None)
    return out


def run_variant(base_cfg: dict, variant: str, sweep_dir: Path, args) -> dict:
    """Run index → query → batch → evaluate for one embedding variant."""
    print(f"\n{'='*70}")
    print(f"  VARIANT: {variant}")
    print(f"{'='*70}")

    cfg = copy.deepcopy(base_cfg)
    variant_dir = sweep_dir / variant
    indices_dir = variant_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    # Point this run at its OWN index/metadata + embedding variant so it
    # can't collide with (or silently reuse) another variant's FAISS index.
    cfg["rag"]["embedding_variant"] = variant
    cfg["rag"]["index_path"] = str(indices_dir / f"{variant}__G_vuln__hnsw.index")
    cfg["rag"]["metadata_path"] = str(indices_dir / f"{variant}__G_vuln__hnsw_meta.json")

    # ── Resolve prebuilt index / query pair files ───────────────────
    # When pipeline_data_v0/ (or equivalent) already has the exact
    # index_pairs_entries.json + query_pairs_entries.json produced by
    # data_pipeline.py, use them directly as the dataset input rather
    # than going through split_info re-matching (which can silently drop
    # entries whose func_name doesn't match exactly).
    split_dir_str = cfg.get("experiment", {}).get("split", {}).get("precomputed_split_dir")
    index_cfg = cfg
    query_cfg = cfg
    if split_dir_str:
        _sd = Path(split_dir_str)
        _if = _sd / "index_pairs_entries.json"
        _qf = _sd / "query_pairs_entries.json"
        if _if.exists() and _qf.exists():
            index_cfg = _make_direct_cfg(cfg, _if)
            query_cfg = _make_direct_cfg(cfg, _qf)
            print(f"  Using prebuilt pairs directly:")
            print(f"    index → {_if}")
            print(f"    query → {_qf}")

    # Save query_cfg as the canonical variant config (governs query +
    # patching + evaluation; index_cfg differs only in input_file).
    variant_config_path = variant_dir / "config.yaml"
    with open(variant_config_path, "w") as f:
        yaml.safe_dump(query_cfg, f, sort_keys=False)
    print(f"  Resolved config → {variant_config_path}")

    # ── 1/4 index ──────────────────────────────────────────────────
    print(f"\n--- [1/4] index ({variant}) ---")
    build_index(index_cfg)

    # ── 2/4 query (full retrieval batch, no --max-queries) ─────────
    print(f"\n--- [2/4] query ({variant}) ---")
    query_args = SimpleNamespace(
        index_dir=None,
        embedding_variant=variant,
        max_queries=None,
        output_dir=str(variant_dir / "query"),
    )
    query_run_dir = run_batch_query(query_cfg, query_args)

    # ── 3/4 batch (LLM patching, capped by --max-queries) ───────────
    print(f"\n--- [3/4] batch ({variant}) ---")
    from experiments.exp.prompt.patching_experiment import run_patching_experiment

    agent_cfg = query_cfg.get("agents", {})
    patch_run_dir = run_patching_experiment(
        query_cfg,
        retriever_mode="precomputed",
        model_name=args.model,
        query_run=str(query_run_dir),
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        architecture=args.architecture or agent_cfg.get("architecture", "single_turn"),
        prompt_variant=agent_cfg.get("prompt_variant")
        or query_cfg.get("rag", {}).get("prompt_variant", "default"),
        dataset=query_cfg.get("data", {}).get("active", ["autopatch"])[0],
        output_dir=variant_dir / "patching",
    )

    # ── 4/4 evaluate ─────────────────────────────────────────────
    print(f"\n--- [4/4] evaluate ({variant}) ---")
    from src.evaluate.__main__ import run_all

    results_jsonl = patch_run_dir / "results.jsonl"
    run_all(
        results_path=results_jsonl,
        config_path=str(variant_config_path),
        strip_comments=args.strip_comments,
    )

    return {
        "variant": variant,
        "index_path": index_cfg["rag"]["index_path"],
        "query_run_dir": str(query_run_dir),
        "patch_run_dir": str(patch_run_dir),
        "results_jsonl": str(results_jsonl),
        "split": {
            "index_entries": str(_if) if split_dir_str and _if.exists() else None,
            "query_entries": str(_qf) if split_dir_str and _qf.exists() else None,
            "precomputed_split_dir": split_dir_str,
        },
    }


def summarize_retrieval(query_run_dir: str) -> dict:
    """Cheap retrieval summary from a query run's results.jsonl.

    Each row records only the top-1 hit (see ``src.rag.query._precomputed_row``),
    so this reports hit@1 / cwe@1 rates, not hit@5 / hit@10.
    """
    path = Path(query_run_dir) / "results.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n = len(rows)
    if n == 0:
        return {"n_queries": 0}
    success = [r for r in rows if r.get("status") == "success"]
    cve_hits = sum(1 for r in success if r.get("retrieval", {}).get("cve_match"))
    cwe_hits = sum(1 for r in success if r.get("retrieval", {}).get("cwe_match"))
    return {
        "n_queries": n,
        "n_with_match": len(success),
        "hit@1": round(cve_hits / n, 4),
        "cwe_hit@1": round(cwe_hits / n, 4),
    }


def summarize_patch(patch_run_dir: str) -> dict:
    """Load the evaluation_summary.json written by src.evaluate for one run."""
    path = Path(patch_run_dir) / "evaluation_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_extra_summary(spec: str) -> tuple[str, dict]:
    """Parse a ``name=path/to/run_dir`` spec pointing at a pre-existing run
    (e.g. a prior codebert_seq run) so it can be folded into the final table
    without re-running anything."""
    name, _, run_dir = spec.partition("=")
    run_dir = Path(run_dir)
    retrieval = summarize_retrieval(str(run_dir))
    patch = summarize_patch(str(run_dir))
    return name, {"variant": name, "retrieval": retrieval, "patch": patch}


def print_summary_table(rows: list[dict]) -> None:
    print(f"\n{'='*90}")
    print("  OVERALL RESULTS")
    print(f"{'='*90}")

    header = f"{'variant':<20}{'hit@1':>10}{'cwe_hit@1':>12}{'bleu_4':>10}{'jaccard':>10}{'bertF1':>10}{'rougeL':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        r = row.get("retrieval", {})
        p = row.get("patch", {})
        print(
            f"{row['variant']:<20}"
            f"{r.get('hit@1', float('nan')):>10.4f}"
            f"{r.get('cwe_hit@1', float('nan')):>12.4f}"
            f"{p.get('avg_bleu_4', float('nan')):>10.4f}"
            f"{p.get('avg_token_jaccard', float('nan')):>10.4f}"
            f"{p.get('avg_bertscore_f1', float('nan')):>10.4f}"
            f"{p.get('avg_rougeL_f1', float('nan')):>10.4f}"
        )
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help=f"embedding variants to sweep (default: {DEFAULT_VARIANTS})",
    )
    parser.add_argument("--architecture", default="tool_calling")
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--strip-comments", action="store_true", default=True,
        help="strip C/C++ comments before patch comparison (default: on)",
    )
    parser.add_argument(
        "--tag", default="embedding_sweep",
        help="name for the consolidated output folder",
    )
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT),
        help="parent directory under which the sweep's output folder is created",
    )
    parser.add_argument(
        "--extra-summary", action="append", default=[],
        metavar="NAME=RUN_DIR",
        help="fold in a pre-existing run (e.g. NAME=RUN_DIR pointing at a prior "
        "codebert_seq run dir containing results.jsonl + evaluation_summary.json) "
        "into the final printed table, without re-running it",
    )
    parser.add_argument(
        "--precomputed-split-dir", default=None,
        help="directory containing split_info_{balanced,stratified}.json (as written by "
        "utils/build_balanced_split.py). Pins ALL variants to this exact index/query "
        "partition instead of each one resampling its own split.",
    )
    parser.add_argument(
        "--precomputed-split-variant", default="balanced", choices=["balanced", "stratified"],
        help="which split_info_<variant>.json to use under --precomputed-split-dir (default: balanced)",
    )
    parser.add_argument(
        "--judge-provider", default="azure", choices=["azure", "anthropic", "gemini"],
        help="LLM provider for patch verification (default: azure). Set via LLM_JUDGE_PROVIDER env var.",
    )
    args = parser.parse_args()

    base_cfg = build_base_cfg(args.config)

    if args.precomputed_split_dir:
        split_cfg = base_cfg.setdefault("experiment", {}).setdefault("split", {})
        split_cfg["enabled"] = True
        split_cfg["precomputed_split_dir"] = args.precomputed_split_dir
        split_cfg["precomputed_split_variant"] = args.precomputed_split_variant
        print(
            f"Pinned split: {args.precomputed_split_dir} "
            f"(variant={args.precomputed_split_variant}) — applied to all variants"
        )

    _run_id, sweep_dir = make_run_dir(args.tag, output_dir=Path(args.output_root))
    print(f"Consolidated output folder: {sweep_dir}")

    variant_results = []
    for variant in args.variants:
        result = run_variant(base_cfg, variant, sweep_dir, args)
        result["retrieval"] = summarize_retrieval(result["query_run_dir"])
        result["patch"] = summarize_patch(result["patch_run_dir"])
        variant_results.append(result)

    for spec in args.extra_summary:
        name, extra = load_extra_summary(spec)
        variant_results.append(extra)

    summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps(variant_results, indent=2, default=str))
    print(f"\nCombined summary written to: {summary_path}")

    print_summary_table(variant_results)


if __name__ == "__main__":
    main()
