#!/usr/bin/env python3
"""
File-level CVEfixes retrieval experiment.

Evaluates CVEfixes at FILE GRANULARITY instead of function granularity.
Loads files directly from the database and merges all methods within each file.

This enables:
  - Function-level vs file-level retrieval performance comparison
  - Understanding how file-level aggregation affects embedding quality
  - True file-level vulnerability signatures

Usage:
    python -m cvefixes_experiments.scripts.pipeline_verification.exp_cvefixes_retrieval_grid_file_level \
      --config config.yaml \
      --output-dir cvefixes_experiments/output/cvefixes_retrieval_grid_file_level \
      --embedders codebert_seq codebert_pattern combined gin
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from experiments.exp.retrieval_experiment import RetrievalGridExperiment
from experiments.base import ExperimentOutput
from src.data.cvefixes import CVEFixesDataset
from src.embeddings import REGISTRY as EMBEDDER_REGISTRY


def _normalize_embedder_name(name: str) -> str:
    aliases = {
        "codebert-seq": "codebert_seq",
        "codebert-pattern": "codebert_pattern",
    }
    key = name.strip()
    return aliases.get(key, key)


def _sample_cve_aware(
    pairs: list,
    *,
    mode: str,
    seed: int = 42,
    min_cves_per_cwe: int = 3,
    total: int | None = None,
) -> tuple[list, dict[str, Any]]:
    """Sample pairs at CVE granularity to reshape the CWE distribution.

    Goal: build two samples that serve the same purpose but have different
    *shapes*, so we can measure how the dataset's CWE distribution affects
    retrieval.

    Modes:
      - ``"proportional"``: keep the dataset's natural CWE proportions.
      - ``"balanced"``: cap every CWE to the same budget so the distribution is
        approximately uniform across CWEs (under-samples the majority CWEs).

    CVE-aware guarantees (both modes):
      - Whole CVE groups are kept together (a CVE never straddles the sample
        boundary), so the downstream leakage-safe split can always find
        same-CVE support for its queries.
      - At least ``min_cves_per_cwe`` CVEs are retained per CWE when available.
      - Multi-pair CVEs are preferred, so each CWE keeps CVEs that can serve as
        both index and query.

    Args:
        pairs: file-level ``FunctionPair`` objects (graphs not required yet).
        mode: ``"proportional"`` or ``"balanced"``.
        seed: RNG seed for reproducible CVE selection.
        min_cves_per_cwe: minimum CVEs kept per CWE (whole groups).
        total: target number of pairs; share it across modes for a fair
            comparison. If ``None``: proportional keeps all pairs; balanced
            under-samples every CWE to the smallest CWE's pair count.

    Returns:
        ``(sampled_pairs, info)`` where ``info`` records the realized per-CWE
        CVE/pair counts and the requested target.
    """
    if mode not in ("proportional", "balanced"):
        return pairs[:], {"mode": mode or "none", "applied": False}

    rng = random.Random(seed)

    by_cwe: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        cwe = getattr(pair, "cwe_id", "UNKNOWN") or "UNKNOWN"
        cve = getattr(pair, "cve_id", "UNKNOWN") or "UNKNOWN"
        by_cwe[cwe][cve].append(pair)

    cwes = sorted(by_cwe)
    n_cwes = max(1, len(cwes))
    cwe_pair_counts = {c: sum(len(v) for v in by_cwe[c].values()) for c in cwes}
    total_pairs = sum(cwe_pair_counts.values()) or 1

    if mode == "balanced":
        if total and total > 0:
            base = max(1, int(total) // n_cwes)
        else:
            base = min(cwe_pair_counts.values())
        pair_quota = {c: base for c in cwes}
    else:  # proportional
        target_total = int(total) if total and total > 0 else total_pairs
        pair_quota = {
            c: max(1, round(target_total * cwe_pair_counts[c] / total_pairs))
            for c in cwes
        }

    sampled: list = []
    per_cwe: dict[str, dict[str, int]] = {}
    for cwe in cwes:
        groups = list(by_cwe[cwe].items())  # (cve_id, [pairs])
        rng.shuffle(groups)
        # Prefer multi-pair CVEs so each CWE keeps query-capable groups.
        groups.sort(key=lambda kv: len(kv[1]), reverse=True)

        quota = pair_quota[cwe]
        picked: list = []
        n_cve = 0
        for _cve, grp in groups:
            if n_cve < min_cves_per_cwe or len(picked) < quota:
                picked.extend(grp)
                n_cve += 1
            else:
                break
        sampled.extend(picked)
        per_cwe[cwe] = {"cves": n_cve, "pairs": len(picked)}

    rng.shuffle(sampled)

    info = {
        "mode": mode,
        "applied": True,
        "seed": seed,
        "min_cves_per_cwe": min_cves_per_cwe,
        "target_total": int(total) if total and total > 0 else None,
        "result_total_pairs": len(sampled),
        "result_total_cves": sum(v["cves"] for v in per_cwe.values()),
        "per_cwe": per_cwe,
    }
    return sampled, info


class CVEFixesFileLevelRetrievalExperiment(RetrievalGridExperiment):
    """Retrieval grid for file-level CVEfixes evaluation."""

    def __init__(
        self,
        *,
        run_leave_one_out: bool = False,
        run_self_retrieval: bool = True,
        ks: list[int] | None = None,
    ):
        super().__init__(
            run_leave_one_out=run_leave_one_out,
            run_self_retrieval=run_self_retrieval,
            ks=ks,
            preloaded_pairs=None,
        )

    @property
    def name(self) -> str:
        return "cvefixes_retrieval_grid_file_level"

    def load_data(self, cfg: dict) -> dict[str, Any]:
        """Load file-level pairs: metadata from filesystem, split, then load graphs for needed pairs."""
        from src.data.pipeline import cpg_dir_for, load_cpg_dirs_parallel, compute_graph_diff
        from src.data.split import build_split
        from collections import Counter
        import time
        
        # Use CVEFixesDataset to load file-level pairs
        cvefixes_cfg = cfg.get("data", {}).get("cvefixes", {})
        dataset = CVEFixesDataset(cvefixes_cfg)
        self._dataset = dataset
        graphml_root = cvefixes_cfg.get("graphml_root", "graphml_cvefixes_file")
        
        # Step 1: Load metadata from filesystem (instant — reads metadata.json files)
        # Falls back to DB query if no metadata.json files found
        from pathlib import Path
        metadata_files = list(Path(graphml_root).glob("*/metadata.json")) if Path(graphml_root).exists() else []
        
        if metadata_files:
            all_file_pairs = dataset.load_from_filesystem()
            print(f"  [CVEfixes] loaded {len(all_file_pairs)} file-level pairs from filesystem")
        else:
            print(f"  [CVEfixes] no metadata.json found in {graphml_root}, falling back to DB query")
            all_file_pairs = dataset.load_lightweight_file_level()
            print(f"  [CVEfixes] loaded {len(all_file_pairs)} file-level pairs from DB")
        
        # Step 1b: Optional CVE-aware distribution sampling. Builds a reshaped
        # sample (natural vs. uniform CWE mix) so we can measure how the dataset
        # distribution affects retrieval. Whole CVE groups are kept together.
        sampling_cfg = cfg.get("experiment", {}).get("sampling", {})
        sample_mode = str(sampling_cfg.get("mode", "none") or "none")
        if sample_mode in ("proportional", "balanced"):
            all_file_pairs, sample_info = _sample_cve_aware(
                all_file_pairs,
                mode=sample_mode,
                seed=int(sampling_cfg.get("seed", 42)),
                min_cves_per_cwe=int(sampling_cfg.get("min_cves_per_cwe", 3)),
                total=sampling_cfg.get("total"),
            )
            per_cwe_pairs = {c: v["pairs"] for c, v in sample_info["per_cwe"].items()}
            print(
                f"  [Sampler] mode={sample_mode} -> {sample_info['result_total_pairs']} pairs, "
                f"{sample_info['result_total_cves']} CVEs across {len(sample_info['per_cwe'])} CWEs"
            )
            print(f"  [Sampler] per-CWE pairs: {per_cwe_pairs}")
        else:
            sample_info = {"mode": "none", "applied": False}

        # Step 2: Leakage-safe train/test split BEFORE loading graphs.
        # Reuses the same CVE/CWE-aware split as the method-level experiment so
        # index/query are disjoint and every query has same-CVE support in index.
        # Settings come from config experiment.split (seed, test_ratio, enabled).
        from cvefixes_experiments.scripts.pipeline_verification.exp_cvefixes_retrieval_grid import (
            _split_cvefixes_pairs,
        )
        split_cfg = cfg.get("experiment", {}).get("split", {})
        if split_cfg.get("enabled", True):
            index_pairs, query_pairs, split_info = _split_cvefixes_pairs(
                all_file_pairs,
                test_ratio=float(split_cfg.get("test_ratio", 0.2)),
                seed=int(split_cfg.get("seed", 42)),
                min_pairs_per_cve=int(split_cfg.get("min_pairs_per_cve", 2)),
            )
        else:
            index_pairs, query_pairs, split_info = build_split(all_file_pairs, cfg)
        split_info["sampling"] = sample_info
        print(f"  Split -> index={len(index_pairs)} files  query={len(query_pairs)} files")
        print(f"  Index CWE dist: {dict(Counter(p.cwe_id for p in index_pairs))}")
        print(f"  Query CWE dist: {dict(Counter(p.cwe_id for p in query_pairs))}")
        
        # Step 3: Load graphs ONLY for pairs we'll actually use (index + query)
        # File-level graphs: one CPG per file (dir_name = CVE-XXXX_fYYYYY)
        needed_pairs = list(index_pairs) + list(query_pairs)
        
        graph_dirs_before = []
        graph_dirs_after = []
        for p in needed_pairs:
            dir_name = p.meta["dir_name"]
            graph_dirs_before.append(cpg_dir_for(graphml_root, cve_id=dir_name, variant="original", version="before"))
            graph_dirs_after.append(cpg_dir_for(graphml_root, cve_id=dir_name, variant="original", version="after"))
        
        all_dirs = graph_dirs_before + graph_dirs_after
        print(f"  Loading {len(all_dirs)} graphs in parallel for {len(needed_pairs)} pairs...")
        t0 = time.perf_counter()
        loaded_graphs = load_cpg_dirs_parallel(all_dirs, max_workers=8)
        print(f"  Loaded {len(loaded_graphs)}/{len(all_dirs)} graphs in {time.perf_counter()-t0:.1f}s")
        
        # Assign graphs to pairs
        loaded_count = 0
        for i, p in enumerate(needed_pairs):
            dir_b = graph_dirs_before[i]
            dir_a = graph_dirs_after[i]
            
            if dir_b in loaded_graphs and dir_a in loaded_graphs:
                G_before = loaded_graphs[dir_b]
                G_after = loaded_graphs[dir_a]
                if G_before.number_of_nodes() > 0 and G_after.number_of_nodes() > 0:
                    p.G_before = G_before
                    p.G_after = G_after
                    p.G_vuln = compute_graph_diff(G_before, G_after)
                    loaded_count += 1
        
        print(f"  Successfully populated {loaded_count}/{len(needed_pairs)} pairs with graphs")
        
        # Filter out pairs that have no graphs
        index_pairs = [p for p in index_pairs if p.G_before.number_of_nodes() > 0]
        query_pairs = [p for p in query_pairs if p.G_before.number_of_nodes() > 0]
        print(f"  After graph filter: index={len(index_pairs)}  query={len(query_pairs)}")
        
        return {
            "pairs": all_file_pairs,
            "index_pairs": index_pairs,
            "query_pairs": query_pairs,
            "split_info": split_info,
            "sample_info": sample_info,
        }


    def after_run(self, output: ExperimentOutput) -> None:
        """Generate dashboard after run."""
        super().after_run(output)
        
        try:
            from experiments.dashboard_scripts.dashboard import generate_html_dashboard
            generate_html_dashboard(str(output.run_dir))
            print(f"Dashboard -> {output.run_dir / 'dashboard.html'}")
        except Exception as exc:
            print(f"Dashboard generation skipped: {exc}")


def _prepare_cfg(
    cfg: dict,
    *,
    db_path: str,
    graphml_root: str,
    embedders: list[str],
    target_cwes: list[str] | None,
    ks: list[int],
) -> dict:
    cfg.setdefault("data", {})
    cfg["data"]["active"] = ["cvefixes"]
    cfg["data"].setdefault("cvefixes", {})
    cfg["data"]["cvefixes"]["db_path"] = db_path
    cfg["data"]["cvefixes"]["graphml_root"] = graphml_root

    if target_cwes is not None:
        cfg["data"]["cvefixes"]["target_cwes"] = target_cwes

    cfg.setdefault("embeddings", {})
    cfg["embeddings"]["active"] = embedders

    cfg.setdefault("experiment", {})
    cfg["experiment"]["graph_variants"] = ["G_vuln"]
    cfg["experiment"]["ks"] = ks

    return cfg


def _sync_top_level_split(cfg: dict) -> None:
    """Ensure the split is enabled while preserving config values.

    Reads existing ``experiment.split`` from config (seed, test_ratio,
    min_pairs_per_cve, ...) and only forces ``enabled: True`` so the
    leakage-safe train/test split in load_data() is always applied.
    """
    cfg.setdefault("experiment", {})
    split_cfg = cfg["experiment"].setdefault("split", {})
    split_cfg.setdefault("enabled", True)
    split_cfg.setdefault("stratified", True)
    split_cfg.setdefault("seed", 42)
    split_cfg.setdefault("test_ratio", 0.2)


def run_experiment(
    *,
    config_path: str = "config.yaml",
    output_dir: str = "cvefixes_experiments/output/cvefixes_retrieval_grid_file_level",
    graphml_root: str = "graphml_fixedcves_file",
    db_path: str = "data/cvefixes/CVEfixes.db",
    embedders: list[str] | None = None,
    target_cwes: list[str] | None = None,
    ks: list[int] | None = None,
    run_leave_one_out: bool = False,
    run_self_retrieval: bool = True,
    sample_mode: str = "none",
    sample_total: int | None = None,
    min_cves_per_cwe: int = 3,
    seed: int | None = None,
) -> ExperimentOutput:
    """Run file-level retrieval experiment."""
    if embedders is None:
        embedders = ["codebert_seq", "codebert_pattern", "combined", "gin"]
    embedders = [_normalize_embedder_name(name) for name in embedders]

    unknown = [name for name in embedders if name not in EMBEDDER_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown embedders: {unknown}. Available: {sorted(EMBEDDER_REGISTRY.keys())}"
        )

    if ks is None:
        ks = [1, 5, 10]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg = _prepare_cfg(
        cfg,
        db_path=db_path,
        graphml_root=graphml_root,
        embedders=embedders,
        target_cwes=target_cwes,
        ks=ks,
    )
    _sync_top_level_split(cfg)

    if seed is not None:
        cfg["experiment"]["split"]["seed"] = int(seed)

    cfg.setdefault("experiment", {})
    cfg["experiment"]["sampling"] = {
        "mode": sample_mode,
        "total": sample_total,
        "min_cves_per_cwe": min_cves_per_cwe,
        "seed": int(cfg["experiment"].get("split", {}).get("seed", 42)),
    }

    # Keep proportional vs. balanced comparison runs in separate folders.
    if sample_mode in ("proportional", "balanced"):
        output_dir = str(Path(output_dir) / sample_mode)

    print("=" * 70)
    print("EXPERIMENT: CVEfixes File-Level Retrieval Grid")
    print("  Load files directly from database; aggregate all methods per file")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"DB: {db_path}")
    print(f"GraphML root: {graphml_root}")
    print(f"Embedders: {embedders}")
    print(f"KS: {ks}")
    print(
        f"Sampling: mode={sample_mode} total={sample_total} "
        f"min_cves_per_cwe={min_cves_per_cwe}"
    )

    exp = CVEFixesFileLevelRetrievalExperiment(
        run_leave_one_out=run_leave_one_out,
        run_self_retrieval=run_self_retrieval,
        ks=ks,
    )

    return exp.run(cfg, output_dir=Path(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="File-level CVEfixes retrieval experiment"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        default="cvefixes_experiments/output/cvefixes_retrieval_grid_file_level",
        help="Directory where run outputs are written",
    )
    parser.add_argument(
        "--graphml-root",
        default="graphml_fixedcves_file",
        help="GraphML root directory for CVEfixes exported CPGs",
    )
    parser.add_argument(
        "--db-path",
        default="data/cvefixes/CVEfixes.db",
        help="Path to CVEfixes SQLite database",
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=["codebert_seq", "codebert_pattern", "combined", "gin"],
        help="Embedders to compare",
    )
    parser.add_argument(
        "--target-cwes",
        nargs="*",
        default=None,
        help="Optional CWE filter",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Retrieval cutoffs",
    )
    parser.add_argument(
        "--run-leave-one-out",
        action="store_true",
        help="Enable expensive leave-one-out evaluation",
    )
    parser.add_argument(
        "--no-self-retrieval",
        action="store_true",
        help="Disable same-CVE self-retrieval metrics",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["none", "proportional", "balanced"],
        default="none",
        help=(
            "CVE-aware distribution sampling: 'proportional' keeps the natural "
            "CWE distribution, 'balanced' makes it ~uniform per CWE, 'none' disables. "
            "Run once per mode with the same --sample-total to compare distributions."
        ),
    )
    parser.add_argument(
        "--sample-total",
        type=int,
        default=None,
        help=(
            "Target number of pairs to sample (share across modes for a fair "
            "comparison). Default: proportional keeps all; balanced under-samples "
            "each CWE to the smallest CWE."
        ),
    )
    parser.add_argument(
        "--min-cves-per-cwe",
        type=int,
        default=3,
        help="Keep at least this many whole CVE groups per CWE so retrieval stays feasible.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override split/sampling seed (default: config experiment.split.seed).",
    )

    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        graphml_root=args.graphml_root,
        db_path=args.db_path,
        embedders=args.embedders,
        target_cwes=args.target_cwes,
        ks=args.ks,
        run_leave_one_out=args.run_leave_one_out,
        run_self_retrieval=not args.no_self_retrieval,
        sample_mode=args.sample_mode,
        sample_total=args.sample_total,
        min_cves_per_cwe=args.min_cves_per_cwe,
        seed=args.seed,
    )

