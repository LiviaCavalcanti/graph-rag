#!/usr/bin/env python3
"""
Structured CVEfixes retrieval experiment on graphml_selected_cves.

Compares retrieval performance for:
  - codebert_seq
  - codebert_pattern
  - combined

This script is a reproducible wrapper around experiments.exp.retrieval_experiment
with a CVE/CWE-aware split to avoid query/index leakage when using pure
CVEfixes original graphs.

Usage:
    python -m cvefixes_experiments.scripts.pipeline_verification.exp_cvefixes_retrieval_grid \
      --config config.yaml \
      --output-dir cvefixes_experiments/output/cvefixes_retrieval_grid
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from experiments.exp.retrieval_experiment import RetrievalGridExperiment
from experiments.base import ExperimentOutput
from src.data import load_pairs
from src.embeddings import REGISTRY as EMBEDDER_REGISTRY


def _normalize_embedder_name(name: str) -> str:
    aliases = {
        "codebert-seq": "codebert_seq",
        "codebert-pattern": "codebert_pattern",
    }
    key = name.strip()
    return aliases.get(key, key)


def _sample_per_cwe(pairs: list, max_per_cwe: int, seed: int) -> list:
    if max_per_cwe <= 0:
        return pairs

    rng = random.Random(seed)
    by_cwe: dict[str, list] = defaultdict(list)
    for pair in pairs:
        by_cwe[getattr(pair, "cwe_id", "UNKNOWN") or "UNKNOWN"].append(pair)

    sampled: list = []
    for cwe_id in sorted(by_cwe):
        items = by_cwe[cwe_id][:]
        rng.shuffle(items)
        sampled.extend(items[:max_per_cwe])

    rng.shuffle(sampled)
    return sampled


def _split_cvefixes_pairs(
    pairs: list,
    *,
    test_ratio: float,
    seed: int,
    min_pairs_per_cve: int = 2,
) -> tuple[list, list, dict[str, Any]]:
    """Split pairs so every query has same-CVE support in index.

    Strategy:
      - Group by CWE, then by CVE.
      - For CVEs with >= min_pairs_per_cve entries: move a slice to query.
      - Keep at least one entry from each such CVE in index.
      - Singletons stay in index only.
    """
    rng = random.Random(seed)
    ratio = max(0.0, min(0.9, float(test_ratio)))

    by_cwe: dict[str, list] = defaultdict(list)
    for pair in pairs:
        cwe = getattr(pair, "cwe_id", "UNKNOWN") or "UNKNOWN"
        by_cwe[cwe].append(pair)

    index_pairs: list = []
    query_pairs: list = []

    for cwe in sorted(by_cwe):
        cwe_pairs = by_cwe[cwe]
        by_cve: dict[str, list] = defaultdict(list)
        for pair in cwe_pairs:
            by_cve[getattr(pair, "cve_id", "UNKNOWN") or "UNKNOWN"].append(pair)

        for cve_id in sorted(by_cve):
            cve_pairs = by_cve[cve_id][:]
            rng.shuffle(cve_pairs)

            if len(cve_pairs) >= min_pairs_per_cve:
                n_query = int(round(len(cve_pairs) * ratio))
                n_query = max(1, min(len(cve_pairs) - 1, n_query))
                query_pairs.extend(cve_pairs[:n_query])
                index_pairs.extend(cve_pairs[n_query:])
            else:
                index_pairs.extend(cve_pairs)

    # Safety filter: remove any query CVE missing from index.
    index_cves = {p.cve_id for p in index_pairs}
    query_pairs = [p for p in query_pairs if p.cve_id in index_cves]

    # Fallback to avoid empty query in tiny subsets.
    if not query_pairs:
        by_cve_all: dict[str, list] = defaultdict(list)
        for pair in pairs:
            by_cve_all[pair.cve_id].append(pair)
        for cve_id, items in by_cve_all.items():
            if len(items) >= min_pairs_per_cve:
                items = items[:]
                rng.shuffle(items)
                q = items[0]
                query_pairs = [q]
                index_pairs = [p for p in pairs if p is not q]
                break

    rng.shuffle(index_pairs)
    rng.shuffle(query_pairs)

    split_info = {
        "enabled": True,
        "seed": seed,
        "test_ratio": ratio,
        "strategy": "by_cwe_then_cve_with_same_cve_support",
        "min_pairs_per_cve": min_pairs_per_cve,
        "counts": {
            "total": len(pairs),
            "index_total": len(index_pairs),
            "query_total": len(query_pairs),
            "index_cve_unique": len({p.cve_id for p in index_pairs}),
            "query_cve_unique": len({p.cve_id for p in query_pairs}),
        },
        "index_cwe_dist": dict(Counter(p.cwe_id for p in index_pairs)),
        "query_cwe_dist": dict(Counter(p.cwe_id for p in query_pairs)),
    }

    return index_pairs, query_pairs, split_info


class CVEFixesRetrievalComparisonExperiment(RetrievalGridExperiment):
    """Retrieval grid for CVEfixes with leakage-safe split and dashboard export."""

    def __init__(
        self,
        *,
        split_seed: int,
        test_ratio: float,
        min_pairs_per_cve: int,
        max_pairs_per_cwe: int,
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
        self._split_seed = split_seed
        self._test_ratio = test_ratio
        self._min_pairs_per_cve = min_pairs_per_cve
        self._max_pairs_per_cwe = max_pairs_per_cwe

    @property
    def name(self) -> str:
        return "cvefixes_retrieval_grid"

    def load_data(self, cfg: dict) -> dict[str, Any]:
        pairs = load_pairs(cfg)

        # Keep only pairs with non-empty vulnerability slices.
        pairs = [p for p in pairs if getattr(p, "G_vuln", None) and p.G_vuln.number_of_nodes() > 0]

        if self._max_pairs_per_cwe > 0:
            pairs = _sample_per_cwe(pairs, self._max_pairs_per_cwe, self._split_seed)

        index_pairs, query_pairs, split_info = _split_cvefixes_pairs(
            pairs,
            test_ratio=self._test_ratio,
            seed=self._split_seed,
            min_pairs_per_cve=self._min_pairs_per_cve,
        )

        print(f"  [CVEfixes] loaded {len(pairs)} usable pairs")
        print(f"  Split -> index={len(index_pairs)}  query={len(query_pairs)}")
        print(f"  Index CWE dist: {dict(Counter(p.cwe_id for p in index_pairs))}")
        print(f"  Query CWE dist: {dict(Counter(p.cwe_id for p in query_pairs))}")

        return {
            "pairs": pairs,
            "index_pairs": index_pairs,
            "query_pairs": query_pairs,
            "split_info": split_info,
        }

    def after_run(self, output: ExperimentOutput) -> None:
        # Keep base behavior: results.json + visualizations + summary.json/all_runs.json
        super().after_run(output)

        # Generate the unified HTML dashboard used by pipeline_verification.
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


def _sync_top_level_split(cfg: dict, seed: int, test_ratio: float) -> None:
    """Keep split metadata visible in dashboard dataset_info fields."""
    cfg.setdefault("experiment", {})
    cfg["experiment"]["split"] = {
        "enabled": True,
        "stratified": True,
        "seed": int(seed),
        "test_ratio": float(test_ratio),
        "include_real_in_index": True,
        "augmented_train_ratio": 1.0,
        "query_source": "cve_aware_query",
    }


def run_experiment(
    *,
    config_path: str = "config.yaml",
    output_dir: str = "cvefixes_experiments/output/cvefixes_retrieval_grid",
    graphml_root: str = "graphml_selected_cves",
    db_path: str = "data/cvefixes/CVEfixes.db",
    embedders: list[str] | None = None,
    target_cwes: list[str] | None = None,
    ks: list[int] | None = None,
    seed: int = 42,
    test_ratio: float = 0.2,
    min_pairs_per_cve: int = 5,
    max_pairs_per_cwe: int = 20,
    run_leave_one_out: bool = False,
    run_self_retrieval: bool = True,
) -> ExperimentOutput:
    if embedders is None:
        embedders = ["codebert_seq", "codebert_pattern", "combined"]
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
    _sync_top_level_split(cfg, seed=seed, test_ratio=test_ratio)

    print("=" * 70)
    print("EXPERIMENT: CVEfixes Retrieval Grid (Structured)")
    print("  Compare codebert_seq vs codebert_pattern vs combined")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"DB: {db_path}")
    print(f"GraphML root: {graphml_root}")
    print(f"Embedders: {embedders}")
    print(f"KS: {ks}")
    print(f"Split seed={seed}, test_ratio={test_ratio}")

    exp = CVEFixesRetrievalComparisonExperiment(
        split_seed=seed,
        test_ratio=test_ratio,
        min_pairs_per_cve=min_pairs_per_cve,
        max_pairs_per_cwe=max_pairs_per_cwe,
        run_leave_one_out=run_leave_one_out,
        run_self_retrieval=run_self_retrieval,
        ks=ks,
    )

    return exp.run(cfg, output_dir=Path(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Structured CVEfixes retrieval experiment on graphml_selected_cves"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        default="cvefixes_experiments/output/cvefixes_retrieval_grid",
        help="Directory where run outputs are written",
    )
    parser.add_argument(
        "--graphml-root",
        default="graphml_selected_cves",
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
        default=["codebert_seq", "codebert_pattern", "combined"],
        help="Embedders to compare (supports codebert-seq/codebert-pattern aliases)",
    )
    parser.add_argument(
        "--target-cwes",
        nargs="*",
        default=None,
        help="Optional CWE filter, e.g. CWE-20 CWE-476 CWE-787",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Retrieval cutoffs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split random seed")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of each multi-entry CVE assigned to query",
    )
    parser.add_argument(
        "--min-pairs-per-cve",
        type=int,
        default=5,
        help="Minimum entries per CVE to allow query allocation",
    )
    parser.add_argument(
        "--max-pairs-per-cwe",
        type=int,
        default=20,
        help="Optional cap per CWE before splitting (0 = no cap)",
    )
    parser.add_argument(
        "--run-leave-one-out",
        action="store_true",
        help="Enable expensive leave-one-out evaluation for small index sets",
    )
    parser.add_argument(
        "--no-self-retrieval",
        action="store_true",
        help="Disable same-CVE self-retrieval metrics",
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
        seed=args.seed,
        test_ratio=args.test_ratio,
        min_pairs_per_cve=args.min_pairs_per_cve,
        max_pairs_per_cwe=args.max_pairs_per_cwe,
        run_leave_one_out=args.run_leave_one_out,
        run_self_retrieval=not args.no_self_retrieval,
    )
