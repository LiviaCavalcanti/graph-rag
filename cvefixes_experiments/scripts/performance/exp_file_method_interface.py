#!/usr/bin/env python3
"""
Method-level vs file-level CVEfixes retrieval comparison, standardized as a
``RetrievalGridExperiment`` subclass.

Unlike ``exp_method_vs_file_level.py`` (which builds paired CPGs on the fly
with Joern), this experiment loads ALREADY-EXPORTED CPGs at both
granularities (one graphml root for method-level graphs, one for
file-level graphs) and reuses a single precomputed, CVE-aware split (as
produced by ``utils/build_balanced_split.py``: ``split_info_balanced.json``
/ ``split_info_stratified.json``) so that the SAME set of CVEs is used for
index/query at both granularities.

``level`` (method vs file) is exposed as an experiment axis alongside
embedder / graph_variant / backend, so the standard grid machinery
(metrics, dashboard, summary) applies uniformly, and an additional
``comparison.json`` with method-vs-file deltas is written after the run.

Usage:
    python -m cvefixes_experiments.scripts.performance.exp_file_method_interface \\
      --config config.yaml \\
      --method-graphml-root graphml_cvefixes_fixed \\
      --file-graphml-root graphml_fixedcves_file \\
      --precomputed-split-dir cvefixes_experiments/output/method_vs_file_level \\
      --precomputed-split-variant balanced \\
      --embedders codebert_seq codebert_pattern combined \\
      --output-dir cvefixes_experiments/output/file_method_interface
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from experiments.base import Axis, CellContext, ExperimentOutput
from experiments.exp.retrieval_experiment import (
    BACKEND_REGISTRY,
    RetrievalGridExperiment,
)
from src.data.cvefixes import CVEFixesDataset
from src.data.pipeline import load_cpg_dirs_parallel, cpg_dir_for, compute_graph_diff
from src.embeddings import REGISTRY as EMBEDDER_REGISTRY, build_embedders
from src.metrics.metrics import embedding_space_stats
from src.metrics.retrieval_eval import retrieve_all
from src.rag.utils import populate_index

_SPLIT_VARIANT_FILENAMES = {
    "balanced": "split_info_balanced.json",
    "stratified": "split_info_stratified.json",
}

def _load_precomputed_split(
    split_dir: str | Path, variant: str, all_pairs: list
) -> tuple[list, list, dict[str, Any]]:
    """Build index/query pairs from a precomputed split_info json folder.

    Only the CVE membership (index vs. query) from the spec is used — the
    spec's own entries (func_name) may be at a different granularity (e.g.
    method-level) than the file-level pairs being split here. Every whole
    CVE stays on one side, preserving the leakage-safe guarantee.
    """
    variant = variant.lower()
    fname = _SPLIT_VARIANT_FILENAMES.get(variant)
    if fname is None:
        raise ValueError(
            f"Unknown split variant '{variant}', expected one of {sorted(_SPLIT_VARIANT_FILENAMES)}"
        )

    split_path = Path(split_dir) / fname
    if not split_path.exists():
        raise FileNotFoundError(f"Precomputed split file not found: {split_path}")

    spec = json.loads(split_path.read_text(encoding="utf-8"))
    index_cves = {e["cve_id"] for e in spec.get("index", [])}
    query_cves = {e["cve_id"] for e in spec.get("query", [])}

    overlap = index_cves & query_cves
    if overlap:
        print(
            f"  [PrecomputedSplit] WARNING: {len(overlap)} CVEs appear in both "
            "index and query specs; treating as index-only."
        )
        query_cves -= overlap

    index_pairs = [p for p in all_pairs if p.cve_id in index_cves]
    query_pairs = [p for p in all_pairs if p.cve_id in query_cves]

    split_info = {
        "enabled": True,
        "source": "precomputed",
        "split_dir": str(split_dir),
        "variant": variant,
        "counts": {
            "total": len(all_pairs),
            "index_total": len(index_pairs),
            "query_total": len(query_pairs),
            "index_cve_unique": len({p.cve_id for p in index_pairs}),
            "query_cve_unique": len({p.cve_id for p in query_pairs}),
        },
        "index_cwe_dist": dict(Counter(p.cwe_id for p in index_pairs)),
        "query_cwe_dist": dict(Counter(p.cwe_id for p in query_pairs)),
    }
    return index_pairs, query_pairs, split_info


def _normalize_embedder_name(name: str) -> str:
    aliases = {
        "codebert-seq": "codebert_seq",
        "codebert-pattern": "codebert_pattern",
    }
    key = name.strip()
    return aliases.get(key, key)

def _normalize_embedder_name(name: str) -> str:
    aliases = {
        "codebert-seq": "codebert_seq",
        "codebert-pattern": "codebert_pattern",
    }
    key = name.strip()
    return aliases.get(key, key)


def _load_file_level_pairs(cvefixes_cfg: dict) -> list:
    """Load file-level pairs with graphs, from filesystem metadata or DB."""
    dataset = CVEFixesDataset(cvefixes_cfg)
    graphml_root = cvefixes_cfg.get("graphml_root", "graphml_cvefixes_file")

    metadata_files = (
        list(Path(graphml_root).glob("*/metadata.json"))
        if Path(graphml_root).exists()
        else []
    )
    if metadata_files:
        pairs = dataset.load_from_filesystem()
        print(f"  [CVEfixes/file] loaded {len(pairs)} pairs from filesystem")
        return pairs

    print(f"  [CVEfixes/file] no metadata.json found in {graphml_root}, falling back to DB query")
    pairs = dataset.load_lightweight_file_level()
    print(f"  [CVEfixes/file] loaded {len(pairs)} lightweight pairs from DB, loading graphs...")

    graph_dirs_before, graph_dirs_after = [], []
    for p in pairs:
        dir_name = p.meta["dir_name"]
        graph_dirs_before.append(cpg_dir_for(graphml_root, cve_id=dir_name, variant="original", version="before"))
        graph_dirs_after.append(cpg_dir_for(graphml_root, cve_id=dir_name, variant="original", version="after"))

    all_dirs = graph_dirs_before + graph_dirs_after
    t0 = time.perf_counter()
    loaded_graphs = load_cpg_dirs_parallel(all_dirs, max_workers=8)
    print(f"  [CVEfixes/file] loaded {len(loaded_graphs)}/{len(all_dirs)} graphs in {time.perf_counter()-t0:.1f}s")

    populated = []
    for i, p in enumerate(pairs):
        dir_b, dir_a = graph_dirs_before[i], graph_dirs_after[i]
        if dir_b in loaded_graphs and dir_a in loaded_graphs:
            g_before, g_after = loaded_graphs[dir_b], loaded_graphs[dir_a]
            if g_before.number_of_nodes() > 0 and g_after.number_of_nodes() > 0:
                p.G_before, p.G_after = g_before, g_after
                p.G_vuln = compute_graph_diff(g_before, g_after)
                populated.append(p)
    print(f"  [CVEfixes/file] {len(populated)}/{len(pairs)} pairs have valid graphs")
    return populated


class CVEFixesFilevsMethodExperiment(RetrievalGridExperiment):
    """Retrieval grid comparing method-level vs file-level CVEfixes CPGs.

    Both levels share the SAME CVE-aware index/query split, loaded from a
    precomputed ``split_info_{balanced,stratified}.json`` folder (see
    ``utils/build_balanced_split.py``). ``level`` is an axis, so the grid
    covers ``level × embedder × graph_variant × backend``.
    """

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
        return "cvefixes_file_method_interface"

    def load_data(self, cfg: dict) -> dict[str, Any]:
        """Load method-level + file-level pairs and split both by the same
        precomputed CVE membership."""
        cvefixes_cfg = cfg.get("data", {}).get("cvefixes", {})
        mvf_cfg = cfg.get("experiment", {}).get("method_vs_file", {})

        # ── method-level pairs (with graphs) ──────────────────────────
        method_cfg = {**cvefixes_cfg, "graphml_root": mvf_cfg["method_graphml_root"]}
        method_dataset = CVEFixesDataset(method_cfg)
        all_method_pairs = method_dataset.load_all()
        print(f"  [CVEfixes/method] loaded {len(all_method_pairs)} pairs with graphs")

        # ── file-level pairs (with graphs) ────────────────────────────
        file_cfg = {**cvefixes_cfg, "graphml_root": mvf_cfg["file_graphml_root"]}
        all_file_pairs = _load_file_level_pairs(file_cfg)

        self._method_dataset = method_dataset

        # ── shared precomputed split (by CVE membership) ─────────────
        precomputed_cfg = cfg.get("experiment", {}).get("precomputed_split", {})
        precomputed_dir = precomputed_cfg.get("dir")
        if not precomputed_dir:
            raise ValueError(
                "cfg['experiment']['precomputed_split']['dir'] is required for "
                f"{self.name}; run utils/build_balanced_split.py first."
            )
        variant = precomputed_cfg.get("variant", "balanced")

        method_index, method_query, method_split_info = _load_precomputed_split(
            precomputed_dir, variant, all_method_pairs
        )
        file_index, file_query, file_split_info = _load_precomputed_split(
            precomputed_dir, variant, all_file_pairs
        )

        print(
            f"  Split (method) -> index={len(method_index)}  query={len(method_query)}"
        )
        print(
            f"  Split (file)   -> index={len(file_index)}  query={len(file_query)}"
        )

        return {
            "pairs": all_method_pairs + all_file_pairs,
            "index_pairs": method_index,
            "query_pairs": method_query,
            "split_info": {"method": method_split_info, "file": file_split_info},
            "index_pairs_by_level": {"method": method_index, "file": file_index},
            "query_pairs_by_level": {"method": method_query, "file": file_query},
            "split_info_by_level": {"method": method_split_info, "file": file_split_info},
        }

    def axes(self, cfg: dict) -> list[Axis]:
        embedders = build_embedders(cfg)
        backends = list(BACKEND_REGISTRY.keys())
        graph_variants = cfg.get("experiment", {}).get("graph_variants", ["G_vuln"])
        return [
            Axis("level", ["method", "file"], description="Code granularity"),
            Axis("embedder", embedders, description="Embedding model"),
            Axis("graph_variant", graph_variants, description="Which graph to embed"),
            Axis("backend", backends, description="Vector index backend"),
        ]

    def run_cell(self, ctx: CellContext) -> dict[str, Any]:
        level = ctx.coords["level"]
        embedder = ctx.coords["embedder"]
        backend_name = ctx.coords["backend"]
        graph_variant = ctx.coords["graph_variant"]
        index_pairs = ctx.data["index_pairs_by_level"][level]
        query_pairs = ctx.data["query_pairs_by_level"][level]
        ks = self._ks or ctx.cfg.get("experiment", {}).get("ks", [1, 5, 10])

        # ── embedding (cached per level across backends) ─────────────
        cache_key = f"{level}__{embedder.name}__{graph_variant}"
        if cache_key not in ctx.cache:
            t0 = time.perf_counter()
            graphs = [getattr(p, graph_variant) for p in index_pairs]
            embeddings = embedder.embed_many(graphs)
            embed_time = time.perf_counter() - t0
            print(f"    [{level}] embedded {len(graphs)} graphs in {embed_time:.1f}s")

            stats = embedding_space_stats(embeddings)
            print(
                f"    [{level}] eff_dim={stats['effective_dim']:.1f}  "
                f"mean_sim={stats['mean_pairwise_sim']:.3f}"
            )
            ctx.cache[cache_key] = {
                "embeddings": embeddings,
                "embed_time_s": embed_time,
                "space_stats": stats,
            }

        cached = ctx.cache[cache_key]
        index_embeddings = cached["embeddings"]

        # ── build index (kept per level to avoid path collisions) ────
        t0 = time.perf_counter()
        index_dir = ctx.run_dir / "indices" / level
        index_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{embedder.name}__{graph_variant}"
        index = BACKEND_REGISTRY[backend_name](
            dim=embedder.dim,
            index_path=str(index_dir / f"{stem}__{backend_name}.index"),
            metadata_path=str(index_dir / f"{stem}__{backend_name}_meta.json"),
        )
        retriever = populate_index(
            index, index_pairs, index_embeddings, embedder.name, top_k=max(ks)
        )
        build_time = time.perf_counter() - t0

        # ── retrieve all queries (batch) ──────────────────────────────
        qr = retrieve_all(query_pairs, embedder, retriever, top_k=max(ks))

        # ── populate artifacts for MetricSpecs ────────────────────────
        ctx.artifacts["index_embeddings"] = index_embeddings
        ctx.artifacts["space_stats"] = cached["space_stats"]
        ctx.artifacts["retriever"] = retriever
        ctx.artifacts["index_metadata"] = index.metadata
        ctx.artifacts["query_results"] = qr

        return {
            "level": level,
            "embedder": embedder.name,
            "backend": backend_name,
            "graph_variant": graph_variant,
            "n_index": len(index_pairs),
            "n_query": len(query_pairs),
            "embed_time_s": cached["embed_time_s"],
            "index_build_s": round(build_time, 3),
        }

    def after_run(self, output: ExperimentOutput) -> None:
        """Generate the standard dashboard/summary, plus a method-vs-file
        comparison.json with deltas."""
        super().after_run(output)
        self._write_comparison(output)

    def _write_comparison(self, output: ExperimentOutput) -> None:
        ks = self._ks or [1, 5, 10]
        by_level: dict[str, dict[str, dict]] = {"method": {}, "file": {}}
        for cell in output.cells:
            m = cell.metrics
            level = m.get("level")
            if level in by_level and m.get("embedder"):
                by_level[level][m["embedder"]] = m

        metrics_list = [f"hit@{k}" for k in ks] + ["mrr"]
        deltas = []
        embedders_seen = sorted(set(by_level["method"]) | set(by_level["file"]))
        for emb in embedders_seen:
            m = by_level["method"].get(emb)
            f = by_level["file"].get(emb)
            if not m or not f:
                continue
            for metric in metrics_list:
                mv = (m.get("self_retrieval") or {}).get(metric, 0.0)
                fv = (f.get("self_retrieval") or {}).get(metric, 0.0)
                deltas.append({
                    "embedder": emb, "metric": metric,
                    "method": round(mv, 4), "file": round(fv, 4),
                    "delta_file_minus_method": round(fv - mv, 4),
                })
            mcwe = (m.get("cwe_recall") or {}).get("macro_avg", 0.0)
            fcwe = (f.get("cwe_recall") or {}).get("macro_avg", 0.0)
            deltas.append({
                "embedder": emb, "metric": "cwe_recall",
                "method": round(mcwe, 4), "file": round(fcwe, 4),
                "delta_file_minus_method": round(fcwe - mcwe, 4),
            })

        comparison = {
            "run_id": output.run_id,
            "description": "Method-level vs file-level CVEfixes retrieval (shared precomputed split)",
            "deltas": deltas,
        }
        comparison_path = output.run_dir / "comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2, default=str))
        print(f"\nComparison → {comparison_path}")
        print(f"  {'embedder':<16} {'metric':<12} {'method':>8} {'file':>8} {'delta':>8}")
        for d in deltas:
            arrow = "^" if d["delta_file_minus_method"] > 0 else ("v" if d["delta_file_minus_method"] < 0 else "=")
            print(
                f"  {d['embedder']:<16} {d['metric']:<12} "
                f"{d['method']:>8.3f} {d['file']:>8.3f} "
                f"{d['delta_file_minus_method']:>+8.3f} {arrow}"
            )


def _prepare_cfg(
    cfg: dict,
    *,
    db_path: str,
    method_graphml_root: str,
    file_graphml_root: str,
    embedders: list[str],
    target_cwes: list[str] | None,
    ks: list[int],
    precomputed_split_dir: str,
    precomputed_split_variant: str,
) -> dict:
    cfg.setdefault("data", {})
    cfg["data"]["active"] = ["cvefixes"]
    cfg["data"].setdefault("cvefixes", {})
    cfg["data"]["cvefixes"]["db_path"] = db_path

    if target_cwes is not None:
        cfg["data"]["cvefixes"]["target_cwes"] = target_cwes

    cfg.setdefault("embeddings", {})
    cfg["embeddings"]["active"] = embedders

    cfg.setdefault("experiment", {})
    cfg["experiment"]["graph_variants"] = ["G_vuln"]
    cfg["experiment"]["ks"] = ks
    cfg["experiment"]["method_vs_file"] = {
        "method_graphml_root": method_graphml_root,
        "file_graphml_root": file_graphml_root,
    }
    cfg["experiment"]["precomputed_split"] = {
        "dir": precomputed_split_dir,
        "variant": precomputed_split_variant,
    }

    return cfg


def run_experiment(
    *,
    config_path: str = "config.yaml",
    output_dir: str = "cvefixes_experiments/output/file_method_interface",
    method_graphml_root: str = "graphml_cvefixes_fixed",
    file_graphml_root: str = "graphml_fixedcves_file",
    db_path: str = "data/cvefixes/CVEfixes.db",
    embedders: list[str] | None = None,
    target_cwes: list[str] | None = None,
    ks: list[int] | None = None,
    precomputed_split_dir: str = "cvefixes_experiments/output/method_vs_file_level",
    precomputed_split_variant: str = "balanced",
    run_leave_one_out: bool = False,
    run_self_retrieval: bool = True,
) -> ExperimentOutput:
    """Run the method-vs-file retrieval comparison from precomputed CPGs."""
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
        method_graphml_root=method_graphml_root,
        file_graphml_root=file_graphml_root,
        embedders=embedders,
        target_cwes=target_cwes,
        ks=ks,
        precomputed_split_dir=precomputed_split_dir,
        precomputed_split_variant=precomputed_split_variant,
    )

    print("=" * 70)
    print("EXPERIMENT: CVEfixes Method-level vs File-level Retrieval")
    print("  Shared precomputed CVE-aware split across both granularities")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"DB: {db_path}")
    print(f"Method graphml root: {method_graphml_root}")
    print(f"File graphml root: {file_graphml_root}")
    print(f"Embedders: {embedders}")
    print(f"KS: {ks}")
    print(f"Precomputed split: dir={precomputed_split_dir} variant={precomputed_split_variant}")

    exp = CVEFixesFilevsMethodExperiment(
        run_leave_one_out=run_leave_one_out,
        run_self_retrieval=run_self_retrieval,
        ks=ks,
    )

    return exp.run(cfg, output_dir=Path(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Method-level vs file-level CVEfixes retrieval comparison"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        default="cvefixes_experiments/output/file_method_interface",
        help="Directory where run outputs are written",
    )
    parser.add_argument(
        "--method-graphml-root",
        default="graphml_cvefixes_fixed",
        help="GraphML root directory for method-level CVEfixes exported CPGs",
    )
    parser.add_argument(
        "--file-graphml-root",
        default="graphml_fixedcves_file",
        help="GraphML root directory for file-level CVEfixes exported CPGs",
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
        "--precomputed-split-dir",
        required=True,
        help=(
            "Path to a folder containing split_info_balanced.json / "
            "split_info_stratified.json (as produced by utils/build_balanced_split.py)."
        ),
    )
    parser.add_argument(
        "--precomputed-split-variant",
        choices=["balanced", "stratified"],
        default="balanced",
        help="Which precomputed split file to use from --precomputed-split-dir.",
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

    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        method_graphml_root=args.method_graphml_root,
        file_graphml_root=args.file_graphml_root,
        db_path=args.db_path,
        embedders=args.embedders,
        target_cwes=args.target_cwes,
        ks=args.ks,
        precomputed_split_dir=args.precomputed_split_dir,
        precomputed_split_variant=args.precomputed_split_variant,
        run_leave_one_out=args.run_leave_one_out,
        run_self_retrieval=not args.no_self_retrieval,
    )

