#!/usr/bin/env python3
"""
Experiment: CVEfixes Pipeline Verification (Retrieval Correctness)

════════════════════════════════════════════════════════════════════════
GOAL
════════════════════════════════════════════════════════════════════════
Verify that the full graph-RAG pipeline (Joern CPG → graph diff →
embedding → retrieval) correctly groups vulnerability instances by
their CWE class and CVE identity. This serves as a correctness
sanity-check: if the pipeline is working, queries should retrieve
entries of the same vulnerability (same CVE or at least same CWE).

════════════════════════════════════════════════════════════════════════
DATA
════════════════════════════════════════════════════════════════════════
Source: cvefixes_experiments/data/cvefixes_filtered_by_cwe.json
  - Only CWEs with sufficient multi-function CVEs are included
    (≥20 entries AND ≥5 CVEs with multiple affected functions)
  - Selected CWEs: CWE-20, CWE-476, CWE-362, CWE-416, CWE-787,
    CWE-400, CWE-190
  - Only entries where the CVE has ≥2 entries are kept (so same-CVE
    retrieval is possible)
  - Code size filtered: 10–120 lines (excludes trivial/huge functions)
  - Target: ~100 index + ~25 query entries

════════════════════════════════════════════════════════════════════════
PROTOCOL
════════════════════════════════════════════════════════════════════════
  1. Filter dataset to eligible entries (multi-function CVEs, balanced CWEs)
  2. Generate CPGs via Joern (before + after)
  3. Compute graph diff → G_vuln (vulnerability slice, depth=2)
  4. Split into index (80%) / query (20%), stratified by CWE
  5. Embed with each configured embedder
  6. Build flat index, retrieve top-k for each query
  7. Compute metrics:
     - hit@1, hit@5, hit@10 (same CVE in top-k)
     - MRR (mean reciprocal rank of first same-CVE result)
     - CWE recall (same CWE class in top-k)

════════════════════════════════════════════════════════════════════════
METRICS & SUCCESS CRITERIA
════════════════════════════════════════════════════════════════════════
The pipeline is "correct" if:
  - CWE recall > 0.5 (retrieved entries share vulnerability class)
  - MRR > 0.2 (same-CVE entries appear in top ranks)
  - hit@5 > 0.3 (at least some same-CVE retrieval works)

Low scores indicate: broken embeddings, degenerate graph diffs, or
a mismatch between graph structure and vulnerability semantics.

════════════════════════════════════════════════════════════════════════
USAGE
════════════════════════════════════════════════════════════════════════
    python -m cvefixes_experiments.scripts.pipeline_verification.exp_pipeline_verification [--config config.yaml]

Output: cvefixes_experiments/output/pipeline_verification/
  - results.json (dashboard-compatible)
  - dashboard.html (auto-generated if available)
  - split_info.json (reproducibility)
"""

from __future__ import annotations

import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from src.data.base import FunctionPair
from src.data.pipeline import (
    compute_graph_diff,
    load_cpg_dir,
    run_joern_export,
    write_c_file,
)
from src.embeddings import REGISTRY as EMBEDDER_REGISTRY
from src.metrics.metrics import embedding_space_stats
from src.metrics.retrieval_eval import (
    cve_retrieval_metrics,
    cwe_recall_metrics,
    retrieve_all,
)
from src.rag.hnsw import HNSWIndex
from src.rag.utils import populate_index

# ── Configuration ────────────────────────────────────────────────────

JOERN_BIN_DIR = "/home/z0050s2b/bin/joern/joern-cli"
DATA_FILE = Path("cvefixes_experiments/data/cvefixes_filtered_by_cwe.json")
OUTPUT_DIR = Path("cvefixes_experiments/output/pipeline_verification")
WORK_DIR = OUTPUT_DIR / "cpg_cache"

SEED = 42
SLICE_DEPTH = 2

# CWEs with enough multi-function CVEs for meaningful retrieval
TARGET_CWES = ["CWE-20", "CWE-476", "CWE-362", "CWE-416", "CWE-787", "CWE-400", "CWE-190"]

# Per-CWE sampling: aim for balanced representation
SAMPLES_PER_CWE = 20  # target per CWE → ~140 total → 112 index + 28 query

# Code size bounds (lines)
MIN_LINES = 10
MAX_LINES = 120

# Embedders to evaluate
EMBEDDER_NAMES = ["gin", "combined", "codebert_pattern"]

KS = [1, 5, 10]


# ── Data selection ───────────────────────────────────────────────────


def select_entries(data_file: Path, seed: int) -> list[dict]:
    """
    Select entries for the experiment:
    - Only from TARGET_CWES
    - Only CVEs that have ≥2 entries (same-CVE retrieval is possible)
    - Code size within bounds
    - Balanced sampling across CWEs
    """
    with open(data_file) as f:
        data = json.load(f)

    entries = data["entries"]
    rng = random.Random(seed)

    # Filter to target CWEs with valid code
    eligible = []
    for e in entries:
        cwe_id = e["cwe"][0]["cwe_id"]
        if cwe_id not in TARGET_CWES:
            continue
        code_before = e.get("code_before") or ""
        code_after = e.get("code_after") or ""
        if not code_before or not code_after:
            continue
        n_lines = len(code_before.split("\n"))
        if n_lines < MIN_LINES or n_lines > MAX_LINES:
            continue
        eligible.append(e)

    # Keep only entries whose CVE has ≥2 entries (so retrieval target exists)
    cve_counts = Counter(e["cve_id"] for e in eligible)
    eligible = [e for e in eligible if cve_counts[e["cve_id"]] >= 2]

    print(f"  Eligible entries (multi-function CVEs, {MIN_LINES}-{MAX_LINES} lines): {len(eligible)}")

    # Balanced sampling: up to SAMPLES_PER_CWE per CWE
    by_cwe = defaultdict(list)
    for e in eligible:
        by_cwe[e["cwe"][0]["cwe_id"]].append(e)

    selected = []
    for cwe in TARGET_CWES:
        pool = by_cwe.get(cwe, [])
        rng.shuffle(pool)
        # Take SAMPLES_PER_CWE * 1.5 to account for Joern failures
        n_take = int(SAMPLES_PER_CWE * 1.5)
        selected.extend(pool[:n_take])

    rng.shuffle(selected)
    print(f"  Selected {len(selected)} candidates across {len(TARGET_CWES)} CWEs")
    print(f"  CWE distribution: {dict(Counter(e['cwe'][0]['cwe_id'] for e in selected))}")
    return selected


# ── CPG generation ───────────────────────────────────────────────────


def generate_cpg_pair(entry: dict, work_dir: Path) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph] | None:
    """Generate before/after CPGs. Returns (G_before, G_after) or None."""
    func_name = entry.get("method_name") or "function"
    func_safe = "".join(c if c.isalnum() or c == "_" else "_" for c in func_name)

    before_dir = work_dir / "before"
    after_dir = work_dir / "after"

    try:
        src_before = write_c_file(entry["code_before"], before_dir / f"{func_safe}.cpp")
        ok = run_joern_export(JOERN_BIN_DIR, str(src_before), str(before_dir), str(before_dir / "graph"))
        if not ok:
            return None

        src_after = write_c_file(entry["code_after"], after_dir / f"{func_safe}.cpp")
        ok = run_joern_export(JOERN_BIN_DIR, str(src_after), str(after_dir), str(after_dir / "graph"))
        if not ok:
            return None

        G_before = load_cpg_dir(str(before_dir / "graph"))
        G_after = load_cpg_dir(str(after_dir / "graph"))

        if G_before.number_of_nodes() < 10 or G_after.number_of_nodes() < 10:
            return None

        return G_before, G_after
    except Exception:
        return None


# ── Build FunctionPairs ──────────────────────────────────────────────


def build_pairs(entries: list[dict], work_dir: Path) -> list[FunctionPair]:
    """Generate CPGs and build FunctionPair objects with graph diffs."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    cwe_counts = Counter()
    t_start = time.perf_counter()

    for i, entry in enumerate(entries):
        cwe_id = entry["cwe"][0]["cwe_id"]

        # Stop if we have enough for this CWE
        if cwe_counts[cwe_id] >= SAMPLES_PER_CWE:
            continue

        cve_id = entry["cve_id"]
        func_name = entry.get("method_name") or "func"
        func_safe = "".join(c if c.isalnum() or c == "_" else "_" for c in func_name)
        entry_dir = work_dir / f"{i:04d}_{cve_id}_{func_safe}"

        # Try to load from cache
        G_before, G_after = None, None
        if (entry_dir / "before" / "graph").exists() and (entry_dir / "after" / "graph").exists():
            try:
                G_before = load_cpg_dir(str(entry_dir / "before" / "graph"))
                G_after = load_cpg_dir(str(entry_dir / "after" / "graph"))
                if G_before.number_of_nodes() < 10 or G_after.number_of_nodes() < 10:
                    G_before, G_after = None, None
            except Exception:
                G_before, G_after = None, None

        # Generate fresh if not cached
        if G_before is None:
            if entry_dir.exists():
                shutil.rmtree(entry_dir)
            result = generate_cpg_pair(entry, entry_dir)
            if result is None:
                continue
            G_before, G_after = result

        # Compute graph diff (vulnerability slice)
        G_vuln = compute_graph_diff(G_before, G_after)
        if G_vuln.number_of_nodes() == 0:
            continue

        pairs.append(FunctionPair(
            cve_id=cve_id,
            cwe_id=cwe_id,
            func_name=func_name,
            project=entry.get("project", ""),
            G_before=G_before,
            G_after=G_after,
            G_vuln=G_vuln,
            meta={
                "dataset": "CVEfixes",
                "variant": "original",
                "filename": entry.get("filename", ""),
                "language": entry.get("programming_language", "C"),
                "source_before": entry.get("code_before", ""),
            },
        ))
        cwe_counts[cwe_id] += 1

        elapsed = time.perf_counter() - t_start
        print(f"  [{len(pairs):3d}] {cve_id}/{func_name} [{cwe_id}]  "
              f"nodes: {G_before.number_of_nodes()}/{G_after.number_of_nodes()} "
              f"→ slice: {G_vuln.number_of_nodes()}")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n  Built {len(pairs)} pairs in {elapsed_total:.0f}s")
    print(f"  Per-CWE: {dict(cwe_counts)}")
    return pairs


# ── Train/test split ─────────────────────────────────────────────────


def stratified_split(pairs: list[FunctionPair], test_ratio: float, seed: int):
    """
    Stratified split ensuring:
    - Each CWE has representation in both index and query
    - Query entries always have at least one same-CVE entry in index
    """
    rng = random.Random(seed)

    # Group by CWE
    by_cwe = defaultdict(list)
    for p in pairs:
        by_cwe[p.cwe_id].append(p)

    # Group by CVE within each CWE
    query_pairs = []
    index_pairs = []

    for cwe, cwe_entries in by_cwe.items():
        by_cve = defaultdict(list)
        for p in cwe_entries:
            by_cve[p.cve_id].append(p)

        # For multi-entry CVEs: put some in query, rest in index
        # For single-entry CVEs: only index (can't do same-CVE retrieval)
        cwe_query = []
        cwe_index = []

        for cve_id, cve_entries in by_cve.items():
            rng.shuffle(cve_entries)
            if len(cve_entries) >= 2:
                # Put 1 in query, rest in index
                n_query = max(1, int(len(cve_entries) * test_ratio))
                cwe_query.extend(cve_entries[:n_query])
                cwe_index.extend(cve_entries[n_query:])
            else:
                # Single entry → index only
                cwe_index.extend(cve_entries)

        query_pairs.extend(cwe_query)
        index_pairs.extend(cwe_index)

    rng.shuffle(query_pairs)
    rng.shuffle(index_pairs)

    return index_pairs, query_pairs


# ── Main experiment ──────────────────────────────────────────────────


def run_experiment(cfg_path: str = "config.yaml"):
    """Run the full pipeline verification experiment."""
    import yaml

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print("EXPERIMENT: CVEfixes Pipeline Verification")
    print("  Verifying that graph-RAG retrieves same-CVE / same-CWE entries")
    print("=" * 70)

    # 1. Select data
    print(f"\n[1/5] Selecting entries from {DATA_FILE}...")
    entries = select_entries(DATA_FILE, SEED)

    # 2. Generate CPGs and build pairs
    print(f"\n[2/5] Building CPGs and graph diffs (slice depth={SLICE_DEPTH})...")
    pairs = build_pairs(entries, WORK_DIR)

    if len(pairs) < 20:
        print(f"FATAL: only {len(pairs)} pairs — need at least 20")
        return

    # 3. Split
    print(f"\n[3/5] Splitting into index/query (stratified by CWE)...")
    index_pairs, query_pairs = stratified_split(pairs, test_ratio=0.2, seed=SEED)

    # Verify query entries have support in index
    index_cves = set(p.cve_id for p in index_pairs)
    query_pairs = [p for p in query_pairs if p.cve_id in index_cves]

    print(f"  Index: {len(index_pairs)} entries")
    print(f"  Query: {len(query_pairs)} entries (all have same-CVE support in index)")
    print(f"  Index CWE dist: {dict(Counter(p.cwe_id for p in index_pairs))}")
    print(f"  Query CWE dist: {dict(Counter(p.cwe_id for p in query_pairs))}")

    split_info = {
        "seed": SEED,
        "n_index": len(index_pairs),
        "n_query": len(query_pairs),
        "index_cwe_dist": dict(Counter(p.cwe_id for p in index_pairs)),
        "query_cwe_dist": dict(Counter(p.cwe_id for p in query_pairs)),
        "index_cve_unique": len(set(p.cve_id for p in index_pairs)),
        "query_cve_unique": len(set(p.cve_id for p in query_pairs)),
    }

    # 4. Embed + retrieve for each embedder
    print(f"\n[4/5] Running retrieval for {len(EMBEDDER_NAMES)} embedders...")
    emb_cfg = cfg.get("embeddings", {})
    cells = []

    for emb_name in EMBEDDER_NAMES:
        if emb_name not in EMBEDDER_REGISTRY:
            print(f"  WARNING: embedder '{emb_name}' not in registry, skipping")
            continue

        embedder = EMBEDDER_REGISTRY[emb_name](emb_cfg)
        print(f"\n  ── {embedder.name} ──")

        # Embed index
        t0 = time.perf_counter()
        index_graphs = [p.G_vuln for p in index_pairs]
        index_embeddings = embedder.embed_many(index_graphs)
        embed_time = time.perf_counter() - t0
        print(f"    Embedded {len(index_graphs)} graphs in {embed_time:.1f}s")

        # Check for degenerate embeddings
        norms = np.linalg.norm(index_embeddings, axis=1)
        n_zero = int(np.sum(norms < 1e-6))
        if n_zero == len(index_embeddings):
            print(f"    SKIP — all embeddings are zero")
            continue

        space_stats = embedding_space_stats(index_embeddings)
        print(f"    eff_dim={space_stats['effective_dim']:.1f}  "
              f"mean_sim={space_stats['mean_pairwise_sim']:.3f}")

        # Build index
        dim = index_embeddings.shape[1]
        index_dir = OUTPUT_DIR / "indices"
        index_dir.mkdir(parents=True, exist_ok=True)
        index = HNSWIndex(
            dim=dim,
            index_path=str(index_dir / f"{embedder.name}__hnsw.index"),
            metadata_path=str(index_dir / f"{embedder.name}__hnsw_meta.json"),
        )
        retriever = populate_index(
            index, index_pairs, index_embeddings, embedder.name, top_k=max(KS)
        )

        # Retrieve
        qr = retrieve_all(query_pairs, embedder, retriever, top_k=max(KS))
        print(f"    Retrieved for {len(qr)} queries")

        # Compute metrics
        cve_metrics = cve_retrieval_metrics(qr, ks=KS, index_metadata=index.metadata)
        cwe_metrics = cwe_recall_metrics(qr, index.metadata, top_k=max(KS))

        # CWE hit@k: binary — did ANY same-CWE entry appear in top-k?
        cwe_hit = {}
        for k in KS:
            hits = 0
            total = 0
            for pair, results in qr:
                cwe = pair.cwe_id
                if not cwe or cwe == "UNKNOWN":
                    continue
                total += 1
                if any(r.get("cwe_id") == cwe for r in results[:k]):
                    hits += 1
            cwe_hit[k] = hits / total if total > 0 else 0.0

        hit1 = cve_metrics.get("hit@1", 0)
        hit5 = cve_metrics.get("hit@5", 0)
        hit10 = cve_metrics.get("hit@10", 0)
        mrr = cve_metrics.get("mrr", 0)
        cwe_recall = cwe_metrics.get("macro_avg", 0)

        print(f"    hit@1={hit1:.3f}  hit@5={hit5:.3f}  hit@10={hit10:.3f}  "
              f"MRR={mrr:.3f}  CWE_recall={cwe_recall:.3f}")
        print(f"    CWE_hit@1={cwe_hit.get(1,0):.3f}  "
              f"CWE_hit@5={cwe_hit.get(5,0):.3f}  "
              f"CWE_hit@10={cwe_hit.get(10,0):.3f}")

        cells.append({
            "embedder": embedder.name,
            "backend": "hnsw",
            "graph_variant": "G_vuln",
            "n_samples": len(index_pairs),
            "embed_time_s": round(embed_time, 2),
            "space_stats": space_stats,
            "self_retrieval": cve_metrics,
            "cwe_recall": cwe_metrics,
            "cwe_hit": {f"hit@{k}": v for k, v in cwe_hit.items()},
        })

    # 5. Save results
    print(f"\n[5/5] Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "run_id": f"pipeline_verification_{SEED}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "CVEfixes pipeline verification — same-CVE/CWE retrieval correctness check",
        "config": {
            "seed": SEED,
            "slice_depth": SLICE_DEPTH,
            "target_cwes": TARGET_CWES,
            "samples_per_cwe": SAMPLES_PER_CWE,
            "min_lines": MIN_LINES,
            "max_lines": MAX_LINES,
            "embedders": EMBEDDER_NAMES,
            "ks": KS,
        },
        "dataset_info": {
            "source": str(DATA_FILE),
            "n_pairs_total": len(pairs),
            "split": split_info,
        },
        "cells": cells,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results → {results_path}")

    # Save split info for reproducibility
    split_path = OUTPUT_DIR / "split_info.json"
    with open(split_path, "w") as f:
        json.dump({
            **split_info,
            "index_entries": [{"cve_id": p.cve_id, "cwe_id": p.cwe_id, "func_name": p.func_name} for p in index_pairs],
            "query_entries": [{"cve_id": p.cve_id, "cwe_id": p.cwe_id, "func_name": p.func_name} for p in query_pairs],
        }, f, indent=2)
    print(f"  Split info → {split_path}")

    # Generate dashboard if available
    try:
        from experiments.dashboard_scripts.dashboard import generate_html_dashboard
        generate_html_dashboard(str(OUTPUT_DIR))
        print(f"  Dashboard → {OUTPUT_DIR / 'dashboard.html'}")
    except Exception as e:
        print(f"  Dashboard generation skipped: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for cell in cells:
        sr = cell["self_retrieval"]
        cwe = cell["cwe_recall"]
        ch = cell.get("cwe_hit", {})
        print(f"  {cell['embedder']:<20}  "
              f"hit@1={sr.get('hit@1',0):.3f}  "
              f"hit@5={sr.get('hit@5',0):.3f}  "
              f"hit@10={sr.get('hit@10',0):.3f}  "
              f"MRR={sr.get('mrr',0):.3f}  "
              f"CWE_recall={cwe.get('macro_avg',0):.3f}  "
              f"CWE_hit@5={ch.get('hit@5',0):.3f}")

    # Correctness verdict
    if cells:
        best_mrr = max(c["self_retrieval"].get("mrr", 0) for c in cells)
        best_cwe = max(c["cwe_recall"].get("macro_avg", 0) for c in cells)
        best_hit5 = max(c["self_retrieval"].get("hit@5", 0) for c in cells)
        print(f"\n  Best MRR:        {best_mrr:.3f} {'✓' if best_mrr > 0.2 else '✗'} (threshold: 0.2)")
        print(f"  Best CWE recall: {best_cwe:.3f} {'✓' if best_cwe > 0.5 else '✗'} (threshold: 0.5)")
        print(f"  Best hit@5:      {best_hit5:.3f} {'✓' if best_hit5 > 0.3 else '✗'} (threshold: 0.3)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CVEfixes pipeline verification experiment")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    args = parser.parse_args()
    run_experiment(cfg_path=args.config)
