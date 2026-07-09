#!/usr/bin/env python3
"""
Experiment: Method-level vs File-level input comparison (SAME CVEs).

════════════════════════════════════════════════════════════════════════
GOAL
════════════════════════════════════════════════════════════════════════
Measure how retrieval performance changes when the CPG input granularity
switches from METHOD-level code to FILE-level code, holding the CVE/CWE
sample set constant.

Both granularities are built from the SAME pinned entries
(experiments_cves/selected_entries.json — the same selection used by the
pipeline-verification baseline, seed=42). Each entry carries both the
method body (code_before/after) and the full file (file_code_before/after),
so the ONLY thing that varies between the two runs is the input granularity.

════════════════════════════════════════════════════════════════════════
PROTOCOL
════════════════════════════════════════════════════════════════════════
  1. Load pinned entries (same CVEs as the baseline).
  2. For every entry build TWO CPG pairs:
       - method: Joern on the method body      → G_vuln(method)
       - file:   Joern on the whole file        → G_vuln(file)
     Existing CPG caches are reused when present; missing method CPGs are
     generated with Joern (file CPGs are already fully cached).
  3. Keep only entries that produced valid graphs at BOTH levels
     (identical instance set → fair paired comparison).
  4. Split ONCE on the shared (cve, func) instances (stratified by CWE,
     same-CVE support guaranteed, seed=42) and apply the SAME
     index/query partition to both levels.
  5. Run the same retrieval grid (HNSW) for each level and each embedder.
  6. Report method-vs-file metrics side by side with deltas.

════════════════════════════════════════════════════════════════════════
USAGE
════════════════════════════════════════════════════════════════════════
    python -m cvefixes_experiments.scripts.pipeline_verification.exp_method_vs_file_level \
      --config config.yaml \
      --embedders wl combined codebert_seq \
      --output-dir cvefixes_experiments/output/method_vs_file_level

    # Instant run using only already-cached CPGs (no Joern):
    python -m ...exp_method_vs_file_level --reuse-only

Output (in --output-dir):
  - comparison.json        method vs file metrics + deltas
  - method/results.json    per-level dashboard-compatible results
  - file/results.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from src.data.base import FunctionPair
from src.data.pipeline import (
    compute_graph_diff,
    load_cpg_dir,
    run_joern_export,
    write_c_file,
)
from src.embeddings import REGISTRY as EMBEDDER_REGISTRY
from src.metrics.metrics import embedding_space_stats
from src.metrics.retrieval_eval import cve_retrieval_metrics, cwe_recall_metrics
from src.rag.hnsw import HNSWIndex
from src.rag.utils import populate_index


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_ENTRIES = Path("experiments_cves/selected_entries.json")
DEFAULT_OUTPUT_DIR = Path("cvefixes_experiments/output/method_vs_file_level")
DEFAULT_JOERN_BIN = "/home/z0050s2b/bin/joern/joern-cli"

# Existing CPG caches to reuse (built by exp_pipeline_verification.py).
FILE_CACHES = [
    Path("cvefixes_experiments/output/"
         "pipeline_verification_structure_file_ctxt_combined_code/cpg_cache"),
]
METHOD_CACHES = [
    Path("cvefixes_experiments/output/pipeline_verification_method_ctxt/cpg_cache"),
]

# Source data (with BOTH method and file code) used to recover code for
# instances anchored from a baseline split_info.json.
DEFAULT_SOURCE = Path("cvefixes_experiments/data/cvefixes_filtered_by_cwe_with_files.json")

MIN_GRAPH_NODES = 10
DEFAULT_EMBEDDERS = ["wl", "combined", "codebert_seq"]
DEFAULT_KS = [1, 5, 10]
DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.2


# ── Helpers ──────────────────────────────────────────────────────────


def _func_safe(name: str | None) -> str:
    """Sanitize a function name the same way the cache builder did."""
    src = name or "func"
    return "".join(c if c.isalnum() or c == "_" else "_" for c in src)


def _entry_key(entry: dict) -> tuple[str, str]:
    """Stable identity for a vulnerability instance: (cve_id, safe func)."""
    return (entry["cve_id"], _func_safe(entry.get("method_name")))


def _cwe_of(entry: dict) -> str:
    cwe = entry.get("cwe")
    if isinstance(cwe, list) and cwe:
        return cwe[0].get("cwe_id", "UNKNOWN")
    return entry.get("cwe_id", "UNKNOWN")


def _build_cache_map(cache_dir: Path) -> dict[tuple[str, str], Path]:
    """Map (cve_id, func_safe) -> entry dir for an existing CPG cache.

    Cache dir names look like ``{i:04d}_{cve_id}_{func_safe}`` where cve_id
    is hyphenated (e.g. CVE-2018-8788) so the FIRST underscore after the
    index separates the CVE from the function name.
    """
    mapping: dict[tuple[str, str], Path] = {}
    if not cache_dir.exists():
        return mapping
    pattern = re.compile(r"^\d+_(CVE-\d+-\d+)_(.*)$")
    for child in cache_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m:
            continue
        mapping.setdefault((m.group(1), m.group(2)), child)
    return mapping


def _build_cache_maps(dirs: list[Path]) -> dict[tuple[str, str], Path]:
    """Merge cache maps from several cache directories (first hit wins)."""
    merged: dict[tuple[str, str], Path] = {}
    for d in dirs:
        for k, v in _build_cache_map(Path(d)).items():
            merged.setdefault(k, v)
    return merged


def _load_pair_graphs(entry_dir: Path):
    """Load (G_before, G_after) from a cached entry dir, or None on failure."""
    before = entry_dir / "before" / "graph"
    after = entry_dir / "after" / "graph"
    if not before.exists() or not after.exists():
        return None
    try:
        g_before = load_cpg_dir(str(before))
        g_after = load_cpg_dir(str(after))
    except Exception:
        return None
    if g_before.number_of_nodes() < MIN_GRAPH_NODES or g_after.number_of_nodes() < MIN_GRAPH_NODES:
        return None
    return g_before, g_after


def _joern_build(code_before: str, code_after: str, out_dir: Path, func_safe: str, joern_bin: str):
    """Generate before/after CPGs with Joern into out_dir; return graphs or None."""
    before_dir = out_dir / "before"
    after_dir = out_dir / "after"
    try:
        src_before = write_c_file(code_before, before_dir / f"{func_safe}.cpp")
        if not run_joern_export(joern_bin, str(src_before), str(before_dir), str(before_dir / "graph")):
            return None
        src_after = write_c_file(code_after, after_dir / f"{func_safe}.cpp")
        if not run_joern_export(joern_bin, str(src_after), str(after_dir), str(after_dir / "graph")):
            return None
        return _load_pair_graphs(out_dir)
    except Exception:
        return None


def _make_pair(entry: dict, g_before, g_after, level: str) -> FunctionPair:
    return FunctionPair(
        cve_id=entry["cve_id"],
        cwe_id=_cwe_of(entry),
        func_name=entry.get("method_name") or "func",
        project=entry.get("project", ""),
        G_before=g_before,
        G_after=g_after,
        G_vuln=compute_graph_diff(g_before, g_after),
        meta={
            "dataset": "CVEfixes",
            "level": level,
            "variant": "original",
            "filename": entry.get("filename", ""),
            "language": entry.get("programming_language", "C"),
            "file_change_id": entry.get("file_change_id"),
        },
    )


# ── Paired build (both granularities) ────────────────────────────────


def build_paired_pairs(
    entries: list[dict],
    *,
    output_dir: Path,
    joern_bin: str,
    reuse_only: bool,
    max_workers: int = 8,
) -> dict[tuple[str, str], dict[str, FunctionPair]]:
    """Build method + file CPG pairs for each entry.

    Returns {key: {"method": FunctionPair, "file": FunctionPair}} containing
    ONLY entries that produced valid graphs at both levels.
    """
    file_map = _build_cache_maps(FILE_CACHES)
    method_map = _build_cache_maps(METHOD_CACHES)
    method_build_dir = output_dir / "cpg_cache_method"
    file_build_dir = output_dir / "cpg_cache_file"
    method_build_dir.mkdir(parents=True, exist_ok=True)
    file_build_dir.mkdir(parents=True, exist_ok=True)

    print(f"  File cache entries:   {len(file_map)}")
    print(f"  Method cache entries: {len(method_map)}")

    def _resolve(entry, cache_map, build_dir, code_before, code_after):
        """Reuse a cached CPG for (cve,func) at this level, else Joern-build."""
        key = _entry_key(entry)
        cve_id, func_safe = key
        graphs = None
        cached = cache_map.get(key)
        if cached is not None:
            graphs = _load_pair_graphs(cached)
        if graphs is None:
            fresh_dir = build_dir / f"{cve_id}_{func_safe}"
            if (fresh_dir / "before" / "graph").exists():
                graphs = _load_pair_graphs(fresh_dir)
            if graphs is None and not reuse_only and code_before and code_after:
                if fresh_dir.exists():
                    shutil.rmtree(fresh_dir, ignore_errors=True)
                graphs = _joern_build(code_before, code_after, fresh_dir, func_safe, joern_bin)
        return graphs

    def process(entry: dict):
        key = _entry_key(entry)

        # file level: reuse cache, else Joern-build from FILE code
        file_graphs = _resolve(
            entry, file_map, file_build_dir,
            entry.get("file_code_before") or "", entry.get("file_code_after") or "",
        )
        if file_graphs is None:
            return None

        # method level: reuse cache, else Joern-build from METHOD code
        method_graphs = _resolve(
            entry, method_map, method_build_dir,
            entry.get("code_before") or "", entry.get("code_after") or "",
        )
        if method_graphs is None:
            return None

        try:
            m_pair = _make_pair(entry, method_graphs[0], method_graphs[1], "method")
            f_pair = _make_pair(entry, file_graphs[0], file_graphs[1], "file")
        except Exception:
            return None
        if m_pair.G_vuln.number_of_nodes() == 0 or f_pair.G_vuln.number_of_nodes() == 0:
            return None
        return key, {"method": m_pair, "file": f_pair}

    paired: dict[tuple[str, str], dict[str, FunctionPair]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, e): e for e in entries}
        for fut in tqdm(as_completed(futures), total=len(entries), desc="Building method+file pairs"):
            res = fut.result()
            if res is not None:
                key, pairs = res
                paired[key] = pairs
    return paired


# ── Split (shared across levels) ─────────────────────────────────────


def stratified_split_keys(
    entities: list[dict], test_ratio: float, seed: int
) -> tuple[set, set]:
    """Split (cve, func) entities into index/query keys.

    Stratified by CWE; multi-entry CVEs contribute queries while keeping
    same-CVE support in the index. Singleton CVEs stay index-only.
    """
    rng = random.Random(seed)
    by_cwe: dict[str, list[dict]] = defaultdict(list)
    for ent in entities:
        by_cwe[ent["cwe_id"]].append(ent)

    index_keys: set = set()
    query_keys: set = set()
    for cwe_entities in by_cwe.values():
        by_cve: dict[str, list[dict]] = defaultdict(list)
        for ent in cwe_entities:
            by_cve[ent["cve_id"]].append(ent)
        for cve_entities in by_cve.values():
            items = cve_entities[:]
            rng.shuffle(items)
            if len(items) >= 2:
                n_query = max(1, int(len(items) * test_ratio))
                for ent in items[:n_query]:
                    query_keys.add(ent["key"])
                for ent in items[n_query:]:
                    index_keys.add(ent["key"])
            else:
                index_keys.add(items[0]["key"])

    # Keep only queries whose CVE still has support in the index.
    index_cves = {k[0] for k in index_keys}
    query_keys = {k for k in query_keys if k[0] in index_cves}
    return index_keys, query_keys


def _sample_paired_cve_aware(
    paired: dict[tuple[str, str], dict[str, FunctionPair]],
    *,
    mode: str,
    seed: int,
    total: int | None = None,
    min_cves_per_cwe: int = 3,
) -> tuple[dict[tuple[str, str], dict[str, FunctionPair]], dict[str, Any]]:
    """Reshape the paired instance set to a proportional/balanced CWE mix.

    Applied to the PAIRED dict, so the method and file levels use the identical
    reshaped sample. Whole CVE groups are kept together (a CVE never straddles
    the sample boundary) so the downstream stratified split can still find
    same-CVE support. ``balanced`` caps every CWE to the same budget (a
    ~uniform CWE mix); ``proportional`` keeps the natural CWE proportions.
    """
    if mode not in ("proportional", "balanced"):
        return paired, {"mode": mode or "none", "applied": False}

    rng = random.Random(seed)
    by_cwe: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for key, pair in paired.items():
        cwe = pair["file"].cwe_id or "UNKNOWN"
        by_cwe[cwe][key[0]].append(key)  # key[0] = cve_id

    cwes = sorted(by_cwe)
    n_cwes = max(1, len(cwes))
    cwe_pair_counts = {c: sum(len(v) for v in by_cwe[c].values()) for c in cwes}
    total_pairs = sum(cwe_pair_counts.values()) or 1

    if mode == "balanced":
        # Uniform CWE mix: cap every CWE to the same budget. When the shared
        # ``total`` budget (calibrated for a larger pool) does not bind, fall
        # back to the smallest CWE's availability so the result is genuinely
        # uniform rather than the natural (unbalanced) distribution.
        uniform_cap = min(cwe_pair_counts.values())
        if total and total > 0:
            base = min(max(1, int(total) // n_cwes), uniform_cap)
        else:
            base = uniform_cap
        quota = {c: base for c in cwes}
    else:  # proportional
        tgt = int(total) if total and total > 0 else total_pairs
        quota = {
            c: max(1, round(tgt * cwe_pair_counts[c] / total_pairs)) for c in cwes
        }

    selected: list = []
    per_cwe: dict[str, dict[str, int]] = {}
    for cwe in cwes:
        groups = list(by_cwe[cwe].items())  # (cve_id, [keys])
        rng.shuffle(groups)
        groups.sort(key=lambda kv: len(kv[1]), reverse=True)  # prefer multi-pair CVEs
        picked: list = []
        n_cve = 0
        for _cve, keys in groups:
            if n_cve < min_cves_per_cwe or len(picked) < quota[cwe]:
                picked.extend(keys)
                n_cve += 1
            else:
                break
        selected.extend(picked)
        per_cwe[cwe] = {"cves": n_cve, "pairs": len(picked)}

    rng.shuffle(selected)
    sampled = {k: paired[k] for k in selected}
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


# ── Per-level retrieval ──────────────────────────────────────────────


def run_level_retrieval(
    level: str,
    paired: dict[tuple[str, str], dict[str, FunctionPair]],
    index_keys: set,
    query_keys: set,
    *,
    embedder_names: list[str],
    ks: list[int],
    emb_cfg: dict,
    out_dir: Path,
) -> list[dict]:
    """Run the HNSW retrieval grid for a single granularity level."""
    index_pairs = [paired[k][level] for k in index_keys if k in paired]
    query_pairs = [paired[k][level] for k in query_keys if k in paired]

    level_dir = out_dir / level
    index_dir = level_dir / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── level={level}  index={len(index_pairs)}  query={len(query_pairs)} ──")
    cells: list[dict] = []

    for emb_name in embedder_names:
        if emb_name not in EMBEDDER_REGISTRY:
            print(f"  WARNING: embedder '{emb_name}' not in registry, skipping")
            continue
        embedder = EMBEDDER_REGISTRY[emb_name](emb_cfg)

        t0 = time.perf_counter()
        index_embeddings = embedder.embed_many([p.G_vuln for p in index_pairs])
        embed_time = time.perf_counter() - t0

        norms = np.linalg.norm(index_embeddings, axis=1)
        if int(np.sum(norms < 1e-6)) == len(index_embeddings):
            print(f"  [{emb_name}] all-zero embeddings, skipping")
            continue

        space_stats = embedding_space_stats(index_embeddings)
        dim = index_embeddings.shape[1]
        index = HNSWIndex(
            dim=dim,
            index_path=str(index_dir / f"{embedder.name}__hnsw.index"),
            metadata_path=str(index_dir / f"{embedder.name}__hnsw_meta.json"),
        )
        retriever = populate_index(
            index, index_pairs, index_embeddings, embedder.name, top_k=max(ks)
        )

        query_embeddings = embedder.embed_many([p.G_vuln for p in query_pairs])
        qr = []
        for pair, qvec in zip(query_pairs, query_embeddings):
            if np.linalg.norm(qvec) < 1e-6:
                continue
            qr.append((pair, retriever.query(qvec, top_k=max(ks))))

        cve_metrics = cve_retrieval_metrics(qr, ks=ks, index_metadata=index.metadata)
        cwe_metrics = cwe_recall_metrics(qr, index.metadata, top_k=max(ks))

        cwe_hit = {}
        for k in ks:
            hits = total = 0
            for pair, results in qr:
                if not pair.cwe_id or pair.cwe_id == "UNKNOWN":
                    continue
                total += 1
                if any(r.get("cwe_id") == pair.cwe_id for r in results[:k]):
                    hits += 1
            cwe_hit[f"hit@{k}"] = hits / total if total else 0.0

        print(
            f"  [{emb_name:<16}] hit@1={cve_metrics.get('hit@1',0):.3f} "
            f"hit@5={cve_metrics.get('hit@5',0):.3f} "
            f"mrr={cve_metrics.get('mrr',0):.3f} "
            f"cwe_recall={cwe_metrics.get('macro_avg',0):.3f}"
        )

        cells.append({
            "embedder": embedder.name,
            "backend": "hnsw",
            "graph_variant": "G_vuln",
            "level": level,
            "n_index": len(index_pairs),
            "n_query": len(qr),
            "embed_time_s": round(embed_time, 2),
            "space_stats": space_stats,
            "cve_retrieval": cve_metrics,
            "cwe_recall": cwe_metrics,
            "cwe_hit": cwe_hit,
        })

    # Dashboard-compatible per-level results.json
    level_dir.mkdir(parents=True, exist_ok=True)
    (level_dir / "results.json").write_text(json.dumps({
        "run_id": f"method_vs_file_{level}",
        "description": f"CVEfixes retrieval — {level}-level input",
        "cells": cells,
    }, indent=2, default=str))

    return cells


# ── Orchestration ────────────────────────────────────────────────────


def load_anchor_entries(
    split_info_path: Path, source_path: Path
) -> tuple[list[dict], set, set]:
    """Reproduce a baseline run's EXACT instances and index/query split.

    Reads ``split_info.json`` (its ``index_entries`` / ``query_entries`` list
    (cve_id, cwe_id, func_name) per instance) and joins each instance back to
    the source dataset (which carries both method and file code) by
    ``(cve_id, method_name)``.

    Returns ``(entries, index_keys, query_keys)`` where entries carry full
    code and the key sets are the fixed baseline split (by (cve, safe-func)).
    """
    si = json.loads(Path(split_info_path).read_text())
    src_entries = json.loads(Path(source_path).read_text())["entries"]
    src: dict[tuple[str, str], dict] = {}
    for e in src_entries:
        src.setdefault((e["cve_id"], e.get("method_name")), e)

    def _resolve(records: list[dict]):
        resolved, missing = [], 0
        for r in records:
            e = src.get((r["cve_id"], r.get("func_name")))
            if e is None:
                missing += 1
                continue
            resolved.append(e)
        return resolved, missing

    idx_entries, miss_i = _resolve(si.get("index_entries", []))
    qry_entries, miss_q = _resolve(si.get("query_entries", []))
    if miss_i or miss_q:
        print(f"  [anchor] unresolved in source: index={miss_i} query={miss_q}")

    entries = idx_entries + qry_entries
    index_keys = {_entry_key(e) for e in idx_entries}
    query_keys = {_entry_key(e) for e in qry_entries}
    return entries, index_keys, query_keys


def run_comparison(
    *,
    config_path: str = "config.yaml",
    entries_path: Path = DEFAULT_ENTRIES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    embedders: list[str] | None = None,
    ks: list[int] | None = None,
    seed: int = DEFAULT_SEED,
    test_ratio: float = DEFAULT_TEST_RATIO,
    reuse_only: bool = False,
    joern_bin: str | None = None,
    anchor_split_info: str | None = None,
    source_data: Path = DEFAULT_SOURCE,
    sample_mode: str = "none",
    sample_total: int | None = None,
) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg.get("experiment", {}).get("pipeline_verification", {})
    embedders = embedders or exp_cfg.get("embedders", DEFAULT_EMBEDDERS)
    ks = ks or exp_cfg.get("ks", DEFAULT_KS)
    emb_cfg = cfg.get("embeddings", {})
    joern_bin = joern_bin or cfg.get("joern", {}).get("bin_dir") or DEFAULT_JOERN_BIN

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: Method-level vs File-level input (same CVEs)")
    print("=" * 70)
    print(f"Embedders: {embedders}")
    print(f"KS:        {ks}   seed={seed}   test_ratio={test_ratio}")
    print(f"Reuse-only (no Joern): {reuse_only}")
    print(f"Sampling:  mode={sample_mode}  total={sample_total}")

    if anchor_split_info:
        print(f"Anchor:    {anchor_split_info}  (strictly same instances & split)")
        entries, fixed_index_keys, fixed_query_keys = load_anchor_entries(
            Path(anchor_split_info), Path(source_data)
        )
        print(f"\n[1/4] Anchored on baseline split_info: {len(entries)} instances "
              f"(index={len(fixed_index_keys)} query={len(fixed_query_keys)}, "
              f"{len({e['cve_id'] for e in entries})} CVEs)")
    else:
        print(f"Entries:   {entries_path}")
        entries = json.loads(Path(entries_path).read_text())["entries"]
        fixed_index_keys = fixed_query_keys = None
        print(f"\n[1/4] Loaded {len(entries)} pinned entries "
              f"({len({e['cve_id'] for e in entries})} unique CVEs)")

    print("\n[2/4] Building method + file CPG pairs...")
    paired = build_paired_pairs(
        entries, output_dir=output_dir, joern_bin=joern_bin, reuse_only=reuse_only
    )
    print(f"  Paired instances built at BOTH levels: {len(paired)}")
    if len(paired) < 20:
        print(f"FATAL: only {len(paired)} paired instances — need at least 20")
        return {}

    # Optional CWE-mix reshape (balanced/proportional) applied to the paired
    # set BEFORE the split, so both levels share the same reshaped sample.
    sample_info: dict[str, Any] = {"mode": sample_mode, "applied": False}

    # Split: fixed baseline split when anchored, else stratified over paired.
    if fixed_index_keys is not None:
        index_keys = {k for k in fixed_index_keys if k in paired}
        query_keys = {k for k in fixed_query_keys if k in paired}
        index_cves = {k[0] for k in index_keys}
        n_before = len(query_keys)
        query_keys = {k for k in query_keys if k[0] in index_cves}
        dropped = n_before - len(query_keys)
        print(f"\n[3/4] Baseline split (strictly same): "
              f"index={len(index_keys)}/{len(fixed_index_keys)}  "
              f"query={len(query_keys)}/{len(fixed_query_keys)}"
              + (f"  ({dropped} queries dropped: no same-CVE support after graph filter)"
                 if dropped else ""))
        split_mode = "baseline_split_info"
    else:
        if sample_mode in ("proportional", "balanced"):
            paired, sample_info = _sample_paired_cve_aware(
                paired, mode=sample_mode, seed=seed, total=sample_total,
            )
            per_cwe_pairs = {c: v["pairs"] for c, v in sample_info["per_cwe"].items()}
            print(
                f"  [Sampler] mode={sample_mode} → {sample_info['result_total_pairs']} "
                f"paired instances, {sample_info['result_total_cves']} CVEs across "
                f"{len(sample_info['per_cwe'])} CWEs"
            )
            print(f"  [Sampler] per-CWE pairs: {per_cwe_pairs}")
            if len(paired) < 20:
                print(f"FATAL: only {len(paired)} paired instances after sampling — need ≥ 20")
                return {}
        entities = [
            {"key": k, "cve_id": k[0], "cwe_id": v["file"].cwe_id}
            for k, v in paired.items()
        ]
        index_keys, query_keys = stratified_split_keys(entities, test_ratio, seed)
        print(f"\n[3/4] Split (shared across levels): "
              f"index={len(index_keys)}  query={len(query_keys)}")
        split_mode = "stratified_seed"
    idx_cwe = Counter(v["file"].cwe_id for k, v in paired.items() if k in index_keys)
    qry_cwe = Counter(v["file"].cwe_id for k, v in paired.items() if k in query_keys)
    print(f"  Index CWE dist: {dict(idx_cwe)}")
    print(f"  Query CWE dist: {dict(qry_cwe)}")

    print("\n[4/4] Running retrieval per level...")
    method_cells = run_level_retrieval(
        "method", paired, index_keys, query_keys,
        embedder_names=embedders, ks=ks, emb_cfg=emb_cfg, out_dir=output_dir,
    )
    file_cells = run_level_retrieval(
        "file", paired, index_keys, query_keys,
        embedder_names=embedders, ks=ks, emb_cfg=emb_cfg, out_dir=output_dir,
    )

    comparison = _build_comparison(
        method_cells, file_cells, ks=ks, seed=seed,
        n_paired=len(paired), n_index=len(index_keys), n_query=len(query_keys),
        embedders=embedders, split_mode=split_mode,
    )
    comparison.setdefault("dataset_info", {})["sampling"] = sample_info
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    print(f"\nComparison → {output_dir / 'comparison.json'}")
    _print_comparison(comparison)
    return comparison


def _build_comparison(method_cells, file_cells, *, ks, seed, n_paired, n_index, n_query, embedders, split_mode="stratified_seed"):
    by_emb = {"method": {c["embedder"]: c for c in method_cells},
              "file": {c["embedder"]: c for c in file_cells}}
    metrics = [f"hit@{k}" for k in ks] + ["mrr"]
    deltas = []
    for emb in embedders:
        m = by_emb["method"].get(emb)
        f = by_emb["file"].get(emb)
        if not m or not f:
            continue
        for metric in metrics:
            mv = (m.get("cve_retrieval") or m.get("self_retrieval") or {}).get(metric, 0.0)
            fv = (f.get("cve_retrieval") or f.get("self_retrieval") or {}).get(metric, 0.0)
            deltas.append({
                "embedder": emb, "metric": metric,
                "method": round(mv, 4), "file": round(fv, 4),
                "delta_file_minus_method": round(fv - mv, 4),
            })
        mcwe = m["cwe_recall"].get("macro_avg", 0.0)
        fcwe = f["cwe_recall"].get("macro_avg", 0.0)
        deltas.append({
            "embedder": emb, "metric": "cwe_recall",
            "method": round(mcwe, 4), "file": round(fcwe, 4),
            "delta_file_minus_method": round(fcwe - mcwe, 4),
        })
    return {
        "run_id": "method_vs_file_level",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Retrieval performance: method-level vs file-level CPG input on the same CVEs",
        "config": {"seed": seed, "ks": ks, "embedders": embedders},
        "dataset_info": {"n_paired_instances": n_paired, "n_index": n_index,
                         "n_query": n_query, "split_mode": split_mode},
        "method_cells": method_cells,
        "file_cells": file_cells,
        "deltas": deltas,
    }


def _print_comparison(comparison: dict) -> None:
    print("\n" + "=" * 70)
    print("METHOD vs FILE  (delta = file - method)")
    print("=" * 70)
    print(f"  {'embedder':<16} {'metric':<12} {'method':>8} {'file':>8} {'delta':>8}")
    print("  " + "-" * 56)
    for d in comparison["deltas"]:
        arrow = "▲" if d["delta_file_minus_method"] > 0 else ("▼" if d["delta_file_minus_method"] < 0 else "=")
        print(f"  {d['embedder']:<16} {d['metric']:<12} "
              f"{d['method']:>8.3f} {d['file']:>8.3f} "
              f"{d['delta_file_minus_method']:>+8.3f} {arrow}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Method-level vs File-level retrieval comparison")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--entries", default=str(DEFAULT_ENTRIES),
                        help="Pinned selected_entries.json (same CVEs anchor)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embedders", nargs="+", default=None,
                        help="Embedders to compare (default: config pipeline_verification.embedders)")
    parser.add_argument("--ks", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--reuse-only", action="store_true",
                        help="Only use already-cached CPGs (no Joern). Smaller paired set, instant.")
    parser.add_argument("--joern-bin", default=None, help="Path to joern-cli dir")
    parser.add_argument("--anchor-split-info", default=None,
                        help="Reproduce the EXACT instances and index/query split from a "
                             "baseline split_info.json (strictly same sample).")
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE),
                        help="Source JSON with file code, used to recover code for anchored instances")
    parser.add_argument("--sample-mode", choices=["none", "proportional", "balanced"],
                        default="none",
                        help="Reshape the paired set to a CWE mix before the split "
                             "(ignored when --anchor-split-info is used).")
    parser.add_argument("--sample-total", type=int, default=None,
                        help="Target paired-instance budget for --sample-mode.")
    args = parser.parse_args()

    run_comparison(
        config_path=args.config,
        entries_path=Path(args.entries),
        output_dir=Path(args.output_dir),
        embedders=args.embedders,
        ks=args.ks,
        seed=args.seed,
        test_ratio=args.test_ratio,
        reuse_only=args.reuse_only,
        joern_bin=args.joern_bin,
        anchor_split_info=args.anchor_split_info,
        source_data=Path(args.source_data),
        sample_mode=args.sample_mode,
        sample_total=args.sample_total,
    )


if __name__ == "__main__":
    main()
