#!/usr/bin/env python3
"""
Slicing depth study on CVEfixes data.

Compares different slice configurations for vulnerability graph construction:
  - depth_1:  1-hop flow expansion from changed nodes
  - depth_2:  2-hop flow expansion
  - depth_3:  3-hop flow expansion (current default)
  - changed_only:  no flow expansion — only removed + fix_adjacent nodes

For each slice config × embedder (gin, combined, codebert_pattern),
evaluates retrieval (hit@1/5/10, MRR) and CWE recall on CVEfixes data
generated on-the-fly from the extraction JSON.

Usage:
    python -m experiments.exp.slicing_depth_study [--config config.yaml] [--n-index 100] [--n-query 20]
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.common import (
    build_flat_index,
    evaluate_cwe_recall,
    evaluate_retrieval,
    make_run_dir,
    save_json,
)
from src.data.base import FunctionPair
from src.data.pipeline import (
    compute_graph_diff,
    load_cpg_dir,
    run_joern_export,
    write_c_file,
)
from src.embeddings import REGISTRY as EMBEDDER_REGISTRY
from src.metrics.metrics import embedding_space_stats

# ── Configuration ────────────────────────────────────────────────────

JOERN_BIN_DIR = "/usr/local/bin"
JSON_PATH = Path("cvefixes_experiments/data/cvefixes_code_extraction.json")
WORK_DIR = Path("cvefixes_experiments/output/slicing_depth_study_work")

EMBEDDER_NAMES = ["gin", "combined", "codebert_pattern"]

# Embedders for token-guided slicing (includes enriched variants that see diff/token_entry)
TOKEN_SLICE_EMBEDDER_NAMES = ["gin", "gin_enriched", "combined", "combined_enriched", "codebert_pattern", "codebert_flow", "codebert_flow_pattern"]

SEED = 42

# CWE families — group related CWEs into vulnerability classes
CWE_FAMILIES = {
    "memory_corruption": ["CWE-119", "CWE-120", "CWE-121", "CWE-122", "CWE-125", "CWE-787", "CWE-805"],
    "use_after_free": ["CWE-416", "CWE-415", "CWE-763"],
    "integer_issues": ["CWE-190", "CWE-191", "CWE-189", "CWE-681", "CWE-369"],
    "null_ptr_deref": ["CWE-476", "CWE-824"],
    "race_condition": ["CWE-362", "CWE-367"],
}

# Reverse mapping: CWE-ID → family name
CWE_TO_FAMILY = {}
for fam, cwes in CWE_FAMILIES.items():
    for cwe in cwes:
        CWE_TO_FAMILY[cwe] = fam

N_FAMILIES = 5
SAMPLES_PER_FAMILY = 20
# Slice depth variants — each is a (SLICE_DEPTH, weight_threshold) tuple.
# weight_threshold=None means keep all nodes in the slice.
# weight_threshold=0.3 means keep only nodes with diff_weight > threshold.
SLICE_CONFIGS = {
    "depth_1": {"depth": 1, "weight_threshold": None},
    "depth_2": {"depth": 2, "weight_threshold": None},
    "depth_3": {"depth": 3, "weight_threshold": None},
    "changed_only": {"depth": 0, "weight_threshold": 0.3},
    "removed_only": {"depth": 0, "weight_threshold": 0.9},
    "tight_1hop": {"depth": 1, "weight_threshold": 0.5},
}

# ── Discriminative tokens per vulnerability family ───────────────────
# (from TF-IDF + LinearSVC feature importance analysis)
DISCRIMINATIVE_TOKENS = {
    "integer_issues": [
        "string", "it_overrun", "val", "scale", "sma", "int", "acl_len",
    ],
    "memory_corruption": [
        "ptr", "buf", "size_t", "buffer", "str_len", "ndo", "nd_tcheck",
    ],
    "null_ptr_deref": [
        "unlock", "syscall_nr", "fs_uuid", "name", "page", "dir",
    ],
    "race_condition": [
        "sk", "shutting_down", "arg", "get_block", "struct", "opt",
    ],
    "use_after_free": [
        "bfqq", "bfq_idle_slice_timer_body", "retval", "free", "size2",
        "head", "create_spnego_ctx", "sc",
    ],
}

# Flattened set for family-agnostic matching
ALL_DISCRIMINATIVE_TOKENS = set()
for _tokens in DISCRIMINATIVE_TOKENS.values():
    ALL_DISCRIMINATIVE_TOKENS.update(_tokens)


# ── Custom graph diff with configurable depth ────────────────────────


def compute_graph_diff_configurable(
    G_before: nx.MultiDiGraph,
    G_after: nx.MultiDiGraph,
    slice_depth: int = 3,
    weight_threshold: float | None = None,
) -> nx.MultiDiGraph:
    """
    Same logic as pipeline.compute_graph_diff but with configurable
    slice depth and optional weight-based filtering.
    """
    from collections import Counter

    NOISE_TYPES = {
        "TYPE_DECL", "FILE", "NAMESPACE_BLOCK", "COMMENT", "UNKNOWN", "METHOD_RETURN",
    }
    FLOW_EDGES = {"CFG", "CDG", "REACHING_DEF", "PDG", "DDG"}
    CHANGE_WEIGHT = {
        "removed": 1.0,
        "fix_adjacent": 0.8,
        "edge_changed": 0.6,
        "context": 0.2,
    }

    def _code(attrs: dict) -> str:
        v = attrs.get("CODE")
        return str(v).strip() if v else ""

    def _node_fp(G, n):
        a = G.nodes[n]
        return (a.get("labelV", ""), _code(a), str(a.get("LINE_NUMBER", "")))

    def _edge_fp(G, u, v, d):
        return (_node_fp(G, u), _node_fp(G, v), d.get("labelE") or d.get("label", ""))

    def _is_semantic(G, n):
        return G.nodes[n].get("labelV") not in NOISE_TYPES

    # ── 1. semantic node diff
    before_fps = Counter(_node_fp(G_before, n) for n in G_before if _is_semantic(G_before, n))
    after_fps = Counter(_node_fp(G_after, n) for n in G_after if _is_semantic(G_after, n))

    removed_fps = {fp for fp in before_fps if before_fps[fp] > after_fps.get(fp, 0)}
    added_fps = {fp for fp in after_fps if after_fps[fp] > before_fps.get(fp, 0)}

    changed = set()
    diff_label = {}

    for n in G_before:
        if _is_semantic(G_before, n) and _node_fp(G_before, n) in removed_fps:
            changed.add(n)
            diff_label[n] = "removed"

    after_fp_to_nodes = {}
    for n in G_after:
        after_fp_to_nodes.setdefault(_node_fp(G_after, n), []).append(n)

    before_fp_to_nodes = {}
    for n in G_before:
        before_fp_to_nodes.setdefault(_node_fp(G_before, n), []).append(n)

    for fp in added_fps:
        for n_after in after_fp_to_nodes.get(fp, []):
            neighbors = set(G_after.predecessors(n_after)) | set(G_after.successors(n_after))
            for nb in neighbors:
                nb_fp = _node_fp(G_after, nb)
                for n_before in before_fp_to_nodes.get(nb_fp, []):
                    if n_before not in diff_label:
                        changed.add(n_before)
                        diff_label[n_before] = "fix_adjacent"

    # ── 2. semantic edge diff
    before_efps = Counter(_edge_fp(G_before, u, v, d) for u, v, d in G_before.edges(data=True))
    after_efps = Counter(_edge_fp(G_after, u, v, d) for u, v, d in G_after.edges(data=True))

    changed_efps = {
        efp for efp in before_efps | after_efps
        if before_efps.get(efp, 0) != after_efps.get(efp, 0)
    }

    for u, v, d in G_before.edges(data=True):
        if _edge_fp(G_before, u, v, d) in changed_efps:
            for nd in (u, v):
                changed.add(nd)
                diff_label.setdefault(nd, "edge_changed")

    # ── 3. bounded program slice (configurable depth)
    slice_nodes = set(changed)
    frontier = set(changed)

    for _ in range(slice_depth):
        next_frontier = set()
        for n in frontier:
            if n not in G_before:
                continue
            for _, tgt, d in G_before.out_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in FLOW_EDGES and tgt not in slice_nodes:
                    next_frontier.add(tgt)
            for src, _, d in G_before.in_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in FLOW_EDGES and src not in slice_nodes:
                    next_frontier.add(src)
        slice_nodes |= next_frontier
        frontier = next_frontier

    # ── 4. filter noise types
    slice_nodes = {n for n in slice_nodes if n in G_before and _is_semantic(G_before, n)}

    if not slice_nodes:
        return nx.MultiDiGraph()

    # ── 5. build subgraph with diff labels + weights
    G_vuln = G_before.subgraph(slice_nodes).copy()
    for n in G_vuln:
        dlabel = diff_label.get(n, "context")
        G_vuln.nodes[n]["diff"] = dlabel
        G_vuln.nodes[n]["diff_weight"] = CHANGE_WEIGHT.get(dlabel, 0.2)

    # ── 6. optional weight threshold (keep only high-signal nodes)
    if weight_threshold is not None:
        keep = {n for n in G_vuln if G_vuln.nodes[n]["diff_weight"] > weight_threshold}
        if keep:
            G_vuln = G_vuln.subgraph(keep).copy()
        else:
            return nx.MultiDiGraph()

    return G_vuln


# ── Token-guided slicing ─────────────────────────────────────────────


def compute_token_guided_slice(
    G: nx.MultiDiGraph,
    tokens: list[str],
    slice_depth: int = 2,
) -> nx.MultiDiGraph:
    """
    Build a program slice starting from nodes whose CODE contains any of the
    discriminative tokens. Expands outward via flow edges (CFG, CDG, REACHING_DEF).

    This tests the hypothesis: if certain tokens are lexically discriminative
    for a vulnerability family, do the graph neighborhoods around those tokens
    also capture discriminative structural patterns?

    Args:
        G: Full CPG (before or after version)
        tokens: List of discriminative tokens to search for in node CODE
        slice_depth: How many hops to expand from entry points

    Returns:
        Subgraph containing the token-guided slice with 'token_entry' attribute
        on seed nodes.
    """
    NOISE_TYPES = {
        "TYPE_DECL", "FILE", "NAMESPACE_BLOCK", "COMMENT", "UNKNOWN", "METHOD_RETURN",
    }
    FLOW_EDGES = {"CFG", "CDG", "REACHING_DEF", "PDG", "DDG"}

    def _code(attrs: dict) -> str:
        v = attrs.get("CODE")
        return str(v).strip() if v else ""

    def _is_semantic(n):
        return G.nodes[n].get("labelV") not in NOISE_TYPES

    # ── 1. Find entry-point nodes: nodes whose CODE contains a discriminative token
    entry_nodes = set()
    token_matches = {}  # node → matched tokens

    for n in G:
        if not _is_semantic(n):
            continue
        code = _code(G.nodes[n]).lower()
        if not code:
            continue
        matched = [t for t in tokens if t.lower() in code]
        if matched:
            entry_nodes.add(n)
            token_matches[n] = matched

    if not entry_nodes:
        return nx.MultiDiGraph()

    # ── 2. Bounded program slice from entry points
    slice_nodes = set(entry_nodes)
    frontier = set(entry_nodes)

    for _ in range(slice_depth):
        next_frontier = set()
        for n in frontier:
            if n not in G:
                continue
            for _, tgt, d in G.out_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in FLOW_EDGES and tgt not in slice_nodes:
                    next_frontier.add(tgt)
            for src, _, d in G.in_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in FLOW_EDGES and src not in slice_nodes:
                    next_frontier.add(src)
        slice_nodes |= next_frontier
        frontier = next_frontier

    # ── 3. Filter noise
    slice_nodes = {n for n in slice_nodes if n in G and _is_semantic(n)}
    if not slice_nodes:
        return nx.MultiDiGraph()

    # ── 4. Build subgraph with token_entry annotations
    G_slice = G.subgraph(slice_nodes).copy()
    for n in G_slice:
        is_entry = n in entry_nodes
        G_slice.nodes[n]["token_entry"] = is_entry
        # Weight: entry nodes get high weight, neighbors get distance-based decay
        G_slice.nodes[n]["diff_weight"] = 1.0 if is_entry else 0.4
        # Use distinct label "token_entry" — NOT "removed" — so GIN can
        # distinguish token-guided entry points from actual diff changes
        G_slice.nodes[n]["diff"] = "token_entry" if is_entry else "context"
        if is_entry:
            G_slice.nodes[n]["matched_tokens"] = ",".join(token_matches[n])

    return G_slice


def compute_token_guided_slice_family_aware(
    G: nx.MultiDiGraph,
    family: str,
    slice_depth: int = 2,
) -> nx.MultiDiGraph:
    """
    Family-aware version: uses only the tokens for the given family.
    This tests whether family-specific tokens produce family-discriminative slices.
    """
    tokens = DISCRIMINATIVE_TOKENS.get(family, [])
    if not tokens:
        return nx.MultiDiGraph()
    return compute_token_guided_slice(G, tokens, slice_depth=slice_depth)


def compute_token_guided_slice_all(
    G: nx.MultiDiGraph,
    slice_depth: int = 2,
) -> nx.MultiDiGraph:
    """
    Family-agnostic version: uses all discriminative tokens.
    Entry-point nodes carry which family's tokens they matched.
    """
    return compute_token_guided_slice(G, list(ALL_DISCRIMINATIVE_TOKENS), slice_depth=slice_depth)


# ── Data preparation ─────────────────────────────────────────────────


def select_by_family(entries: list[dict], samples_per_family: int = SAMPLES_PER_FAMILY, seed: int = 42) -> list[dict]:
    """
    Select entries grouped by CWE family.
    Returns samples_per_family entries per family (with extra for Joern failures).
    Each entry gets a 'cwe_family' key added.
    """
    rng = random.Random(seed)

    # Group entries by family
    by_family = defaultdict(list)
    for e in entries:
        cwes = e.get("cwe", [])
        if not cwes:
            continue
        cwe_id = cwes[0]["cwe_id"]
        family = CWE_TO_FAMILY.get(cwe_id)
        if family is None:
            continue
        cb = e.get("code_before") or ""
        ca = e.get("code_after") or ""
        if not cb or not ca:
            continue
        bl = len(cb.split("\n"))
        al = len(ca.split("\n"))
        if 10 < bl < 80 and 10 < al < 80:
            by_family[family].append(e)

    # Select samples_per_family * 1.5 per family (extra for failures)
    n_select = int(samples_per_family * 1.5)
    selected = []
    for family in CWE_FAMILIES:
        pool = by_family.get(family, [])
        rng.shuffle(pool)
        for e in pool[:n_select]:
            e["cwe_family"] = family
            selected.append(e)

    rng.shuffle(selected)
    print(f"  Family selection: {', '.join(f'{fam}={min(len(by_family[fam]), n_select)}' for fam in CWE_FAMILIES)}")
    return selected


def generate_cpg_pair(entry: dict, work_dir: Path) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph] | None:
    """Generate before/after CPGs for a single entry. Returns (G_before, G_after) or None on failure."""
    func_name = entry.get("method_name") or "function"
    # sanitize filename
    func_name_safe = "".join(c if c.isalnum() or c == "_" else "_" for c in func_name)

    before_dir = work_dir / "before"
    after_dir = work_dir / "after"

    try:
        # Before
        src_before = write_c_file(entry["code_before"], before_dir / f"{func_name_safe}.cpp")
        graph_before_dir = before_dir / "graph"
        ok = run_joern_export(JOERN_BIN_DIR, str(src_before), str(before_dir), str(graph_before_dir))
        if not ok:
            return None

        # After
        src_after = write_c_file(entry["code_after"], after_dir / f"{func_name_safe}.cpp")
        graph_after_dir = after_dir / "graph"
        ok = run_joern_export(JOERN_BIN_DIR, str(src_after), str(after_dir), str(graph_after_dir))
        if not ok:
            return None

        G_before = load_cpg_dir(str(graph_before_dir))
        G_after = load_cpg_dir(str(graph_after_dir))

        # Reject if Joern failed to parse (< 10 nodes → likely CPPASTProblem)
        if G_before.number_of_nodes() < 10 or G_after.number_of_nodes() < 10:
            return None

        return G_before, G_after

    except Exception:
        return None


def prepare_dataset(
    n_total: int, seed: int = SEED
) -> list[dict]:
    """
    Load JSON, select diverse examples by family, generate CPGs.
    Returns list of dicts with keys: entry, G_before, G_after, work_dir.
    """
    with open(JSON_PATH) as f:
        data = json.load(f)

    candidates = select_by_family(data["entries"], samples_per_family=SAMPLES_PER_FAMILY, seed=seed)
    print(f"Selected {len(candidates)} candidates (targeting {n_total} successful)")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    prepared = []
    family_counts = Counter()
    t_start = time.perf_counter()

    for i, entry in enumerate(candidates):
        # Stop when we have enough per family
        family = entry.get("cwe_family", "UNKNOWN")
        if family_counts[family] >= SAMPLES_PER_FAMILY:
            continue
        if sum(min(v, SAMPLES_PER_FAMILY) for v in family_counts.values()) >= n_total:
            break

        cve = entry["cve_id"]
        func = entry.get("method_name") or "func"
        func_safe = "".join(c if c.isalnum() or c == "_" else "_" for c in func)
        entry_dir = WORK_DIR / f"{i:04d}_{cve}_{func_safe}"

        # Skip if already generated
        if (entry_dir / "before" / "graph").exists() and (entry_dir / "after" / "graph").exists():
            try:
                G_before = load_cpg_dir(str(entry_dir / "before" / "graph"))
                G_after = load_cpg_dir(str(entry_dir / "after" / "graph"))
                if G_before.number_of_nodes() >= 10 and G_after.number_of_nodes() >= 10:
                    prepared.append({
                        "entry": entry,
                        "G_before": G_before,
                        "G_after": G_after,
                        "work_dir": entry_dir,
                    })
                    family_counts[family] += 1
                    continue
            except Exception:
                pass

        # Generate fresh
        if entry_dir.exists():
            shutil.rmtree(entry_dir)

        result = generate_cpg_pair(entry, entry_dir)
        if result is None:
            print(f"  [{i+1}] SKIP {cve}/{func} (Joern parse failure)")
            continue

        G_before, G_after = result
        prepared.append({
            "entry": entry,
            "G_before": G_before,
            "G_after": G_after,
            "work_dir": entry_dir,
        })
        family_counts[family] += 1
        elapsed = time.perf_counter() - t_start
        rate = elapsed / len(prepared)
        eta = rate * (n_total - len(prepared))
        print(f"  [{len(prepared):3d}/{n_total}] {cve}/{func} [{family}]  "
              f"nodes: {G_before.number_of_nodes()}/{G_after.number_of_nodes()}  "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed_total = time.perf_counter() - t_start
    print(f"\nPrepared {len(prepared)} examples in {elapsed_total:.0f}s")
    print(f"  Per-family: {dict(family_counts)}")
    return prepared


# ── Build FunctionPair objects per slice config ──────────────────────


def build_pairs_for_config(
    prepared: list[dict], slice_cfg: dict
) -> list[FunctionPair]:
    """Build FunctionPair list using specified slice config."""
    depth = slice_cfg["depth"]
    threshold = slice_cfg["weight_threshold"]
    pairs = []

    for item in prepared:
        entry = item["entry"]
        G_before = item["G_before"]
        G_after = item["G_after"]

        G_vuln = compute_graph_diff_configurable(
            G_before, G_after,
            slice_depth=depth,
            weight_threshold=threshold,
        )

        if G_vuln.number_of_nodes() == 0:
            continue

        # Use family as the class label for retrieval
        cwe_family = entry.get("cwe_family", "UNKNOWN")

        pairs.append(FunctionPair(
            cve_id=entry["cve_id"],
            cwe_id=cwe_family,
            func_name=entry.get("method_name") or "",
            project="",
            G_before=G_before,
            G_after=G_after,
            G_vuln=G_vuln,
            meta={
                "dataset": "CVEFixes",
                "variant": "original",
                "filename": entry.get("filename", ""),
                "language": entry.get("programming_language", ""),
            },
        ))

    return pairs


# ── Stratified train/test split ──────────────────────────────────────


def stratified_split(pairs: list[FunctionPair], n_test: int, seed: int = SEED):
    """Split into index/query sets ensuring each CWE family has representation in both."""
    rng = random.Random(seed)

    by_family = defaultdict(list)
    for p in pairs:
        by_family[p.cwe_id].append(p)  # cwe_id holds family name

    query_pairs = []
    index_pairs = []

    for family, items in by_family.items():
        rng.shuffle(items)
        # Put 20% of each family into query (min 1)
        n_q = max(1, len(items) // 5)
        query_pairs.extend(items[:n_q])
        index_pairs.extend(items[n_q:])

    # Trim to target if we overshot
    if len(query_pairs) > n_test:
        rng.shuffle(query_pairs)
        excess = query_pairs[n_test:]
        query_pairs = query_pairs[:n_test]
        index_pairs.extend(excess)

    return index_pairs, query_pairs


# ── Main experiment loop ─────────────────────────────────────────────


def run_experiment(n_index: int = 100, n_query: int = 20, cfg_path: str | None = None):
    """Run the full slicing depth comparison experiment."""
    import yaml

    # Load config for embedder settings
    cfg_file = cfg_path or "config.yaml"
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    run_id, run_dir = make_run_dir("slice_depth")
    print(f"Run: {run_id}")
    print(f"Output: {run_dir}")
    print(f"Target: {n_index} index + {n_query} query pairs")
    print()

    # ── 1. Prepare data (generate CPGs)
    n_total = N_FAMILIES * SAMPLES_PER_FAMILY  # 5 families × 10 = 50
    print("=" * 60)
    print(f"  Phase 1: Generating CPGs via Joern ({N_FAMILIES} families × {SAMPLES_PER_FAMILY} samples)")
    print("=" * 60)
    prepared = prepare_dataset(n_total, seed=SEED)

    if len(prepared) < 25:
        print(f"FATAL: only {len(prepared)} examples — need at least 25")
        return {}

    # Determine split sizes from what we actually got
    n_query = min(n_query, len(prepared) // 4)
    n_index = len(prepared) - n_query
    print(f"  Final split: {n_index} index + {n_query} query = {len(prepared)} total")

    # ── 2. Build embedders
    emb_cfg = cfg["embeddings"]
    embedders = []
    for name in EMBEDDER_NAMES:
        if name in EMBEDDER_REGISTRY:
            embedders.append(EMBEDDER_REGISTRY[name](emb_cfg))
        else:
            print(f"  WARNING: embedder '{name}' not in registry, skipping")

    # ── 3. Run experiment grid: slice_config × embedder
    print("\n" + "=" * 60)
    print("  Phase 2: Evaluating slice configs × embedders")
    print("=" * 60)

    all_results = []
    ks = [1, 5, 10]

    for slice_name, slice_cfg in SLICE_CONFIGS.items():
        print(f"\n{'─' * 60}")
        print(f"  Slice config: {slice_name} (depth={slice_cfg['depth']}, threshold={slice_cfg['weight_threshold']})")
        print(f"{'─' * 60}")

        # Build pairs with this slice config
        pairs = build_pairs_for_config(prepared, slice_cfg)
        if len(pairs) < n_query + 5:
            print(f"  SKIP — only {len(pairs)} pairs produced (need ≥{n_query + 5})")
            continue

        # Split
        index_pairs, query_pairs = stratified_split(pairs, n_query, seed=SEED)
        print(f"  Index: {len(index_pairs)}, Query: {len(query_pairs)}")

        # Slice stats
        slice_sizes = [p.G_vuln.number_of_nodes() for p in pairs]
        before_sizes = [p.G_before.number_of_nodes() for p in pairs]
        ratios = [s / b for s, b in zip(slice_sizes, before_sizes) if b > 0]
        diff_dist = Counter()
        for p in pairs:
            for n in p.G_vuln.nodes():
                diff_dist[p.G_vuln.nodes[n].get("diff", "?")] += 1

        slice_stats = {
            "mean_slice_nodes": round(np.mean(slice_sizes), 1),
            "mean_before_nodes": round(np.mean(before_sizes), 1),
            "mean_slice_ratio": round(np.mean(ratios), 3),
            "diff_distribution": dict(diff_dist),
            "n_pairs_valid": len(pairs),
        }
        print(f"  Slice stats: mean={slice_stats['mean_slice_nodes']:.0f} nodes, "
              f"ratio={slice_stats['mean_slice_ratio']:.1%} of before graph")
        print(f"  Diff dist: {dict(diff_dist)}")

        for embedder in embedders:
            print(f"\n    [{embedder.name}]")

            # Reset PCA state for fresh fit
            if hasattr(embedder, "_fitted"):
                embedder._fitted = False
            if hasattr(embedder, "_pca"):
                embedder._pca = None

            try:
                # Embed index
                t0 = time.perf_counter()
                index_graphs = [p.G_vuln for p in index_pairs]
                index_embeddings = embedder.embed_many(index_graphs)
                embed_time = time.perf_counter() - t0

                # Check for degenerate embeddings
                norms = np.linalg.norm(index_embeddings, axis=1)
                n_zero = int(np.sum(norms < 1e-6))
                if n_zero == len(index_embeddings):
                    print(f"      SKIP — all embeddings zero")
                    all_results.append({
                        "slice_config": slice_name,
                        "embedder": embedder.name,
                        "error": "all embeddings zero",
                        **{f"hit@{k}": 0 for k in ks},
                        "mrr": 0, "cwe_recall": 0,
                    })
                    continue

                space_stats = embedding_space_stats(index_embeddings)

                # Build flat index (deterministic, no HNSW randomness)
                actual_dim = index_embeddings.shape[1]
                index, retriever = build_flat_index(
                    index_pairs, index_embeddings, embedder.name, actual_dim
                )

                # Embed queries
                query_graphs = [p.G_vuln for p in query_pairs]
                query_embeddings = embedder.embed_many(query_graphs)

                # Evaluate
                sr = evaluate_retrieval(
                    query_pairs, query_embeddings, retriever, index_pairs, ks=ks
                )
                cwe_result = evaluate_cwe_recall(
                    query_pairs, query_embeddings, retriever, index.metadata, top_k=max(ks)
                )
                cwe_recall = cwe_result["macro_avg"]

                hit1 = sr.get("hit@1", 0)
                hit5 = sr.get("hit@5", 0)
                hit10 = sr.get("hit@10", 0)
                mrr = sr.get("mrr", 0)

                print(f"      hit@1={hit1:.3f}  hit@5={hit5:.3f}  hit@10={hit10:.3f}  "
                      f"MRR={mrr:.3f}  CWE_recall={cwe_recall:.3f}  "
                      f"({embed_time:.1f}s, {n_zero} zero)")

                all_results.append({
                    "slice_config": slice_name,
                    "slice_depth": slice_cfg["depth"],
                    "weight_threshold": slice_cfg["weight_threshold"],
                    "embedder": embedder.name,
                    "hit@1": round(hit1, 4),
                    "hit@5": round(hit5, 4),
                    "hit@10": round(hit10, 4),
                    "mrr": round(mrr, 4),
                    "cve_precision": round(sr.get("cve_precision", 0), 4),
                    "cve_recall": round(sr.get("cve_recall", 0), 4),
                    "cve_f1": round(sr.get("cve_f1", 0), 4),
                    "cwe_recall": round(cwe_recall, 4),
                    "effective_dim": round(space_stats.get("effective_dim", 0), 1),
                    "mean_pairwise_sim": round(space_stats.get("mean_pairwise_sim", 0), 4),
                    "n_index": len(index_pairs),
                    "n_query": len(query_pairs),
                    "embed_time_s": round(embed_time, 1),
                    "slice_stats": slice_stats,
                })

            except Exception as e:
                print(f"      ERROR: {type(e).__name__}: {e}")
                all_results.append({
                    "slice_config": slice_name,
                    "embedder": embedder.name,
                    "error": str(e),
                    **{f"hit@{k}": 0 for k in ks},
                    "mrr": 0, "cwe_recall": 0,
                })

    # ── 4. Report
    report = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_index_target": n_index,
            "n_query_target": n_query,
            "seed": SEED,
            "slice_configs": SLICE_CONFIGS,
            "embedders": EMBEDDER_NAMES,
        },
        "results": all_results,
    }

    output_path = run_dir / "slicing_depth_study.json"
    save_json(report, output_path)
    print(f"\nResults saved to: {output_path}")

    _print_summary(report)

    # ── 5. Diff-line classification/clustering
    entries_for_diff = [item["entry"] for item in prepared]
    diff_results = run_diff_classification(entries_for_diff, run_dir)
    report["diff_classification"] = diff_results

    # ── 6. Token-guided slicing evaluation
    token_results = run_token_guided_slicing(prepared, embedders, run_dir)
    report["token_guided_slicing"] = token_results

    # Re-save with all results included
    save_json(report, output_path)

    return report


def _print_summary(report: dict):
    """Print comparison table."""
    results = report["results"]
    if not results:
        print("No results to display.")
        return

    print(f"\n{'=' * 90}")
    print("  SLICING DEPTH STUDY — RESULTS SUMMARY")
    print(f"{'=' * 90}")

    # Group by embedder
    embedders_seen = sorted(set(r["embedder"] for r in results))
    configs_seen = list(SLICE_CONFIGS.keys())

    for emb in embedders_seen:
        print(f"\n  ── {emb} {'─' * (70 - len(emb))}")
        print(f"    {'Config':<16s} {'hit@1':>7s} {'hit@5':>7s} {'hit@10':>7s} {'MRR':>7s} {'CWE':>7s} {'Nodes':>7s} {'Ratio':>7s}")
        print(f"    {'─' * 72}")

        for cfg_name in configs_seen:
            row = next((r for r in results if r["embedder"] == emb and r["slice_config"] == cfg_name), None)
            if row is None:
                continue
            ss = row.get("slice_stats", {})
            print(
                f"    {cfg_name:<16s} "
                f"{row.get('hit@1', 0):>7.3f} "
                f"{row.get('hit@5', 0):>7.3f} "
                f"{row.get('hit@10', 0):>7.3f} "
                f"{row.get('mrr', 0):>7.3f} "
                f"{row.get('cwe_recall', 0):>7.3f} "
                f"{ss.get('mean_slice_nodes', 0):>7.0f} "
                f"{ss.get('mean_slice_ratio', 0):>6.1%}"
            )

    print()


# ── Diff-line extraction & classification/clustering ─────────────────


def extract_diff_lines(code_before: str, code_after: str) -> dict[str, list[str]]:
    """
    Extract added and removed lines from a unified diff.
    Returns dict with keys: 'removed', 'added', 'all_changed'.
    """
    before_lines = (code_before or "").splitlines(keepends=True)
    after_lines = (code_after or "").splitlines(keepends=True)

    diff = list(difflib.unified_diff(before_lines, after_lines, n=0))

    removed = []
    added = []
    for line in diff:
        if line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())

    # Filter out empty lines and braces-only lines
    removed = [l for l in removed if l and l not in ("{", "}", "")]
    added = [l for l in added if l and l not in ("{", "}", "")]

    return {
        "removed": removed,
        "added": added,
        "all_changed": removed + added,
    }


def build_diff_corpus(entries: list[dict]) -> tuple[list[str], list[str]]:
    """
    Build a corpus of diff-line text per entry and corresponding CWE family labels.
    Returns (documents, labels) where each document is the joined diff lines.
    """
    documents = []
    labels = []

    for entry in entries:
        cb = entry.get("code_before") or ""
        ca = entry.get("code_after") or ""
        if not cb and not ca:
            continue

        diff = extract_diff_lines(cb, ca)
        text = " ".join(diff["all_changed"])
        if not text.strip():
            continue

        # Get CWE family
        cwes = entry.get("cwe", [])
        if not cwes:
            continue
        cwe_id = cwes[0]["cwe_id"]
        family = CWE_TO_FAMILY.get(cwe_id)
        if family is None:
            continue

        documents.append(text)
        labels.append(family)

    return documents, labels


def run_diff_classification(entries: list[dict], run_dir: Path | None = None) -> dict:
    """
    Classify vulnerability entries by CWE family using diff-line TF-IDF features.

    Uses SVM with stratified cross-validation. Also runs K-means clustering
    and reports Adjusted Rand Index + homogeneity.
    """
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        adjusted_rand_score,
        classification_report,
        homogeneity_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import LinearSVC

    print("\n" + "=" * 60)
    print("  Phase 3: Diff-line classification & clustering")
    print("=" * 60)

    documents, labels = build_diff_corpus(entries)
    print(f"  Corpus: {len(documents)} samples, {len(set(labels))} classes")
    print(f"  Class distribution: {Counter(labels)}")

    if len(documents) < 20 or len(set(labels)) < 2:
        print("  SKIP — insufficient data for classification")
        return {}

    # ── TF-IDF vectorization (token-level, keeping C identifiers)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[<>=!&|]{1,2}|[+\-*/]",
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(documents)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = list(le.classes_)

    print(f"  TF-IDF matrix: {X.shape[0]} × {X.shape[1]}")

    # ── Classification: LinearSVC with stratified 5-fold CV
    n_splits = min(5, min(Counter(y).values()))
    if n_splits < 2:
        print("  SKIP classification — some classes have < 2 samples")
        clf_report = {}
    else:
        clf = make_pipeline(LinearSVC(max_iter=5000, random_state=SEED))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        y_pred = cross_val_predict(clf, X, y, cv=skf)

        report = classification_report(
            y, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        clf_report = {
            "accuracy": round(report["accuracy"], 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            "per_class": {
                name: {
                    "precision": round(report[name]["precision"], 4),
                    "recall": round(report[name]["recall"], 4),
                    "f1": round(report[name]["f1-score"], 4),
                    "support": report[name]["support"],
                }
                for name in class_names
            },
        }
        print(f"\n  Classification (LinearSVC, {n_splits}-fold CV):")
        print(f"    Accuracy:    {clf_report['accuracy']:.3f}")
        print(f"    Macro F1:    {clf_report['macro_f1']:.3f}")
        print(f"    Weighted F1: {clf_report['weighted_f1']:.3f}")
        for name in class_names:
            pc = clf_report["per_class"][name]
            print(f"      {name:<20s}  P={pc['precision']:.3f}  R={pc['recall']:.3f}  F1={pc['f1']:.3f}  (n={pc['support']})")

    # ── Clustering: K-means
    n_clusters = len(set(labels))
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    cluster_labels = km.fit_predict(X.toarray())

    ari = adjusted_rand_score(y, cluster_labels)
    homogeneity = homogeneity_score(y, cluster_labels)

    cluster_report = {
        "n_clusters": n_clusters,
        "adjusted_rand_index": round(ari, 4),
        "homogeneity": round(homogeneity, 4),
    }
    print(f"\n  Clustering (K-means, k={n_clusters}):")
    print(f"    Adjusted Rand Index: {ari:.3f}")
    print(f"    Homogeneity:         {homogeneity:.3f}")

    # ── Top features per class
    if n_splits >= 2:
        # Fit a final model on all data to extract feature importances
        final_clf = LinearSVC(max_iter=5000, random_state=SEED)
        final_clf.fit(X, y)
        feature_names = vectorizer.get_feature_names_out()

        top_features = {}
        for i, name in enumerate(class_names):
            if len(class_names) == 2:
                coefs = final_clf.coef_[0]
                top_idx = np.argsort(coefs if i == 1 else -coefs)[-10:]
            else:
                top_idx = np.argsort(final_clf.coef_[i])[-10:]
            top_features[name] = [feature_names[j] for j in top_idx]

        print(f"\n  Top discriminative tokens per family:")
        for name, feats in top_features.items():
            print(f"    {name:<20s}: {', '.join(feats)}")
    else:
        top_features = {}

    results = {
        "n_samples": len(documents),
        "n_classes": len(class_names),
        "class_names": class_names,
        "classification": clf_report,
        "clustering": cluster_report,
        "top_features": top_features,
    }

    if run_dir:
        save_json(results, run_dir / "diff_classification.json")
        print(f"\n  Saved to: {run_dir / 'diff_classification.json'}")

    return results


# ── Token-guided slicing → unsupervised clustering ───────────────────


def _random_slice(G: nx.MultiDiGraph, target_size: int, seed: int = 42) -> nx.MultiDiGraph:
    """Build a random slice of target_size nodes for baseline comparison."""
    NOISE_TYPES = {
        "TYPE_DECL", "FILE", "NAMESPACE_BLOCK", "COMMENT", "UNKNOWN", "METHOD_RETURN",
    }
    FLOW_EDGES = {"CFG", "CDG", "REACHING_DEF", "PDG", "DDG"}

    semantic_nodes = [
        n for n in G
        if G.nodes[n].get("labelV") not in NOISE_TYPES
    ]
    if not semantic_nodes or target_size < 1:
        return nx.MultiDiGraph()

    rng = random.Random(seed)
    # Pick a random seed node, expand via flow edges
    seed_node = rng.choice(semantic_nodes)
    slice_nodes = {seed_node}
    frontier = {seed_node}

    while len(slice_nodes) < target_size and frontier:
        next_frontier = set()
        for n in frontier:
            for _, tgt, d in G.out_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in FLOW_EDGES and tgt not in slice_nodes:
                    next_frontier.add(tgt)
                    slice_nodes.add(tgt)
                    if len(slice_nodes) >= target_size:
                        break
            if len(slice_nodes) >= target_size:
                break
            for src, _, d in G.in_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in FLOW_EDGES and src not in slice_nodes:
                    next_frontier.add(src)
                    slice_nodes.add(src)
                    if len(slice_nodes) >= target_size:
                        break
            if len(slice_nodes) >= target_size:
                break
        frontier = next_frontier

    slice_nodes = {n for n in slice_nodes if n in G and G.nodes[n].get("labelV") not in NOISE_TYPES}
    if not slice_nodes:
        return nx.MultiDiGraph()

    G_slice = G.subgraph(slice_nodes).copy()
    for n in G_slice:
        G_slice.nodes[n]["diff"] = "context"
        G_slice.nodes[n]["diff_weight"] = 0.4
    return G_slice


def run_token_guided_slicing(
    prepared: list[dict],
    embedders: list,
    run_dir: Path,
) -> dict:
    """
    Phase 6: Token-guided slicing → graph embedding → unsupervised clustering.

    Tests whether graph slices rooted at discriminative-token nodes naturally
    cluster into vulnerability families without supervised labels.

    Compares three slicing strategies:
      - token_guided: entry points = nodes containing discriminative tokens
      - diff_based:   entry points = changed nodes (existing approach, depth=2)
      - random:       entry points = random nodes (structural null hypothesis)

    For each strategy × embedder:
      - Build graph slices
      - Embed via GIN/combined/codebert_pattern
      - K-means cluster (k = n_families)
      - Measure ARI, homogeneity, NMI vs true family labels
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        homogeneity_score,
        normalized_mutual_info_score,
    )

    print("\n" + "=" * 60)
    print("  Phase 4: Token-guided slicing → clustering")
    print("  Hypothesis: graph neighborhoods of discriminative tokens")
    print("  naturally cluster by vulnerability family")
    print("=" * 60)

    n_families = len(CWE_FAMILIES)
    strategies = ["token_guided", "diff_based", "random"]

    # ── Build slices for each strategy
    slices_by_strategy: dict[str, list[tuple[nx.MultiDiGraph, str]]] = {
        s: [] for s in strategies
    }

    token_hit_count = 0
    total_count = 0

    for item in prepared:
        entry = item["entry"]
        G_before = item["G_before"]
        G_after = item["G_after"]
        family = entry.get("cwe_family")
        if not family:
            continue

        total_count += 1

        # Strategy 1: Token-guided (all discriminative tokens, depth=2)
        G_token = compute_token_guided_slice_all(G_before, slice_depth=2)
        if G_token.number_of_nodes() >= 3:
            slices_by_strategy["token_guided"].append((G_token, family))
            token_hit_count += 1

        # Strategy 2: Diff-based (existing approach, depth=2)
        G_diff = compute_graph_diff_configurable(G_before, G_after, slice_depth=2)
        if G_diff.number_of_nodes() >= 3:
            slices_by_strategy["diff_based"].append((G_diff, family))

        # Strategy 3: Random slice (match token-guided size)
        target_size = G_token.number_of_nodes() if G_token.number_of_nodes() >= 3 else 15
        G_rand = _random_slice(G_before, target_size, seed=SEED + total_count)
        if G_rand.number_of_nodes() >= 3:
            slices_by_strategy["random"].append((G_rand, family))

    token_hit_rate = token_hit_count / total_count if total_count > 0 else 0
    print(f"\n  Token hit rate: {token_hit_count}/{total_count} = {token_hit_rate:.1%}")
    for s in strategies:
        print(f"  {s}: {len(slices_by_strategy[s])} valid slices")

    # ── Embed and cluster for each strategy × embedder
    all_cluster_results = []

    for strategy in strategies:
        slices = slices_by_strategy[strategy]
        if len(slices) < 10:
            print(f"\n  SKIP {strategy} — only {len(slices)} slices (need ≥10)")
            continue

        graphs = [g for g, _ in slices]
        true_labels = [lbl for _, lbl in slices]

        # Encode labels to ints
        label_set = sorted(set(true_labels))
        label_to_int = {l: i for i, l in enumerate(label_set)}
        y_true = np.array([label_to_int[l] for l in true_labels])

        # Slice size stats
        sizes = [g.number_of_nodes() for g in graphs]
        print(f"\n  ── {strategy} (n={len(graphs)}, mean_nodes={np.mean(sizes):.1f}) ──")

        for embedder in embedders:
            try:
                # Reset embedder state
                if hasattr(embedder, "_fitted"):
                    embedder._fitted = False
                if hasattr(embedder, "_pca"):
                    embedder._pca = None

                embeddings = embedder.embed_many(graphs)

                # Check for degenerate
                norms = np.linalg.norm(embeddings, axis=1)
                if np.sum(norms < 1e-6) == len(embeddings):
                    print(f"    [{embedder.name}] SKIP — all zero embeddings")
                    continue

                # K-means clustering
                km = KMeans(n_clusters=n_families, random_state=SEED, n_init=10)
                y_pred = km.fit_predict(embeddings)

                ari = adjusted_rand_score(y_true, y_pred)
                homogeneity = homogeneity_score(y_true, y_pred)
                nmi = normalized_mutual_info_score(y_true, y_pred)

                print(f"    [{embedder.name}] ARI={ari:.3f}  Homogeneity={homogeneity:.3f}  NMI={nmi:.3f}")

                all_cluster_results.append({
                    "strategy": strategy,
                    "embedder": embedder.name,
                    "n_samples": len(graphs),
                    "n_clusters": n_families,
                    "mean_slice_nodes": round(float(np.mean(sizes)), 1),
                    "ari": round(ari, 4),
                    "homogeneity": round(homogeneity, 4),
                    "nmi": round(nmi, 4),
                })

            except Exception as e:
                print(f"    [{embedder.name}] ERROR: {type(e).__name__}: {e}")
                all_cluster_results.append({
                    "strategy": strategy,
                    "embedder": embedder.name,
                    "error": str(e),
                })

    # ── Summary comparison
    print(f"\n{'=' * 70}")
    print("  TOKEN-GUIDED SLICING — CLUSTERING RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Strategy':<16s} {'Embedder':<20s} {'ARI':>7s} {'Homog':>7s} {'NMI':>7s} {'N':>5s}")
    print(f"  {'─' * 62}")
    for r in all_cluster_results:
        if "error" in r:
            continue
        print(f"  {r['strategy']:<16s} {r['embedder']:<20s} "
              f"{r['ari']:>7.3f} {r['homogeneity']:>7.3f} {r['nmi']:>7.3f} {r['n_samples']:>5d}")

    results = {
        "token_hit_rate": round(token_hit_rate, 4),
        "n_total": total_count,
        "strategy_counts": {s: len(slices_by_strategy[s]) for s in strategies},
        "clustering_results": all_cluster_results,
    }

    save_json(results, run_dir / "token_guided_clustering.json")
    print(f"\n  Saved to: {run_dir / 'token_guided_clustering.json'}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--n-index", type=int, default=100, help="Number of index pairs")
    parser.add_argument("--n-query", type=int, default=20, help="Number of query pairs")
    parser.add_argument("--diff-only", action="store_true",
                        help="Run only the diff-line classification/clustering (skip graph embedding)")
    parser.add_argument("--token-slice", action="store_true",
                        help="Run token-guided slicing experiment (requires Joern-generated CPGs)")
    args = parser.parse_args()

    if args.diff_only:
        # Quick mode: just run diff classification on the JSON data
        with open(JSON_PATH) as f:
            data = json.load(f)
        entries = select_by_family(data["entries"], samples_per_family=50, seed=SEED)
        _, run_dir = make_run_dir("diff_classify")
        run_diff_classification(entries, run_dir)
    elif args.token_slice:
        # Token-guided slicing only (needs pre-generated CPGs)
        import yaml
        cfg_file = args.config
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        emb_cfg = cfg["embeddings"]
        embedders = []
        for name in TOKEN_SLICE_EMBEDDER_NAMES:
            if name in EMBEDDER_REGISTRY:
                embedders.append(EMBEDDER_REGISTRY[name](emb_cfg))
        n_total = N_FAMILIES * SAMPLES_PER_FAMILY
        prepared = prepare_dataset(n_total, seed=SEED)
        _, run_dir = make_run_dir("token_slice")
        run_token_guided_slicing(prepared, embedders, run_dir)
    else:
        run_experiment(n_index=args.n_index, n_query=args.n_query, cfg_path=args.config)


if __name__ == "__main__":
    main()
