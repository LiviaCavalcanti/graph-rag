"""
Compare graph properties between CVEfixes and AutoPatch datasets.

Analyzes:
  - Node type distribution
  - Graph size (node/edge counts)
  - Edge type distribution
  - Intra-CWE and intra-CVE graph similarity
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.pipeline import load_cpg_dir


# ── Config ────────────────────────────────────────────────────────────

AUTOPATCH_GRAPHML = Path("graphml_augmented")
CVEFIXES_GRAPHML = Path("graphml_cvefixes_fixed")
CVEFIXES_CACHED = Path("workspace/graphfdg_cwe_train")  # Properly Joern-generated
AUTOPATCH_CVE_LIST = Path("CVE-list")
MIN_NODES = 10  # Filter scaffold-only graphs


# ── Graph loading ─────────────────────────────────────────────────────

def load_autopatch_graphs() -> list[dict]:
    """Load AutoPatch original graphs with metadata."""
    results = []
    for cve_dir in sorted(AUTOPATCH_GRAPHML.iterdir()):
        if not cve_dir.is_dir():
            continue
        before_dir = cve_dir / "original" / "before" / "graph"
        if not before_dir.exists():
            continue
        try:
            G = load_cpg_dir(str(before_dir))
        except Exception:
            continue
        if G.number_of_nodes() < 3:
            continue

        # Get CWE from info.json
        cve_id = cve_dir.name
        cwe_id = "UNKNOWN"
        info_path = AUTOPATCH_CVE_LIST / cve_id / "info.json"
        if not info_path.exists():
            # Try suffix matching
            for d in AUTOPATCH_CVE_LIST.iterdir():
                if d.is_dir() and d.name.startswith(cve_id):
                    info_path = d / "info.json"
                    break
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
                cwe_id = info.get("cwe_id", info.get("cwe_type", "UNKNOWN"))
            except Exception:
                pass

        results.append({
            "cve_id": cve_id,
            "cwe_id": cwe_id,
            "graph": G,
            "dataset": "AutoPatch",
        })
    return results


def load_cvefixes_graphs() -> list[dict]:
    """Load CVEfixes graphs from cached Joern-generated CPGs."""
    # Load CWE family mapping
    CWE_FAMILIES = {
        "memory_corruption": ["CWE-119","CWE-120","CWE-121","CWE-122","CWE-125","CWE-787","CWE-805"],
        "use_after_free": ["CWE-416","CWE-415","CWE-763"],
        "integer_issues": ["CWE-190","CWE-191","CWE-189","CWE-681","CWE-369"],
        "null_ptr_deref": ["CWE-476","CWE-824"],
        "race_condition": ["CWE-362","CWE-367","CWE-667"],
    }
    CWE_TO_FAMILY = {}
    for fam, cwes in CWE_FAMILIES.items():
        for cwe in cwes:
            CWE_TO_FAMILY[cwe] = fam

    # Load CWE mapping from filtered JSON
    cwe_map = {}
    for json_path in [
        Path("cvefixes_filtered_by_cwe.json"),
        Path("cvefixes_experiments/data/cvefixes_filtered_by_cwe.json"),
    ]:
        if json_path.exists():
            data = json.load(open(json_path))
            for e in data["entries"]:
                cve_id = e["cve_id"]
                cwes = [c["cwe_id"] for c in e.get("cwe", [])]
                if cwes:
                    cwe_map[cve_id] = cwes[0]
            break

    results = []
    # Use cached Joern-generated graphs (from the training experiment)
    if CVEFIXES_CACHED.exists():
        for entry_dir in sorted(CVEFIXES_CACHED.iterdir()):
            if not entry_dir.is_dir():
                continue
            before_dir = entry_dir / "before" / "graph"
            if not before_dir.exists():
                continue
            try:
                G = load_cpg_dir(str(before_dir))
            except Exception:
                continue
            if G.number_of_nodes() < MIN_NODES:
                continue

            # Parse dir name: 0383_CVE-2023-3863_llcp_sock_connect
            parts = entry_dir.name.split("_", 2)
            cve_id = parts[1] if len(parts) > 1 else entry_dir.name
            # Try to find CVE-YYYY-NNNNN pattern
            import re
            m = re.search(r"(CVE-\d{4}-\d+)", entry_dir.name)
            if m:
                cve_id = m.group(1)
            cwe_id = cwe_map.get(cve_id, "UNKNOWN")

            results.append({
                "cve_id": cve_id,
                "cwe_id": cwe_id,
                "graph": G,
                "dataset": "CVEfixes",
            })
    return results


# ── Analysis functions ────────────────────────────────────────────────

def analyze_node_types(graphs: list[dict]) -> dict:
    """Compute node type distribution."""
    type_counter = Counter()
    total_nodes = 0
    for g in graphs:
        for _, attr in g["graph"].nodes(data=True):
            label = attr.get("labelV", "UNKNOWN")
            type_counter[label] += 1
            total_nodes += 1
    # Normalize
    dist = {k: v / total_nodes for k, v in type_counter.most_common(20)}
    return {"distribution": dist, "total_nodes": total_nodes, "unique_types": len(type_counter)}


def analyze_edge_types(graphs: list[dict]) -> dict:
    """Compute edge type distribution."""
    type_counter = Counter()
    total_edges = 0
    for g in graphs:
        for _, _, data in g["graph"].edges(data=True):
            label = data.get("labelE") or data.get("label", "UNKNOWN")
            type_counter[label] += 1
            total_edges += 1
    dist = {k: v / total_edges for k, v in type_counter.most_common(20)}
    return {"distribution": dist, "total_edges": total_edges, "unique_types": len(type_counter)}


def analyze_graph_sizes(graphs: list[dict]) -> dict:
    """Compute graph size statistics."""
    nodes = [g["graph"].number_of_nodes() for g in graphs]
    edges = [g["graph"].number_of_edges() for g in graphs]
    degrees = []
    for g in graphs:
        G = g["graph"]
        if G.number_of_nodes() > 0:
            degrees.append(G.number_of_edges() / G.number_of_nodes())

    return {
        "n_graphs": len(graphs),
        "nodes": {
            "mean": np.mean(nodes),
            "median": np.median(nodes),
            "std": np.std(nodes),
            "min": int(np.min(nodes)),
            "max": int(np.max(nodes)),
            "q25": np.percentile(nodes, 25),
            "q75": np.percentile(nodes, 75),
        },
        "edges": {
            "mean": np.mean(edges),
            "median": np.median(edges),
            "std": np.std(edges),
            "min": int(np.min(edges)),
            "max": int(np.max(edges)),
            "q25": np.percentile(edges, 25),
            "q75": np.percentile(edges, 75),
        },
        "avg_degree": {
            "mean": np.mean(degrees),
            "median": np.median(degrees),
            "std": np.std(degrees),
        },
    }


def compute_graph_fingerprint(G: nx.MultiDiGraph) -> np.ndarray:
    """Compute a lightweight fingerprint for graph similarity comparison.
    Uses node-type histogram + edge-type histogram (normalized).
    """
    node_types = [
        "METHOD", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD_RETURN",
        "BLOCK", "LOCAL", "CALL", "IDENTIFIER", "LITERAL", "RETURN",
        "CONTROL_STRUCTURE", "FIELD_IDENTIFIER", "UNKNOWN",
    ]
    edge_types = ["AST", "CFG", "CDG", "REACHING_DEF", "REF", "ARGUMENT", "RECEIVER", "CALL"]

    type_idx = {t: i for i, t in enumerate(node_types)}
    etype_idx = {t: i for i, t in enumerate(edge_types)}

    node_hist = np.zeros(len(node_types), dtype=np.float32)
    for _, attr in G.nodes(data=True):
        label = attr.get("labelV", "UNKNOWN")
        node_hist[type_idx.get(label, type_idx["UNKNOWN"])] += 1
    n_nodes = node_hist.sum()
    if n_nodes > 0:
        node_hist /= n_nodes

    edge_hist = np.zeros(len(edge_types), dtype=np.float32)
    for _, _, data in G.edges(data=True):
        label = data.get("labelE") or data.get("label", "")
        idx = etype_idx.get(label, -1)
        if idx >= 0:
            edge_hist[idx] += 1
    n_edges = edge_hist.sum()
    if n_edges > 0:
        edge_hist /= n_edges

    return np.concatenate([node_hist, edge_hist])


def analyze_similarity(graphs: list[dict]) -> dict:
    """Compute intra-CWE and intra-CVE similarity using graph fingerprints."""
    # Compute fingerprints
    fingerprints = np.array([compute_graph_fingerprint(g["graph"]) for g in graphs])

    # Normalize for cosine similarity
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    fingerprints_norm = fingerprints / norms

    # Overall pairwise similarity
    n = len(fingerprints_norm)
    if n > 500:
        # Sample to keep it tractable
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=min(500, n), replace=False)
        sample = fingerprints_norm[sample_idx]
    else:
        sample = fingerprints_norm
        sample_idx = np.arange(n)

    sim_matrix = sample @ sample.T
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones(sim_matrix.shape, dtype=bool), k=1)
    overall_sims = sim_matrix[mask]

    # Intra-CWE similarity
    cwe_groups = defaultdict(list)
    for i, g in enumerate(graphs):
        if g["cwe_id"] != "UNKNOWN":
            cwe_groups[g["cwe_id"]].append(i)

    intra_cwe_sims = []
    inter_cwe_sims = []
    for cwe, indices in cwe_groups.items():
        if len(indices) < 2:
            continue
        fps = fingerprints_norm[indices]
        sim = fps @ fps.T
        m = np.triu(np.ones(sim.shape, dtype=bool), k=1)
        intra_cwe_sims.extend(sim[m].tolist())

    # Inter-CWE: sample pairs from different CWEs
    cwe_list = [c for c, idx in cwe_groups.items() if len(idx) >= 2]
    rng = np.random.default_rng(42)
    for _ in range(min(5000, len(graphs) * 10)):
        if len(cwe_list) < 2:
            break
        c1, c2 = rng.choice(cwe_list, size=2, replace=False)
        i1 = rng.choice(cwe_groups[c1])
        i2 = rng.choice(cwe_groups[c2])
        sim = float(fingerprints_norm[i1] @ fingerprints_norm[i2])
        inter_cwe_sims.append(sim)

    # Intra-CVE similarity
    cve_groups = defaultdict(list)
    for i, g in enumerate(graphs):
        cve_groups[g["cve_id"]].append(i)

    intra_cve_sims = []
    for cve, indices in cve_groups.items():
        if len(indices) < 2:
            continue
        fps = fingerprints_norm[indices]
        sim = fps @ fps.T
        m = np.triu(np.ones(sim.shape, dtype=bool), k=1)
        intra_cve_sims.extend(sim[m].tolist())

    return {
        "overall": {
            "mean": float(np.mean(overall_sims)),
            "std": float(np.std(overall_sims)),
            "median": float(np.median(overall_sims)),
        },
        "intra_cwe": {
            "mean": float(np.mean(intra_cwe_sims)) if intra_cwe_sims else None,
            "std": float(np.std(intra_cwe_sims)) if intra_cwe_sims else None,
            "n_pairs": len(intra_cwe_sims),
            "n_cwes": len([c for c in cwe_groups if len(cwe_groups[c]) >= 2]),
        },
        "inter_cwe": {
            "mean": float(np.mean(inter_cwe_sims)) if inter_cwe_sims else None,
            "std": float(np.std(inter_cwe_sims)) if inter_cwe_sims else None,
            "n_pairs": len(inter_cwe_sims),
        },
        "intra_cve": {
            "mean": float(np.mean(intra_cve_sims)) if intra_cve_sims else None,
            "std": float(np.std(intra_cve_sims)) if intra_cve_sims else None,
            "n_pairs": len(intra_cve_sims),
            "n_cves": len([c for c in cve_groups if len(cve_groups[c]) >= 2]),
        },
        "discriminability": {
            "cwe_gap": (
                (float(np.mean(intra_cwe_sims)) - float(np.mean(inter_cwe_sims)))
                if intra_cwe_sims and inter_cwe_sims else None
            ),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GRAPH DISTRIBUTION ANALYSIS: CVEfixes vs AutoPatch")
    print("=" * 70)

    # Load graphs
    print("\n[1/5] Loading AutoPatch graphs...")
    autopatch = load_autopatch_graphs()
    print(f"  Loaded {len(autopatch)} graphs")

    print("\n[2/5] Loading CVEfixes graphs...")
    cvefixes = load_cvefixes_graphs()
    print(f"  Loaded {len(cvefixes)} graphs")

    # Analyze
    print("\n[3/5] Analyzing graph sizes...")
    ap_sizes = analyze_graph_sizes(autopatch)
    cf_sizes = analyze_graph_sizes(cvefixes)

    print("\n[4/5] Analyzing node/edge type distributions...")
    ap_nodes = analyze_node_types(autopatch)
    cf_nodes = analyze_node_types(cvefixes)
    ap_edges = analyze_edge_types(autopatch)
    cf_edges = analyze_edge_types(cvefixes)

    print("\n[5/5] Computing graph similarities...")
    ap_sim = analyze_similarity(autopatch)
    cf_sim = analyze_similarity(cvefixes)

    # ── Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Graph sizes
    print(f"\n{'─'*70}")
    print(f"  GRAPH SIZES")
    print(f"{'─'*70}")
    print(f"  {'Metric':<20s} {'AutoPatch':>15s} {'CVEfixes':>15s}")
    print(f"  {'─'*50}")
    print(f"  {'N graphs':<20s} {ap_sizes['n_graphs']:>15d} {cf_sizes['n_graphs']:>15d}")
    print(f"  {'Nodes (mean)':<20s} {ap_sizes['nodes']['mean']:>15.1f} {cf_sizes['nodes']['mean']:>15.1f}")
    print(f"  {'Nodes (median)':<20s} {ap_sizes['nodes']['median']:>15.1f} {cf_sizes['nodes']['median']:>15.1f}")
    print(f"  {'Nodes (std)':<20s} {ap_sizes['nodes']['std']:>15.1f} {cf_sizes['nodes']['std']:>15.1f}")
    print(f"  {'Nodes (min-max)':<20s} {ap_sizes['nodes']['min']:>6d}–{ap_sizes['nodes']['max']:<8d} {cf_sizes['nodes']['min']:>6d}–{cf_sizes['nodes']['max']:<8d}")
    print(f"  {'Edges (mean)':<20s} {ap_sizes['edges']['mean']:>15.1f} {cf_sizes['edges']['mean']:>15.1f}")
    print(f"  {'Edges (median)':<20s} {ap_sizes['edges']['median']:>15.1f} {cf_sizes['edges']['median']:>15.1f}")
    print(f"  {'Edges (std)':<20s} {ap_sizes['edges']['std']:>15.1f} {cf_sizes['edges']['std']:>15.1f}")
    print(f"  {'Avg degree (mean)':<20s} {ap_sizes['avg_degree']['mean']:>15.2f} {cf_sizes['avg_degree']['mean']:>15.2f}")

    # Node types
    print(f"\n{'─'*70}")
    print(f"  NODE TYPE DISTRIBUTION (top 10)")
    print(f"{'─'*70}")
    all_types = sorted(
        set(list(ap_nodes['distribution'].keys()) + list(cf_nodes['distribution'].keys())),
        key=lambda t: -(ap_nodes['distribution'].get(t, 0) + cf_nodes['distribution'].get(t, 0))
    )[:10]
    print(f"  {'Type':<25s} {'AutoPatch':>12s} {'CVEfixes':>12s} {'Δ':>8s}")
    print(f"  {'─'*57}")
    for t in all_types:
        ap_v = ap_nodes['distribution'].get(t, 0)
        cf_v = cf_nodes['distribution'].get(t, 0)
        delta = cf_v - ap_v
        print(f"  {t:<25s} {ap_v:>11.1%} {cf_v:>11.1%} {delta:>+7.1%}")

    # Edge types
    print(f"\n{'─'*70}")
    print(f"  EDGE TYPE DISTRIBUTION (top 10)")
    print(f"{'─'*70}")
    all_etypes = sorted(
        set(list(ap_edges['distribution'].keys()) + list(cf_edges['distribution'].keys())),
        key=lambda t: -(ap_edges['distribution'].get(t, 0) + cf_edges['distribution'].get(t, 0))
    )[:10]
    print(f"  {'Type':<25s} {'AutoPatch':>12s} {'CVEfixes':>12s} {'Δ':>8s}")
    print(f"  {'─'*57}")
    for t in all_etypes:
        ap_v = ap_edges['distribution'].get(t, 0)
        cf_v = cf_edges['distribution'].get(t, 0)
        delta = cf_v - ap_v
        print(f"  {t:<25s} {ap_v:>11.1%} {cf_v:>11.1%} {delta:>+7.1%}")

    # Similarity
    print(f"\n{'─'*70}")
    print(f"  GRAPH SIMILARITY (cosine on node+edge type fingerprint)")
    print(f"{'─'*70}")
    print(f"  {'Metric':<30s} {'AutoPatch':>15s} {'CVEfixes':>15s}")
    print(f"  {'─'*60}")
    print(f"  {'Overall pairwise (mean)':<30s} {ap_sim['overall']['mean']:>15.4f} {cf_sim['overall']['mean']:>15.4f}")
    print(f"  {'Overall pairwise (std)':<30s} {ap_sim['overall']['std']:>15.4f} {cf_sim['overall']['std']:>15.4f}")
    if ap_sim['intra_cwe']['mean'] is not None:
        print(f"  {'Intra-CWE (mean)':<30s} {ap_sim['intra_cwe']['mean']:>15.4f} {cf_sim['intra_cwe']['mean']:>15.4f}")
        print(f"  {'Intra-CWE (n_pairs)':<30s} {ap_sim['intra_cwe']['n_pairs']:>15d} {cf_sim['intra_cwe']['n_pairs']:>15d}")
        print(f"  {'Intra-CWE (n_cwes)':<30s} {ap_sim['intra_cwe']['n_cwes']:>15d} {cf_sim['intra_cwe']['n_cwes']:>15d}")
    if ap_sim['inter_cwe']['mean'] is not None:
        print(f"  {'Inter-CWE (mean)':<30s} {ap_sim['inter_cwe']['mean']:>15.4f} {cf_sim['inter_cwe']['mean']:>15.4f}")
    if ap_sim['intra_cve']['mean'] is not None or cf_sim['intra_cve']['mean'] is not None:
        ap_cve = f"{ap_sim['intra_cve']['mean']:.4f}" if ap_sim['intra_cve']['mean'] else "N/A"
        cf_cve = f"{cf_sim['intra_cve']['mean']:.4f}" if cf_sim['intra_cve']['mean'] else "N/A"
        print(f"  {'Intra-CVE (mean)':<30s} {ap_cve:>15s} {cf_cve:>15s}")
        ap_n = str(ap_sim['intra_cve']['n_cves']) if ap_sim['intra_cve']['n_cves'] else "0"
        cf_n = str(cf_sim['intra_cve']['n_cves']) if cf_sim['intra_cve']['n_cves'] else "0"
        print(f"  {'Intra-CVE (n_cves w/ ≥2)':<30s} {ap_n:>15s} {cf_n:>15s}")
    if ap_sim['discriminability']['cwe_gap'] is not None:
        print(f"  {'CWE gap (intra-inter)':<30s} {ap_sim['discriminability']['cwe_gap']:>15.4f} {cf_sim['discriminability']['cwe_gap']:>15.4f}")

    # Interpretation
    print(f"\n{'─'*70}")
    print(f"  INTERPRETATION")
    print(f"{'─'*70}")

    # Size comparison
    size_ratio = cf_sizes['nodes']['mean'] / ap_sizes['nodes']['mean']
    if size_ratio > 1.5:
        print(f"  • CVEfixes graphs are {size_ratio:.1f}x LARGER on average")
    elif size_ratio < 0.67:
        print(f"  • CVEfixes graphs are {1/size_ratio:.1f}x SMALLER on average")
    else:
        print(f"  • Similar graph sizes (ratio={size_ratio:.2f})")

    # Similarity comparison
    if cf_sim['overall']['mean'] > ap_sim['overall']['mean'] + 0.05:
        print(f"  • CVEfixes has HIGHER internal similarity ({cf_sim['overall']['mean']:.3f} vs {ap_sim['overall']['mean']:.3f})")
        print(f"    → More homogeneous graph structure (explains GIN collapse)")
    elif ap_sim['overall']['mean'] > cf_sim['overall']['mean'] + 0.05:
        print(f"  • AutoPatch has HIGHER internal similarity")
    else:
        print(f"  • Similar internal similarity levels")

    # CWE discriminability
    ap_gap = ap_sim['discriminability']['cwe_gap']
    cf_gap = cf_sim['discriminability']['cwe_gap']
    if ap_gap is not None and cf_gap is not None:
        if ap_gap > cf_gap + 0.02:
            print(f"  • AutoPatch CWEs are MORE separable by structure (gap={ap_gap:.4f} vs {cf_gap:.4f})")
        elif cf_gap > ap_gap + 0.02:
            print(f"  • CVEfixes CWEs are MORE separable by structure (gap={cf_gap:.4f} vs {ap_gap:.4f})")
        else:
            print(f"  • Similar CWE separability")

    # Save JSON
    output = {
        "autopatch": {
            "sizes": ap_sizes,
            "node_types": ap_nodes,
            "edge_types": ap_edges,
            "similarity": ap_sim,
        },
        "cvefixes": {
            "sizes": cf_sizes,
            "node_types": cf_nodes,
            "edge_types": cf_edges,
            "similarity": cf_sim,
        },
    }
    out_path = Path("cvefixes_experiments/output/graph_distribution_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\n  Full results saved → {out_path}")


if __name__ == "__main__":
    main()
