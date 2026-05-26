"""
Experiment: KB-Guided Sink Ranking (Precision Improvement via Graph Retrieval)

Problem:
  Joern + regex sink patterns find many candidate sink nodes in a CPG,
  but most are FALSE positives (they match the pattern but aren't the
  actual vulnerability). E.g., in a 437-node CPG, 69 nodes match the
  sink pattern but only 5 overlap with the actual vulnerability fix.

Hypothesis:
  A knowledge base (KB) of confirmed vulnerability subgraphs can help
  RANK which candidate sinks are most likely to be real vulnerabilities.
  If the local graph neighbourhood around a candidate sink is "similar"
  to a known vulnerability in the KB, it's more likely to be a true positive.

Protocol:
  1. For each of the 87 verified entries:
     a) Build CPG from code_before
     b) Find ALL sink candidate nodes (regex match)
     c) Label each: TRUE (code overlaps with changed/fixed lines) or FALSE
     d) Extract k-hop subgraph around each candidate sink node
  2. For each entry (leave-one-out):
     a) Build KB from the TRUE-positive subgraphs of all OTHER entries
     b) For each candidate sink in the held-out entry:
        compute max cosine similarity to any KB subgraph
     c) Use this similarity score to rank TRUE above FALSE candidates
  3. Evaluate ranking quality:
     - AUC-ROC: can the KB similarity separate TRUE from FALSE sinks?
     - Precision@K: if we take the top-K ranked sinks, what fraction are TRUE?
     - Reduction factor: how much does KB filtering reduce false positives?

This directly tests whether graph-based retrieval improves vulnerability
detection precision beyond what Joern's pattern matching alone achieves.
"""

import json
import re
import difflib
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.pipeline import load_cpg_dir, run_joern_export, write_c_file
from src.embeddings.vuln_pattern import (
    RE_PTR_DEREF,
    RE_ALLOC,
    RE_FREE,
    RE_LOCK,
    RE_UNLOCK,
    RE_NULL_CHECK,
    RE_ARITH,
    RE_BOUNDS,
    RE_CAST,
    RE_CHECK,
)

# ── Configuration ──
RANDOM_SEED = 42
JOERN_BIN_DIR = "/home/z0050s2b/bin/joern/joern-cli"
SUBSET_FILE = "cvefixes_experiments/output/joern_sink_subset_100.json"
RESULTS_FILE = "cvefixes_experiments/output/joern_sink_results.json"
OUTPUT_FILE = "cvefixes_experiments/output/joern_kb_ranking_results.json"
SINK_HOP_RADIUS = 2  # k-hop subgraph around each candidate sink
EMBEDDING_DIM = 36
EMBEDDER = "codebert"  # will switch to "codebert" in next iteration

# ── Sink patterns (per CWE) ──
SINK_PATTERNS = {
    "CWE-416": [re.compile(r"\b(free|kfree|vfree|kvfree|delete|release)\b"), re.compile(r"\*\w|->")],
    "CWE-476": [re.compile(r"\*\w|->"), re.compile(r"\b(NULL|null|nullptr)\b"), re.compile(r"\b(memcpy|memmove|memset)\b")],
    "CWE-190": [re.compile(r"[+\-*/]|<<|>>"), re.compile(r"\b(int|long|short|unsigned|size_t)\s+\w+")],
    "CWE-191": [re.compile(r"[+\-*/]|<<|>>")],
    "CWE-20": [re.compile(r"\b(if|assert|BUG_ON|WARN_ON|check|verify|IS_ERR)\b"), re.compile(r"->")],
    "CWE-362": [re.compile(r"->"), re.compile(r"\b(memcpy|memmove|copy_from_user|copy_to_user)\b"), re.compile(r"\*\w")],
    "CWE-400": [re.compile(r"\b(malloc|calloc|kmalloc|kzalloc|alloc)\b"), re.compile(r"->"), re.compile(r"\b(perf_|event_|overflow)\w*\b")],
    "CWE-401": [re.compile(r"\b(malloc|calloc|kmalloc|kzalloc|alloc|skb)\b"), re.compile(r"->")],
    "CWE-787": [re.compile(r"[+\-*/]|<<|>>"), re.compile(r"\*\w|->")],
    "CWE-667": [re.compile(r"->"), re.compile(r"\*\w")],
}

DEFAULT_SINK_PATTERNS = [
    re.compile(r"\b(free|kfree|vfree|kvfree|delete|release)\b"),
    re.compile(r"\b(malloc|calloc|kmalloc|kzalloc|alloc)\b"),
    re.compile(r"\*\w|->"),
    re.compile(r"\b(if|assert|BUG_ON|WARN_ON|check|verify|IS_ERR)\b"),
    re.compile(r"[+\-*/]|<<|>>"),
]


# ── Helpers ──


def get_changed_lines(code_before: str, code_after: str) -> set[str]:
    """Return the set of changed line contents (both removed and added, stripped)."""
    before_lines = code_before.split("\n")
    after_lines = code_after.split("\n")
    diff = list(difflib.unified_diff(before_lines, after_lines, lineterm=""))
    changed = set()
    for line in diff:
        if line.startswith("-") and not line.startswith("---"):
            changed.add(line[1:].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            changed.add(line[1:].strip())
    return {l for l in changed if l}  # remove empty


def find_all_sink_nodes(G: nx.MultiDiGraph, cwe_id: str) -> list[dict]:
    """
    Find ALL nodes matching sink patterns. Returns list of dicts with
    node_id, code, label.
    """
    patterns = SINK_PATTERNS.get(cwe_id, DEFAULT_SINK_PATTERNS)
    sinks = []
    for n, attr in G.nodes(data=True):
        code = (attr.get("CODE") or "").strip()
        if not code:
            continue
        label = attr.get("labelV", "")
        if label in ("METHOD", "BLOCK", "METHOD_RETURN"):
            continue
        for pat in patterns:
            if pat.search(code):
                sinks.append({"node_id": n, "code": code, "label": label})
                break
    return sinks


def label_sinks(sinks: list[dict], changed_lines: set[str]) -> list[dict]:
    """
    Label each sink as TRUE (its code overlaps a changed line) or FALSE.
    This is the ground truth: TRUE sinks are confirmed vulnerability sites.
    """
    for sink in sinks:
        sink_code = sink["code"]
        is_true = any(
            sink_code in cl or cl in sink_code
            for cl in changed_lines
        )
        sink["is_true"] = is_true
    return sinks


def extract_node_subgraph(G: nx.MultiDiGraph, center_node, k_hop: int) -> nx.MultiDiGraph:
    """Extract k-hop neighbourhood around a single node."""
    neighbourhood = {center_node}
    frontier = {center_node}
    for _ in range(k_hop):
        next_frontier = set()
        for node in frontier:
            for _, succ in G.out_edges(node):
                if succ not in neighbourhood:
                    next_frontier.add(succ)
            for pred, _ in G.in_edges(node):
                if pred not in neighbourhood:
                    next_frontier.add(pred)
        neighbourhood |= next_frontier
        frontier = next_frontier
    return G.subgraph(neighbourhood).copy()


# ── Subgraph embedding (lightweight, no external model needed) ──


def embed_subgraph(G: nx.MultiDiGraph) -> np.ndarray:
    """
    Embed a small subgraph into a fixed-size vector.
    Combines: node-type distribution + edge-type distribution +
    code-pattern density + structural stats.
    Total: 36 dimensions.
    """
    # Node type histogram (13 dims)
    node_types = [
        "METHOD", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD_RETURN",
        "BLOCK", "LOCAL", "CALL", "IDENTIFIER", "LITERAL", "RETURN",
        "CONTROL_STRUCTURE", "FIELD_IDENTIFIER", "UNKNOWN",
    ]
    type_idx = {t: i for i, t in enumerate(node_types)}
    node_hist = np.zeros(len(node_types), dtype=np.float32)
    for _, attr in G.nodes(data=True):
        label = attr.get("labelV", "UNKNOWN")
        node_hist[type_idx.get(label, type_idx["UNKNOWN"])] += 1
    n_nodes = node_hist.sum()
    if n_nodes > 0:
        node_hist /= n_nodes

    # Edge type histogram (8 dims)
    edge_types = ["AST", "CFG", "CDG", "REACHING_DEF", "REF", "ARGUMENT", "RECEIVER", "CALL"]
    etype_idx = {t: i for i, t in enumerate(edge_types)}
    edge_hist = np.zeros(len(edge_types), dtype=np.float32)
    for _, _, data in G.edges(data=True):
        label = data.get("labelE") or data.get("label", "")
        idx = etype_idx.get(label, -1)
        if idx >= 0:
            edge_hist[idx] += 1
    n_edges = edge_hist.sum()
    if n_edges > 0:
        edge_hist /= n_edges

    # Code pattern density (10 dims)
    patterns = [RE_PTR_DEREF, RE_ALLOC, RE_FREE, RE_LOCK, RE_UNLOCK,
                RE_NULL_CHECK, RE_ARITH, RE_BOUNDS, RE_CAST, RE_CHECK]
    code_pats = np.zeros(len(patterns), dtype=np.float32)
    code_nodes = 0
    for _, attr in G.nodes(data=True):
        code = (attr.get("CODE") or "").strip()
        if not code:
            continue
        code_nodes += 1
        for i, pat in enumerate(patterns):
            if pat.search(code):
                code_pats[i] += 1
    if code_nodes > 0:
        code_pats /= code_nodes

    # Structural stats (5 dims)
    n_n = G.number_of_nodes()
    n_e = G.number_of_edges()
    avg_deg = n_e / n_n if n_n > 0 else 0
    n_calls = sum(1 for _, a in G.nodes(data=True) if a.get("labelV") == "CALL")
    n_ctrl = sum(1 for _, a in G.nodes(data=True) if a.get("labelV") == "CONTROL_STRUCTURE")
    stats = np.array([
        min(n_n / 50.0, 1.0),
        min(n_e / 200.0, 1.0),
        min(avg_deg / 8.0, 1.0),
        n_calls / max(n_n, 1),
        n_ctrl / max(n_n, 1),
    ], dtype=np.float32)

    return np.concatenate([node_hist, edge_hist, code_pats, stats])


def embed_subgraph_codebert(G: nx.MultiDiGraph, tokenizer, model) -> np.ndarray:
    """
    Embed a subgraph using CodeBERT on the concatenated code of its nodes.
    Returns a 768-dim vector (CodeBERT hidden size).
    """
    import torch

    # Collect code from all nodes, ordered by node id for reproducibility
    code_parts = []
    for n in sorted(G.nodes()):
        code = (G.nodes[n].get("CODE") or "").strip()
        if code:
            code_parts.append(code)
    text = " ".join(code_parts)[:512]  # truncate to model max

    if not text.strip():
        return np.zeros(768, dtype=np.float32)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding
    return outputs.last_hidden_state[0, 0].numpy().astype(np.float32)


def get_embedder():
    """Return the embedding function and dimensionality based on EMBEDDER config."""
    if EMBEDDER == "codebert":
        import os
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", local_files_only=True)
        model = AutoModel.from_pretrained("microsoft/codebert-base", local_files_only=True)
        model.eval()
        def embed_fn(G):
            return embed_subgraph_codebert(G, tokenizer, model)
        return embed_fn, 768
    else:
        return embed_subgraph, EMBEDDING_DIM


# ── Main experiment ──


def run_experiment():
    np.random.seed(RANDOM_SEED)
    print("=" * 70)
    print("EXPERIMENT: KB-Guided Sink Ranking")
    print(f"  Embedder: {EMBEDDER}")
    print("  Can a knowledge base of known vulnerability subgraphs improve")
    print("  precision when ranking Joern-found sink candidates?")
    print("=" * 70)

    # Initialize embedder
    embed_fn, emb_dim = get_embedder()
    print(f"  Embedding dimension: {emb_dim}")

    # Load data
    print("\n[1/4] Loading verified entries...")
    with open(SUBSET_FILE) as f:
        subset = json.load(f)
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # Only use entries where sinks were successfully verified
    hit_keys = set()
    for r in results["details"]:
        if r.get("status") == "success" and r.get("hit"):
            hit_keys.add((r["cve_id"], r["method_name"]))

    entries = [e for e in subset["entries"]
               if (e["cve_id"], e["method_name"]) in hit_keys]
    print(f"  {len(entries)} verified entries loaded")

    # Build CPGs and label sinks
    print(f"\n[2/4] Building CPGs and labelling sink candidates...")
    entry_data = []

    for i, entry in enumerate(entries):
        cve_id = entry["cve_id"]
        cwe_id = entry["cwe"][0]["cwe_id"]
        code_before = entry["code_before"]
        code_after = entry["code_after"]

        # Get changed lines (ground truth)
        changed_lines = get_changed_lines(code_before, code_after)
        if not changed_lines:
            continue

        with tempfile.TemporaryDirectory(prefix=f"kb_{cve_id}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_path = write_c_file(code_before, tmpdir_path / "vuln.c")
            cpg_dir = str(tmpdir_path / "cpg")
            graph_dir = str(tmpdir_path / "graph")

            ok = run_joern_export(JOERN_BIN_DIR, str(src_path), cpg_dir, graph_dir)
            if not ok:
                continue
            try:
                G = load_cpg_dir(graph_dir)
            except Exception:
                continue
            if G.number_of_nodes() < 10:
                continue

            # Find and label ALL sink candidates
            sinks = find_all_sink_nodes(G, cwe_id)
            if not sinks:
                continue
            sinks = label_sinks(sinks, changed_lines)

            n_true = sum(1 for s in sinks if s["is_true"])
            n_false = len(sinks) - n_true
            if n_true == 0:
                continue  # need at least some true sinks

            # Extract subgraph around each sink and embed it
            true_embeddings = []
            false_embeddings = []
            all_embeddings = []
            all_labels = []

            for sink in sinks:
                sub_G = extract_node_subgraph(G, sink["node_id"], SINK_HOP_RADIUS)
                emb = embed_fn(sub_G)
                all_embeddings.append(emb)
                all_labels.append(sink["is_true"])
                if sink["is_true"]:
                    true_embeddings.append(emb)
                else:
                    false_embeddings.append(emb)

            entry_data.append({
                "cve_id": cve_id,
                "cwe_id": cwe_id,
                "n_sinks": len(sinks),
                "n_true": n_true,
                "n_false": n_false,
                "true_embeddings": np.array(true_embeddings),
                "false_embeddings": np.array(false_embeddings) if false_embeddings else np.zeros((0, emb_dim)),
                "all_embeddings": np.array(all_embeddings),
                "all_labels": all_labels,
            })

        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(entries)}")

    print(f"  Built data for {len(entry_data)} entries")
    total_sinks = sum(d["n_sinks"] for d in entry_data)
    total_true = sum(d["n_true"] for d in entry_data)
    total_false = sum(d["n_false"] for d in entry_data)
    print(f"  Total sink candidates: {total_sinks}")
    print(f"  TRUE (actual vuln): {total_true} ({total_true/total_sinks:.1%})")
    print(f"  FALSE (not vuln):   {total_false} ({total_false/total_sinks:.1%})")
    print(f"  Baseline precision (no KB): {total_true/total_sinks:.1%}")

    # Leave-one-out KB ranking
    print(f"\n[3/4] Leave-one-out KB-guided ranking...")

    all_auc_scores = []
    all_ap_scores = []
    all_precision_at_k = {1: [], 3: [], 5: []}
    per_cwe_results = defaultdict(list)
    detailed_results = []

    for hold_idx in range(len(entry_data)):
        held_out = entry_data[hold_idx]

        # Skip if held-out entry has all TRUE or all FALSE (can't compute AUC)
        if held_out["n_true"] == held_out["n_sinks"] or held_out["n_false"] == 0:
            continue

        # Build KB from all other entries' TRUE subgraphs
        kb_embeddings = []
        for j, other in enumerate(entry_data):
            if j == hold_idx:
                continue
            kb_embeddings.append(other["true_embeddings"])

        if not kb_embeddings:
            continue
        kb_matrix = np.vstack(kb_embeddings)  # (KB_size, 36)

        # For each sink candidate in held-out entry, compute similarity to KB
        candidate_embs = held_out["all_embeddings"]  # (n_sinks, 36)
        labels = held_out["all_labels"]

        # Cosine similarity: each candidate vs entire KB
        norms_cand = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8
        norms_kb = np.linalg.norm(kb_matrix, axis=1, keepdims=True) + 1e-8
        sim_matrix = (candidate_embs / norms_cand) @ (kb_matrix / norms_kb).T

        # Score = max similarity to KB (nearest known vulnerability)
        scores = sim_matrix.max(axis=1)

        # Evaluate: can these scores separate TRUE from FALSE?
        labels_binary = np.array(labels, dtype=int)

        try:
            auc = roc_auc_score(labels_binary, scores)
            ap = average_precision_score(labels_binary, scores)
            all_auc_scores.append(auc)
            all_ap_scores.append(ap)
            per_cwe_results[held_out["cwe_id"]].append(auc)
        except ValueError:
            continue

        # Precision@K: rank by score, check top-K
        ranked_indices = np.argsort(-scores)
        for k in [1, 3, 5]:
            if k <= len(ranked_indices):
                top_k_labels = [labels[idx] for idx in ranked_indices[:k]]
                prec = sum(top_k_labels) / k
                all_precision_at_k[k].append(prec)

        detailed_results.append({
            "cve_id": held_out["cve_id"],
            "cwe_id": held_out["cwe_id"],
            "n_sinks": held_out["n_sinks"],
            "n_true": held_out["n_true"],
            "n_false": held_out["n_false"],
            "baseline_precision": held_out["n_true"] / held_out["n_sinks"],
            "auc_roc": float(auc),
            "avg_precision": float(ap),
        })

    # Results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  Entries evaluated: {len(all_auc_scores)}")
    print(f"  (Excluded {len(entry_data) - len(all_auc_scores)} entries with all-TRUE sinks)")

    if all_auc_scores:
        mean_auc = np.mean(all_auc_scores)
        mean_ap = np.mean(all_ap_scores)
        print(f"\n  --- KB-Guided Ranking (max-similarity to nearest KB neighbour) ---")
        print(f"  Mean AUC-ROC:          {mean_auc:.3f}  (0.5 = random, 1.0 = perfect)")
        print(f"  Mean Avg Precision:    {mean_ap:.3f}")
        print(f"  Baseline precision:    {total_true/total_sinks:.3f} (no ranking)")

        for k in [1, 3, 5]:
            if all_precision_at_k[k]:
                mean_pk = np.mean(all_precision_at_k[k])
                print(f"  Mean Precision@{k}:      {mean_pk:.3f}")

        # Interpretation
        print(f"\n  --- Interpretation ---")
        if mean_auc > 0.7:
            print(f"  ✓ KB similarity is a USEFUL signal (AUC={mean_auc:.3f} >> 0.5)")
            print(f"    → Graph neighbourhood around true vulnerabilities IS distinguishable")
            print(f"    → KB-guided ranking can improve precision over raw pattern matching")
        elif mean_auc > 0.55:
            print(f"  ~ KB similarity provides WEAK signal (AUC={mean_auc:.3f} > 0.5)")
            print(f"    → Some discriminative power but not sufficient alone")
        else:
            print(f"  ✗ KB similarity is NOT useful (AUC={mean_auc:.3f} ≈ 0.5)")
            print(f"    → Local graph structure around true/false sinks looks the same")

        # Per-CWE breakdown
        print(f"\n  --- Per-CWE AUC-ROC ---")
        for cwe, aucs in sorted(per_cwe_results.items(), key=lambda x: -np.mean(x[1])):
            mean_cwe_auc = np.mean(aucs)
            marker = "✓" if mean_cwe_auc > 0.6 else "~" if mean_cwe_auc > 0.5 else "✗"
            print(f"    {marker} {cwe}: {mean_cwe_auc:.3f} (n={len(aucs)})")

    # Save results
    output = {
        "config": {
            "random_seed": RANDOM_SEED,
            "sink_hop_radius": SINK_HOP_RADIUS,
            "embedding_dim": emb_dim,
            "embedder": EMBEDDER,
            "subset_file": SUBSET_FILE,
            "results_file": RESULTS_FILE,
            "n_entries_evaluated": len(all_auc_scores),
            "total_sinks": total_sinks,
            "total_true": total_true,
            "total_false": total_false,
            "baseline_precision": total_true / total_sinks,
        },
        "summary": {
            "mean_auc_roc": float(np.mean(all_auc_scores)) if all_auc_scores else None,
            "mean_avg_precision": float(np.mean(all_ap_scores)) if all_ap_scores else None,
            "mean_precision_at_1": float(np.mean(all_precision_at_k[1])) if all_precision_at_k[1] else None,
            "mean_precision_at_3": float(np.mean(all_precision_at_k[3])) if all_precision_at_k[3] else None,
            "mean_precision_at_5": float(np.mean(all_precision_at_k[5])) if all_precision_at_k[5] else None,
        },
        "per_cwe": {cwe: float(np.mean(aucs)) for cwe, aucs in per_cwe_results.items()},
        "details": detailed_results,
    }
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    run_experiment()
