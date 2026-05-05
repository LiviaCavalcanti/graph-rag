#!/usr/bin/env python3
"""
Post-patch verification via diff-based retrieval.

For each generated patch in a results.jsonl, this script:
  1. Runs Joern on the generated patch to build G_generated (full CPG)
  2. Computes G_diff = compute_graph_diff(G_before, G_generated) — the
     structural diff between the original vulnerable code and the patch
  3. Embeds G_diff and queries the G_vuln FAISS index for top-k neighbours
  4. If the same CVE's G_vuln is retrieved at top-1 → the patch modifies the
     same vulnerability region in a structurally similar way (patch likely correct)
     If not → the patch changes something different (patch likely wrong)

Rationale: G_vuln = diff(G_before, G_after) captures what the ground-truth fix
changed.  G_diff = diff(G_before, G_generated) captures what the generated patch
changed.  If the patch is correct, G_diff should be structurally similar to G_vuln
and retrieve the same CVE's vulnerability fingerprint from the index.

Output:
  - patch_verification.jsonl   (per-entry results)
  - patch_verification_summary.json  (aggregate patch accuracy)

Usage:
    python -m src.evaluate.patch_verification <results.jsonl> [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.data.autopatch import load_pairs
from src.io.read_write import load_config
from src.data.split import build_split
from src.data.pipeline import (compute_graph_diff, load_cpg_dir,
                               run_joern_export, write_c_file)
from src.embeddings import build_embedders
from src.rag.faiss_index import FAISSIndex
from src.rag.retriever import Retriever
from src.rag.utils import populate_index

# ── helpers ──────────────────────────────────────────────────────────


def _extract_diff_lines(G_diff) -> list[dict]:
    """Extract code lines from G_diff nodes, grouped by diff label.

    Returns a list of dicts with keys: line, code, labelV, diff, diff_weight.
    Sorted by LINE_NUMBER.  Only includes nodes that carry a CODE attribute.
    """
    lines = []
    for n, attrs in G_diff.nodes(data=True):
        code = str(attrs.get("CODE", "")).strip()
        if not code:
            continue
        lines.append({
            "line": attrs.get("LINE_NUMBER"),
            "code": code,
            "labelV": attrs.get("labelV", ""),
            "diff": attrs.get("diff", "context"),
            "diff_weight": attrs.get("diff_weight", 0.2),
        })
    lines.sort(key=lambda x: (x["line"] or 0, x["code"]))
    return lines


def _summarize_diff_lines(diff_lines: list[dict]) -> dict:
    """Compact summary: unique code lines per diff category."""
    by_cat: dict[str, list[str]] = {}
    for entry in diff_lines:
        cat = entry["diff"]
        code = entry["code"]
        by_cat.setdefault(cat, [])
        if code not in by_cat[cat]:
            by_cat[cat].append(code)
    return by_cat


def _load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_index_and_retriever(
    index_pairs,
    cfg,
    top_k,
    embedder_name=None,
    run_dir: Path | None = None,
):
    """Build embedder, FAISS index, and retriever.

    Tries to load a pre-built index from *run_dir* first (the directory that
    contains the results.jsonl being evaluated).  Falls back to the global
    index configured in ``rag.index_path``, and finally rebuilds from scratch.
    When a fresh index is built it is saved into *run_dir* for reuse.
    """
    rag_cfg = cfg["rag"]
    dim = cfg["embeddings"]["dim"]

    embedders = build_embedders(cfg)
    if embedder_name:
        matches = [e for e in embedders if e.name == embedder_name]
        if not matches:
            raise ValueError(
                f"Embedder '{embedder_name}' not found. Available: {[e.name for e in embedders]}"
            )
        embedder = matches[0]
    else:
        embedder = embedders[0]
    print(f"Using embedder: {embedder.name}  dim={dim}")

    # --- try loading an existing index (run_dir first, then global) ---
    candidates: list[tuple[Path, Path]] = []
    if run_dir is not None:
        candidates.append((run_dir / "faiss.index", run_dir / "faiss_metadata.json"))
    candidates.append((Path(rag_cfg["index_path"]), Path(rag_cfg["metadata_path"])))

    index = None
    for idx_path, meta_path in candidates:
        if idx_path.exists() and meta_path.exists():
            trial = FAISSIndex(
                dim=dim, index_path=str(idx_path), metadata_path=str(meta_path)
            )
            trial.load()
            if trial.index.d == dim and trial.index.ntotal == len(index_pairs):
                print(
                    f"Loaded index from {idx_path}: {trial.index.ntotal} vectors, dim={trial.index.d}"
                )
                index = trial
                break
            else:
                print(
                    f"Index at {idx_path} incompatible (d={trial.index.d}, n={trial.index.ntotal}), skipping..."
                )

    # --- fit embedder PCA (always needed for query-time embed_one) ---
    graphs = [p.G_vuln for p in index_pairs]
    print(f"Fitting embedder on {len(graphs)} index graphs...")
    embeddings = embedder.embed_many(graphs)

    # --- build fresh index if none loaded ---
    if index is None:
        save_path = (
            run_dir if run_dir is not None else Path(rag_cfg["index_path"]).parent
        )
        idx_path = save_path / "faiss.index"
        meta_path = save_path / "faiss_metadata.json"
        index = FAISSIndex(
            dim=dim, index_path=str(idx_path), metadata_path=str(meta_path)
        )
        retriever = populate_index(index, index_pairs, embeddings, embedder.name, top_k=top_k)
        print(f"Built and saved index: {index.index.ntotal} vectors → {idx_path}")
    else:
        retriever = Retriever(index, top_k=top_k)

    return embedder, retriever


def _get_supplementary_code(cve_id: str, variant: str, data_root: Path) -> str:
    """Load supplementary code for Joern scaffolding."""
    # Try variant-specific JSON first
    variant_json = data_root / cve_id / "out_v2" / "code" / f"{variant}.json"
    if variant_json.exists():
        try:
            d = json.loads(variant_json.read_text())
            supp = d.get("supplementary_code", "")
            if supp:
                return supp
        except (json.JSONDecodeError, KeyError):
            pass

    # Fall back to db_entry.json
    db_path = data_root / cve_id / "out_v2" / "db_entry.json"
    if db_path.exists():
        try:
            d = json.loads(db_path.read_text())
            return d.get("supplementary_code", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return ""


def _run_joern_on_patch(
    generated_patch: str,
    supplementary_code: str,
    joern_bin_dir: str,
    work_dir: Path,
):
    """Write generated patch to file, run Joern, return loaded graph or None."""
    source_path = work_dir / "patched.c"
    cpg_dir = work_dir / "cpg"
    graph_dir = work_dir / "graph"

    write_c_file(generated_patch, source_path, supplementary_code)

    ok = run_joern_export(joern_bin_dir, str(source_path), str(cpg_dir), str(graph_dir))
    if not ok:
        return None

    return load_cpg_dir(str(graph_dir))


def verify_one(
    rec: dict,
    embedder,
    retriever,
    cfg: dict,
    data_root: Path,
    top_k: int,
    query_pair_lookup: dict,
) -> dict:
    """Run post-patch verification for a single record.

    Builds a full CPG from the generated patch, computes the structural diff
    against G_before, embeds the diff, and queries the G_vuln FAISS index.
    If the same CVE's G_vuln is the nearest neighbour, the patch modifies the
    vulnerability region in a structurally similar way to the ground-truth fix.
    """
    q_cve = rec.get("query_cve", "?")
    q_var = rec.get("query_variant", "?")
    q_cwe = rec.get("query_cwe", "?")
    generated = (rec.get("generated_patch") or "").strip()

    base_result = {
        "query_cve": q_cve,
        "query_cwe": q_cwe,
        "query_variant": q_var,
    }

    if not generated:
        return {**base_result, "status": "no_patch", "retrieved": []}

    # Look up the original query pair to get G_before
    pair_key = (q_cve, q_var)
    qp = query_pair_lookup.get(pair_key)
    if qp is None:
        return {**base_result, "status": "pair_not_found", "retrieved": []}

    G_before = qp.G_before

    # 1. Run Joern on generated patch → G_generated (full CPG)
    supp = _get_supplementary_code(q_cve, q_var, data_root)
    joern_bin = cfg["joern"]["bin_dir"]

    with tempfile.TemporaryDirectory(prefix=f"pv_{q_cve}_{q_var}_") as tmpdir:
        G_generated = _run_joern_on_patch(generated, supp, joern_bin, Path(tmpdir))

    if G_generated is None or G_generated.number_of_nodes() == 0:
        return {**base_result, "status": "joern_failed", "retrieved": []}

    n_nodes_generated = G_generated.number_of_nodes()

    # 2. Compute structural diff: what did the patch change vs vulnerable code?
    G_diff = compute_graph_diff(G_before, G_generated)
    n_diff_nodes = G_diff.number_of_nodes()

    # Extract code lines from the generated-patch diff
    patch_diff_lines = _extract_diff_lines(G_diff) if n_diff_nodes > 0 else []
    patch_diff_summary = _summarize_diff_lines(patch_diff_lines) if patch_diff_lines else {}

    # Extract code lines from the ground-truth vulnerability diff (G_vuln)
    gt_vuln_lines = _extract_diff_lines(qp.G_vuln) if qp.G_vuln and qp.G_vuln.number_of_nodes() > 0 else []
    gt_vuln_summary = _summarize_diff_lines(gt_vuln_lines) if gt_vuln_lines else {}

    if n_diff_nodes == 0:
        # No structural difference detected — patch is identical to vulnerable code
        return {
            **base_result,
            "status": "no_diff",
            "g_generated_nodes": n_nodes_generated,
            "g_diff_nodes": 0,
            "patch_diff_lines": [],
            "patch_diff_summary": {},
            "gt_vuln_lines": gt_vuln_lines,
            "gt_vuln_summary": gt_vuln_summary,
            "retrieved": [],
        }

    # 3. Embed the diff and query the G_vuln index
    try:
        emb = embedder.embed_one(G_diff)
    except Exception as e:
        return {
            **base_result,
            "status": "embedding_error",
            "error": str(e),
            "g_generated_nodes": n_nodes_generated,
            "g_diff_nodes": n_diff_nodes,
            "patch_diff_lines": patch_diff_lines,
            "patch_diff_summary": patch_diff_summary,
            "gt_vuln_lines": gt_vuln_lines,
            "gt_vuln_summary": gt_vuln_summary,
            "retrieved": [],
        }

    results = retriever.query(emb, top_k=top_k)
    top1_cve = results[0]["cve_id"] if results else None
    top1_score = results[0].get("score", 0.0) if results else 0.0
    same_cve_in_topk = any(r["cve_id"] == q_cve for r in results)
    same_cve_at_top1 = top1_cve == q_cve

    return {
        **base_result,
        "status": "verified",
        "g_generated_nodes": n_nodes_generated,
        "g_diff_nodes": n_diff_nodes,
        "same_cve_at_top1": same_cve_at_top1,
        "same_cve_in_topk": same_cve_in_topk,
        "top1_cve": top1_cve,
        "top1_score": round(top1_score, 6),
        "patch_diff_lines": patch_diff_lines,
        "patch_diff_summary": patch_diff_summary,
        "gt_vuln_lines": gt_vuln_lines,
        "gt_vuln_summary": gt_vuln_summary,
        "retrieved": [
            {
                "rank": j + 1,
                "cve_id": r.get("cve_id"),
                "variant": r.get("variant"),
                "score": round(r.get("score", 0.0), 6),
            }
            for j, r in enumerate(results)
        ],
    }


# ── aggregation ──────────────────────────────────────────────────────


def aggregate_verification(entries: list[dict]) -> dict:
    """Compute patch accuracy.

    Definitions (diff-based retrieval verification):
      - same CVE at top-1 = patch modifies the same region as the GT fix → SUCCESS
      - different CVE at top-1 = patch changes something else → FAIL
      - "Not vulnerable" = retriever does NOT match same CVE at top-1

    Patch accuracy = fraction where same CVE IS retrieved at top-1.
    """
    verified = [e for e in entries if e.get("status") == "verified"]
    n = len(verified)
    if n == 0:
        return {"n_verified": 0}

    patch_success = sum(1 for e in verified if e["same_cve_at_top1"])
    patch_fail = sum(1 for e in verified if not e["same_cve_at_top1"])

    # Also check top-k
    topk_success = sum(1 for e in verified if e["same_cve_in_topk"])
    topk_fail = sum(1 for e in verified if not e["same_cve_in_topk"])

    # Score distribution
    scores = [e["top1_score"] for e in verified]

    # By-CWE breakdown
    by_cwe = {}
    for e in verified:
        cwe = e.get("query_cwe", "?")
        by_cwe.setdefault(cwe, {"n": 0, "success": 0})
        by_cwe[cwe]["n"] += 1
        if e["same_cve_at_top1"]:
            by_cwe[cwe]["success"] += 1
    for cwe in by_cwe:
        by_cwe[cwe]["accuracy"] = round(by_cwe[cwe]["success"] / by_cwe[cwe]["n"], 4)

    # Graph size statistics
    g_nodes = [e["g_generated_nodes"] for e in verified]
    g_diff = [e["g_diff_nodes"] for e in verified]

    return {
        "n_total": len(entries),
        "n_verified": n,
        "n_skipped": len(entries) - n,
        "skip_reasons": _count_skip_reasons(entries),
        "patch_accuracy_top1": round(patch_success / n, 4),
        "patch_accuracy_topk": round(topk_success / n, 4),
        "patch_success_top1": patch_success,
        "patch_fail_top1": patch_fail,
        "patch_success_topk": topk_success,
        "patch_fail_topk": topk_fail,
        "top1_score_stats": {
            "mean": round(float(np.mean(scores)), 4),
            "median": round(float(np.median(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
        },
        "g_generated_node_stats": {
            "mean": round(float(np.mean(g_nodes)), 1),
            "median": round(float(np.median(g_nodes)), 1),
        },
        "g_diff_node_stats": {
            "mean": round(float(np.mean(g_diff)), 1),
            "median": round(float(np.median(g_diff)), 1),
        },
        "by_cwe": by_cwe,
    }


def _count_skip_reasons(entries):
    reasons = {}
    for e in entries:
        s = e.get("status", "?")
        if s != "verified":
            reasons[s] = reasons.get(s, 0) + 1
    return reasons


# ── main orchestrator ────────────────────────────────────────────────


def run_patch_verification(
    results_path: Path,
    cfg: dict,
    top_k: int = 5,
    out_dir: Path | None = None,
    embedder_name: str | None = None,
) -> Path:
    """Run the full post-patch verification pipeline."""
    out = out_dir or results_path.parent
    data_root = Path(cfg["data"]["autopatch"]["root"])

    # Load dataset and build split (same as batch inference)
    pairs = load_pairs(cfg)
    index_pairs, query_pairs, _split_info = build_split(pairs, cfg)
    print(f"Dataset: {len(index_pairs)} index, {len(query_pairs)} query")

    # Build index + retriever (reuse saved index from results dir if available)
    run_dir = results_path.parent
    embedder, retriever = _build_index_and_retriever(
        index_pairs,
        cfg,
        top_k,
        embedder_name,
        run_dir=run_dir,
    )

    # Build lookup from (cve_id, variant) → query pair for G_before access
    query_pair_lookup = {}
    for qp in query_pairs:
        key = (qp.cve_id, qp.meta.get("variant", ""))
        query_pair_lookup[key] = qp
    print(f"Query pair lookup: {len(query_pair_lookup)} entries")

    # Load batch results
    records = _load_records(results_path)
    print(f"Loaded {len(records)} batch results from {results_path}")

    # Run verification for each record
    entries = []
    for i, rec in enumerate(records):
        label = f"{rec.get('query_cve', '?')}/{rec.get('query_variant', '?')}"
        print(f"  [{i+1}/{len(records)}] {label} ...", end=" ", flush=True)

        try:
            entry = verify_one(
                rec, embedder, retriever, cfg, data_root, top_k, query_pair_lookup
            )
        except Exception as exc:
            entry = {
                "query_cve": rec.get("query_cve", "?"),
                "query_cwe": rec.get("query_cwe", "?"),
                "query_variant": rec.get("query_variant", "?"),
                "status": "error",
                "error": str(exc),
                "retrieved": [],
            }
            print(f"ERROR: {exc}")
        else:
            status = entry["status"]
            if status == "verified":
                matched = "SAME CVE" if entry["same_cve_at_top1"] else "DIFFERENT"
                print(
                    f"{matched}  (score={entry['top1_score']:.4f}, nodes={entry['g_generated_nodes']}, diff={entry['g_diff_nodes']})"
                )
            else:
                print(f"SKIP: {status}")

        entries.append(entry)

    # Write results
    out_path = out / "patch_verification.jsonl"
    with open(out_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, default=str) + "\n")
    print(f"\nWrote {len(entries)} entries to {out_path}")

    # Aggregate
    summary = aggregate_verification(entries)
    summary_path = out / "patch_verification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    n = summary.get("n_verified", 0)
    if n > 0:
        acc1 = summary["patch_accuracy_top1"]
        acck = summary["patch_accuracy_topk"]
        print(f"\n{'═'*60}")
        print(f"  POST-PATCH VERIFICATION RESULTS")
        print(f"{'═'*60}")
        print(f"  Verified:           {n}/{summary['n_total']}")
        print(
            f"  Patch accuracy @1:  {acc1:.1%}  ({summary['patch_success_top1']}/{n} diffs match GT)"
        )
        print(
            f"  Patch accuracy @k:  {acck:.1%}  ({summary['patch_success_topk']}/{n} diffs match GT)"
        )
        print(
            f"  Score mean/median:  {summary['top1_score_stats']['mean']:.4f} / {summary['top1_score_stats']['median']:.4f}"
        )
        print(f"\n  By CWE:")
        for cwe, stats in sorted(
            summary["by_cwe"].items(), key=lambda x: -x[1]["accuracy"]
        ):
            print(
                f"    {cwe:30s}  {stats['accuracy']:.0%}  ({stats['success']}/{stats['n']})"
            )
        print(f"{'═'*60}")

    return out_path


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Post-patch verification via retrieval"
    )
    parser.add_argument("results", type=Path, help="Path to results.jsonl")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--embedder",
        type=str,
        default=None,
        help="Embedder name (default: first active)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_patch_verification(
        args.results,
        cfg,
        top_k=args.top_k,
        out_dir=args.out_dir,
        embedder_name=args.embedder,
    )


if __name__ == "__main__":
    main()
