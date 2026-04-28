#!/usr/bin/env python3
"""
Evaluate RAG retrieval quality for batch inference results.

For each query in a results.jsonl produced by batch inference, this script:
  1. Loads all FunctionPair objects from the dataset.
  2. Builds an index/query split (same as batch mode).
  3. Embeds index pairs using CombinedEmbedder and builds a FAISS index.
  4. Embeds each query pair's G_vuln graph.
  5. Queries the FAISS index for top-k neighbours.
  6. Writes retrieval results (top-k with scores) to a JSONL file.

Usage:
    python -m src.evaluate.retrieval_eval <results.jsonl> [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.common import load_config, load_pairs, build_split
from src.embeddings import build_embedders
from src.rag.index import FAISSIndex
from src.rag.retriever import Retriever
from src.evaluate.preprocessing import extract_function_body
from src.metrics.metrics import hits_at_k, mean_reciprocal_rank


def _build_index_and_retriever(
    index_pairs: list,
    cfg: dict,
    top_k: int,
):
    """Build embedder, FAISS index, and retriever.

    Tries to load the pre-existing index from disk (like main.run_query).
    Falls back to building from scratch if the index doesn't exist or the
    dimension doesn't match the active embedder.

    Returns (embedder, retriever).
    """
    rag_cfg = cfg["rag"]
    dim = cfg["embeddings"]["dim"]

    # build embedder (same as main.run_pipeline)
    embedders = build_embedders(cfg)
    embedder = embedders[0]  # first active embedder from config
    print(f"Using embedder: {embedder.name}  dim={dim}")

    # try loading pre-existing index from disk
    index_path = Path(rag_cfg["index_path"])
    meta_path = Path(rag_cfg["metadata_path"])
    loaded = False

    if index_path.exists() and meta_path.exists():
        index = FAISSIndex(
            dim=dim,
            index_path=rag_cfg["index_path"],
            metadata_path=rag_cfg["metadata_path"],
        )
        index.load()
        if index.index.d == dim:
            print(f"Loaded pre-existing index: {index.index.ntotal} vectors, dim={index.index.d}")
            loaded = True
        else:
            print(f"Index dim mismatch (index={index.index.d}, embedder={dim}), rebuilding...")

    # fit embedder (PCA for combined) — always needed for query-time embed_one
    graphs = [p.G_vuln for p in index_pairs]
    print(f"Fitting embedder on {len(graphs)} index graphs...")
    embeddings = embedder.embed_many(graphs)

    # rebuild index if we couldn't load one
    if not loaded:
        index = FAISSIndex(
            dim=dim,
            index_path=rag_cfg["index_path"],
            metadata_path=rag_cfg["metadata_path"],
        )
        for pair, vec in zip(index_pairs, embeddings):
            index.add(pair, vec, embedder.name)
        print(f"Built fresh index: {index.index.ntotal} vectors, dim={dim}")

    retriever = Retriever(index, top_k=top_k)
    return embedder, retriever


def _load_records(results_path: Path) -> list[dict]:
    """Read results.jsonl into a list of dicts."""
    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_pair_lookup(query_pairs: list) -> dict[tuple[str, str], object]:
    """Build a (cve_id, variant) → FunctionPair lookup + cve-only fallback."""
    by_key: dict[tuple[str, str], object] = {}
    by_cve: dict[str, object] = {}
    for p in query_pairs:
        by_key[(p.cve_id, p.meta.get("variant", ""))] = p
        by_cve.setdefault(p.cve_id, p)
    return by_key, by_cve


def _match_pair(record: dict, by_key, by_cve):
    """Find the FunctionPair for a record using O(1) lookup."""
    q_cve = record.get("query_cve")
    q_var = record.get("query_variant", "")
    return by_key.get((q_cve, q_var)) or by_cve.get(q_cve)


def _evaluate_one(
    rec: dict,
    pair,
    embedder,
    retriever,
    top_k: int,
    base_dir: Path,
) -> dict:
    """Run retrieval for a single record. Returns the evaluation entry."""
    q_cve = rec.get("query_cve", "?")
    q_var = rec.get("query_variant", "?")

    if pair is None:
        return {
            "query_cve": q_cve,
            "query_variant": q_var,
            "status": "no_matching_pair",
            "retrieved": [],
        }

    # embed query graph
    try:
        q_emb = embedder.embed_one(pair.G_vuln)
    except Exception as e:
        return {
            "query_cve": q_cve,
            "query_variant": q_var,
            "status": "embedding_error",
            "error": str(e),
            "retrieved": [],
        }

    # query FAISS
    results = retriever.query(q_emb, top_k=top_k)
    hit_cve = any(r["cve_id"] == q_cve for r in results)

    # ground-truth preview
    gt_code = rec.get("ground_truth_patch", "")
    gt_preview = ""
    if gt_code:
        gt_preview = extract_function_body(gt_code)[:200]

    return {
        "query_cve": q_cve,
        "query_cwe": rec.get("query_cwe"),
        "query_variant": q_var,
        "status": "evaluated",
        "hit_cve_in_top_k": hit_cve,
        "ground_truth_patch": gt_code[:500] if gt_code else "",
        "ground_truth_preview": gt_preview,
        "retrieved": [
            {
                "rank": j + 1,
                "cve_id": r.get("cve_id"),
                "cwe_id": r.get("cwe_id"),
                "variant": r.get("variant"),
                "func_name": r.get("func_name"),
                "score": round(r.get("score", 0.0), 6),
                "n_nodes": r.get("n_nodes"),
            }
            for j, r in enumerate(results)
        ],
    }


def _run_all_queries(
    records: list[dict],
    query_pairs: list,
    embedder,
    retriever,
    top_k: int,
    base_dir: Path,
) -> list[dict]:
    """Embed + retrieve for every record. Pure compute, no I/O."""
    by_key, by_cve = _build_pair_lookup(query_pairs)
    evaluated = []
    total = len(records)

    for i, rec in enumerate(records):
        pair = _match_pair(rec, by_key, by_cve)
        entry = _evaluate_one(rec, pair, embedder, retriever, top_k, base_dir)
        evaluated.append(entry)

        label = f"{entry['query_cve']}/{entry.get('query_variant', '?')}"
        status = entry["status"]
        if status == "evaluated":
            top1 = entry["retrieved"][0] if entry["retrieved"] else {}
            print(
                f"  [{i+1}/{total}] {label}  "
                f"hit_cve={entry['hit_cve_in_top_k']}  "
                f"top1={top1.get('cve_id','?')}/{top1.get('variant','?')} "
                f"score={top1.get('score', 0):.4f}"
            )
        else:
            print(f"  [{i+1}/{total}] {label}  {status}")

    return evaluated


def _write_results(evaluated: list[dict], out_path: Path) -> None:
    """Write per-record evaluations to JSONL (single fast flush)."""
    with open(out_path, "w") as f:
        for entry in evaluated:
            f.write(json.dumps(entry) + "\n")


def _aggregate(evaluated: list[dict], top_k: int, split_info: dict) -> dict:
    """Compute aggregate retrieval metrics from evaluated entries."""
    total = len(evaluated)
    matched = [e for e in evaluated if e["status"] == "evaluated"]
    n = len(matched)

    summary = {
        "total_records": total,
        "matched": n,
        "skipped": total - n,
        "top_k": top_k,
    }

    if n == 0:
        summary["split_info"] = split_info
        return summary

    # reuse src.metrics.metrics for per-query hit/mrr
    hit_at_1_total = 0
    hit_at_k_total = 0
    mrrs = []
    cwe_hits = 0

    for e in matched:
        retrieved = e["retrieved"]
        q_cve = e["query_cve"]

        hit_at_1_total += hits_at_k(retrieved, q_cve, 1)
        hit_at_k_total += hits_at_k(retrieved, q_cve, top_k)
        mrrs.append(mean_reciprocal_rank(retrieved, q_cve))

        if any(r["cwe_id"] == e.get("query_cwe") for r in retrieved):
            cwe_hits += 1

    summary["hit_at_1"] = hit_at_1_total
    summary["hit_rate_at_1"] = round(hit_at_1_total / n, 4)
    summary["hit_cve_at_k"] = hit_at_k_total
    summary["hit_rate_at_k"] = round(hit_at_k_total / n, 4)
    summary["mrr"] = round(float(np.mean(mrrs)), 4)
    summary["cwe_hit_at_k"] = cwe_hits
    summary["cwe_hit_rate_at_k"] = round(cwe_hits / n, 4)

    summary["split_info"] = split_info
    return summary


def _print_summary(summary: dict, top_k: int) -> None:
    n = summary["matched"]
    print(f"\n{'═'*60}")
    print(f"  RETRIEVAL EVALUATION SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total records:   {summary['total_records']}")
    print(f"  Matched/queried: {n}")
    print(f"  Top-k:           {top_k}")
    print(f"  Hit@1:           {summary.get('hit_at_1', 0)}  ({summary.get('hit_rate_at_1', 0):.1%})")
    print(f"  Hit@{top_k}:          {summary.get('hit_cve_at_k', 0)}  ({summary.get('hit_rate_at_k', 0):.1%})")
    print(f"  MRR:             {summary.get('mrr', 0):.4f}")
    print(f"  CWE Hit@{top_k}:      {summary.get('cwe_hit_at_k', 0)}  ({summary.get('cwe_hit_rate_at_k', 0):.1%})")
    print(f"{'═'*60}")


def evaluate_retrieval(
    results_path: Path,
    cfg: dict,
    top_k: int = 5,
    out_path: Path | None = None,
):
    """Main evaluation entry point."""
    # 1. load inputs
    records = _load_records(results_path)
    print(f"Loaded {len(records)} records from {results_path}")

    pairs = load_pairs(cfg)
    index_pairs, query_pairs, split_info = build_split(pairs, cfg)
    print(f"Split: {len(index_pairs)} index pairs, {len(query_pairs)} query pairs")

    # 2. build index (heavy compute)
    embedder, retriever = _build_index_and_retriever(index_pairs, cfg, top_k)

    # 3. run all queries (compute only, no file handles open)
    evaluated = _run_all_queries(
        records, query_pairs, embedder, retriever, top_k, Path.cwd(),
    )

    # 4. write results (single fast I/O pass)
    out = out_path or results_path.parent / "retrieval_eval.jsonl"
    summary_path = out.with_name("retrieval_eval_summary.json")

    _write_results(evaluated, out)
    print(f"\nPer-record retrieval written to: {out}")

    # 5. aggregate and write summary
    summary = _aggregate(evaluated, top_k, split_info)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary written to:              {summary_path}")

    _print_summary(summary, top_k)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality for batch inference results."
    )
    parser.add_argument(
        "results_jsonl",
        help="Path to results.jsonl from batch inference",
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of neighbours to retrieve (default: 5)")
    parser.add_argument("--out", default=None, help="Output JSONL path (default: <input_dir>/retrieval_eval.jsonl)")
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    cfg = load_config(args.config)
    out_path = Path(args.out) if args.out else None
    evaluate_retrieval(results_path, cfg, top_k=args.top_k, out_path=out_path)


if __name__ == "__main__":
    main()
