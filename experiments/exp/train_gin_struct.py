"""
Training script for GIN-Struct with triplet loss (CVE-level positives).

Uses the same 11-d node-type features as the frozen GIN but learns
a metric space where same-CVE variants cluster together.

Usage:
    python -m experiments.exp.train_gin_struct [--config config.yaml] [--epochs 100]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.common import build_hnsw, build_split, evaluate_cwe_recall, evaluate_retrieval, load_pairs
from src.embeddings.gin_struct import GINStructEmbedder
from src.embeddings.gin_struct_model import GINStructModel
from src.training.struct_trainer import StructTripletDataset, StructTripletTrainer


def _get_cve_labels(pairs) -> list[str]:
    """Extract CVE-ID labels (for triplet mining positives)."""
    return [p.cve_id for p in pairs]


def _get_dir_labels(pairs) -> list[str]:
    """Extract dir_name labels — groups variants of the same function."""
    return [p.meta.get("dir_name", p.cve_id) for p in pairs]


def run_training(cfg: dict, output_dir: Path | None = None) -> dict:
    """
    Full training pipeline:
      1. Load data & split
      2. Build triplet datasets (CVE-level grouping)
      3. Train structural GIN with triplet loss
      4. Evaluate on query set (CVE + CWE metrics)
      5. Save checkpoint & results
    """
    if output_dir is None:
        output_dir = Path("experiments/output/gin_struct_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GIN-Struct Triplet Training (CVE-level positives)")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    pairs = load_pairs(cfg)
    index_pairs, query_pairs, split_info = build_split(pairs, cfg)
    print(f"  Index: {len(index_pairs)} pairs, Query: {len(query_pairs)} pairs")

    # Use CVE-ID as label for triplet mining
    gs_cfg = cfg.get("gin_struct", {}).get("training", {})
    label_mode = gs_cfg.get("label_mode", "cve")

    if label_mode == "dir_name":
        index_labels = _get_dir_labels(index_pairs)
        label_desc = "dir_name (function-level)"
    else:
        index_labels = _get_cve_labels(index_pairs)
        label_desc = "CVE-ID"

    unique_labels = sorted(set(index_labels))
    print(f"  Label mode: {label_desc}")
    print(f"  Unique labels: {len(unique_labels)}")

    # Show label distribution
    from collections import Counter
    label_counts = Counter(index_labels)
    multi = {k: v for k, v in label_counts.items() if v >= 2}
    print(f"  Labels with ≥2 samples (trainable): {len(multi)} "
          f"(covering {sum(multi.values())} graphs)")

    # ── 2. Build datasets ────────────────────────────────────────────
    print("\n[2/5] Building training dataset...")
    index_graphs = [p.G_vuln for p in index_pairs]
    train_dataset = StructTripletDataset(index_graphs, index_labels)

    # Validation dataset from query pairs
    val_dataset = None
    if len(query_pairs) >= 4:
        query_graphs_all = [p.G_vuln for p in query_pairs]
        if label_mode == "dir_name":
            query_labels = _get_dir_labels(query_pairs)
        else:
            query_labels = _get_cve_labels(query_pairs)
        label_counts_q = Counter(query_labels)
        if any(c >= 2 for c in label_counts_q.values()):
            val_dataset = StructTripletDataset(query_graphs_all, query_labels)

    # ── 3. Train ─────────────────────────────────────────────────────
    print("\n[3/5] Training GIN-Struct model...")
    emb_cfg = cfg.get("embeddings", {})
    gs_model_cfg = cfg.get("gin_struct", {})

    # Warm start from frozen GIN weights (preserves good random geometry)
    warm_start = gs_model_cfg.get("training", {}).get("warm_start", True)

    model = GINStructModel(
        hidden_dim=gs_model_cfg.get("hidden_dim", 128),
        out_dim=emb_cfg.get("dim", 128),
        num_layers=gs_model_cfg.get("num_layers", 3),
        dropout=gs_model_cfg.get("dropout", 0.2),
        frozen_compat=warm_start,
    )

    if warm_start:
        model.warm_start_from_frozen(emb_cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    device = "cuda" if __import__("torch").cuda.is_available() else gs_model_cfg.get("device", "cpu")
    trainer = StructTripletTrainer(model, cfg, device=device)

    t0 = time.perf_counter()
    history = trainer.train(train_dataset, val_dataset)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")

    # ── 4. Save checkpoint ───────────────────────────────────────────
    print("\n[4/5] Saving checkpoint...")
    checkpoint_path = output_dir / "gin_struct_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path, extra={
        "train_time_s": train_time,
        "n_index": len(index_pairs),
        "n_query": len(query_pairs),
        "n_labels": len(unique_labels),
        "label_mode": label_mode,
    })

    # ── 5. Evaluate ──────────────────────────────────────────────────
    print("\n[5/5] Evaluating trained model...")
    eval_cfg = dict(cfg.get("embeddings", {}))
    eval_cfg["gin_struct"] = dict(gs_model_cfg)
    eval_cfg["gin_struct"]["checkpoint_path"] = str(checkpoint_path)
    embedder = GINStructEmbedder(eval_cfg)

    # Embed index
    index_embs = embedder.embed_many(index_graphs)
    print(f"  Index embeddings: {index_embs.shape}")

    # Embed queries
    query_graphs = [p.G_vuln for p in query_pairs]
    query_embs = embedder.embed_many(query_graphs)
    print(f"  Query embeddings: {query_embs.shape}")

    # Build HNSW index
    ks = cfg.get("experiment", {}).get("ks", [1, 5, 10])
    index, retriever = build_hnsw(
        index_pairs, index_embs, "gin_struct", embedder.dim, output_dir,
        tag="trained",
    )

    # Index metadata for CWE recall
    index_metadata = [
        {"cve_id": p.cve_id, "cwe_id": p.cwe_id, "func_name": p.func_name,
         "variant": p.meta.get("variant", ""), **p.meta}
        for p in index_pairs
    ]

    # CWE recall
    cwe_recall = evaluate_cwe_recall(query_pairs, query_embs, retriever, index_metadata, top_k=10)
    cwe_recall.pop("raw_queries", None)
    cwe_recall.pop("per_cwe", None)

    # CVE retrieval
    retrieval = evaluate_retrieval(query_pairs, query_embs, retriever, index_pairs, ks=ks)
    retrieval.pop("raw_queries", None)

    # Print results
    print(f"\n{'='*60}")
    print("CWE Recall:")
    print(f"  macro_avg:    {cwe_recall.get('macro_avg', 0):.4f}")
    print(f"  ranx_recall:  {cwe_recall.get('ranx_recall', 0):.4f}")
    print(f"\nCVE Retrieval:")
    print(f"  hit@1:  {retrieval.get('hit@1', 0):.4f}")
    print(f"  hit@5:  {retrieval.get('hit@5', 0):.4f}")
    print(f"  hit@10: {retrieval.get('hit@10', 0):.4f}")
    print(f"  MRR:    {retrieval.get('mrr', 0):.4f}")
    print(f"  nDCG@10:{retrieval.get('ndcg@10', 0):.4f}")
    print(f"  MAP@10: {retrieval.get('map@10', 0):.4f}")
    print(f"{'='*60}")

    # Save results
    results = {
        "training": {
            "epochs_run": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "train_time_s": train_time,
            "label_mode": label_mode,
        },
        "cwe_recall": cwe_recall,
        "retrieval": retrieval,
        "config": {
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "out_dim": model.out_dim,
            "dropout": model.dropout,
            "margin": trainer.margin,
            "lr": trainer.lr,
            "n_index": len(index_pairs),
            "n_query": len(query_pairs),
            "n_labels": len(unique_labels),
            "label_mode": label_mode,
        },
        "checkpoint_path": str(checkpoint_path),
    }

    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results → {results_path}")
    print(f"  Checkpoint → {checkpoint_path}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from src.io import load_config

    parser = argparse.ArgumentParser(description="Train GIN-Struct with triplet loss (CVE-level)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--label-mode", choices=["cve", "dir_name"], default=None,
                        help="Label grouping for positives: cve (default) or dir_name")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Disable warm start from frozen GIN weights")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg.setdefault("gin_struct", {}).setdefault("training", {})["epochs"] = args.epochs
    if args.label_mode is not None:
        cfg.setdefault("gin_struct", {}).setdefault("training", {})["label_mode"] = args.label_mode
    if args.no_warm_start:
        cfg.setdefault("gin_struct", {}).setdefault("training", {})["warm_start"] = False

    output_dir = Path(args.output) if args.output else None
    run_training(cfg, output_dir)
