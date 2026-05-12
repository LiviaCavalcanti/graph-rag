"""
Training script for GIN-CodeBERT with triplet loss.

Loads vulnerability pairs, extracts per-node CodeBERT features (cached),
trains a GIN model with triplet loss using CWE-ID as class labels,
and evaluates on a held-out query set.

Usage:
    python -m experiments.exp.train_gin_codebert [--config config.yaml] [--epochs 50]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.common import build_hnsw, build_split, evaluate_cwe_recall, evaluate_retrieval, load_pairs
from src.embeddings.gin_codebert import GINCodeBERTEmbedder
from src.embeddings.gin_model import GINCodeBERTModel, nx_to_pyg_codebert
from src.embeddings.node_codebert import NodeCodeBERTEncoder
from src.training import TripletDataset, TripletTrainer


def _get_labels(pairs) -> list[str]:
    """Extract CWE-ID labels from pairs."""
    return [p.cwe_id or "UNKNOWN" for p in pairs]


def run_training(cfg: dict, output_dir: Path | None = None) -> dict:
    """
    Full training pipeline:
      1. Load data & split
      2. Encode node features (cached)
      3. Train GIN with triplet loss
      4. Evaluate on query set
      5. Save checkpoint & results
    """
    if output_dir is None:
        output_dir = Path("experiments/output/gin_codebert_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GIN-CodeBERT Triplet Training")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    pairs = load_pairs(cfg)
    index_pairs, query_pairs, split_info = build_split(pairs, cfg)
    print(f"  Index: {len(index_pairs)} pairs, Query: {len(query_pairs)} pairs")

    index_labels = _get_labels(index_pairs)
    unique_labels = sorted(set(index_labels))
    print(f"  CWE classes: {len(unique_labels)} — {unique_labels[:5]}...")

    # ── 2. Build datasets ────────────────────────────────────────────
    print("\n[2/5] Building training dataset (encoding node features)...")
    emb_cfg = cfg.get("embeddings", {})
    node_encoder = NodeCodeBERTEncoder(emb_cfg)

    index_graphs = [p.G_vuln for p in index_pairs]
    train_dataset = TripletDataset(index_graphs, index_labels, node_encoder)

    # Validation dataset from query pairs (if enough)
    val_dataset = None
    if len(query_pairs) >= 4:
        query_graphs = [p.G_vuln for p in query_pairs]
        query_labels = _get_labels(query_pairs)
        # Only use for validation if we have classes with ≥2 samples
        label_counts = {}
        for l in query_labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        if any(c >= 2 for c in label_counts.values()):
            val_dataset = TripletDataset(query_graphs, query_labels, node_encoder)

    # ── 3. Train ─────────────────────────────────────────────────────
    print("\n[3/5] Training GIN-CodeBERT model...")
    gc_cfg = cfg.get("gin_codebert", {})
    train_cfg = gc_cfg.get("training", {})

    model = GINCodeBERTModel(
        in_dim=768,
        hidden_dim=gc_cfg.get("hidden_dim", 128),
        out_dim=emb_cfg.get("dim", 128),
        num_layers=gc_cfg.get("num_layers", 3),
        dropout=gc_cfg.get("dropout", 0.3),
    )

    device = "cuda" if __import__("torch").cuda.is_available() else gc_cfg.get("device", "cpu")
    trainer = TripletTrainer(model, cfg, device=device)

    # Override epochs from CLI if provided
    t0 = time.perf_counter()
    history = trainer.train(train_dataset, val_dataset)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")

    # ── 4. Save checkpoint ───────────────────────────────────────────
    print("\n[4/5] Saving checkpoint...")
    checkpoint_path = output_dir / "gin_codebert_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path, extra={
        "train_time_s": train_time,
        "n_index": len(index_pairs),
        "n_query": len(query_pairs),
        "n_classes": len(unique_labels),
    })

    # ── 5. Evaluate ──────────────────────────────────────────────────
    print("\n[5/5] Evaluating trained model...")
    # Create embedder wrapper pointing to the new checkpoint
    eval_cfg = dict(cfg.get("embeddings", {}))
    eval_cfg["gin_codebert"] = dict(gc_cfg)
    eval_cfg["gin_codebert"]["checkpoint_path"] = str(checkpoint_path)
    embedder = GINCodeBERTEmbedder(eval_cfg)

    # Embed index
    index_embs = embedder.embed_many(index_graphs)
    print(f"  Index embeddings: {index_embs.shape}")

    # Embed queries
    query_graphs = [p.G_vuln for p in query_pairs]
    query_embs = embedder.embed_many(query_graphs)
    print(f"  Query embeddings: {query_embs.shape}")

    # Build index and evaluate retrieval
    ks = cfg.get("experiment", {}).get("ks", [1, 5, 10])
    index, retriever = build_hnsw(
        index_pairs, index_embs, "gin_codebert", embedder.dim, output_dir,
        tag="trained",
    )

    # Build index metadata for CWE recall
    index_metadata = [
        {"cve_id": p.cve_id, "cwe_id": p.cwe_id, "func_name": p.func_name,
         "variant": p.meta.get("variant", ""), **p.meta}
        for p in index_pairs
    ]

    # CWE-level evaluation (aligned with training objective)
    cwe_recall = evaluate_cwe_recall(query_pairs, query_embs, retriever, index_metadata, top_k=10)
    cwe_recall.pop("raw_queries", None)
    cwe_recall.pop("per_cwe", None)

    # CVE-level evaluation (standard retrieval)
    retrieval = evaluate_retrieval(query_pairs, query_embs, retriever, index_pairs, ks=ks)
    retrieval.pop("raw_queries", None)

    # Print results
    print(f"\n{'='*60}")
    print("CWE Recall (training objective):")
    print(f"  macro_avg:    {cwe_recall.get('macro_avg', 0):.4f}")
    print(f"  ranx_recall:  {cwe_recall.get('ranx_recall', 0):.4f}")
    print(f"  n_cwes:       {cwe_recall.get('n_cwes', 0)}")
    print(f"  n_singletons: {cwe_recall.get('n_singletons', 0)}")
    print(f"\nCVE Retrieval (downstream task):")
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
            "n_classes": len(unique_labels),
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

    parser = argparse.ArgumentParser(description="Train GIN-CodeBERT with triplet loss")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Allow CLI override of epochs
    if args.epochs is not None:
        cfg.setdefault("gin_codebert", {}).setdefault("training", {})["epochs"] = args.epochs

    output_dir = Path(args.output) if args.output else None
    run_training(cfg, output_dir)
