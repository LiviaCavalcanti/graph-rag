"""
Repeated runner for combining strategy experiments.

Usage:
    uv run python -m experiments.exp.combining_repeated --config config.yaml [--runs 10] [--experiment A|B|both]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from experiments.common import build_flat_index, evaluate_retrieval
from experiments.exp.combining_strategies.scripts._combining_strategies import NormConcatPCA, NormPCAConcat, PCAConcatPCA
from experiments.repeated_experience_runner import run_repeated
from src.io import load_config

EXPERIMENTS = {
    "A": {
        "norm_concat_pca": {"cls": NormConcatPCA},
        "pca_concat_pca": {"cls": PCAConcatPCA},
    },
    "B": {
        "norm_concat_pca": {"cls": NormConcatPCA},
        "norm_pca_concat": {"cls": NormPCAConcat},
    },
    "both": {
        "norm_concat_pca": {"cls": NormConcatPCA},
        "pca_concat_pca": {"cls": PCAConcatPCA},
        "norm_pca_concat": {"cls": NormPCAConcat},
    },
}


def evaluate_combining(index_pairs, query_pairs, embedder, variant_name, variant_def, run_dir, ks, **kw):
    """Evaluate a single combining strategy on one split."""
    np.random.seed(42)
    torch.manual_seed(42)

    emb_cfg = kw["emb_cfg"]
    strategy = variant_def["cls"](emb_cfg)

    graphs = [p.G_vuln for p in index_pairs]
    index_embs = strategy.embed_many(graphs)

    query_graphs = [p.G_vuln for p in query_pairs]
    query_embs = strategy.embed_many(query_graphs)

    _, retriever = build_flat_index(index_pairs, index_embs, strategy.name, strategy.dim)
    sr = evaluate_retrieval(query_pairs, query_embs, retriever, index_pairs, ks=ks)
    sr.pop("raw_queries", None)

    return {"variant": variant_name, **sr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeated combining strategy experiments")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--experiment", choices=["A", "B", "both"], default="both")
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant_defs = EXPERIMENTS[args.experiment]

    run_repeated(
        cfg,
        n_runs=args.runs,
        base_seed=args.base_seed,
        evaluate_fn=evaluate_combining,
        variant_defs=variant_defs,
        embedders=None,
        pre_variant_hook=None,
        tag=f"combining_{args.experiment}_repeated{args.runs}",
        evaluate_kw={"emb_cfg": cfg.get("embeddings", {})},
    )
