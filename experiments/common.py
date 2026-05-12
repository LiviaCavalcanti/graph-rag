"""
Shared experiment primitives — data loading, index building, evaluation, I/O.

Every experiment script should use these instead of reimplementing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.autopatch import load_pairs  # noqa: F401
from src.data.split import build_split  # noqa: F401
from src.rag.hnsw import HNSWIndex
from src.rag.retriever import Retriever


# ── index building ───────────────────────────────────────────────────

def build_hnsw(
    pairs: list,
    embeddings: np.ndarray,
    embedder_name: str,
    dim: int,
    run_dir: Path,
    tag: str = "",
) -> tuple[HNSWIndex, Retriever]:
    """Build, save, reload an HNSW index.  Returns (index, retriever)."""
    idx_dir = run_dir / "indices"
    idx_dir.mkdir(exist_ok=True)
    stem = f"{embedder_name}__{tag}" if tag else embedder_name
    index = HNSWIndex(
        dim=dim,
        index_path=str(idx_dir / f"{stem}__hnsw.index"),
        metadata_path=str(idx_dir / f"{stem}__hnsw_meta.json"),
    )
    for pair, vec in zip(pairs, embeddings):
        index.add(pair, vec, embedder_name)
    index.save()
    index.load()
    return index, Retriever(index, top_k=10)


# ── evaluation primitives (canonical home: src/metrics/retrieval_eval) ─
from src.metrics.retrieval_eval import evaluate_cwe_recall, evaluate_retrieval

# ── uncertainty helpers (used by analyze_misses & verify_crossing) ───

def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over retrieval scores."""
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float64) / temperature
    arr -= arr.max()
    exp = np.exp(arr)
    return (exp / exp.sum()).tolist()


def is_uncertain(prob: float, margin: float, prob_floor: float = 0.12, margin_floor: float = 0.005) -> bool:
    return prob < prob_floor or margin < margin_floor


# ── I/O helpers (canonical home: src/io) ─────────────────────────────
from src.io import load_config, make_run_dir, save_json  # noqa: F401
