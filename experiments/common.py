"""
Shared experiment primitives — data loading, index building, evaluation, I/O.

Every experiment script should use these instead of reimplementing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import load_pairs  # noqa: F401
from src.data.split import build_split  # noqa: F401
import faiss

from src.rag.hnsw import HNSWIndex
from src.rag.retriever import Retriever


# ── index building ───────────────────────────────────────────────────


class _FlatIndex:
    """Minimal flat inner-product index with the same interface as HNSWIndex."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add_batch(self, pairs: list, embeddings: np.ndarray, variant: str):
        self.index.add(embeddings.astype(np.float32))
        for pair in pairs:
            self.metadata.append(
                {
                    "cve_id": pair.cve_id,
                    "cwe_id": pair.cwe_id,
                    "func_name": pair.func_name,
                    "project": pair.project,
                    "variant": variant,
                    "n_nodes": pair.G_vuln.number_of_nodes(),
                    **pair.meta,
                }
            )


def build_flat_index(
    pairs: list,
    embeddings: np.ndarray,
    embedder_name: str,
    dim: int,
) -> tuple[_FlatIndex, Retriever]:
    """Build a deterministic exact-search (brute-force) index.

    Unlike HNSW, this has no random level assignment — results are
    perfectly reproducible across runs.
    """
    index = _FlatIndex(dim=dim)
    index.add_batch(pairs, embeddings, embedder_name)
    return index, Retriever(index, top_k=10)


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
