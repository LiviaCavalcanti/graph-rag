"""RAG index utilities — build, populate, and load-or-build helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.rag.base import VectorIndex
from src.rag.retriever import Retriever


def populate_index(
    index: VectorIndex,
    pairs: list,
    embeddings: np.ndarray,
    embedder_name: str,
    top_k: int = 10,
) -> Retriever:
    """Add pairs + embeddings to *index*, save, reload, return a Retriever."""
    for pair, vec in zip(pairs, embeddings):
        index.add(pair, vec, embedder_name)
    index.save()
    index.load()
    return Retriever(index, top_k=top_k)


def load_or_build(
    index: VectorIndex,
    pairs: list,
    embeddings: np.ndarray,
    embedder_name: str,
    top_k: int = 10,
) -> Retriever:
    """Try loading an existing index from disk; fall back to building fresh.

    The index is considered valid when both files exist on disk and the
    stored vector count matches ``len(pairs)``.
    """
    if index.index_path.exists() and index.metadata_path.exists():
        index.load()
        if index.index.ntotal == len(pairs):
            return Retriever(index, top_k=top_k)
    return populate_index(index, pairs, embeddings, embedder_name, top_k=top_k)
