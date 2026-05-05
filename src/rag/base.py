"""Protocol defining the shared interface for all vector index backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorIndex(Protocol):
    """Structural interface shared by FAISSIndex and HNSWIndex."""

    dim: int
    index_path: Path
    metadata_path: Path
    metadata: list[dict]

    def add(self, pair, embedding: np.ndarray, variant: str) -> None: ...
    def add_raw(self, embedding: np.ndarray, meta: dict) -> None: ...
    def save(self) -> None: ...
    def load(self) -> None: ...
