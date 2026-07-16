from abc import ABC, abstractmethod

import networkx as nx
import numpy as np


class BaseEmbedder(ABC):
    def __init__(self, cfg: dict, apply_norm: bool = True):
        self.cfg = cfg
        self.dim = cfg.get("dim", 128)
        self.projection = cfg.get("projection", "pca")  # "pca" or "none"
        self.l2_normalize = cfg.get("l2_normalize", True)
        self.apply_norm = apply_norm

    def _norm_vec(self, v: np.ndarray) -> np.ndarray:
        """L2-normalize a single vector if l2_normalize is enabled."""
        if not self.l2_normalize:
            return v.astype(np.float32)
        norm = np.linalg.norm(v)
        return (v / (norm + 1e-8)).astype(np.float32)

    def _norm_mat(self, M: np.ndarray) -> np.ndarray:
        """L2-normalize each row of a matrix if l2_normalize is enabled."""
        if not self.l2_normalize:
            return M.astype(np.float32)
        from sklearn.preprocessing import normalize

        return normalize(M.astype(np.float32), norm="l2")

    @abstractmethod
    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray: ...

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        results = []
        for G in graphs:
            try:
                results.append(self.embed_one(G))
            except Exception:
                results.append(np.zeros(self.dim, dtype=np.float32))

        return np.stack(results).astype(np.float32)

    @property
    @abstractmethod
    def name(self) -> str: ...

    # ── PCA lifecycle contract ───────────────────────────────────────────
    # Embedders that apply a data-dependent projection (PCA) set this to
    # True and implement fit() / get_pca_state() / load_pca_state().
    # Non-PCA embedders (WL, GIN, NetLSD, …) leave all three as no-ops.
    #
    # Training rule:  fit() once on the full index corpus inside build_index.
    # Inference rule: load_pca_state() from the persisted artifact; never
    #                 call fit() again at query / batch time.
    requires_fitting: bool = False

    def fit(self, graphs: list) -> None:
        """Fit PCA on a representative corpus.  No-op for non-PCA embedders."""

    def get_pca_state(self) -> "dict | None":
        """Return serialisable PCA state for persistence.  None if no PCA."""
        return None

    def load_pca_state(self, state: dict) -> None:
        """Restore PCA state produced by get_pca_state().  No-op if no PCA."""



def resolve_codebert_path(cfg: dict) -> str:
    """Resolve the local CodeBERT model path from an ``embeddings`` config.

    Single source of truth for the CodeBERT-based embedders.  Prefers the
    canonical ``codebert.model_path`` key, falls back to the legacy
    ``rgcn.codebert_model`` key, then to a repo-relative default.  Keeping the
    path in config (not hard-coded in source) makes the repo portable.
    """
    return (
        cfg.get("codebert", {}).get("model_path")
        or cfg.get("rgcn", {}).get("codebert_model")
        or "models/codebert-base/"
    )
