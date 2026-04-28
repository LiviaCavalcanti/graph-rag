"""
CodeXGLUE-style baseline embedder.

Embeds the FULL function text through CodeBERT [CLS], ignoring all
graph structure and diff information.  This mimics what a CodeXGLUE
defect detection model would produce as a function representation.

Comparison with codebert_seq (which embeds only changed code selected
via diff_weight from the CPG diff) shows whether graph-aware code
selection adds value beyond a flat-sequence approach.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import torch

from .base import BaseEmbedder
from .codebert_seq import CodeBERTSeqEmbedder


def extract_full_function(G: nx.MultiDiGraph, max_tokens: int = 500) -> str:
    """
    Concatenate ALL node CODE in LINE_NUMBER order to reconstruct
    the full function text.  No diff awareness.
    """
    nodes_with_code = []
    for n in G.nodes():
        code = (G.nodes[n].get("CODE") or "").strip()
        if not code:
            continue
        line = int(G.nodes[n].get("LINE_NUMBER", 9999) or 9999)
        nodes_with_code.append((line, code))

    nodes_with_code.sort()

    parts, tok_count = [], 0
    for _, code in nodes_with_code:
        words = code.split()
        if tok_count + len(words) > max_tokens:
            remaining = max_tokens - tok_count
            if remaining > 0:
                parts.append(" ".join(words[:remaining]))
            break
        parts.append(code)
        tok_count += len(words)

    return " ".join(parts)


class CodeXGLUEBaselineEmbedder(BaseEmbedder):
    """
    Full-function CodeBERT — no graph awareness, no diff awareness.
    CodeXGLUE defect detection style representation.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._cb = CodeBERTSeqEmbedder(cfg)
        self._pca = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "codexglue_baseline"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if self.projection != "none" and not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        self._cb._load_codebert()
        text = extract_full_function(G)
        raw = self._cb.encode_batch([text])

        if self.projection == "none":
            return self._norm_vec(raw[0])

        projected = self._pca.transform(raw)[0].astype(np.float32)
        if projected.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[: projected.shape[0]] = projected
            projected = padded
        return self._norm_vec(projected)

    def embed_many(self, graphs: list) -> np.ndarray:
        self._cb._load_codebert()

        texts = [extract_full_function(G) for G in graphs]
        raw = self._cb.encode_batch(texts)

        valid = np.linalg.norm(raw, axis=1) > 1e-8

        if self.projection == "none":
            self.dim = raw.shape[1]
            out = self._norm_mat(raw)
            out[~valid] = 0.0
            print(f"    [codexglue_baseline] no projection — dim={self.dim}")
            return out

        from sklearn.decomposition import PCA

        out = np.zeros((len(graphs), self.dim), dtype=np.float32)
        if not valid.any():
            return out

        if not self._fitted:
            valid_raw = raw[valid]
            n_comp = min(self.dim, valid_raw.shape[0] - 1, valid_raw.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(valid_raw)
            self._fitted = True
            expl = self._pca.explained_variance_ratio_.sum()
            print(
                f"    [codexglue_baseline] PCA fitted — {n_comp} comp, "
                f"explained variance: {expl:.2%}"
            )

        projected = self._pca.transform(raw).astype(np.float32)
        if projected.shape[1] < self.dim:
            padded = np.zeros((projected.shape[0], self.dim), dtype=np.float32)
            padded[:, : projected.shape[1]] = projected
            projected = padded
        projected = self._norm_mat(projected)
        projected[~valid] = 0.0
        return projected
