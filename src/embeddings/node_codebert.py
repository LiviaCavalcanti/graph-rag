"""
Per-node CodeBERT feature extraction with disk caching.

Encodes each node's CODE attribute through frozen CodeBERT → 768-d vector.
Results are cached to disk keyed by a hash of the graph's node codes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

_CACHE_DIR = Path("workspace/node_codebert_cache")


def _graph_hash(G: nx.MultiDiGraph) -> str:
    """Deterministic hash of a graph's node CODE content for caching."""
    codes = []
    for n in sorted(G.nodes()):
        code = (G.nodes[n].get("CODE", "") or "").strip()
        codes.append(code)
    raw = "\n".join(codes).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class NodeCodeBERTEncoder:
    """
    Extracts per-node CodeBERT [CLS] embeddings for all nodes in a graph.

    Nodes with empty/missing CODE get zero vectors.
    Results are cached to disk for efficiency.
    """

    def __init__(self, cfg: dict):
        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else cfg.get("rgcn", {}).get("device", "cpu")
        )
        self._model_path = cfg.get("codebert", {}).get(
            "model_path",
            cfg.get("rgcn", {}).get(
                "codebert_model",
                "/home/z0050s2b/code/graph-rag/models/codebert-base/",
            ),
        )
        self._batch_size = cfg.get("gin_codebert", {}).get("cb_batch_size", 32)
        self._cache_dir = Path(
            cfg.get("gin_codebert", {}).get("node_cache_dir", str(_CACHE_DIR))
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._codebert_dim = 768

    def _load(self):
        if self._loaded:
            return
        from transformers import AutoModel, AutoTokenizer

        print(f"  [node_codebert] Loading CodeBERT from {self._model_path} on {self._device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, local_files_only=True
        )
        self._model = AutoModel.from_pretrained(
            self._model_path, local_files_only=True
        )
        self._model.eval().to(self._device)
        self._loaded = True
        print(f"  [node_codebert] CodeBERT loaded on {self._device}")

    def _cache_path(self, graph_hash: str) -> Path:
        return self._cache_dir / f"{graph_hash}.npy"

    def encode_graph(self, G: nx.MultiDiGraph) -> np.ndarray:
        """
        Encode all nodes in G → (n_nodes, 768) array.

        Uses disk cache if available.
        """
        gh = _graph_hash(G)
        cache_file = self._cache_path(gh)
        if cache_file.exists():
            return np.load(cache_file)

        self._load()
        nodes = list(G.nodes())
        n = len(nodes)
        out = np.zeros((n, self._codebert_dim), dtype=np.float32)

        # Collect code strings
        code_strings = []
        for nd in nodes:
            code = (G.nodes[nd].get("CODE", "") or "").strip()
            code_strings.append(code)

        # Find non-empty indices
        nonempty = [(i, s) for i, s in enumerate(code_strings) if s]
        if nonempty:
            indices, strings = zip(*nonempty)
            all_cls = []
            bs = self._batch_size
            for start in range(0, len(strings), bs):
                batch = list(strings[start:start + bs])
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,  # per-node code is short
                    return_tensors="pt",
                ).to(self._device)
                with torch.no_grad():
                    cls = self._model(**enc).last_hidden_state[:, 0, :]
                all_cls.append(cls.cpu())

            cls_mat = torch.cat(all_cls, dim=0).numpy()
            for pos, orig_idx in enumerate(indices):
                out[orig_idx] = cls_mat[pos]

        # Cache to disk
        np.save(cache_file, out)
        return out

    def encode_graphs_batch(
        self, graphs: list[nx.MultiDiGraph]
    ) -> list[np.ndarray]:
        """Encode multiple graphs, returning list of (n_nodes_i, 768) arrays."""
        results = []
        for G in graphs:
            results.append(self.encode_graph(G))
        return results
