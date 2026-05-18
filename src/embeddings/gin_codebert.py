"""
GIN-CodeBERT Embedder — BaseEmbedder wrapper for the trained GIN model.

Loads a trained GINCodeBERTModel checkpoint and provides the standard
embed_one / embed_many interface for use in the experiment framework.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch

from .base import BaseEmbedder
from .gin_model import GINCodeBERTModel, nx_to_pyg_codebert
from .node_codebert import NodeCodeBERTEncoder


class GINCodeBERTEmbedder(BaseEmbedder):
    """
    Trained GIN with CodeBERT node features.

    Requires a trained checkpoint. If no checkpoint exists, falls back
    to random initialization (useful for first-time training experiments).

    Config keys (under embeddings.gin_codebert):
        checkpoint_path: path to saved .pt checkpoint
        hidden_dim: GIN hidden dimension (default 128)
        num_layers: GIN layers (default 3)
        dropout: dropout rate (default 0.3)
        cb_batch_size: CodeBERT batch size (default 32)
    """

    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg, apply_norm)
        gc_cfg = cfg.get("gin_codebert", {})
        self._checkpoint_path = gc_cfg.get("checkpoint_path", None)
        self._hidden_dim = gc_cfg.get("hidden_dim", 128)
        self._num_layers = gc_cfg.get("num_layers", 3)
        self._dropout = gc_cfg.get("dropout", 0.3)
        self._device = (
            "cuda" if torch.cuda.is_available()
            else gc_cfg.get("device", "cpu")
        )

        # Node encoder (shared, caches to disk)
        self._node_encoder = NodeCodeBERTEncoder(cfg)

        # Load or create model
        self._model: GINCodeBERTModel | None = None
        self._loaded = False

    def _ensure_model(self):
        if self._loaded:
            return
        if self._checkpoint_path and Path(self._checkpoint_path).exists():
            from src.training import TripletTrainer
            self._model = TripletTrainer.load_checkpoint(
                Path(self._checkpoint_path), device=self._device
            )
            self.dim = self._model.out_dim
            print(f"  [gin_codebert] Loaded checkpoint: {self._checkpoint_path}")
        else:
            self._model = GINCodeBERTModel(
                in_dim=768,
                hidden_dim=self._hidden_dim,
                out_dim=self.dim,
                num_layers=self._num_layers,
                dropout=self._dropout,
            ).to(self._device)
            self._model.eval()
            if self._checkpoint_path:
                print(f"  [gin_codebert] No checkpoint at {self._checkpoint_path}, using random init")
            else:
                print(f"  [gin_codebert] No checkpoint configured, using random init")
        self._loaded = True

    @property
    def name(self) -> str:
        return "gin_codebert"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        self._ensure_model()
        node_feats = self._node_encoder.encode_graph(G)
        data = nx_to_pyg_codebert(G, node_feats)
        if data is None:
            return np.zeros(self.dim, dtype=np.float32)

        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        data = data.to(self._device)

        self._model.eval()
        with torch.no_grad():
            emb = self._model(data).cpu().numpy()[0]

        return self._norm_vec(emb) if self.apply_norm else emb

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        self._ensure_model()
        self._model.eval()

        # Encode node features for all graphs
        data_list = []
        valid_idx = []
        for i, G in enumerate(graphs):
            node_feats = self._node_encoder.encode_graph(G)
            data = nx_to_pyg_codebert(G, node_feats)
            if data is not None:
                data_list.append(data)
                valid_idx.append(i)

        results = np.zeros((len(graphs), self.dim), dtype=np.float32)
        if not data_list:
            return results

        # Batch inference
        batch_size = 32
        all_embs = []
        with torch.no_grad():
            for start in range(0, len(data_list), batch_size):
                batch = Batch.from_data_list(data_list[start:start + batch_size])
                batch = batch.to(self._device)
                embs = self._model(batch).cpu().numpy()
                all_embs.append(embs)

        all_embs = np.concatenate(all_embs, axis=0)
        for j, orig_idx in enumerate(valid_idx):
            results[orig_idx] = all_embs[j]

        return self._norm_mat(results) if self.apply_norm else results
