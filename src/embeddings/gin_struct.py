"""
GIN-Struct Embedder — BaseEmbedder wrapper for the trained structural GIN.

Uses the same 11-d node-type one-hot features as the frozen GIN,
but with weights trained via triplet loss (CVE-level positives).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch

from .base import BaseEmbedder
from .gin_struct_model import GINStructModel
from .wl import NODE_TYPES, nx_to_pyg

import torch.nn.functional as F


class GINStructEmbedder(BaseEmbedder):
    """
    Trained structural GIN with node-type features.

    Requires a trained checkpoint. Without one, uses random initialization.

    Config keys (under embeddings.gin_struct):
        checkpoint_path: path to saved .pt checkpoint
        hidden_dim: GIN hidden dimension (default 128)
        num_layers: GIN layers (default 3)
        dropout: dropout rate (default 0.2)
    """

    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg, apply_norm)
        gs_cfg = cfg.get("gin_struct", {})
        self._checkpoint_path = gs_cfg.get("checkpoint_path", None)
        self._hidden_dim = gs_cfg.get("hidden_dim", 128)
        self._num_layers = gs_cfg.get("num_layers", 3)
        self._dropout = gs_cfg.get("dropout", 0.2)
        self._device = (
            "cuda" if torch.cuda.is_available()
            else gs_cfg.get("device", "cpu")
        )
        self._model: GINStructModel | None = None
        self._loaded = False

    def _ensure_model(self):
        if self._loaded:
            return
        if self._checkpoint_path and Path(self._checkpoint_path).exists():
            from src.training.struct_trainer import StructTripletTrainer
            self._model = StructTripletTrainer.load_checkpoint(
                Path(self._checkpoint_path), device=self._device
            )
            self.dim = self._model.out_dim
            print(f"  [gin_struct] Loaded checkpoint: {self._checkpoint_path}")
        else:
            self._model = GINStructModel(
                hidden_dim=self._hidden_dim,
                out_dim=self.dim,
                num_layers=self._num_layers,
                dropout=self._dropout,
            ).to(self._device)
            self._model.eval()
            if self._checkpoint_path:
                print(f"  [gin_struct] No checkpoint at {self._checkpoint_path}, using random init")
            else:
                print(f"  [gin_struct] No checkpoint configured, using random init")
        self._loaded = True

    @property
    def name(self) -> str:
        return "gin_struct"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        self._ensure_model()
        data = nx_to_pyg(G)
        if data is None or data.x.shape[0] < 2:
            return np.zeros(self.dim, dtype=np.float32)

        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        data = data.to(self._device)

        # Use train mode for BN (per-batch stats, like frozen GIN) but no grad
        self._model.train()
        with torch.no_grad():
            emb = self._model(data).cpu().numpy()[0]

        return self._norm_vec(emb) if self.apply_norm else emb

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        self._ensure_model()
        self._model.train()  # BN uses per-batch stats (matches frozen GIN)

        data_list = []
        valid_idx = []
        for i, G in enumerate(graphs):
            data = nx_to_pyg(G)
            if data is not None and data.x.shape[0] >= 2:
                data_list.append(data)
                valid_idx.append(i)

        results = np.zeros((len(graphs), self.dim), dtype=np.float32)
        if not data_list:
            return results

        # Batch inference
        batch_size = 64
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
