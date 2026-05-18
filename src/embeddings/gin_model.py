"""
Trainable GIN model that accepts CodeBERT node features (768-d).

Architecture:
  input_proj(768 → hidden) → N×GINConv(MLP) + BatchNorm + Dropout
  → global_add_pool ‖ global_mean_pool → readout(hidden*2 → dim)

Unlike the frozen GIN in gin.py, this model has trainable parameters
and is optimized with triplet loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

import networkx as nx
import numpy as np

from .wl import NODE_TYPES, NODE_TYPE_IDX


def nx_to_pyg_codebert(
    G: nx.MultiDiGraph, node_features: np.ndarray
) -> Data | None:
    """
    Convert a NetworkX graph to PyG Data using precomputed CodeBERT
    node features (n_nodes, 768).
    """
    nodes = list(G.nodes())
    if not nodes or node_features.shape[0] < 2:
        return None

    idx = {n: i for i, n in enumerate(nodes)}

    edge_index = [[], []]
    for u, v in G.edges():
        if u in idx and v in idx:
            edge_index[0].append(idx[u])
            edge_index[1].append(idx[v])

    x = torch.tensor(node_features, dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long) if edge_index[0] else torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=ei)


class GINCodeBERTModel(nn.Module):
    """
    Trainable GIN that takes 768-d CodeBERT node features as input.

    Parameters:
        in_dim: Input feature dimension (768 for CodeBERT)
        hidden_dim: Hidden dimension for GIN layers (default 128)
        out_dim: Output embedding dimension (default 128)
        num_layers: Number of GIN convolution layers (default 3)
        dropout: Dropout rate (default 0.3)
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Project 768-d CodeBERT features to hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Readout: concat add_pool and mean_pool → project to out_dim
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Batch with x=(total_nodes, in_dim), edge_index, batch

        Returns:
            Tensor of shape (batch_size, out_dim), L2-normalized
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.input_proj(x))

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level readout
        out = torch.cat(
            [global_add_pool(x, batch), global_mean_pool(x, batch)],
            dim=1,
        )
        out = self.readout(out)

        # L2 normalize
        out = F.normalize(out, p=2, dim=1)
        return out

    def embed_graph(self, data: Data) -> torch.Tensor:
        """Embed a single graph (no batch dimension assumed)."""
        if data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
        self.eval()
        with torch.no_grad():
            return self.forward(data)
