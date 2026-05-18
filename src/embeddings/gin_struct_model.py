"""
Trainable GIN model using structural node-type features (11-d one-hot).

Same architecture as the frozen GIN in gin.py, but with trainable weights
optimized via triplet loss using CVE/dir_name-level positives.

This approach preserves the structural signal that makes the frozen GIN
effective while learning a metric space aligned with the retrieval task.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

import networkx as nx
import numpy as np

from .wl import NODE_TYPES, NODE_TYPE_IDX, nx_to_pyg


NUM_NODE_TYPES = len(NODE_TYPES)


class GINStructModel(nn.Module):
    """
    Trainable GIN with 11-d node-type one-hot input features.

    Architecture matches the frozen GIN exactly, but weights are learnable:
      one_hot(11) → input_proj(11→hidden) → NxGINConv(MLP) + BN
      → global_add_pool ‖ global_mean_pool → readout(hidden*2 → out_dim)

    Parameters:
        hidden_dim: Hidden dimension (default 128)
        out_dim: Output embedding dimension (default 128)
        num_layers: Number of GIN convolution layers (default 3)
        dropout: Dropout rate applied between conv layers during training (default 0.2)
        frozen_compat: If True, use exact frozen GIN architecture (BatchNorm,
                       no MLP dropout, single Linear readout) for warm start
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        frozen_compat: bool = False,
    ):
        super().__init__()
        self.in_dim = NUM_NODE_TYPES
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.frozen_compat = frozen_compat

        self.input_proj = nn.Linear(NUM_NODE_TYPES, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            if frozen_compat:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINConv(mlp, train_eps=False))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINConv(mlp, train_eps=True))
                self.bns.append(nn.LayerNorm(hidden_dim))

        if frozen_compat:
            self.readout = nn.Linear(hidden_dim * 2, out_dim)
        else:
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

    def warm_start_from_frozen(self, cfg: dict) -> None:
        """
        Initialize weights from the frozen GIN (gin.py) which uses seed 42.

        When frozen_compat=True, this copies ALL weights exactly (architectures match).
        When frozen_compat=False, transfers what can be matched between architectures.
        """
        from src.embeddings.gin import GINEmbedder

        frozen = GINEmbedder(cfg, apply_norm=False)

        # input_proj: exact match (Linear 11→128)
        self.input_proj.weight.data.copy_(frozen.input_proj.weight.data)
        self.input_proj.bias.data.copy_(frozen.input_proj.bias.data)

        if self.frozen_compat:
            # Exact architecture match — copy everything directly
            for frozen_conv, struct_conv in zip(frozen.convs, self.convs):
                frozen_mlp = frozen_conv.nn
                struct_mlp = struct_conv.nn
                # Both: Sequential(Linear, ReLU, Linear)
                struct_mlp[0].weight.data.copy_(frozen_mlp[0].weight.data)
                struct_mlp[0].bias.data.copy_(frozen_mlp[0].bias.data)
                struct_mlp[2].weight.data.copy_(frozen_mlp[2].weight.data)
                struct_mlp[2].bias.data.copy_(frozen_mlp[2].bias.data)

            # BatchNorm: copy running stats and affine params
            for frozen_bn, struct_bn in zip(frozen.bns, self.bns):
                struct_bn.load_state_dict(frozen_bn.state_dict())

            # Readout: both are Linear(256→128)
            self.readout.weight.data.copy_(frozen.readout.weight.data)
            self.readout.bias.data.copy_(frozen.readout.bias.data)
        else:
            # Architecture differs — transfer what we can
            for frozen_conv, struct_conv in zip(frozen.convs, self.convs):
                frozen_mlp = frozen_conv.nn  # Sequential(Linear, ReLU, Linear)
                struct_mlp = struct_conv.nn  # Sequential(Linear, ReLU, Dropout, Linear)
                struct_mlp[0].weight.data.copy_(frozen_mlp[0].weight.data)
                struct_mlp[0].bias.data.copy_(frozen_mlp[0].bias.data)
                struct_mlp[3].weight.data.copy_(frozen_mlp[2].weight.data)
                struct_mlp[3].bias.data.copy_(frozen_mlp[2].bias.data)

            # Readout: frozen is Linear(256→128), struct is Sequential(Linear(256→128), ...)
            self.readout[0].weight.data.copy_(frozen.readout.weight.data)
            self.readout[0].bias.data.copy_(frozen.readout.bias.data)

        print(f"  [gin_struct] Warm-started from frozen GIN (seed={cfg.get('gin', {}).get('seed', 42)}, compat={self.frozen_compat})")

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Batch with x=(total_nodes,) long tensor of node type indices,
                  edge_index, batch

        Returns:
            Tensor of shape (batch_size, out_dim), L2-normalized
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # One-hot encode node types
        x = F.one_hot(x, num_classes=NUM_NODE_TYPES).float()
        x = F.relu(self.input_proj(x))

        if self.frozen_compat:
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, edge_index)))
        else:
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

        # L2 normalize (eps prevents NaN for zero vectors)
        out = F.normalize(out, p=2, dim=1, eps=1e-8)
        return out
