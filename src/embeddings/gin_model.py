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

from .base import BaseEmbedder
from .codebert_seq import CodeBERTSeqEmbedder
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

    Architecture: Two-stage design to avoid early information bottleneck
      1. Minimal projection: 768 → hidden_dim (preserves semantic richness)
      2. GIN message passing: Refine node features with graph structure
      3. Graph-level pooling & readout: Compress to final dimension

    Parameters:
        in_dim: Input feature dimension (768 for CodeBERT)
        hidden_dim: Hidden dimension for GIN layers (default 256, NOT 128)
        out_dim: Output embedding dimension (default 128)
        num_layers: Number of GIN convolution layers (default 3)
        dropout: Dropout rate (default 0.3)
    
    Key insight: Don't compress too early (768→128 loses 87% of info).
    Instead: 768→256 (preserves ~33% more signal than 768→128),
    then GIN refines with structure, then compress to out_dim at the end.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,  # INCREASED from 128 to preserve semantic signal
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

        # Semantic projection: 768 → hidden_dim (preserve information for GIN to work with)
        # This is a careful balance: compress enough to fit in GPU, not too much to lose signal
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


# ── GINCodeBERTEmbedder: Embedder interface wrapper ──────────────────


class GINCodeBERTEmbedder(BaseEmbedder):
    """
    Embedder wrapper for GINCodeBERTModel.
    
    Uses CodeBERT to embed node code, then refines with graph structure
    via trainable GIN. Output is fixed-dimensional and L2-normalized.
    
    Configuration:
        gin_model.in_dim: 768 (CodeBERT output)
        gin_model.hidden_dim: 256 (internal GIN dimension, default)
        gin_model.out_dim: 128 (or cfg['dim'])
        gin_model.num_layers: 3
        gin_model.dropout: 0.3
        gin_model.device: "cuda" or "cpu"
    """
    
    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg, apply_norm=apply_norm)
        
        # Get GIN model configuration
        gin_cfg = cfg.get("gin_model", {})
        self.in_dim = gin_cfg.get("in_dim", 768)
        self.hidden_dim = gin_cfg.get("hidden_dim", 256)
        self.out_dim = self.dim  # Use base class dim as output dimension
        self.num_layers = gin_cfg.get("num_layers", 3)
        self.dropout = gin_cfg.get("dropout", 0.3)
        
        # Handle device string conversion (gpu -> cuda)
        device_str = gin_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "gpu":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device_str
        
        # Initialize CodeBERT for node features (without compression)
        # CodeBERTSeqEmbedder reads: cfg["codebert"]["dim"] and cfg["rgcn"]["cb_batch_size"]
        # To get raw 768-d vectors, we need to set codebert.dim >= 768
        cb_cfg = {
            "codebert": {
                **cfg.get("codebert", {"model_path": "models/codebert-base/", "seed": 42}),
                "dim": 768,  # Override to get raw 768-d CodeBERT output (no PCA compression)
            },
            "rgcn": cfg.get("rgcn", {"cb_batch_size": 64}),
            "dim": 768,                # Set to 768 so no compression triggers in embed_many
            "projection": "none",      # Disable PCA projection
            "l2_normalize": False,     # Disable L2 norm to preserve raw embeddings
        }
        self.codebert_embedder = CodeBERTSeqEmbedder(cb_cfg, apply_norm=False)
        
        # Initialize GIN model
        self.gin_model = GINCodeBERTModel(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self._device)
        self.gin_model.eval()
    
    @property
    def name(self) -> str:
        return "gin_codebert"
    
    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        """Embed a single graph via CodeBERT node features + GIN refinement."""
        if G.number_of_nodes() == 0:
            return np.zeros(self.out_dim, dtype=np.float32)
        
        # Get CodeBERT node features
        nodes = list(G.nodes())
        code_snippets = [
            (G.nodes[n].get("CODE") or "").strip() or f"node_{n}"
            for n in nodes
        ]
        
        node_features = self.codebert_embedder.encode_batch(code_snippets)  # (N, 768)
        
        # Validate node features shape
        if node_features.shape[1] != self.in_dim:
            raise ValueError(
                f"CodeBERT node features have shape {node_features.shape}, "
                f"but GINCodeBERTModel expects input dimension {self.in_dim}. "
                f"Ensure cfg['projection']='none' and cfg['dim']=768."
            )
        
        # Convert to PyG and embed
        pyg_data = nx_to_pyg_codebert(G, node_features)
        if pyg_data is None or pyg_data.x.shape[0] == 0:
            return np.zeros(self.out_dim, dtype=np.float32)
        
        pyg_data = pyg_data.to(self._device)
        with torch.no_grad():
            embedding = self.gin_model.embed_graph(pyg_data)
        
        # embed_graph returns (1, out_dim) for a single graph; drop batch dim.
        return embedding.cpu().numpy().astype(np.float32)[0]
    
    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        """Embed multiple graphs via CodeBERT + GIN batch processing."""
        embeddings = []
        
        for G in graphs:
            if G.number_of_nodes() == 0:
                embeddings.append(np.zeros(self.out_dim, dtype=np.float32))
                continue
            
            # Get CodeBERT node features
            nodes = list(G.nodes())
            code_snippets = [
                (G.nodes[n].get("CODE") or "").strip() or f"node_{n}"
                for n in nodes
            ]
            
            node_features = self.codebert_embedder.encode_batch(code_snippets)  # (N, 768)
            
            # Validate node features shape
            if node_features.shape[1] != self.in_dim:
                raise ValueError(
                    f"CodeBERT node features have shape {node_features.shape}, "
                    f"but GINCodeBERTModel expects input dimension {self.in_dim}. "
                    f"Ensure cfg['projection']='none' and cfg['dim']=768."
                )
            
            # Convert to PyG and embed
            pyg_data = nx_to_pyg_codebert(G, node_features)
            if pyg_data is None or pyg_data.x.shape[0] == 0:
                embeddings.append(np.zeros(self.out_dim, dtype=np.float32))
                continue
            
            pyg_data = pyg_data.to(self._device)
            with torch.no_grad():
                embedding = self.gin_model.embed_graph(pyg_data)
            
            # embed_graph returns (1, out_dim) for a single graph; drop batch dim.
            embeddings.append(embedding.cpu().numpy().astype(np.float32)[0])
        
        result = np.stack(embeddings) if embeddings else np.zeros((0, self.out_dim), dtype=np.float32)
        
        # Apply normalization if enabled
        if self.l2_normalize:
            result = self._norm_mat(result)
        
        return result
