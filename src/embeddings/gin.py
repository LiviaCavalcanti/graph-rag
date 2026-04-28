import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

from .base import BaseEmbedder
from .wl import NODE_TYPES, nx_to_pyg


class GINEmbedder(BaseEmbedder):
    """
    Untrained GIN — random MLP weights still produce structurally
    discriminative embeddings. Outperforms NetLSD on small graphs
    because it operates on node features, not just the spectrum.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        in_dim = len(NODE_TYPES)
        hidden_dim = cfg.get("gin", {}).get("hidden_dim", 128)
        num_layers = cfg.get("gin", {}).get("num_layers", 3)

        seed = cfg.get("gin", {}).get("seed", 42)
        torch.manual_seed(seed)

        self.input_proj = torch.nn.Linear(in_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.readout = torch.nn.Linear(hidden_dim * 2, self.dim)

        # freeze — we use this as a fixed feature extractor
        for p in self.parameters():
            p.requires_grad_(False)

    def parameters(self):
        return (
            list(self.input_proj.parameters())
            + list(self.convs.parameters())
            + list(self.bns.parameters())
            + list(self.readout.parameters())
        )

    @property
    def name(self) -> str:
        return "gin"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        data = nx_to_pyg(G)
        if data is None or data.x.shape[0] < 2:
            return np.zeros(self.dim, dtype=np.float32)

        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)

        # one-hot from integer colour
        x = F.one_hot(data.x, num_classes=len(NODE_TYPES)).float()
        x = F.relu(self.input_proj(x))

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index)))

        out = torch.cat(
            [
                global_add_pool(x, data.batch),
                global_mean_pool(x, data.batch),
            ],
            dim=1,
        )
        out = self.readout(out).detach().numpy()[0]

        return self._norm_vec(out)
