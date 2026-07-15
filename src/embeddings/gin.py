import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

from .base import BaseEmbedder
from .wl import DIFF_LABELS, NODE_TYPES, nx_to_pyg, nx_to_pyg_enriched


class GINEmbedder(BaseEmbedder):
    """
    Untrained GIN — random MLP weights still produce structurally
    discriminative embeddings. Outperforms NetLSD on small graphs
    because it operates on node features, not just the spectrum.
    """

    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg, apply_norm)
        in_dim = len(NODE_TYPES)
        hidden_dim = cfg.get("gin", {}).get("hidden_dim", 128)
        num_layers = cfg.get("gin", {}).get("num_layers", 3)

        seed = cfg.get("gin", {}).get("seed", 42)
        torch.manual_seed(seed)

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # move model components to device
        self.input_proj = self.input_proj.to(self.device)
        self.convs = self.convs.to(self.device)
        self.bns = self.bns.to(self.device)
        self.readout = self.readout.to(self.device)

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
        # move data to device
        data = data.to(self.device)

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
        out = self.readout(out).detach().cpu().numpy()[0]

        return self._norm_vec(out) if self.apply_norm else out

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        # convert graphs to PyG format
        pyg_data_list = []
        valid_indices = []
        print(f"[GIN] Converting {len(graphs)} graphs to PyG format...")
        for i, G in enumerate(graphs):
            data = nx_to_pyg(G)
            if data is not None and data.x.shape[0] >= 2:
                pyg_data_list.append(data)
                valid_indices.append(i)
            if (i + 1) % max(1, len(graphs) // 10) == 0:
                print(f"  [{i + 1}/{len(graphs)}] graphs processed")

        results = np.zeros((len(graphs), self.dim), dtype=np.float32)
        if not pyg_data_list:
            print("  [GIN] Warning: No valid graphs. Returning zeroed results")
            return results

        print(f"[GIN] Batching {len(pyg_data_list)} valid graphs...")
        batch_data = Batch.from_data_list(pyg_data_list)
        # move batch to device
        batch_data = batch_data.to(self.device)

        print(f"[GIN] Running forward pass on GPU batch...")
        x = batch_data.x

        if isinstance(x, torch.Tensor) and x.dtype == torch.long:
            x = F.one_hot(x, num_classes=len(NODE_TYPES)).float()
        else:
            x = x.float()

        x = F.relu(self.input_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, batch_data.edge_index)))

        # global pooling
        out = torch.cat(
            [
                global_add_pool(x, batch_data.batch),
                global_mean_pool(x, batch_data.batch),
            ],
            dim=1,
        )

        out = self.readout(out).detach().cpu().numpy()
        print(f"[GIN] Normalizing embeddings...")

        # assign normalized embeddings
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = self._norm_vec(out[i]) if self.apply_norm else out[i]

        print(f"[GIN] Done! Embedded {len(pyg_data_list)}/{len(graphs)} graphs")
        return results


class GINEnrichedEmbedder(BaseEmbedder):
    """
    GIN with enriched node features: [labelV(11) || diff(4) || diff_weight(1) || token_entry(1)]

    Unlike the standard GIN which only sees node types, this variant
    can distinguish entry-point nodes from context, enabling it to learn
    that token-matched nodes form discriminative neighborhoods.
    """

    # enriched feature dim: 11 (node types) + 5 (diff labels) + 1 (weight) = 17
    IN_DIM = len(NODE_TYPES) + len(DIFF_LABELS) + 1

    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg, apply_norm)
        in_dim = self.IN_DIM
        hidden_dim = cfg.get("gin", {}).get("hidden_dim", 128)
        num_layers = cfg.get("gin", {}).get("num_layers", 3)

        seed = cfg.get("gin", {}).get("seed", 42)
        torch.manual_seed(seed)

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"running Gin with device {self.device}")
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

        # move model components to device
        self.input_proj = self.input_proj.to(self.device)
        self.convs = self.convs.to(self.device)
        self.bns = self.bns.to(self.device)
        self.readout = self.readout.to(self.device)

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
        return "gin_enriched"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        data = nx_to_pyg_enriched(G)
        if data is None or data.x.shape[0] < 2:
            return np.zeros(self.dim, dtype=np.float32)

        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        # move data to device
        data = data.to(self.device)

        # features are already continuous floats from nx_to_pyg_enriched
        x = data.x
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
        out = self.readout(out).detach().cpu().numpy()[0]

        return self._norm_vec(out) if self.apply_norm else out

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        # convert graphs to PyG enriched format
        pyg_data_list = []
        valid_indices = []
        print(
            f"[GIN Enriched] Converting {len(graphs)} graphs to enriched PyG format..."
        )
        for i, G in enumerate(graphs):
            data = nx_to_pyg_enriched(G)
            if data is not None and data.x.shape[0] >= 2:
                pyg_data_list.append(data)
                valid_indices.append(i)
            if (i + 1) % max(1, len(graphs) // 10) == 0:
                print(f"  [{i + 1}/{len(graphs)}] graphs processed")

        results = np.zeros((len(graphs), self.dim), dtype=np.float32)
        if not pyg_data_list:
            print("  [GIN Enriched] Warning: No valid graphs. Returning zeroed results")
            return results

        print(f"[GIN Enriched] Batching {len(pyg_data_list)} valid graphs...")
        batch_data = Batch.from_data_list(pyg_data_list)
        # move batch to device
        batch_data = batch_data.to(self.device)

        print(f"[GIN Enriched] Running forward pass on GPU batch...")
        x = batch_data.x
        x = x.float()

        x = F.relu(self.input_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, batch_data.edge_index)))

        # global pooling
        out = torch.cat(
            [
                global_add_pool(x, batch_data.batch),
                global_mean_pool(x, batch_data.batch),
            ],
            dim=1,
        )

        out = self.readout(out).detach().cpu().numpy()
        print(f"[GIN Enriched] Normalizing embeddings...")

        # assign normalized embeddings
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = self._norm_vec(out[i]) if self.apply_norm else out[i]

        print(
            f"[GIN Enriched] Done! Embedded {len(pyg_data_list)}/{len(graphs)} graphs"
        )
        return results
