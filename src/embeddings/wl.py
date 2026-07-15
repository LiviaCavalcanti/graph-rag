import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import WLConv, global_add_pool

from .base import BaseEmbedder

NODE_TYPES = [
    "METHOD",
    "METHOD_PARAMETER_IN",
    "BLOCK",
    "LOCAL",
    "CALL",
    "IDENTIFIER",
    "LITERAL",
    "RETURN",
    "CONTROL_STRUCTURE",
    "FIELD_IDENTIFIER",
    "UNKNOWN",
]
NODE_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}


def nx_to_pyg(G: nx.MultiDiGraph) -> Data | None:
    nodes = list(G.nodes())
    if not nodes:
        return None
    idx = {n: i for i, n in enumerate(nodes)}

    colours = []
    for n in nodes:
        ntype = G.nodes[n].get("labelV", "UNKNOWN")
        colours.append(NODE_TYPE_IDX.get(ntype, len(NODE_TYPES) - 1))

    edge_index = [[], []]
    for u, v in G.edges():
        if u in idx and v in idx:
            edge_index[0].append(idx[u])
            edge_index[1].append(idx[v])

    return Data(
        x=torch.tensor(colours, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


# ── Diff-weight encoding (visible to structural embedders) ──────────

DIFF_LABELS = ["removed", "fix_adjacent", "edge_changed", "context", "token_entry"]
DIFF_LABEL_IDX = {d: i for i, d in enumerate(DIFF_LABELS)}


def nx_to_pyg_enriched(G: nx.MultiDiGraph) -> Data | None:
    """
    Extended feature encoding: [labelV one-hot (11) || diff one-hot (5) || diff_weight (1)]
    Total: 17-dim continuous features per node.

    The diff one-hot includes 'token_entry' as a distinct label, so GIN can
    distinguish token-guided entry points from diff-based removed nodes.
    """
    nodes = list(G.nodes())
    if not nodes:
        return None
    idx = {n: i for i, n in enumerate(nodes)}

    features = []
    for n in nodes:
        attr = G.nodes[n]
        # labelV one-hot (11 dims)
        ntype = attr.get("labelV", "UNKNOWN")
        type_idx = NODE_TYPE_IDX.get(ntype, len(NODE_TYPES) - 1)
        type_onehot = [0.0] * len(NODE_TYPES)
        type_onehot[type_idx] = 1.0

        # diff label one-hot (5 dims) — includes token_entry as distinct class
        diff = attr.get("diff", "context")
        diff_idx = DIFF_LABEL_IDX.get(diff, 3)
        diff_onehot = [0.0] * len(DIFF_LABELS)
        diff_onehot[diff_idx] = 1.0

        # diff_weight (1 dim, continuous 0-1)
        dw = float(attr.get("diff_weight", 0.2))

        features.append(type_onehot + diff_onehot + [dw])

    edge_index = [[], []]
    for u, v in G.edges():
        if u in idx and v in idx:
            edge_index[0].append(idx[u])
            edge_index[1].append(idx[v])

    return Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


class WLEmbedder(BaseEmbedder):
    """
    Weisfeiler-Lehman colour refinement → sum pool per iteration
    → concatenate → linear projection.
    No training required.
    """

    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg, apply_norm)
        self.num_iterations = cfg.get("wl", {}).get("num_iterations", 4)
        self.hidden_dim = cfg.get("wl", {}).get("hidden_dim", 64)

        seed = cfg.get("wl", {}).get("seed", 42)
        torch.manual_seed(seed)

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.convs = torch.nn.ModuleList([WLConv() for _ in range(self.num_iterations)])
        self.embedding = torch.nn.Embedding(8192, self.hidden_dim)
        self.proj = torch.nn.Linear(self.hidden_dim * self.num_iterations, self.dim)

        # move model components to device
        self.convs = self.convs.to(self.device)
        self.embedding = self.embedding.to(self.device)
        self.proj = self.proj.to(self.device)

    @property
    def name(self) -> str:
        return "wl"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if G.number_of_nodes() < 3:
            return np.zeros(self.dim, dtype=np.float32)

        data = nx_to_pyg(G)
        if data is None:
            return np.zeros(self.dim, dtype=np.float32)

        # add fake batch dimension
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        # move data to device
        data = data.to(self.device)

        colours = data.x
        pooled = []
        for conv in self.convs:
            colours = conv(colours, data.edge_index)
            colours = colours % self.embedding.num_embeddings
            emb = self.embedding(colours)
            pooled.append(global_add_pool(emb, data.batch))  # (1, hidden_dim)

        out = torch.cat(pooled, dim=1)  # (1, hidden_dim * num_iterations)
        out = self.proj(out).detach().cpu().numpy()[0]  # (dim,)

        return self._norm_vec(out) if self.apply_norm else out

    def embed_many(self, graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        """Batch process multiple graphs for efficiency."""
        # convert graphs to PyG format, filtering out invalid ones
        print("[WL] Starting embed to many ")
        pyg_data_list = []
        valid_indices = []
        print(f"[WL] Converting {len(graphs)} graphs to PyG format...")
        for i, G in enumerate(graphs):
            if G.number_of_nodes() < 3:
                continue
            data = nx_to_pyg(G)
            if data is not None:
                pyg_data_list.append(data)
                valid_indices.append(i)
            if (i + 1) % max(1, len(graphs) // 10) == 0:
                print(f"  [{i + 1}/{len(graphs)}] graphs processed")

        # allocate result array
        results = np.zeros((len(graphs), self.dim), dtype=np.float32)

        if not pyg_data_list:
            print("  [WL] Warning: No valid graphs. Returning zeroed results")
            return results  # all graphs invalid

        print(f"[WL] Batching {len(pyg_data_list)} valid graphs...")
        # batch all valid graphs
        batch_data = Batch.from_data_list(pyg_data_list)
        # move batch to device
        batch_data = batch_data.to(self.device)

        print(f"[WL] Running forward pass on GPU batch...")
        colours = batch_data.x
        pooled = []
        for conv in self.convs:
            colours = conv(colours, batch_data.edge_index)
            colours = colours % self.embedding.num_embeddings
            emb = self.embedding(colours)
            pooled.append(
                global_add_pool(emb, batch_data.batch)
            )  # (batch_size, hidden_dim)

        out = torch.cat(pooled, dim=1)  # (batch_size, hidden_dim * num_iterations)
        out = self.proj(out).detach().cpu().numpy()  # (batch_size, dim)

        print(f"[WL] Normalizing embeddings...")
        # assign normalized embeddings to valid indices
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = self._norm_vec(out[i]) if self.apply_norm else out[i]

        print(f"[WL] Done! Embedded {len(pyg_data_list)}/{len(graphs)} graphs")
        return results
