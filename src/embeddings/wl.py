import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import WLConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import normalize
from .base import BaseEmbedder

NODE_TYPES = [
    'METHOD', 'METHOD_PARAMETER_IN', 'BLOCK', 'LOCAL',
    'CALL', 'IDENTIFIER', 'LITERAL', 'RETURN',
    'CONTROL_STRUCTURE', 'FIELD_IDENTIFIER', 'UNKNOWN'
]
NODE_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}


def nx_to_pyg(G: nx.MultiDiGraph) -> Data | None:
    nodes = list(G.nodes())
    if not nodes:
        return None
    idx = {n: i for i, n in enumerate(nodes)}

    colours = []
    for n in nodes:
        ntype = G.nodes[n].get('labelV', 'UNKNOWN')
        colours.append(NODE_TYPE_IDX.get(ntype, len(NODE_TYPES) - 1))

    edge_index = [[], []]
    for u, v in G.edges():
        if u in idx and v in idx:
            edge_index[0].append(idx[u])
            edge_index[1].append(idx[v])

    return Data(
        x          = torch.tensor(colours, dtype=torch.long),
        edge_index = torch.tensor(edge_index, dtype=torch.long),
    )


class WLEmbedder(BaseEmbedder):
    """
    Weisfeiler-Lehman colour refinement → sum pool per iteration
    → concatenate → linear projection.
    No training required.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.num_iterations = cfg.get('wl', {}).get('num_iterations', 4)
        self.hidden_dim     = cfg.get('wl', {}).get('hidden_dim', 64)

        self.convs     = torch.nn.ModuleList([WLConv() for _ in range(self.num_iterations)])
        self.embedding = torch.nn.Embedding(8192, self.hidden_dim)
        self.proj      = torch.nn.Linear(self.hidden_dim * self.num_iterations, self.dim)

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

        colours = data.x
        pooled  = []
        for conv in self.convs:
            colours = conv(colours, data.edge_index)
            colours = colours % self.embedding.num_embeddings
            emb     = self.embedding(colours)
            pooled.append(global_add_pool(emb, data.batch))  # (1, hidden_dim)

            
        out = torch.cat(pooled, dim=1)           # (1, hidden_dim * num_iterations)
        out = self.proj(out).detach().numpy()[0] # (dim,)

        norm = np.linalg.norm(out)
        return (out / (norm + 1e-8)).astype(np.float32)