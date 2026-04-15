"""
R-GCN embedder with heterogeneous edge types and rich node features.

Architecture:
  node features = [type_onehot(15) || diff_onehot(8) || diff_weight(1)
                   || codebert_cls(768) || semantic_flags(6)]
  → R-GCN layers (one weight matrix per relation type)
  → weighted global pooling (diff_weight)
  → MLP projection → L2-normalised output

Uses CodeBERT (lazy-loaded) for semantic code embeddings.
Frozen random R-GCN weights — no training needed.
"""

from pathlib import Path
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import FastRGCNConv, global_add_pool, global_mean_pool
from .base import BaseEmbedder


# ── constants ──────────────────────────────────────────────────────────

NODE_TYPES = [
    'METHOD', 'METHOD_PARAMETER_IN', 'METHOD_PARAMETER_OUT',
    'METHOD_RETURN', 'BLOCK', 'LOCAL', 'CALL', 'IDENTIFIER',
    'LITERAL', 'RETURN', 'CONTROL_STRUCTURE', 'FIELD_IDENTIFIER',
    'JUMP_TARGET', 'TYPE_REF', 'UNKNOWN',
]
EDGE_TYPES = [
    'AST', 'CFG', 'CDG', 'REACHING_DEF',
    'REF', 'ARGUMENT', 'RECEIVER', 'CALL',
]
NODE_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
EDGE_TYPE_IDX = {t: i for i, t in enumerate(EDGE_TYPES)}

DIFF_TYPES = ['removed', 'added', 'mutated', 'rewired',
              'context1', 'context2', 'context3', '']
DIFF_IDX   = {t: i for i, t in enumerate(DIFF_TYPES)}


# ── graph size budget ──────────────────────────────────────────────────

def trim_graph(
    G:          nx.MultiDiGraph,
    max_nodes:  int = 200,
) -> nx.MultiDiGraph:
    """
    Keep the top-max_nodes nodes by diff_weight, breaking ties by
    in-degree (more connected = more central to the vuln).

    Called before embedding when G_vuln is too large.
    Preserves all edges between kept nodes.
    """
    if G.number_of_nodes() <= max_nodes:
        return G

    def score(n):
        attr   = G.nodes[n]
        dw     = attr.get('diff_weight', 0.2)
        in_deg = G.in_degree(n)
        return dw * 2 + in_deg / (G.number_of_nodes() + 1)

    top_nodes = sorted(G.nodes(), key=score, reverse=True)[:max_nodes]
    return G.subgraph(top_nodes).copy()


# ── node feature builders ──────────────────────────────────────────────

def _type_onehot(node_type: str) -> torch.Tensor:
    vec = torch.zeros(len(NODE_TYPES))
    vec[NODE_TYPE_IDX.get(node_type, len(NODE_TYPES) - 1)] = 1.0
    return vec


def _diff_onehot(diff_type: str) -> torch.Tensor:
    vec = torch.zeros(len(DIFF_TYPES))
    vec[DIFF_IDX.get(diff_type, len(DIFF_TYPES) - 1)] = 1.0
    return vec


def _code_hash_embedding(code: str, dim: int = 64) -> torch.Tensor:
    """Kept for backwards compatibility only — not used by default."""
    import hashlib
    if not code or not code.strip():
        return torch.zeros(dim)
    h    = hashlib.sha256(code.encode()).digest()
    seed = int.from_bytes(h[:4], 'big')
    rng  = np.random.default_rng(seed)
    vec  = rng.standard_normal(dim).astype(np.float32)
    return torch.from_numpy(vec / (np.linalg.norm(vec) + 1e-8))


def _build_node_features_structural(
    G:        nx.MultiDiGraph,
) -> torch.Tensor:
    """
    Structural-only node features (no pretrained model).
    Each node: [type_onehot(15) || diff_onehot(8) || diff_weight(1) || semantic_flags(6)]
    Total: 30 dims
    """
    rows = []
    for n in G.nodes():
        attr      = G.nodes[n]
        code      = attr.get('CODE', '') or ''
        ntype     = attr.get('labelV', 'UNKNOWN')
        diff      = attr.get('diff', '')
        dw        = float(attr.get('diff_weight', 0.2))

        type_feat = _type_onehot(ntype)
        diff_feat = _diff_onehot(diff)
        dw_feat   = torch.tensor([dw])

        flags = torch.tensor([
            float(bool(re.search(r'[*&]|->', code))),
            float(bool(re.search(r'\b(malloc|alloc|new)\b', code))),
            float(bool(re.search(r'\b(free|delete|kfree)\b', code))),
            float(bool(re.search(r'\b(lock|mutex|spin)\b', code))),
            float(bool(re.search(r'\b(if|assert|check)\b', code))),
            float(len(code.split()) / 20.0),
        ])

        rows.append(torch.cat([type_feat, diff_feat, dw_feat, flags]))

    return torch.stack(rows) if rows else torch.zeros(1, STRUCTURAL_DIM)


# structural feature width: 15 + 8 + 1 + 6
STRUCTURAL_DIM = len(NODE_TYPES) + len(DIFF_TYPES) + 1 + 6  # = 30


def _build_node_features_codebert(
    G:      nx.MultiDiGraph,
    model,
    tokenizer,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Node features with CodeBERT [CLS] token as code representation.
    Each node: [type_onehot(15) || diff_onehot(8) || diff_weight(1)
                || codebert_cls(768) || semantic_flags(6)]
    Total: 798 dims — projected down by RGCN input layer.
    """
    import re
    nodes = list(G.nodes())
    rows  = []

    # batch CODE strings for efficiency
    codes = [
        (G.nodes[n].get('CODE', '') or '')[:128]  # truncate long nodes
        for n in nodes
    ]

    # tokenise all at once
    enc = tokenizer(
        codes,
        padding       = True,
        truncation    = True,
        max_length    = 64,
        return_tensors= 'pt',
    ).to(device)

    with torch.no_grad():
        out = model(**enc).last_hidden_state[:, 0, :]  # [CLS] token, (N, 768)
    out = F.normalize(out, p=2, dim=1).cpu()  # L2-norm to match structural scale

    for i, n in enumerate(nodes):
        attr      = G.nodes[n]
        code      = codes[i]
        ntype     = attr.get('labelV', 'UNKNOWN')
        diff      = attr.get('diff', '')
        dw        = float(attr.get('diff_weight', 0.2))

        type_feat = _type_onehot(ntype)
        diff_feat = _diff_onehot(diff)
        dw_feat   = torch.tensor([dw])
        code_feat = out[i]

        flags = torch.tensor([
            float(bool(re.search(r'[*&]|->', code))),
            float(bool(re.search(r'\b(malloc|alloc|new)\b', code))),
            float(bool(re.search(r'\b(free|delete|kfree)\b', code))),
            float(bool(re.search(r'\b(lock|mutex|spin)\b', code))),
            float(bool(re.search(r'\b(if|assert|check)\b', code))),
            float(len(code.split()) / 20.0),
        ])

        rows.append(torch.cat([type_feat, diff_feat, dw_feat, code_feat, flags]))

    return torch.stack(rows)


def _build_edge_index_and_types(
    G:       nx.MultiDiGraph,
    node_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns edge_index (2, E) and edge_type (E,) for R-GCN.
    Each edge type gets its own relation index.
    """
    srcs, dsts, etypes = [], [], []
    for u, v, data in G.edges(data=True):
        if u not in node_idx or v not in node_idx:
            continue
        label = data.get('label', 'AST')
        srcs.append(node_idx[u])
        dsts.append(node_idx[v])
        etypes.append(EDGE_TYPE_IDX.get(label, 0))

    if not srcs:
        return (torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long))

    return (torch.tensor([srcs, dsts], dtype=torch.long),
            torch.tensor(etypes, dtype=torch.long))


def nx_to_rgcn_data(
    G:        nx.MultiDiGraph,
    x:        torch.Tensor,
    max_nodes: int = 200,
) -> Data | None:
    """Build a PyG Data object with edge types for R-GCN."""
    # G = trim_graph(G, max_nodes=max_nodes)
    if G.number_of_nodes() == 0:
        return None

    nodes    = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    if x.shape[0] != len(nodes):
        return None

    edge_index, edge_type = _build_edge_index_and_types(G, node_idx)

    diff_weights = torch.tensor(
        [G.nodes[n].get('diff_weight', 0.2) for n in nodes],
        dtype=torch.float,
    )

    return Data(
        x           = x,
        edge_index  = edge_index,
        edge_type   = edge_type,
        diff_weights= diff_weights,
        num_nodes   = len(nodes),
    )


# ── R-GCN model ────────────────────────────────────────────────────────

class RGCNModel(nn.Module):
    """
    Relational GCN with diff-weighted pooling.

    Key design choices:
      - One weight matrix per edge type (num_relations = 8)
      - FastRGCNConv: basis decomposition reduces parameter count
        (num_bases=4 means 8 relation matrices are expressed as
         linear combinations of 4 basis matrices — regularises
         rare edge types like RECEIVER)
      - Pooling weighted by diff_weight: removed/added nodes
        contribute more to the graph-level embedding than context
      - Sum + mean concatenation for richer graph representation
    """

    def __init__(
        self,
        in_dim:       int,
        hidden_dim:   int = 256,
        out_dim:      int = 128,
        num_layers:   int = 3,
        num_relations: int = len(EDGE_TYPES),
        num_bases:    int = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(FastRGCNConv(
                hidden_dim, hidden_dim,
                num_relations = num_relations,
                num_bases     = num_bases,
            ))
            # self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.bns.append(nn.LayerNorm(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_type = (
            data.x, data.edge_index, data.edge_type
        )
        batch        = data.batch if hasattr(data, 'batch') and data.batch is not None \
                       else torch.zeros(x.shape[0], dtype=torch.long)
        diff_weights = data.diff_weights if hasattr(data, 'diff_weights') \
                       else torch.ones(x.shape[0])

        x = F.relu(self.input_proj(x))

        for conv, bn in zip(self.convs, self.bns):
            if edge_index.shape[1] > 0:
                x = F.relu(bn(conv(x, edge_index, edge_type)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # diff-weighted pooling: nodes with higher diff_weight
        # contribute more to the graph-level vector
        w = diff_weights.unsqueeze(1).to(x.device)
        x_weighted = x * w

        x_sum  = global_add_pool(x_weighted, batch)
        x_mean = global_mean_pool(x_weighted, batch)
        out    = torch.cat([x_sum, x_mean], dim=1)

        return self.readout(out)


# ── embedder classes ───────────────────────────────────────────────────

class RGCNEmbedder(BaseEmbedder):
    """
    CodeBERT + structural node features → diff-weighted pooling → PCA.

    Node features per node:
        [type_onehot(15) || diff_onehot(8) || diff_weight(1)
         || L2-normalised codebert_cls(768) || semantic_flags(6)]
    Graph-level vector:
        diff-weighted mean pool over all nodes (no trimming).
    Projection:
        PCA fitted on the first embed_many() call — optimal linear
        projection that maximises preserved variance (Johnson-Lindenstrauss
        guarantee for the Gaussian case; PCA is even tighter).
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._device = (
            'cuda' if torch.cuda.is_available()
            else cfg.get('rgcn', {}).get('device', 'cpu')
        )
        self._model_name = cfg.get('rgcn', {}).get(
            'codebert_model',
            '/home/z0050s2b/code/graph-rag/models/codebert-base/',
        )
        self._codebert_dim = 768
        self._in_dim = STRUCTURAL_DIM + self._codebert_dim  # 798
        self._cb_batch_size = cfg.get('rgcn', {}).get('cb_batch_size', 512)

        # lazy-loaded CodeBERT
        self._cb_model     = None
        self._cb_tokenizer = None
        self._cb_available = None  # None = not checked yet

        # PCA projection (fitted on first embed_many call)
        self._pca    = None
        self._fitted = False

    def _load_codebert(self):
        if self._cb_available is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModel
            print(f"  [rgcn] loading {self._model_name} on {self._device}...")
            if not Path(self._model_name).exists():
                raise ValueError(f"Model path {self._model_name} does not exist.")
            self._cb_tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, local_files_only=True,
            )
            self._cb_model = AutoModel.from_pretrained(
                self._model_name, local_files_only=True,
            )
            self._cb_model.eval().to(self._device)
            self._cb_available = True
            print(f"  [rgcn] CodeBERT loaded on {self._device}")
        except Exception as e:
            print(f"  [rgcn] CodeBERT unavailable ({e}), "
                  f"falling back to structural features")
            self._cb_available = False
            self._in_dim = STRUCTURAL_DIM

    @property
    def name(self) -> str:
        return "rgcn"

    # ── batched CodeBERT encoding ──────────────────────────────────

    def _encode_codebert_batched(
        self, all_codes: list[str],
    ) -> torch.Tensor:
        """Run CodeBERT on all code strings in large batches."""
        all_cls: list[torch.Tensor] = []
        bs = self._cb_batch_size
        for start in range(0, len(all_codes), bs):
            batch = all_codes[start:start + bs]
            enc = self._cb_tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors='pt',
            ).to(self._device)
            with torch.no_grad():
                out = self._cb_model(**enc).last_hidden_state[:, 0, :]
            all_cls.append(F.normalize(out, p=2, dim=1).cpu())
        return torch.cat(all_cls, dim=0)

    # ── node-level feature assembly ────────────────────────────────

    @staticmethod
    def _make_structural_row(
        ntype: str, diff: str, dw: float, code: str,
    ) -> torch.Tensor:
        """Build the non-CodeBERT part of a node feature vector."""
        return torch.cat([
            _type_onehot(ntype),
            _diff_onehot(diff),
            torch.tensor([dw]),
            torch.tensor([
                float(bool(re.search(r'[*&]|->', code))),
                float(bool(re.search(r'\b(malloc|alloc|new)\b', code))),
                float(bool(re.search(r'\b(free|delete|kfree)\b', code))),
                float(bool(re.search(r'\b(lock|mutex|spin)\b', code))),
                float(bool(re.search(r'\b(if|assert|check)\b', code))),
                float(len(code.split()) / 20.0),
            ]),
        ])

    # ── graph-level pooling ────────────────────────────────────────

    @staticmethod
    def _pool_graph(
        x: torch.Tensor, diff_weights: list[float],
    ) -> np.ndarray:
        """Diff-weighted mean pool: changed nodes contribute more."""
        w = torch.tensor(diff_weights, dtype=torch.float).unsqueeze(1)
        pooled = (x * w).sum(0) / (w.sum() + 1e-8)
        return pooled.numpy().astype(np.float32)

    # ── build pooled vectors for all graphs at once ────────────────

    def _build_all_pooled(
        self, graphs: list[nx.MultiDiGraph],
    ) -> tuple[np.ndarray, list[int]]:
        """
        Build node features for *all* graphs with a single batched
        CodeBERT pass, then diff-weighted pool each graph.
        Returns (pooled_matrix, valid_graph_indices).
        """
        # 1. collect all code strings + node metadata
        all_codes:  list[str]        = []
        graph_meta: list[list[dict]] = []
        for G in graphs:
            meta = []
            for n in G.nodes():
                attr = G.nodes[n]
                code = (attr.get('CODE', '') or '')[:128]
                all_codes.append(code)
                meta.append({
                    'ntype': attr.get('labelV', 'UNKNOWN'),
                    'diff':  attr.get('diff', ''),
                    'dw':    float(attr.get('diff_weight', 0.2)),
                    'code':  code,
                })
            graph_meta.append(meta)

        # 2. one batched CodeBERT pass (GPU if available)
        if self._cb_available and all_codes:
            all_cls = self._encode_codebert_batched(all_codes)
        else:
            all_cls = None

        # 3. assemble per-node features and pool each graph
        pooled_vecs: list[np.ndarray] = []
        valid_idx:   list[int]        = []
        offset = 0
        for gi, G in enumerate(graphs):
            meta = graph_meta[gi]
            n_nodes = len(meta)
            if n_nodes == 0:
                offset += n_nodes
                continue

            rows = []
            dws  = []
            for j in range(n_nodes):
                nd   = meta[j]
                base = self._make_structural_row(
                    nd['ntype'], nd['diff'], nd['dw'], nd['code'],
                )
                if all_cls is not None:
                    row = torch.cat([base[:24], all_cls[offset + j], base[24:]])
                    # layout: type(15) + diff(8) + dw(1) + codebert(768) + flags(6)
                else:
                    row = base
                rows.append(row)
                dws.append(nd['dw'])

            x = torch.stack(rows)
            pooled_vecs.append(self._pool_graph(x, dws))
            valid_idx.append(gi)
            offset += n_nodes

        if not pooled_vecs:
            return np.zeros((0, self._in_dim), dtype=np.float32), []
        return np.stack(pooled_vecs), valid_idx

    # ── public API ─────────────────────────────────────────────────

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        self._load_codebert()
        if G.number_of_nodes() == 0:
            return np.zeros(self.dim, dtype=np.float32)
        # single-graph path reuses _build_all_pooled for consistency
        raw, valid = self._build_all_pooled([G])
        if raw.shape[0] == 0:
            return np.zeros(self.dim, dtype=np.float32)
        projected = self._pca.transform(raw)[0].astype(np.float32)
        norm = np.linalg.norm(projected)
        return (projected / (norm + 1e-8)).astype(np.float32)

    def embed_many(self, graphs: list) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize
        self._load_codebert()

        raw, valid_idx = self._build_all_pooled(graphs)

        out = np.zeros((len(graphs), self.dim), dtype=np.float32)
        if raw.shape[0] == 0:
            return out

        # fit PCA on the first call
        if not self._fitted:
            n_comp = min(self.dim, raw.shape[0] - 1, raw.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(raw)
            self._fitted = True
            explained = self._pca.explained_variance_ratio_.sum()
            print(f"    [rgcn] PCA fitted — {n_comp} components, "
                  f"explained variance: {explained:.2%}")

        projected = self._pca.transform(raw).astype(np.float32)
        projected = normalize(projected, norm='l2')

        for pos, orig_idx in enumerate(valid_idx):
            out[orig_idx] = projected[pos]
        return out


class RGCNCodeBERTEmbedder(RGCNEmbedder):
    """Alias for backwards compatibility — RGCNEmbedder now uses CodeBERT by default."""

    @property
    def name(self) -> str:
        return "rgcn_codebert"
