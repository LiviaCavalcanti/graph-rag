"""
R-GCN embedder: structural histogram + CodeBERT on changed code.

Architecture:
  Stream A — structural histogram (78 dims): node/edge type counts,
             diff-type counts, semantic flags, graph stats, cross-features.
  Stream B — CodeBERT [CLS] on concatenated changed-code per graph (768 dims).
  Projection — PCA on the concatenation (846 dims → self.dim).
"""

import re
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from .base import BaseEmbedder
from .codebert_seq import collect_changed_code

# ── constants ──────────────────────────────────────────────────────────

NODE_TYPES = [
    "METHOD",
    "METHOD_PARAMETER_IN",
    "METHOD_PARAMETER_OUT",
    "METHOD_RETURN",
    "BLOCK",
    "LOCAL",
    "CALL",
    "IDENTIFIER",
    "LITERAL",
    "RETURN",
    "CONTROL_STRUCTURE",
    "FIELD_IDENTIFIER",
    "JUMP_TARGET",
    "TYPE_REF",
    "UNKNOWN",
]
EDGE_TYPES = [
    "AST",
    "CFG",
    "CDG",
    "REACHING_DEF",
    "REF",
    "ARGUMENT",
    "RECEIVER",
    "CALL",
]
NODE_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
EDGE_TYPE_IDX = {t: i for i, t in enumerate(EDGE_TYPES)}

DIFF_TYPES = [
    "removed",
    "added",
    "mutated",
    "rewired",
    "context1",
    "context2",
    "context3",
    "",
]
DIFF_IDX = {t: i for i, t in enumerate(DIFF_TYPES)}


# ── embedder classes ───────────────────────────────────────────────────

# threshold: nodes with diff_weight above this are considered "changed"
_CHANGED_THRESH = 0.3


def _build_structural_histogram(G: nx.MultiDiGraph) -> np.ndarray:
    """
    Graph-level structural histogram — no mean-pooling needed.

    Layout (78 dims):
      changed_node_type_counts  (15)  — node types among changed nodes
      context_node_type_counts  (15)  — node types among context nodes
      diff_type_counts          (8)   — count per diff category
      edge_type_counts          (8)   — count per edge relation
      flag_fracs_changed        (6)   — fraction of changed nodes with each flag
      flag_fracs_all            (6)   — fraction of all nodes with each flag
      graph_stats               (8)   — n_nodes, n_changed, n_edges, density,
                                        mean_dw, max_dw, frac_changed, n_edge_types
      cross_features            (12)  — changed_node_types × {removed, fix_adj, other_changed}
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros(78, dtype=np.float32)

    # per-node attributes
    changed_type_hist = np.zeros(len(NODE_TYPES), dtype=np.float32)
    context_type_hist = np.zeros(len(NODE_TYPES), dtype=np.float32)
    diff_hist = np.zeros(len(DIFF_TYPES), dtype=np.float32)
    edge_hist = np.zeros(len(EDGE_TYPES), dtype=np.float32)

    flags_changed = np.zeros(6, dtype=np.float32)
    flags_all = np.zeros(6, dtype=np.float32)
    n_changed = 0
    dws = []

    # cross: for each of top-4 node types, count in {removed, fix_adjacent, other_changed}
    # collapsed to 4 types × 3 categories = 12
    TOP_TYPES = ["CALL", "IDENTIFIER", "LITERAL", "CONTROL_STRUCTURE"]
    top_idx = {t: i for i, t in enumerate(TOP_TYPES)}
    cross = np.zeros((4, 3), dtype=np.float32)  # [type][removed|fix_adj|other]

    for nd in nodes:
        attr = G.nodes[nd]
        ntype = attr.get("labelV", "UNKNOWN")
        diff = attr.get("diff", "")
        dw = float(attr.get("diff_weight", 0.2))
        code = attr.get("CODE", "") or ""

        tidx = NODE_TYPE_IDX.get(ntype, len(NODE_TYPES) - 1)
        didx = DIFF_IDX.get(diff, len(DIFF_TYPES) - 1)
        diff_hist[didx] += 1
        dws.append(dw)

        fl = np.array(
            [
                float(bool(re.search(r"[*&]|->", code))),
                float(bool(re.search(r"\b(malloc|alloc|new)\b", code))),
                float(bool(re.search(r"\b(free|delete|kfree)\b", code))),
                float(bool(re.search(r"\b(lock|mutex|spin)\b", code))),
                float(bool(re.search(r"\b(if|assert|check)\b", code))),
                float(len(code.split()) / 20.0),
            ],
            dtype=np.float32,
        )
        flags_all += fl

        if dw > _CHANGED_THRESH:
            changed_type_hist[tidx] += 1
            n_changed += 1
            flags_changed += fl
            # cross features
            if ntype in top_idx:
                cat = 0 if diff == "removed" else (1 if diff == "fix_adjacent" else 2)
                cross[top_idx[ntype], cat] += 1
        else:
            context_type_hist[tidx] += 1

    for _, _, data in G.edges(data=True):
        eidx = EDGE_TYPE_IDX.get(data.get("label", "AST"), 0)
        edge_hist[eidx] += 1

    # normalise histograms to fractions
    if n > 0:
        flags_all /= n
    if n_changed > 0:
        changed_type_hist /= n_changed
        flags_changed /= n_changed

    n_ctx = n - n_changed
    if n_ctx > 0:
        context_type_hist /= n_ctx

    # normalise edge hist
    e_total = edge_hist.sum()
    if e_total > 0:
        edge_hist /= e_total

    dws_arr = np.array(dws, dtype=np.float32)
    n_edges = G.number_of_edges()
    density = n_edges / (n * (n - 1) + 1e-8) if n > 1 else 0.0
    n_edge_types = len(set(d.get("label", "AST") for _, _, d in G.edges(data=True)))

    stats = np.array(
        [
            np.log1p(n),
            np.log1p(n_changed),
            np.log1p(n_edges),
            density,
            dws_arr.mean(),
            dws_arr.max(),
            n_changed / (n + 1e-8),
            n_edge_types / len(EDGE_TYPES),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            changed_type_hist,  # 15
            context_type_hist,  # 15
            diff_hist / (n + 1e-8),  # 8 (normalised)
            edge_hist,  # 8
            flags_changed,  # 6
            flags_all,  # 6
            stats,  # 8
            cross.ravel(),  # 12
        ]
    )  # total = 78


STRUCTURAL_HIST_DIM = 78


class RGCNEmbedder(BaseEmbedder):
    """
    Two-stream embedder: structural histogram + CodeBERT on changed code.

    Stream A — Structural histogram (78 dims):
        Bag-of-node-types, bag-of-edge-types, diff-type counts, semantic
        flags, graph stats, cross-features.  Fixed-size, no mean-pooling.

    Stream B — CodeBERT on changed code (768 dims):
        Concatenate CODE of changed nodes (diff_weight > 0.3) into one
        string, run CodeBERT once per graph → one [CLS] embedding.
        Avoids the per-node mean-pool collapse.

    Projection: PCA on the concatenation (846 dims → self.dim).
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else cfg.get("rgcn", {}).get("device", "cpu")
        )
        self._model_name = cfg.get("rgcn", {}).get(
            "codebert_model",
            "/home/z0050s2b/code/graph-rag/models/codebert-base/",
        )
        self._codebert_dim = 768
        self._in_dim = STRUCTURAL_HIST_DIM + self._codebert_dim  # 78 + 768 = 846
        self._cb_batch_size = cfg.get("rgcn", {}).get("cb_batch_size", 64)

        # lazy-loaded CodeBERT
        self._cb_model = None
        self._cb_tokenizer = None
        self._cb_available = None  # None = not checked yet

        # PCA projection (fitted on first embed_many call)
        self._pca = None
        self._fitted = False

    def _load_codebert(self):
        if self._cb_available is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer

            print(f"  [rgcn] loading {self._model_name} on {self._device}...")
            if not Path(self._model_name).exists():
                raise ValueError(f"Model path {self._model_name} does not exist.")
            self._cb_tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                local_files_only=True,
            )
            self._cb_model = AutoModel.from_pretrained(
                self._model_name,
                local_files_only=True,
            )
            self._cb_model.eval().to(self._device)
            self._cb_available = True
            print(f"  [rgcn] CodeBERT loaded on {self._device}")
        except Exception as e:
            print(
                f"  [rgcn] CodeBERT unavailable ({e}), "
                f"falling back to structural-only features"
            )
            self._cb_available = False
            self._in_dim = STRUCTURAL_HIST_DIM

    @property
    def name(self) -> str:
        return "rgcn"

    # ── batched CodeBERT: one [CLS] per graph ──────────────────────

    def _encode_graphs_codebert(
        self,
        code_strings: list[str],
    ) -> np.ndarray:
        """
        Run CodeBERT on one code string per graph (changed code concat).
        Returns (N, 768) array.  Empty strings get a zero vector.
        """
        n = len(code_strings)
        out = np.zeros((n, self._codebert_dim), dtype=np.float32)

        # separate non-empty for batched encoding
        nonempty_idx = [i for i, s in enumerate(code_strings) if s.strip()]
        if not nonempty_idx:
            return out

        nonempty_strs = [code_strings[i] for i in nonempty_idx]
        all_cls: list[torch.Tensor] = []
        bs = self._cb_batch_size
        for start in range(0, len(nonempty_strs), bs):
            batch = nonempty_strs[start : start + bs]
            enc = self._cb_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                cls = self._cb_model(**enc).last_hidden_state[:, 0, :]
            all_cls.append(cls.cpu())

        cls_mat = torch.cat(all_cls, dim=0).numpy()
        for pos, orig_idx in enumerate(nonempty_idx):
            out[orig_idx] = cls_mat[pos]

        return out

    # ── build graph-level vectors ──────────────────────────────────

    def _build_all_vectors(
        self,
        graphs: list[nx.MultiDiGraph],
    ) -> tuple[np.ndarray, list[int]]:
        """
        Stream A: structural histogram per graph.
        Stream B: CodeBERT [CLS] on changed-code string per graph.
        Returns (matrix, valid_indices).
        """
        struct_vecs = []
        code_strings = []
        valid_idx = []

        for gi, G in enumerate(graphs):
            if G.number_of_nodes() == 0:
                continue
            struct_vecs.append(_build_structural_histogram(G))
            code_strings.append(collect_changed_code(G))
            valid_idx.append(gi)

        if not struct_vecs:
            return np.zeros((0, self._in_dim), dtype=np.float32), []

        struct_mat = np.stack(struct_vecs)  # (N, 78)

        if self._cb_available:
            cb_mat = self._encode_graphs_codebert(code_strings)  # (N, 768)
            raw = np.concatenate([struct_mat, cb_mat], axis=1)  # (N, 846)
        else:
            raw = struct_mat

        return raw, valid_idx

    # ── public API ─────────────────────────────────────────────────

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if self.projection != "none" and not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        self._load_codebert()
        if G.number_of_nodes() == 0:
            return np.zeros(self.dim, dtype=np.float32)
        raw, valid = self._build_all_vectors([G])
        if raw.shape[0] == 0:
            return np.zeros(self.dim, dtype=np.float32)

        if self.projection == "none":
            return self._norm_vec(raw[0])

        projected = self._pca.transform(raw)[0].astype(np.float32)
        if projected.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[: projected.shape[0]] = projected
            projected = padded
        return self._norm_vec(projected)

    def embed_many(self, graphs: list) -> np.ndarray:
        self._load_codebert()

        raw, valid_idx = self._build_all_vectors(graphs)

        if raw.shape[0] == 0:
            return np.zeros((len(graphs), self.dim), dtype=np.float32)

        if self.projection == "none":
            self.dim = raw.shape[1]
            out = np.zeros((len(graphs), self.dim), dtype=np.float32)
            projected = self._norm_mat(raw)
            for pos, orig_idx in enumerate(valid_idx):
                out[orig_idx] = projected[pos]
            print(f"    [rgcn] no projection — dim={self.dim}")
            return out

        from sklearn.decomposition import PCA

        out = np.zeros((len(graphs), self.dim), dtype=np.float32)

        # fit PCA on the first call
        if not self._fitted:
            n_comp = min(self.dim, raw.shape[0] - 1, raw.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(raw)
            self._fitted = True
            explained = self._pca.explained_variance_ratio_.sum()
            print(
                f"    [rgcn] PCA fitted — {n_comp} components, "
                f"explained variance: {explained:.2%}"
            )

        projected = self._pca.transform(raw).astype(np.float32)
        if projected.shape[1] < self.dim:
            padded = np.zeros((projected.shape[0], self.dim), dtype=np.float32)
            padded[:, : projected.shape[1]] = projected
            projected = padded
        projected = self._norm_mat(projected)

        for pos, orig_idx in enumerate(valid_idx):
            out[orig_idx] = projected[pos]
        return out
