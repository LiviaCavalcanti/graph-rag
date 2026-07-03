"""
CodeBERT-only embedder — semantic baseline for ablation.

Embeds the concatenated changed-code from a vulnerability diff as a
single sequence through CodeBERT [CLS].  NO graph structural features.

Purpose: isolate the semantic contribution so that any improvement from
adding structural features (vuln_pattern, combined, etc.) proves the
graph's added value.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import torch

from .base import BaseEmbedder, resolve_codebert_path

# ── shared utility ─────────────────────────────────────────────────────

_CHANGED_THRESH = 0.3


def collect_changed_code(
    G: nx.MultiDiGraph,
    max_tokens: int = 400,
    dw_thresh: float = _CHANGED_THRESH,
) -> str:
    """
    Concatenate CODE from changed nodes (diff_weight > threshold)
    into one string for CodeBERT.  Ordered by importance then line.
    """
    changed = []
    for nd in G.nodes():
        attr = G.nodes[nd]
        dw = float(attr.get("diff_weight", 0.2))
        if dw <= dw_thresh:
            continue
        code = (attr.get("CODE", "") or "").strip()
        if not code:
            continue
        line = int(attr.get("LINE_NUMBER", 9999) or 9999)
        changed.append((dw, line, code))

    # Fallback: no nodes passed the diff threshold → use ALL code nodes
    if not changed:
        for nd in G.nodes():
            code = (G.nodes[nd].get("CODE", "") or "").strip()
            if not code:
                continue
            line = int(G.nodes[nd].get("LINE_NUMBER", 9999) or 9999)
            changed.append((0.0, line, code))

    if not changed:
        return ""

    changed.sort(key=lambda t: (-t[0], t[1]))
    parts, tok_count = [], 0
    for _, _, code in changed:
        words = code.split()
        if tok_count + len(words) > max_tokens:
            remaining = max_tokens - tok_count
            if remaining > 0:
                parts.append(" ".join(words[:remaining]))
            break
        parts.append(code)
        tok_count += len(words)

    return " ".join(parts)


def collect_flow_ordered_code(
    G: nx.MultiDiGraph,
    max_tokens: int = 400,
    dw_thresh: float = _CHANGED_THRESH,
) -> str:
    """
    Concatenate CODE from nodes ordered by graph data-flow traversal.

    Instead of sorting by line number, follows REACHING_DEF and CFG edges
    from high-importance seed nodes outward. This makes adjacent tokens in
    the sequence flow-connected in the graph, so CodeBERT's self-attention
    implicitly captures data-flow relationships.

    Traversal order:
      1. Start from highest diff_weight nodes (removed > fix_adjacent > context)
      2. Follow outgoing REACHING_DEF edges (data flows FROM here)
      3. Then outgoing CFG edges (control flows FROM here)
      4. BFS expansion until max_tokens reached
    """
    FLOW_EDGES = {"REACHING_DEF", "CFG", "CDG"}

    # Collect all nodes with code
    node_info = {}  # node_id → (code, diff_weight)
    for nd in G.nodes():
        attr = G.nodes[nd]
        code = (attr.get("CODE", "") or "").strip()
        if not code:
            continue
        dw = float(attr.get("diff_weight", 0.2))
        node_info[nd] = (code, dw)

    if not node_info:
        return ""

    # Seed nodes: those above threshold, sorted by importance
    seeds = [(nd, dw) for nd, (code, dw) in node_info.items() if dw > dw_thresh]
    if not seeds:
        # Fallback: use all nodes, pick highest-degree as seeds
        seeds = [(nd, dw) for nd, (code, dw) in node_info.items()]
    seeds.sort(key=lambda x: -x[1])

    # BFS from seeds following flow edges (REACHING_DEF preferred over CFG)
    visited_order = []
    visited = set()

    def _flow_successors(n):
        """Get successors ordered: REACHING_DEF first, then CFG, then CDG."""
        by_type = {"REACHING_DEF": [], "CFG": [], "CDG": []}
        for _, tgt, d in G.out_edges(n, data=True):
            et = d.get("labelE") or d.get("label", "")
            if et in by_type and tgt in node_info and tgt not in visited:
                by_type[et].append(tgt)
        return by_type["REACHING_DEF"] + by_type["CFG"] + by_type["CDG"]

    # Start BFS from each seed
    from collections import deque

    queue = deque()
    for nd, _ in seeds:
        if nd not in visited:
            visited.add(nd)
            visited_order.append(nd)
            queue.append(nd)

    while queue:
        n = queue.popleft()
        for succ in _flow_successors(n):
            if succ not in visited:
                visited.add(succ)
                visited_order.append(succ)
                queue.append(succ)

    # Also add unvisited nodes (disconnected from flow) at the end
    for nd in node_info:
        if nd not in visited:
            visited_order.append(nd)

    # Build text sequence in traversal order
    parts, tok_count = [], 0
    for nd in visited_order:
        code, _ = node_info[nd]
        words = code.split()
        if tok_count + len(words) > max_tokens:
            remaining = max_tokens - tok_count
            if remaining > 0:
                parts.append(" ".join(words[:remaining]))
            break
        parts.append(code)
        tok_count += len(words)

    return " ".join(parts)


# ── CodeBERT-only embedder ─────────────────────────────────────────────


class CodeBERTSeqEmbedder(BaseEmbedder):
    """
    Pure CodeBERT baseline: embed changed-code text only, no structure.
    Serves as the semantic-only control for ablation experiments.
    """

    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg, apply_norm=apply_norm)
        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else cfg.get("rgcn", {}).get("device", "cpu")
        )
        self._model_name = resolve_codebert_path(cfg)
        self._codebert_dim = cfg.get("codebert").get("dim", 128)
        self._cb_batch_size = cfg.get("rgcn", {}).get("cb_batch_size", 64)
        self._cb_model = None
        self._cb_tokenizer = None
        self._cb_available = None
        self._pca = None
        self._fitted = False
        print(
            f"Device: { self._device} \n Final dimension: {self._codebert_dim} \n Batch size: {self._cb_batch_size}"
        )

    def _load_codebert(self):
        if self._cb_available is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer

            print(
                f"  [codebert_seq] loading {self._model_name} " f"on {self._device}..."
            )
            if not Path(self._model_name).exists():
                raise ValueError(f"Model path {self._model_name} does not exist.")
            self._cb_tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, local_files_only=True
            )
            self._cb_model = AutoModel.from_pretrained(
                self._model_name, local_files_only=True
            )
            self._cb_model.eval().to(self._device)
            self._cb_available = True
            print(f"  [codebert_seq] CodeBERT loaded on {self._device}")
        except Exception as e:
            print(f"  [codebert_seq] CodeBERT unavailable ({e})")
            self._cb_available = False

    @property
    def name(self) -> str:
        return "codebert_seq"

    def encode_batch(self, code_strings: list[str]) -> np.ndarray:
        """
        Run CodeBERT on a list of code strings, return (N, 768).
        Empty strings get zero vectors.
        Public so the fusion embedder can call it.
        """
        n = len(code_strings)
        raw_dim = 768  # always return full CLS dimension; callers handle projection
        out = np.zeros((n, raw_dim), dtype=np.float32)
        if not self._cb_available:
            return out

        nonempty = [(i, s) for i, s in enumerate(code_strings) if s.strip()]
        if not nonempty:
            return out

        indices, strings = zip(*nonempty)
        all_cls: list[torch.Tensor] = []
        bs = self._cb_batch_size
        for start in range(0, len(strings), bs):
            batch = list(strings[start : start + bs])
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
        for pos, orig_idx in enumerate(indices):
            out[orig_idx] = cls_mat[pos]
        return out

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        raw_dim = 768
        should_compress = self.projection != "none" or (
            self.dim is not None and self.dim < raw_dim
        )

        if should_compress and not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")

        self._load_codebert()
        code = collect_changed_code(G)
        raw = self.encode_batch([code])

        if not should_compress:
            return self._norm_vec(raw[0]) if self.apply_norm else raw[0]

        # Apply learned PCA transformation
        target_dim = self.dim if self.dim is not None else raw_dim
        projected = self._pca.transform(raw)[0].astype(np.float32)
        if projected.shape[0] < target_dim:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[: projected.shape[0]] = projected
            projected = padded
        return self._norm_vec(projected) if self.apply_norm else projected

    def embed_many(self, graphs: list) -> np.ndarray:
        self._load_codebert()

        code_strings = [collect_changed_code(G) for G in graphs]
        raw = self.encode_batch(code_strings)  # Shape: (N, 768)

        valid = np.linalg.norm(raw, axis=1) > 1e-8

        # Determine if we need to compress: either projection is set OR self.dim < raw_dim
        raw_dim = raw.shape[1]
        should_compress = self.projection != "none" or (
            self.dim is not None and self.dim < raw_dim
        )

        if not should_compress:
            # No compression needed - keep full dimensionality
            self.dim = raw_dim
            out = self._norm_mat(raw)
            out[~valid] = 0.0
            print(f"    [codebert_seq] no compression — dim={self.dim}")
            return out

        # Apply PCA compression to target dimensionality
        from sklearn.decomposition import PCA

        # Determine target components
        target_dim = self.dim if self.dim is not None else raw_dim
        out = np.zeros((len(graphs), target_dim), dtype=np.float32)
        if not valid.any():
            return out

        if not self._fitted:
            valid_raw = raw[valid]
            n_comp = min(target_dim, valid_raw.shape[0] - 1, valid_raw.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(valid_raw)
            self._fitted = True
            expl = self._pca.explained_variance_ratio_.sum()
            print(
                f"    [codebert_seq] PCA compression fitted — {n_comp} comp, "
                f"explained variance: {expl:.2%} (768 → {target_dim})"
            )

        projected = self._pca.transform(raw).astype(np.float32)
        if projected.shape[1] < target_dim:
            padded = np.zeros((projected.shape[0], target_dim), dtype=np.float32)
            padded[:, : projected.shape[1]] = projected
            projected = padded
        projected = self._norm_mat(projected)
        projected[~valid] = 0.0
        return projected


class CodeBERTFlowEmbedder(CodeBERTSeqEmbedder):
    """
    CodeBERT with graph-flow-ordered code sequencing.

    Instead of sorting code by line number, follows REACHING_DEF → CFG → CDG
    edges from seed nodes. This makes adjacent tokens in the input sequence
    data-flow-connected, so CodeBERT's self-attention implicitly captures
    structural relationships between statements.

    Inspired by GraphFVD's use of graph-aware code representation.
    """

    @property
    def name(self) -> str:
        return "codebert_flow"

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        raw_dim = 768
        should_compress = self.projection != "none" or (
            self.dim is not None and self.dim < raw_dim
        )

        if should_compress and not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")

        self._load_codebert()
        code = collect_flow_ordered_code(G)
        raw = self.encode_batch([code])

        if not should_compress:
            return self._norm_vec(raw[0]) if self.apply_norm else raw[0]

        # Apply learned PCA transformation
        target_dim = self.dim if self.dim is not None else raw_dim
        projected = self._pca.transform(raw)[0].astype(np.float32)
        if projected.shape[0] < target_dim:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[: projected.shape[0]] = projected
            projected = padded
        return self._norm_vec(projected) if self.apply_norm else projected

    def embed_many(self, graphs: list) -> np.ndarray:
        self._load_codebert()

        code_strings = [collect_flow_ordered_code(G) for G in graphs]
        raw = self.encode_batch(code_strings)  # Shape: (N, 768)

        valid = np.linalg.norm(raw, axis=1) > 1e-8

        # Determine if we need to compress: either projection is set OR self.dim < raw_dim
        raw_dim = raw.shape[1]
        should_compress = self.projection != "none" or (
            self.dim is not None and self.dim < raw_dim
        )

        if not should_compress:
            # No compression needed - keep full dimensionality
            self.dim = raw_dim
            out = self._norm_mat(raw)
            out[~valid] = 0.0
            print(f"    [codebert_flow] no compression — dim={self.dim}")
            return out

        # Apply PCA compression to target dimensionality
        from sklearn.decomposition import PCA

        # Determine target components
        target_dim = self.dim if self.dim is not None else raw_dim
        out = np.zeros((len(graphs), target_dim), dtype=np.float32)
        if not valid.any():
            return out

        if not self._fitted:
            valid_raw = raw[valid]
            n_comp = min(target_dim, valid_raw.shape[0] - 1, valid_raw.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(valid_raw)
            self._fitted = True
            expl = self._pca.explained_variance_ratio_.sum()
            print(
                f"    [codebert_flow] PCA compression fitted — {n_comp} comp, "
                f"explained variance: {expl:.2%} (768 → {target_dim})"
            )

        projected = self._pca.transform(raw).astype(np.float32)
        if projected.shape[1] < target_dim:
            padded = np.zeros((projected.shape[0], target_dim), dtype=np.float32)
            padded[:, : projected.shape[1]] = projected
            projected = padded
        projected = self._norm_mat(projected)
        projected[~valid] = 0.0
        return projected
