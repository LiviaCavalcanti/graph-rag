"""
Vulnerability-specific structural pattern embedder.

Extracts features that **require CPG edge connectivity** — they encode
inter-statement relationships (data flow, control dependence, control
flow) that code text alone cannot capture.

Feature groups (34 dims total):
  A. Vulnerability flow patterns  (8)  — REACHING_DEF / CDG reachability
  B. Diff edge composition        (8)  — edge type distribution of changes
  C. Changed↔context boundary     (6)  — flow direction across diff boundary
  D. Diff topology stats          (6)  — subgraph shape of the change
  E. Changed node roles           (6)  — node type distribution of changes

Also provides a fusion embedder (CodeBERTPatternEmbedder) that
concatenates these structural features with CodeBERT on changed code,
then PCA-projects — the key ablation variant.
"""

import re

import networkx as nx
import numpy as np

from .base import BaseEmbedder

# ── constants ──────────────────────────────────────────────────────────

ALL_EDGE_TYPES = [
    "AST",
    "CFG",
    "CDG",
    "REACHING_DEF",
    "REF",
    "ARGUMENT",
    "RECEIVER",
    "CALL",
]
_EDGE_IDX = {t: i for i, t in enumerate(ALL_EDGE_TYPES)}

FLOW_EDGE_TYPES = ("CFG", "CDG", "REACHING_DEF")

CHANGED_THRESH = 0.3

VULN_PATTERN_DIM = 34  # total raw feature dimension

# ── regex patterns for code classification ─────────────────────────────

RE_PTR_DEREF = re.compile(r"\*\w|->")
RE_ALLOC = re.compile(r"\b(malloc|calloc|kmalloc|kzalloc|alloc|realloc|krealloc|new)\b")
RE_FREE = re.compile(r"\b(free|kfree|vfree|kvfree|delete|release)\b")
RE_LOCK = re.compile(r"\b(lock|mutex_lock|spin_lock|down_read|down_write|rtnl_lock)\b")
RE_UNLOCK = re.compile(
    r"\b(unlock|mutex_unlock|spin_unlock|up_read|up_write|rtnl_unlock)\b"
)
RE_NULL_CHECK = re.compile(r"\b(NULL|null|nullptr)\b")
RE_ARITH = re.compile(r"[+\-*/]|<<|>>")
RE_BOUNDS = re.compile(
    r"\b(MAX|MIN|SIZE_MAX|UINT_MAX|INT_MAX|limit|bound|clamp)\b", re.I
)
RE_CAST = re.compile(
    r"\(\s*(?:unsigned|signed|int|long|short|char|void|size_t"
    r"|u?int\d+_t)\s*\*?\s*\)"
)
RE_CHECK = re.compile(r"\b(if|assert|BUG_ON|WARN_ON|check|verify|IS_ERR)\b")


# ── helpers ────────────────────────────────────────────────────────────


def _edge_type(data: dict) -> str:
    return data.get("labelE") or data.get("label", "")


def _code(G: nx.MultiDiGraph, n) -> str:
    return G.nodes[n].get("CODE", "") or ""


def _ntype(G: nx.MultiDiGraph, n) -> str:
    return G.nodes[n].get("labelV", "UNKNOWN")


def _is_changed(G: nx.MultiDiGraph, n) -> bool:
    return float(G.nodes[n].get("diff_weight", 0.2)) > CHANGED_THRESH


def _out_by_type(G, n, etype):
    """Set of successors via edges of given type."""
    return {v for _, v, d in G.out_edges(n, data=True) if _edge_type(d) == etype}


def _in_by_type(G, n, etype):
    """Set of predecessors via edges of given type."""
    return {u for u, _, d in G.in_edges(n, data=True) if _edge_type(d) == etype}


# ── main feature extraction ───────────────────────────────────────────


def build_vuln_pattern_features(G: nx.MultiDiGraph) -> np.ndarray:
    """
    34-dimensional feature vector encoding vulnerability-specific
    structural patterns.  Every feature depends on CPG edge connectivity.

    Returns np.ndarray of shape (34,).
    """
    nodes = list(G.nodes())
    n_total = len(nodes)
    if n_total == 0:
        return np.zeros(VULN_PATTERN_DIM, dtype=np.float32)

    changed = set()
    context = set()
    for n in nodes:
        (changed if _is_changed(G, n) else context).add(n)
    n_changed = len(changed)

    # Fallback: no diff labels → treat ALL nodes as changed
    if n_changed == 0:
        changed = set(nodes)
        context = set()
        n_changed = len(changed)

    # ── classify changed nodes ─────────────────────────────────────
    free_nodes = set()
    alloc_nodes = set()
    lock_nodes = set()
    unlock_nodes = set()
    deref_nodes = set()  # pointer dereferences
    arith_nodes = set()  # arithmetic operations
    cast_nodes = set()  # type casts
    check_nodes = set()  # if / assert / null-check
    changed_calls = set()
    changed_ids = set()

    for n in changed:
        nt = _ntype(G, n)
        code = _code(G, n)

        if nt == "CALL":
            changed_calls.add(n)
            if RE_FREE.search(code):
                free_nodes.add(n)
            if RE_ALLOC.search(code):
                alloc_nodes.add(n)
            if RE_LOCK.search(code):
                lock_nodes.add(n)
            if RE_UNLOCK.search(code):
                unlock_nodes.add(n)
        elif nt == "IDENTIFIER":
            changed_ids.add(n)
            if RE_PTR_DEREF.search(code):
                deref_nodes.add(n)
        elif nt == "CONTROL_STRUCTURE":
            if RE_CHECK.search(code) or RE_NULL_CHECK.search(code):
                check_nodes.add(n)

        if RE_ARITH.search(code):
            arith_nodes.add(n)
        if RE_CAST.search(code):
            cast_nodes.add(n)

    # ── Group A: vulnerability flow patterns (8) ───────────────────
    # Each requires graph edge connectivity (REACHING_DEF / CDG / CFG).

    # A1: UAF — free() has REACHING_DEF path to subsequent identifier use
    a1 = 0.0
    for fn in free_nodes:
        # 1-hop: direct REACHING_DEF from free to an identifier
        rd = _out_by_type(G, fn, "REACHING_DEF")
        if rd & changed_ids:
            a1 = 1.0
            break
        # 2-hop: free → X → identifier
        for mid in rd:
            if _out_by_type(G, mid, "REACHING_DEF") & changed_ids:
                a1 = 1.0
                break
        if a1:
            break

    # A2: NPD — ptr dereference without null-check in CDG predecessors
    a2 = 0.0
    if deref_nodes:
        unguarded = 0
        for dn in deref_nodes:
            cdg_preds = _in_by_type(G, dn, "CDG")
            guarded = any(n in check_nodes for n in cdg_preds)
            if not guarded:
                # 2-hop CDG check
                for p in cdg_preds:
                    if _in_by_type(G, p, "CDG") & check_nodes:
                        guarded = True
                        break
            if not guarded:
                unguarded += 1
        a2 = unguarded / len(deref_nodes)

    # A3: Unchecked return — CALL whose REACHING_DEF chain doesn't
    #     reach a CONTROL_STRUCTURE (i.e. return value is never checked)
    a3 = 0.0
    if changed_calls:
        unchecked = 0
        for cn in changed_calls:
            succs = _out_by_type(G, cn, "REACHING_DEF")
            has_check = any(_ntype(G, s) == "CONTROL_STRUCTURE" for s in succs)
            if not has_check:
                # 2-hop
                for s in succs:
                    s2 = _out_by_type(G, s, "REACHING_DEF")
                    if any(_ntype(G, x) == "CONTROL_STRUCTURE" for x in s2):
                        has_check = True
                        break
            if not has_check:
                unchecked += 1
        a3 = unchecked / len(changed_calls)

    # A4: Lock/unlock imbalance (race condition / deadlock signal)
    nl, nu = len(lock_nodes), len(unlock_nodes)
    a4 = abs(nl - nu) / (nl + nu + 1)

    # A5: Arithmetic without bounds guard in CDG neighbourhood
    a5 = 0.0
    if arith_nodes:
        unguarded = 0
        for an in arith_nodes:
            neigh = _in_by_type(G, an, "CDG") | _out_by_type(G, an, "CDG")
            has_bound = any(RE_BOUNDS.search(_code(G, nb)) for nb in neigh)
            if not has_bound:
                unguarded += 1
        a5 = unguarded / len(arith_nodes)

    # A6: Use without definition — IDENTIFIER with no incoming
    #     REACHING_DEF (uninitialised variable signal)
    a6 = 0.0
    if changed_ids:
        no_def = sum(
            1 for idn in changed_ids if not _in_by_type(G, idn, "REACHING_DEF")
        )
        a6 = no_def / len(changed_ids)

    # A7: Alloc/free imbalance (resource leak signal)
    na, nf = len(alloc_nodes), len(free_nodes)
    a7 = abs(na - nf) / (na + nf + 1)

    # A8: Type cast connected via REACHING_DEF to other changed nodes
    a8 = 0.0
    for cn in cast_nodes:
        rd_in = _in_by_type(G, cn, "REACHING_DEF")
        rd_out = _out_by_type(G, cn, "REACHING_DEF")
        if (rd_in | rd_out) & changed:
            a8 = 1.0
            break

    flow_patterns = np.array([a1, a2, a3, a4, a5, a6, a7, a8], dtype=np.float32)

    # ── Group B: diff edge composition (8) ─────────────────────────
    # Fraction of each edge type among edges touching changed nodes.
    edge_counts = np.zeros(len(ALL_EDGE_TYPES), dtype=np.float32)
    for u, v, d in G.edges(data=True):
        if u in changed or v in changed:
            idx = _EDGE_IDX.get(_edge_type(d))
            if idx is not None:
                edge_counts[idx] += 1
    edge_fracs = edge_counts / (edge_counts.sum() + 1e-8)

    # ── Group C: boundary flow direction (6) ───────────────────────
    # For CFG, CDG, REACHING_DEF: count changed→context vs context→changed.
    # This captures whether the change is at a flow source or sink.
    boundary = np.zeros(6, dtype=np.float32)
    for u, v, d in G.edges(data=True):
        et = _edge_type(d)
        u_ch, v_ch = u in changed, v in changed
        if u_ch == v_ch:
            continue  # both changed or both context — not a boundary
        if et == "CFG":
            boundary[0 if u_ch else 1] += 1
        elif et == "CDG":
            boundary[2 if u_ch else 3] += 1
        elif et == "REACHING_DEF":
            boundary[4 if u_ch else 5] += 1
    b_total = boundary.sum()
    boundary = boundary / (b_total + 1e-8)

    # ── Group D: diff topology stats (6) ───────────────────────────
    changed_sub = G.subgraph(changed)
    n_edges_ch = changed_sub.number_of_edges()

    d1 = n_changed / (n_total + 1e-8)  # changed frac
    d2 = (
        n_edges_ch / (n_changed * max(n_changed - 1, 1)) if n_changed > 1 else 0.0
    )  # density
    # connected components (undirected view)
    d3 = nx.number_connected_components(nx.Graph(changed_sub)) / (n_changed + 1e-8)
    degs = [changed_sub.degree(n) for n in changed]
    d4 = np.mean(degs) / max(n_changed, 1)  # mean degree
    d5 = max(degs) / max(n_changed, 1) if degs else 0.0  # max degree
    # internal vs crossing edges
    n_crossing = sum(
        1 for u, v, _ in G.edges(data=True) if (u in changed) != (v in changed)
    )
    d6 = n_edges_ch / (n_edges_ch + n_crossing + 1e-8)

    topology = np.array([d1, d2, d3, d4, d5, d6], dtype=np.float32)

    # ── Group E: changed node role distribution (6) ────────────────
    ROLE_TYPES = ["CALL", "CONTROL_STRUCTURE", "IDENTIFIER", "LITERAL", "BLOCK"]
    role_counts = np.zeros(6, dtype=np.float32)  # 5 named + OTHER
    for n in changed:
        nt = _ntype(G, n)
        if nt in ROLE_TYPES:
            role_counts[ROLE_TYPES.index(nt)] += 1
        else:
            role_counts[5] += 1
    roles = role_counts / (n_changed + 1e-8)

    return np.concatenate(
        [
            flow_patterns,  # 8
            edge_fracs,  # 8
            boundary,  # 6
            topology,  # 6
            roles,  # 6
        ]
    )  # total = 34


# ── VulnPattern-only embedder ─────────────────────────────────────────


class VulnPatternEmbedder(BaseEmbedder):
    """
    Structural-only embedder using vulnerability-specific graph patterns.
    No CodeBERT — proves whether graph structure alone is discriminative.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._pca = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "vuln_pattern"

    @staticmethod
    def build_raw_many(graphs: list[nx.MultiDiGraph]) -> np.ndarray:
        """Raw pattern features for a list of graphs (N, 34)."""
        return np.stack([build_vuln_pattern_features(G) for G in graphs]).astype(
            np.float32
        )

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        raw = build_vuln_pattern_features(G).reshape(1, -1)
        if self.projection == "none":
            return self._norm_vec(raw[0])
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        proj = self._pca.transform(raw)[0].astype(np.float32)
        if proj.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[: proj.shape[0]] = proj
            proj = padded
        return self._norm_vec(proj)

    def embed_many(self, graphs: list) -> np.ndarray:
        raw = self.build_raw_many(graphs)
        if raw.shape[0] == 0:
            return np.zeros((len(graphs), self.dim), dtype=np.float32)

        if self.projection == "none":
            self.dim = raw.shape[1]
            print(f"    [vuln_pattern] no projection — dim={self.dim}")
            return self._norm_mat(raw)

        from sklearn.decomposition import PCA

        out = np.zeros((len(graphs), self.dim), dtype=np.float32)

        if not self._fitted:
            n_comp = min(self.dim, raw.shape[0] - 1, raw.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(raw)
            self._fitted = True
            expl = self._pca.explained_variance_ratio_.sum()
            print(
                f"    [vuln_pattern] PCA fitted — {n_comp} comp, "
                f"explained variance: {expl:.2%}"
            )

        projected = self._pca.transform(raw).astype(np.float32)
        if projected.shape[1] < self.dim:
            padded = np.zeros((projected.shape[0], self.dim), dtype=np.float32)
            padded[:, : projected.shape[1]] = projected
            projected = padded
        return self._norm_mat(projected)


# ── CodeBERT + VulnPattern fusion ──────────────────────────────────────


class CodeBERTPatternEmbedder(BaseEmbedder):
    """
    Fusion: vulnerability graph patterns (34d) + CodeBERT changed-code
    (768d) → PCA → L2-normalised embedding.

    Key ablation variant.  If this beats codebert_seq, graph structure
    adds value.  If it beats vuln_pattern, CodeBERT adds value.
    """

    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg)
        # lazily import to avoid circular deps
        from .codebert_seq import CodeBERTSeqEmbedder, collect_changed_code

        self._cb_embedder = CodeBERTSeqEmbedder(cfg, apply_norm=apply_norm)
        self._collect = collect_changed_code
        self._pca = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "codebert_pattern"

    def _build_raw(
        self,
        graphs: list[nx.MultiDiGraph],
    ) -> tuple[np.ndarray, list[int]]:
        """Concatenated [pattern(34) || codebert(768)] for each graph."""
        # structural patterns (fast, no model)
        pattern_raw = VulnPatternEmbedder.build_raw_many(graphs)

        # CodeBERT on changed code (slow, needs model)
        self._cb_embedder._load_codebert()
        code_strings = [self._collect(G) for G in graphs]
        cb_raw = self._cb_embedder.encode_batch(code_strings)

        raw = np.concatenate([pattern_raw, cb_raw], axis=1)

        # valid = non-degenerate graphs
        valid = []
        for i in range(len(graphs)):
            if (
                np.linalg.norm(pattern_raw[i]) > 1e-8
                or np.linalg.norm(cb_raw[i]) > 1e-8
            ):
                valid.append(i)

        return raw, valid

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        raw, _ = self._build_raw([G])
        if self.projection == "none":
            return self._norm_vec(raw[0])
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        proj = self._pca.transform(raw)[0].astype(np.float32)
        if proj.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[: proj.shape[0]] = proj
            proj = padded
        return self._norm_vec(proj)

    def embed_many(self, graphs: list) -> np.ndarray:
        raw, valid_idx = self._build_raw(graphs)
        if not valid_idx:
            return np.zeros((len(graphs), self.dim), dtype=np.float32)

        if self.projection == "none":
            self.dim = raw.shape[1]
            out = np.zeros((len(graphs), self.dim), dtype=np.float32)
            projected = self._norm_mat(raw)
            for i in set(valid_idx):
                out[i] = projected[i]
            print(f"    [codebert_pattern] no projection — dim={self.dim}")
            return out

        from sklearn.decomposition import PCA

        out = np.zeros((len(graphs), self.dim), dtype=np.float32)
        valid_raw = raw[valid_idx]

        if not self._fitted:
            n_comp = min(self.dim, valid_raw.shape[0] - 1, valid_raw.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(valid_raw)
            self._fitted = True
            expl = self._pca.explained_variance_ratio_.sum()
            print(
                f"    [codebert_pattern] PCA fitted — {n_comp} comp, "
                f"explained variance: {expl:.2%}"
            )

        projected = self._pca.transform(raw).astype(np.float32)
        if projected.shape[1] < self.dim:
            padded = np.zeros((projected.shape[0], self.dim), dtype=np.float32)
            padded[:, : projected.shape[1]] = projected
            projected = padded
        projected = self._norm_mat(projected)

        for i in range(len(graphs)):
            if i in set(valid_idx):
                out[i] = projected[i]
        return out


class CodeBERTFlowPatternEmbedder(BaseEmbedder):
    """
    Fusion: vulnerability graph patterns (34d) + CodeBERT with flow-ordered
    code sequencing (768d) → PCA → L2-normalised embedding.

    Improvement over CodeBERTPatternEmbedder: instead of concatenating code
    by line number, orders code by data-flow traversal (REACHING_DEF → CFG)
    so CodeBERT's self-attention sees flow-connected statements as adjacent
    tokens. Inspired by GraphFVD's graph-aware code representation.
    """

    def __init__(self, cfg: dict, apply_norm: bool = True):
        super().__init__(cfg)
        from .codebert_seq import CodeBERTFlowEmbedder, collect_flow_ordered_code

        self._cb_embedder = CodeBERTFlowEmbedder(cfg, apply_norm=apply_norm)
        self._collect = collect_flow_ordered_code
        self._pca = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "codebert_flow_pattern"

    def _build_raw(
        self,
        graphs: list[nx.MultiDiGraph],
    ) -> tuple[np.ndarray, list[int]]:
        """Concatenated [pattern(34) || codebert_flow(768)] for each graph."""
        pattern_raw = VulnPatternEmbedder.build_raw_many(graphs)

        self._cb_embedder._load_codebert()
        code_strings = [self._collect(G) for G in graphs]
        cb_raw = self._cb_embedder.encode_batch(code_strings)

        raw = np.concatenate([pattern_raw, cb_raw], axis=1)

        valid = []
        for i in range(len(graphs)):
            if (
                np.linalg.norm(pattern_raw[i]) > 1e-8
                or np.linalg.norm(cb_raw[i]) > 1e-8
            ):
                valid.append(i)

        return raw, valid

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        raw, _ = self._build_raw([G])
        if self.projection == "none":
            return self._norm_vec(raw[0])
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        proj = self._pca.transform(raw)[0].astype(np.float32)
        if proj.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[: proj.shape[0]] = proj
            proj = padded
        return self._norm_vec(proj)

    def embed_many(self, graphs: list) -> np.ndarray:
        raw, valid_idx = self._build_raw(graphs)
        if not valid_idx:
            return np.zeros((len(graphs), self.dim), dtype=np.float32)

        if self.projection == "none":
            self.dim = raw.shape[1]
            out = np.zeros((len(graphs), self.dim), dtype=np.float32)
            projected = self._norm_mat(raw)
            for i in set(valid_idx):
                out[i] = projected[i]
            print(f"    [codebert_flow_pattern] no projection — dim={self.dim}")
            return out

        from sklearn.decomposition import PCA

        out = np.zeros((len(graphs), self.dim), dtype=np.float32)
        valid_raw = raw[valid_idx]

        if not self._fitted:
            n_comp = min(self.dim, valid_raw.shape[0] - 1, valid_raw.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(valid_raw)
            self._fitted = True
            expl = self._pca.explained_variance_ratio_.sum()
            print(
                f"    [codebert_flow_pattern] PCA fitted — {n_comp} comp, "
                f"explained variance: {expl:.2%}"
            )

        projected = self._pca.transform(raw).astype(np.float32)
        if projected.shape[1] < self.dim:
            padded = np.zeros((projected.shape[0], self.dim), dtype=np.float32)
            padded[:, : projected.shape[1]] = projected
            projected = padded
        projected = self._norm_mat(projected)

        for i in range(len(graphs)):
            if i in set(valid_idx):
                out[i] = projected[i]
        return out
