"""
MotifEmbedder: combines three signals into one vector.

  1. Motif histogram    — structural vulnerability patterns (len=8)
  2. Semantic features  — CODE token features (ptr/alloc/free/lock)
  3. NetLSD descriptor  — spectral shape of PDG slice

Concatenated and projected to out_dim via a fixed random projection
(no training needed — random projections preserve distances by
Johnson-Lindenstrauss lemma).

This is the key insight: motif + semantic features are CWE-discriminative
even when pure topology is not.
"""

import numpy as np
import networkx as nx
from sklearn.random_projection import GaussianRandomProjection
from .base import BaseEmbedder


class MotifEmbedder(BaseEmbedder):
    """
    Motif histogram + semantic CODE features + NetLSD.
    No training. Random projection to fixed output dim.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._projector = None   # fit on first embed_many call

    @property
    def name(self) -> str:
        return "motif"

    def _raw_features(self, G: nx.MultiDiGraph) -> np.ndarray:
        import re
        if G.number_of_nodes() == 0:
            return np.zeros(64, dtype=np.float32)

        # ── motif histogram ────────────────────────────────────────
        changed = {n for n,a in G.nodes(data=True)
                   if a.get('diff') in ('removed','added','mutated','rewired')}
        from knowledge.slicing import extract_motif_histogram
        motifs = extract_motif_histogram(G, changed or set(G.nodes()))

        # ── semantic CODE features ─────────────────────────────────
        PTR   = re.compile(r'[*&]|->|\[')
        ALLOC = re.compile(r'\b(malloc|calloc|kmalloc|alloc|new)\b')
        FREE  = re.compile(r'\b(free|kfree|delete|release)\b')
        LOCK  = re.compile(r'\b(lock|mutex|spin|acquire)\b')
        CHECK = re.compile(r'\b(if|assert|check|verify|validate)\b')
        CAST  = re.compile(r'\(\s*\w[\w\s\*]+\)')

        n          = G.number_of_nodes()
        has_ptr    = has_alloc = has_free = has_lock = has_check = has_cast = 0
        diff_wts   = []
        tok_counts = []

        for _, attr in G.nodes(data=True):
            code = attr.get('CODE', '') or ''
            has_ptr   += bool(PTR.search(code))
            has_alloc += bool(ALLOC.search(code))
            has_free  += bool(FREE.search(code))
            has_lock  += bool(LOCK.search(code))
            has_check += bool(CHECK.search(code))
            has_cast  += bool(CAST.search(code))
            diff_wts.append(attr.get('diff_weight', 0.1))
            tok_counts.append(len(code.split()))

        # node type distribution
        from collections import Counter
        ntypes    = Counter(a.get('labelV','UNKNOWN')
                            for _,a in G.nodes(data=True))
        etypes    = Counter(d.get('label','')
                            for _,_,d in G.edges(data=True))
        NODE_T    = ['METHOD','CALL','IDENTIFIER','LITERAL','RETURN',
                     'BLOCK','CONTROL_STRUCTURE','LOCAL','PARAM']
        EDGE_T    = ['AST','CFG','CDG','REF','ARGUMENT','REACHING_DEF']
        node_hist = [ntypes.get(t,0)/n for t in NODE_T]
        e         = max(G.number_of_edges(), 1)
        edge_hist = [etypes.get(t,0)/e for t in EDGE_T]

        semantic = [
            has_ptr/n, has_alloc/n, has_free/n,
            has_lock/n, has_check/n, has_cast/n,
            float(np.mean(diff_wts)),
            float(np.std(diff_wts)),
            float(np.mean(tok_counts)),
            float(np.max(tok_counts)) / 50.0,
            G.number_of_nodes() / 100.0,
            G.number_of_edges() / 200.0,
        ]

        # ── NetLSD on the slice ────────────────────────────────────
        try:
            import netlsd
            H  = nx.Graph(G)
            H.remove_nodes_from(list(nx.isolates(H)))
            if H.number_of_nodes() >= 3:
                ts  = np.logspace(-2, 2, 16)
                lsd = netlsd.heat(H, timescales=ts).astype(np.float32)
                lsd = lsd / (np.linalg.norm(lsd) + 1e-8)
            else:
                lsd = np.zeros(16, dtype=np.float32)
        except Exception:
            lsd = np.zeros(16, dtype=np.float32)

        raw = np.concatenate([
            motifs,     # 8
            node_hist,  # 9
            edge_hist,  # 6
            semantic,   # 12
            lsd,        # 16
        ]).astype(np.float32)   # total: 51

        return raw

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if self._projector is None:
            raise RuntimeError("Call embed_many() first to fit projector")
        raw = self._raw_features(G).reshape(1, -1)
        out = self._projector.transform(raw)[0].astype(np.float32)
        return self._norm_vec(out)

    def embed_many(self, graphs: list) -> np.ndarray:
        raws = np.stack([self._raw_features(G) for G in graphs])

        if self._projector is None:
            self._projector = GaussianRandomProjection(
                n_components = self.dim,
                random_state = 42,
            )
            self._projector.fit(raws)

        projected = self._projector.transform(raws).astype(np.float32)
        return self._norm_mat(projected)