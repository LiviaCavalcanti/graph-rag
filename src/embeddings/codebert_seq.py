"""
CodeBERT-only embedder — semantic baseline for ablation.

Embeds the concatenated changed-code from a vulnerability diff as a
single sequence through CodeBERT [CLS].  NO graph structural features.

Purpose: isolate the semantic contribution so that any improvement from
adding structural features (vuln_pattern, combined, etc.) proves the
graph's added value.
"""

from pathlib import Path
import numpy as np
import torch
import networkx as nx
from .base import BaseEmbedder


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
        dw   = float(attr.get('diff_weight', 0.2))
        if dw <= dw_thresh:
            continue
        code = (attr.get('CODE', '') or '').strip()
        if not code:
            continue
        line = int(attr.get('LINE_NUMBER', 9999) or 9999)
        changed.append((dw, line, code))

    # Fallback: no nodes passed the diff threshold → use ALL code nodes
    if not changed:
        for nd in G.nodes():
            code = (G.nodes[nd].get('CODE', '') or '').strip()
            if not code:
                continue
            line = int(G.nodes[nd].get('LINE_NUMBER', 9999) or 9999)
            changed.append((0.0, line, code))

    if not changed:
        return ''

    changed.sort(key=lambda t: (-t[0], t[1]))
    parts, tok_count = [], 0
    for _, _, code in changed:
        words = code.split()
        if tok_count + len(words) > max_tokens:
            remaining = max_tokens - tok_count
            if remaining > 0:
                parts.append(' '.join(words[:remaining]))
            break
        parts.append(code)
        tok_count += len(words)

    return ' '.join(parts)


# ── CodeBERT-only embedder ─────────────────────────────────────────────

class CodeBERTSeqEmbedder(BaseEmbedder):
    """
    Pure CodeBERT baseline: embed changed-code text only, no structure.
    Serves as the semantic-only control for ablation experiments.
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
        self._cb_batch_size = cfg.get('rgcn', {}).get('cb_batch_size', 64)
        self._cb_model     = None
        self._cb_tokenizer = None
        self._cb_available = None
        self._pca    = None
        self._fitted = False

    def _load_codebert(self):
        if self._cb_available is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModel
            print(f"  [codebert_seq] loading {self._model_name} "
                  f"on {self._device}...")
            if not Path(self._model_name).exists():
                raise ValueError(
                    f"Model path {self._model_name} does not exist.")
            self._cb_tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, local_files_only=True)
            self._cb_model = AutoModel.from_pretrained(
                self._model_name, local_files_only=True)
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
        out = np.zeros((n, self._codebert_dim), dtype=np.float32)
        if not self._cb_available:
            return out

        nonempty = [(i, s) for i, s in enumerate(code_strings) if s.strip()]
        if not nonempty:
            return out

        indices, strings = zip(*nonempty)
        all_cls: list[torch.Tensor] = []
        bs = self._cb_batch_size
        for start in range(0, len(strings), bs):
            batch = list(strings[start:start + bs])
            enc = self._cb_tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors='pt',
            ).to(self._device)
            with torch.no_grad():
                cls = self._cb_model(**enc).last_hidden_state[:, 0, :]
            all_cls.append(cls.cpu())

        cls_mat = torch.cat(all_cls, dim=0).numpy()
        for pos, orig_idx in enumerate(indices):
            out[orig_idx] = cls_mat[pos]
        return out

    def embed_one(self, G: nx.MultiDiGraph) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_many() first to fit PCA")
        self._load_codebert()
        code = collect_changed_code(G)
        raw = self.encode_batch([code])
        projected = self._pca.transform(raw)[0].astype(np.float32)
        if projected.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:projected.shape[0]] = projected
            projected = padded
        norm = np.linalg.norm(projected)
        return (projected / (norm + 1e-8)).astype(np.float32)

    def embed_many(self, graphs: list) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize
        self._load_codebert()

        code_strings = [collect_changed_code(G) for G in graphs]
        raw = self.encode_batch(code_strings)

        out = np.zeros((len(graphs), self.dim), dtype=np.float32)
        valid = np.linalg.norm(raw, axis=1) > 1e-8
        if not valid.any():
            return out

        if not self._fitted:
            valid_raw = raw[valid]
            n_comp = min(self.dim, valid_raw.shape[0] - 1,
                         valid_raw.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            self._pca.fit(valid_raw)
            self._fitted = True
            expl = self._pca.explained_variance_ratio_.sum()
            print(f"    [codebert_seq] PCA fitted — {n_comp} comp, "
                  f"explained variance: {expl:.2%}")

        projected = self._pca.transform(raw).astype(np.float32)
        if projected.shape[1] < self.dim:
            padded = np.zeros((projected.shape[0], self.dim),
                              dtype=np.float32)
            padded[:, :projected.shape[1]] = projected
            projected = padded
        projected = normalize(projected, norm='l2')
        projected[~valid] = 0.0
        return projected
