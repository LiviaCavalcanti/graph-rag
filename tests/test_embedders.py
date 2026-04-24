"""
Extensible behaviour tests for embedders.

Design principles
─────────────────
• **Parametrised over embedder classes** – add a new embedder to
  ``STANDALONE_EMBEDDERS`` or ``PCA_EMBEDDERS`` and existing tests
  automatically cover it.
• **Shared graph fixtures** – realistic CPG‑like ``nx.MultiDiGraph``
  objects built from constants used by the real embedders.
• **Marker‑based categories** – ``@standalone`` for embedders that
  need no fitting, ``@pca_dependent`` for those that require
  ``embed_many`` first.
• **Each test checks exactly one behavioural property** so failures
  are easy to diagnose.

To add a NEW embedder test:
  1. If the embedder is standalone, add its class to
     ``STANDALONE_EMBEDDERS``.
  2. If it requires embed_many fitting, add it to ``PCA_EMBEDDERS``.
  3. If it needs CodeBERT, add it to ``CODEBERT_EMBEDDERS`` so it is
     skipped when the model is unavailable.
  4. Add property‑specific tests in their own ``class Test…`` group.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from src.embeddings.base import BaseEmbedder
from src.embeddings.netlsd import NetLSDEmbedder
from src.embeddings.wl import WLEmbedder, NODE_TYPES as WL_NODE_TYPES
from src.embeddings.gin import GINEmbedder
from src.embeddings.combined import CombinedEmbedder
from src.embeddings.vuln_pattern import VulnPatternEmbedder, CodeBERTPatternEmbedder
from src.embeddings.codebert_seq import CodeBERTSeqEmbedder
from src.embeddings.rgcn import RGCNEmbedder
from src.embeddings.codexglue_baseline import CodeXGLUEBaselineEmbedder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIM = 32  # small dim for fast tests

BASE_CFG: dict = {
    "dim": DIM,
    "wl":  {"num_iterations": 4, "hidden_dim": 64},
    "gin": {"hidden_dim": 64, "num_layers": 3},
    "rgcn": {
        "device": "cpu",
        "codebert_model": "/home/z0050s2b/code/graph-rag/models/codebert-base/",
        "cb_batch_size": 4,
    },
}

# ---------------------------------------------------------------------------
# Embedder registries – extend these to cover new embedders automatically
# ---------------------------------------------------------------------------

STANDALONE_EMBEDDERS: list[type[BaseEmbedder]] = [
    NetLSDEmbedder,
    WLEmbedder,
    GINEmbedder,
]

PCA_EMBEDDERS: list[type[BaseEmbedder]] = [
    CombinedEmbedder,
    VulnPatternEmbedder,
]

# These need a CodeBERT model on disk; skipped if unavailable.
CODEBERT_EMBEDDERS: list[type[BaseEmbedder]] = [
    CodeBERTSeqEmbedder,
    CodeBERTPatternEmbedder,
    RGCNEmbedder,
    CodeXGLUEBaselineEmbedder,
]

ALL_EMBEDDERS = STANDALONE_EMBEDDERS + PCA_EMBEDDERS + CODEBERT_EMBEDDERS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EDGE_TYPES = ["AST", "CFG", "CDG", "REACHING_DEF", "REF", "ARGUMENT",
               "RECEIVER", "CALL"]


def _codebert_available() -> bool:
    """Return True if the local CodeBERT model can be loaded."""
    try:
        from pathlib import Path
        return Path(BASE_CFG["rgcn"]["codebert_model"]).exists()
    except Exception:
        return False


CODEBERT_AVAILABLE = _codebert_available()

_standalone_ids = [c.__name__ for c in STANDALONE_EMBEDDERS]
_pca_ids = [c.__name__ for c in PCA_EMBEDDERS]
_codebert_ids = [c.__name__ for c in CODEBERT_EMBEDDERS]
_all_ids = [c.__name__ for c in ALL_EMBEDDERS]


def _skip_if_needs_codebert(cls: type) -> None:
    if cls in CODEBERT_EMBEDDERS and not CODEBERT_AVAILABLE:
        pytest.skip("CodeBERT model not available")


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

def _make_empty_graph() -> nx.MultiDiGraph:
    return nx.MultiDiGraph()


def _make_tiny_graph(n_nodes: int = 2) -> nx.MultiDiGraph:
    """Graph with 1-2 nodes, below the minimum for most embedders."""
    G = nx.MultiDiGraph()
    for i in range(n_nodes):
        G.add_node(i, labelV="IDENTIFIER", CODE=f"x{i}",
                   diff="added", diff_weight=0.8, LINE_NUMBER=i + 1)
    if n_nodes >= 2:
        G.add_edge(0, 1, labelE="AST")
    return G


def _make_small_graph(seed: int = 0) -> nx.MultiDiGraph:
    """
    A realistic 10-node CPG-like graph with mixed node types,
    edge types, CODE attributes, and diff labels.
    """
    rng = np.random.RandomState(seed)
    G = nx.MultiDiGraph()

    node_types = ["METHOD", "BLOCK", "CALL", "IDENTIFIER", "IDENTIFIER",
                  "LITERAL", "RETURN", "CONTROL_STRUCTURE", "LOCAL", "CALL"]
    codes = [
        "void vuln_func(int *p)",
        "{ ... }",
        "free(p)",
        "p",
        "q",
        "42",
        "return result",
        "if (p != NULL)",
        "int result = 0",
        "malloc(sizeof(int))",
    ]
    diffs = ["unchanged", "unchanged", "removed", "added", "added",
             "unchanged", "mutated", "added", "unchanged", "added"]
    diff_weights = [0.1, 0.1, 0.9, 0.8, 0.7,
                    0.1, 0.6, 0.8, 0.1, 0.9]

    for i in range(10):
        G.add_node(i, labelV=node_types[i], CODE=codes[i],
                   diff=diffs[i], diff_weight=diff_weights[i],
                   LINE_NUMBER=i + 1)

    # Structural edges (AST tree)
    ast_edges = [(0, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                 (1, 6), (1, 7), (7, 8), (7, 9)]
    for u, v in ast_edges:
        G.add_edge(u, v, labelE="AST")

    # Control flow
    cfg_edges = [(2, 7), (7, 6), (9, 3)]
    for u, v in cfg_edges:
        G.add_edge(u, v, labelE="CFG")

    # Data flow
    G.add_edge(2, 3, labelE="REACHING_DEF")  # free→use (UAF pattern)
    G.add_edge(9, 4, labelE="REACHING_DEF")  # alloc→use
    G.add_edge(8, 6, labelE="REACHING_DEF")

    # Control dependence
    G.add_edge(7, 2, labelE="CDG")
    G.add_edge(7, 9, labelE="CDG")

    return G


def _make_different_graph(seed: int = 99) -> nx.MultiDiGraph:
    """A structurally different graph for discrimination tests."""
    G = nx.MultiDiGraph()
    node_types = ["METHOD", "BLOCK", "LOCAL", "LOCAL", "RETURN",
                  "LITERAL", "IDENTIFIER"]
    codes = [
        "int safe_func()",
        "{ ... }",
        "int a = 0",
        "int b = 1",
        "return a + b",
        "0",
        "a",
    ]
    for i in range(7):
        G.add_node(i, labelV=node_types[i], CODE=codes[i],
                   diff="unchanged", diff_weight=0.1,
                   LINE_NUMBER=i + 1)

    for u, v in [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (4, 6)]:
        G.add_edge(u, v, labelE="AST")
    G.add_edge(2, 4, labelE="CFG")
    G.add_edge(3, 4, labelE="CFG")
    G.add_edge(5, 6, labelE="REACHING_DEF")
    return G


def _make_graph_with_diff_labels() -> nx.MultiDiGraph:
    """Graph where several nodes have high diff_weight."""
    return _make_small_graph(seed=0)  # already has diff labels


def _make_graph_without_diff_labels() -> nx.MultiDiGraph:
    """Same topology, all diff_weight = 0 (no changed nodes)."""
    G = _make_small_graph(seed=0)
    for n in G.nodes():
        G.nodes[n]["diff_weight"] = 0.0
        G.nodes[n]["diff"] = "unchanged"
    return G


# ---------------------------------------------------------------------------
# Shared pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> dict:
    return dict(BASE_CFG)


@pytest.fixture
def empty_graph() -> nx.MultiDiGraph:
    return _make_empty_graph()


@pytest.fixture
def tiny_graph() -> nx.MultiDiGraph:
    return _make_tiny_graph(n_nodes=2)


@pytest.fixture
def small_graph() -> nx.MultiDiGraph:
    return _make_small_graph()


@pytest.fixture
def different_graph() -> nx.MultiDiGraph:
    return _make_different_graph()


@pytest.fixture
def graph_batch() -> list[nx.MultiDiGraph]:
    """A batch large enough for PCA fitting (need > dim samples)."""
    graphs = []
    for seed in range(DIM + 10):
        G = _make_small_graph(seed=seed % 5)
        # add slight random perturbation so rows aren't identical
        if seed >= 5:
            extra = nx.MultiDiGraph(G)
            extra.add_node(100 + seed, labelV="IDENTIFIER",
                           CODE=f"var_{seed}", diff_weight=0.5,
                           diff="added", LINE_NUMBER=20 + seed)
            extra.add_edge(0, 100 + seed, labelE="AST")
            graphs.append(extra)
        else:
            graphs.append(G)
    return graphs


# ===================================================================
# Test classes — grouped by behavioural property
# ===================================================================

class TestOutputShape:
    """embed_one → (dim,),  embed_many → (N, dim)."""

    @pytest.mark.parametrize("cls", STANDALONE_EMBEDDERS,
                             ids=_standalone_ids)
    def test_embed_one_shape_standalone(self, cls, cfg, small_graph):
        emb = cls(cfg)
        vec = emb.embed_one(small_graph)
        assert vec.shape == (DIM,), f"Expected ({DIM},), got {vec.shape}"
        assert vec.dtype == np.float32

    @pytest.mark.parametrize("cls", STANDALONE_EMBEDDERS,
                             ids=_standalone_ids)
    def test_embed_many_shape_standalone(self, cls, cfg, graph_batch):
        emb = cls(cfg)
        mat = emb.embed_many(graph_batch)
        n = len(graph_batch)
        assert mat.shape == (n, DIM), f"Expected ({n},{DIM}), got {mat.shape}"
        assert mat.dtype == np.float32

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_embed_many_shape_pca(self, cls, cfg, graph_batch):
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        mat = emb.embed_many(graph_batch)
        n = len(graph_batch)
        assert mat.shape == (n, DIM), f"Expected ({n},{DIM}), got {mat.shape}"

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_embed_one_after_fit_shape(self, cls, cfg, graph_batch,
                                       small_graph):
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        emb.embed_many(graph_batch)
        vec = emb.embed_one(small_graph)
        assert vec.shape == (DIM,)
        assert vec.dtype == np.float32


class TestL2Normalisation:
    """Non-zero outputs should have unit L2 norm (tolerance 1e-3)."""

    @pytest.mark.parametrize("cls", STANDALONE_EMBEDDERS,
                             ids=_standalone_ids)
    def test_unit_norm_standalone(self, cls, cfg, small_graph):
        emb = cls(cfg)
        vec = emb.embed_one(small_graph)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:  # non-zero output
            assert abs(norm - 1.0) < 1e-3, f"norm={norm}"

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_unit_norm_pca(self, cls, cfg, graph_batch, small_graph):
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        emb.embed_many(graph_batch)
        vec = emb.embed_one(small_graph)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            assert abs(norm - 1.0) < 1e-3, f"norm={norm}"

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_batch_norms_pca(self, cls, cfg, graph_batch):
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        mat = emb.embed_many(graph_batch)
        norms = np.linalg.norm(mat, axis=1)
        nonzero = norms[norms > 1e-6]
        if len(nonzero) > 0:
            assert np.allclose(nonzero, 1.0, atol=1e-3), \
                f"norms range: [{nonzero.min():.4f}, {nonzero.max():.4f}]"


class TestDegenerateGraphs:
    """Empty / tiny graphs should produce zero vectors, never crash."""

    @pytest.mark.parametrize("cls", STANDALONE_EMBEDDERS,
                             ids=_standalone_ids)
    def test_empty_graph_standalone(self, cls, cfg, empty_graph):
        emb = cls(cfg)
        vec = emb.embed_one(empty_graph)
        assert vec.shape == (DIM,)
        assert np.allclose(vec, 0.0), "Empty graph should yield zero vector"

    @pytest.mark.parametrize("cls", STANDALONE_EMBEDDERS,
                             ids=_standalone_ids)
    def test_tiny_graph_standalone(self, cls, cfg, tiny_graph):
        emb = cls(cfg)
        vec = emb.embed_one(tiny_graph)
        assert vec.shape == (DIM,)
        # Should be zero for embedders that require >= 3 nodes,
        # or a valid embedding otherwise — never NaN/Inf.
        assert np.all(np.isfinite(vec))

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_empty_in_batch(self, cls, cfg, graph_batch):
        """An empty graph inside a batch should get a zero row."""
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        batch = graph_batch + [_make_empty_graph()]
        mat = emb.embed_many(batch)
        assert mat.shape[0] == len(batch)
        # last row (empty graph) might be zero or small — must be finite
        assert np.all(np.isfinite(mat[-1]))


class TestPCALifecycle:
    """PCA-dependent embedders: embed_one fails before fit, works after."""

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_embed_one_before_fit_raises(self, cls, cfg, small_graph):
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        with pytest.raises(RuntimeError, match="embed_many|PCA"):
            emb.embed_one(small_graph)

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_embed_one_after_fit_works(self, cls, cfg, graph_batch,
                                       small_graph):
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        emb.embed_many(graph_batch)
        vec = emb.embed_one(small_graph)
        assert vec.shape == (DIM,)
        assert np.all(np.isfinite(vec))

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_pca_reset_and_refit(self, cls, cfg, graph_batch):
        """After resetting _pca/_fitted, embed_many re-fits fresh PCA."""
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        mat1 = emb.embed_many(graph_batch)
        assert emb._fitted

        # reset
        emb._pca = None
        emb._fitted = False
        assert not emb._fitted

        mat2 = emb.embed_many(graph_batch)
        assert emb._fitted
        # Same data → same PCA → same output
        assert np.allclose(mat1, mat2, atol=1e-5)


class TestDeterminism:
    """Same graph → same embedding (given same weights)."""

    @pytest.mark.parametrize("cls", STANDALONE_EMBEDDERS,
                             ids=_standalone_ids)
    def test_deterministic_standalone(self, cls, cfg, small_graph):
        emb = cls(cfg)
        v1 = emb.embed_one(small_graph)
        v2 = emb.embed_one(small_graph)
        np.testing.assert_array_equal(v1, v2)

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_deterministic_pca(self, cls, cfg, graph_batch, small_graph):
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        emb.embed_many(graph_batch)
        v1 = emb.embed_one(small_graph)
        v2 = emb.embed_one(small_graph)
        np.testing.assert_array_equal(v1, v2)


class TestStructuralSensitivity:
    """Different graph structures should (usually) produce different embeddings."""

    @pytest.mark.parametrize("cls", STANDALONE_EMBEDDERS,
                             ids=_standalone_ids)
    def test_different_graphs_standalone(self, cls, cfg, small_graph,
                                         different_graph):
        emb = cls(cfg)
        v1 = emb.embed_one(small_graph)
        v2 = emb.embed_one(different_graph)
        # At least one dimension should differ
        assert not np.allclose(v1, v2, atol=1e-4), \
            "Structurally different graphs produced identical embeddings"

    @pytest.mark.parametrize("cls", PCA_EMBEDDERS, ids=_pca_ids)
    def test_different_graphs_pca(self, cls, cfg, graph_batch,
                                   small_graph, different_graph):
        _skip_if_needs_codebert(cls)
        emb = cls(cfg)
        emb.embed_many(graph_batch)
        v1 = emb.embed_one(small_graph)
        v2 = emb.embed_one(different_graph)
        assert not np.allclose(v1, v2, atol=1e-4), \
            "Structurally different graphs produced identical embeddings"


class TestEmbedManyConsistency:
    """embed_many should be consistent with individual embed_one calls
    (for standalone embedders that don't need fitting)."""

    @pytest.mark.parametrize("cls", STANDALONE_EMBEDDERS,
                             ids=_standalone_ids)
    def test_many_matches_one(self, cls, cfg):
        graphs = [_make_small_graph(seed=i % 5) for i in range(5)]
        emb = cls(cfg)
        mat = emb.embed_many(graphs)
        for i, G in enumerate(graphs):
            vec = emb.embed_one(G)
            np.testing.assert_allclose(mat[i], vec, atol=1e-5,
                err_msg=f"embed_many row {i} != embed_one")


class TestNameProperty:
    """Every embedder must expose a non-empty string name."""

    @pytest.mark.parametrize("cls", ALL_EMBEDDERS, ids=_all_ids)
    def test_name_is_string(self, cls, cfg):
        emb = cls(cfg)
        assert isinstance(emb.name, str)
        assert len(emb.name) > 0


class TestBaseEmbedderContract:
    """embed_many default (from BaseEmbedder) catches exceptions → zero row."""

    def test_exception_in_embed_one_yields_zero(self):
        class BrokenEmbedder(BaseEmbedder):
            @property
            def name(self):
                return "broken"

            def embed_one(self, G):
                raise ValueError("intentional")

        emb = BrokenEmbedder({"dim": DIM})
        mat = emb.embed_many([_make_small_graph()])
        assert mat.shape == (1, DIM)
        assert np.allclose(mat[0], 0.0)


# ===================================================================
# VulnPattern-specific tests
# ===================================================================

class TestVulnPatternFeatures:
    """Test the raw 34-d feature extraction independently."""

    def test_raw_dim(self):
        from src.embeddings.vuln_pattern import build_vuln_pattern_features
        G = _make_small_graph()
        raw = build_vuln_pattern_features(G)
        assert raw.shape == (34,)
        assert raw.dtype == np.float32

    def test_empty_graph_raw(self):
        from src.embeddings.vuln_pattern import build_vuln_pattern_features
        raw = build_vuln_pattern_features(_make_empty_graph())
        assert raw.shape == (34,)
        assert np.allclose(raw, 0.0)

    def test_diff_awareness(self):
        """Diff labels should affect pattern features."""
        from src.embeddings.vuln_pattern import build_vuln_pattern_features
        with_diff = build_vuln_pattern_features(_make_graph_with_diff_labels())
        without_diff = build_vuln_pattern_features(_make_graph_without_diff_labels())
        assert not np.allclose(with_diff, without_diff, atol=1e-5), \
            "Diff labels had no effect on vuln pattern features"

    def test_uaf_pattern_detected(self):
        """The small_graph has free→use via REACHING_DEF (UAF signal)."""
        from src.embeddings.vuln_pattern import build_vuln_pattern_features
        raw = build_vuln_pattern_features(_make_small_graph())
        # flow_patterns[0] = a1 = UAF signal
        assert raw[0] > 0.0, "UAF pattern should be detected"

    def test_build_raw_many(self):
        graphs = [_make_small_graph(i) for i in range(5)]
        mat = VulnPatternEmbedder.build_raw_many(graphs)
        assert mat.shape == (5, 34)


# ===================================================================
# WL / GIN specific tests
# ===================================================================

class TestGraphConversion:
    """nx_to_pyg should produce valid PyG Data objects."""

    def test_nx_to_pyg_basic(self):
        from src.embeddings.wl import nx_to_pyg
        G = _make_small_graph()
        data = nx_to_pyg(G)
        assert data is not None
        assert data.x.shape[0] == G.number_of_nodes()
        # Colours should be in [0, len(NODE_TYPES)-1]
        assert data.x.min() >= 0
        assert data.x.max() < len(WL_NODE_TYPES)

    def test_nx_to_pyg_empty(self):
        from src.embeddings.wl import nx_to_pyg
        data = nx_to_pyg(_make_empty_graph())
        assert data is None

    def test_unknown_node_type_mapped(self):
        """A node type not in NODE_TYPES should map to UNKNOWN index."""
        from src.embeddings.wl import nx_to_pyg, NODE_TYPE_IDX
        G = nx.MultiDiGraph()
        G.add_node(0, labelV="TOTALLY_FAKE_TYPE")
        G.add_node(1, labelV="METHOD")
        G.add_edge(0, 1, labelE="AST")
        data = nx_to_pyg(G)
        assert data is not None
        # Node 0 should get the UNKNOWN index
        unknown_idx = NODE_TYPE_IDX["UNKNOWN"]
        assert data.x[0].item() == unknown_idx


# ===================================================================
# Combined embedder specific tests
# ===================================================================

class TestCombinedEmbedder:
    """CombinedEmbedder should concatenate NetLSD+WL+GIN then PCA."""

    def test_raw_concatenation(self, cfg, small_graph):
        emb = CombinedEmbedder(cfg)
        raw = emb._raw_one(small_graph)
        # Raw = NetLSD(dim) + WL(dim) + GIN(dim) = 3 * dim
        assert raw.shape == (3 * DIM,)

    def test_sub_embedders_consistent(self, cfg, small_graph):
        """Raw vector should equal concat of individual sub-embedder outputs."""
        emb = CombinedEmbedder(cfg)
        raw = emb._raw_one(small_graph)

        a = emb._netlsd.embed_one(small_graph)
        b = emb._wl.embed_one(small_graph)
        c = emb._gin.embed_one(small_graph)

        expected = np.concatenate([a, b, c])
        np.testing.assert_allclose(raw, expected, atol=1e-6)


# ===================================================================
# CodeBERT-dependent tests (skipped if model unavailable)
# ===================================================================

class TestCodeBERTEmbedders:
    """Tests that require the CodeBERT model."""

    @pytest.fixture(autouse=True)
    def _skip_no_codebert(self):
        if not CODEBERT_AVAILABLE:
            pytest.skip("CodeBERT model not available")

    def test_codebert_seq_output_shape(self, cfg, graph_batch,
                                       small_graph):
        emb = CodeBERTSeqEmbedder(cfg)
        mat = emb.embed_many(graph_batch)
        assert mat.shape == (len(graph_batch), DIM)
        vec = emb.embed_one(small_graph)
        assert vec.shape == (DIM,)

    def test_codebert_seq_before_fit_raises(self, cfg, small_graph):
        emb = CodeBERTSeqEmbedder(cfg)
        with pytest.raises(RuntimeError, match="embed_many|PCA"):
            emb.embed_one(small_graph)

    def test_codebert_pattern_output_shape(self, cfg, graph_batch,
                                           small_graph):
        emb = CodeBERTPatternEmbedder(cfg)
        mat = emb.embed_many(graph_batch)
        assert mat.shape == (len(graph_batch), DIM)
        vec = emb.embed_one(small_graph)
        assert vec.shape == (DIM,)

    def test_rgcn_output_shape(self, cfg, graph_batch, small_graph):
        emb = RGCNEmbedder(cfg)
        mat = emb.embed_many(graph_batch)
        assert mat.shape == (len(graph_batch), DIM)

    def test_codexglue_output_shape(self, cfg, graph_batch, small_graph):
        emb = CodeXGLUEBaselineEmbedder(cfg)
        mat = emb.embed_many(graph_batch)
        assert mat.shape == (len(graph_batch), DIM)

    def test_collect_changed_code(self, small_graph):
        from src.embeddings.codebert_seq import collect_changed_code
        code = collect_changed_code(small_graph)
        assert isinstance(code, str)
        assert len(code) > 0  # small_graph has nodes with CODE

    def test_collect_changed_code_empty(self):
        from src.embeddings.codebert_seq import collect_changed_code
        code = collect_changed_code(_make_empty_graph())
        assert code == ""

    def test_collect_changed_code_no_diff(self):
        """When no nodes pass threshold, falls back to all CODE nodes."""
        from src.embeddings.codebert_seq import collect_changed_code
        G = _make_graph_without_diff_labels()
        code = collect_changed_code(G)
        # Should fall back to all nodes — still non-empty
        assert len(code) > 0


# ===================================================================
# Extensibility: collect_changed_code standalone tests
# ===================================================================

class TestCollectChangedCode:
    """Unit tests for the shared collect_changed_code utility."""

    def test_respects_threshold(self):
        from src.embeddings.codebert_seq import collect_changed_code
        G = nx.MultiDiGraph()
        G.add_node(0, CODE="below_thresh", diff_weight=0.1)
        G.add_node(1, CODE="above_thresh", diff_weight=0.9)
        code = collect_changed_code(G)
        assert "above_thresh" in code
        # Below-threshold code should NOT appear (when above-thresh exists)
        assert "below_thresh" not in code

    def test_fallback_when_nothing_passes(self):
        from src.embeddings.codebert_seq import collect_changed_code
        G = nx.MultiDiGraph()
        G.add_node(0, CODE="only_code", diff_weight=0.0)
        code = collect_changed_code(G)
        assert "only_code" in code

    def test_max_tokens_respected(self):
        from src.embeddings.codebert_seq import collect_changed_code
        G = nx.MultiDiGraph()
        # Create a node with many tokens
        long_code = " ".join([f"token{i}" for i in range(500)])
        G.add_node(0, CODE=long_code, diff_weight=0.9)
        code = collect_changed_code(G, max_tokens=10)
        assert len(code.split()) <= 10

    def test_ordering_by_importance_then_line(self):
        from src.embeddings.codebert_seq import collect_changed_code
        G = nx.MultiDiGraph()
        G.add_node(0, CODE="low_weight", diff_weight=0.5,
                   LINE_NUMBER=1)
        G.add_node(1, CODE="high_weight", diff_weight=0.9,
                   LINE_NUMBER=10)
        code = collect_changed_code(G)
        # high_weight should come first (higher diff_weight)
        pos_high = code.find("high_weight")
        pos_low = code.find("low_weight")
        assert pos_high < pos_low
