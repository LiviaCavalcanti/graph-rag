import json
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from rag.index import FAISSIndex
from rag.retriever import Retriever
from data.base import FunctionPair
import networkx as nx


# ── helpers ────────────────────────────────────────────────────────────

def make_pair(cve_id='CVE-2025-0001', func_name='foo', cwe_id='CWE-119', project='test'):
    G = nx.MultiDiGraph()
    G.add_node('n1', labelV='METHOD')
    G.add_node('n2', labelV='CALL')
    G.add_edge('n1', 'n2', label='CFG')
    return FunctionPair(
        cve_id    = cve_id,
        cwe_id    = cwe_id,
        func_name = func_name,
        project   = project,
        G_before  = G,
        G_after   = G,
        G_vuln    = G,
        meta      = {'dataset': 'test', 'variant': 'original'},
    )


def make_index(tmp_path, dim=128):
    return FAISSIndex(
        dim           = dim,
        index_path    = str(tmp_path / 'faiss.index'),
        metadata_path = str(tmp_path / 'metadata.json'),
    )


def random_vec(dim=128, seed=None):
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── FAISSIndex.add ─────────────────────────────────────────────────────

def test_add_increments_count(tmp_path):
    index = make_index(tmp_path)
    index.add(make_pair(), random_vec(), 'netlsd')
    assert index.index.ntotal == 1


def test_add_multiple(tmp_path):
    index = make_index(tmp_path)
    for i in range(5):
        index.add(make_pair(cve_id=f'CVE-2025-{i:04d}'), random_vec(seed=i), 'netlsd')
    assert index.index.ntotal == 5
    assert len(index.metadata) == 5


def test_add_stores_metadata_fields(tmp_path):
    index = make_index(tmp_path)
    pair  = make_pair(cve_id='CVE-2025-0001', func_name='vuln_fn', cwe_id='CWE-119')
    index.add(pair, random_vec(), 'netlsd')

    m = index.metadata[0]
    assert m['cve_id']    == 'CVE-2025-0001'
    assert m['func_name'] == 'vuln_fn'
    assert m['cwe_id']    == 'CWE-119'
    assert m['variant']   == 'netlsd'


def test_add_stores_node_count(tmp_path):
    index = make_index(tmp_path)
    index.add(make_pair(), random_vec(), 'netlsd')
    assert index.metadata[0]['n_nodes'] == 2  # G_vuln has 2 nodes


def test_add_stores_extra_meta_fields(tmp_path):
    index = make_index(tmp_path)
    pair  = make_pair()
    pair.meta['dataset'] = 'autopatch'
    index.add(pair, random_vec(), 'netlsd')
    assert index.metadata[0]['dataset'] == 'autopatch'


# ── FAISSIndex.save / load ─────────────────────────────────────────────

def test_save_creates_files(tmp_path):
    index = make_index(tmp_path)
    index.add(make_pair(), random_vec(), 'netlsd')
    index.save()
    assert Path(tmp_path / 'faiss.index').exists()
    assert Path(tmp_path / 'metadata.json').exists()


def test_save_metadata_is_valid_json(tmp_path):
    index = make_index(tmp_path)
    index.add(make_pair(cve_id='CVE-2025-0001'), random_vec(), 'netlsd')
    index.save()
    data = json.loads(Path(tmp_path / 'metadata.json').read_text())
    assert isinstance(data, list)
    assert data[0]['cve_id'] == 'CVE-2025-0001'


def test_save_creates_parent_dirs(tmp_path):
    index = FAISSIndex(
        dim           = 128,
        index_path    = str(tmp_path / 'nested' / 'deep' / 'faiss.index'),
        metadata_path = str(tmp_path / 'nested' / 'deep' / 'metadata.json'),
    )
    index.add(make_pair(), random_vec(), 'netlsd')
    index.save()
    assert (tmp_path / 'nested' / 'deep' / 'faiss.index').exists()


def test_load_restores_count(tmp_path):
    index = make_index(tmp_path)
    for i in range(3):
        index.add(make_pair(cve_id=f'CVE-2025-{i:04d}'), random_vec(seed=i), 'netlsd')
    index.save()

    index2 = make_index(tmp_path)
    index2.load()
    assert index2.index.ntotal == 3


def test_load_restores_metadata(tmp_path):
    index = make_index(tmp_path)
    index.add(make_pair(cve_id='CVE-2025-0001'), random_vec(), 'netlsd')
    index.save()

    index2 = make_index(tmp_path)
    index2.load()
    assert index2.metadata[0]['cve_id'] == 'CVE-2025-0001'


def test_roundtrip_preserves_vector(tmp_path):
    """Vector retrieved after save/load must be close to the one inserted."""
    index = make_index(tmp_path)
    vec   = random_vec(seed=42)
    index.add(make_pair(), vec, 'netlsd')
    index.save()

    index2 = make_index(tmp_path)
    index2.load()

    # search for the same vector — should get distance ~1.0 (cosine on L2-normed)
    distances, indices = index2.index.search(vec.reshape(1, -1), 1)
    assert indices[0][0] == 0
    assert distances[0][0] == pytest.approx(1.0, abs=1e-5)


# ── Retriever.query ────────────────────────────────────────────────────

def make_loaded_index(tmp_path, n=5, dim=128):
    index = make_index(tmp_path, dim=dim)
    vecs  = []
    for i in range(n):
        v = random_vec(seed=i)
        index.add(make_pair(cve_id=f'CVE-2025-{i:04d}', func_name=f'fn_{i}'), v, 'netlsd')
        vecs.append(v)
    index.save()
    index.load()
    return index, vecs


def test_query_returns_top_k(tmp_path):
    index, vecs = make_loaded_index(tmp_path, n=5)
    retriever   = Retriever(index, top_k=3)
    results     = retriever.query(vecs[0])
    assert len(results) == 3


def test_query_top_k_override(tmp_path):
    index, vecs = make_loaded_index(tmp_path, n=5)
    retriever   = Retriever(index, top_k=3)
    results     = retriever.query(vecs[0], top_k=2)
    assert len(results) == 2


def test_query_returns_score(tmp_path):
    index, vecs = make_loaded_index(tmp_path, n=3)
    retriever   = Retriever(index, top_k=1)
    results     = retriever.query(vecs[0])
    assert 'score' in results[0]
    assert isinstance(results[0]['score'], float)


def test_query_self_is_top_result(tmp_path):
    """Querying with an indexed vector must return itself as rank 1."""
    index, vecs = make_loaded_index(tmp_path, n=5)
    retriever   = Retriever(index, top_k=1)
    results     = retriever.query(vecs[2])
    assert results[0]['cve_id'] == 'CVE-2025-0002'


def test_query_results_contain_metadata(tmp_path):
    index, vecs = make_loaded_index(tmp_path, n=3)
    retriever   = Retriever(index, top_k=1)
    results     = retriever.query(vecs[0])
    assert 'cve_id'    in results[0]
    assert 'func_name' in results[0]
    assert 'cwe_id'    in results[0]


def test_query_scores_descending(tmp_path):
    """Higher scores must come first (inner product = cosine on normed vecs)."""
    index, vecs = make_loaded_index(tmp_path, n=5)
    retriever   = Retriever(index, top_k=5)
    results     = retriever.query(vecs[0])
    scores      = [r['score'] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_accepts_2d_input(tmp_path):
    """Query vector can be (1, dim) shaped."""
    index, vecs = make_loaded_index(tmp_path, n=3)
    retriever   = Retriever(index, top_k=1)
    results     = retriever.query(vecs[0].reshape(1, -1))
    assert len(results) == 1


def test_query_top_k_capped_at_index_size(tmp_path):
    """top_k larger than index should not crash — FAISS returns what it has."""
    index, vecs = make_loaded_index(tmp_path, n=2)
    retriever   = Retriever(index, top_k=10)
    results     = retriever.query(vecs[0])
    assert len(results) == 2


# ── Retriever.query_by_cve ─────────────────────────────────────────────

def test_query_by_cve_found(tmp_path):
    index, vecs = make_loaded_index(tmp_path, n=3)
    retriever   = Retriever(index, top_k=5)
    results     = retriever.query_by_cve('CVE-2025-0001')
    assert len(results) == 1
    assert results[0]['cve_id'] == 'CVE-2025-0001'


def test_query_by_cve_not_found(tmp_path):
    index, vecs = make_loaded_index(tmp_path, n=3)
    retriever   = Retriever(index, top_k=5)
    results     = retriever.query_by_cve('CVE-9999-9999')
    assert results == []


def test_query_by_cve_multiple_variants(tmp_path):
    """Same CVE indexed twice (e.g. before + after) returns both."""
    index = make_index(tmp_path)
    pair  = make_pair(cve_id='CVE-2025-0001')
    index.add(pair, random_vec(seed=0), 'netlsd')
    index.add(pair, random_vec(seed=1), 'wl')
    index.save()
    index.load()

    retriever = Retriever(index, top_k=5)
    results   = retriever.query_by_cve('CVE-2025-0001')
    assert len(results) == 2


# ── dim mismatch guard ─────────────────────────────────────────────────

def test_query_wrong_dim_raises(tmp_path):
    index, vecs = make_loaded_index(tmp_path, n=2, dim=128)
    retriever   = Retriever(index, top_k=1)
    bad_vec     = random_vec(dim=64)
    with pytest.raises(Exception):
        retriever.query(bad_vec)