import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import networkx as nx

from data.autopatch import AutoPatchDataset, _VARIANTS


# ── fixtures ───────────────────────────────────────────────────────────

def make_cve_dir(tmp_path: Path, cve_id: str, db_entry: dict,
                 has_original: bool = True,
                 has_supplementary: bool = False,
                 variants: list[dict] | None = None) -> Path:
    """
    Build a minimal CVE directory matching AutoPatch on-disk structure.
    """
    cve_dir = tmp_path / cve_id
    cve_dir.mkdir()

    # db_entry.json lives under out_v2/
    out_v2 = cve_dir / 'out_v2'
    out_v2.mkdir(parents=True, exist_ok=True)
    (out_v2 / 'db_entry.json').write_text(json.dumps(db_entry))

    if has_original:
        (cve_dir / 'original_code.txt').write_text("void vuln_fn() { char buf[10]; buf[20]=0; }")
        (cve_dir / 'original_code_fixed.c').write_text("void vuln_fn() { char buf[10]; buf[9]=0; }")
        # stream() checks for vuln_patch.txt as the fixed version
        (cve_dir / 'vuln_patch.txt').write_text("void vuln_fn() { char buf[10]; buf[9]=0; }")

    if has_supplementary:
        (cve_dir / 'supplementary_code.txt').write_text(
            "```c\nstruct MyStruct { int x; };\n```\n```c\nvoid helper(){}\n```"
        )

    if variants:
        code_dir = cve_dir / 'out_v2' / 'code'
        code_dir.mkdir(parents=True, exist_ok=True)
        for v in variants:
            (code_dir / v['json_file']).write_text(json.dumps({
                'is_vulnerable': v.get('is_vulnerable', True),
                're_implemented_code': v.get('code', 'void reimpl(){}'),
                'supplementary_code': v.get('supp', ''),
            }))
            (code_dir / v['fixed_file']).write_text("void reimpl_fixed(){}")

    return cve_dir


SAMPLE_DB = {
    'cve_id': 'CVE-2025-0001',
    'cwe_type': 'CWE-119',
    'function_name': 'vuln_fn',
    'function_prototype': 'void vuln_fn()',
    'root_cause': 'buffer overflow',
    'fix_list': ['bounds check'],
}


# ── name() ────────────────────────────────────────────────────────────

def test_name():
    ds = AutoPatchDataset({'root': '/tmp', 'graphml_root': '/tmp/graphs'})
    assert ds.name() == 'AutoPatch'


# ── _load_db_entry ────────────────────────────────────────────────────

def test_load_db_entry_valid(tmp_path):
    cve_dir = make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': '/tmp'})
    result = ds._load_db_entry(cve_dir)
    assert result['cve_id'] == 'CVE-2025-0001'
    assert result['cwe_type'] == 'CWE-119'


def test_load_db_entry_missing(tmp_path):
    cve_dir = tmp_path / 'CVE-MISSING'
    cve_dir.mkdir()
    # out_v2 exists but no db_entry.json inside
    (cve_dir / 'out_v2').mkdir()
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': '/tmp'})
    assert ds._load_db_entry(cve_dir) is None


def test_load_db_entry_malformed_json(tmp_path):
    cve_dir = tmp_path / 'CVE-BAD'
    cve_dir.mkdir()
    out_v2 = cve_dir / 'out_v2'
    out_v2.mkdir()
    (out_v2 / 'db_entry.json').write_text("{ not valid json >>>")
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': '/tmp'})
    assert ds._load_db_entry(cve_dir) is None


# ── find_cve_dir ──────────────────────────────────────────────────────

def _make_cve_list(tmp_path: Path, dir_names: list[str]) -> Path:
    """Create a base_dir with CVE-list/<dir_names> sub-directories."""
    cve_list = tmp_path / "CVE-list"
    cve_list.mkdir()
    for name in dir_names:
        (cve_list / name).mkdir()
    return tmp_path


def test_find_cve_dir_exact_match(tmp_path):
    base = _make_cve_list(tmp_path, ["CVE-2024-4741"])
    result = AutoPatchDataset.find_cve_dir("CVE-2024-4741", base)
    assert result is not None
    assert result.name == "CVE-2024-4741"


def test_find_cve_dir_suffix_fallback(tmp_path):
    base = _make_cve_list(tmp_path, ["CVE-2024-53142_1", "CVE-2024-53142_2"])
    result = AutoPatchDataset.find_cve_dir("CVE-2024-53142", base)
    assert result is not None
    assert result.name == "CVE-2024-53142_1"  # first sorted suffix


def test_find_cve_dir_exact_preferred_over_suffix(tmp_path):
    """Exact match wins even when suffixed dirs also exist."""
    base = _make_cve_list(tmp_path, ["CVE-2024-100", "CVE-2024-100_1", "CVE-2024-100_2"])
    result = AutoPatchDataset.find_cve_dir("CVE-2024-100", base)
    assert result is not None
    assert result.name == "CVE-2024-100"


def test_find_cve_dir_no_match(tmp_path):
    base = _make_cve_list(tmp_path, ["CVE-2024-9999"])
    result = AutoPatchDataset.find_cve_dir("CVE-2025-0001", base)
    assert result is None


def test_find_cve_dir_no_cve_list_dir(tmp_path):
    """base_dir exists but has no CVE-list sub-directory."""
    result = AutoPatchDataset.find_cve_dir("CVE-2025-0001", tmp_path)
    assert result is None


def test_find_cve_dir_ignores_files(tmp_path):
    """Files matching the prefix should not be returned."""
    cve_list = tmp_path / "CVE-list"
    cve_list.mkdir()
    (cve_list / "CVE-2024-100_1").write_text("not a dir")  # file, not dir
    result = AutoPatchDataset.find_cve_dir("CVE-2024-100", tmp_path)
    assert result is None


def test_find_cve_dir_no_partial_prefix_match(tmp_path):
    """CVE-2024-10 should NOT match CVE-2024-100_1 (prefix must be exact + '_')."""
    base = _make_cve_list(tmp_path, ["CVE-2024-100_1"])
    result = AutoPatchDataset.find_cve_dir("CVE-2024-10", base)
    assert result is None


# ── load_ground_truth ─────────────────────────────────────────────────

def _make_gt_file(tmp_path: Path, cve_id: str, variant: str, content: str) -> Path:
    """Create the canonical ground-truth file on disk."""
    code_dir = tmp_path / "CVE-list" / cve_id / "out_v2" / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    gt_file = code_dir / f"{variant}_fixed.c"
    gt_file.write_text(content)
    return gt_file


def test_load_ground_truth_canonical_file(tmp_path):
    _make_gt_file(tmp_path, "CVE-2025-0001", "augmented", "void fixed(){}")
    result = AutoPatchDataset.load_ground_truth(
        "CVE-2025-0001", "augmented", "", tmp_path
    )
    assert result == "void fixed(){}"


def test_load_ground_truth_fallback_to_relative_path(tmp_path):
    """When canonical file doesn't exist, falls back to gt_path_str as relative path."""
    rel = "some/other/file.c"
    (tmp_path / "some" / "other").mkdir(parents=True)
    (tmp_path / rel).write_text("fallback content")
    result = AutoPatchDataset.load_ground_truth("CVE-MISSING", "v1", rel, tmp_path)
    assert result == "fallback content"


def test_load_ground_truth_fallback_to_inline_code(tmp_path):
    """When gt_path_str contains newlines, treat it as inline code."""
    inline = "void f() {\n    return;\n}"
    result = AutoPatchDataset.load_ground_truth("CVE-MISSING", "v1", inline, tmp_path)
    assert result == inline


def test_load_ground_truth_fallback_to_long_string(tmp_path):
    """Strings longer than 260 chars are treated as inline code."""
    long_code = "x" * 300
    result = AutoPatchDataset.load_ground_truth("CVE-MISSING", "v1", long_code, tmp_path)
    assert result == long_code


def test_load_ground_truth_canonical_preferred_over_fallback(tmp_path):
    """Canonical file takes priority even when gt_path_str is also valid."""
    _make_gt_file(tmp_path, "CVE-2025-0001", "augmented", "canonical")
    (tmp_path / "fallback.c").write_text("fallback")
    result = AutoPatchDataset.load_ground_truth(
        "CVE-2025-0001", "augmented", "fallback.c", tmp_path
    )
    assert result == "canonical"


def test_load_ground_truth_returns_none_when_nothing_found(tmp_path):
    result = AutoPatchDataset.load_ground_truth("CVE-MISSING", "v1", "", tmp_path)
    assert result is None


def test_load_ground_truth_empty_cve_id_uses_fallback(tmp_path):
    """Empty cve_id skips canonical lookup, goes straight to fallback."""
    inline = "void f() {\n}"
    result = AutoPatchDataset.load_ground_truth("", "augmented", inline, tmp_path)
    assert result == inline


def test_load_ground_truth_empty_variant_uses_fallback(tmp_path):
    """Empty variant skips canonical lookup, goes straight to fallback."""
    (tmp_path / "gt.c").write_text("from path")
    result = AutoPatchDataset.load_ground_truth("CVE-2025-0001", "", "gt.c", tmp_path)
    assert result == "from path"


def test_load_ground_truth_suffix_cve_dir(tmp_path):
    """Works with suffixed CVE dirs (e.g. CVE-2024-53142_1)."""
    _make_gt_file(tmp_path, "CVE-2024-53142_1", "augmented", "suffixed content")
    result = AutoPatchDataset.load_ground_truth(
        "CVE-2024-53142", "augmented", "", tmp_path
    )
    assert result == "suffixed content"


def test_load_ground_truth_nonexistent_relative_path(tmp_path):
    """Non-existent relative path with no newlines returns None."""
    result = AutoPatchDataset.load_ground_truth(
        "CVE-MISSING", "v1", "does/not/exist.c", tmp_path
    )
    assert result is None


# ── load_db_cache ─────────────────────────────────────────────────────

def test_load_db_cache_loads_all(tmp_path):
    for name in ["CVE-A", "CVE-B"]:
        d = tmp_path / name
        d.mkdir()
        out = d / "out_v2"
        out.mkdir()
        (out / "db_entry.json").write_text(json.dumps({"cve_id": name}))
    cache = AutoPatchDataset.load_db_cache(tmp_path)
    assert len(cache) == 2
    assert cache["CVE-A"]["cve_id"] == "CVE-A"
    assert cache["CVE-B"]["cve_id"] == "CVE-B"


def test_load_db_cache_skips_files(tmp_path):
    (tmp_path / "not-a-dir.txt").write_text("file")
    d = tmp_path / "CVE-A"
    d.mkdir()
    out = d / "out_v2"
    out.mkdir()
    (out / "db_entry.json").write_text(json.dumps({"cve_id": "CVE-A"}))
    cache = AutoPatchDataset.load_db_cache(tmp_path)
    assert len(cache) == 1


def test_load_db_cache_skips_missing_db_entry(tmp_path):
    d = tmp_path / "CVE-NO-DB"
    d.mkdir()
    (d / "out_v2").mkdir()
    # no db_entry.json
    cache = AutoPatchDataset.load_db_cache(tmp_path)
    assert len(cache) == 0


def test_load_db_cache_skips_malformed_json(tmp_path):
    d = tmp_path / "CVE-BAD"
    d.mkdir()
    out = d / "out_v2"
    out.mkdir()
    (out / "db_entry.json").write_text("{broken json")
    cache = AutoPatchDataset.load_db_cache(tmp_path)
    assert len(cache) == 0


def test_load_db_cache_empty_root(tmp_path):
    cache = AutoPatchDataset.load_db_cache(tmp_path)
    assert cache == {}


def test_load_db_cache_sorted_keys(tmp_path):
    for name in ["CVE-C", "CVE-A", "CVE-B"]:
        d = tmp_path / name
        d.mkdir()
        out = d / "out_v2"
        out.mkdir()
        (out / "db_entry.json").write_text(json.dumps({"cve_id": name}))
    cache = AutoPatchDataset.load_db_cache(tmp_path)
    assert list(cache.keys()) == ["CVE-A", "CVE-B", "CVE-C"]


# ── _variants_to_use ──────────────────────────────────────────────────

def test_variants_to_use_default():
    ds = AutoPatchDataset({'root': '/tmp', 'graphml_root': '/tmp', 'include_variants': False})
    variants = ds._variants_to_use()
    assert len(variants) == 1
    assert variants[0][0] == 'augmented.json'


def test_variants_to_use_all():
    ds = AutoPatchDataset({'root': '/tmp', 'graphml_root': '/tmp', 'include_variants': True})
    variants = ds._variants_to_use()
    assert len(variants) == len(_VARIANTS)


# ── export_jobs ────────────────────────────────────────────────────────

def test_export_jobs_original_pair(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB, has_original=True)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))

    assert len(jobs) == 2
    versions = {j.version for j in jobs}
    assert versions == {'before', 'after'}


def test_export_jobs_original_source_code(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))

    before_job = next(j for j in jobs if j.version == 'before')
    assert 'vuln_fn' in before_job.source_code
    assert before_job.variant == 'original'


def test_export_jobs_supplementary_code_attached(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB, has_supplementary=True)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))

    for job in jobs:
        # read_supplementary_code extracts C snippets from markdown fences
        assert 'MyStruct' in job.supplementary_code or 'helper' in job.supplementary_code


def test_export_jobs_no_supplementary_when_missing(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB, has_supplementary=False)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))

    for job in jobs:
        assert job.supplementary_code == ''


def test_export_jobs_skips_missing_original(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB, has_original=False)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))
    assert len(jobs) == 0


def test_export_jobs_variants_included(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB, variants=[
        {'json_file': 'augmented.json', 'fixed_file': 'augmented_fixed.c'},
    ])
    ds = AutoPatchDataset({
        'root': str(tmp_path),
        'graphml_root': str(tmp_path / 'graphs'),
        'include_variants': True,
    })
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))
    variant_jobs = [j for j in jobs if j.variant == 'augmented']
    assert len(variant_jobs) == 2  # before + after


def test_export_jobs_variants_excluded_by_default(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB,
                 has_original=True,
                 variants=[{'json_file': 'augmented.json', 'fixed_file': 'augmented_fixed.c'}])
    ds = AutoPatchDataset({
        'root': str(tmp_path),
        'graphml_root': str(tmp_path / 'graphs'),
        'include_variants': False,
    })
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))
    # only original pair
    assert all(j.variant == 'original' for j in jobs)


def test_export_jobs_skips_non_vulnerable_variants(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB, variants=[
        {'json_file': 'augmented.json', 'fixed_file': 'augmented_fixed.c', 'is_vulnerable': False},
    ])
    ds = AutoPatchDataset({
        'root': str(tmp_path),
        'graphml_root': str(tmp_path / 'graphs'),
        'include_variants': True,
    })
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))
    variant_jobs = [j for j in jobs if j.variant == 'augmented']
    assert len(variant_jobs) == 0


def test_export_jobs_out_dir_follows_convention(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB)
    graphml_root = str(tmp_path / 'graphs')
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': graphml_root})
    jobs = list(ds.export_jobs(graphml_root))

    before_job = next(j for j in jobs if j.version == 'before')
    assert 'CVE-2025-0001' in before_job.out_dir
    assert 'original' in before_job.out_dir
    assert 'before' in before_job.out_dir


def test_export_jobs_skips_dir_without_db_entry(tmp_path):
    bad_dir = tmp_path / 'not-a-cve'
    bad_dir.mkdir()
    # no out_v2/db_entry.json → should be skipped
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))
    assert len(jobs) == 0


def test_export_jobs_multiple_cves(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', {**SAMPLE_DB, 'cve_id': 'CVE-2025-0001'})
    make_cve_dir(tmp_path, 'CVE-2025-0002', {**SAMPLE_DB, 'cve_id': 'CVE-2025-0002'})
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})
    jobs = list(ds.export_jobs(str(tmp_path / 'graphs')))
    assert len(jobs) == 4  # 2 CVEs × 2 versions


# ── stream ─────────────────────────────────────────────────────────────

def _mock_graph(n_nodes=3):
    G = nx.MultiDiGraph()
    for i in range(n_nodes):
        G.add_node(str(i), labelV='METHOD')
    G.add_edge('0', '1', label='CFG')
    G.add_edge('1', '2', label='AST')
    return G


def test_stream_yields_function_pair(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})

    mock_graph = _mock_graph()
    with patch('data.autopatch.load_cpg_dir', return_value=mock_graph), \
         patch('data.autopatch.cpg_dir_for', return_value='/fake/path'), \
         patch('data.autopatch.compute_graph_diff', return_value=mock_graph):
        pairs = list(ds.stream())

    assert len(pairs) == 1
    assert pairs[0].cve_id == 'CVE-2025-0001'
    assert pairs[0].func_name == 'vuln_fn'
    assert pairs[0].cwe_id == 'CWE-119'
    assert pairs[0].project == 'autopatch'


def test_stream_meta_fields(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})

    with patch('data.autopatch.load_cpg_dir', return_value=_mock_graph()), \
         patch('data.autopatch.cpg_dir_for', return_value='/fake/path'), \
         patch('data.autopatch.compute_graph_diff', return_value=_mock_graph()):
        pairs = list(ds.stream())

    meta = pairs[0].meta
    assert meta['dataset'] == 'AutoPatch'
    assert meta['variant'] == 'original'
    assert meta['root_cause'] == 'buffer overflow'
    assert meta['fix_list'] == ['bounds check']


def test_stream_skips_when_graphs_missing(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})

    with patch('data.autopatch.load_cpg_dir', return_value=None), \
         patch('data.autopatch.cpg_dir_for', return_value='/fake/path'):
        pairs = list(ds.stream())

    assert len(pairs) == 0


def test_stream_skips_empty_graph(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})

    with patch('data.autopatch.load_cpg_dir', return_value=nx.MultiDiGraph()), \
         patch('data.autopatch.cpg_dir_for', return_value='/fake/path'), \
         patch('data.autopatch.compute_graph_diff', return_value=nx.MultiDiGraph()):
        pairs = list(ds.stream())

    assert len(pairs) == 0


def test_stream_g_vuln_is_subgraph_of_g_before(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})

    G_before = _mock_graph(4)
    G_after  = _mock_graph(3)   # one node removed
    G_vuln   = _mock_graph(2)   # diff subgraph

    call_count = {'n': 0}
    def side_effect(_path):
        call_count['n'] += 1
        return G_before if call_count['n'] == 1 else G_after

    with patch('data.autopatch.load_cpg_dir', side_effect=side_effect), \
         patch('data.autopatch.cpg_dir_for', return_value='/fake/path'), \
         patch('data.autopatch.compute_graph_diff', return_value=G_vuln):
        pairs = list(ds.stream())

    if pairs:
        vuln_nodes = set(pairs[0].G_vuln.nodes())
        before_nodes = set(pairs[0].G_before.nodes())
        assert vuln_nodes.issubset(before_nodes)


def test_stream_supplementary_in_meta(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB, has_supplementary=True)
    ds = AutoPatchDataset({'root': str(tmp_path), 'graphml_root': str(tmp_path / 'graphs')})

    with patch('data.autopatch.load_cpg_dir', return_value=_mock_graph()), \
         patch('data.autopatch.cpg_dir_for', return_value='/fake/path'), \
         patch('data.autopatch.compute_graph_diff', return_value=_mock_graph()):
        pairs = list(ds.stream())

    assert 'supplementary_code' in pairs[0].meta
    assert 'MyStruct' in pairs[0].meta['supplementary_code']


def test_stream_variants_not_yielded_by_default(tmp_path):
    make_cve_dir(tmp_path, 'CVE-2025-0001', SAMPLE_DB, variants=[
        {'json_file': 'augmented.json', 'fixed_file': 'augmented_fixed.c'},
    ])
    ds = AutoPatchDataset({
        'root': str(tmp_path),
        'graphml_root': str(tmp_path / 'graphs'),
        'include_variants': False,
    })
    with patch('data.autopatch.load_cpg_dir', return_value=_mock_graph()), \
         patch('data.autopatch.cpg_dir_for', return_value='/fake/path'), \
         patch('data.autopatch.compute_graph_diff', return_value=_mock_graph()):
        pairs = list(ds.stream())

    assert all(p.meta['variant'] == 'original' for p in pairs)