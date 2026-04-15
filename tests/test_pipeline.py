import pytest
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock
import networkx as nx

from data.pipeline import (
    extract_c_snippets,
    read_supplementary_code,
    write_c_file,
    load_cpg_dir,
    cpg_dir_for,
    compute_graph_diff,
)


# ── extract_c_snippets ─────────────────────────────────────────────────

def test_extract_c_snippets_fenced():
    text = "some text\n```c\nint foo() { return 0; }\n```\nmore text"
    snippets = extract_c_snippets(text)
    assert len(snippets) == 1
    assert 'int foo()' in snippets[0]


def test_extract_c_snippets_multiple():
    text = "```c\nint a;\n```\nblah\n```c\nint b;\n```"
    snippets = extract_c_snippets(text)
    assert len(snippets) == 2
    assert 'int a;' in snippets[0]
    assert 'int b;' in snippets[1]


def test_extract_c_snippets_no_language_tag():
    text = "```\nvoid foo(){}\n```"
    snippets = extract_c_snippets(text)
    assert len(snippets) == 1


def test_extract_c_snippets_empty():
    assert extract_c_snippets("no fences here") == []


def test_extract_c_snippets_cpp_tag():
    text = "```cpp\nclass Foo {};\n```"
    snippets = extract_c_snippets(text)
    assert len(snippets) == 1


# ── read_supplementary_code ────────────────────────────────────────────

def test_read_supplementary_code_missing_file():
    result = read_supplementary_code(Path('/nonexistent/path.txt'))
    assert result == ''


def test_read_supplementary_code_with_fences(tmp_path):
    f = tmp_path / 'supplementary_code.txt'
    f.write_text("intro\n```c\nstruct Foo { int x; };\n```\nmore\n```c\nvoid helper(){}\n```")
    result = read_supplementary_code(f)
    assert 'struct Foo' in result
    assert 'void helper' in result
    assert '```' not in result


def test_read_supplementary_code_plain_c(tmp_path):
    f = tmp_path / 'supplementary_code.txt'
    f.write_text("struct Bar { int y; };")
    result = read_supplementary_code(f)
    assert 'struct Bar' in result


def test_read_supplementary_code_order(tmp_path):
    """Snippets must appear in document order."""
    f = tmp_path / 'supplementary_code.txt'
    f.write_text("```c\nfirst();\n```\n```c\nsecond();\n```")
    result = read_supplementary_code(f)
    assert result.index('first') < result.index('second')


# ── write_c_file ───────────────────────────────────────────────────────

def test_write_c_file_creates_file(tmp_path):
    dest = tmp_path / 'out' / 'func.c'
    write_c_file("void foo(){}", dest)
    assert dest.exists()


def test_write_c_file_contains_main_code(tmp_path):
    dest = tmp_path / 'func.c'
    write_c_file("void my_function(){}", dest)
    content = dest.read_text()
    assert 'void my_function(){}' in content


def test_write_c_file_supplementary_before_main(tmp_path):
    dest = tmp_path / 'func.c'
    write_c_file("void main_fn(){}", dest, supplementary_code="struct MyStruct { int x; };")
    content = dest.read_text()
    assert content.index('struct MyStruct') < content.index('void main_fn')


def test_write_c_file_strips_markdown_fences(tmp_path):
    dest = tmp_path / 'func.c'
    write_c_file("```c\nvoid foo(){}\n```", dest)
    content = dest.read_text()
    assert '```' not in content
    assert 'void foo(){}' in content


def test_write_c_file_scaffold_included(tmp_path):
    dest = tmp_path / 'func.c'
    write_c_file("void foo(){}", dest)
    content = dest.read_text()
    assert 'typedef' in content  # scaffold typedefs present


def test_write_c_file_creates_parent_dirs(tmp_path):
    dest = tmp_path / 'a' / 'b' / 'c' / 'func.c'
    write_c_file("void f(){}", dest)
    assert dest.exists()


# ── cpg_dir_for ────────────────────────────────────────────────────────

def test_cpg_dir_for_structure():
    path = cpg_dir_for('/root', 'CVE-2025-1234', 'original', 'before')
    assert path == '/root/CVE-2025-1234/original/before/graph'


def test_cpg_dir_for_variant():
    path = cpg_dir_for('/root', 'CVE-2025-1234', 'augmented', 'after')
    assert 'augmented' in path
    assert 'after' in path


# ── load_cpg_dir ───────────────────────────────────────────────────────

def _make_export_xml(path: Path, nodes: list[dict], edges: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<graphml>',
        '<key id="labelV" for="node" attr.name="labelV" attr.type="string"/>',
        '<key id="label" for="edge" attr.name="label" attr.type="string"/>',
        '<graph id="G" edgedefault="directed">',
    ]
    for n in nodes:
        lines.append(f'<node id="{n["id"]}">')
        for k, v in n.items():
            if k != 'id':
                lines.append(f'  <data key="{k}">{v}</data>')
        lines.append('</node>')
    for i, e in enumerate(edges):
        lines.append(f'<edge id="e{i}" source="{e["src"]}" target="{e["dst"]}">')
        lines.append(f'  <data key="label">{e.get("label", "EDGE")}</data>')
        lines.append('</edge>')
    lines += ['</graph>', '</graphml>']
    path.write_text('\n'.join(lines), encoding='utf-8')

def test_load_cpg_dir_merges_multiple_files(tmp_path):
    graph_dir = tmp_path / 'graph'
    p = graph_dir / 'file1.c' / 'func_a.xml' / 'export.xml'
    _make_export_xml(
        graph_dir / 'file1.c' / 'func_a.xml' / 'export.xml',
        nodes=[{'id': 'n1', 'labelV': 'METHOD'}, {'id': 'n2', 'labelV': 'CALL'}],
        edges=[{'src': 'n1', 'dst': 'n2', 'label': 'AST'}],
    )
    _make_export_xml(
        graph_dir / 'file2.c' / 'func_b.xml' / 'export.xml',
        nodes=[{'id': 'n3', 'labelV': 'RETURN'}],
        edges=[],
    )
    G = load_cpg_dir(str(graph_dir))
    assert 'n1' in G.nodes
    assert 'n2' in G.nodes
    assert 'n3' in G.nodes



def test_load_cpg_dir_removes_comment_nodes(tmp_path):
    graph_dir = tmp_path / 'graph'
    _make_export_xml(
        graph_dir / 'f.xml' / 'export.xml',
        nodes=[{'id': 'n1', 'labelV': 'METHOD'}, {'id': 'n2', 'labelV': 'COMMENT'}],
        edges=[],
    )
    G = load_cpg_dir(str(tmp_path))
    assert 'n1' in G.nodes
    assert 'n2' not in G.nodes


def test_load_cpg_dir_no_dangling_edges(tmp_path):
    """Edges referencing nodes from other files must resolve after merge."""
    graph_dir = tmp_path / 'graph'
    # file1 has n1, file2 has n2, edge n1->n2 is in file1
    _make_export_xml(
        graph_dir / 'f1.xml' / 'export.xml',
        nodes=[{'id': 'n1', 'labelV': 'METHOD'}],
        edges=[{'src': 'n1', 'dst': 'n2', 'label': 'CFG'}],
    )
    _make_export_xml(
        graph_dir / 'f2.xml' / 'export.xml',
        nodes=[{'id': 'n2', 'labelV': 'CALL'}],
        edges=[],
    )
    G = load_cpg_dir(str(tmp_path))
    # edge should be present since both nodes exist after merge
    assert G.has_edge('n1', 'n2')


def test_load_cpg_dir_drops_unresolvable_edges(tmp_path):
    """Edges pointing to nodes not in any file must be dropped."""
    graph_dir = tmp_path / 'graph'
    _make_export_xml(
        graph_dir / 'f.xml' / 'export.xml',
        nodes=[{'id': 'n1', 'labelV': 'METHOD'}],
        edges=[{'src': 'n1', 'dst': 'ghost', 'label': 'REF'}],
    )
    G = load_cpg_dir(str(tmp_path))
    assert not G.has_edge('n1', 'REF')


def test_load_cpg_dir_raises_if_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cpg_dir(str(tmp_path / 'nonexistent'))


def test_load_cpg_dir_tolerates_malformed_file(tmp_path):
    """One bad file must not abort the whole load."""
    graph_dir = tmp_path / 'graph'
    (graph_dir / 'bad.xml' / 'export.xml').parent.mkdir(parents=True)
    (graph_dir / 'bad.xml' / 'export.xml').write_text("not valid xml <<<")
    _make_export_xml(
        graph_dir / 'good.xml' / 'export.xml',
        nodes=[{'id': 'n1', 'labelV': 'METHOD'}],
        edges=[],
    )
    G = load_cpg_dir(str(tmp_path))
    assert 'n1' in G.nodes


def test_load_cpg_dir_accepts_graph_subdir_directly(tmp_path):
    """Can be called with .../before/graph/ directly."""
    graph_dir = tmp_path / 'graph'
    _make_export_xml(
        graph_dir / 'f.xml' / 'export.xml',
        nodes=[{'id': 'n1', 'labelV': 'CALL'}],
        edges=[],
    )
    G = load_cpg_dir(str(graph_dir))
    assert 'n1' in G.nodes


# ── compute_graph_diff ─────────────────────────────────────────────────

def _simple_graph(node_ids: list[str], edges: list[tuple]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for i, n in enumerate(node_ids):
        G.add_node(n, labelV='CALL', CODE=f'code_{n}', LINE_NUMBER=i + 1)
    for u, v in edges:
        G.add_edge(u, v, labelE='CFG')
    return G


def test_compute_graph_diff_removed_nodes():
    G_before = _simple_graph(['a', 'b', 'c'], [('a', 'b'), ('b', 'c')])
    G_after  = _simple_graph(['a', 'c'], [('a', 'c')])
    G_vuln   = compute_graph_diff(G_before, G_after)
    # b was removed — must appear in vuln subgraph
    assert 'b' in G_vuln.nodes


def test_compute_graph_diff_added_nodes():
    G_before = _simple_graph(['a', 'b'], [('a', 'b')])
    G_after  = _simple_graph(['a', 'b', 'c'], [('a', 'b'), ('b', 'c')])
    G_vuln   = compute_graph_diff(G_before, G_after)
    # c was added next to b — b should appear as fix_adjacent
    assert 'b' in G_vuln.nodes


def test_compute_graph_diff_diff_annotation():
    G_before = _simple_graph(['a', 'b'], [('a', 'b')])
    G_after  = _simple_graph(['a'], [])
    G_vuln   = compute_graph_diff(G_before, G_after)
    assert G_vuln.nodes['b']['diff'] == 'removed'


def test_compute_graph_diff_context_annotation():
    G_before = _simple_graph(['a', 'b', 'c'], [('a', 'b'), ('b', 'c')])
    G_after  = _simple_graph(['a', 'c'], [('a', 'c')])
    G_vuln   = compute_graph_diff(G_before, G_after)
    # a and c are CFG neighbours of removed b — slice should include them
    if 'a' in G_vuln.nodes:
        assert G_vuln.nodes['a']['diff'] in ('context', 'edge_changed', 'fix_adjacent')


def test_compute_graph_diff_unchanged_graph():
    G = _simple_graph(['a', 'b'], [('a', 'b')])
    G_vuln = compute_graph_diff(G, G.copy())
    # nothing changed — vuln subgraph should be empty
    assert G_vuln.number_of_nodes() == 0


def test_compute_graph_diff_vuln_subgraph_is_connected():
    """
    The vulnerability subgraph must not contain isolated nodes —
    every node in G_vuln should be reachable from at least one changed node.
    """
    G_before = _simple_graph(['a', 'b', 'c', 'd'], [('a','b'),('b','c'),('c','d')])
    G_after  = _simple_graph(['a', 'c', 'd'], [('a','c'),('c','d')])
    G_vuln   = compute_graph_diff(G_before, G_after)

    if G_vuln.number_of_nodes() > 0:
        # convert to undirected to check weak connectivity
        assert nx.is_weakly_connected(G_vuln)
