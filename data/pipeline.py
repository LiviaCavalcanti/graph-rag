import subprocess
import glob
import tempfile
import textwrap
from pathlib import Path

import networkx as nx


def load_single_graph(export_xml_path: str) -> nx.MultiDiGraph | None:
    G = nx.read_graphml(export_xml_path, node_type=str, force_multigraph=True)
    nodes_to_remove = [
        n
        for n, attr in G.nodes(data=True)
        if attr.get("lableV") in ("COMMENT", "UNKNOWN")
    ]
    G.remove_nodes_from(nodes_to_remove)
    return G


def load_function_graph(
    graphml_root: str,
    cve_id: str,
    version: str,
    func_name: str,
    hint: str | None = None,
) -> nx.MultiDiGraph | None:
    patterns = [
        str(Path(graphml_root) / f"*{cve_id}*{version}*" / "**" / f"{func_name}.xml"),
        str(
            Path(graphml_root)
            / f"*{version}*"
            / "**"
            / f"{func_name}.xml"
            / "export.xml"
        ),
        str(Path(graphml_root) / "**" / f"{func_name}.xml" / "export.xml"),
    ]

    if hint:
        patterns.append(
            str(
                Path(graphml_root)
                / cve_id
                / hint
                / version
                / "**"
                / f"{func_name}.xml"
                / "export.xml"
            )
        )

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        print(f"matches: {matches}")
        if matches:
            return load_single_graph(matches[0])

    return None


def compute_graph_diff(
    G_before: nx.MultiDiGraph, G_after: nx.MultiDiGraph
) -> nx.MultiDiGraph:
    def edge_set(G):
        return {(u, v, d.get("label", "")) for u, v, d in G.edges(data=True)}

    nodes_before = set(G_before.nodes())
    nodes_after = set(G_after.nodes())

    removed_nodes = nodes_before - nodes_after
    added_nodes = nodes_after - nodes_before

    changed_nodes = removed_nodes | added_nodes

    removed_edges = edge_set(G_before) - edge_set(G_after)
    added_edges = edge_set(G_after) - edge_set(G_after)

    changed_nodes |= {u for u, v, _ in removed_edges | added_edges}
    changed_nodes |= {v for u, v, _ in removed_edges | added_edges}

    neighbourhood = set()
    for n in changed_nodes:
        if n in G_before:
            neighbourhood |= set(G_before.predecessors(n))
            neighbourhood |= set(G_before.successors(n))
        vuln_nodes = changed_nodes | neighbourhood
        G_vuln = G_before.subgraph(vuln_nodes).copy()

        for n in G_vuln.nodes():
            if n in removed_nodes:
                G_vuln.nodes[n]["diff"] = "removed"
            elif n in added_nodes:
                G_vuln.nodes[n]["diff"] = "added"
            else:
                G_vuln.nodes[n]["diff"] = "context"

    return G_vuln


def write_c_file(source_code: str, dest_path: Path) -> Path:
    """
    Write raw source (function snippet or full file) to a .c file.
    Wraps in a minimal compilable scaffold if it looks like a bare function.
    """
    stripped = source_code.strip()

    # strip markdown code fences if present (AutoPatch LLM outputs)
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    # minimal scaffold so Joern can parse without errors
    scaffold = textwrap.dedent("""\
        /* auto-generated wrapper for Joern CPG export */
        typedef unsigned int u32;
        typedef int bool;
        #define NULL ((void*)0)
        #define false 0
        #define true 1

        {code}
    """).format(code=stripped)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(scaffold)
    return dest_path


def run_joern_export(
    joern_bir_dir: str, source_file: str, out_dir: str, graph_dir: str
) -> bool:
    joern_bin = Path(joern_bir_dir)
    source = Path(source_file)
    out = Path(out_dir)
    graph_out = Path(graph_dir)
    cpg_file = out / "cpg.bin"

    out.mkdir(parents=True, exist_ok=True)

    parse_cmd = [
        str(joern_bin / "joern-parse"),
        str(source),
        "--output",
        str(cpg_file),
    ]

    result = subprocess.run(parse_cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return False
    export_cmd = [
        str(joern_bin / "joern-export"),
        "--repr",
        "cpg",
        "--format",
        "graphml",
        str(cpg_file),
        "--out",
        str(graph_out),
    ]

    result = subprocess.run(
        export_cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    return result.returncode == 0
