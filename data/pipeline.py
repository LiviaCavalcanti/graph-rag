import glob
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
