import subprocess
import glob
import textwrap
from pathlib import Path
import re
import networkx as nx


def extract_c_snippets(text: str) -> list[str]:
    pattern = re.compile(r"```(?:c|cpp)?\s*\n(.*?)```", re.DOTALL)
    return [m.group(1).strip() for m in pattern.finditer(text)]


def read_supplementary_code(supp_path: Path) -> str:
    if not supp_path.exists():
        print(f"Path for supplementary code does not exist {supp_path}")
        return ""

    raw = supp_path.read_text(encoding="utf-8", errors="replace")
    snippets = extract_c_snippets(raw)
    if not snippets:
        return raw.strip()

    return "\n\n".join(snippets)


def cpg_dir_for(graphml_root: str, cve_id: str, variant: str, version: str) -> str:
    return str(Path(graphml_root) / cve_id / variant / version / "graph")


def load_cpg_dir(graph_dir: str) -> nx.MultiDiGraph:
    root = Path(graph_dir)
    print(f"Loading CPG from {root}")
    if not (root / "graph").exists() and root.name != "graph":
        root = root / "graph"

    files = glob.glob(str(root / "**" / "export.xml"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No export.xml found under {root}")
    G = nx.MultiDiGraph()
    
    # track which node IDs were explicitly declared in a <node> element
    # vs implicitly created by NetworkX when an edge referenced them
    declared_nodes: set[str] = set()
    for f in files:
        try:
            sub = nx.read_graphml(f, node_type=str, force_multigraph=True)
            declared_nodes.update(sub.nodes())
            G.update(sub)
        except Exception as e:
            print(f" warning: could not parse {f}: {e} \n Content was:\n{Path(f).read_text()}")

    noise = {
        n for n, attr in G.nodes(data=True)
        if attr.get('labelV') in ('COMMENT', 'UNKNOWN')
    }
    
    declared_nodes -= noise
    G.remove_nodes_from(noise)
    phantom_nodes = set(G.nodes()) - declared_nodes
    G.remove_nodes_from(phantom_nodes)

    # clean edges of removed phantom 
    dangling = [
        (u, v, k) for u, v, k in G.edges(keys=True)
        if u not in G._node or v not in G._node
    ]
    print(f"{graph_dir} -- Declared nodes: {len(declared_nodes)}, noise: {len(noise)}, dangling nodes: {len(dangling)}")
    G.remove_edges_from(dangling)
    return G


def compute_graph_diff(
    G_before: nx.MultiDiGraph, G_after: nx.MultiDiGraph
) -> nx.MultiDiGraph:
    def edge_set(G):
        return {(u, v, d.get("labelE") or d.get("label", "")) for u, v, d in G.edges(data=True)}

    nodes_before = set(G_before.nodes())
    nodes_after = set(G_after.nodes())

    removed_nodes = nodes_before - nodes_after
    added_nodes = nodes_after - nodes_before

    changed_nodes = removed_nodes | added_nodes

    removed_edges = edge_set(G_before) - edge_set(G_after)
    added_edges = edge_set(G_after) - edge_set(G_before)

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
    if changed_nodes == set():
        print("No patch changes to graph")
        G_vuln = nx.MultiDiGraph()
    return G_vuln


def write_c_file(
    source_code: str, dest_path: Path, supplementary_code: str = ""
) -> Path:
    """
    Write raw source (function snippet or full file) to a .c file.
    Wraps in a minimal compilable scaffold if it looks like a bare function.
    """

    def strip_fences(code: str) -> str:
        stripped = source_code.strip()

        # strip markdown code fences if present (AutoPatch LLM outputs)
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()
        return stripped

    main_code = strip_fences(source_code)
    supp_code = strip_fences(supplementary_code) if supplementary_code else ''

    # minimal scaffold so Joern can parse without errors
    scaffold = textwrap.dedent("""\
        /* auto-generated wrapper for Joern CPG export */
        typedef unsigned int u32;
        typedef int bool;
        #define NULL ((void*)0)
        #define false 0
        #define true 1

        {supplementary}
                               
        {code}
    """).format(code=main_code, supplementary=supp_code)

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
        "--language", "c",
    ]

    result = subprocess.run(parse_cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"EXPORT ERROR: {result.stderr} , {result}")
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
    if result.returncode != 0:
        print(result)
        print(f"EXPORT ERROR: {result.stderr}")
    return result.returncode == 0
