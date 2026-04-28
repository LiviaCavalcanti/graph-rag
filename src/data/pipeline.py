import glob
import re
import subprocess
import textwrap
from pathlib import Path

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
            print(
                f" warning: could not parse {f}: {e} \n Content was:\n{Path(f).read_text()}"
            )

    noise = {
        n
        for n, attr in G.nodes(data=True)
        if attr.get("labelV") in ("COMMENT", "UNKNOWN")
    }

    declared_nodes -= noise
    G.remove_nodes_from(noise)
    phantom_nodes = set(G.nodes()) - declared_nodes
    G.remove_nodes_from(phantom_nodes)

    # clean edges of removed phantom
    dangling = [
        (u, v, k)
        for u, v, k in G.edges(keys=True)
        if u not in G._node or v not in G._node
    ]
    # print(f"{graph_dir} -- Declared nodes: {len(declared_nodes)}, noise: {len(noise)}, dangling nodes: {len(dangling)}")
    G.remove_edges_from(dangling)
    return G


# diff-type → weight mapping (used by both compute_graph_diff and slicing)
CHANGE_WEIGHT = {
    "removed": 1.0,
    "fix_adjacent": 0.8,
    "edge_changed": 0.6,
    "context": 0.2,
}


def compute_graph_diff(
    G_before: nx.MultiDiGraph, G_after: nx.MultiDiGraph
) -> nx.MultiDiGraph:
    """
    Semantic graph diff + vulnerability-aware program slice.

    Matches nodes by (labelV, CODE, LINE_NUMBER) instead of node ID
    so that ID renumbering between before/after does not produce false
    changes.  Extracts a bounded program slice by following CFG,
    REACHING_DEF and CDG edges from the truly changed nodes.
    """
    from collections import Counter

    # ── config ───────────────────────────────────────────────────
    NOISE_TYPES = {
        "TYPE_DECL",
        "FILE",
        "NAMESPACE_BLOCK",
        "COMMENT",
        "UNKNOWN",
        "METHOD_RETURN",
    }
    FLOW_EDGES = {"CFG", "CDG", "REACHING_DEF", "PDG", "DDG"}
    SLICE_DEPTH = 3  # hops along flow edges from seed nodes

    # ── helpers ──────────────────────────────────────────────────
    def _code(attrs: dict) -> str:
        v = attrs.get("CODE")
        return str(v).strip() if v else ""

    def _node_fp(G: nx.MultiDiGraph, n) -> tuple:
        """Semantic fingerprint resilient to ID renumbering."""
        a = G.nodes[n]
        return (a.get("labelV", ""), _code(a), str(a.get("LINE_NUMBER", "")))

    def _edge_fp(G, u, v, d) -> tuple:
        return (_node_fp(G, u), _node_fp(G, v), d.get("labelE") or d.get("label", ""))

    def _is_semantic(G, n) -> bool:
        return G.nodes[n].get("labelV") not in NOISE_TYPES

    # ── 1. semantic node diff ────────────────────────────────────
    before_fps = Counter(
        _node_fp(G_before, n) for n in G_before if _is_semantic(G_before, n)
    )
    after_fps = Counter(
        _node_fp(G_after, n) for n in G_after if _is_semantic(G_after, n)
    )

    removed_fps = {fp for fp in before_fps if before_fps[fp] > after_fps.get(fp, 0)}
    added_fps = {fp for fp in after_fps if after_fps[fp] > before_fps.get(fp, 0)}

    changed = set()
    diff_label = {}

    # nodes whose code was removed/reduced in the patch
    for n in G_before:
        if _is_semantic(G_before, n) and _node_fp(G_before, n) in removed_fps:
            changed.add(n)
            diff_label[n] = "removed"

    # for *added* code in the patch: the fix was inserted next to some
    # existing nodes – find those neighbors in G_before to mark where
    # the vulnerability sits
    after_fp_to_nodes = {}
    for n in G_after:
        after_fp_to_nodes.setdefault(_node_fp(G_after, n), []).append(n)

    before_fp_to_nodes = {}
    for n in G_before:
        before_fp_to_nodes.setdefault(_node_fp(G_before, n), []).append(n)

    # for the added nodes check their neighbors in G_after,
    # then find any nodes in G_before with the same fingerprint as those neighbors,
    # and mark them as 'fix_adjacent' (if not already marked as 'removed')
    for fp in added_fps:
        for n_after in after_fp_to_nodes.get(fp, []):
            neighbors = set(G_after.predecessors(n_after)) | set(
                G_after.successors(n_after)
            )
            for nb in neighbors:
                nb_fp = _node_fp(G_after, nb)
                for n_before in before_fp_to_nodes.get(nb_fp, []):
                    if n_before not in diff_label:
                        changed.add(n_before)
                        diff_label[n_before] = "fix_adjacent"

    # ── 2. semantic edge diff ────────────────────────────────────
    before_efps = Counter(
        _edge_fp(G_before, u, v, d) for u, v, d in G_before.edges(data=True)
    )
    after_efps = Counter(
        _edge_fp(G_after, u, v, d) for u, v, d in G_after.edges(data=True)
    )

    changed_efps = {
        efp
        for efp in before_efps | after_efps
        if before_efps.get(efp, 0) != after_efps.get(efp, 0)
    }

    for u, v, d in G_before.edges(data=True):
        if _edge_fp(G_before, u, v, d) in changed_efps:
            for nd in (u, v):
                changed.add(nd)
                diff_label.setdefault(nd, "edge_changed")

    # ── 3. bounded program slice along flow edges ────────────────
    slice_nodes = set(changed)
    frontier = set(changed)

    for _ in range(SLICE_DEPTH):
        next_frontier = set()
        for n in frontier:
            if n not in G_before:
                continue
            for _, tgt, d in G_before.out_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in FLOW_EDGES and tgt not in slice_nodes:
                    next_frontier.add(tgt)
            for src, _, d in G_before.in_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in FLOW_EDGES and src not in slice_nodes:
                    next_frontier.add(src)
        slice_nodes |= next_frontier
        frontier = next_frontier

    # ── 4. filter noise types ────────────────────────────────────
    slice_nodes = {
        n for n in slice_nodes if n in G_before and _is_semantic(G_before, n)
    }

    if not slice_nodes:
        print("No semantic changes detected between before/after graphs")
        return nx.MultiDiGraph()

    # ── 5. build subgraph with diff labels + weights ────────────
    G_vuln = G_before.subgraph(slice_nodes).copy()
    for n in G_vuln:
        dlabel = diff_label.get(n, "context")
        G_vuln.nodes[n]["diff"] = dlabel
        G_vuln.nodes[n]["diff_weight"] = CHANGE_WEIGHT.get(dlabel, 0.2)

    return G_vuln


def write_c_file(
    source_code: str, dest_path: Path, supplementary_code: str = ""
) -> Path:
    """
    Write raw source (function snippet or full file) to a .c file.
    Wraps in a minimal compilable scaffold if it looks like a bare function.
    #TODO: add supplementary_code
    """

    def strip_fences(code: str) -> str:
        stripped = code.strip()

        # strip markdown code fences if present (AutoPatch LLM outputs)
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()
        return stripped

    main_code = strip_fences(source_code)
    supp_code = strip_fences(supplementary_code) if supplementary_code else ""

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

    # use newFrontend for better C++ support, but it can be much slower and more memory-hungry than the default frontend
    parse_cmd = [
        str(joern_bin / "joern-parse"),
        str(source),
        "--output",
        str(cpg_file),
        "--language",
        "newc",
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
