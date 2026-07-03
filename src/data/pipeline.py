import glob
import logging
import re
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


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


def _cache_path_for(graph_dir: str) -> Path:
    """Return the pickle cache path for a graph directory."""
    return Path(graph_dir).parent / ".graph_cache.pkl"


def load_cpg_dir(graph_dir: str, use_cache: bool = True) -> nx.MultiDiGraph:
    """Load a CPG from export.xml files, using pickle cache when available.

    First run parses all XML files and saves a .graph_cache.pkl alongside them.
    Subsequent runs load the pickle directly (~100x faster).
    """
    import pickle

    cache_file = _cache_path_for(graph_dir)

    # Try loading from cache first
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass  # Fall through to full parse

    # Full XML parse
    root = Path(graph_dir)
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
    G.remove_edges_from(dangling)

    logger.info(
        f"Declared nodes {declared_nodes}. Phantom nodes: {phantom_nodes}. Prunned dangling edges after removing phantoms: {dangling}"
    )

    # Save to cache for next time
    if use_cache:
        try:
            with open(cache_file, "wb") as fh:
                pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Could not write graph cache {cache_file}: {e}")

    return G


def load_cpg_dir_safe(graph_dir: str) -> tuple[str, nx.MultiDiGraph | None]:
    """Load CPG and return (graph_dir, graph) or (graph_dir, None) on failure."""
    try:
        return (graph_dir, load_cpg_dir(graph_dir))
    except Exception as e:
        logger.warning(f"Failed to load {graph_dir}: {e}")
        return (graph_dir, None)


def load_cpg_dirs_parallel(
    graph_dirs: list[str], max_workers: int = 8
) -> dict[str, nx.MultiDiGraph]:
    """Load multiple CPG directories in parallel using ThreadPoolExecutor.

    ThreadPoolExecutor is ideal for I/O-bound operations like file loading.

    Args:
        graph_dirs: List of graph directory paths
        max_workers: Number of parallel workers (default 8 for I/O)

    Returns:
        Dict mapping graph_dir -> loaded graph (skips failed paths)
    """
    graphs = {}
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_cpg_dir_safe, gd): gd for gd in graph_dirs}

        from tqdm import tqdm as tqdm_module

        for future in tqdm_module(
            as_completed(futures), total=len(futures), desc="Loading graphs (parallel)"
        ):
            graph_dir, graph = future.result()
            if graph is not None:
                graphs[graph_dir] = graph
            else:
                failed.append(graph_dir)

    if failed:
        logger.warning(f"Failed to load {len(failed)} graphs: {failed[:5]}...")

    return graphs


# ── graph-diff defaults ──────────────────────────────────────────────
# Single source of truth for compute_graph_diff parameters.  Override per
# run via the config `graph:` section (see graph_diff_params()).  Defaults
# preserve the historical hard-coded behavior.
DEFAULT_SLICE_DEPTH = 3  # hops along flow edges from seed nodes
DEFAULT_CHANGE_WEIGHT = {
    "removed": 1.0,
    "fix_adjacent": 0.8,
    "edge_changed": 0.6,
    "context": 0.2,
}
DEFAULT_NOISE_TYPES = {
    "TYPE_DECL",
    "FILE",
    "NAMESPACE_BLOCK",
    "COMMENT",
    "UNKNOWN",
    "METHOD_RETURN",
}
DEFAULT_FLOW_EDGES = {"CFG", "CDG", "REACHING_DEF", "PDG", "DDG"}

# Backwards-compatible alias (was the only module-level constant before the
# parameters became configurable).
CHANGE_WEIGHT = DEFAULT_CHANGE_WEIGHT


def graph_diff_params(cfg: dict | None) -> dict:
    """Extract compute_graph_diff() overrides from a config mapping.

    Accepts either a full config dict (reads its ``graph`` section) or a
    graph-section dict directly.  Only keys explicitly present are returned,
    so anything omitted falls back to the ``DEFAULT_*`` values above — i.e.
    passing an empty or foreign config is behavior-preserving.
    """
    if not cfg:
        return {}
    section = cfg.get("graph", cfg) if isinstance(cfg, dict) else cfg
    if not isinstance(section, dict):
        return {}
    params: dict = {}
    for key in ("slice_depth", "change_weight", "noise_types", "flow_edges"):
        if section.get(key) is not None:
            params[key] = section[key]
    return params


def compute_graph_diff(
    G_before: nx.MultiDiGraph,
    G_after: nx.MultiDiGraph,
    *,
    slice_depth: int | None = None,
    change_weight: dict | None = None,
    noise_types=None,
    flow_edges=None,
) -> nx.MultiDiGraph:
    """
    Semantic graph diff + vulnerability-aware program slice.

    Matches nodes by (labelV, CODE, LINE_NUMBER) instead of node ID
    so that ID renumbering between before/after does not produce false
    changes.  Extracts a bounded program slice by following CFG,
    REACHING_DEF and CDG edges from the truly changed nodes.
    """
    from collections import Counter

    # ── resolve parameters (defaults preserve prior behavior) ────
    slice_depth = DEFAULT_SLICE_DEPTH if slice_depth is None else int(slice_depth)
    change_weight = DEFAULT_CHANGE_WEIGHT if change_weight is None else change_weight
    noise_types = DEFAULT_NOISE_TYPES if noise_types is None else set(noise_types)
    flow_edges = DEFAULT_FLOW_EDGES if flow_edges is None else set(flow_edges)

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
        return G.nodes[n].get("labelV") not in noise_types

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

    for _ in range(slice_depth):
        next_frontier = set()
        for n in frontier:
            if n not in G_before:
                continue
            for _, tgt, d in G_before.out_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in flow_edges and tgt not in slice_nodes:
                    next_frontier.add(tgt)
            for src, _, d in G_before.in_edges(n, data=True):
                el = d.get("labelE") or d.get("label", "")
                if el in flow_edges and src not in slice_nodes:
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
        G_vuln.nodes[n]["diff_weight"] = change_weight.get(dlabel, 0.2)

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


DEFAULT_JOERN_TIMEOUT_S = 120
DEFAULT_JOERN_LANGUAGE = "newc"


def run_joern_export(
    joern_bir_dir: str,
    source_file: str,
    out_dir: str,
    graph_dir: str,
    *,
    timeout_s: int = DEFAULT_JOERN_TIMEOUT_S,
    language: str = DEFAULT_JOERN_LANGUAGE,
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
        language,
    ]

    result = subprocess.run(
        parse_cmd, capture_output=True, text=True, timeout=timeout_s
    )
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
        timeout=timeout_s,
    )
    if result.returncode != 0:
        print(result)
        print(f"EXPORT ERROR: {result.stderr}")
    return result.returncode == 0
