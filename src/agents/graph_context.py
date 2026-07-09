"""
Serialize a G_vuln MultiDiGraph into structured text for LLM prompts.

Produces three sections:
  [Vulnerability Diff]   — diff-annotated code lines grouped by change type
  [Data Flow]            — REACHING_DEF chains from changed nodes
  [Control Dependencies] — CDG edges showing guard conditions
"""

from __future__ import annotations

import networkx as nx

# Diff categories in display order (most important first)
_DIFF_ORDER = ["removed", "fix_adjacent", "edge_changed", "context"]

_DIFF_LABELS = {
    "removed": "REMOVED (patch deleted these)",
    "fix_adjacent": "FIX_ADJACENT (near patch insertion points)",
    "edge_changed": "EDGE_CHANGED (data/control flow changed by patch)",
    "context": "CONTEXT (surrounding code in program slice)",
}


def _node_label(G: nx.MultiDiGraph, n) -> str:
    """Compact one-line label for a node: 'code (Lnn) [TYPE]'."""
    attrs = G.nodes[n]
    code = str(attrs.get("CODE", "")).strip()
    line = attrs.get("LINE_NUMBER")
    lv = attrs.get("labelV", "")
    parts = []
    if code:
        parts.append(code)
    if line is not None:
        parts.append(f"L{line}")
    if lv:
        parts.append(f"[{lv}]")
    return " ".join(parts) if parts else str(n)


# ── section builders ────────────────────────────────────────────────


def format_diff(G: nx.MultiDiGraph) -> str:
    """Diff-annotated code lines grouped by change type."""
    by_cat: dict[str, list[dict]] = {cat: [] for cat in _DIFF_ORDER}

    for n, attrs in G.nodes(data=True):
        code = str(attrs.get("CODE", "")).strip()
        if not code:
            continue
        cat = attrs.get("diff", "context")
        if cat not in by_cat:
            cat = "context"
        # ── noise filters ──
        line = attrs.get("LINE_NUMBER")
        lv = attrs.get("labelV", "")
        # Skip generic parameter stubs without line numbers
        if line is None and lv in (
            "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD", "BLOCK",
        ):
            continue
        # Skip full function body nodes (redundant with [Target Code])
        if lv in ("METHOD", "BLOCK") and "\n" in code:
            continue
        by_cat[cat].append({
            "line": line,
            "code": code,
            "labelV": lv,
        })

    # Sort each category by line number
    for entries in by_cat.values():
        entries.sort(key=lambda e: (e["line"] or 0, e["code"]))

    lines: list[str] = []
    for cat in _DIFF_ORDER:
        entries = by_cat[cat]
        if not entries:
            continue
        lines.append(f"{_DIFF_LABELS[cat]}:")
        for e in entries:
            ln = f"L{e['line']}" if e["line"] is not None else "L?"
            lines.append(f"  {ln}: {e['code']}   [{e['labelV']}]")
    return "\n".join(lines) if lines else "No diff information available."


def format_data_flow(G: nx.MultiDiGraph) -> str:
    """REACHING_DEF chains starting from changed nodes."""
    seed_nodes = {
        n for n, attrs in G.nodes(data=True)
        if attrs.get("diff") in ("removed", "fix_adjacent", "edge_changed")
    }
    if not seed_nodes:
        return "No data-flow information available."

    seen_edges: set[tuple] = set()
    chains: list[str] = []

    for n in sorted(seed_nodes, key=lambda x: G.nodes[x].get("LINE_NUMBER") or 0):
        # Walk outgoing REACHING_DEF edges (up to 3 hops)
        frontier = [n]
        visited = {n}
        for _ in range(3):
            next_frontier = []
            for cur in frontier:
                for _, tgt, d in G.out_edges(cur, data=True):
                    edge_type = d.get("labelE") or d.get("label", "")
                    if edge_type != "REACHING_DEF":
                        continue
                    edge_key = (cur, tgt)
                    if edge_key in seen_edges:
                        continue
                    # Skip edges between nodes without line numbers (noise)
                    src_line = G.nodes[cur].get("LINE_NUMBER")
                    tgt_line = G.nodes[tgt].get("LINE_NUMBER")
                    if src_line is None and tgt_line is None:
                        continue
                    seen_edges.add(edge_key)
                    chains.append(
                        f"  {_node_label(G, cur)} ──REACHING_DEF──▸ {_node_label(G, tgt)}"
                    )
                    if tgt not in visited:
                        visited.add(tgt)
                        next_frontier.append(tgt)
            frontier = next_frontier

        # Walk incoming REACHING_DEF edges (1 hop — show where data comes from)
        for src, _, d in G.in_edges(n, data=True):
            edge_type = d.get("labelE") or d.get("label", "")
            if edge_type != "REACHING_DEF":
                continue
            edge_key = (src, n)
            if edge_key in seen_edges:
                continue
            # Skip edges between nodes without line numbers (noise)
            src_line = G.nodes[src].get("LINE_NUMBER")
            n_line = G.nodes[n].get("LINE_NUMBER")
            if src_line is None and n_line is None:
                continue
            seen_edges.add(edge_key)
            chains.append(
                f"  {_node_label(G, src)} ──REACHING_DEF──▸ {_node_label(G, n)}"
            )

    return "\n".join(chains) if chains else "No data-flow edges in the program slice."


def format_control_deps(G: nx.MultiDiGraph) -> str:
    """CDG edges showing what conditions guard the vulnerable code."""
    seed_nodes = {
        n for n, attrs in G.nodes(data=True)
        if attrs.get("diff") in ("removed", "fix_adjacent", "edge_changed")
    }
    if not seed_nodes:
        return "No control-dependency information available."

    seen: set[tuple] = set()
    deps: list[str] = []

    for n in sorted(seed_nodes, key=lambda x: G.nodes[x].get("LINE_NUMBER") or 0):
        # Incoming CDG: what controls this node
        for src, _, d in G.in_edges(n, data=True):
            edge_type = d.get("labelE") or d.get("label", "")
            if edge_type != "CDG":
                continue
            key = (src, n)
            if key in seen:
                continue
            seen.add(key)
            deps.append(
                f"  {_node_label(G, n)} is control-dependent on: {_node_label(G, src)}"
            )
        # Outgoing CDG: what this node controls
        for _, tgt, d in G.out_edges(n, data=True):
            edge_type = d.get("labelE") or d.get("label", "")
            if edge_type != "CDG":
                continue
            key = (n, tgt)
            if key in seen:
                continue
            seen.add(key)
            deps.append(
                f"  {_node_label(G, n)} controls: {_node_label(G, tgt)}"
            )

    return "\n".join(deps) if deps else "No control-dependency edges in the program slice."


# ── main entry point ────────────────────────────────────────────────


def serialize_graph_context(G_vuln: nx.MultiDiGraph | None) -> str:
    """Serialize G_vuln into the three prompt sections.

    Returns a single string with [Patch Diff], [Data Flow],
    and [Control Dependencies] sections ready to paste into a prompt.

    Returns "None" if G_vuln is None or empty (keeps prompt clean).
    """
    if G_vuln is None or len(G_vuln) == 0:
        return "None"

    sections = [
        f"[Patch Diff]\n{format_diff(G_vuln)}",
        f"[Data Flow]\n{format_data_flow(G_vuln)}",
        f"[Control Dependencies]\n{format_control_deps(G_vuln)}",
    ]
    return "\n\n".join(sections)
