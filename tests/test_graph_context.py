"""Tests for src.agents.graph_context — graph-to-text serialization."""

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph_context import (
    format_control_deps,
    format_data_flow,
    format_diff,
    serialize_graph_context,
)


# ── test fixtures ────────────────────────────────────────────────────


def _make_graph() -> nx.MultiDiGraph:
    """Build a small G_vuln-like graph for testing.

    Simulates a buffer-overflow vulnerability:
      L10: void process(char *input)  [METHOD]       context
      L12: int len = strlen(input)    [CALL]          fix_adjacent
      L13: if (len > MAX)             [CONTROL_STRUCTURE] removed
      L14: memcpy(dest, input, len)   [CALL]          fix_adjacent
      L16: return                     [RETURN]         context

    Edges:
      L12 --REACHING_DEF--> L13  (len flows to condition)
      L12 --REACHING_DEF--> L14  (len flows to memcpy arg)
      L13 --CDG--> L14           (memcpy guarded by if)
      L10 --CFG--> L12           (control flow)
    """
    G = nx.MultiDiGraph()

    G.add_node(10, CODE="void process(char *input)", LINE_NUMBER=10,
               labelV="METHOD", diff="context", diff_weight=0.2)
    G.add_node(12, CODE="int len = strlen(input)", LINE_NUMBER=12,
               labelV="CALL", diff="fix_adjacent", diff_weight=0.8)
    G.add_node(13, CODE="if (len > MAX)", LINE_NUMBER=13,
               labelV="CONTROL_STRUCTURE", diff="removed", diff_weight=1.0)
    G.add_node(14, CODE="memcpy(dest, input, len)", LINE_NUMBER=14,
               labelV="CALL", diff="fix_adjacent", diff_weight=0.8)
    G.add_node(16, CODE="return", LINE_NUMBER=16,
               labelV="RETURN", diff="context", diff_weight=0.2)

    G.add_edge(12, 13, labelE="REACHING_DEF")
    G.add_edge(12, 14, labelE="REACHING_DEF")
    G.add_edge(13, 14, labelE="CDG")
    G.add_edge(10, 12, labelE="CFG")

    return G


# ── format_diff ──────────────────────────────────────────────────────


class TestFormatDiff:

    def test_groups_by_category(self):
        G = _make_graph()
        result = format_diff(G)
        # REMOVED section should appear before FIX_ADJACENT
        removed_pos = result.index("REMOVED")
        adjacent_pos = result.index("FIX_ADJACENT")
        context_pos = result.index("CONTEXT")
        assert removed_pos < adjacent_pos < context_pos

    def test_contains_all_code_lines(self):
        G = _make_graph()
        result = format_diff(G)
        assert "if (len > MAX)" in result
        assert "memcpy(dest, input, len)" in result
        assert "void process(char *input)" in result
        assert "return" in result

    def test_shows_line_numbers(self):
        G = _make_graph()
        result = format_diff(G)
        assert "L13:" in result
        assert "L14:" in result

    def test_shows_label_types(self):
        G = _make_graph()
        result = format_diff(G)
        assert "[CONTROL_STRUCTURE]" in result
        assert "[METHOD]" in result

    def test_empty_graph(self):
        G = nx.MultiDiGraph()
        result = format_diff(G)
        assert "No diff" in result

    def test_skips_nodes_without_code(self):
        G = nx.MultiDiGraph()
        G.add_node(1, LINE_NUMBER=1, labelV="BLOCK", diff="context", diff_weight=0.2)
        G.add_node(2, CODE="x = 1", LINE_NUMBER=2, labelV="LOCAL",
                   diff="removed", diff_weight=1.0)
        result = format_diff(G)
        assert "x = 1" in result
        assert "BLOCK" not in result


# ── format_data_flow ─────────────────────────────────────────────────


class TestFormatDataFlow:

    def test_shows_reaching_def_edges(self):
        G = _make_graph()
        result = format_data_flow(G)
        assert "REACHING_DEF" in result
        assert "strlen" in result
        assert "memcpy" in result

    def test_shows_chain_from_changed_nodes(self):
        G = _make_graph()
        result = format_data_flow(G)
        # L12 (fix_adjacent) should show its outgoing REACHING_DEF to L13 and L14
        assert "L12" in result
        assert "L13" in result or "L14" in result

    def test_no_seed_nodes(self):
        G = nx.MultiDiGraph()
        G.add_node(1, CODE="x", LINE_NUMBER=1, labelV="LOCAL",
                   diff="context", diff_weight=0.2)
        result = format_data_flow(G)
        assert "No data-flow" in result

    def test_no_reaching_def_edges(self):
        G = nx.MultiDiGraph()
        G.add_node(1, CODE="x", LINE_NUMBER=1, labelV="LOCAL",
                   diff="removed", diff_weight=1.0)
        G.add_node(2, CODE="y", LINE_NUMBER=2, labelV="LOCAL",
                   diff="context", diff_weight=0.2)
        G.add_edge(1, 2, labelE="CFG")  # not REACHING_DEF
        result = format_data_flow(G)
        assert "No data-flow edges" in result


# ── format_control_deps ──────────────────────────────────────────────


class TestFormatControlDeps:

    def test_shows_cdg_edges(self):
        G = _make_graph()
        result = format_control_deps(G)
        assert "control-dependent on" in result or "controls:" in result
        assert "if (len > MAX)" in result
        assert "memcpy" in result

    def test_no_seed_nodes(self):
        G = nx.MultiDiGraph()
        G.add_node(1, CODE="x", LINE_NUMBER=1, labelV="LOCAL",
                   diff="context", diff_weight=0.2)
        result = format_control_deps(G)
        assert "No control-dependency" in result

    def test_no_cdg_edges(self):
        G = nx.MultiDiGraph()
        G.add_node(1, CODE="x", LINE_NUMBER=1, labelV="LOCAL",
                   diff="removed", diff_weight=1.0)
        G.add_node(2, CODE="y", LINE_NUMBER=2, labelV="LOCAL",
                   diff="context", diff_weight=0.2)
        G.add_edge(1, 2, labelE="CFG")
        result = format_control_deps(G)
        assert "No control-dependency edges" in result


# ── serialize_graph_context ──────────────────────────────────────────


class TestSerializeGraphContext:

    def test_none_graph(self):
        assert serialize_graph_context(None) == "None"

    def test_empty_graph(self):
        assert serialize_graph_context(nx.MultiDiGraph()) == "None"

    def test_contains_all_sections(self):
        G = _make_graph()
        result = serialize_graph_context(G)
        assert "[Patch Diff]" in result
        assert "[Data Flow]" in result
        assert "[Control Dependencies]" in result

    def test_sections_have_content(self):
        G = _make_graph()
        result = serialize_graph_context(G)
        assert "REMOVED" in result
        assert "REACHING_DEF" in result
        assert "memcpy" in result
