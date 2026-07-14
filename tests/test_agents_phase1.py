"""
Offline tests for the agent architecture seam (Phase 1).

Covers the agent registry, the SingleTurnAgent, and — critically — byte-for-byte
prompt parity between the new SingleTurnAgent and the original patch_one, so the
refactor of the batch path is provably behaviour-preserving. Everything runs via
MockBackend; no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from src.agents.architectures.single_turn import SingleTurnAgent
from src.agents.backends import MockBackend
from src.agents.base import AgentResult, PatchContext
from src.agents.graph_context import serialize_graph_context
from src.agents.patcher import patch_one
from src.agents.registry import build_agent, list_agents, register_agent

# ── minimal, realistic inputs ────────────────────────────────────────

EXAMPLE_DB = {
    "cve_id": "CVE-2024-0001",
    "cwe_type": "CWE-787",
    "root_cause": "missing bounds check before memcpy",
    "original_code": "int f(char *d, char *s, int n) { memcpy(d, s, n); return 0; }",
    "vuln_patch": "int f(char *d, char *s, int n) { if (n > MAX) return -1; memcpy(d, s, n); return 0; }",
    "fix_list": "Validate n before memcpy.",
}
TARGET_DB = {
    "cve_id": "CVE-2025-9999",
    "cwe_type": "CWE-787",
    "root_cause": "unchecked length in copy",
}
TARGET_CODE = "int g(char *d, char *s, int n) { memcpy(d, s, n); return 0; }"


def _graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node(
        "n1",
        CODE="memcpy(dst, src, len)",
        LINE_NUMBER=10,
        labelV="CALL",
        diff="removed",
    )
    g.add_node(
        "n2",
        CODE="if (len > MAX)",
        LINE_NUMBER=9,
        labelV="CONTROL_STRUCTURE",
        diff="fix_adjacent",
    )
    return g


# ── registry ─────────────────────────────────────────────────────────


def test_single_turn_registered():
    assert "single_turn" in list_agents()


def test_build_agent_unknown_raises():
    with pytest.raises(ValueError):
        build_agent("no_such_architecture")


def test_build_agent_ignores_bespoke_kwargs():
    # Extra kwargs (tool providers, step recipes) must be forwarded harmlessly.
    agent = build_agent(
        "single_turn",
        prompt_variant="default",
        backend=MockBackend.from_patch("r", "p"),
        tools={"provider": "none"},
        max_tool_iters=6,
    )
    assert isinstance(agent, SingleTurnAgent)


def test_register_agent_roundtrip():
    register_agent(
        "dummy_arch",
        lambda **kw: SingleTurnAgent(
            **{
                k: v
                for k, v in kw.items()
                if k in {"prompt_variant", "model_name", "llm_params", "backend"}
            }
        ),
    )
    assert "dummy_arch" in list_agents()


# ── SingleTurnAgent behaviour ────────────────────────────────────────


def test_single_turn_run_offline():
    mock = MockBackend.from_patch("reason about bounds", "int g(){ /*fixed*/ }")
    agent = SingleTurnAgent(prompt_variant="default", backend=mock)
    res = agent.run(PatchContext(EXAMPLE_DB, TARGET_DB, TARGET_CODE))

    assert isinstance(res, AgentResult)
    assert res.architecture == "single_turn"
    assert res.prompt_variant == "default"
    assert res.parsed is not None
    assert res.parsed.vuln_patch == "int g(){ /*fixed*/ }"
    assert res.n_turns == 1
    assert res.n_tool_calls == 0
    assert res.prompt_fingerprint  # non-empty content hash
    assert res.prompt_tokens > 0


def test_single_turn_prompt_parity_with_patch_one():
    """The agent must send byte-identical prompt messages to patch_one."""
    mock_agent = MockBackend.from_patch("r", "p")
    agent = SingleTurnAgent(prompt_variant="default", backend=mock_agent)
    res = agent.run(PatchContext(EXAMPLE_DB, TARGET_DB, TARGET_CODE))

    mock_po = MockBackend.from_patch("r", "p")
    raw, parsed, _ = patch_one(
        example_db=EXAMPLE_DB,
        target_db=TARGET_DB,
        target_code=TARGET_CODE,
        prompt_variant="default",
        backend=mock_po,
    )

    assert mock_agent.calls == mock_po.calls  # identical messages
    assert res.raw_output == raw
    assert res.parsed.vuln_patch == parsed.vuln_patch


def test_graph_variant_serializes_context_into_prompt():
    g = _graph()
    mock_agent = MockBackend.from_patch("r", "p")
    agent = SingleTurnAgent(prompt_variant="graph", backend=mock_agent)
    agent.run(
        PatchContext(
            EXAMPLE_DB,
            TARGET_DB,
            TARGET_CODE,
            query_pair=SimpleNamespace(G_vuln=g),
        )
    )
    last_user_msg = mock_agent.calls[0][-1]["content"]
    assert "memcpy(dst, src, len)" in last_user_msg  # graph context injected


def test_graph_variant_parity_with_explicit_context():
    """On-demand serialization must equal passing graph_context to patch_one."""
    g = _graph()
    gc = serialize_graph_context(g)

    mock_agent = MockBackend.from_patch("r", "p")
    agent = SingleTurnAgent(prompt_variant="graph", backend=mock_agent)
    agent.run(
        PatchContext(
            EXAMPLE_DB, TARGET_DB, TARGET_CODE, query_pair=SimpleNamespace(G_vuln=g)
        )
    )

    mock_po = MockBackend.from_patch("r", "p")
    patch_one(
        example_db=EXAMPLE_DB,
        target_db=TARGET_DB,
        target_code=TARGET_CODE,
        prompt_variant="graph",
        graph_context=gc,
        backend=mock_po,
    )
    assert mock_agent.calls == mock_po.calls


def test_default_variant_has_no_graph_context():
    # default must NOT serialize graph context even if a graph is present.
    g = _graph()
    mock = MockBackend.from_patch("r", "p")
    agent = SingleTurnAgent(prompt_variant="default", backend=mock)
    agent.run(
        PatchContext(
            EXAMPLE_DB, TARGET_DB, TARGET_CODE, query_pair=SimpleNamespace(G_vuln=g)
        )
    )
    joined = "\n".join(m["content"] for m in mock.calls[0])
    assert "memcpy(dst, src, len)" not in joined
