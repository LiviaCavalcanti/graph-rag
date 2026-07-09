"""
Agent harness — a self-contained, offline-first toolkit to **debug, test,
evolve, and evaluate** the vulnerability-patching agent.

Layers
------
- backends (``src.agents.backends``): pluggable LLM backends (live / mock /
  replay / record) + ``use_backend`` for process-wide overrides. This is what
  makes every command below runnable with no network.
- scenario: the unit of work — a self-contained agent test case (JSON fixture).
- registry: inspect / diff / render / register prompt variants (evolve).
- evaluator: one unified, offline scorecard (metrics + golden expectations +
  optional LLM judge) (evaluate).
- runner: run scenarios, capture Traces, pretty-print & replay (debug/test).
- compare: N-way prompt-variant comparison with deltas (evolve).

CLI
---
``python -m src.agents.harness <variants|debug|record|replay|eval|compare>``

Quick start (offline)
---------------------
    from src.agents.backends import MockBackend
    from src.agents.harness import AgentHarness, load_scenarios, format_trace

    scns = load_scenarios("tests/fixtures/agent/scenarios.json")
    backend = MockBackend.from_patch("reasoning", scns[0].ground_truth)
    trace = AgentHarness(prompt_variant="default").run_scenario(scns[0], backend=backend)
    print(format_trace(trace))
"""

from src.agents.harness.compare import ComparisonReport, compare_variants
from src.agents.harness.evaluator import AgentEvaluator, ScoreCard
from src.agents.harness.registry import (
    describe,
    diff_variants,
    list_variants,
    register_variant,
    render_messages,
)
from src.agents.harness.runner import (
    AgentHarness,
    SuiteResult,
    Trace,
    format_trace,
    replay_record,
    save_traces,
)
from src.agents.harness.scenario import Scenario, load_scenarios, save_scenarios

__all__ = [
    # scenario
    "Scenario",
    "load_scenarios",
    "save_scenarios",
    # registry (evolve)
    "list_variants",
    "describe",
    "diff_variants",
    "render_messages",
    "register_variant",
    # evaluate
    "AgentEvaluator",
    "ScoreCard",
    # runner (debug/test)
    "AgentHarness",
    "Trace",
    "SuiteResult",
    "format_trace",
    "replay_record",
    "save_traces",
    # compare (evolve)
    "compare_variants",
    "ComparisonReport",
]
