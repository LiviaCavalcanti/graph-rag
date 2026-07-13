"""
Agent abstraction — the "architecture" seam.

An :class:`Agent` turns a :class:`PatchContext` (the retrieved example plus the
target to patch) into an :class:`AgentResult` (the parsed patch plus a full,
reproducible trace of every LLM turn and tool call). This generalizes the
original single-turn ``patch_one`` so that multi-step, tool-using, and
MCP-based architectures can be swapped in behind one interface and tracked
identically.

The LLM transport stays behind :class:`~src.agents.backends.CompletionBackend`;
this layer only decides *how many* calls to make and *what* to put in them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.agents.patcher import InvocationRecord, PatchResult


@dataclass
class PatchContext:
    """Everything an agent needs to generate one patch.

    Assembled by the batch runner from the retrieved example, the target, and
    (optionally) the query ``FunctionPair`` so tool-using agents can reach its
    CPGs (``query_pair.G_vuln`` etc.).
    """

    example_db: dict
    target_db: dict
    target_code: str
    target_supplementary: str = ""
    graph_context: str = ""
    query_pair: Any = None


@dataclass
class AgentResult:
    """Structured, reproducible output of one agent run.

    ``turns`` holds one :class:`InvocationRecord` per LLM call (single-turn
    agents have exactly one); ``tool_calls`` logs any tool invocations. Token
    counts and ``finish_reason`` aggregate across turns so downstream code that
    previously read a single record keeps working.
    """

    architecture: str
    prompt_variant: str
    raw_output: str = ""
    parsed: PatchResult | None = None
    turns: list[InvocationRecord] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    prompt_fingerprint: str = ""
    error: str | None = None

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens for t in self.turns)

    @property
    def finish_reason(self) -> str:
        return self.turns[-1].finish_reason if self.turns else ""


@runtime_checkable
class Agent(Protocol):
    """A patch-generating architecture.

    Implementations are registered in :mod:`src.agents.registry` and built by
    name so experiments can vary ``architecture`` as a grid axis.
    """

    name: str

    def run(self, ctx: PatchContext) -> AgentResult: ...
