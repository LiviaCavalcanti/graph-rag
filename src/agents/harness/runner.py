"""
AgentHarness — run scenarios, capture traces, and pretty-print for debugging.

This is the orchestration hub the CLI drives. It runs a Scenario through the
real ``patch_one`` code path (with any backend injected), scores the result
with :class:`~src.agents.harness.evaluator.AgentEvaluator`, and packages
everything into a :class:`Trace` you can print, persist, and later replay.

Everything here is backend-agnostic: pass a MockBackend for offline unit runs,
a ReplayBackend for deterministic regression, or a LiveBackend for real calls.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.agents.backends import CompletionBackend, use_backend
from src.agents.harness.evaluator import AgentEvaluator, ScoreCard
from src.agents.harness.scenario import Scenario


@dataclass
class Trace:
    """A single scenario run: inputs, the LLM record, and the scorecard."""

    scenario_id: str
    status: str  # success | parse_error | error
    coords: dict[str, Any] = field(default_factory=dict)  # variant, model
    generated_patch: str = ""
    cached: bool = False
    elapsed_s: float = 0.0
    error: str | None = None
    record: dict[str, Any] = field(
        default_factory=dict
    )  # InvocationRecord.model_dump()
    score: dict[str, Any] = field(default_factory=dict)  # ScoreCard.to_dict()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SuiteResult:
    """Aggregate over a set of traces from one suite run."""

    traces: list[Trace]
    summary: dict[str, Any]
    coords: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "coords": self.coords,
            "summary": self.summary,
            "traces": [t.to_dict() for t in self.traces],
        }


class AgentHarness:
    """Drive scenarios through the agent with a chosen backend and evaluator."""

    def __init__(
        self,
        *,
        evaluator: AgentEvaluator | None = None,
        model_name: str | None = None,
        prompt_variant: str = "default",
        llm_params: dict | None = None,
    ):
        self.evaluator = evaluator or AgentEvaluator()
        self.model_name = model_name
        self.prompt_variant = prompt_variant
        self.llm_params = llm_params or {}

    # ── single scenario ──────────────────────────────────────────────

    def run_scenario(
        self,
        scenario: Scenario,
        *,
        backend: CompletionBackend | None = None,
        prompt_variant: str | None = None,
        model_name: str | None = None,
    ) -> Trace:
        from src.agents.patcher import patch_one

        variant = prompt_variant or self.prompt_variant
        model = model_name or self.model_name
        coords = {"variant": variant, "model": model}
        t0 = time.perf_counter()
        try:
            raw, parsed, record = patch_one(
                example_db=scenario.example_db,
                target_db=scenario.target_db,
                target_code=scenario.target_code,
                target_supplementary=scenario.target_supplementary,
                model_name=model,
                prompt_variant=variant,
                graph_context=scenario.graph_context,
                llm_params=self.llm_params,
                backend=backend,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a failed trace
            return Trace(
                scenario_id=scenario.id,
                status="error",
                coords=coords,
                elapsed_s=round(time.perf_counter() - t0, 3),
                error=str(exc),
            )

        generated = parsed.vuln_patch if parsed else ""
        card = self.evaluator.score_scenario(scenario, generated)
        return Trace(
            scenario_id=scenario.id,
            status="success" if parsed else "parse_error",
            coords=coords,
            generated_patch=generated,
            cached=record.cached,
            elapsed_s=record.elapsed_s,
            record=record.model_dump(),
            score=card.to_dict(),
        )

    # ── suite ────────────────────────────────────────────────────────

    def run_suite(
        self,
        scenarios: list[Scenario],
        *,
        backend: CompletionBackend | None = None,
        prompt_variant: str | None = None,
        model_name: str | None = None,
        progress: bool = False,
    ) -> SuiteResult:
        traces: list[Trace] = []
        # Install the backend once so nested patch_one calls all use it.
        ctx = use_backend(backend) if backend is not None else _null_ctx()
        with ctx:
            for i, scn in enumerate(scenarios, 1):
                if progress:
                    print(f"  [{i}/{len(scenarios)}] {scn.id} …", flush=True)
                traces.append(
                    self.run_scenario(
                        scn, prompt_variant=prompt_variant, model_name=model_name
                    )
                )
        cards = [ScoreCard(**t.score) for t in traces if t.score]
        summary = self.evaluator.aggregate(cards)
        summary["status_counts"] = _status_counts(traces)
        return SuiteResult(
            traces=traces,
            summary=summary,
            coords={
                "variant": prompt_variant or self.prompt_variant,
                "model": model_name or self.model_name,
            },
        )


# ── replay a saved InvocationRecord (pure, no LLM) ───────────────────


def replay_record(
    record: str | Path | dict,
    *,
    reference: str = "",
    expected: dict | None = None,
    evaluator: AgentEvaluator | None = None,
) -> ScoreCard:
    """Re-parse and re-score a previously saved InvocationRecord.

    Useful for debugging a past run offline: feed a saved trace JSON and see how
    the current parser + metrics score it, without any LLM call.
    """
    from src.agents.backends import MockBackend
    from src.agents.patcher import AutoPatchPatcher

    if isinstance(record, (str, Path)):
        record = json.loads(Path(record).read_text())

    raw_output = record.get("raw_output", "")
    parser = AutoPatchPatcher(backend=MockBackend(""))
    parsed = parser.parse(raw_output)
    generated = parsed.vuln_patch if parsed else ""
    ev = evaluator or AgentEvaluator()
    return ev.score(generated, reference, expected=expected)


# ── pretty printing (debug) ──────────────────────────────────────────


def format_trace(
    trace: Trace, *, max_msg_chars: int = 1600, max_out_chars: int = 2000
) -> str:
    """Human-readable rendering of a Trace for the debug CLI."""
    rec = trace.record
    score = trace.score
    lines: list[str] = []
    bar = "━" * 72
    coord = "  ".join(f"{k}={v}" for k, v in trace.coords.items())
    lines.append(bar)
    lines.append(f"Trace: {trace.scenario_id}   [{coord}  cached={trace.cached}]")
    lines.append(bar)
    tok = (
        f"{rec.get('prompt_tokens', 0)}/{rec.get('completion_tokens', 0)}/"
        f"{rec.get('total_tokens', 0)}"
        if rec
        else "-"
    )
    lines.append(
        f"status: {trace.status}   elapsed: {trace.elapsed_s}s   tokens(p/c/t): {tok}"
        f"   finish: {rec.get('finish_reason', '-') if rec else '-'}"
    )
    if trace.error:
        lines.append(f"error: {trace.error}")

    messages = rec.get("messages", []) if rec else []
    if messages:
        lines.append(f"\n── Prompt ({len(messages)} messages) " + "─" * 40)
        for m in messages:
            content = m.get("content", "")
            if len(content) > max_msg_chars:
                content = (
                    content[:max_msg_chars]
                    + f"\n… [+{len(content) - max_msg_chars} chars]"
                )
            lines.append(f"[{m.get('role', '?')}]\n{content}")

    raw = rec.get("raw_output", "") if rec else ""
    if raw:
        shown = (
            raw
            if len(raw) <= max_out_chars
            else raw[:max_out_chars] + f"\n… [+{len(raw) - max_out_chars} chars]"
        )
        lines.append(f"\n── Raw output ({len(raw)} chars) " + "─" * 44)
        lines.append(shown)

    if trace.generated_patch:
        lines.append("\n── Parsed patch " + "─" * 56)
        lines.append(trace.generated_patch)

    if score:
        lines.append("\n── Scores " + "─" * 62)
        lines.append(
            f"parsed={score.get('parsed')}  exact={score.get('exact_match')}  "
            f"similarity={score.get('similarity')}  bleu4={score.get('bleu_4')}  "
            f"rougeL={score.get('rouge_l_f1')}  jaccard={score.get('token_jaccard')}  "
            f"editdist={score.get('edit_distance_norm')}"
        )
        if score.get("expectations_checked"):
            status = "PASS" if score.get("expectations_passed") else "FAIL"
            lines.append(f"expectations: {status}")
            for f in score.get("expectation_failures", []):
                lines.append(f"   ✗ {f}")
        if score.get("judge_verdict"):
            lines.append(
                f"judge: {score['judge_verdict']} "
                f"(confidence={score.get('judge_confidence')})"
            )
    lines.append(bar)
    return "\n".join(lines)


def save_traces(traces: list[Trace], path: str | Path) -> Path:
    """Persist traces as JSONL for later inspection or replay."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), default=str) + "\n")
    return p


# ── internals ────────────────────────────────────────────────────────


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _status_counts(traces: list[Trace]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in traces:
        counts[t.status] = counts.get(t.status, 0) + 1
    return counts


__all__ = [
    "Trace",
    "SuiteResult",
    "AgentHarness",
    "replay_record",
    "format_trace",
    "save_traces",
]
