"""
Single-turn patcher agent — one LLM call, few-shot prompt, marker-parsed patch.

This expresses the original ``patch_one`` behaviour as an :class:`Agent`, so it
can be selected by name (``architecture: single_turn``) and compared head to
head with multi-step / tool-using architectures under identical tracking.
"""

from __future__ import annotations

from src.agents import prompt_registry
from src.agents.backends import CompletionBackend
from src.agents.base import AgentResult, PatchContext
from src.agents.graph_context import serialize_graph_context
from src.agents.patcher import AutoPatchPatcher, build_input_dict


class SingleTurnAgent:
    """One-shot patch generation (registry key: ``single_turn``)."""

    name = "single_turn"

    def __init__(
        self,
        *,
        prompt_variant: str = "default",
        model_name: str | None = None,
        llm_params: dict | None = None,
        backend: CompletionBackend | None = None,
    ):
        self.prompt_variant = prompt_variant
        self.model_name = model_name
        self.llm_params = llm_params or {}
        self.backend = backend

    def run(self, ctx: PatchContext) -> AgentResult:
        # Serialize graph context on demand when the variant asks for it and the
        # caller hasn't already supplied it. Preserves the historical behaviour
        # (graph variants inject a serialized G_vuln) while keeping the batch
        # runner architecture-agnostic.
        graph_context = ctx.graph_context
        if not graph_context and prompt_registry.needs_graph_context(
            self.prompt_variant
        ):
            graph_context = serialize_graph_context(
                getattr(ctx.query_pair, "G_vuln", None)
            )

        input_dict = build_input_dict(
            ctx.example_db,
            ctx.target_db,
            ctx.target_code,
            ctx.target_supplementary,
            graph_context,
        )
        patcher = AutoPatchPatcher(
            self.model_name,
            prompt_variant=self.prompt_variant,
            backend=self.backend,
            **self.llm_params,
        )
        record = patcher.invoke(input_dict)
        return AgentResult(
            architecture=self.name,
            prompt_variant=self.prompt_variant,
            raw_output=record.raw_output,
            parsed=record.parsed,
            turns=[record],
            prompt_fingerprint=_safe_fingerprint(self.prompt_variant),
        )


def _safe_fingerprint(variant: str) -> str:
    """Fingerprint the variant, falling back to its live templates.

    Variants registered at runtime (via the harness) are absent from the
    declarative manifest; fingerprint their in-memory templates instead.
    """
    try:
        return prompt_registry.variant_fingerprint(variant)
    except KeyError:
        from src.agents import patcher as _patcher

        tmpls = _patcher._VARIANTS.get(variant, [])
        return prompt_registry.fingerprint(tmpls) if tmpls else ""
