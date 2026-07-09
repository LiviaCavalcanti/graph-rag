"""
Agent architecture registry.

Maps an ``architecture`` name to a builder so the batch runner and experiments
can select a patching architecture by string (grid axis / CLI flag / config),
mirroring the embedder registry in :mod:`src.embeddings`.

New architectures (multi-step, tool-using, MCP) register themselves here in
later phases; ``build_agent`` forwards bespoke keyword args (tool providers,
step recipes) untouched so this signature stays stable.
"""

from __future__ import annotations

from typing import Callable

from src.agents.backends import CompletionBackend
from src.agents.base import Agent

# name → factory(*, prompt_variant, model_name, llm_params, backend, **extra) -> Agent
_BUILDERS: dict[str, Callable[..., Agent]] = {}


def register_agent(name: str, builder: Callable[..., Agent]) -> None:
    """Register an architecture builder under *name* (overwrites if present)."""
    _BUILDERS[name] = builder


def list_agents() -> list[str]:
    return sorted(_BUILDERS)


def build_agent(
    architecture: str,
    *,
    prompt_variant: str = "default",
    model_name: str | None = None,
    llm_params: dict | None = None,
    backend: CompletionBackend | None = None,
    **extra,
) -> Agent:
    """Construct an agent by architecture name.

    Extra keyword args are forwarded to the builder so new architectures can
    take bespoke config without changing this signature.
    """
    if architecture not in _BUILDERS:
        raise ValueError(
            f"Unknown architecture {architecture!r}. Available: {list_agents()}"
        )
    return _BUILDERS[architecture](
        prompt_variant=prompt_variant,
        model_name=model_name,
        llm_params=llm_params,
        backend=backend,
        **extra,
    )


def _register_builtins() -> None:
    from src.agents.architectures.single_turn import SingleTurnAgent

    def _single_turn(*, prompt_variant, model_name, llm_params, backend, **_):
        return SingleTurnAgent(
            prompt_variant=prompt_variant,
            model_name=model_name,
            llm_params=llm_params,
            backend=backend,
        )

    register_agent("single_turn", _single_turn)


_register_builtins()
