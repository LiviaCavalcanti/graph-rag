"""Concrete patching-agent architectures.

Importing this package makes the built-in architectures importable; they are
registered with the agent registry in :mod:`src.agents.registry`.
"""

from src.agents.architectures.single_turn import SingleTurnAgent

__all__ = ["SingleTurnAgent"]
