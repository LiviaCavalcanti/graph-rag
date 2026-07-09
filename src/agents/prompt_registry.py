"""
Declarative, versioned prompt-variant registry.

Prompt variants are DATA, not code: they live in ``prompts/registry.yaml`` as an
ordered list of chat messages (``role`` + template ``file``). This module loads
that manifest into the ``{name: [(role, template_text), ...]}`` shape that
``src.agents.patcher._VARIANTS`` and the harness already expect, and exposes
per-variant metadata (version, description, whether the variant needs serialized
graph context) plus a stable content fingerprint for reproducibility.

Adding / editing a prompt variant = edit ``registry.yaml`` (and the ``.txt``
files it points at). No Python changes required.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
REGISTRY_PATH: Path = PROMPTS_DIR / "registry.yaml"

Template = tuple[str, str]  # (role, template_text)
_VALID_ROLES = {"system", "user", "assistant"}


def fingerprint(templates: list[Template]) -> str:
    """Stable 12-char sha256 over a variant's ``(role, template)`` messages.

    Uses the same algorithm as the harness registry so both agree on IDs.
    """
    joined = "\x00".join(f"{role}\x01{tmpl}" for role, tmpl in templates)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _read_template(fname: str) -> str:
    return (PROMPTS_DIR / fname).read_text()


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Prompt registry manifest not found: {REGISTRY_PATH}")
    data = yaml.safe_load(REGISTRY_PATH.read_text()) or {}
    variants = data.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"{REGISTRY_PATH} has no non-empty 'variants' mapping")
    return variants


def _build_entry(name: str, spec: dict) -> dict:
    messages = spec.get("messages")
    if not messages:
        raise ValueError(f"Prompt variant {name!r} has no 'messages'")
    templates: list[Template] = []
    for i, msg in enumerate(messages):
        role = msg.get("role")
        fname = msg.get("file")
        if role not in _VALID_ROLES:
            raise ValueError(
                f"Prompt variant {name!r} message {i}: invalid role {role!r} "
                f"(expected one of {sorted(_VALID_ROLES)})"
            )
        if not fname:
            raise ValueError(f"Prompt variant {name!r} message {i}: missing 'file'")
        templates.append((role, _read_template(fname)))
    return {
        "name": name,
        "version": spec.get("version", 1),
        "description": spec.get("description", ""),
        "needs_graph_context": bool(spec.get("needs_graph_context", False)),
        "templates": templates,
        "files": [m.get("file") for m in messages],
        "fingerprint": fingerprint(templates),
    }


@lru_cache(maxsize=1)
def _entries() -> dict[str, dict]:
    return {name: _build_entry(name, spec) for name, spec in _load_manifest().items()}


def load_variants() -> dict[str, list[Template]]:
    """Return ``{name: [(role, template_text), ...]}`` for every variant.

    This is the exact shape ``src.agents.patcher._VARIANTS`` historically held,
    so the patcher and harness keep working unchanged.
    """
    return {name: list(e["templates"]) for name, e in _entries().items()}


def list_variants() -> list[str]:
    return sorted(_entries())


def variant_meta(name: str) -> dict:
    """Return metadata for a variant (raises ``KeyError`` if unknown)."""
    entries = _entries()
    if name not in entries:
        raise KeyError(f"Unknown prompt_variant {name!r}; known: {list_variants()}")
    e = entries[name]
    return {
        "name": e["name"],
        "version": e["version"],
        "description": e["description"],
        "needs_graph_context": e["needs_graph_context"],
        "fingerprint": e["fingerprint"],
        "files": list(e["files"]),
        "n_messages": len(e["templates"]),
    }


def variant_fingerprint(name: str) -> str:
    return variant_meta(name)["fingerprint"]


def needs_graph_context(name: str) -> bool:
    """Whether *name* wants serialized ``G_vuln`` context injected.

    Falls back to a name heuristic for variants registered at runtime (via the
    harness) that are not present in the declarative manifest.
    """
    entries = _entries()
    if name in entries:
        return entries[name]["needs_graph_context"]
    return name.startswith("graph")
