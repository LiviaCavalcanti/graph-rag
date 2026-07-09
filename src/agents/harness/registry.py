"""
PromptRegistry — inspect, diff, render, and evolve prompt variants.

The agent's prompt variants live in :data:`src.agents.patcher._VARIANTS`
(name → ordered list of ``(role, template)``). This module wraps that mapping
with tooling for the *evolve* workflow:

    - list / describe registered variants and their content fingerprints
    - render the full prompt a variant would send for a given scenario
      (no LLM call) so you can eyeball or unit-test prompt construction
    - diff two variants message-by-message (unified diff)
    - register a brand-new variant at runtime (so ``patch_one`` / the harness
      can immediately A/B it) and optionally persist its templates to files

Nothing here calls the network.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path

from src.agents import patcher as _patcher
from src.agents.harness.scenario import Scenario

PROMPTS_DIR: Path = _patcher._PROMPTS_DIR


def _fingerprint(templates: list[tuple[str, str]]) -> str:
    joined = "\x00".join(f"{role}\x01{tmpl}" for role, tmpl in templates)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def list_variants() -> list[str]:
    """Registered prompt-variant names."""
    return sorted(_patcher._VARIANTS)


def get_templates(variant: str) -> list[tuple[str, str]]:
    """Return the ``(role, template)`` list for a variant (raises if unknown)."""
    if variant not in _patcher._VARIANTS:
        raise KeyError(f"Unknown prompt_variant {variant!r}; known: {list_variants()}")
    return list(_patcher._VARIANTS[variant])


def describe(variant: str | None = None) -> dict | list[dict]:
    """Summarise one variant (or all): roles, message count, fingerprint."""

    def _one(name: str) -> dict:
        tmpls = get_templates(name)
        return {
            "variant": name,
            "n_messages": len(tmpls),
            "roles": [role for role, _ in tmpls],
            "fingerprint": _fingerprint(tmpls),
            "approx_template_chars": sum(len(t) for _, t in tmpls),
        }

    if variant is not None:
        return _one(variant)
    return [_one(n) for n in list_variants()]


def render_messages(scenario: Scenario, variant: str) -> list[dict]:
    """Build the exact chat messages a variant would send for *scenario*.

    Uses a mock backend so constructing the patcher never touches the network.
    """
    from src.agents.backends import MockBackend

    input_dict = _patcher.build_input_dict(
        scenario.example_db,
        scenario.target_db,
        scenario.target_code,
        scenario.target_supplementary,
        scenario.graph_context,
    )
    p = _patcher.AutoPatchPatcher(prompt_variant=variant, backend=MockBackend(""))
    return p._build_messages(input_dict)


def diff_variants(a: str, b: str) -> list[dict]:
    """Message-by-message diff between two variants.

    Returns one entry per aligned message slot with the roles, whether the
    template text changed, and a unified diff when it did.
    """
    ta, tb = get_templates(a), get_templates(b)
    out: list[dict] = []
    for i in range(max(len(ta), len(tb))):
        ra, sa = ta[i] if i < len(ta) else ("<none>", "")
        rb, sb = tb[i] if i < len(tb) else ("<none>", "")
        changed = sa != sb or ra != rb
        entry = {"index": i, "role_a": ra, "role_b": rb, "changed": changed}
        if changed:
            entry["diff"] = "\n".join(
                difflib.unified_diff(
                    sa.splitlines(),
                    sb.splitlines(),
                    fromfile=f"{a}[{i}:{ra}]",
                    tofile=f"{b}[{i}:{rb}]",
                    lineterm="",
                )
            )
        out.append(entry)
    return out


def register_variant(
    name: str,
    templates: list[tuple[str, str]],
    *,
    overwrite: bool = False,
) -> str:
    """Register a new prompt variant at runtime.

    After this call, ``AutoPatchPatcher(prompt_variant=name)``, ``patch_one``,
    and the harness can all use *name* immediately. Returns its fingerprint.
    """
    if name in _patcher._VARIANTS and not overwrite:
        raise ValueError(f"Variant {name!r} already exists (pass overwrite=True)")
    if not templates:
        raise ValueError("templates must be a non-empty list of (role, template)")
    roles = {r for r, _ in templates}
    unknown = roles - {"system", "user", "assistant"}
    if unknown:
        raise ValueError(f"Invalid message roles: {sorted(unknown)}")
    _patcher._VARIANTS[name] = [(r, t) for r, t in templates]
    return _fingerprint(templates)


def register_from_files(
    name: str,
    spec: list[tuple[str, str]],
    *,
    prompts_dir: Path | None = None,
    overwrite: bool = False,
) -> str:
    """Register a variant from ``(role, filename)`` pairs under the prompts dir."""
    base = prompts_dir or PROMPTS_DIR
    templates = [(role, (base / fname).read_text()) for role, fname in spec]
    return register_variant(name, templates, overwrite=overwrite)


def save_variant_templates(
    variant: str, out_dir: str | Path, *, prefix: str | None = None
) -> list[Path]:
    """Persist a variant's templates to ``<out_dir>/<prefix>_<idx>_<role>.txt``.

    Handy for capturing an evolved prompt so it can be committed and reloaded.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfx = prefix or variant
    written: list[Path] = []
    for i, (role, tmpl) in enumerate(get_templates(variant)):
        f = out / f"{pfx}_{i}_{role}.txt"
        f.write_text(tmpl)
        written.append(f)
    return written


__all__ = [
    "PROMPTS_DIR",
    "list_variants",
    "get_templates",
    "describe",
    "render_messages",
    "diff_variants",
    "register_variant",
    "register_from_files",
    "save_variant_templates",
]
