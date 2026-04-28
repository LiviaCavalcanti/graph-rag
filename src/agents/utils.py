"""
Shared agent utilities — text helpers, output parsing, code extraction.

Used by:
  - src/agents/patcher.py (prompt building, LLM invocation, patch_one)
  - experiments/agent_experiment.py
  - experiments/batch_inference.py
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_NAME = MODEL_NAME if MODEL_NAME else "deepseek-R1"


# ── text helpers ─────────────────────────────────────────────────────


def fmt_mapping(mapping: dict | str | None) -> str:
    """Format variable/function mapping for the prompt."""
    if not mapping:
        return "None"
    if isinstance(mapping, str):
        return mapping
    return "\n".join(f"- {k}: {v}" for k, v in mapping.items())


def strip_code_fences(code: str) -> str:
    """Remove markdown code fences."""
    code = re.sub(r"^```[a-zA-Z]*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code.strip()


def read_code_file(path: str | None) -> str:
    """Read source code from a file path or inline string."""
    if not path:
        return ""
    p = Path(path)
    if p.exists():
        try:
            return p.read_text(errors="replace")
        except Exception:
            return ""
    if len(path) > 50:
        return path
    return ""


# ── code extraction from FunctionPair ────────────────────────────────


def get_target_code(pair) -> str:
    """Get the vulnerable code for a query pair."""
    code = pair.meta.get("source_before", "")
    if not code:
        lines = []
        for n, d in pair.G_before.nodes(data=True):
            if d.get("CODE"):
                lines.append(d["CODE"])
        code = "\n".join(lines)
    return strip_code_fences(code)


def get_ground_truth_patch(pair) -> str:
    """Get the patched code (ground truth) for evaluation."""
    code = pair.meta.get("source_after", "")
    return strip_code_fences(code)


# ── output parsing ───────────────────────────────────────────────────


def parse_patch(output: str) -> dict | None:
    """Parse CoT and patched code from LLM output."""
    try:
        cot_start = output.find("[CoT START]")
        cot_end = output.find("[CoT END]")
        patch_start = output.find("[Patched Code START]")
        patch_end = output.find("[Patched Code END]")

        cot = ""
        if cot_start >= 0 and cot_end > cot_start:
            cot = output[cot_start + len("[CoT START]") : cot_end].strip()

        vuln_patch = ""
        if patch_start >= 0 and patch_end > patch_start:
            vuln_patch = output[
                patch_start + len("[Patched Code START]") : patch_end
            ].strip()

        if vuln_patch:
            return {"cot": cot, "vuln_patch": strip_code_fences(vuln_patch)}
    except Exception:
        pass
    return None


def code_similarity(generated: str, reference: str) -> float:
    """Compute normalized similarity between generated and reference code."""
    if not generated or not reference:
        return 0.0
    gen_lines = [l.strip() for l in generated.splitlines() if l.strip()]
    ref_lines = [l.strip() for l in reference.splitlines() if l.strip()]
    return SequenceMatcher(None, gen_lines, ref_lines).ratio()
