"""
Tool definitions and implementations for the tool-calling patching agent.

Each tool is:
1. An OpenAI-compatible schema (in TOOL_SCHEMAS)
2. A Python implementation function that takes parsed arguments + context

The agent loop calls tools by name, passing JSON-parsed arguments.
"""

from __future__ import annotations

import difflib
import json
import logging
from typing import Any

from src.agents.cwe_descriptions import get_cwe_info

logger = logging.getLogger(__name__)


# ── Tool Schemas (OpenAI function-calling format) ─────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_vulnerability_context",
            "description": (
                "Get detailed information about the target vulnerability: "
                "CVE description, CWE category with typical fix patterns, "
                "severity, and known root cause. Call this first to understand "
                "what needs to be fixed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_example_fix",
            "description": (
                "Analyze the retrieved example's fix: see what changed between "
                "the vulnerable and patched code, and understand the fix pattern "
                "used. Returns a structured diff and inferred fix strategy."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_c_syntax",
            "description": (
                "Check whether a C code snippet parses correctly. Returns 'OK' "
                "if the syntax is valid, or a list of parse errors. Use this to "
                "verify your generated patch before submitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The C code to syntax-check.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_fix_correctness",
            "description": (
                "Ask a verifier whether your proposed patch actually addresses "
                "the root cause of the vulnerability. Returns a verdict (correct/"
                "incorrect/uncertain) with reasoning. Use after generating a patch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis": {
                        "type": "string",
                        "description": "Your diagnosis of the root cause.",
                    },
                    "patch": {
                        "type": "string",
                        "description": "The proposed patched code.",
                    },
                },
                "required": ["diagnosis", "patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": (
                "Submit your final patched code. Only call this after you have "
                "verified the patch is correct. This ends the patching session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "The final patched C code.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of what was fixed and why.",
                    },
                },
                "required": ["patch", "reasoning"],
            },
        },
    },
]


# ── Tool Implementations ──────────────────────────────────────────────


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """Execute a tool by name with given arguments and context.

    Returns the tool's output as a string (to be fed back to the LLM).
    """
    fn = _TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return fn(arguments, context)
    except Exception as e:
        logger.warning("Tool %s failed: %s", tool_name, e)
        return f"Error executing {tool_name}: {e}"


def _tool_get_vulnerability_context(args: dict, ctx: dict) -> str:
    """Return CVE/CWE description, severity, root cause."""
    target_db = ctx.get("target_db", {})
    example_db = ctx.get("example_db", {})

    cwe_id = target_db.get("cwe_type", "") or example_db.get("cwe_type", "")
    cve_id = target_db.get("cve_id", "") or example_db.get("cve_id", "")

    cwe_info = get_cwe_info(cwe_id)

    root_cause = target_db.get("root_cause", "Not specified")
    cve_description = target_db.get("cve_description", "Not available")

    sections = [
        f"## CVE: {cve_id}",
        f"**Description:** {cve_description}",
        "",
        f"## CWE: {cwe_id} — {cwe_info['name']}",
        f"**Category description:** {cwe_info['description']}",
        f"**Typical fix patterns for this CWE:** {cwe_info['typical_fixes']}",
        "",
        f"## Root cause (from analysis): {root_cause}",
    ]
    return "\n".join(sections)


def _tool_analyze_example_fix(args: dict, ctx: dict) -> str:
    """Analyze the example's code_before → code_after transformation."""
    example_db = ctx.get("example_db", {})

    code_before = example_db.get("original_code", "")
    code_after = example_db.get("vuln_patch", "")
    fix_list = example_db.get("fix_list", "")
    example_cwe = example_db.get("cwe_type", "Unknown")
    example_cve = example_db.get("cve_id", "Unknown")

    diff_text = _smart_diff(code_before, code_after, token_budget=2000)

    sections = [
        f"## Example: {example_cve} ({example_cwe})",
        "",
        f"### Fix description:\n{fix_list or 'Not available'}",
        "",
        f"### Diff (vulnerable → patched):\n```diff\n{diff_text}\n```",
        "",
        "### Key observations:",
        "- Analyze what structural transformation was applied",
        "- Identify the fix pattern (added locking? NULL check? refactor into helper?)",
        "- Consider whether the same pattern applies to the target code",
    ]
    return "\n".join(sections)


def _smart_diff(before: str, after: str, token_budget: int = 2000) -> str:
    """Produce a diff that fits within a token budget (~4 chars/token).

    Strategy:
    1. If the full unified diff fits, use it.
    2. Otherwise, extract only the hunks (changed lines + 2 lines context).
    3. If still too large, show a summary + the first N hunks.
    """
    char_budget = token_budget * 4

    # Full diff with reduced context (2 lines instead of default 3)
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="vulnerable",
            tofile="patched",
            lineterm="",
            n=2,
        )
    )

    full_text = "\n".join(diff_lines)
    if len(full_text) <= char_budget:
        return full_text

    # Extract only the change lines (+ and - lines) with minimal context
    hunks = _extract_hunks(diff_lines)
    total_hunks = len(hunks)

    # Build output hunk by hunk until budget is exhausted
    output_parts = []
    chars_used = 0
    summary = (
        f"[{total_hunks} change hunks, showing those that fit in budget]\n"
        f"[Total diff: {len(diff_lines)} lines, "
        f"{sum(1 for l in diff_lines if l.startswith('+') and not l.startswith('+++'))} additions, "
        f"{sum(1 for l in diff_lines if l.startswith('-') and not l.startswith('---'))} deletions]\n"
    )
    chars_used += len(summary)
    output_parts.append(summary)

    for i, hunk in enumerate(hunks):
        hunk_text = "\n".join(hunk)
        if chars_used + len(hunk_text) + 2 > char_budget:
            remaining = total_hunks - i
            output_parts.append(f"\n... ({remaining} more hunks omitted)")
            break
        output_parts.append(hunk_text)
        chars_used += len(hunk_text) + 1

    return "\n".join(output_parts)


def _extract_hunks(diff_lines: list[str]) -> list[list[str]]:
    """Split unified diff lines into individual hunks (each starting with @@)."""
    hunks: list[list[str]] = []
    current: list[str] = []

    for line in diff_lines:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            current = [line]
        elif line.startswith("---") or line.startswith("+++"):
            continue  # skip file headers
        elif current:
            current.append(line)

    if current:
        hunks.append(current)
    return hunks


def _tool_check_c_syntax(args: dict, ctx: dict) -> str:
    """Check C syntax using tree-sitter if available, else basic heuristics."""
    code = args.get("code", "")
    if not code.strip():
        return "Error: empty code provided"

    # Try tree-sitter first
    try:
        return _check_with_tree_sitter(code)
    except ImportError:
        pass

    # Fallback: basic brace/bracket matching + common issues
    return _check_basic_syntax(code)


def _check_with_tree_sitter(code: str) -> str:
    """Use tree-sitter-c to parse and report errors."""
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser

    C_LANGUAGE = Language(tsc.language())
    parser = Parser(C_LANGUAGE)

    tree = parser.parse(bytes(code, "utf-8"))
    errors = []
    _collect_errors(tree.root_node, errors)

    if not errors:
        return "OK — code parses successfully with no syntax errors."

    result = f"SYNTAX ERRORS ({len(errors)} found):\n"
    for i, (line, col, text) in enumerate(errors[:10], 1):
        result += f"  {i}. Line {line + 1}, col {col}: {text}\n"
    if len(errors) > 10:
        result += f"  ... and {len(errors) - 10} more errors\n"
    return result


def _collect_errors(node, errors: list, max_depth: int = 50) -> None:
    """Recursively collect ERROR and MISSING nodes from a tree-sitter parse tree."""
    if max_depth <= 0:
        return
    if node.type == "ERROR" or node.is_missing:
        text = node.text.decode("utf-8", errors="replace")[:80] if node.text else ""
        errors.append((node.start_point[0], node.start_point[1], text or node.type))
    for child in node.children:
        _collect_errors(child, errors, max_depth - 1)


def _check_basic_syntax(code: str) -> str:
    """Fallback syntax check: brace matching and common hallucination patterns."""
    issues = []

    # Check brace balance
    opens = code.count("{")
    closes = code.count("}")
    if opens != closes:
        issues.append(f"Mismatched braces: {opens} opening vs {closes} closing")

    # Check paren balance
    opens_p = code.count("(")
    closes_p = code.count(")")
    if opens_p != closes_p:
        issues.append(
            f"Mismatched parentheses: {opens_p} opening vs {closes_p} closing"
        )

    # Check for common hallucination patterns
    # (struct members that don't exist are hard to catch without context,
    #  but we can flag suspicious patterns)
    if "pthread_mutex" in code and "linux/" in code:
        issues.append(
            "Warning: mixing pthread_mutex with kernel code — kernel uses "
            "mutex_lock/spin_lock, not pthread"
        )

    if not issues:
        return "OK — basic syntax checks pass (braces/parentheses balanced)."

    return "ISSUES:\n" + "\n".join(f"  - {i}" for i in issues)


def _tool_verify_fix_correctness(args: dict, ctx: dict) -> str:
    """Use an LLM-as-judge to verify the patch addresses the root cause.

    Falls back to a heuristic check if no verification backend is available.
    """
    diagnosis = args.get("diagnosis", "")
    patch = args.get("patch", "")
    target_code = ctx.get("target_code", "")
    target_db = ctx.get("target_db", {})
    cwe_id = target_db.get("cwe_type", "")

    # If we have a verification backend, use it
    verify_backend = ctx.get("verify_backend")
    if verify_backend is not None:
        return _llm_verify(verify_backend, diagnosis, patch, target_code, cwe_id, ctx)

    # Fallback: basic heuristic verification
    return _heuristic_verify(diagnosis, patch, target_code, cwe_id)


def _llm_verify(
    backend, diagnosis: str, patch: str, original: str, cwe_id: str, ctx: dict
) -> str:
    """Call a cheap LLM to judge patch correctness."""
    from src.agents.utils import MODEL_NAME

    model = ctx.get("verify_model") or f"azure/{MODEL_NAME}"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a security code reviewer. Given a vulnerability diagnosis "
                "and a proposed patch, determine if the patch correctly fixes the "
                "root cause. Be strict: the patch must address the actual vulnerability, "
                "not just add superficial checks. Respond with VERDICT: CORRECT, "
                "INCORRECT, or UNCERTAIN, followed by brief reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## CWE: {cwe_id}\n\n"
                f"## Diagnosis:\n{diagnosis}\n\n"
                f"## Original vulnerable code:\n```c\n{original}\n```\n\n"
                f"## Proposed patch:\n```c\n{patch}\n```\n\n"
                "Does this patch correctly fix the described vulnerability?"
            ),
        },
    ]

    try:
        result = backend.complete(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        return result.content
    except Exception as e:
        logger.warning("Verification LLM call failed: %s", e)
        return f"UNCERTAIN — verification call failed: {e}"


def _heuristic_verify(diagnosis: str, patch: str, original: str, cwe_id: str) -> str:
    """Basic heuristic checks when no LLM verifier is available."""
    checks = []

    if cwe_id == "CWE-476" and "NULL" not in patch.upper() and "!=" not in patch:
        checks.append(
            "WARNING: CWE-476 (NULL deref) but no NULL check visible in patch"
        )

    if cwe_id == "CWE-362" and not any(
        kw in patch for kw in ["lock", "mutex", "atomic", "rcu", "xchg"]
    ):
        checks.append(
            "WARNING: CWE-362 (race) but no synchronization primitives in patch"
        )

    if cwe_id == "CWE-416" and not any(
        kw in patch for kw in ["rcu", "ref", "get_", "put_", "try_get", "NULL"]
    ):
        checks.append("WARNING: CWE-416 (UAF) but no refcounting/RCU pattern visible")

    if not checks:
        return "UNCERTAIN — basic heuristic checks pass, but cannot verify semantic correctness without LLM judge."

    return "\n".join(checks)


def _tool_submit_patch(args: dict, ctx: dict) -> str:
    """Terminal tool — marks the patch as final."""
    patch = args.get("patch", "")
    reasoning = args.get("reasoning", "")
    if not patch.strip():
        return "Error: cannot submit empty patch"
    # The agent loop checks for this tool name to terminate
    return f"SUBMITTED. Reasoning: {reasoning}"


# ── Tool Registry ─────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, Any] = {
    "get_vulnerability_context": _tool_get_vulnerability_context,
    "analyze_example_fix": _tool_analyze_example_fix,
    "check_c_syntax": _tool_check_c_syntax,
    "verify_fix_correctness": _tool_verify_fix_correctness,
    "submit_patch": _tool_submit_patch,
}
