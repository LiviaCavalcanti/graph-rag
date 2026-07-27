"""
Tool-calling patching agent — iterative diagnosis, generation, and verification.

Uses a ReAct-style loop: the LLM reasons, calls tools to gather context and
verify its work, and iterates until it submits a final patch or hits the
iteration cap.

Registry key: ``tool_calling``
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time

from src.agents.backends import CompletionBackend, CompletionResult, get_default_backend
from src.agents.base import AgentResult, PatchContext
from src.agents.patcher import InvocationRecord, PatchResult
from src.agents.tools import TOOL_SCHEMAS, execute_tool
from src.agents.utils import MODEL_NAME

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert C/C++ security engineer specializing in vulnerability patching.

Your task: given vulnerable C code and a retrieved similar vulnerability example, \
generate a minimal unified diff that fixes the vulnerability.

## Workflow

1. **Diagnose**: Call `get_vulnerability_context` to understand the CVE, CWE category, \
and typical fix patterns for this class of vulnerability.
2. **Study the example**: Call `analyze_example_fix` to see how a similar vulnerability \
was fixed. Extract the fix pattern.
3. **Generate a patch**: Produce a unified diff (`diff -u` format) containing ONLY the \
changed lines. Do NOT output the entire source file.
4. **Verify**: Call `check_c_syntax` on the patched function body only (the `+++` side of \
your diff). Pass the modified function, not the whole file.
5. **Submit**: Call `submit_patch` with your unified diff.

## Rules

- The patch MUST address the actual root cause, not just add superficial checks.
- Do NOT invent struct members, functions, or APIs that don't exist in the codebase.
- Keep the patch minimal — change only what's necessary to fix the vulnerability.
- If the example's CWE differs from the target's CWE, adapt your approach accordingly.
- If syntax check fails, fix the issues and re-check before submitting.
- You MUST call `submit_patch` to finalize your answer.

## Output discipline

- Do NOT write explanatory text between tool calls. Message content must be empty or \
at most one short sentence.
- All reasoning belongs in the `reasoning` argument of `submit_patch`, not in message \
content.
- The `patch` argument to `submit_patch` must be a unified diff string (lines starting \
with `---`, `+++`, `@@`, ` `, `+`, `-`). Never regenerate the entire source file.
"""


class ToolCallingAgent:
    """ReAct-style tool-calling patch agent (registry key: ``tool_calling``)."""

    name = "tool_calling"

    def __init__(
        self,
        *,
        prompt_variant: str = "default",
        model_name: str | None = None,
        llm_params: dict | None = None,
        backend: CompletionBackend | None = None,
        max_iterations: int = 6,
        verify_backend: CompletionBackend | None = None,
        verify_model: str | None = None,
    ):
        self.prompt_variant = prompt_variant
        self.model_name = model_name or MODEL_NAME
        self.llm_params = llm_params or {}
        self.backend = backend or get_default_backend()
        self.max_iterations = max_iterations
        self.verify_backend = verify_backend
        # Default to the same deployment as the main model: Azure requires the
        # model string to match an actual deployment name on the resource, and
        # there is no guarantee a separate "gpt-4o-mini" deployment exists.
        self.verify_model = verify_model or f"azure/{self.model_name}"

    def run(self, ctx: PatchContext) -> AgentResult:
        model = f"azure/{self.model_name}"
        temperature = self.llm_params.get("temperature", 0.2)
        max_tokens = self.llm_params.get("max_tokens", 8192)

        # Build tool execution context
        tool_ctx = {
            "target_db": ctx.target_db,
            "example_db": ctx.example_db,
            "target_code": ctx.target_code,
            "verify_backend": self.verify_backend or self.backend,
            "verify_model": self.verify_model,
        }

        # Initial messages
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_user_message(ctx),
            },
        ]

        turns: list[InvocationRecord] = []
        tool_call_log: list[dict] = []
        submitted_patch: str | None = None
        submitted_reasoning: str = ""

        for iteration in range(self.max_iterations):
            logger.info(
                "Tool-calling agent iteration %d/%d (messages=%d)",
                iteration + 1,
                self.max_iterations,
                len(messages),
            )

            # Call LLM
            t0 = time.perf_counter()
            try:
                result = self.backend.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                turns.append(
                    InvocationRecord(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=messages.copy(),
                        error=str(e),
                        elapsed_s=round(time.perf_counter() - t0, 3),
                    )
                )
                break

            elapsed = round(time.perf_counter() - t0, 3)
            record = InvocationRecord(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages.copy(),
                raw_output=result.content,
                finish_reason=result.finish_reason,
                response_id=result.response_id,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                elapsed_s=elapsed,
            )
            turns.append(record)

            # Handle tool calls
            if result.tool_calls:
                # Append assistant message with tool calls
                assistant_msg: dict = {"role": "assistant", "content": result.content}
                assistant_msg["tool_calls"] = result.tool_calls
                messages.append(assistant_msg)

                for tc in result.tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args_str = tc["function"]["arguments"]
                    tc_id = tc["id"]

                    try:
                        fn_args = json.loads(fn_args_str)
                    except json.JSONDecodeError:
                        fn_args = {}

                    logger.info("Executing tool: %s(%s)", fn_name, list(fn_args.keys()))

                    # Execute
                    tool_output = execute_tool(fn_name, fn_args, tool_ctx)

                    # Log
                    tool_call_log.append(
                        {
                            "iteration": iteration,
                            "tool": fn_name,
                            "arguments": fn_args,
                            "output_preview": tool_output[:500],
                        }
                    )

                    # Check for terminal tool
                    if fn_name == "submit_patch":
                        # Handle the gpt-4o errors of returning true to patch field instead of a string
                        raw_patch = fn_args.get("patch", "")
                        if isinstance(raw_patch, str) and raw_patch.strip():
                            patch_text = raw_patch
                            # Phase 2: if the model returned a unified diff,
                            # apply it to the target code server-side so that
                            # `generated_patch` in results.jsonl is always a
                            # full file — evaluation metrics unchanged.
                            if self._is_unified_diff(patch_text):
                                applied = self._apply_diff(ctx.target_code, patch_text)
                                if applied is not None:
                                    logger.info(
                                        "Applied unified diff → full file (%d chars)",
                                        len(applied),
                                    )
                                    patch_text = applied
                            submitted_patch = patch_text
                            submitted_reasoning = fn_args.get("reasoning", "")
                        else:
                            # Model passed a non-string (e.g. boolean True) or empty
                            # string. The tool already returned an error message that
                            # will be fed back so the model can retry.
                            logger.warning(
                                "submit_patch: 'patch' arg is %s (not a non-empty string)"
                                " — ignoring submission, model will receive error feedback",
                                type(raw_patch).__name__,
                            )

                    # Append tool result message
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": tool_output,
                        }
                    )

                # If patch was submitted, we're done
                if submitted_patch is not None:
                    break

            elif result.finish_reason == "stop":
                # LLM stopped without tool calls — try to parse output directly
                # (fallback: the model may have generated the patch inline)
                if result.content:
                    submitted_patch = self._try_extract_code(result.content)
                    submitted_reasoning = result.content[:200]
                break
            elif result.finish_reason == "length":
                # Completion was cut off by the token limit. Inject a compact
                # recovery prompt and continue so the model can retry with a
                # tool call rather than verbose prose.
                logger.warning(
                    "finish_reason=length at iteration %d — injecting recovery prompt",
                    iteration,
                )
                if iteration < self.max_iterations - 1:
                    if result.content:
                        messages.append(
                            {"role": "assistant", "content": result.content}
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your last response was cut off because it was too long. "
                                "Do NOT write any text. "
                                "Output ONLY a tool call with no explanatory prose."
                            ),
                        }
                    )
                    continue
                else:
                    logger.warning("finish_reason=length on last iteration — giving up")
                    break
            else:
                # Unexpected finish reason
                logger.warning(
                    "Unexpected finish_reason=%s at iteration %d",
                    result.finish_reason,
                    iteration,
                )
                break

        # Build final result
        parsed = None
        raw_output = ""
        if submitted_patch:
            parsed = PatchResult(
                cot=submitted_reasoning,
                vuln_patch=submitted_patch,
            )
            raw_output = submitted_patch
        elif turns:
            raw_output = turns[-1].raw_output

        return AgentResult(
            architecture=self.name,
            prompt_variant=self.prompt_variant,
            raw_output=raw_output,
            parsed=parsed,
            turns=turns,
            tool_calls=tool_call_log,
            prompt_fingerprint="",
        )

    def _build_user_message(self, ctx: PatchContext) -> str:
        """Build the initial user message with target code and metadata."""
        target_db = ctx.target_db
        parts = [
            "## Target Vulnerable Code",
            "",
            f"**CWE:** {target_db.get('cwe_type', 'Unknown')}",
            f"**CVE:** {target_db.get('cve_id', 'Unknown')}",
            "",
            "```c",
            ctx.target_code,
            "```",
            "",
            "Generate a patched version of this code that fixes the vulnerability.",
            "Use the available tools to understand the vulnerability and verify your fix.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _is_unified_diff(text: str) -> bool:
        """Return True when text looks like a unified diff (contains @@ hunk headers)."""
        return bool(re.search(r"^@@\s+-\d+", text, re.MULTILINE))

    @staticmethod
    def _apply_diff(original: str, diff: str) -> str | None:
        """Apply a unified diff to original code using the system ``patch`` utility.

        Returns the full patched file content, or None if the application
        failed (caller should fall back to the raw diff string).
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                orig_path = os.path.join(tmpdir, "original.c")
                patch_path = os.path.join(tmpdir, "changes.patch")
                out_path = os.path.join(tmpdir, "patched.c")

                with open(orig_path, "w") as f:
                    f.write(original)
                with open(patch_path, "w") as f:
                    f.write(diff)

                proc = subprocess.run(
                    ["patch", "-u", orig_path, patch_path, "-o", out_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and os.path.exists(out_path):
                    with open(out_path) as f:
                        return f.read()
                logger.warning(
                    "patch apply failed (rc=%d): %s",
                    proc.returncode,
                    proc.stderr[:300],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("_apply_diff error: %s", exc)
        return None

    @staticmethod
    def _try_extract_code(text: str) -> str | None:
        """Try to extract code from the LLM's free-form response.

        Priority order:
        1. Explicit markers ([Patched Code START]...[Patched Code END])
        2. Longest ```c fenced block (most likely the full patched function)
        3. Longest ``` fenced block
        4. If the text looks like bare C code (has braces, semicolons), use it directly
        """
        # 1. Explicit markers
        start_marker = "[Patched Code START]"
        end_marker = "[Patched Code END]"
        if start_marker in text and end_marker in text:
            s = text.index(start_marker) + len(start_marker)
            e = text.index(end_marker)
            return text[s:e].strip()

        # 2. Longest ```c block (pick the largest — most likely the full patch)
        if "```c" in text:
            blocks = []
            parts = text.split("```c")
            for part in parts[1:]:
                if "```" in part:
                    blocks.append(part.split("```")[0].strip())
            if blocks:
                return max(blocks, key=len)

        # 3. Longest ``` block
        if "```" in text:
            parts = text.split("```")
            # Odd-indexed parts are inside fences
            blocks = [parts[i].strip() for i in range(1, len(parts), 2) if parts[i].strip()]
            if blocks:
                return max(blocks, key=len)

        # 4. Bare code heuristic: if it has function signatures and braces
        stripped = text.strip()
        if (
            stripped.count("{") >= 1
            and stripped.count("}") >= 1
            and stripped.count(";") >= 1
            and "{" in stripped[:200]  # function body starts near the top
        ):
            return stripped

        return None
