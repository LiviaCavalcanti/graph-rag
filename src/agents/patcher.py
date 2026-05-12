import json
import logging
import os
import re
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.agents.utils import MODEL_NAME, fmt_mapping, strip_code_fences

logger = logging.getLogger(__name__)

load_dotenv()
# litellm._turn_on_debug()

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def sanitize_after_index(s, start, until):
    before = s[:start]
    target = s[start:until]
    after = s[until:]

    # Only escape unescaped quotes (not already preceded by \)
    target = re.sub(r'(?<!\\)"', r"\"", target)

    # Only escape real newlines, not already escaped ones (i.e., not \\n)
    # This works by replacing actual newline characters, not literal \n
    target = target.replace("\n", r"\n")
    target = target.replace("\t", r"\t")

    return before + target + after


# ── prompt templates ─────────────────────────────────────────────────

_FORMAT_INSTRUCTIONS = _load_prompt("format_instructions.txt")

# Prompt variant → (system, user_example, assistant, user_target) templates
_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "default": [
        ("system", _load_prompt("system.txt")),
        ("user", _load_prompt("user_example.txt")),
        ("assistant", _load_prompt("assistant.txt")),
        ("user", _load_prompt("user_target.txt")),
    ],
    "graph": [
        ("system", _load_prompt("graph_system.txt")),
        ("user", _load_prompt("user_example.txt")),
        ("assistant", _load_prompt("assistant.txt")),
        ("user", _load_prompt("graph_user_target.txt")),
    ],
    "graph_v2": [
        ("system", _load_prompt("graph_system_v2.txt")),
        ("user", _load_prompt("user_example.txt")),
        ("assistant", _load_prompt("assistant_v2.txt")),
        ("user", _load_prompt("graph_user_target_v2.txt")),
    ],
    "default_v2": [
        ("system", _load_prompt("system_v2.txt")),
        ("user", _load_prompt("user_example.txt")),
        ("assistant", _load_prompt("assistant.txt")),
        ("user", _load_prompt("user_target.txt")),
    ],
}


class PatchResult(BaseModel):
    """Structured output from the patcher LLM."""
    cot: str
    vuln_patch: str


class InvocationRecord(BaseModel):
    """Everything needed to reproduce / debug a single LLM call."""
    model: str
    temperature: float
    max_tokens: int
    messages: list[dict]
    raw_output: str = ""
    parsed: PatchResult | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    response_id: str = ""

    def save(self, path: str | Path) -> Path:
        """Persist record as JSON for later replay / debugging."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(), indent=2, default=str))
        logger.debug("Saved invocation record → %s", p)
        return p


class AutoPatchPatcher:

    def __init__(self, model_name: str | None = None, prompt_variant: str = "default"):
        self.model_name = model_name or MODEL_NAME
        if prompt_variant not in _VARIANTS:
            raise ValueError(
                f"Unknown prompt_variant {prompt_variant!r}, "
                f"expected one of {list(_VARIANTS)}"
            )
        self.prompt_variant = prompt_variant
        self._templates = _VARIANTS[prompt_variant]

    def _build_messages(self, input_dict: dict) -> list[dict]:
        fmt_vars = {**input_dict, "format_instructions": _FORMAT_INSTRUCTIONS}
        return [
            {"role": role, "content": tmpl.format(**fmt_vars)}
            for role, tmpl in self._templates
        ]

    @staticmethod
    def _extract_between(text: str, start_marker: str, end_marker: str) -> str | None:
        start = text.find(start_marker)
        if start == -1:
            return None
        start += len(start_marker)
        end = text.find(end_marker, start)
        if end == -1:
            return None
        return text[start:end].strip()

    def parse(self, output: str) -> PatchResult | None:
        cot = self._extract_between(output, "[CoT START]", "[CoT END]")
        vuln_patch = self._extract_between(output, "[Patched Code START]", "[Patched Code END]")

        if cot is None or vuln_patch is None:
            logger.warning("Missing markers in LLM output (len=%d)", len(output))
            return None

        try:
            return PatchResult(cot=cot, vuln_patch=vuln_patch)
        except ValidationError as e:
            logger.warning("Pydantic validation failed: %s", e)
            return None

    def invoke(self, input_dict: dict) -> InvocationRecord:
        messages = self._build_messages(input_dict)
        params = dict(
            model=f"azure/{self.model_name}",
            temperature=0.2,
            max_tokens=4096,
        )
        record = InvocationRecord(
            model=params["model"],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            messages=messages,
        )

        logger.info(
            "Invoking %s  (prompt_msgs=%d, max_tokens=%d)",
            params["model"], len(messages), params["max_tokens"],
        )
        t0 = time.perf_counter()

        try:
            response = litellm.completion(
                **params,
                messages=messages,
                api_key=os.getenv("AZURE_API_KEY"),
                api_base=os.getenv("AZURE_API_BASEURL"),
                api_version="2024-12-01-preview",
            )
            record.elapsed_s = round(time.perf_counter() - t0, 3)
            record.raw_output = response.choices[0].message.content or ""
            record.finish_reason = response.choices[0].finish_reason or ""
            record.response_id = getattr(response, "id", "") or ""

            usage = getattr(response, "usage", None)
            if usage:
                record.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                record.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                record.total_tokens = getattr(usage, "total_tokens", 0) or 0

            logger.info(
                "LLM responded  (tokens=%d/%d/%d, finish=%s, elapsed=%.1fs)",
                record.prompt_tokens, record.completion_tokens, record.total_tokens,
                record.finish_reason, record.elapsed_s,
            )

            record.parsed = self.parse(record.raw_output)
            if record.parsed is None:
                logger.warning("Parse failed for response_id=%s", record.response_id)

        except Exception as exc:
            record.elapsed_s = round(time.perf_counter() - t0, 3)
            record.error = str(exc)
            logger.error(
                "LLM call failed after %.1fs: %s", record.elapsed_s, exc,
                exc_info=True,
            )
            raise

        return record


# ── single-shot patcher ─────────────────────────────────────────────


def patch_one(
    example_db: dict,
    target_db: dict,
    target_code: str,
    target_supplementary: str = "",
    model_name: str | None = None,
    trace_dir: str | Path | None = None,
    prompt_variant: str = "default",
    graph_context: str = "",
) -> tuple[str, PatchResult | None, InvocationRecord]:
    """Build prompt, invoke LLM via litellm (google-adk Azure backend), parse result.

    Returns (raw_output, parsed, record) where *parsed* is a PatchResult
    with 'cot' and 'vuln_patch' fields (or None on parse failure), and
    *record* is an InvocationRecord capturing everything for reproducibility.

    If *trace_dir* is provided, the record is saved as a JSON file there.

    Args:
        prompt_variant: "default" for original AutoPatch prompts,
                        "graph" for graph-enhanced prompts.
        graph_context:  Serialized graph analysis text (from
                        graph_context.serialize_graph_context). Only used
                        when prompt_variant="graph".
    """
    input_dict = {
        "example_target_cwe_type": example_db.get("cwe_type", "Unknown"),
        "example_target_cve_id": example_db.get("cve_id", "Unknown"),
        "example_anonymized_target_fix_list": example_db.get("fix_list", "None"),
        "example_target_vulnerability_related_variable_mapping": fmt_mapping(
            example_db.get("vulnerability_related_variables")
        ),
        "example_target_vulnerability_related_function_mapping": fmt_mapping(
            example_db.get("vulnerability_related_functions")
        ),
        "example_target_root_cause": example_db.get("root_cause", "Unknown"),
        "example_target_code": strip_code_fences(example_db.get("original_code", "")),
        "example_target_patch_cot": example_db.get("patch_cot", ""),
        "example_target_vuln_patch": strip_code_fences(
            example_db.get("vuln_patch", "")
        ),
        "target_supplementary_code": target_supplementary or "None",
        "target_vulnerability_related_variable_mapping": fmt_mapping(
            target_db.get("vulnerability_related_variables")
        ),
        "target_vulnerability_related_function_mapping": fmt_mapping(
            target_db.get("vulnerability_related_functions")
        ),
        "target_root_cause": target_db.get("root_cause", "Unknown"),
        "target_code": target_code,
        "target_graph_context": graph_context or "None",
    }

    patcher = AutoPatchPatcher(model_name, prompt_variant=prompt_variant)
    record = patcher.invoke(input_dict)

    if trace_dir:
        cve_id = example_db.get("cve_id", "unknown")
        fname = f"{cve_id}_{int(time.time())}.json"
        record.save(Path(trace_dir) / fname)

    return record.raw_output, record.parsed, record
