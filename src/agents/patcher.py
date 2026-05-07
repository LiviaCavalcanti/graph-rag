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
_SYSTEM_TMPL = _load_prompt("system.txt")
_USER_EXAMPLE_TMPL = _load_prompt("user_example.txt")
_ASSISTANT_TMPL = _load_prompt("assistant.txt")
_USER_TARGET_TMPL = _load_prompt("user_target.txt")

_MESSAGE_TEMPLATES = [
    ("system", _SYSTEM_TMPL),
    ("user", _USER_EXAMPLE_TMPL),
    ("assistant", _ASSISTANT_TMPL),
    ("user", _USER_TARGET_TMPL),
]


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

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or MODEL_NAME

    def _build_messages(self, input_dict: dict) -> list[dict]:
        fmt_vars = {**input_dict, "format_instructions": _FORMAT_INSTRUCTIONS}
        return [
            {"role": role, "content": tmpl.format(**fmt_vars)}
            for role, tmpl in _MESSAGE_TEMPLATES
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

    def invoke(self, input_dict: dict) -> str:
        messages = self._build_messages(input_dict)
        response = litellm.completion(
            model=f"azure/{self.model_name}",
            messages=messages,
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASEURL"),
            api_version="2024-12-01-preview",
            temperature=0.2,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""


# ── single-shot patcher ─────────────────────────────────────────────


def patch_one(
    example_db: dict,
    target_db: dict,
    target_code: str,
    target_supplementary: str = "",
    model_name: str | None = None,
) -> tuple[str, PatchResult | None]:
    """Build prompt, invoke LLM via litellm (google-adk Azure backend), parse result.

    Returns (raw_output, parsed) where *parsed* is a PatchResult
    with 'cot' and 'vuln_patch' fields, or None on parse failure.
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
    }

    patcher = AutoPatchPatcher(model_name)
    raw_output = patcher.invoke(input_dict)
    parsed = patcher.parse(raw_output)
    return raw_output, parsed
