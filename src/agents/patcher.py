import os
import re
from pathlib import Path

import litellm
from dotenv import load_dotenv

from src.agents.utils import MODEL_NAME, fmt_mapping, strip_code_fences

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


class AutoPatchPatcher:

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or MODEL_NAME

    def _build_messages(self, input_dict: dict) -> list[dict]:
        fmt_vars = {**input_dict, "format_instructions": _FORMAT_INSTRUCTIONS}
        return [
            {"role": role, "content": tmpl.format(**fmt_vars)}
            for role, tmpl in _MESSAGE_TEMPLATES
        ]

    def parse(self, output):
        json_output = None
        try:
            cot = output[
                output.find("[CoT START]")
                + len("[CoT START]") : output.find("[CoT END]")
            ]
            vuln_patch = output[
                output.find("[Patched Code START]")
                + len("[Patched Code START]") : output.find("[Patched Code END]")
            ]
            json_output = {"cot": cot, "vuln_patch": vuln_patch}
        except Exception as e:
            print("LLM output not directly JSON2. Need manual parsing.")
            print(e)
            print(output)
            return None
        return json_output

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
) -> tuple[str, dict | None]:
    """Build prompt, invoke LLM via litellm (google-adk Azure backend), parse result.

    Returns (raw_output, parsed) where *parsed* is a dict with
    'cot' and 'vuln_patch' keys, or None on parse failure.
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
