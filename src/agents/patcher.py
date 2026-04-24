import os
import re
import textwrap

import litellm
from dotenv import load_dotenv

from src.agents.utils import (
    MODEL_NAME,
    fmt_mapping,
    strip_code_fences,
)

load_dotenv()
# litellm._turn_on_debug()

def sanitize_after_index(s, start,until):
    before = s[:start]
    target = s[start:until]
    after = s[until:]

    # Only escape unescaped quotes (not already preceded by \)
    target = re.sub(r'(?<!\\)"', r'\"', target)
    
    # Only escape real newlines, not already escaped ones (i.e., not \\n)
    # This works by replacing actual newline characters, not literal \n
    target = target.replace('\n', r'\n')
    target = target.replace('\t', r'\t')

    return before + target + after


# ── prompt templates ─────────────────────────────────────────────────

_FORMAT_INSTRUCTIONS = textwrap.dedent("""\
in the following format, including the leading and trailing "```" and "```" 
```
[CoT START]
<the thinking process for the vulnerability patching (Step 1)>
[CoT END]

[Patched Code START]
<the patched code>
[Patched Code END]
```""")

_SYSTEM_TMPL = textwrap.dedent("""\
You are an expert software security engineer. Your goal is to patch user-provided [Target Code] having a vulnerabilitiy of {example_target_cwe_type} similar to {example_target_cve_id}. To patch a vulnerability similar to {example_target_cve_id}, you will mainly focus on the following [Fix-List for {example_target_cve_id}], where mapping from actual variables to each symbolic variables/functions in [Fix-List for {example_target_cve_id}] should be provided by user with [Variable Mapping] and [Function Mapping]. 

[Fix-List for {example_target_cve_id}]
{example_anonymized_target_fix_list}

Perform the followings step by step and show the reasoning in each step. Start answering with "Let's think step-by-step."
1) Based on [Fix-List for {example_target_cve_id}] along with the user-provided [Variable Mapping] and [Function Mapping], describe how to patch [Target Code] for fixing {example_target_cwe_type} similar to {example_target_cve_id}.
2) Use the patch description from Step 1 to generate a patched code.
3) Provide the results {format_instructions}""")

_USER_EXAMPLE_TMPL = textwrap.dedent("""\
[Supplementary Code]
None

[Variable Mapping]
{example_target_vulnerability_related_variable_mapping}

[Function Mapping]
{example_target_vulnerability_related_function_mapping}

[Root Cause]
{example_target_root_cause}

[Target Code]
{example_target_code}""")

_ASSISTANT_TMPL = textwrap.dedent("""\
Now, I will patch user-provided [Target Code] having a vulnerabilitiy of {example_target_cwe_type} similar to {example_target_cve_id}. Then, I will provide the results {format_instructions}

Let's think step-by-step.

Step 1. Describe how to patch [Target Code] to fix {example_target_cwe_type} similar to {example_target_cve_id}.
{example_target_patch_cot}

Step 2. Generate a patched code based on Step 1.
{example_target_vuln_patch}

Step 3. Provide the results.
```
[CoT START]
 {example_target_patch_cot}
[CoT END]

[Patched Code START]
{example_target_vuln_patch}
[Patched Code END]
```""")

_USER_TARGET_TMPL = textwrap.dedent("""\
[Supplementary Code]
{target_supplementary_code}

[Variable Mapping]
{target_vulnerability_related_variable_mapping}

[Function Mapping]
{target_vulnerability_related_function_mapping}

[Root Cause]
{target_root_cause}

[Target Code]
{target_code}""")

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
            cot = output[output.find('[CoT START]') + len('[CoT START]'): output.find('[CoT END]')]
            vuln_patch = output[output.find('[Patched Code START]') + len('[Patched Code START]'): output.find('[Patched Code END]')]
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
        "example_target_vuln_patch": strip_code_fences(example_db.get("vuln_patch", "")),
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


