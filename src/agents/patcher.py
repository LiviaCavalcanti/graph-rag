import json
import logging
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from src.agents import prompt_registry
from src.agents.backends import CompletionBackend, get_default_backend
from src.agents.utils import MODEL_NAME, fmt_mapping, strip_code_fences

logger = logging.getLogger(__name__)

load_dotenv()

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# ── LLM invocation defaults (override per-call via AutoPatchPatcher(...) / config) ──
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096
DEFAULT_API_VERSION = "2024-12-01-preview"


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

# Prompt variants are loaded from the declarative registry
# (prompts/registry.yaml). Kept as a module-level dict of the historical shape
# ``name → [(role, template), ...]`` so the harness (src.agents.harness.registry)
# and runtime variant registration keep working unchanged.
_VARIANTS: dict[str, list[tuple[str, str]]] = prompt_registry.load_variants()


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
    cached: bool = False

    def save(self, path: str | Path) -> Path:
        """Persist record as JSON for later replay / debugging."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(), indent=2, default=str))
        logger.debug("Saved invocation record → %s", p)
        return p


class AutoPatchPatcher:

    def __init__(
        self,
        model_name: str | None = None,
        prompt_variant: str = "default",
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_version: str = DEFAULT_API_VERSION,
        backend: CompletionBackend | None = None,
    ):
        self.model_name = model_name or MODEL_NAME
        if prompt_variant not in _VARIANTS:
            raise ValueError(
                f"Unknown prompt_variant {prompt_variant!r}, "
                f"expected one of {list(_VARIANTS)}"
            )
        self.prompt_variant = prompt_variant
        self._templates = _VARIANTS[prompt_variant]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_version = api_version
        # Pluggable LLM backend (live / mock / replay / record). Resolved at
        # construction so an active use_backend(...) scope is picked up.
        self._backend = backend or get_default_backend()

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
        vuln_patch = self._extract_between(
            output, "[Patched Code START]", "[Patched Code END]"
        )

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
        model = f"azure/{self.model_name}"
        record = InvocationRecord(
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )

        logger.info(
            "Invoking %s  (backend=%s, prompt_msgs=%d, max_tokens=%d)",
            model,
            type(self._backend).__name__,
            len(messages),
            self.max_tokens,
        )
        t0 = time.perf_counter()

        try:
            result = self._backend.complete(
                model=model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_version=self.api_version,
            )
            record.elapsed_s = round(time.perf_counter() - t0, 3)
            record.raw_output = result.content or ""
            record.finish_reason = result.finish_reason or ""
            record.response_id = result.response_id or ""
            record.prompt_tokens = result.prompt_tokens
            record.completion_tokens = result.completion_tokens
            record.total_tokens = result.total_tokens
            record.cached = result.cached

            logger.info(
                "LLM responded  (tokens=%d/%d/%d, finish=%s, cached=%s, elapsed=%.1fs)",
                record.prompt_tokens,
                record.completion_tokens,
                record.total_tokens,
                record.finish_reason,
                record.cached,
                record.elapsed_s,
            )

            record.parsed = self.parse(record.raw_output)
            if record.parsed is None:
                logger.warning("Parse failed for response_id=%s", record.response_id)

        except Exception as exc:
            record.elapsed_s = round(time.perf_counter() - t0, 3)
            record.error = str(exc)
            logger.error(
                "LLM call failed after %.1fs: %s",
                record.elapsed_s,
                exc,
                exc_info=True,
            )
            raise

        return record


# ── single-shot patcher ─────────────────────────────────────────────


def build_input_dict(
    example_db: dict,
    target_db: dict,
    target_code: str,
    target_supplementary: str = "",
    graph_context: str = "",
) -> dict:
    """Assemble the prompt template variables from the retrieved example and target.

    Shared by :func:`patch_one` and the harness prompt registry so that debug /
    diff tooling renders exactly the same prompt the agent would send.
    """
    return {
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


def patch_one(
    example_db: dict,
    target_db: dict,
    target_code: str,
    target_supplementary: str = "",
    model_name: str | None = None,
    trace_dir: str | Path | None = None,
    prompt_variant: str = "default",
    graph_context: str = "",
    llm_params: dict | None = None,
    backend: CompletionBackend | None = None,
) -> tuple[str, PatchResult | None, InvocationRecord]:
    """Build prompt, invoke the LLM backend, and parse the result.

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
        backend:        Optional CompletionBackend override (live/mock/replay/
                        record). Defaults to the process-wide backend.
    """
    input_dict = build_input_dict(
        example_db, target_db, target_code, target_supplementary, graph_context
    )

    patcher = AutoPatchPatcher(
        model_name, prompt_variant=prompt_variant, backend=backend, **(llm_params or {})
    )
    record = patcher.invoke(input_dict)

    if trace_dir:
        cve_id = example_db.get("cve_id", "unknown")
        fname = f"{cve_id}_{int(time.time())}.json"
        record.save(Path(trace_dir) / fname)

    return record.raw_output, record.parsed, record
