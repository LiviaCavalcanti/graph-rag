"""Tests for src.agents.patcher — unit tests (mocked) + live Azure integration test."""

import sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.patcher import (
    sanitize_after_index,
    AutoPatchPatcher,
    PatchResult,
    InvocationRecord,
    patch_one,
)


# ── sanitize_after_index ─────────────────────────────────────────────

class TestSanitizeAfterIndex:

    def test_escapes_quotes_in_target(self):
        s = 'before "hello" after'
        result = sanitize_after_index(s, 7, 14)
        assert result == 'before \\"hello\\" after'

    def test_escapes_newlines_in_target(self):
        s = "before\nhello\nafter"
        result = sanitize_after_index(s, 6, 13)
        assert result == "before\\nhello\\nafter"

    def test_escapes_tabs_in_target(self):
        s = "before\thello\tafter"
        result = sanitize_after_index(s, 6, 13)
        assert result == "before\\thello\\tafter"

    def test_leaves_before_and_after_untouched(self):
        s = '"before"\ttarget\t"after"'
        result = sanitize_after_index(s, 8, 15)
        # only the target region (indices 8..15) gets escaped
        assert result.startswith('"before"')
        assert result.endswith('"after"')

    def test_empty_target_region(self):
        s = "hello world"
        result = sanitize_after_index(s, 5, 5)
        assert result == "hello world"

    def test_does_not_double_escape_quotes(self):
        s = 'prefix \\"already\\" suffix'
        result = sanitize_after_index(s, 7, 19)
        # already-escaped quotes should NOT be double-escaped
        assert '\\\\\\"' not in result


# ── AutoPatchPatcher.parse ───────────────────────────────────────────

class TestAutoPatchPatcherParse:

    @pytest.fixture()
    def patcher(self):
        """Create an AutoPatchPatcher instance just for testing parse()."""
        return AutoPatchPatcher()

    def test_parse_valid_output(self, patcher):
        output = """
Some preamble text.

[CoT START]
We need to add a null check before dereferencing ptr.
[CoT END]

[Patched Code START]
void func() {
    if (ptr != NULL) {
        *ptr = value;
    }
}
[Patched Code END]

Some trailing text.
"""
        result = patcher.parse(output)
        assert result is not None
        assert isinstance(result, PatchResult)
        assert "null check" in result.cot
        assert "if (ptr != NULL)" in result.vuln_patch

    def test_parse_missing_markers(self, patcher):
        output = "No markers at all, just plain text."
        result = patcher.parse(output)
        assert result is None

    def test_parse_only_cot_marker(self, patcher):
        output = "[CoT START]some reasoning[CoT END]"
        result = patcher.parse(output)
        # Missing [Patched Code] markers → None
        assert result is None

    def test_parse_with_code_fences(self, patcher):
        output = """```
[CoT START]
Step 1 reasoning
[CoT END]

[Patched Code START]
```c
int x = 0;
```
[Patched Code END]
```"""
        result = patcher.parse(output)
        assert result is not None
        assert isinstance(result, PatchResult)
        assert "int x = 0;" in result.vuln_patch


# ── InvocationRecord ─────────────────────────────────────────────────

class TestInvocationRecord:

    def test_create_minimal_record(self):
        record = InvocationRecord(
            model="azure/test",
            temperature=0.2,
            max_tokens=4096,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert record.raw_output == ""
        assert record.parsed is None
        assert record.error is None
        assert record.elapsed_s == 0.0
        assert record.prompt_tokens == 0

    def test_record_with_parsed_result(self):
        parsed = PatchResult(cot="reasoning", vuln_patch="fixed code")
        record = InvocationRecord(
            model="azure/test",
            temperature=0.2,
            max_tokens=4096,
            messages=[],
            raw_output="some output",
            parsed=parsed,
            elapsed_s=1.5,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            finish_reason="stop",
        )
        assert record.parsed.vuln_patch == "fixed code"
        assert record.total_tokens == 150

    def test_save_and_load(self, tmp_path):
        parsed = PatchResult(cot="cot text", vuln_patch="patch text")
        record = InvocationRecord(
            model="azure/test",
            temperature=0.2,
            max_tokens=4096,
            messages=[{"role": "system", "content": "you are an expert"}],
            raw_output="raw llm output",
            parsed=parsed,
            elapsed_s=2.3,
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            finish_reason="stop",
            response_id="chatcmpl-abc123",
        )
        import json
        out_path = record.save(tmp_path / "traces" / "test.json")
        assert out_path.exists()

        loaded = json.loads(out_path.read_text())
        assert loaded["model"] == "azure/test"
        assert loaded["parsed"]["vuln_patch"] == "patch text"
        assert loaded["prompt_tokens"] == 200
        assert loaded["response_id"] == "chatcmpl-abc123"
        assert loaded["messages"][0]["role"] == "system"

    def test_save_creates_parent_dirs(self, tmp_path):
        record = InvocationRecord(
            model="azure/test",
            temperature=0.2,
            max_tokens=4096,
            messages=[],
        )
        deep_path = tmp_path / "a" / "b" / "c" / "record.json"
        out = record.save(deep_path)
        assert out.exists()

    def test_model_dump_roundtrip(self):
        parsed = PatchResult(cot="cot", vuln_patch="patch")
        record = InvocationRecord(
            model="azure/test",
            temperature=0.2,
            max_tokens=4096,
            messages=[{"role": "user", "content": "test"}],
            parsed=parsed,
            error=None,
        )
        data = record.model_dump()
        restored = InvocationRecord(**data)
        assert restored.parsed.vuln_patch == "patch"
        assert restored.model == "azure/test"


