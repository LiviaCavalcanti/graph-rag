"""Tests for batch_inference helpers: _get_target_code, _get_ground_truth, _run_single_query.

These tests verify the code extraction and ground-truth resolution logic
that was refactored to work with db_entry.json instead of graph traversal.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from agents.base import AgentResult
from agents.batch_inference import (
    ForbiddenError,
    _get_ground_truth,
    _get_target_code,
    _run_single_query,
)
from agents.patcher import InvocationRecord, PatchResult
from data.base import FunctionPair

# ── helpers ───────────────────────────────────────────────────────────


def _mock_record(**overrides):
    """Build a minimal InvocationRecord for test mocks."""
    defaults = dict(
        model="azure/test-model",
        temperature=0.2,
        max_tokens=4096,
        messages=[],
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        finish_reason="stop",
    )
    defaults.update(overrides)
    return InvocationRecord(**defaults)


class _FakeAgent:
    """Capturing stand-in for a patching agent (replaces mocking patch_one).

    _run_single_query now delegates to an Agent; these tests exercise its
    orchestration (db resolution, code extraction, metric computation, error
    handling) independent of any specific architecture.
    """

    name = "single_turn"

    def __init__(self, *, result=None, error=None):
        self._result = result
        self._error = error
        self.last_ctx = None

    def run(self, ctx):
        self.last_ctx = ctx
        if self._error is not None:
            raise self._error
        return self._result


def _agent_result(parsed=None, raw="raw", record=None):
    """Build an AgentResult wrapping a single InvocationRecord turn."""
    return AgentResult(
        architecture="single_turn",
        prompt_variant="default",
        raw_output=raw,
        parsed=parsed,
        turns=[record or _mock_record(parsed=parsed)],
        prompt_fingerprint="deadbeef0000",
    )


def _make_pair(
    cve_id="CVE-2025-0001",
    variant="original",
    dir_name="CVE-2025-0001",
    source_before="",
    source_after="",
    supplementary_code="",
    **extra_meta,
):
    G = nx.MultiDiGraph()
    meta = {
        "variant": variant,
        "dir_name": dir_name,
        "source_before": source_before,
        "source_after": source_after,
        "supplementary_code": supplementary_code,
        **extra_meta,
    }
    return FunctionPair(
        cve_id=cve_id,
        cwe_id="CWE-119",
        func_name="vuln_fn",
        project="autopatch",
        G_before=G,
        G_after=G,
        G_vuln=G,
        meta=meta,
    )


SAMPLE_DB = {
    "cve_id": "CVE-2025-0001",
    "cwe_type": "CWE-119",
    "function_name": "vuln_fn",
    "original_code": "void vuln_fn() { buf[20]=0; }",
    "vuln_patch": "void vuln_fn() { buf[9]=0; }",
    "supplementary_code": "struct Ctx { int x; };",
    "root_cause": "buffer overflow",
    "fix_list": ["bounds check"],
}


# ── _get_target_code ──────────────────────────────────────────────────


class TestGetTargetCode:

    def test_original_variant_uses_db_entry_code(self, tmp_path):
        """For original variants, source_before is a file path.
        _get_target_code should return db_entry['original_code'] instead."""
        code_file = tmp_path / "original_code.txt"
        code_file.write_text("void vuln_fn() {\n\tbuf[20]=0;\n}")

        pair = _make_pair(variant="original", source_before=str(code_file))
        result = _get_target_code(pair, SAMPLE_DB)
        assert result == "void vuln_fn() { buf[20]=0; }"

    def test_augmented_variant_uses_inline_code(self):
        """For augmented variants, source_before IS the code (inline string).
        _get_target_code should return it directly."""
        inline_code = "int reimpl() { char *p = 0; *p = 1; }"
        pair = _make_pair(variant="augmented", source_before=inline_code)
        result = _get_target_code(pair, SAMPLE_DB)
        assert result == inline_code

    def test_strips_code_fences(self):
        """Code wrapped in markdown fences should have fences removed."""
        fenced = "```c\nvoid foo() { return; }\n```"
        pair = _make_pair(variant="augmented", source_before=fenced)
        result = _get_target_code(pair, SAMPLE_DB)
        assert "```" not in result
        assert "void foo() { return; }" in result

    def test_empty_source_before_falls_back_to_db(self):
        pair = _make_pair(variant="original", source_before="")
        result = _get_target_code(pair, SAMPLE_DB)
        assert "buf[20]=0" in result

    def test_returns_empty_string_when_no_code_available(self):
        pair = _make_pair(variant="original", source_before="")
        result = _get_target_code(pair, {})
        assert result == ""


# ── _get_ground_truth ─────────────────────────────────────────────────


class TestGetGroundTruth:

    def test_reads_from_source_after_file(self, tmp_path):
        """When source_after is a file path, should read file contents."""
        fixed_file = tmp_path / "vuln_patch.txt"
        fixed_file.write_text("void vuln_fn() { buf[9]=0; }")
        pair = _make_pair(source_after=str(fixed_file))
        result = _get_ground_truth(pair, {})
        assert "buf[9]=0" in result

    def test_inline_source_after(self):
        """When source_after is long inline code (not a file), use it directly."""
        long_code = "void fixed_fn() { " + "x = 0; " * 20 + "}"
        pair = _make_pair(source_after=long_code)
        result = _get_ground_truth(pair, {})
        assert "fixed_fn" in result

    def test_falls_back_to_db_vuln_patch(self):
        """When source_after is empty, should use db_entry['vuln_patch']."""
        pair = _make_pair(source_after="")
        result = _get_ground_truth(pair, SAMPLE_DB)
        assert "buf[9]=0" in result

    def test_strips_code_fences_from_file(self, tmp_path):
        fixed_file = tmp_path / "fixed.c"
        fixed_file.write_text("```c\nvoid fixed() {}\n```")
        pair = _make_pair(source_after=str(fixed_file))
        result = _get_ground_truth(pair, {})
        assert "```" not in result

    def test_returns_empty_when_nothing_available(self):
        pair = _make_pair(source_after="")
        result = _get_ground_truth(pair, {})
        assert result == ""

    def test_nonexistent_file_path_does_not_crash(self):
        """If source_after points to a missing file, should fall back gracefully."""
        pair = _make_pair(source_after="/nonexistent/path/that/does/not/exist.c")
        result = _get_ground_truth(pair, SAMPLE_DB)
        # should fall back to db_entry
        assert "buf[9]=0" in result


# ── _run_single_query ─────────────────────────────────────────────────


class TestRunSingleQuery:

    def _make_retriever(self, example_pair=None, retrieval_info=None):
        retriever = MagicMock()
        retriever.retrieve.return_value = (
            example_pair,
            retrieval_info or {"cve_match": False, "cwe_match": False},
        )
        return retriever

    def test_skips_when_no_example_found(self):
        query = _make_pair(dir_name="CVE-2025-0001")
        retriever = self._make_retriever(example_pair=None)
        result = _run_single_query(
            query, retriever, {}, _FakeAgent(result=_agent_result())
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "no_example_found"

    def test_skips_when_target_db_missing(self):
        """If db_cache doesn't have the query's dir_name, should skip."""
        query = _make_pair(dir_name="CVE-2025-0001")
        example = _make_pair(dir_name="CVE-2025-0002")
        retriever = self._make_retriever(example_pair=example)
        # db_cache has example but not query
        db_cache = {"CVE-2025-0002": SAMPLE_DB}
        result = _run_single_query(
            query, retriever, db_cache, _FakeAgent(result=_agent_result())
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_db_entry"

    def test_skips_when_example_db_missing(self):
        """If db_cache doesn't have the example's dir_name, should skip."""
        query = _make_pair(dir_name="CVE-2025-0001")
        example = _make_pair(dir_name="CVE-2025-0002")
        retriever = self._make_retriever(example_pair=example)
        # db_cache has query but not example
        db_cache = {"CVE-2025-0001": SAMPLE_DB}
        result = _run_single_query(
            query, retriever, db_cache, _FakeAgent(result=_agent_result())
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_db_entry"

    def test_uses_dir_name_not_cve_id_for_db_lookup(self):
        """Regression: db_cache must be keyed by dir_name, not cve_id.
        Two dirs with same cve_id but different dir_names must resolve separately."""
        query = _make_pair(cve_id="CVE-2025-0001", dir_name="CVE-2025-0001_1")
        example = _make_pair(cve_id="CVE-2025-0001", dir_name="CVE-2025-0001_2")
        retriever = self._make_retriever(example_pair=example)

        db1 = {
            **SAMPLE_DB,
            "function_name": "func_a",
            "original_code": "void func_a(){}",
        }
        db2 = {
            **SAMPLE_DB,
            "function_name": "func_b",
            "original_code": "void func_b(){}",
        }

        # keyed by dir_name, NOT cve_id
        db_cache = {
            "CVE-2025-0001_1": db1,
            "CVE-2025-0001_2": db2,
        }

        parsed = PatchResult(vuln_patch="void func_a_fixed(){}", cot="...")
        agent = _FakeAgent(result=_agent_result(parsed=parsed))
        _run_single_query(query, retriever, db_cache, agent)

        # the agent must receive the correct per-dir db entries via PatchContext
        assert agent.last_ctx.example_db["function_name"] == "func_b"
        assert agent.last_ctx.target_db["function_name"] == "func_a"

    def test_skips_when_no_target_code(self):
        """If target code resolves to empty, should skip with reason no_target_code."""
        query = _make_pair(dir_name="CVE-2025-0001", source_before="")
        example = _make_pair(dir_name="CVE-2025-0002")
        retriever = self._make_retriever(example_pair=example)
        # db with no original_code
        db_cache = {
            "CVE-2025-0001": {"cve_id": "CVE-2025-0001"},
            "CVE-2025-0002": SAMPLE_DB,
        }
        result = _run_single_query(
            query, retriever, db_cache, _FakeAgent(result=_agent_result())
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "no_target_code"

    def test_success_returns_similarity_and_patch(self):
        ground_truth = "void vuln_fn() { buf[9]=0; }"
        parsed = PatchResult(vuln_patch=ground_truth, cot="fixed it")
        agent = _FakeAgent(result=_agent_result(parsed=parsed))

        query = _make_pair(dir_name="CVE-2025-0001", source_after="")
        example = _make_pair(dir_name="CVE-2025-0002")
        retriever = self._make_retriever(
            example_pair=example, retrieval_info={"cve_match": True, "cwe_match": True}
        )
        db_cache = {
            "CVE-2025-0001": SAMPLE_DB,
            "CVE-2025-0002": SAMPLE_DB,
        }
        result = _run_single_query(query, retriever, db_cache, agent)
        assert result["status"] == "success"
        assert result["similarity"] > 0
        assert result["generated_patch"] is not None
        assert "query_cve" in result
        assert "query_cwe" in result

    def test_exact_match_detected(self):
        """When generated patch is whitespace-equivalent to ground truth, exact_match should be True."""
        gt = "void fixed() { return 0; }"
        # same code with different whitespace
        generated = "void fixed()  {  return 0;  }"
        parsed = PatchResult(vuln_patch=generated, cot="...")
        agent = _FakeAgent(result=_agent_result(parsed=parsed))

        query = _make_pair(dir_name="CVE-2025-0001", source_after="")
        example = _make_pair(dir_name="CVE-2025-0002")
        retriever = self._make_retriever(example_pair=example)
        db_entry_with_gt = {**SAMPLE_DB, "vuln_patch": gt}
        db_cache = {
            "CVE-2025-0001": db_entry_with_gt,
            "CVE-2025-0002": SAMPLE_DB,
        }
        result = _run_single_query(query, retriever, db_cache, agent)
        assert result["status"] == "success"
        assert result["exact_match"] is True

    def test_parse_error_when_no_vuln_patch(self):
        """If LLM output doesn't produce a vuln_patch, status should be parse_error."""
        agent = _FakeAgent(result=_agent_result(parsed=None))
        query = _make_pair(dir_name="CVE-2025-0001")
        example = _make_pair(dir_name="CVE-2025-0002")
        retriever = self._make_retriever(example_pair=example)
        db_cache = {
            "CVE-2025-0001": SAMPLE_DB,
            "CVE-2025-0002": SAMPLE_DB,
        }
        result = _run_single_query(query, retriever, db_cache, agent)
        assert result["status"] == "parse_error"

    def test_403_raises_forbidden_error(self):
        """HTTP 403 from the API must raise ForbiddenError to abort the run."""
        agent = _FakeAgent(error=Exception("HTTP 403 Forbidden: Access denied"))
        query = _make_pair(dir_name="CVE-2025-0001")
        example = _make_pair(dir_name="CVE-2025-0002")
        retriever = self._make_retriever(example_pair=example)
        db_cache = {
            "CVE-2025-0001": SAMPLE_DB,
            "CVE-2025-0002": SAMPLE_DB,
        }
        with pytest.raises(ForbiddenError):
            _run_single_query(query, retriever, db_cache, agent)

    def test_non_403_error_returns_error_status(self):
        """Non-403 errors should return an error result, not raise."""
        agent = _FakeAgent(error=RuntimeError("Connection timeout"))
        query = _make_pair(dir_name="CVE-2025-0001")
        example = _make_pair(dir_name="CVE-2025-0002")
        retriever = self._make_retriever(example_pair=example)
        db_cache = {
            "CVE-2025-0001": SAMPLE_DB,
            "CVE-2025-0002": SAMPLE_DB,
        }
        result = _run_single_query(query, retriever, db_cache, agent)
        assert result["status"] == "error"
        assert "Connection timeout" in result["error"]

    def test_result_contains_retrieval_info(self):
        parsed = PatchResult(vuln_patch="fixed()", cot="...")
        agent = _FakeAgent(result=_agent_result(parsed=parsed))
        query = _make_pair(dir_name="CVE-2025-0001")
        example = _make_pair(dir_name="CVE-2025-0002")
        info = {"cve_match": True, "cwe_match": False, "distance": 0.5}
        retriever = self._make_retriever(example_pair=example, retrieval_info=info)
        db_cache = {
            "CVE-2025-0001": SAMPLE_DB,
            "CVE-2025-0002": SAMPLE_DB,
        }
        result = _run_single_query(query, retriever, db_cache, agent)
        assert result["retrieval"] == info
        assert result["cve_match"] is True
        assert result["cwe_match"] is False
