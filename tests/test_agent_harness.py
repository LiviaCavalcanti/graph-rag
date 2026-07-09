"""
Offline test suite for the agent harness (debug / test / evolve / evaluate).

Everything here runs with no network: MockBackend and a temp Cassette stand in
for the LLM. This is the regression net for the whole harness + the backend
seam wired into patcher.patch_one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.backends import (
    Cassette,
    CassetteMiss,
    CompletionResult,
    MockBackend,
    RecordBackend,
    ReplayBackend,
    request_fingerprint,
    use_backend,
)
from src.agents.harness import (
    AgentEvaluator,
    AgentHarness,
    Scenario,
    ScoreCard,
    compare_variants,
    format_trace,
    load_scenarios,
    registry,
    replay_record,
    save_traces,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "agent" / "scenarios.json"


@pytest.fixture()
def scenarios():
    return load_scenarios(FIXTURES)


@pytest.fixture()
def scn(scenarios):
    return scenarios[0]  # cwe476-null-deref


def _echo_backend(scenario: Scenario) -> MockBackend:
    """A mock that returns the scenario's ground-truth patch (→ perfect scores)."""
    return MockBackend.from_patch("reasoning", scenario.ground_truth)


# ── backends ─────────────────────────────────────────────────────────


class TestBackends:
    def test_mock_from_patch_is_parseable(self):
        backend = MockBackend.from_patch("why", "int x = 0;")
        res = backend.complete(
            model="azure/m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=10,
        )
        assert "[Patched Code START]" in res.content
        assert "int x = 0;" in res.content
        assert res.total_tokens == res.prompt_tokens + res.completion_tokens
        assert backend.calls  # request captured

    def test_fingerprint_is_deterministic_and_sensitive(self):
        msgs = [{"role": "user", "content": "a"}]
        k1 = request_fingerprint(
            model="m", messages=msgs, temperature=0.2, max_tokens=5
        )
        k2 = request_fingerprint(
            model="m", messages=msgs, temperature=0.2, max_tokens=5
        )
        k3 = request_fingerprint(
            model="m", messages=msgs, temperature=0.3, max_tokens=5
        )
        assert k1 == k2
        assert k1 != k3

    def test_cassette_record_then_replay_roundtrip(self, tmp_path):
        cass = Cassette(tmp_path / "c.jsonl")
        inner = MockBackend("hello world")
        rec = RecordBackend(inner, cass)
        kw = dict(
            model="azure/m",
            messages=[{"role": "user", "content": "q"}],
            temperature=0.0,
            max_tokens=8,
        )

        first = rec.complete(**kw)
        assert first.content == "hello world" and not first.cached
        assert len(cass) == 1

        # second call served from cassette (no inner invocation)
        inner2 = MockBackend("SHOULD NOT BE USED")
        rec2 = RecordBackend(inner2, cass)
        second = rec2.complete(**kw)
        assert second.content == "hello world" and second.cached

        cass.save()
        replay = ReplayBackend(Cassette(tmp_path / "c.jsonl"))
        third = replay.complete(**kw)
        assert third.content == "hello world" and third.cached

    def test_replay_miss_is_strict(self, tmp_path):
        replay = ReplayBackend(Cassette(tmp_path / "empty.jsonl"))
        with pytest.raises(CassetteMiss):
            replay.complete(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                temperature=0.0,
                max_tokens=1,
            )

    def test_use_backend_scopes_default(self):
        from src.agents.backends import get_default_backend

        sentinel = MockBackend("scoped")
        with use_backend(sentinel):
            assert get_default_backend() is sentinel
        # restored afterwards (fresh LiveBackend, not the sentinel)
        assert get_default_backend() is not sentinel

    def test_completion_result_dict_roundtrip(self):
        r = CompletionResult(content="x", finish_reason="stop", total_tokens=3)
        r2 = CompletionResult.from_dict(r.to_dict(), cached=True)
        assert r2.content == "x" and r2.cached is True


# ── patcher integration via backend seam ─────────────────────────────


class TestPatcherBackendSeam:
    def test_patch_one_uses_injected_backend(self, scn):
        from src.agents.patcher import patch_one

        backend = _echo_backend(scn)
        raw, parsed, record = patch_one(
            example_db=scn.example_db,
            target_db=scn.target_db,
            target_code=scn.target_code,
            model_name="test-model",
            prompt_variant="default",
            backend=backend,
        )
        assert parsed is not None
        assert parsed.vuln_patch.strip() == scn.ground_truth.strip()
        assert record.total_tokens > 0

    def test_use_backend_default_flows_into_patch_one(self, scn):
        from src.agents.patcher import patch_one

        with use_backend(_echo_backend(scn)):
            _, parsed, _ = patch_one(
                example_db=scn.example_db,
                target_db=scn.target_db,
                target_code=scn.target_code,
                model_name="test-model",
            )
        assert parsed is not None


# ── scenario fixtures ────────────────────────────────────────────────


class TestScenario:
    def test_load_fixtures(self, scenarios):
        assert {s.id for s in scenarios} == {"cwe476-null-deref", "cwe787-oob-write"}

    def test_to_from_dict_roundtrip(self, scn):
        restored = Scenario.from_dict(scn.to_dict())
        assert restored.id == scn.id
        assert restored.ground_truth == scn.ground_truth
        assert restored.expected == scn.expected


# ── prompt registry (evolve) ─────────────────────────────────────────


class TestRegistry:
    def test_list_and_describe(self):
        variants = registry.list_variants()
        assert "default" in variants and "graph" in variants
        d = registry.describe("default")
        assert d["n_messages"] == 4 and len(d["fingerprint"]) == 12

    def test_render_messages_contains_target_code(self, scn):
        msgs = registry.render_messages(scn, "default")
        assert msgs[0]["role"] == "system"
        joined = "\n".join(m["content"] for m in msgs)
        assert "process" in joined  # target function name appears

    def test_diff_variants_detects_change(self):
        diff = registry.diff_variants("default", "graph")
        assert any(e["changed"] for e in diff)

    def test_register_variant_then_usable(self, scn):
        name = "test_evolved_variant"
        templates = registry.get_templates("default")
        registry.register_variant(name, templates, overwrite=True)
        assert name in registry.list_variants()
        # a newly-registered variant is immediately runnable
        h = AgentHarness(prompt_variant=name, model_name="m")
        trace = h.run_scenario(scn, backend=_echo_backend(scn))
        assert trace.status == "success"

    def test_register_rejects_bad_role(self):
        with pytest.raises(ValueError):
            registry.register_variant("bad", [("robot", "hi")], overwrite=True)


# ── evaluator (evaluate) ─────────────────────────────────────────────


class TestEvaluator:
    def test_exact_match_scores_perfect(self):
        ev = AgentEvaluator()
        card = ev.score("int x = 0;", "int x = 0;")
        assert card.exact_match and card.similarity == 1.0 and card.parsed

    def test_comment_stripping_makes_exact(self):
        ev = AgentEvaluator(strip_comments=True)
        card = ev.score("int x = 0; // set", "int x = 0;")
        assert card.exact_match

    def test_expectations_pass_and_fail(self):
        ev = AgentEvaluator()
        ok = ev.score(
            "if (p) return;",
            "if (p) return;",
            expected={"contains": ["if"], "min_similarity": 0.5},
        )
        assert ok.expectations_passed and ok.passed
        bad = ev.score(
            "return;",
            "if (p) return;",
            expected={"contains": ["if"], "not_contains": ["return"]},
        )
        assert not bad.expectations_passed
        assert len(bad.expectation_failures) == 2

    def test_empty_generated_does_not_parse(self):
        card = AgentEvaluator().score("", "int x;", expected={"must_parse": True})
        assert not card.parsed and not card.expectations_passed

    def test_aggregate(self):
        ev = AgentEvaluator()
        cards = [
            ev.score("int x = 0;", "int x = 0;"),
            ev.score("int y = 1;", "int x = 0;"),
        ]
        agg = ev.aggregate(cards)
        assert agg["n"] == 2 and 0.0 <= agg["mean_similarity"] <= 1.0
        assert agg["parse_rate"] == 1.0


# ── runner (debug / test) ────────────────────────────────────────────


class TestRunner:
    def test_run_scenario_perfect(self, scn):
        h = AgentHarness(model_name="m", prompt_variant="default")
        trace = h.run_scenario(scn, backend=_echo_backend(scn))
        assert trace.status == "success"
        assert trace.score["exact_match"] and trace.score["similarity"] == 1.0
        assert trace.score["expectations_passed"]

    def test_run_scenario_parse_error(self, scn):
        h = AgentHarness(model_name="m")
        trace = h.run_scenario(scn, backend=MockBackend("no markers here"))
        assert trace.status == "parse_error"
        assert trace.generated_patch == ""

    def test_run_scenario_backend_error_becomes_trace(self, scn):
        class Boom:
            def complete(self, **kw):
                raise RuntimeError("kaboom")

        h = AgentHarness(model_name="m")
        trace = h.run_scenario(scn, backend=Boom())
        assert trace.status == "error" and "kaboom" in trace.error

    def test_run_suite_installs_backend_globally(self, scenarios):
        # No explicit per-call backend: rely on run_suite installing it.
        backend = MockBackend.from_patch("r", scenarios[0].ground_truth)
        h = AgentHarness(model_name="m", prompt_variant="default")
        result = h.run_suite(scenarios, backend=backend)
        assert result.summary["n"] == 2
        assert result.summary["status_counts"]["success"] == 2

    def test_format_trace_has_sections(self, scn):
        h = AgentHarness(model_name="m")
        trace = h.run_scenario(scn, backend=_echo_backend(scn))
        text = format_trace(trace)
        assert "Prompt" in text and "Parsed patch" in text and "Scores" in text

    def test_save_and_replay_record(self, scn, tmp_path):
        h = AgentHarness(model_name="m")
        trace = h.run_scenario(scn, backend=_echo_backend(scn))
        # persist traces
        out = save_traces([trace], tmp_path / "traces.jsonl")
        assert out.exists()
        # replay the saved InvocationRecord and re-score offline
        rec_path = tmp_path / "rec.json"
        rec_path.write_text(json.dumps(trace.record))
        card = replay_record(rec_path, reference=scn.ground_truth)
        assert card.exact_match and card.similarity == 1.0


# ── compare (evolve) ─────────────────────────────────────────────────


class TestCompare:
    def test_compare_variants_offline(self, scenarios):
        backend = MockBackend.from_patch("r", scenarios[0].ground_truth)
        report = compare_variants(
            scenarios,
            ["default", "graph"],
            backend=backend,
            model_name="m",
            baseline="default@m",
        )
        assert set(report.labels) == {"default@m", "graph@m"}
        table = report.render_table()
        assert "variant" in table and "baseline: default@m" in table
        assert report.best("mean_similarity") in report.labels


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_variants_command(self, capsys):
        from src.agents.harness.cli import main

        main(["variants"])
        out = capsys.readouterr().out
        assert "default" in out and "fingerprint" in out

    def test_debug_command_mock(self, capsys):
        from src.agents.harness.cli import main

        main(
            [
                "debug",
                "--fixtures",
                str(FIXTURES),
                "--backend",
                "mock",
                "--variant",
                "default",
                "--model",
                "m",
            ]
        )
        out = capsys.readouterr().out
        assert "Trace:" in out and "Scores" in out

    def test_eval_results_jsonl(self, capsys, tmp_path):
        from src.agents.harness.cli import main

        results = tmp_path / "results.jsonl"
        results.write_text(
            json.dumps(
                {
                    "query_cve": "CVE-1",
                    "generated_patch": "int x = 0;",
                    "ground_truth_patch": "int x = 0;",
                }
            )
            + "\n"
        )
        main(["eval", "--results", str(results)])
        out = capsys.readouterr().out
        assert '"n": 1' in out and '"parse_rate": 1.0' in out
