"""
AgentEvaluator — one unified, offline-first scorecard for a generated patch.

Ties together three evaluation signals that were previously scattered across
``src.evaluate``:

    1. Text/code similarity vs the ground-truth patch (always on, no network):
       exact match, code_similarity, BLEU-4, ROUGE-L, token Jaccard, edit dist.
    2. Golden expectations declared on a Scenario (thresholds / substrings) —
       turns any scenario into a pass/fail regression test.
    3. Optional LLM-as-judge verdict (FIXED / PARTIAL / NOT_FIXED), routed
       through a CompletionBackend so it too can be mocked or replayed offline.

Only (1) and (2) run by default; (3) is opt-in via ``judge=True``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from statistics import mean

from src.agents.harness.scenario import Scenario
from src.metrics.similarity import (
    bleu_score,
    code_similarity,
    exact_match,
    normalised_edit_distance,
    normalised_exact_match,
    rouge_scores,
    token_jaccard,
)


def _strip_c_comments(code: str) -> str:
    """Remove C/C++ block and line comments (self-contained; no heavy imports)."""
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//[^\n]*", "", code)
    return re.sub(r"\n\s*\n", "\n\n", code)


@dataclass
class ScoreCard:
    """All metrics + verdict for one generated patch."""

    scenario_id: str = ""
    parsed: bool = False
    exact_match: bool = False
    normalised_exact: bool = False
    similarity: float = 0.0
    bleu_4: float = 0.0
    rouge_l_f1: float = 0.0
    token_jaccard: float = 0.0
    edit_distance_norm: float = 0.0
    generated_len: int = 0
    reference_len: int = 0
    # golden expectations
    expectations_checked: bool = False
    expectations_passed: bool = True
    expectation_failures: list[str] = field(default_factory=list)
    # optional LLM judge
    judge_verdict: str | None = None
    judge_confidence: float | None = None
    judge_reasoning: str | None = None

    @property
    def passed(self) -> bool:
        """True when the patch parsed and all declared expectations hold."""
        return self.parsed and self.expectations_passed

    def to_dict(self) -> dict:
        return asdict(self)


class AgentEvaluator:
    """Score generated patches against ground truth and golden expectations."""

    def __init__(
        self,
        *,
        judge: bool = False,
        judge_model: str | None = None,
        judge_backend=None,
        strip_comments: bool = True,
    ):
        self.judge_enabled = judge
        self.judge_model = judge_model
        self.judge_backend = judge_backend
        self.strip_comments = strip_comments

    # ── pure text metrics (offline) ──────────────────────────────────

    def score(
        self,
        generated: str,
        reference: str,
        *,
        expected: dict | None = None,
        scenario_id: str = "",
    ) -> ScoreCard:
        gen = (generated or "").strip()
        ref = (reference or "").strip()
        if self.strip_comments:
            gen_cmp = _strip_c_comments(gen).strip()
            ref_cmp = _strip_c_comments(ref).strip()
        else:
            gen_cmp, ref_cmp = gen, ref

        card = ScoreCard(scenario_id=scenario_id, parsed=bool(gen))
        card.generated_len = len(gen)
        card.reference_len = len(ref)

        if gen and ref:
            card.exact_match = exact_match(gen_cmp, ref_cmp)
            card.normalised_exact = normalised_exact_match(gen_cmp, ref_cmp)
            card.similarity = round(code_similarity(gen_cmp, ref_cmp), 4)
            card.bleu_4 = round(bleu_score(gen_cmp, ref_cmp), 4)
            card.rouge_l_f1 = round(rouge_scores(gen_cmp, ref_cmp)["rougeL_f1"], 4)
            card.token_jaccard = round(token_jaccard(gen_cmp, ref_cmp), 4)
            card.edit_distance_norm = round(
                normalised_edit_distance(gen_cmp, ref_cmp), 4
            )

        if expected:
            passed, failures = self._check_expectations(gen, card, expected)
            card.expectations_checked = True
            card.expectations_passed = passed
            card.expectation_failures = failures

        return card

    def score_scenario(self, scenario: Scenario, generated: str) -> ScoreCard:
        """Score one scenario; run the LLM judge when enabled."""
        card = self.score(
            generated,
            scenario.ground_truth,
            expected=scenario.expected or None,
            scenario_id=scenario.id,
        )
        if self.judge_enabled and generated:
            verdict = self.judge(scenario, generated)
            card.judge_verdict = verdict.get("verdict")
            card.judge_confidence = verdict.get("confidence")
            card.judge_reasoning = verdict.get("reasoning")
        return card

    # ── golden expectations ──────────────────────────────────────────

    @staticmethod
    def _check_expectations(
        generated: str, card: ScoreCard, expected: dict
    ) -> tuple[bool, list[str]]:
        failures: list[str] = []
        thresholds = {
            "min_similarity": card.similarity,
            "min_bleu": card.bleu_4,
            "min_rouge_l": card.rouge_l_f1,
            "min_jaccard": card.token_jaccard,
        }
        for key, actual in thresholds.items():
            if key in expected and actual < expected[key]:
                failures.append(f"{key}: {actual:.4f} < {expected[key]}")

        if (
            "max_edit_distance" in expected
            and card.edit_distance_norm > expected["max_edit_distance"]
        ):
            failures.append(
                f"max_edit_distance: {card.edit_distance_norm:.4f} > {expected['max_edit_distance']}"
            )
        if expected.get("exact_match") and not card.exact_match:
            failures.append("exact_match expected but not achieved")
        if expected.get("must_parse", False) and not card.parsed:
            failures.append("must_parse expected but patch was empty")

        for needle in expected.get("contains", []):
            if needle not in generated:
                failures.append(f"missing required substring: {needle!r}")
        for needle in expected.get("not_contains", []):
            if needle in generated:
                failures.append(f"forbidden substring present: {needle!r}")

        return (not failures), failures

    # ── optional LLM-as-judge (backend-routed) ───────────────────────

    def judge(self, scenario: Scenario, generated: str) -> dict:
        """Ask an LLM whether *generated* actually fixes the vulnerability.

        Routed through a CompletionBackend so it can be mocked / replayed.
        """
        from src.agents.backends import get_default_backend
        from src.evaluate.llm_evaluation import (
            SYSTEM_PROMPT,
            USER_PROMPT_TEMPLATE,
            parse_llm_response,
        )

        db = scenario.target_db
        user = USER_PROMPT_TEMPLATE.format(
            cve_id=scenario.cve_id or db.get("cve_id", "Unknown"),
            cwe_type=scenario.cwe_id or db.get("cwe_type", "Unknown"),
            function_name=db.get("function_name", "unknown"),
            function_prototype=db.get("function_prototype", ""),
            language=db.get("language", "c"),
            vulnerable_code=scenario.target_code,
            patched_code=generated,
        )
        backend = self.judge_backend or get_default_backend()
        model = self.judge_model or "gpt-4o"
        result = backend.complete(
            model=f"azure/{model}",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        try:
            return parse_llm_response(result.content)
        except Exception as exc:  # noqa: BLE001 - judge is best-effort
            return {"verdict": "ERROR", "confidence": 0.0, "reasoning": str(exc)}

    # ── suite aggregation ────────────────────────────────────────────

    @staticmethod
    def aggregate(cards: list[ScoreCard]) -> dict:
        """Suite-level summary over a list of scorecards."""
        n = len(cards)
        if n == 0:
            return {"n": 0}
        parsed = [c for c in cards if c.parsed]
        checked = [c for c in cards if c.expectations_checked]

        def avg(attr: str, source: list[ScoreCard]) -> float:
            vals = [getattr(c, attr) for c in source]
            return round(mean(vals), 4) if vals else 0.0

        summary = {
            "n": n,
            "parse_rate": round(len(parsed) / n, 4),
            "exact_match_rate": (
                round(sum(c.exact_match for c in parsed) / len(parsed), 4)
                if parsed
                else 0.0
            ),
            "mean_similarity": avg("similarity", parsed),
            "mean_bleu_4": avg("bleu_4", parsed),
            "mean_rouge_l_f1": avg("rouge_l_f1", parsed),
            "mean_token_jaccard": avg("token_jaccard", parsed),
        }
        if checked:
            summary["expectations_checked"] = len(checked)
            summary["expectations_pass_rate"] = round(
                sum(c.expectations_passed for c in checked) / len(checked), 4
            )
        judged = [c for c in cards if c.judge_verdict]
        if judged:
            verdicts: dict[str, int] = {}
            for c in judged:
                verdicts[c.judge_verdict] = verdicts.get(c.judge_verdict, 0) + 1
            summary["judge_verdicts"] = verdicts
        return summary


__all__ = ["ScoreCard", "AgentEvaluator"]
