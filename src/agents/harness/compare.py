"""
compare_variants — A/B (or N-way) prompt-variant comparison over a fixed suite.

The core of the *evolve* workflow: run several prompt variants (optionally with
different models) against the same set of scenarios and the same backend, then
tabulate aggregate metrics with deltas against a baseline. Because it accepts
any backend, you can evolve prompts entirely offline with a ReplayBackend or
MockBackend and get a reproducible comparison table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.backends import CompletionBackend
from src.agents.harness.evaluator import AgentEvaluator
from src.agents.harness.runner import AgentHarness, SuiteResult
from src.agents.harness.scenario import Scenario

# Metrics shown in the comparison table (key → header).
_TABLE_METRICS = [
    ("parse_rate", "parse"),
    ("exact_match_rate", "exact"),
    ("mean_similarity", "sim"),
    ("mean_bleu_4", "bleu4"),
    ("mean_rouge_l_f1", "rougeL"),
    ("mean_token_jaccard", "jacc"),
    ("expectations_pass_rate", "expect"),
]


@dataclass
class ComparisonReport:
    """Per-variant suite results plus a baseline for delta computation."""

    results: list[SuiteResult]
    labels: list[str]
    baseline_idx: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "baseline": self.labels[self.baseline_idx],
            "variants": [
                {"label": lbl, **res.summary}
                for lbl, res in zip(self.labels, self.results)
            ],
        }

    def best(self, metric: str = "mean_similarity") -> str:
        """Label of the variant with the highest value of *metric*."""
        pairs = [
            (lbl, res.summary.get(metric, 0.0))
            for lbl, res in zip(self.labels, self.results)
        ]
        return max(pairs, key=lambda p: p[1])[0]

    def render_table(self) -> str:
        base = self.results[self.baseline_idx].summary
        w = max((len(l) for l in self.labels), default=7) + 1
        header = (
            "variant".ljust(w)
            + "n   "
            + "  ".join(h.rjust(8) for _, h in _TABLE_METRICS)
        )
        lines = [header, "─" * len(header)]
        for i, (lbl, res) in enumerate(zip(self.labels, self.results)):
            s = res.summary
            cells = [lbl.ljust(w), str(s.get("n", 0)).ljust(4)]
            for key, _ in _TABLE_METRICS:
                val = s.get(key)
                if val is None:
                    cells.append("-".rjust(8))
                    continue
                cell = f"{val:.3f}"
                if i != self.baseline_idx and key in base and base.get(key) is not None:
                    delta = val - base[key]
                    sign = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
                    cell = f"{val:.3f}{sign}"
                cells.append(cell.rjust(8))
            lines.append(cells[0] + cells[1] + "  ".join(cells[2:]))
        lines.append(
            f"\nbaseline: {self.labels[self.baseline_idx]}   best(sim): {self.best()}"
        )
        return "\n".join(lines)


def _normalize(variants: list[str | dict]) -> list[dict]:
    out = []
    for v in variants:
        if isinstance(v, str):
            out.append({"variant": v, "model": None})
        else:
            out.append({"variant": v["variant"], "model": v.get("model")})
    return out


def compare_variants(
    scenarios: list[Scenario],
    variants: list[str | dict],
    *,
    backend: CompletionBackend | None = None,
    model_name: str | None = None,
    evaluator: AgentEvaluator | None = None,
    baseline: str | None = None,
    llm_params: dict | None = None,
    progress: bool = False,
) -> ComparisonReport:
    """Run each variant over *scenarios* and return a :class:`ComparisonReport`.

    Args:
        variants: variant names, or dicts ``{"variant": ..., "model": ...}``.
        backend:  shared CompletionBackend (mock / replay / live).
        baseline: label ``"variant"`` or ``"variant@model"`` used for deltas.
    """
    configs = _normalize(variants)
    ev = evaluator or AgentEvaluator()
    results: list[SuiteResult] = []
    labels: list[str] = []

    for cfg in configs:
        model = cfg["model"] or model_name
        label = cfg["variant"] if model is None else f"{cfg['variant']}@{model}"
        if progress:
            print(f"\n▶ variant={cfg['variant']} model={model or '(default)'}")
        harness = AgentHarness(
            evaluator=ev,
            model_name=model,
            prompt_variant=cfg["variant"],
            llm_params=llm_params,
        )
        res = harness.run_suite(scenarios, backend=backend, progress=progress)
        results.append(res)
        labels.append(label)

    baseline_idx = labels.index(baseline) if baseline in labels else 0
    return ComparisonReport(
        results=results,
        labels=labels,
        baseline_idx=baseline_idx,
        meta={"n_scenarios": len(scenarios), "n_variants": len(configs)},
    )


__all__ = ["ComparisonReport", "compare_variants"]
