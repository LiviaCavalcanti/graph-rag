"""
Scenario — a single, self-contained agent test case.

A :class:`Scenario` carries everything ``patch_one`` needs to run (retrieved
example + target function + graph context) plus the ground-truth patch used to
score the result. Scenarios are the unit of work for every harness command:
debug, replay, eval, and compare.

Fixtures are plain JSON so they can be committed, diffed, and hand-authored::

    {
      "scenarios": [
        {
          "id": "cwe476-null-deref",
          "cve_id": "CVE-2025-0001",
          "cwe_id": "CWE-476",
          "example_db": { "cve_id": "...", "original_code": "...", "vuln_patch": "..." },
          "target_db":  { "cve_id": "...", "original_code": "..." },
          "target_code": "int f(...) { ... }",
          "ground_truth": "int f(...) { if (p) ... }",
          "graph_context": "None",
          "expected": { "min_similarity": 0.6, "contains": ["if"] }
        }
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Scenario:
    """One agent test case: inputs for ``patch_one`` plus the reference patch."""

    id: str
    example_db: dict[str, Any]
    target_db: dict[str, Any]
    target_code: str
    ground_truth: str = ""
    target_supplementary: str = ""
    graph_context: str = "None"
    cve_id: str = ""
    cwe_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    # Optional golden expectations checked by the test/eval harness.
    expected: dict[str, Any] = field(default_factory=dict)

    # ── serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cve_id": self.cve_id,
            "cwe_id": self.cwe_id,
            "example_db": self.example_db,
            "target_db": self.target_db,
            "target_code": self.target_code,
            "target_supplementary": self.target_supplementary,
            "ground_truth": self.ground_truth,
            "graph_context": self.graph_context,
            "meta": self.meta,
            "expected": self.expected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            id=d["id"],
            example_db=d.get("example_db", {}),
            target_db=d.get("target_db", {}),
            target_code=d.get("target_code", ""),
            ground_truth=d.get("ground_truth", ""),
            target_supplementary=d.get("target_supplementary", ""),
            graph_context=d.get("graph_context", "None"),
            cve_id=d.get("cve_id", "") or d.get("target_db", {}).get("cve_id", ""),
            cwe_id=d.get("cwe_id", "") or d.get("target_db", {}).get("cwe_type", ""),
            meta=d.get("meta", {}),
            expected=d.get("expected", {}),
        )

    # ── constructors from live data ──────────────────────────────────

    @classmethod
    def from_pair(
        cls,
        query_pair,
        example_pair,
        db_cache: dict,
        *,
        target_code: str,
        ground_truth: str,
        graph_context: str = "None",
    ) -> "Scenario":
        """Build a Scenario from a retrieved (query, example) FunctionPair pair.

        ``db_cache`` is keyed by ``dir_name`` (see AutoPatchDataset.load_db_cache).
        """
        target_dir = query_pair.meta.get("dir_name", "")
        example_dir = example_pair.meta.get("dir_name", "")
        return cls(
            id=f"{query_pair.cve_id}:{query_pair.meta.get('variant', '')}",
            example_db=db_cache.get(example_dir, {}),
            target_db=db_cache.get(target_dir, {}),
            target_code=target_code,
            ground_truth=ground_truth,
            target_supplementary=query_pair.meta.get("supplementary_code", ""),
            graph_context=graph_context,
            cve_id=query_pair.cve_id,
            cwe_id=query_pair.cwe_id,
            meta={
                "variant": query_pair.meta.get("variant", ""),
                "example_cve": example_pair.cve_id,
            },
        )


# ── fixture I/O ──────────────────────────────────────────────────────


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load scenarios from a JSON file (``{"scenarios": [...]}`` or a bare list)
    or from a directory of ``*.json`` fixture files."""
    p = Path(path)
    if p.is_dir():
        scenarios: list[Scenario] = []
        for f in sorted(p.glob("*.json")):
            scenarios.extend(load_scenarios(f))
        return scenarios

    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("scenarios", [])
    return [Scenario.from_dict(d) for d in data]


def save_scenarios(scenarios: list[Scenario], path: str | Path) -> Path:
    """Write scenarios to a JSON fixture file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {"scenarios": [s.to_dict() for s in scenarios]}, indent=2, default=str
        )
    )
    return p


__all__ = ["Scenario", "load_scenarios", "save_scenarios"]
