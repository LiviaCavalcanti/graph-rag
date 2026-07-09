---
description: Evaluate a batch run directory (patch + retrieval metrics + HTML dashboards)
argument-hint: <run_dir, e.g. experiments/output/<run_id>/>
allowed-tools: Bash(uv run python -m src.evaluate:*)
---

Run `uv run python -m src.evaluate $ARGUMENTS`.

- Pass a run directory (auto-finds `results.jsonl`) or the `results.jsonl` file directly.
- Produces: `evaluation.jsonl`, `evaluation_summary.json`, `retrieval_eval_summary.json`,
  `evaluation_dashboard.html`, `patch_analysis.html`.
- When adding/altering metrics, first read the **eval-dashboard** skill
  ([.github/skills/eval-dashboard/SKILL.md](.github/skills/eval-dashboard/SKILL.md)).
