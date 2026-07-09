---
description: Run the pytest suite (skips live-API integration tests by default)
argument-hint: [optional pytest args, e.g. tests/test_embedders.py -k gin]
allowed-tools: Bash(uv run pytest:*)
---

Run the project tests with `uv`.

- If `$ARGUMENTS` is empty, run the fast unit suite: `uv run pytest -m "not integration" -q`
- Otherwise forward the arguments: `uv run pytest $ARGUMENTS`
- `pythonpath=src` is configured in `pyproject.toml`, so run from the repo root.
- Markers: `severity_high` (bug affects metrics), `severity_low` (edge-case robustness),
  `integration` (needs `AZURE_API_KEY` / `AZURE_API_BASEURL`).

Summarize failures concisely and propose a fix before changing code.
