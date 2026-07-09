---
description: Format the codebase with isort + black
allowed-tools: Bash(uv run isort:*), Bash(uv run black:*)
---

Format Python sources using the project's configured tools:

```bash
uv run isort .
uv run black .
```

Run this after edits and before committing. Both tools are declared in `pyproject.toml`.
