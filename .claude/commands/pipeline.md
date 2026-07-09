---
description: Run a graph-rag pipeline stage (export | index | query | batch | full)
argument-hint: <export|index|query|batch|full> [extra flags]
allowed-tools: Bash(uv run python main.py:*)
---

Run `uv run python main.py --mode $ARGUMENTS`.

Stages (see the [main.py](main.py) docstring):

- `export` — run Joern to generate CPGs (requires Joern installed; `joern.bin_dir` in config).
- `index` — embed CPGs and build the FAISS index.
- `query` — retrieve similar CVEs (add `--cve CVE-YYYY-NNNNN` for a single query).
- `batch` — LLM patching (requires `AZURE_API_KEY` / `AZURE_API_BASEURL` in `.env`; add `--model` / `--max-queries`).
- `full` — retrieval → patching → evaluation.

Confirm the target dataset in `config.yaml` (`data.active`) before running `export`/`index`.
