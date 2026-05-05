# Agents

## Environment

This project uses **uv** as the Python package manager and virtual environment tool.

- Use `uv` to install dependencies and manage the virtual environment.
- Run commands with `uv run` or activate the `.venv` created by `uv`.

## Models

- When a language model is needed, use the **local CodeBERT** model located at `models/codebert-base/`.
- Do not download or use a different model without asking the user first.

## Experiments

- All new experiments **must** subclass `Experiment` from `experiments/base.py`.
- Implement the required abstract members: `name`, `axes(cfg)`, and `run_cell(ctx)`.
- Use the optional hooks (`load_data`, `metrics`, `before_run`, `after_run`, `on_cell_error`) as needed.
- Do not create standalone experiment scripts that bypass the base class.
