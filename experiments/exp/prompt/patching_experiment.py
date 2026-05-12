"""
Patching experiment — LLM-based patch generation using the Experiment abstraction.

Reimplements run_patching_experiment() from agent_experiment.py. Separates:
  - Core logic: retriever construction + batch LLM inference (run_cell)
  - Data: lightweight pair loading + split + db_cache (load_data)
  - I/O: JSONL streaming handled by run_batch_inference, summary in after_run

Batchable: each cell is one (retriever_mode, model_name) combination.
Adding new retrievers or models only requires extending the axes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from experiments.base import Axis, CellContext, Experiment, ExperimentOutput


# ── Experiment class ─────────────────────────────────────────────────


class PatchingExperiment(Experiment):
    """LLM patching experiment: retriever_mode × model_name grid."""

    def __init__(
        self,
        *,
        retriever_modes: list[str] | None = None,
        model_names: list[str] | None = None,
        query_run: str | None = None,
        max_queries: int | None = None,
        batch_size: int = 10,
        resume: str | None = None,
        prompt_variant: str = "default",
        cve_filter: set[str] | None = None,
    ):
        self._retriever_modes = retriever_modes or ["oracle"]
        self._model_names = model_names
        self._query_run = query_run
        self._max_queries = max_queries
        self._batch_size = batch_size
        self._resume = resume
        self._prompt_variant = prompt_variant
        self._cve_filter = cve_filter

    @property
    def name(self) -> str:
        return "patching"

    def load_data(self, cfg: dict) -> dict[str, Any]:
        """Load lightweight pairs (no CPGs), split, and load db_cache."""
        from src.data.autopatch import AutoPatchDataset, load_pairs_lightweight
        from experiments.common import build_split

        pairs = load_pairs_lightweight(cfg)
        print(f"Loaded {len(pairs)} lightweight pairs (no CPGs)")
        index_pairs, query_pairs, split_info = build_split(pairs, cfg)

        if self._cve_filter:
            query_pairs = [p for p in query_pairs if p.cve_id in self._cve_filter]
            print(f"CVE filter applied: {len(query_pairs)} query pairs remaining")

        if self._max_queries:
            query_pairs = query_pairs[: self._max_queries]

        # Load db_cache (shared across all cells)
        cve_root = Path(cfg["data"]["autopatch"]["root"])
        db_cache = AutoPatchDataset.load_db_cache(cve_root)
        print(f"Cached {len(db_cache)} db_entries")

        return {
            "pairs": pairs,
            "index_pairs": index_pairs,
            "query_pairs": query_pairs,
            "split_info": split_info,
            "db_cache": db_cache,
        }

    def axes(self, cfg: dict) -> list[Axis]:
        model_names = self._model_names or [
            os.getenv("MODEL_NAME", cfg.get("agents", {}).get("model", "gpt-4o"))
        ]
        return [
            Axis("retriever_mode", self._retriever_modes, description="Retrieval strategy"),
            Axis("model_name", model_names, description="LLM model/deployment"),
        ]

    def before_run(self, ctx: CellContext) -> None:
        """Validate environment variables before starting."""
        from dotenv import load_dotenv

        load_dotenv()

        if not os.getenv("AZURE_API_KEY") or not os.getenv("AZURE_API_BASEURL"):
            print("ERROR: Set AZURE_API_KEY and AZURE_API_BASEURL in .env")
            sys.exit(1)

    def run_cell(self, ctx: CellContext) -> dict[str, Any]:
        retriever_mode = ctx.coords["retriever_mode"]
        model_name = ctx.coords["model_name"]
        index_pairs = ctx.data["index_pairs"]
        query_pairs = ctx.data["query_pairs"]
        db_cache = ctx.data["db_cache"]
        split_info = ctx.data["split_info"]

        # ── build retriever ──────────────────────────────────────────
        retriever = self._build_retriever(retriever_mode, index_pairs)

        # ── run batch inference ──────────────────────────────────────
        from src.agents.batch_inference import run_batch_inference

        cell_output_dir = ctx.run_dir / f"{retriever_mode}__{model_name}"
        cell_output_dir.mkdir(parents=True, exist_ok=True)

        run_dir = run_batch_inference(
            query_pairs=query_pairs,
            retriever=retriever,
            db_cache=db_cache,
            model_name=model_name,
            batch_size=self._batch_size,
            run_tag=f"batch_{retriever_mode}",
            resume_dir=self._resume,
            meta_extra={"mode": retriever_mode, "split_info": split_info},
            output_dir=cell_output_dir,
            prompt_variant=self._prompt_variant,
        )

        # ── aggregate metrics from JSONL ─────────────────────────────
        metrics = self._aggregate_cell_metrics(run_dir)

        return {
            "retriever_mode": retriever_mode,
            "model_name": model_name,
            "n_queries": len(query_pairs),
            "output_dir": str(run_dir),
            **metrics,
        }

    def _build_retriever(self, mode: str, index_pairs: list):
        """Construct retriever based on mode."""
        if mode == "oracle":
            from src.rag.oracle import OracleRetriever

            retriever = OracleRetriever(index_pairs)
            print(f"Oracle retriever built from {len(index_pairs)} index pairs")
            return retriever

        elif mode == "precomputed":
            from src.rag.precomputed import PrecomputedRetriever

            if not self._query_run:
                print("ERROR: query_run required for precomputed mode")
                sys.exit(1)

            query_results = Path(self._query_run) / "retrieval_results.jsonl"
            if not query_results.exists():
                query_results = Path(self._query_run) / "results.jsonl"
            if not query_results.exists():
                print(f"ERROR: {query_results} not found")
                sys.exit(1)

            return PrecomputedRetriever(query_results)

        else:
            raise ValueError(f"Unknown retriever_mode: {mode!r}")

    def _aggregate_cell_metrics(self, run_dir: Path) -> dict[str, Any]:
        """Read JSONL results and compute aggregate similarity/ROUGE metrics."""
        import json

        results_file = run_dir / "results.jsonl"
        if not results_file.exists():
            return {}

        similarities = []
        exact_matches = 0
        rouge_sums: dict[str, float] = {}
        rouge_count = 0
        total = 0

        with open(results_file) as f:
            for line in f:
                row = json.loads(line)
                if row.get("status") != "success":
                    continue
                total += 1
                similarities.append(row.get("similarity", 0.0))
                if row.get("exact_match"):
                    exact_matches += 1
                rouge = row.get("rouge", {})
                if rouge:
                    rouge_count += 1
                    for k, v in rouge.items():
                        rouge_sums[k] = rouge_sums.get(k, 0.0) + v

        if total == 0:
            return {"n_success": 0}

        avg_rouge = {f"avg_{k}": round(v / rouge_count, 4) for k, v in rouge_sums.items()} if rouge_count else {}

        return {
            "n_success": total,
            "avg_similarity": round(sum(similarities) / total, 4),
            "exact_match_rate": round(exact_matches / total, 4),
            **avg_rouge,
        }

    def on_cell_error(self, ctx: CellContext, error: Exception) -> dict[str, Any] | None:
        """Return partial result on error rather than crashing the grid."""
        return {
            "retriever_mode": ctx.coords.get("retriever_mode"),
            "model_name": ctx.coords.get("model_name"),
            "status": "error",
            "error": str(error),
        }


# ── Entry point (backwards-compatible) ───────────────────────────────


def run_patching_experiment(
    cfg: dict,
    *,
    retriever_mode: str = "oracle",
    model_name: str | None = None,
    query_run: str | None = None,
    max_queries: int | None = None,
    batch_size: int = 10,
    resume: str | None = None,
    output_dir: Path | None = None,
    prompt_variant: str = "default",
    cve_filter: set[str] | None = None,
) -> Path:
    """Run the LLM patching experiment.

    Drop-in replacement for agent_experiment.run_patching_experiment().
    """
    model_names = [model_name] if model_name else None

    exp = PatchingExperiment(
        retriever_modes=[retriever_mode],
        model_names=model_names,
        query_run=query_run,
        max_queries=max_queries,
        batch_size=batch_size,
        resume=resume,
        prompt_variant=prompt_variant,
        cve_filter=cve_filter,
    )

    output = exp.run(cfg, output_dir=output_dir) if output_dir else exp.run(cfg)

    # Return the cell's output directory (backwards compat)
    if output.cells and output.cells[0].metrics.get("output_dir"):
        return Path(output.cells[0].metrics["output_dir"])
    return output.run_dir
