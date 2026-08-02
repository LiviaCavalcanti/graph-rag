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

from experiments.base import Axis, CellContext, Experiment

# ── Experiment class ─────────────────────────────────────────────────


class PatchingExperiment(Experiment):
    """LLM patching experiment: retriever_mode × model_name grid."""

    def __init__(
        self,
        *,
        retriever_modes: list[str] | None = None,
        model_names: list[str] | None = None,
        architectures: list[str] | None = None,
        prompt_variants: list[str] | None = None,
        query_run: str | None = None,
        max_queries: int | None = None,
        batch_size: int = 10,
        resume: str | None = None,
        prompt_variant: str = "default",
        cve_filter: set[str] | None = None,
        cve_root: str | None = None,
        include_variants: bool | None = None,
        dataset: str = "autopatch",
    ):
        self._retriever_modes = retriever_modes or ["oracle"]
        self._model_names = model_names
        self._architectures = architectures
        self._prompt_variants = prompt_variants
        self._query_run = query_run
        self._max_queries = max_queries
        self._batch_size = batch_size
        self._resume = resume
        self._prompt_variant = prompt_variant
        self._cve_filter = cve_filter
        self._cve_root = cve_root
        self._include_variants = include_variants
        if dataset not in ("autopatch", "cvefixes", "cvefixes_file"):
            raise ValueError(
                f"Unknown dataset {dataset!r} for patching; expected "
                "'autopatch', 'cvefixes', or 'cvefixes_file'."
            )
        self._dataset = dataset

    @property
    def name(self) -> str:
        return "patching"

    def load_data(self, cfg: dict) -> dict[str, Any]:
        """Load lightweight pairs (no CPGs), split, and build a db_cache.

        ``dataset`` (constructor arg / ``--dataset`` CLI flag) selects the
        source, independent of ``cfg["data"]["active"]`` — which only
        controls the embedding/retrieval pipeline's datasets:

        - ``"autopatch"`` (default): AutoPatch CVE-list, db_entry.json fields
          (root_cause/fix_list/...) feed the prompt builder directly.
        - ``"cvefixes"``: CVEfixes SQLite DB. It has no db_entry.json, so
          db_cache is derived from the SAME pairs (cve_id/cwe_id/code only;
          root_cause/fix_list are unavailable and default to "Unknown"/"None"
          in the prompt builder). Every pair uses a single implicit variant
          (``"original"``), so retrieval decisions from a prior ``--mode
          query --dataset cvefixes`` run line up by (cve_id, variant).
        - ``"cvefixes_file"``: pinned CVEfixes subset loaded from a JSON
          entries file (``data.cvefixes_file.input_file``), with optional
          CVE-aware CWE resampling. Same db_cache derivation as
          ``"cvefixes"`` (no db_entry.json / root_cause / fix_list).

        Pairs and db_cache always come from the *same* loaded pairs so they
        can never silently diverge.
        """
        from experiments.common import build_split

        if self._dataset == "cvefixes":
            pairs, db_cache = self._load_cvefixes_data(cfg)
        elif self._dataset == "cvefixes_file":
            pairs, db_cache = self._load_cvefixes_file_data(cfg)
        else:
            pairs, db_cache = self._load_autopatch_data(cfg)

        # When using precomputed retrieval, the query JSONL was generated from
        # CPG-valid pairs only (resolve_pairs_from_entries skips missing CPGs),
        # while load_lightweight() above returned ALL entries.  Pairs absent
        # from the JSONL would all become skipped (no_example_found) after the
        # split.  Filter to the JSONL-covered subset BEFORE splitting so every
        # test pair is guaranteed to have a precomputed retrieval result.
        if self._query_run:
            from pathlib import Path as _Path
            from src.rag.precomputed import PrecomputedRetriever
            _qr = _Path(self._query_run)
            _results = _qr / "results.jsonl"
            if not _results.exists():
                _results = _qr / "retrieval_results.jsonl"
            if _results.exists():
                _pr = PrecomputedRetriever(_results)
                before = len(pairs)
                pairs = [p for p in pairs if _pr.retrieve(p)[0] is not None]
                db_cache = {k: v for k, v in db_cache.items()
                            if any(p.meta.get("dir_name") == k for p in pairs)}
                print(
                    f"  [precomputed filter] {before} → {len(pairs)} pairs "
                    f"(kept only those covered by the query JSONL)"
                )

        index_pairs, query_pairs, split_info = build_split(pairs, cfg)

        if self._cve_filter:
            query_pairs = [p for p in query_pairs if p.cve_id in self._cve_filter]
            print(f"CVE filter applied: {len(query_pairs)} query pairs remaining")

        if self._max_queries:
            query_pairs = query_pairs[: self._max_queries]

        return {
            "pairs": pairs,
            "index_pairs": index_pairs,
            "query_pairs": query_pairs,
            "split_info": split_info,
            "db_cache": db_cache,
        }

    def _load_autopatch_data(self, cfg: dict) -> tuple[list, dict]:
        """Load AutoPatch pairs (lightweight) + db_cache from db_entry.json.

        The CVE-list root is resolved in priority order:
        1. ``cve_root`` constructor arg / ``--cve-root`` CLI flag
        2. ``data.autopatch.root`` in config.yaml
        """
        from src.data.autopatch import AutoPatchDataset

        cve_root = self._resolve_cve_root(cfg)
        autopatch_cfg = cfg.get("data", {}).get("autopatch", {}) or {}
        include_variants = (
            self._include_variants
            if self._include_variants is not None
            else autopatch_cfg.get("include_variants", True)
        )

        dataset = AutoPatchDataset(
            {"root": str(cve_root), "include_variants": include_variants}
        )
        pairs = dataset.load_lightweight()
        print(f"Loaded {len(pairs)} lightweight AutoPatch pairs from {cve_root}")

        # db_cache (shared across all cells) is keyed by dir_name, from the
        # SAME root as pairs.
        db_cache = AutoPatchDataset.load_db_cache(cve_root)
        print(f"Cached {len(db_cache)} db_entries")
        return pairs, db_cache

    def _load_cvefixes_data(self, cfg: dict) -> tuple[list, dict]:
        """Load CVEfixes pairs (lightweight) + a db_cache derived from them."""
        from src.data.cvefixes import CVEFixesDataset

        cvefixes_cfg = cfg.get("data", {}).get("cvefixes", {}) or {}
        if not cvefixes_cfg.get("db_path"):
            raise KeyError(
                "No CVEfixes DB configured for patching. Set data.cvefixes.db_path "
                "in config.yaml."
            )

        dataset = CVEFixesDataset(cvefixes_cfg)
        pairs = dataset.load_lightweight()
        print(f"Loaded {len(pairs)} lightweight CVEfixes pairs")

        # CVEfixes has no db_entry.json; build an equivalent db_cache directly
        # from the same pairs' own metadata (single source of truth).
        db_cache = {
            p.meta["dir_name"]: {
                "cve_id": p.cve_id,
                "cwe_type": p.cwe_id,
                "function_name": p.func_name,
                "original_code": p.meta.get("source_before", ""),
                "vuln_patch": p.meta.get("source_after", ""),
                "root_cause": "",
                "fix_list": [],
            }
            for p in pairs
        }
        print(f"Cached {len(db_cache)} db_entries (derived from CVEfixes pairs)")
        return pairs, db_cache

    def _load_cvefixes_file_data(self, cfg: dict) -> tuple[list, dict]:
        """Load CVEfixes-file pairs (lightweight) + a db_cache derived from them."""
        from src.data.cvefixes_file import CVEFixesFileDataset

        cvefixes_file_cfg = cfg.get("data", {}).get("cvefixes_file", {}) or {}

        dataset = CVEFixesFileDataset(cvefixes_file_cfg)
        pairs = dataset.load_lightweight()
        print(f"Loaded {len(pairs)} lightweight CVEfixes-file pairs")

        # CVEfixes-file has no db_entry.json; build an equivalent db_cache
        # directly from the same pairs' own metadata (single source of truth).
        db_cache = {
            p.meta["dir_name"]: {
                "cve_id": p.cve_id,
                "cwe_type": p.cwe_id,
                "function_name": p.func_name,
                "original_code": p.meta.get("source_before", ""),
                "vuln_patch": p.meta.get("source_after", ""),
                "root_cause": "",
                "fix_list": [],
            }
            for p in pairs
        }
        print(
            f"Cached {len(db_cache)} db_entries (derived from CVEfixes-file pairs)"
        )
        return pairs, db_cache

    def _resolve_cve_root(self, cfg: dict) -> Path:
        """Resolve and validate the AutoPatch CVE-list root directory."""
        autopatch_cfg = cfg.get("data", {}).get("autopatch", {}) or {}
        root = self._cve_root or autopatch_cfg.get("root")
        if not root:
            raise KeyError(
                "No CVE-list root configured for patching. Set data.autopatch.root "
                "in config.yaml, pass --cve-root on the CLI, or "
                "PatchingExperiment(cve_root=...)."
            )
        path = Path(root)
        if not path.is_dir():
            raise FileNotFoundError(f"CVE-list root directory not found: {path}")
        return path

    def axes(self, cfg: dict) -> list[Axis]:
        agent_cfg = cfg.get("agents", {})
        architectures = self._architectures or [
            agent_cfg.get("architecture", "single_turn")
        ]
        prompt_variants = self._prompt_variants or [
            self._prompt_variant
            or agent_cfg.get("prompt_variant")
            or cfg.get("rag", {}).get("prompt_variant", "default")
        ]
        model_names = self._model_names or [
            os.getenv("MODEL_NAME", agent_cfg.get("model") or "gpt-4o")
        ]
        return [
            Axis("architecture", architectures, description="Agent architecture"),
            Axis("prompt_variant", prompt_variants, description="Prompt variant"),
            Axis(
                "retriever_mode",
                self._retriever_modes,
                description="Retrieval strategy",
            ),
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
        architecture = ctx.coords["architecture"]
        prompt_variant = ctx.coords["prompt_variant"]
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

        cell_tag = f"{architecture}__{prompt_variant}__{retriever_mode}__{model_name}"
        cell_output_dir = ctx.run_dir / cell_tag
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
            prompt_variant=prompt_variant,
            architecture=architecture,
        )

        # ── aggregate metrics from JSONL ─────────────────────────────
        metrics = self._aggregate_cell_metrics(run_dir)

        return {
            "architecture": architecture,
            "prompt_variant": prompt_variant,
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

        avg_rouge = (
            {f"avg_{k}": round(v / rouge_count, 4) for k, v in rouge_sums.items()}
            if rouge_count
            else {}
        )

        return {
            "n_success": total,
            "avg_similarity": round(sum(similarities) / total, 4),
            "exact_match_rate": round(exact_matches / total, 4),
            **avg_rouge,
        }

    def on_cell_error(
        self, ctx: CellContext, error: Exception
    ) -> dict[str, Any] | None:
        """Return partial result on error rather than crashing the grid."""
        return {
            "architecture": ctx.coords.get("architecture"),
            "prompt_variant": ctx.coords.get("prompt_variant"),
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
    architecture: str = "single_turn",
    cve_filter: set[str] | None = None,
    cve_root: str | None = None,
    dataset: str = "autopatch",
) -> Path:
    """Run the LLM patching experiment.

    Drop-in replacement for agent_experiment.run_patching_experiment().
    """
    model_names = [model_name] if model_name else None

    exp = PatchingExperiment(
        retriever_modes=[retriever_mode],
        model_names=model_names,
        architectures=[architecture],
        prompt_variants=[prompt_variant],
        query_run=query_run,
        max_queries=max_queries,
        batch_size=batch_size,
        resume=resume,
        prompt_variant=prompt_variant,
        cve_filter=cve_filter,
        cve_root=cve_root,
        dataset=dataset,
    )

    output = exp.run(cfg, output_dir=output_dir) if output_dir else exp.run(cfg)

    # Return the cell's output directory (backwards compat)
    if output.cells and output.cells[0].metrics.get("output_dir"):
        return Path(output.cells[0].metrics["output_dir"])
    return output.run_dir
