"""
Experiment base abstraction.

Every experiment is: Data × Variations → Pipeline → Metrics → Output.

Subclasses define:
  - name: short identifier for output directories
  - axes(): independent variables to vary (the grid dimensions)
  - metrics(): what to measure per cell
  - run_cell(): the actual work for one point in the grid

The base class handles: run directory creation, grid iteration,
result serialization, metadata collection, and hook points.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable

from experiments.common import build_split, save_json
from src.data.autopatch import load_pairs
from src.io import make_run_dir

OUTPUT_DIR = Path("experiments/output")


# ── value objects ────────────────────────────────────────────────────


@dataclass
class Axis:
    """One independent variable to vary in the experiment grid."""

    name: str
    values: list[Any]
    description: str = ""


@dataclass
class MetricSpec:
    """A metric to compute on each experimental cell.

    Parameters
    ----------
    name : str
        Key under which the result is stored.
    fn : callable
        Signature: fn(cell_context: CellContext) -> dict | float
    requires : list[str]
        Keys that must be present in CellContext.artifacts before this
        metric can run. Metrics are evaluated in declaration order; if
        a required artifact is missing, the metric is skipped (if optional)
        or raises.
    optional : bool
        If True, skip silently when prerequisites are missing.
    """

    name: str
    fn: Callable[["CellContext"], dict | float]
    requires: list[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class CellContext:
    """State passed into run_cell and metric functions."""

    coords: dict[str, Any]  # axis_name → current value
    data: dict[str, Any]  # from load_data()
    cfg: dict
    run_dir: Path
    cache: dict[str, Any]  # shared mutable cache across cells
    artifacts: dict[str, Any] = field(default_factory=dict)  # per-cell artifacts


@dataclass
class CellResult:
    """Output of one grid cell."""

    coords: dict[str, Any]
    metrics: dict[str, Any]
    timing_s: float
    error: str | None = None

    def coords_as_str(self) -> dict[str, str]:
        """Coords with values converted to strings (for flat serialization)."""
        return {k: (v.name if hasattr(v, "name") else str(v)) for k, v in self.coords.items()}


@dataclass
class ExperimentOutput:
    """Full output of one experiment run."""

    run_id: str
    run_dir: Path
    metadata: dict[str, Any]
    cells: list[CellResult]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "metadata": self.metadata,
            "cells": [
                {
                    "coords": c.coords_as_str(),
                    "metrics": c.metrics,
                    "timing_s": c.timing_s,
                    **({"error": c.error} if c.error else {}),
                }
                for c in self.cells
            ],
        }


# ── base class ───────────────────────────────────────────────────────


class Experiment(ABC):
    """Base class for all grid-style experiments."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used for output directory naming."""
        ...

    @abstractmethod
    def axes(self, cfg: dict) -> list[Axis]:
        """Define the independent variables (grid dimensions)."""
        ...

    @abstractmethod
    def run_cell(self, ctx: CellContext) -> dict[str, Any]:
        """Execute one grid cell. Return metrics dict.

        May also populate ctx.artifacts for downstream MetricSpecs.
        """
        ...

    # ── optional declarative metrics ─────────────────────────────────

    def metrics(self) -> list[MetricSpec]:
        """Declarative metrics evaluated after run_cell.

        Override to add metrics that are computed uniformly across
        experiment types. By default, none — all metrics come from
        run_cell's return value.
        """
        return []

    # ── hooks ────────────────────────────────────────────────────────

    def load_data(self, cfg: dict) -> dict[str, Any]:
        """Load and prepare data. Default: load_pairs + build_split."""
        pairs = load_pairs(cfg)
        index_pairs, query_pairs, split_info = build_split(pairs, cfg)
        return {
            "pairs": pairs,
            "index_pairs": index_pairs,
            "query_pairs": query_pairs,
            "split_info": split_info,
        }

    def before_run(self, ctx: CellContext) -> None:
        """Called once before grid iteration starts."""
        pass

    def after_run(self, output: ExperimentOutput) -> None:
        """Called after all cells complete (e.g., generate dashboard)."""
        pass

    def on_cell_error(self, ctx: CellContext, error: Exception) -> dict[str, Any] | None:
        """Called when run_cell raises. Return a partial metrics dict or None to propagate."""
        return None

    # ── orchestration (not overridden) ───────────────────────────────

    def run(self, cfg: dict, output_dir: Path = OUTPUT_DIR) -> ExperimentOutput:
        """Run the full experiment grid."""
        run_id, run_dir = make_run_dir(self.name, output_dir=output_dir)

        data = self.load_data(cfg)

        cache: dict[str, Any] = {}
        # Provide a context for before_run (coords empty at this stage)
        setup_ctx = CellContext(
            coords={}, data=data, cfg=cfg, run_dir=run_dir, cache=cache
        )
        self.before_run(setup_ctx)

        axes = self.axes(cfg)
        axis_names = [a.name for a in axes]
        axis_values = [a.values for a in axes]

        n_total = 1
        for a in axes:
            n_total *= len(a.values)

        print(f"\n{'='*60}")
        print(f"Experiment: {self.name}  run_id={run_id}")
        print(f"Grid: {' × '.join(f'{a.name}({len(a.values)})' for a in axes)} = {n_total} cells")
        print(f"Data: {len(data.get('pairs', []))} pairs, "
              f"index={len(data.get('index_pairs', []))}, "
              f"query={len(data.get('query_pairs', []))}")
        print(f"{'='*60}")

        cells: list[CellResult] = []
        declared_metrics = self.metrics()

        for i, combo in enumerate(product(*axis_values), 1):
            coords = dict(zip(axis_names, combo))
            ctx = CellContext(
                coords=coords, data=data, cfg=cfg, run_dir=run_dir, cache=cache
            )

            coord_str = "  ".join(f"{k}={_short_repr(v)}" for k, v in coords.items())
            print(f"\n  [{i}/{n_total}] {coord_str}")

            t0 = time.perf_counter()
            try:
                cell_metrics = self.run_cell(ctx)

                # Run declarative metrics
                for spec in declared_metrics:
                    if spec.requires and not all(
                        k in ctx.artifacts for k in spec.requires
                    ):
                        if spec.optional:
                            continue
                        raise RuntimeError(
                            f"Metric '{spec.name}' requires {spec.requires} "
                            f"but artifacts has {list(ctx.artifacts.keys())}"
                        )
                    result = spec.fn(ctx)
                    cell_metrics[spec.name] = result

                elapsed = time.perf_counter() - t0
                cells.append(CellResult(coords=coords, metrics=cell_metrics, timing_s=elapsed))

            except Exception as e:
                elapsed = time.perf_counter() - t0
                fallback = self.on_cell_error(ctx, e)
                if fallback is not None:
                    cells.append(CellResult(
                        coords=coords, metrics=fallback, timing_s=elapsed, error=str(e)
                    ))
                else:
                    print(f"    ERROR: {e}")
                    cells.append(CellResult(
                        coords=coords, metrics={}, timing_s=elapsed, error=str(e)
                    ))

        output = ExperimentOutput(
            run_id=run_id,
            run_dir=run_dir,
            metadata={
                "experiment": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": cfg,
                "axes": [{"name": a.name, "n_values": len(a.values), "description": a.description} for a in axes],
                "data_info": _data_summary(data),
            },
            cells=cells,
        )

        self.after_run(output)

        # Write canonical results (after after_run, so subclasses that
        # write their own results.json in after_run take precedence)
        results_path = run_dir / "results.json"
        if not results_path.exists():
            results_path.write_text(json.dumps(output.to_dict(), indent=2, default=str))
            print(f"\nResults written → {results_path}")

        return output


# ── helpers ──────────────────────────────────────────────────────────


def _short_repr(v: Any) -> str:
    """Short string repr for printing axis values."""
    if hasattr(v, "name"):
        return v.name
    s = str(v)
    return s if len(s) <= 30 else s[:27] + "..."


def _data_summary(data: dict) -> dict:
    """Summarize data dict for metadata."""
    pairs = data.get("pairs", [])
    return {
        "n_pairs": len(pairs),
        "n_index_pairs": len(data.get("index_pairs", [])),
        "n_query_pairs": len(data.get("query_pairs", [])),
        "split_info": data.get("split_info", {}),
        "cwe_ids": list({p.cwe_id for p in pairs if hasattr(p, "cwe_id") and p.cwe_id}),
        "projects": list({p.project for p in pairs if hasattr(p, "project")}),
    }
