"""Generic batched execution with resumability, crash-safe writes, and progress tracking."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from src.io.read_write import BackgroundWriter, load_completed, make_run_dir


def run_batched(
    items: list,
    *,
    process_one: Callable[[Any, int, int], dict],
    run_tag: str = "batch",
    batch_size: int = 10,
    resume: str | None = None,
    output_dir: Path | None = None,
    results_filename: str = "results.jsonl",
    meta_filename: str = "run_meta.json",
    meta: dict | None = None,
    item_key: Callable[[Any], tuple[str, str]] | None = None,
    abort_on: tuple[type, ...] = (),
) -> Path:
    """Run a batch job with resumability and crash-safe JSONL writes.

    Parameters
    ----------
    items : list
        All items to process (before filtering out completed ones).
    process_one : callable
        ``(item, current_index, total) -> dict``.  Called for each pending item.
        Should return a result dict to be written to JSONL.
        May raise an exception in *abort_on* to trigger graceful abort.
    run_tag : str
        Tag for the run directory name (used only when creating a new run).
    batch_size : int
        Items per disk flush.
    resume : str | None
        Path to a previous run directory to resume.
    output_dir : Path | None
        If provided, use this directory (don't create a new one).
    results_filename : str
        Name of the JSONL output file.
    meta_filename : str
        Name of the metadata JSON file.
    meta : dict | None
        Extra metadata to save alongside run info.
    item_key : callable | None
        Extracts a ``(cve_id, variant)`` key from an item for deduplication.
        Defaults to ``(item.cve_id, item.meta.get("variant", ""))``.
    abort_on : tuple[type, ...]
        Exception types that trigger graceful abort (flush current batch, exit).

    Returns
    -------
    Path
        The run directory.
    """
    # ── default key extractor ────────────────────────────────────────
    if item_key is None:
        item_key = lambda p: (p.cve_id, p.meta.get("variant", ""))

    # ── resolve run directory ────────────────────────────────────────
    if resume:
        run_dir = Path(resume)
        if not run_dir.exists():
            print(f"ERROR: resume directory does not exist: {run_dir}")
            sys.exit(1)
        run_id = run_dir.name
        print(f"Resuming run: {run_id}")
    elif output_dir:
        run_dir = output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name
        print(f"Using shared run directory: {run_id}")
    else:
        run_id, run_dir = make_run_dir(run_tag)
        print(f"New run: {run_id}")

    jsonl_path = run_dir / results_filename
    meta_path = run_dir / meta_filename

    # ── load completed items ─────────────────────────────────────────
    completed = load_completed(jsonl_path)
    if completed:
        print(f"Found {len(completed)} completed queries — will skip them")

    pending = [p for p in items if item_key(p) not in completed]
    total = len(items)
    print(f"Total: {total} | Done: {total - len(pending)} | Pending: {len(pending)}")

    if not pending:
        print("All queries already completed.")
        return run_dir

    # ── save run metadata ────────────────────────────────────────────
    run_meta = {
        "run_id": run_id,
        "total_queries": total,
        "resumed": resume is not None,
        **(meta or {}),
    }
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    # ── process in batches ───────────────────────────────────────────
    writer = BackgroundWriter(jsonl_path)
    n_done = len(completed)
    t_start = time.perf_counter()

    try:
        for batch_idx in range(0, len(pending), batch_size):
            batch = pending[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(pending) + batch_size - 1) // batch_size

            print(f"\n{'─'*60}")
            print(
                f"Batch {batch_num}/{total_batches}  "
                f"(queries {n_done+1}–{n_done+len(batch)} of {total})"
            )
            print(f"{'─'*60}")

            batch_results = []
            for i, item in enumerate(batch):
                try:
                    result = process_one(item, n_done + i + 1, total)
                    batch_results.append(result)
                except tuple(abort_on) if abort_on else () as e:
                    print(f"\n{'!'*60}")
                    print(f"  ABORT: {e}")
                    print(f"  Writing {len(batch_results)} results from current batch before exit.")
                    print(f"{'!'*60}")
                    if batch_results:
                        writer.write(batch_results)
                        writer.flush()
                    writer.close()
                    print(f"\nRun directory: {run_dir}")
                    print("Re-run with --resume to continue.")
                    sys.exit(2)

            n_done += len(batch_results)
            writer.write(batch_results)
            elapsed = time.perf_counter() - t_start
            print(
                f"\n  Progress: {n_done}/{total} ({n_done/total:.0%})  |  {elapsed:.0f}s elapsed"
            )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted — flushing writes...")
        writer.flush()
        writer.close()
        print(f"Run directory: {run_dir}")
        print("Re-run with --resume to continue.")
        sys.exit(1)

    writer.flush()
    writer.close()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. {n_done} queries written in {elapsed:.0f}s.")
    print(f"Run directory: {run_dir}")
    return run_dir
