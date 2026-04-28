#!/usr/bin/env python3
"""
Batch inference engine with resumability, crash-safe writes, and 403 abort.

Processes LLM patching queries in batches, writing results to an append-only
JSONL file after each batch.  On restart, already-completed queries are skipped.

This module is a pure execution engine — data loading, split logic, and
retriever construction are handled by the caller (e.g. main.py --mode batch).

Post-process results:
    python -m experiments.postprocess experiments/output/<run_dir>
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from src.agents.patcher import patch_one
from src.agents.utils import (
    MODEL_NAME,
    code_similarity,
    strip_code_fences,
)
from src.io import BackgroundWriter, load_completed
from experiments.common import make_run_dir

RESULTS_FILENAME = "results.jsonl"
META_FILENAME = "run_meta.json"

class ForbiddenError(Exception):
    """Raised when the API returns HTTP 403 — signals immediate abort."""


# ── code extraction helpers ─────────────────────────────────────────

def _get_target_code(query_pair, target_db: dict) -> str:
    """Get the vulnerable source code for the target.

    For augmented variants the code is stored inline in meta['source_before'].
    For original variants it's a file path; fall back to db_entry['original_code'].
    """
    code = query_pair.meta.get("source_before", "")
    if code :
        # inline code (augmented variant)
        return strip_code_fences(code)
    # original variant — prefer db_entry which has the actual code string
    return strip_code_fences(target_db.get("original_code", ""))


def _get_ground_truth(query_pair, target_db: dict) -> str:
    """Get the patched code (ground truth) for evaluation.

    For augmented variants 'source_after' is a file path to the fixed .c file.
    For original variants it may also be a file path, or db_entry has 'vuln_patch'.
    """
    source_after = query_pair.meta.get("source_after", "")
    if source_after:
        p = Path(source_after)
        if p.exists():
            try:
                return strip_code_fences(p.read_text(errors="replace"))
            except OSError:
                pass
        elif len(source_after) > 50:
            # inline code
            return strip_code_fences(source_after)
    return strip_code_fences(target_db.get("vuln_patch", ""))


# ── single-query execution ──────────────────────────────────────────

def _run_single_query(
    query_pair,
    retriever,
    db_cache: dict,
    model_name: str,
) -> dict:
    """
    Execute a single LLM patching query.

    db_cache is keyed by dir_name (e.g. 'CVE-2025-21809_1').
    Returns a result dict.  Raises ForbiddenError on HTTP 403.
    """
    cve_id = query_pair.cve_id
    cwe_id = query_pair.cwe_id
    variant = query_pair.meta.get("variant", "")

    base = {
        "query_cve": cve_id,
        "query_cwe": cwe_id,
        "query_variant": variant,
    }

    # retrieve example
    example_pair, retrieval_info = retriever.retrieve(query_pair)

    if example_pair is None:
        return {**base, "status": "skipped", "reason": "no_example_found",
                "retrieval": retrieval_info}

    # load db_entries by dir_name
    example_dir = example_pair.meta.get("dir_name", "")
    target_dir = query_pair.meta.get("dir_name", "")
    example_db = db_cache.get(example_dir)
    target_db = db_cache.get(target_dir)

    # fallback: when dir_name is empty (e.g. older precomputed results),
    # scan db_cache for a matching cve_id
    if not example_db and example_dir == "":
        ex_cve = getattr(example_pair, "cve_id", None) or example_pair.meta.get("cve_id", "")
        for dname, db in db_cache.items():
            if db.get("cve_id") == ex_cve or dname.startswith(ex_cve):
                example_db = db
                break
    if not target_db and target_dir == "":
        for dname, db in db_cache.items():
            if db.get("cve_id") == cve_id or dname.startswith(cve_id):
                target_db = db
                break

    if not example_db or not target_db:
        return {**base, "status": "skipped", "reason": "missing_db_entry",
                "retrieval": retrieval_info}

    # get target code from meta or db_entry
    target_code = _get_target_code(query_pair, target_db)
    target_supplementary = (
        query_pair.meta.get("supplementary_code", "")
        or target_db.get("supplementary_code", "")
    )
    ground_truth = _get_ground_truth(query_pair, target_db)

    if not target_code:
        return {**base, "status": "skipped", "reason": "no_target_code",
                "retrieval": retrieval_info}

    # invoke patcher (prompt build → LLM → parse)
    t0 = time.perf_counter()
    try:
        raw_output, parsed = patch_one(
            example_db=example_db,
            target_db=target_db,
            target_code=target_code,
            target_supplementary=target_supplementary,
            model_name=model_name,
        )
        elapsed = time.perf_counter() - t0
    except Exception as e:
        elapsed = time.perf_counter() - t0
        err_str = str(e)
        # detect 403 — abort the whole run
        if "403" in err_str or "Forbidden" in err_str:
            raise ForbiddenError(f"HTTP 403 from API: {err_str}") from e
        return {**base, "status": "error", "error": err_str,
                "retrieval": retrieval_info, "elapsed_s": round(elapsed, 2)}

    cve_match = retrieval_info.get("cve_match", False)
    cwe_match = retrieval_info.get("cwe_match", False)

    similarity = 0.0
    is_exact = False
    if parsed and parsed.get("vuln_patch"):
        similarity = code_similarity(parsed["vuln_patch"], ground_truth)
        gen_stripped = re.sub(r"\s+", " ", parsed["vuln_patch"]).strip()
        ref_stripped = re.sub(r"\s+", " ", ground_truth).strip()
        is_exact = gen_stripped == ref_stripped
        status = "success"
    else:
        status = "parse_error"

    return {
        **base,
        "example_cve": example_pair.cve_id,
        "example_cwe": example_pair.cwe_id,
        "example_variant": example_pair.meta.get("variant", ""),
        "status": status,
        "cve_match": cve_match,
        "cwe_match": cwe_match,
        "similarity": round(similarity, 4),
        "exact_match": is_exact,
        "elapsed_s": round(elapsed, 2),
        "retrieval": retrieval_info,
        "raw_output_len": len(raw_output),
        "generated_patch": parsed["vuln_patch"] if parsed else None,
        "ground_truth_patch": ground_truth,
    }


# ── batch runner ─────────────────────────────────────────────────────

def run_batch_inference(
    query_pairs: list,
    retriever,
    db_cache: dict,
    model_name: str | None = None,
    batch_size: int = 10,
    run_tag: str = "batch",
    resume_dir: str | None = None,
    meta_extra: dict | None = None,
) -> Path:
    """
    Run LLM patching in batches, writing results to JSONL incrementally.

    Args:
        query_pairs:  list of FunctionPair to query
        retriever:    object with .retrieve(pair) → (example_pair, info)
        db_cache:     dict of cve_id → db_entry dict
        model_name:   Azure model/deployment name
        batch_size:   queries per disk flush
        run_tag:      tag for the output directory name
        resume_dir:   path to a previous run dir to resume
        meta_extra:   extra metadata to save in run_meta.json

    Returns the path to the run directory.
    """
    resolved_model = model_name or MODEL_NAME

    # ── resolve run directory ────────────────────────────────────────
    if resume_dir:
        run_dir = Path(resume_dir)
        if not run_dir.exists():
            print(f"ERROR: resume directory does not exist: {run_dir}")
            sys.exit(1)
        run_id = run_dir.name
        print(f"Resuming run: {run_id}")
    else:
        run_id, run_dir = make_run_dir(run_tag)
        print(f"New run: {run_id}")

    jsonl_path = run_dir / RESULTS_FILENAME
    meta_path = run_dir / META_FILENAME

    # ── load completed queries ───────────────────────────────────────
    completed = load_completed(jsonl_path)
    if completed:
        print(f"Found {len(completed)} completed queries — will skip them")

    # ── filter out already-done queries ──────────────────────────────
    pending = [
        p for p in query_pairs
        if (p.cve_id, p.meta.get("variant", "")) not in completed
    ]
    total = len(query_pairs)
    print(f"Total queries: {total} | Already done: {total - len(pending)} | Pending: {len(pending)}")

    if not pending:
        print("All queries already completed.  Run postprocess to get summary.")
        return run_dir

    # ── save run metadata ────────────────────────────────────────────
    meta = {
        "run_id": run_id,
        "model": resolved_model,
        "batch_size": batch_size,
        "total_queries": total,
        "resumed": resume_dir is not None,
        **(meta_extra or {}),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # ── process in batches ───────────────────────────────────────────
    writer = BackgroundWriter(jsonl_path)
    n_done = len(completed)
    n_success = 0
    n_errors = 0
    t_start = time.perf_counter()

    try:
        for batch_idx in range(0, len(pending), batch_size):
            batch = pending[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(pending) + batch_size - 1) // batch_size

            print(f"\n{'─'*60}")
            print(f"Batch {batch_num}/{total_batches}  "
                  f"(queries {n_done+1}–{n_done+len(batch)} of {total})")
            print(f"{'─'*60}")

            batch_results = []

            for i, query_pair in enumerate(batch):
                label = (f"  [{n_done+i+1}/{total}] "
                         f"{query_pair.cve_id} ({query_pair.meta.get('variant', '?')})")
                try:
                    result = _run_single_query(
                        query_pair, retriever, db_cache, resolved_model,
                    )
                    status = result["status"]
                    sim = result.get("similarity", "")
                    elapsed = result.get("elapsed_s", "")
                    print(f"{label} → {status}"
                          + (f"  sim={sim}" if sim != "" else "")
                          + (f"  {elapsed}s" if elapsed != "" else ""))

                    if status == "success":
                        n_success += 1
                    elif status == "error":
                        n_errors += 1

                    batch_results.append(result)

                except ForbiddenError as e:
                    print(f"\n{'!'*60}")
                    print(f"  ABORT: {e}")
                    print(f"  Writing {len(batch_results)} results from current batch before exit.")
                    print(f"{'!'*60}")
                    if batch_results:
                        writer.write(batch_results)
                        writer.flush()
                    writer.close()
                    _print_progress(n_done + len(batch_results), total,
                                    n_success, n_errors, t_start)
                    print(f"\nRun directory: {run_dir}")
                    print("Re-run with --resume to continue after fixing credentials.")
                    sys.exit(2)

            # flush batch to disk via background writer
            n_done += len(batch_results)
            writer.write(batch_results)

            _print_progress(n_done, total, n_success, n_errors, t_start)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted — flushing writes...")
        writer.flush()
        writer.close()
        _print_progress(n_done, total, n_success, n_errors, t_start)
        print(f"Run directory: {run_dir}")
        print("Re-run with --resume to continue.")
        sys.exit(1)

    writer.flush()
    writer.close()

    _print_progress(n_done, total, n_success, n_errors, t_start)
    print(f"\nRun directory: {run_dir}")
    print(f"Run postprocess:  python -m experiments.postprocess {run_dir}")

    return run_dir


def _print_progress(done: int, total: int, success: int, errors: int, t_start: float):
    elapsed = time.perf_counter() - t_start
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    print(f"\n  Progress: {done}/{total} ({done/total:.0%})"
          f"  |  OK: {success}  Errors: {errors}"
          f"  |  {elapsed:.0f}s elapsed"
          + (f"  ~{remaining:.0f}s remaining" if done < total else ""))


