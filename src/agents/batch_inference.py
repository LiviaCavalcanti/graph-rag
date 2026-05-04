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

import re
import time
from pathlib import Path

from src.agents.patcher import patch_one
from src.agents.utils import MODEL_NAME, strip_code_fences
from src.metrics.similarity import code_similarity

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
    if code:
        # For original variants, source_before is a file path — prefer db_entry
        if query_pair.meta.get("variant") == "original" and Path(code).exists():
            return strip_code_fences(target_db.get("original_code", ""))
        # inline code (augmented variant)
        return strip_code_fences(code)
    # no source_before — fall back to db_entry
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
        return {
            **base,
            "status": "skipped",
            "reason": "no_example_found",
            "retrieval": retrieval_info,
        }

    # load db_entries by dir_name
    example_dir = example_pair.meta.get("dir_name", "")
    target_dir = query_pair.meta.get("dir_name", "")
    example_db = db_cache.get(example_dir)
    target_db = db_cache.get(target_dir)

    # fallback: when dir_name is empty (e.g. older precomputed results),
    # scan db_cache for a matching cve_id
    if not example_db and example_dir == "":
        ex_cve = getattr(example_pair, "cve_id", None) or example_pair.meta.get(
            "cve_id", ""
        )
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
        return {
            **base,
            "status": "skipped",
            "reason": "missing_db_entry",
            "retrieval": retrieval_info,
        }

    # get target code from meta or db_entry
    target_code = _get_target_code(query_pair, target_db)
    target_supplementary = query_pair.meta.get(
        "supplementary_code", ""
    ) or target_db.get("supplementary_code", "")
    ground_truth = _get_ground_truth(query_pair, target_db)

    if not target_code:
        return {
            **base,
            "status": "skipped",
            "reason": "no_target_code",
            "retrieval": retrieval_info,
        }

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
        return {
            **base,
            "status": "error",
            "error": err_str,
            "retrieval": retrieval_info,
            "elapsed_s": round(elapsed, 2),
        }

    cve_match = retrieval_info.get("cve_match", False)
    cwe_match = retrieval_info.get("cwe_match", False)

    similarity = 0.0
    is_exact = False
    rouge = {}
    if parsed and parsed.get("vuln_patch"):
        similarity = code_similarity(parsed["vuln_patch"], ground_truth)
        gen_stripped = re.sub(r"\s+", " ", parsed["vuln_patch"]).strip()
        ref_stripped = re.sub(r"\s+", " ", ground_truth).strip()
        is_exact = gen_stripped == ref_stripped
        status = "success"
        from src.metrics.similarity import rouge_scores
        rouge = rouge_scores(parsed["vuln_patch"], ground_truth)
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
        "rouge": rouge,
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
    output_dir: Path | None = None,
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
    from src.io.batch import run_batched

    resolved_model = model_name or MODEL_NAME

    # ── per-query callback ───────────────────────────────────────────
    def process_one(query_pair, current: int, total: int) -> dict:
        label = (
            f"  [{current}/{total}] "
            f"{query_pair.cve_id} ({query_pair.meta.get('variant', '?')})"
        )
        result = _run_single_query(
            query_pair,
            retriever,
            db_cache,
            resolved_model,
        )
        status = result["status"]
        sim = result.get("similarity", "")
        elapsed = result.get("elapsed_s", "")
        print(
            f"{label} → {status}"
            + (f"  sim={sim}" if sim != "" else "")
            + (f"  {elapsed}s" if elapsed != "" else "")
        )
        return result

    # ── run batched ──────────────────────────────────────────────────
    return run_batched(
        query_pairs,
        process_one=process_one,
        run_tag=run_tag,
        batch_size=batch_size,
        resume=resume_dir,
        output_dir=output_dir,
        results_filename=RESULTS_FILENAME,
        meta_filename=META_FILENAME,
        meta={
            "model": resolved_model,
            "batch_size": batch_size,
            **(meta_extra or {}),
        },
        abort_on=(ForbiddenError,),
    )
