"""LLM-as-a-Judge for patch evaluation.

Implements the judge design from:
  "Towards a Human-in-the-Loop Framework for Reliable Patch Evaluation
   Using an LLM-as-a-Judge"

Given:
  - Bug context (CVE/CWE description, function name)
  - The patch in unified diff format
  - A golden rubric derived from the CWE category

The judge outputs:
  - <thought>   step-by-step reasoning
  - <verdict>   VALID | INVALID
  - <justification>  concise summary

Output file: <run_dir>/llm_judge_results.jsonl
  {"query_cve": ..., "query_variant": ..., "verdict": "VALID"|"INVALID"|"ERROR",
   "thought": ..., "justification": ..., "cwe_rubric": ..., "model": ..., "elapsed_s": ...}

Usage:
    uv run python -m src.evaluate.llm_judge <run_dir>/results.jsonl [--model gpt-4o]
    uv run python -m src.evaluate.llm_judge <run_dir>/            [--model gpt-4o]
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

# ── CWE Rubric Library ────────────────────────────────────────────────────────

# Maps a CWE number prefix (e.g. "125" from "CWE-125") to a specific rubric
# describing what a VALID fix must accomplish.
_CWE_RUBRICS: dict[str, str] = {
    "119": (
        "The patch must validate buffer size before any copy or write operation. "
        "It must reject or truncate input that would exceed the allocated buffer."
    ),
    "120": (
        "The patch must replace or guard the unsafe memory-copy function with a "
        "size-bounded variant (e.g. strncpy, strlcpy, memcpy with explicit length). "
        "The destination size must be verified before copying."
    ),
    "121": (
        "The patch must prevent a stack-based buffer overflow by adding a bounds check "
        "on the length of data written to a stack-allocated array."
    ),
    "122": (
        "The patch must prevent a heap-based buffer overflow by validating that the "
        "write length does not exceed the heap-allocated buffer size."
    ),
    "125": (
        "The patch must add a bounds check before the out-of-bounds read. "
        "The array index or pointer offset must be validated against the array/buffer "
        "length before any read access."
    ),
    "190": (
        "The patch must check for integer overflow or wrap-around before performing "
        "arithmetic. It must reject or safely handle values that would overflow the "
        "integer type."
    ),
    "191": (
        "The patch must check for integer underflow (wrap-around of unsigned subtraction "
        "or signed result going below minimum) before the operation. Invalid values must "
        "be rejected or clamped."
    ),
    "193": (
        "The patch must correct an off-by-one error; either the loop bound, array index, "
        "or length calculation must be adjusted so that no read or write occurs one element "
        "past the valid range."
    ),
    "20": (
        "The patch must add input validation that rejects or sanitises malformed, "
        "unexpected, or attacker-controlled input before it is used. All relevant code "
        "paths that accept external input must be covered."
    ),
    "362": (
        "The patch must eliminate the race condition by introducing appropriate "
        "synchronisation (mutex, spinlock, atomic operation, or similar). The shared "
        "resource must be protected across all concurrent access paths."
    ),
    "369": (
        "The patch must add a check that the divisor is non-zero before the division "
        "or modulo operation."
    ),
    "400": (
        "The patch must add a limit or cap on the quantity of resources (memory, file "
        "descriptors, loop iterations, etc.) that can be allocated or consumed, or it "
        "must add an error path when allocation exceeds the limit."
    ),
    "401": (
        "The patch must ensure that all memory allocations are released on every "
        "exit path, including error paths. Every code path that exits the function "
        "must free the allocated memory."
    ),
    "415": (
        "The patch must prevent double-free by nulling the pointer after the first free, "
        "adding a flag to track allocation state, or restructuring ownership to ensure "
        "the free is called exactly once."
    ),
    "416": (
        "The patch must prevent use-after-free by either (a) nulling the pointer "
        "immediately after free and checking before use, (b) removing the use that "
        "follows the free, or (c) restructuring lifetime so the object is guaranteed "
        "alive at the point of use."
    ),
    "476": (
        "The patch must add a NULL pointer check before the dereference. Every "
        "code path that could encounter a NULL pointer must be guarded with an "
        "explicit check and a safe error-handling branch."
    ),
    "787": (
        "The patch must validate that the write destination index or pointer is within "
        "the allocated buffer bounds before the write. Off-by-one writes must also be "
        "prevented."
    ),
    "824": (
        "The patch must ensure the pointer is initialised to a valid value or NULL "
        "before use. Any code path that could reach the dereference without prior "
        "initialisation must be fixed."
    ),
}

_CWE_FALLBACK_RUBRIC = (
    "The patch must address the root cause of the reported vulnerability as described "
    "by the CVE. It must introduce the necessary guard, check, initialisation, or "
    "structural change that prevents the specific exploitation vector. Cosmetic changes, "
    "comments, or unrelated modifications do NOT constitute a valid fix. The patch must "
    "not introduce new security vulnerabilities."
)


def get_cwe_rubric(cwe_id: str) -> str:
    """Return the golden rubric for a CWE, falling back to a generic one."""
    if not cwe_id:
        return _CWE_FALLBACK_RUBRIC
    # Accept "CWE-125", "CWE125", "125"
    match = re.search(r"(\d+)", str(cwe_id))
    if match:
        rubric = _CWE_RUBRICS.get(match.group(1))
        if rubric:
            return rubric
    return _CWE_FALLBACK_RUBRIC


# ── Prompt Templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert security code reviewer acting as an impartial judge.

Your task is to evaluate whether a generated patch correctly fixes a reported
software vulnerability (CVE). You will receive:
  1. Bug context — the CVE identifier, CWE type, and the vulnerable function.
  2. The patch in unified diff format (showing what was changed).
  3. A golden rubric — the specific criteria that a valid fix MUST satisfy.

You MUST respond using exactly this XML structure and nothing else:

<thought>
[Your step-by-step analysis: check whether each rubric criterion is met,
identify what the patch changes, and decide whether it addresses the root cause.]
</thought>
<verdict>VALID</verdict>
<justification>
[One or two sentences summarising why the patch is VALID or INVALID.]
</justification>

Rules:
- VALID means the patch meets every criterion in the golden rubric.
- INVALID means the patch fails to meet at least one criterion, is a cosmetic
  change only, or introduces a new vulnerability.
- Do not be lenient: partial fixes are INVALID.
- The diff shows changes from vulnerable to generated code. Lines starting with
  '+' are additions, lines starting with '-' are removals.
""")

_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
## Bug Context

- **CVE ID**: {cve_id}
- **CWE Type**: {cwe_id}
- **Vulnerable Function**: `{function_name}`

## Patch (Unified Diff)

This diff shows the changes the agent made to the vulnerable code.
Lines starting with '+' were added; lines starting with '-' were removed.

```diff
{unified_diff}
```

## Golden Rubric

For this patch to be VALID, it MUST satisfy the following criterion:

> {cwe_rubric}

## Your Judgment

Analyse the patch above against the golden rubric and respond using the required
XML structure (<thought>, <verdict>, <justification>).
""")


# ── Diff Formatting ───────────────────────────────────────────────────────────

def format_unified_diff(vulnerable_code: str, generated_patch: str, context_lines: int = 5) -> str:
    """Produce a unified diff between vulnerable and generated code."""
    a_lines = (vulnerable_code or "").splitlines(keepends=True)
    b_lines = (generated_patch or "").splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        a_lines, b_lines,
        fromfile="vulnerable.c",
        tofile="generated_patch.c",
        n=context_lines,
    ))
    if not diff:
        return "(no differences — patch is identical to vulnerable code)"
    return "".join(diff)


# ── LLM Call ─────────────────────────────────────────────────────────────────

def _call_llm(user_prompt: str, model: str) -> str:
    response = litellm.completion(
        model=f"azure/{model}",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASEURL"),
        api_version="2024-12-01-preview",
        temperature=0.0,
        max_tokens=1500,
    )
    return response.choices[0].message.content or ""


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_judge_response(raw: str) -> dict:
    """Extract thought, verdict, justification from the XML-structured response."""
    def _extract(tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    thought = _extract("thought")
    verdict_text = _extract("verdict").upper().strip()
    justification = _extract("justification")

    # Normalise verdict to VALID / INVALID
    if verdict_text in ("VALID", "INVALID"):
        verdict = verdict_text
    elif "VALID" in verdict_text:
        verdict = "VALID"
    elif "INVALID" in verdict_text:
        verdict = "INVALID"
    else:
        # Fallback: scan full response
        if re.search(r"\bVALID\b", raw, re.IGNORECASE):
            verdict = "VALID"
        elif re.search(r"\bINVALID\b", raw, re.IGNORECASE):
            verdict = "INVALID"
        else:
            verdict = "ERROR"

    return {
        "verdict": verdict,
        "thought": thought,
        "justification": justification or raw[:300],
    }


# ── Per-record Judge ──────────────────────────────────────────────────────────

def judge_one(record: dict, model: str) -> dict:
    """Run the LLM judge on a single results.jsonl record."""
    cve_id = record.get("query_cve", "")
    cwe_id = record.get("query_cwe", "")
    variant = record.get("query_variant", "")

    vulnerable_code = record.get("target_code") or ""
    generated_patch = record.get("generated_patch") or ""

    # Skip records without a generated patch
    if not generated_patch.strip():
        return {
            "query_cve": cve_id,
            "query_variant": variant,
            "query_cwe": cwe_id,
            "verdict": "ERROR",
            "thought": "",
            "justification": "No generated patch available.",
            "cwe_rubric": get_cwe_rubric(cwe_id),
            "model": model,
            "elapsed_s": 0.0,
        }

    # Derive function name from record (best-effort)
    function_name = record.get("query_function_name") or record.get("function_name") or "unknown"

    cwe_rubric = get_cwe_rubric(cwe_id)
    unified_diff = format_unified_diff(vulnerable_code, generated_patch)

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        cve_id=cve_id or "unknown",
        cwe_id=cwe_id or "unknown",
        function_name=function_name,
        unified_diff=unified_diff,
        cwe_rubric=cwe_rubric,
    )

    t0 = time.time()
    try:
        raw = _call_llm(user_prompt, model)
        parsed = parse_judge_response(raw)
    except Exception as exc:
        parsed = {
            "verdict": "ERROR",
            "thought": "",
            "justification": f"LLM call failed: {exc}",
        }
    elapsed = round(time.time() - t0, 2)

    return {
        "query_cve": cve_id,
        "query_variant": variant,
        "query_cwe": cwe_id,
        "verdict": parsed["verdict"],
        "thought": parsed["thought"],
        "justification": parsed["justification"],
        "cwe_rubric": cwe_rubric,
        "model": model,
        "elapsed_s": elapsed,
    }


# ── Batch Runner ──────────────────────────────────────────────────────────────

def run_judge(
    results_path: Path,
    model: str | None = None,
    resume: bool = True,
) -> Path:
    """Run the LLM judge over all records in results.jsonl.

    Writes <run_dir>/llm_judge_results.jsonl and returns its path.
    If resume=True, skips records already present in the output file.
    """
    model = model or os.getenv("MODEL_NAME", "gpt-4o")
    run_dir = results_path.parent
    out_path = run_dir / "llm_judge_results.jsonl"

    # Load records
    records: list[dict] = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Build resume index
    done: set[tuple[str, str]] = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done.add((entry.get("query_cve", ""), entry.get("query_variant", "")))
        print(f"  Resuming: {len(done)} already judged, {len(records) - len(done)} remaining")

    with open(out_path, "a") as out_f:
        for i, rec in enumerate(records):
            key = (rec.get("query_cve", ""), rec.get("query_variant", ""))
            label = f"{key[0]}/{key[1]}"

            if key in done:
                continue

            print(f"  [{i+1}/{len(records)}] Judging {label} ...", end=" ", flush=True)
            result = judge_one(rec, model)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            done.add(key)
            print(f"{result['verdict']} ({result['elapsed_s']}s)")

    print(f"\n  LLM judge results → {out_path}")
    return out_path


def aggregate_judge_results(out_path: Path) -> dict:
    """Compute summary statistics from llm_judge_results.jsonl."""
    entries: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    total = len(entries)
    counts: dict[str, int] = {}
    for e in entries:
        v = e.get("verdict", "ERROR")
        counts[v] = counts.get(v, 0) + 1

    valid = counts.get("VALID", 0)
    invalid = counts.get("INVALID", 0)
    errors = counts.get("ERROR", 0)
    valid_rate = valid / total * 100 if total else 0.0

    # By CWE
    by_cwe: dict[str, dict] = {}
    for e in entries:
        cwe = e.get("query_cwe") or "unknown"
        if cwe not in by_cwe:
            by_cwe[cwe] = {"total": 0, "valid": 0, "invalid": 0, "error": 0}
        by_cwe[cwe]["total"] += 1
        v = e.get("verdict", "ERROR")
        if v == "VALID":
            by_cwe[cwe]["valid"] += 1
        elif v == "INVALID":
            by_cwe[cwe]["invalid"] += 1
        else:
            by_cwe[cwe]["error"] += 1

    for stats in by_cwe.values():
        t = stats["total"]
        stats["valid_rate"] = round(stats["valid"] / t * 100, 1) if t else 0.0

    return {
        "total": total,
        "counts": counts,
        "valid": valid,
        "invalid": invalid,
        "errors": errors,
        "valid_rate": round(valid_rate, 1),
        "by_cwe": by_cwe,
        "entries": entries,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LLM-as-a-Judge over a batch results.jsonl."
    )
    parser.add_argument(
        "results",
        help="Path to results.jsonl (or run directory — will find results.jsonl automatically)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Azure model deployment name (default: MODEL_NAME env var or gpt-4o)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-judge all records even if llm_judge_results.jsonl already exists",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if results_path.is_dir():
        candidate = results_path / "results.jsonl"
        if not candidate.exists():
            # Try results.json (some runs use this)
            candidate = results_path / "results.json"
        results_path = candidate

    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Results:  {results_path}")
    print(f"Model:    {args.model or os.getenv('MODEL_NAME', 'gpt-4o')}")

    out_path = run_judge(results_path, model=args.model, resume=not args.no_resume)
    agg = aggregate_judge_results(out_path)

    print(f"\n{'═'*50}")
    print(f"  LLM JUDGE SUMMARY  ({agg['total']} records)")
    print(f"{'═'*50}")
    print(f"  VALID:    {agg['valid']}  ({agg['valid_rate']:.1f}%)")
    print(f"  INVALID:  {agg['invalid']}")
    print(f"  ERROR:    {agg['errors']}")
    print(f"{'═'*50}")
    if agg["by_cwe"]:
        print("\n  By CWE:")
        for cwe, stats in sorted(agg["by_cwe"].items()):
            print(
                f"    {cwe:12s}  total={stats['total']:3d}  "
                f"valid={stats['valid']}  invalid={stats['invalid']}  "
                f"valid_rate={stats['valid_rate']}%"
            )


if __name__ == "__main__":
    main()
