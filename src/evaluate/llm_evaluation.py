"""LLM-based vulnerability patch verification.

Asks an LLM to judge whether a generated patch actually fixes the CVE
by comparing the vulnerable code against the patched code.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o")
CVE_LIST_DIR = Path("CVE-list")

# ── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are a world-class software security auditor specializing in vulnerability analysis.

Your task: Determine whether a code patch correctly fixes a specific CVE vulnerability.

You must be rigorous. A patch "fixes" the vulnerability ONLY if it introduces the
necessary guard, check, initialization, or structural change that prevents the
specific exploitation vector described by the CVE. Cosmetic changes, comments,
or unrelated modifications do NOT count as a fix.

Respond with a structured JSON verdict. Do not include any other text.
""")

USER_PROMPT_TEMPLATE = textwrap.dedent("""\
## Vulnerability Information

- **CVE ID**: {cve_id}
- **CWE Type**: {cwe_type}
- **Vulnerable Function**: `{function_name}`
- **Function Prototype**: `{function_prototype}`

## Vulnerable Code (BEFORE patch)

This is the original code that contains the vulnerability:

```{language}
{vulnerable_code}
```

## Generated Patch (AFTER patch)

This is the candidate patch that claims to fix the vulnerability:

```{language}
{patched_code}
```

## Your Task

Analyze whether the **Generated Patch** correctly fixes **{cve_id}** ({cwe_type}).

Consider:
1. Does the patch address the root cause of the vulnerability?
2. Does it introduce the necessary safety check, guard condition, or fix?
3. Is the fix complete (not partial)?
4. Does the patch introduce any NEW vulnerabilities?

Respond with ONLY the following JSON (no markdown fences, no extra text):

{{
  "verdict": "FIXED" or "NOT_FIXED" or "PARTIAL",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one paragraph explaining your judgment>",
  "fix_description": "<what the patch does to fix the vulnerability, or what it fails to do>",
  "issues": ["<list of any problems with the patch, empty if none>"]
}}
""")


# ── LLM Call ─────────────────────────────────────────────────────────────────

def call_llm(system: str, user: str, model: str = DEFAULT_MODEL) -> str:
    """Make a single LLM call and return the response text."""
    response = litellm.completion(
        model=f"azure/{model}",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASEURL"),
        api_version="2024-12-01-preview",
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def parse_llm_response(raw: str) -> dict:
    """Parse the LLM's JSON response, handling common formatting issues."""
    text = raw.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from the response
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "reasoning": f"Failed to parse LLM response: {raw[:200]}",
            "fix_description": "",
            "issues": ["LLM response was not valid JSON"],
        }


# ── Core Evaluation Logic ────────────────────────────────────────────────────

def load_vulnerable_code(cve_id: str) -> str:
    """Load the original vulnerable code for a CVE."""
    # Try direct match
    code_path = CVE_LIST_DIR / cve_id / "original_code.txt"
    if code_path.exists():
        return code_path.read_text(encoding="utf-8", errors="replace").strip()

    # Try with _1 suffix (for multi-file CVEs)
    code_path = CVE_LIST_DIR / f"{cve_id}_1" / "original_code.txt"
    if code_path.exists():
        return code_path.read_text(encoding="utf-8", errors="replace").strip()

    # Search for any matching directory
    for d in CVE_LIST_DIR.iterdir():
        if d.is_dir() and d.name.startswith(cve_id):
            candidate = d / "original_code.txt"
            if candidate.exists():
                return candidate.read_text(encoding="utf-8", errors="replace").strip()

    return ""


def load_cve_info(cve_id: str) -> dict:
    """Load info.json for a CVE."""
    info_path = CVE_LIST_DIR / cve_id / "info.json"
    if not info_path.exists():
        info_path = CVE_LIST_DIR / f"{cve_id}_1" / "info.json"
    if not info_path.exists():
        for d in CVE_LIST_DIR.iterdir():
            if d.is_dir() and d.name.startswith(cve_id):
                candidate = d / "info.json"
                if candidate.exists():
                    info_path = candidate
                    break

    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    return {}


def evaluate_single(
    cve_id: str,
    cwe_type: str,
    generated_patch: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Evaluate a single generated patch against its CVE."""
    info = load_cve_info(cve_id)
    vulnerable_code = load_vulnerable_code(cve_id)

    if not vulnerable_code:
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "reasoning": f"Could not load vulnerable code for {cve_id}",
            "fix_description": "",
            "issues": ["Missing vulnerable source code"],
        }

    if not generated_patch.strip():
        return {
            "verdict": "NOT_FIXED",
            "confidence": 1.0,
            "reasoning": "No patch was generated.",
            "fix_description": "",
            "issues": ["Empty patch"],
        }

    # Strip code fences from vulnerable code
    vuln_code_clean = re.sub(r"^```\w*\n?", "", vulnerable_code)
    vuln_code_clean = re.sub(r"\n?```\s*$", "", vuln_code_clean).strip()

    language = info.get("programming_language", "c")
    function_name = info.get("function_name", "unknown")
    function_proto = info.get("function_prototype", function_name)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        cve_id=cve_id,
        cwe_type=cwe_type or info.get("cwe_id", "Unknown"),
        function_name=function_name,
        function_prototype=function_proto,
        language=language,
        vulnerable_code=vuln_code_clean,
        patched_code=generated_patch.strip(),
    )

    raw_response = call_llm(SYSTEM_PROMPT, user_prompt, model=model)
    result = parse_llm_response(raw_response)
    result["raw_response"] = raw_response
    return result


# ── Batch Evaluation ─────────────────────────────────────────────────────────

def evaluate_results_file(
    results_path: Path,
    output_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    max_items: int | None = None,
    delay: float = 1.0,
) -> Path:
    """Evaluate all patches in a results.jsonl file.

    Parameters
    ----------
    results_path : Path to results.jsonl with generated patches
    output_path : Where to write the evaluation JSONL (default: same dir)
    model : LLM model to use for evaluation
    max_items : Limit number of items to evaluate (for testing)
    delay : Seconds to wait between LLM calls (rate limiting)
    """
    if output_path is None:
        output_path = results_path.parent / "llm_vulnerability_eval.jsonl"

    # Load results
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if max_items:
        results = results[:max_items]

    print(f"Evaluating {len(results)} patches with LLM ({model})...")
    print(f"Output: {output_path}")

    # Check for existing progress (resume support)
    completed_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    key = f"{entry['query_cve']}_{entry['query_variant']}"
                    completed_keys.add(key)
        print(f"Resuming: {len(completed_keys)} already evaluated")

    stats = defaultdict(int)

    with open(output_path, "a") as out_f:
        for i, result in enumerate(results):
            query_cve = result.get("query_cve", "")
            query_variant = result.get("query_variant", "")
            query_cwe = result.get("query_cwe", "")
            key = f"{query_cve}_{query_variant}"

            if key in completed_keys:
                continue

            print(f"  [{i+1}/{len(results)}] {query_cve} ({query_variant})...", end=" ", flush=True)

            try:
                evaluation = evaluate_single(
                    cve_id=query_cve,
                    cwe_type=query_cwe,
                    generated_patch=result.get("generated_patch", ""),
                    model=model,
                )
            except Exception as e:
                evaluation = {
                    "verdict": "ERROR",
                    "confidence": 0.0,
                    "reasoning": f"Exception: {e}",
                    "fix_description": "",
                    "issues": [str(e)],
                }

            entry = {
                "query_cve": query_cve,
                "query_variant": query_variant,
                "query_cwe": query_cwe,
                "verdict": evaluation.get("verdict", "ERROR"),
                "confidence": evaluation.get("confidence", 0.0),
                "reasoning": evaluation.get("reasoning", ""),
                "fix_description": evaluation.get("fix_description", ""),
                "issues": evaluation.get("issues", []),
            }

            out_f.write(json.dumps(entry) + "\n")
            out_f.flush()

            verdict = evaluation.get("verdict", "ERROR")
            stats[verdict] += 1
            print(f"{verdict} (conf={evaluation.get('confidence', 0):.2f})")

            if delay > 0:
                time.sleep(delay)

    # Generate summary
    print(f"\n{'='*70}")
    print("LLM VULNERABILITY EVALUATION SUMMARY")
    print(f"{'='*70}")

    # Reload all results for final summary
    all_entries = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                all_entries.append(json.loads(line))

    total = len(all_entries)
    verdict_counts = defaultdict(int)
    cwe_verdicts = defaultdict(lambda: defaultdict(int))
    cve_verdicts = defaultdict(lambda: defaultdict(int))

    for entry in all_entries:
        v = entry.get("verdict", "ERROR")
        verdict_counts[v] += 1
        cwe_verdicts[entry.get("query_cwe", "Unknown")][v] += 1
        cve_verdicts[entry.get("query_cve", "Unknown")][v] += 1

    print(f"\nTotal evaluated: {total}")
    for v in ["FIXED", "PARTIAL", "NOT_FIXED", "ERROR"]:
        count = verdict_counts.get(v, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {v:<12}: {count:>4} ({pct:.1f}%)")

    fix_rate = (verdict_counts.get("FIXED", 0) + verdict_counts.get("PARTIAL", 0)) / total * 100 if total else 0
    print(f"\n  Fix rate (FIXED+PARTIAL): {fix_rate:.1f}%")

    # CWE breakdown
    print(f"\n{'CWE Type':<45} {'FIXED':>6} {'PARTIAL':>8} {'NOT_FIXED':>10} {'Total':>6}")
    print(f"{'-'*45} {'-'*6} {'-'*8} {'-'*10} {'-'*6}")
    for cwe in sorted(cwe_verdicts.keys()):
        d = cwe_verdicts[cwe]
        t = sum(d.values())
        print(f"{cwe:<45} {d.get('FIXED',0):>6} {d.get('PARTIAL',0):>8} {d.get('NOT_FIXED',0):>10} {t:>6}")

    # Save summary JSON
    summary_path = output_path.with_name("llm_vulnerability_eval_summary.json")
    summary = {
        "total": total,
        "verdicts": dict(verdict_counts),
        "fix_rate_percent": fix_rate,
        "model": model,
        "cwe_breakdown": {k: dict(v) for k, v in cwe_verdicts.items()},
        "cve_breakdown": {k: dict(v) for k, v in cve_verdicts.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based vulnerability patch verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m src.evaluate.llm_evaluation results.jsonl
              python -m src.evaluate.llm_evaluation results.jsonl --model gpt-4o --max 10
              python -m src.evaluate.llm_evaluation results.jsonl --output eval_llm.jsonl
        """),
    )
    parser.add_argument("results", type=Path, help="Path to results.jsonl file")
    parser.add_argument("--output", "-o", type=Path, help="Output JSONL path")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--max", type=int, default=None, help="Max items to evaluate (for testing)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between LLM calls in seconds")

    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found")
        sys.exit(1)

    evaluate_results_file(
        results_path=args.results,
        output_path=args.output,
        model=args.model,
        max_items=args.max,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
