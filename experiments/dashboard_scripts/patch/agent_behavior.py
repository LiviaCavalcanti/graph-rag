"""
Agent behavior analysis — extracts and aggregates agent-level signals from the
``transcript``, ``n_turns``, and ``n_tool_calls`` fields logged in results.jsonl.

Answers questions such as:
  - How many rounds/turns did the agent take per case and per CWE?
  - Did the agent call verify_fix_correctness?
  - Did it respect an INCORRECT verdict or submit anyway?
  - How often did submit_patch fail (SEARCH block not found) before succeeding?
  - Do these patterns correlate with patch quality (BERTScore)?
  - Do C++ cases have higher error rates than C cases?
  - Is retrieval worse for C++ code?
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


_TOOL_ORDER = [
    "get_vulnerability_context",
    "analyze_example_fix",
    "verify_fix_correctness",
    "submit_patch",
]

_TOOL_LABELS = {
    "get_vulnerability_context": "get_vuln_ctx",
    "analyze_example_fix": "analyze_example",
    "verify_fix_correctness": "verify",
    "submit_patch": "submit",
}

_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT|UNCERTAIN)", re.IGNORECASE)
_SUBMIT_ERROR_RE = re.compile(r"Error:.*SEARCH block", re.IGNORECASE)


# ── Language index ────────────────────────────────────────────────────────────

def load_language_index(split_dir: str | Path | None) -> dict:
    """
    Build a two-tier language lookup from the dataset's query_pairs.json.

    Tier 1: (cve_id, variant) → language  (fast, works for ~98% of records)
    Tier 2: source_before_prefix → language  (fallback for mixed-language CVEs)

    Returns a dict with keys "by_cve_variant" and "by_code_prefix".
    Returns an empty index gracefully when split_dir is None or file not found.
    """
    index: dict[str, Any] = {"by_cve_variant": {}, "by_code_prefix": {}}
    if split_dir is None:
        return index

    qp_path = Path(split_dir) / "query_pairs.json"
    if not qp_path.exists():
        return index

    try:
        data = json.loads(qp_path.read_text())
        entries = data.get("entries", []) if isinstance(data, dict) else data
    except Exception:
        return index

    # Find (cve_id, variant) groups where language is mixed — those need code fallback
    from collections import defaultdict
    by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for e in entries:
        lang = e.get("language", "")
        if lang:
            by_key[(e.get("cve_id", ""), e.get("variant", ""))].add(lang)

    mixed_keys: set[tuple[str, str]] = {k for k, langs in by_key.items() if len(langs) > 1}

    for e in entries:
        cve_id = e.get("cve_id", "")
        variant = e.get("variant", "")
        lang = e.get("language", "")
        if not lang:
            continue
        key = (cve_id, variant)
        if key not in mixed_keys:
            # Safe to store by cve+variant (single language)
            index["by_cve_variant"][key] = lang
        # Always build code-prefix fallback for individual functions
        source_before = e.get("source_before", "")
        if source_before:
            prefix = source_before[:300]
            index["by_code_prefix"][prefix] = lang

    return index


def infer_language(record: dict, lang_index: dict) -> str:
    """
    Infer C/C++ language for a record using the two-tier index.
    Returns "C", "C++", or "unknown".
    """
    if not lang_index or (not lang_index.get("by_cve_variant") and not lang_index.get("by_code_prefix")):
        return "unknown"

    cve_id = record.get("query_cve", "")
    variant = record.get("query_variant", "")

    # Tier 1: fast lookup by (cve_id, variant)
    lang = lang_index["by_cve_variant"].get((cve_id, variant))
    if lang:
        return lang

    # Tier 2: code prefix match (for mixed-language CVEs)
    target_code = record.get("_raw_result", {}).get("target_code", "")
    if target_code:
        prefix = target_code[:300]
        lang = lang_index["by_code_prefix"].get(prefix)
        if lang:
            return lang

    return "unknown"


# ── Per-record behavior extraction ───────────────────────────────────────────

def extract_behavior(record: dict) -> dict:
    """
    Parse behavior signals from a single enriched record (as returned by
    ``build_record``).  Safe for records that have no ``transcript`` (e.g.
    single-turn agents) — all fields default gracefully.
    """
    raw = record.get("_raw_result", {})
    transcript: dict[str, Any] = raw.get("transcript") or {}
    tool_calls: list[dict] = transcript.get("tool_calls") or []
    turns: list[dict] = transcript.get("turns") or []

    n_turns = raw.get("n_turns", len(turns) or None)
    n_tool_calls = raw.get("n_tool_calls", len(tool_calls) or None)
    status = raw.get("status", record.get("status", "unknown"))
    finish_reason = raw.get("finish_reason", "unknown")
    architecture = raw.get("architecture", "unknown")

    tool_sequence = [tc.get("tool", "") for tc in tool_calls]

    # ── verify_fix_correctness analysis ──────────────────────────
    verify_calls = [tc for tc in tool_calls if tc.get("tool") == "verify_fix_correctness"]
    called_verify = len(verify_calls) > 0

    # Parse verdict from the output_preview of the LAST verify call
    verify_verdict: str | None = None
    if verify_calls:
        last_verify = verify_calls[-1]
        preview = last_verify.get("output_preview", "")
        m = _VERDICT_RE.search(preview)
        if m:
            verify_verdict = m.group(1).upper()

    # ── submit_patch analysis ─────────────────────────────────────
    submit_calls = [tc for tc in tool_calls if tc.get("tool") == "submit_patch"]
    n_submit_attempts = len(submit_calls)

    # A submit attempt "failed" if the output_preview contains the SEARCH block
    # error message (the agent receives this and should retry).
    submit_errors = [
        tc for tc in submit_calls
        if _SUBMIT_ERROR_RE.search(tc.get("output_preview", ""))
    ]
    submit_failed_once = len(submit_errors) > 0

    # Did the agent submit AFTER an INCORRECT verdict?
    submitted_after_incorrect = False
    if verify_verdict == "INCORRECT" and n_submit_attempts > 0:
        # Find the iteration of the last INCORRECT verdict, then check if any
        # submit_patch came after it (higher iteration index)
        last_incorrect_iter = max(
            (tc.get("iteration", 0) for tc in verify_calls
             if _VERDICT_RE.search(tc.get("output_preview", ""), )),
            default=0,
        )
        last_submit_iter = max(
            (tc.get("iteration", 0) for tc in submit_calls),
            default=-1,
        )
        submitted_after_incorrect = last_submit_iter >= last_incorrect_iter

    # ── tool coverage flags ───────────────────────────────────────
    called_get_vuln = any(tc.get("tool") == "get_vulnerability_context" for tc in tool_calls)
    called_analyze_example = any(tc.get("tool") == "analyze_example_fix" for tc in tool_calls)

    # ── per-turn finish reasons ───────────────────────────────────
    per_turn_finish = [t.get("finish_reason", "") for t in turns]
    hit_length_limit = "length" in per_turn_finish

    return {
        "architecture": architecture,
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "status": status,
        "finish_reason": finish_reason,
        "tool_sequence": tool_sequence,
        "called_get_vuln": called_get_vuln,
        "called_analyze_example": called_analyze_example,
        "called_verify": called_verify,
        "verify_verdict": verify_verdict,  # CORRECT / INCORRECT / UNCERTAIN / None
        "submitted_after_incorrect": submitted_after_incorrect,
        "n_submit_attempts": n_submit_attempts,
        "submit_failed_once": submit_failed_once,
        "hit_length_limit": hit_length_limit,
        "per_turn_finish": per_turn_finish,
    }


# ── Population-level aggregation ─────────────────────────────────────────────

def aggregate_behavior(records: list[dict]) -> dict:
    """
    Compute population-level behavior aggregates from a list of enriched records.
    Each record must already have ``behavior``, ``scores``, and ``language`` sub-dicts/fields.
    """
    behavs = [r.get("behavior", {}) for r in records]
    # Filter out records with no behavior data (e.g. no transcript)
    with_data = [b for b in behavs if b.get("n_turns") is not None]

    total = len(records)
    n_with_data = len(with_data)

    def _pct(numerator: int, denom: int = n_with_data) -> float:
        return round(numerator / denom * 100, 1) if denom else 0.0

    def _safe_mean(vals: list[float]) -> float | None:
        return round(mean(vals), 3) if vals else None

    def _safe_median(vals: list[float]) -> float | None:
        return round(median(vals), 1) if vals else None

    # Basic turn/tool stats
    turns_vals = [b["n_turns"] for b in with_data if b.get("n_turns") is not None]
    tool_vals = [b["n_tool_calls"] for b in with_data if b.get("n_tool_calls") is not None]

    n_called_verify = sum(1 for b in with_data if b.get("called_verify"))
    n_incorrect_verdict = sum(1 for b in with_data if b.get("verify_verdict") == "INCORRECT")
    n_submitted_after_incorrect = sum(1 for b in with_data if b.get("submitted_after_incorrect"))
    n_submit_failed = sum(1 for b in with_data if b.get("submit_failed_once"))
    n_hit_length = sum(1 for b in with_data if b.get("hit_length_limit"))

    # Status distribution
    status_counts: dict[str, int] = Counter(b.get("status", "unknown") for b in behavs)

    # Verdict distribution (among those that called verify)
    verify_verdict_counts: dict[str, int] = Counter(
        b.get("verify_verdict") for b in with_data if b.get("called_verify") and b.get("verify_verdict")
    )

    # Tool usage across all records
    tool_usage: dict[str, int] = Counter(
        tool for b in with_data for tool in b.get("tool_sequence", [])
    )

    # Turns histogram  {n_turns: count}
    turns_hist: dict[int, int] = Counter(t for t in turns_vals if t is not None)

    # ── per-CWE breakdown ──────────────────────────────────────────
    by_cwe: dict[str, list[tuple[dict, dict]]] = {}
    for rec in records:
        cwe = rec.get("query_cwe") or "unknown"
        by_cwe.setdefault(cwe, []).append((rec.get("behavior", {}), rec.get("scores", {})))

    cwe_breakdown: dict[str, dict] = {}
    for cwe, pairs in sorted(by_cwe.items()):
        bs = [b for b, _ in pairs if b.get("n_turns") is not None]
        ss = [s for _, s in pairs]
        n = len(pairs)
        n_b = len(bs)

        cwe_turns = [b["n_turns"] for b in bs if b.get("n_turns") is not None]
        cwe_bert = [s.get("bertscore_f1") for s in ss if isinstance(s.get("bertscore_f1"), float)]
        cwe_verify = sum(1 for b in bs if b.get("called_verify"))
        cwe_violations = sum(1 for b in bs if b.get("submitted_after_incorrect"))
        cwe_success = sum(1 for b in bs if b.get("status") == "success")
        cwe_fails = sum(1 for b in bs if b.get("submit_failed_once"))

        cwe_breakdown[cwe] = {
            "count": n,
            "avg_turns": _safe_mean([float(x) for x in cwe_turns]),
            "avg_bertscore": _safe_mean([float(x) for x in cwe_bert]),
            "verify_rate": _pct(cwe_verify, n_b) if n_b else None,
            "violation_rate": _pct(cwe_violations, n_b) if n_b else None,
            "success_rate": _pct(cwe_success, n_b) if n_b else None,
            "submit_fail_rate": _pct(cwe_fails, n_b) if n_b else None,
        }

    # ── per-language breakdown ─────────────────────────────────────
    by_lang: dict[str, list[dict]] = {}
    for rec in records:
        lang = rec.get("language") or "unknown"
        by_lang.setdefault(lang, []).append(rec)

    lang_breakdown: dict[str, dict] = {}
    for lang, recs in sorted(by_lang.items()):
        bs_recs = [r.get("behavior", {}) for r in recs if r.get("behavior", {}).get("n_turns") is not None]
        ss_recs = [r.get("scores", {}) for r in recs]
        n = len(recs)
        n_b = len(bs_recs)

        lang_turns = [b["n_turns"] for b in bs_recs if b.get("n_turns") is not None]
        lang_bert = [s.get("bertscore_f1") for s in ss_recs if isinstance(s.get("bertscore_f1"), float)]
        lang_bleu = [s.get("bleu_4") for s in ss_recs if isinstance(s.get("bleu_4"), float)]
        lang_verify = sum(1 for b in bs_recs if b.get("called_verify"))
        lang_violations = sum(1 for b in bs_recs if b.get("submitted_after_incorrect"))
        lang_success = sum(1 for b in bs_recs if b.get("status") == "success")
        lang_fails = sum(1 for b in bs_recs if b.get("submit_failed_once"))
        # Retrieval metrics
        lang_cve_match = sum(1 for r in recs if r.get("retrieval", {}).get("cve_match"))
        lang_cwe_match = sum(1 for r in recs if r.get("retrieval", {}).get("cwe_match"))
        lang_sim_vals = [r.get("retrieval", {}).get("similarity") for r in recs
                         if isinstance(r.get("retrieval", {}).get("similarity"), float)]

        lang_breakdown[lang] = {
            "count": n,
            "avg_turns": _safe_mean([float(x) for x in lang_turns]),
            "avg_bertscore": _safe_mean([float(x) for x in lang_bert]),
            "avg_bleu_4": _safe_mean([float(x) for x in lang_bleu]),
            "verify_rate": _pct(lang_verify, n_b) if n_b else None,
            "violation_rate": _pct(lang_violations, n_b) if n_b else None,
            "success_rate": _pct(lang_success, n_b) if n_b else None,
            "submit_fail_rate": _pct(lang_fails, n_b) if n_b else None,
            "error_rate": round((n - lang_success) / n * 100, 1) if n else 0.0,
            "cve_match_rate": _pct(lang_cve_match, n),
            "cwe_match_rate": _pct(lang_cwe_match, n),
            "avg_retrieval_sim": _safe_mean([float(x) for x in lang_sim_vals]),
        }

    # ── per-language × CWE breakdown (detect which lang×CWE combos are worst) ──
    by_lang_cwe: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        lang = rec.get("language") or "unknown"
        cwe = rec.get("query_cwe") or "unknown"
        by_lang_cwe.setdefault((lang, cwe), []).append(rec)

    lang_cwe_breakdown: dict[str, dict] = {}
    for (lang, cwe), recs in sorted(by_lang_cwe.items()):
        ss = [r.get("scores", {}) for r in recs]
        bs_recs = [r.get("behavior", {}) for r in recs if r.get("behavior", {}).get("n_turns") is not None]
        n = len(recs)
        bert_vals = [s.get("bertscore_f1") for s in ss if isinstance(s.get("bertscore_f1"), float)]
        success_n = sum(1 for b in bs_recs if b.get("status") == "success")
        lang_cwe_breakdown[f"{lang}|{cwe}"] = {
            "language": lang,
            "cwe": cwe,
            "count": n,
            "avg_bertscore": _safe_mean([float(x) for x in bert_vals]),
            "error_rate": round((n - success_n) / n * 100, 1) if n else 0.0,
        }

    # ── verification violations list ──────────────────────────────
    violations = []
    for rec in records:
        b = rec.get("behavior", {})
        if b.get("submitted_after_incorrect"):
            violations.append({
                "query_cve": rec.get("query_cve", ""),
                "query_cwe": rec.get("query_cwe", ""),
                "query_variant": rec.get("query_variant", ""),
                "language": rec.get("language", "unknown"),
                "verify_verdict": b.get("verify_verdict"),
                "n_turns": b.get("n_turns"),
                "bertscore_f1": rec.get("scores", {}).get("bertscore_f1"),
                "bleu_4": rec.get("scores", {}).get("bleu_4"),
                "status": b.get("status"),
            })

    return {
        "total_records": total,
        "records_with_behavior_data": n_with_data,
        # turn/tool overview
        "avg_turns": _safe_mean([float(x) for x in turns_vals]),
        "median_turns": _safe_median([float(x) for x in turns_vals]),
        "avg_tool_calls": _safe_mean([float(x) for x in tool_vals]),
        "turns_histogram": dict(sorted(turns_hist.items())),
        # tool coverage
        "tool_usage": dict(tool_usage.most_common()),
        "pct_called_verify": _pct(n_called_verify),
        "pct_called_get_vuln": _pct(sum(1 for b in with_data if b.get("called_get_vuln"))),
        "pct_called_analyze_example": _pct(sum(1 for b in with_data if b.get("called_analyze_example"))),
        # verification compliance
        "n_called_verify": n_called_verify,
        "verify_verdict_counts": dict(verify_verdict_counts),
        "n_incorrect_verdicts": n_incorrect_verdict,
        "n_submitted_after_incorrect": n_submitted_after_incorrect,
        "pct_violated_verdict": _pct(n_submitted_after_incorrect, max(n_incorrect_verdict, 1)),
        # submission errors
        "n_submit_failed_once": n_submit_failed,
        "pct_submit_failed": _pct(n_submit_failed),
        # token pressure
        "n_hit_length_limit": n_hit_length,
        "pct_hit_length": _pct(n_hit_length),
        # status
        "status_counts": dict(status_counts),
        # breakdowns
        "by_cwe": cwe_breakdown,
        "by_language": lang_breakdown,
        "by_language_cwe": lang_cwe_breakdown,
        "violations": violations,
    }


Answers questions such as:
  - How many rounds/turns did the agent take per case and per CWE?
  - Did the agent call verify_fix_correctness?
  - Did it respect an INCORRECT verdict or submit anyway?
  - How often did submit_patch fail (SEARCH block not found) before succeeding?
  - Do these patterns correlate with patch quality (BERTScore)?
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean, median
from typing import Any


_TOOL_ORDER = [
    "get_vulnerability_context",
    "analyze_example_fix",
    "verify_fix_correctness",
    "submit_patch",
]

_TOOL_LABELS = {
    "get_vulnerability_context": "get_vuln_ctx",
    "analyze_example_fix": "analyze_example",
    "verify_fix_correctness": "verify",
    "submit_patch": "submit",
}

_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT|UNCERTAIN)", re.IGNORECASE)
_SUBMIT_ERROR_RE = re.compile(r"Error:.*SEARCH block", re.IGNORECASE)


def extract_behavior(record: dict) -> dict:
    """
    Parse behavior signals from a single enriched record (as returned by
    ``build_record``).  Safe for records that have no ``transcript`` (e.g.
    single-turn agents) — all fields default gracefully.
    """
    raw = record.get("_raw_result", {})
    transcript: dict[str, Any] = raw.get("transcript") or {}
    tool_calls: list[dict] = transcript.get("tool_calls") or []
    turns: list[dict] = transcript.get("turns") or []

    n_turns = raw.get("n_turns", len(turns) or None)
    n_tool_calls = raw.get("n_tool_calls", len(tool_calls) or None)
    status = raw.get("status", record.get("status", "unknown"))
    finish_reason = raw.get("finish_reason", "unknown")
    architecture = raw.get("architecture", "unknown")

    tool_sequence = [tc.get("tool", "") for tc in tool_calls]

    # ── verify_fix_correctness analysis ──────────────────────────
    verify_calls = [tc for tc in tool_calls if tc.get("tool") == "verify_fix_correctness"]
    called_verify = len(verify_calls) > 0

    # Parse verdict from the output_preview of the LAST verify call
    verify_verdict: str | None = None
    if verify_calls:
        last_verify = verify_calls[-1]
        preview = last_verify.get("output_preview", "")
        m = _VERDICT_RE.search(preview)
        if m:
            verify_verdict = m.group(1).upper()

    # ── submit_patch analysis ─────────────────────────────────────
    submit_calls = [tc for tc in tool_calls if tc.get("tool") == "submit_patch"]
    n_submit_attempts = len(submit_calls)

    # A submit attempt "failed" if the output_preview contains the SEARCH block
    # error message (the agent receives this and should retry).
    submit_errors = [
        tc for tc in submit_calls
        if _SUBMIT_ERROR_RE.search(tc.get("output_preview", ""))
    ]
    submit_failed_once = len(submit_errors) > 0

    # Did the agent submit AFTER an INCORRECT verdict?
    submitted_after_incorrect = False
    if verify_verdict == "INCORRECT" and n_submit_attempts > 0:
        # Find the iteration of the last INCORRECT verdict, then check if any
        # submit_patch came after it (higher iteration index)
        last_incorrect_iter = max(
            (tc.get("iteration", 0) for tc in verify_calls
             if _VERDICT_RE.search(tc.get("output_preview", ""), )),
            default=0,
        )
        last_submit_iter = max(
            (tc.get("iteration", 0) for tc in submit_calls),
            default=-1,
        )
        submitted_after_incorrect = last_submit_iter >= last_incorrect_iter

    # ── tool coverage flags ───────────────────────────────────────
    called_get_vuln = any(tc.get("tool") == "get_vulnerability_context" for tc in tool_calls)
    called_analyze_example = any(tc.get("tool") == "analyze_example_fix" for tc in tool_calls)

    # ── per-turn finish reasons ───────────────────────────────────
    per_turn_finish = [t.get("finish_reason", "") for t in turns]
    hit_length_limit = "length" in per_turn_finish

    return {
        "architecture": architecture,
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "status": status,
        "finish_reason": finish_reason,
        "tool_sequence": tool_sequence,
        "called_get_vuln": called_get_vuln,
        "called_analyze_example": called_analyze_example,
        "called_verify": called_verify,
        "verify_verdict": verify_verdict,  # CORRECT / INCORRECT / UNCERTAIN / None
        "submitted_after_incorrect": submitted_after_incorrect,
        "n_submit_attempts": n_submit_attempts,
        "submit_failed_once": submit_failed_once,
        "hit_length_limit": hit_length_limit,
        "per_turn_finish": per_turn_finish,
    }


def aggregate_behavior(records: list[dict]) -> dict:
    """
    Compute population-level behavior aggregates from a list of enriched records.
    Each record must already have ``behavior`` and ``scores`` sub-dicts.
    """
    behavs = [r.get("behavior", {}) for r in records]
    # Filter out records with no behavior data (e.g. no transcript)
    with_data = [b for b in behavs if b.get("n_turns") is not None]

    total = len(records)
    n_with_data = len(with_data)

    def _pct(numerator: int, denom: int = n_with_data) -> float:
        return round(numerator / denom * 100, 1) if denom else 0.0

    def _safe_mean(vals: list[float]) -> float | None:
        return round(mean(vals), 3) if vals else None

    def _safe_median(vals: list[float]) -> float | None:
        return round(median(vals), 1) if vals else None

    # Basic turn/tool stats
    turns_vals = [b["n_turns"] for b in with_data if b.get("n_turns") is not None]
    tool_vals = [b["n_tool_calls"] for b in with_data if b.get("n_tool_calls") is not None]

    n_called_verify = sum(1 for b in with_data if b.get("called_verify"))
    n_incorrect_verdict = sum(1 for b in with_data if b.get("verify_verdict") == "INCORRECT")
    n_submitted_after_incorrect = sum(1 for b in with_data if b.get("submitted_after_incorrect"))
    n_submit_failed = sum(1 for b in with_data if b.get("submit_failed_once"))
    n_hit_length = sum(1 for b in with_data if b.get("hit_length_limit"))

    # Status distribution
    status_counts: dict[str, int] = Counter(b.get("status", "unknown") for b in behavs)

    # Verdict distribution (among those that called verify)
    verify_verdict_counts: dict[str, int] = Counter(
        b.get("verify_verdict") for b in with_data if b.get("called_verify") and b.get("verify_verdict")
    )

    # Tool usage across all records
    tool_usage: dict[str, int] = Counter(
        tool for b in with_data for tool in b.get("tool_sequence", [])
    )

    # Turns histogram  {n_turns: count}
    turns_hist: dict[int, int] = Counter(t for t in turns_vals if t is not None)

    # ── per-CWE breakdown ──────────────────────────────────────────
    by_cwe: dict[str, list[tuple[dict, dict]]] = {}
    for rec in records:
        cwe = rec.get("query_cwe") or "unknown"
        by_cwe.setdefault(cwe, []).append((rec.get("behavior", {}), rec.get("scores", {})))

    cwe_breakdown: dict[str, dict] = {}
    for cwe, pairs in sorted(by_cwe.items()):
        bs = [b for b, _ in pairs if b.get("n_turns") is not None]
        ss = [s for _, s in pairs]
        n = len(pairs)
        n_b = len(bs)

        cwe_turns = [b["n_turns"] for b in bs if b.get("n_turns") is not None]
        cwe_bert = [s.get("bertscore_f1") for s in ss if isinstance(s.get("bertscore_f1"), float)]
        cwe_verify = sum(1 for b in bs if b.get("called_verify"))
        cwe_violations = sum(1 for b in bs if b.get("submitted_after_incorrect"))
        cwe_success = sum(1 for b in bs if b.get("status") == "success")
        cwe_fails = sum(1 for b in bs if b.get("submit_failed_once"))

        cwe_breakdown[cwe] = {
            "count": n,
            "avg_turns": _safe_mean([float(x) for x in cwe_turns]),
            "avg_bertscore": _safe_mean([float(x) for x in cwe_bert]),
            "verify_rate": _pct(cwe_verify, n_b) if n_b else None,
            "violation_rate": _pct(cwe_violations, n_b) if n_b else None,
            "success_rate": _pct(cwe_success, n_b) if n_b else None,
            "submit_fail_rate": _pct(cwe_fails, n_b) if n_b else None,
        }

    # ── verification violations list ──────────────────────────────
    violations = []
    for rec in records:
        b = rec.get("behavior", {})
        if b.get("submitted_after_incorrect"):
            violations.append({
                "query_cve": rec.get("query_cve", ""),
                "query_cwe": rec.get("query_cwe", ""),
                "query_variant": rec.get("query_variant", ""),
                "verify_verdict": b.get("verify_verdict"),
                "n_turns": b.get("n_turns"),
                "bertscore_f1": rec.get("scores", {}).get("bertscore_f1"),
                "bleu_4": rec.get("scores", {}).get("bleu_4"),
                "status": b.get("status"),
            })

    return {
        "total_records": total,
        "records_with_behavior_data": n_with_data,
        # turn/tool overview
        "avg_turns": _safe_mean([float(x) for x in turns_vals]),
        "median_turns": _safe_median([float(x) for x in turns_vals]),
        "avg_tool_calls": _safe_mean([float(x) for x in tool_vals]),
        "turns_histogram": dict(sorted(turns_hist.items())),
        # tool coverage
        "tool_usage": dict(tool_usage.most_common()),
        "pct_called_verify": _pct(n_called_verify),
        "pct_called_get_vuln": _pct(sum(1 for b in with_data if b.get("called_get_vuln"))),
        "pct_called_analyze_example": _pct(sum(1 for b in with_data if b.get("called_analyze_example"))),
        # verification compliance
        "n_called_verify": n_called_verify,
        "verify_verdict_counts": dict(verify_verdict_counts),
        "n_incorrect_verdicts": n_incorrect_verdict,
        "n_submitted_after_incorrect": n_submitted_after_incorrect,
        "pct_violated_verdict": _pct(n_submitted_after_incorrect, max(n_incorrect_verdict, 1)),
        # submission errors
        "n_submit_failed_once": n_submit_failed,
        "pct_submit_failed": _pct(n_submit_failed),
        # token pressure
        "n_hit_length_limit": n_hit_length,
        "pct_hit_length": _pct(n_hit_length),
        # status
        "status_counts": dict(status_counts),
        # breakdowns
        "by_cwe": cwe_breakdown,
        "violations": violations,
    }
