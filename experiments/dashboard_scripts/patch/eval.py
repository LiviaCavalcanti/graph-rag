"""
Data formatting and analysis logic for patch evaluation.

Responsibilities:
  - Build per-record analysis from results + evaluation JSONL
  - Load LLM evaluations and human labels
  - Compute aggregates (per-metric, per-CWE, per-variant)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

from helpers import load_jsonl, load_input_code, load_ground_truth, summarize_metric


# ── core analysis ────────────────────────────────────────────────────

def build_record(result: dict, evaluation: dict, base_dir: Path) -> dict:
    """Merge a results.jsonl record with its evaluation.jsonl counterpart."""
    cve_id = result.get("query_cve", "")
    variant = result.get("query_variant", "")

    input_code = load_input_code(cve_id, base_dir)
    ground_truth = load_ground_truth(cve_id, variant, base_dir)
    generated = (result.get("generated_patch") or "").strip()

    retrieval = result.get("retrieval", {})
    metrics = evaluation.get("metrics_vs_function_body", {})
    size = evaluation.get("size_info", {})

    return {
        "query_cve": cve_id,
        "query_cwe": result.get("query_cwe"),
        "query_variant": variant,
        "example_cve": result.get("example_cve"),
        "example_variant": result.get("example_variant"),
        # retrieval
        "retrieval": {
            "cve_match": result.get("cve_match"),
            "cwe_match": result.get("cwe_match"),
            "similarity": result.get("similarity"),
            "retrieved_variant": retrieval.get("retrieved_variant"),
        },
        # code triple
        "input_code": input_code,
        "ground_truth": ground_truth,
        "generated_patch": generated,
        # evaluation scores
        "scores": metrics,
        "size_info": size,
        "status": result.get("status"),
        "elapsed_s": result.get("elapsed_s"),
    }


def _load_llm_eval(run_dir: Path) -> dict[tuple[str, str], dict]:
    """Load LLM vulnerability evaluation results if available."""
    llm_path = run_dir / "llm_vulnerability_eval.jsonl"
    if not llm_path.exists():
        return {}
    index: dict[tuple[str, str], dict] = {}
    for entry in load_jsonl(llm_path):
        key = (entry.get("query_cve", ""), entry.get("query_variant", ""))
        index[key] = entry
    return index


def _load_human_labels(run_dir: Path) -> dict[tuple[str, str], dict]:
    """Load human labeling results if available.

    Expected format per line:
        {"query_cve": "...", "query_variant": "...", "verdict": "FIXED|PARTIAL|NOT_FIXED",
         "notes": "...", "labeler": "..."}
    """
    label_path = run_dir / "human_labels.jsonl"
    if not label_path.exists():
        return {}
    index: dict[tuple[str, str], dict] = {}
    for entry in load_jsonl(label_path):
        key = (entry.get("query_cve", ""), entry.get("query_variant", ""))
        index[key] = entry
    return index


def analyze(
    results_path: Path,
    evaluation_path: Path,
    base_dir: Path,
) -> dict:
    """Run full analysis pipeline: load data, build records, compute aggregates."""
    results = load_jsonl(results_path)
    evaluations = load_jsonl(evaluation_path)
    llm_eval_index = _load_llm_eval(results_path.parent)
    human_label_index = _load_human_labels(results_path.parent)

    # Index evaluations by (query_cve, query_variant)
    eval_index: dict[tuple[str, str], dict] = {}
    for ev in evaluations:
        key = (ev.get("query_cve", ""), ev.get("query_variant", ""))
        eval_index[key] = ev

    records = []
    for r in results:
        key = (r.get("query_cve", ""), r.get("query_variant", ""))
        ev = eval_index.get(key, {})
        rec = build_record(r, ev, base_dir)

        # Attach LLM evaluation if available
        llm = llm_eval_index.get(key)
        if llm:
            rec["llm_eval"] = {
                "verdict": llm.get("verdict", ""),
                "confidence": llm.get("confidence", 0.0),
                "reasoning": llm.get("reasoning", ""),
                "fix_description": llm.get("fix_description", ""),
                "issues": llm.get("issues", []),
            }
        else:
            rec["llm_eval"] = None

        # Attach human label if available
        hl = human_label_index.get(key)
        if hl:
            rec["human_label"] = {
                "verdict": hl.get("verdict", ""),
                "notes": hl.get("notes", ""),
                "labeler": hl.get("labeler", ""),
            }
        else:
            rec["human_label"] = None

        records.append(rec)

    # Aggregate scores
    metric_keys = [
        "exact_match", "normalised_exact_match",
        "char_sequence_ratio", "line_sequence_ratio",
        "normalised_edit_distance",
        "token_jaccard", "token_jaccard_multiset",
        "bleu_1", "bleu_2", "bleu_4",
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
        "rouge1_f1", "rouge2_f1", "rougeL_f1", "rougeL_precision", "rougeL_recall",
    ]
    aggregates = {}
    for mk in metric_keys:
        vals = [
            rec["scores"][mk]
            for rec in records
            if mk in rec.get("scores", {}) and isinstance(rec["scores"][mk], (int, float))
        ]
        aggregates[mk] = summarize_metric([float(v) for v in vals])

    # Per-CWE breakdown of key metrics
    by_cwe: dict[str, list[dict]] = {}
    for rec in records:
        cwe = rec.get("query_cwe", "unknown")
        by_cwe.setdefault(cwe, []).append(rec)

    cwe_summary = {}
    for cwe, recs in sorted(by_cwe.items()):
        n = len(recs)
        def _cwe_avg(key, _recs=recs):
            vals = [r["scores"][key] for r in _recs if key in r.get("scores", {}) and isinstance(r["scores"].get(key), (int, float))]
            return round(mean(vals), 4) if vals else None
        cwe_summary[cwe] = {
            "count": n,
            "avg_bleu_4": _cwe_avg("bleu_4"),
            "avg_bertscore_f1": _cwe_avg("bertscore_f1"),
            "avg_token_jaccard": _cwe_avg("token_jaccard"),
            "avg_rouge1_f1": _cwe_avg("rouge1_f1"),
            "avg_rouge2_f1": _cwe_avg("rouge2_f1"),
            "avg_rougeL_f1": _cwe_avg("rougeL_f1"),
        }

    # Per-variant breakdown
    by_variant: dict[str, list[dict]] = {}
    for rec in records:
        var = rec.get("query_variant", "unknown")
        by_variant.setdefault(var, []).append(rec)

    variant_summary = {}
    for var, recs in sorted(by_variant.items()):
        n = len(recs)
        def _var_avg(key, _recs=recs):
            vals = [r["scores"][key] for r in _recs if key in r.get("scores", {}) and isinstance(r["scores"].get(key), (int, float))]
            return round(mean(vals), 4) if vals else None
        variant_summary[var] = {
            "count": n,
            "avg_bleu_4": _var_avg("bleu_4"),
            "avg_bertscore_f1": _var_avg("bertscore_f1"),
            "avg_token_jaccard": _var_avg("token_jaccard"),
            "avg_rouge1_f1": _var_avg("rouge1_f1"),
            "avg_rouge2_f1": _var_avg("rouge2_f1"),
            "avg_rougeL_f1": _var_avg("rougeL_f1"),
        }

    # LLM evaluation summary
    llm_summary = None
    llm_records = [r for r in records if r.get("llm_eval")]
    if llm_records:
        verdict_counts = Counter(r["llm_eval"]["verdict"] for r in llm_records)
        total_llm = len(llm_records)
        llm_summary = {
            "total": total_llm,
            "verdicts": dict(verdict_counts),
            "fix_rate": round(
                (verdict_counts.get("FIXED", 0) + verdict_counts.get("PARTIAL", 0))
                / total_llm * 100, 1
            ) if total_llm else 0,
            "avg_confidence": round(
                mean(r["llm_eval"]["confidence"] for r in llm_records), 3
            ),
        }

    # Human labeling summary
    human_summary = None
    human_records = [r for r in records if r.get("human_label")]
    if human_records:
        h_verdict_counts = Counter(r["human_label"]["verdict"] for r in human_records)
        total_human = len(human_records)
        human_summary = {
            "total": total_human,
            "verdicts": dict(h_verdict_counts),
            "fix_rate": round(
                (h_verdict_counts.get("FIXED", 0) + h_verdict_counts.get("PARTIAL", 0))
                / total_human * 100, 1
            ) if total_human else 0,
        }

    return {
        "source": {
            "results": str(results_path),
            "evaluation": str(evaluation_path),
            "base_dir": str(base_dir),
        },
        "total_records": len(records),
        "aggregates": aggregates,
        "by_cwe": cwe_summary,
        "by_variant": variant_summary,
        "llm_evaluation": llm_summary,
        "human_evaluation": human_summary,
        "records": records,
    }