#!/usr/bin/env python3
"""
Verify whether crossing (combining) retrieval strategies can recover
the combined approach's uncertain misses.

For each query, collects the top-1 prediction and ranking from every
approach, then simulates several fusion strategies:

  1. Oracle:        any approach hits → hit  (theoretical ceiling)
  2. Majority vote: pick the CVE most approaches rank at #1
  3. Confidence-weighted vote: weight each vote by (1 - uncertainty)
  4. Rank fusion (RRF): reciprocal-rank fusion across all approaches
  5. Fallback cascade: use combined; when uncertain, fall back to
     the next-best approach that IS confident

Reads:  experiments/output/<run_id>/results.json
Writes: experiments/output/<run_id>/crossing_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── helpers ──────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    temperature = max(temperature, 1e-8)
    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps)
    return [e / z for e in exps] if z > 0 else [0.0] * len(exps)


def _is_uncertain(query: dict, prob_floor: float, margin_floor: float,
                  temperature: float) -> bool:
    """Return True when the retrieval for this query is uncertain."""
    retrieved = query.get("retrieved") or []
    if len(retrieved) < 2:
        return True
    scores = [_safe_float(r.get("score")) for r in retrieved]
    probs = _softmax(scores, temperature)
    if probs[0] < prob_floor:
        return True
    margin = scores[0] - scores[1]
    if margin < margin_floor:
        return True
    return False


def _hit_at_1(query: dict) -> bool:
    retrieved = query.get("retrieved") or []
    if not retrieved:
        return False
    return retrieved[0].get("cve_id") == query.get("query_cve")


def _top1_cve(query: dict) -> str | None:
    retrieved = query.get("retrieved") or []
    return retrieved[0].get("cve_id") if retrieved else None


def _cve_rank(query: dict, cve: str) -> int | None:
    """1-based rank of `cve` in the retrieved list, or None."""
    for i, r in enumerate(query.get("retrieved") or [], start=1):
        if r.get("cve_id") == cve:
            return i
    return None


# ── fusion strategies ────────────────────────────────────────────────

def oracle_fusion(per_approach: dict[str, dict]) -> str | None:
    """Return the correct CVE if ANY approach gets hit@1."""
    for q in per_approach.values():
        if _hit_at_1(q):
            return q["query_cve"]
    return _top1_cve(next(iter(per_approach.values())))


def majority_vote(per_approach: dict[str, dict]) -> str | None:
    """Pick the CVE that most approaches place at rank 1."""
    votes: Counter = Counter()
    for q in per_approach.values():
        cve = _top1_cve(q)
        if cve:
            votes[cve] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def confidence_weighted_vote(
    per_approach: dict[str, dict],
    prob_floor: float,
    margin_floor: float,
    temperature: float,
) -> str | None:
    """Weight each approach's rank-1 vote by its confidence."""
    weighted: defaultdict[str, float] = defaultdict(float)
    for name, q in per_approach.items():
        cve = _top1_cve(q)
        if not cve:
            continue
        uncertain = _is_uncertain(q, prob_floor, margin_floor, temperature)
        weight = 0.25 if uncertain else 1.0
        weighted[cve] += weight
    if not weighted:
        return None
    return max(weighted, key=weighted.get)  # type: ignore[arg-type]


def reciprocal_rank_fusion(
    per_approach: dict[str, dict],
    k: int = 60,
    top_k: int = 10,
) -> str | None:
    """
    Reciprocal Rank Fusion (Cormack+ 2009).
    RRF_score(cve) = sum over approaches of 1 / (k + rank_in_that_approach).
    """
    rrf: defaultdict[str, float] = defaultdict(float)
    for q in per_approach.values():
        for r in (q.get("retrieved") or [])[:top_k]:
            cve = r.get("cve_id")
            rank = r.get("rank", 999)
            if cve:
                rrf[cve] += 1.0 / (k + rank)
    if not rrf:
        return None
    return max(rrf, key=rrf.get)  # type: ignore[arg-type]


def fallback_cascade(
    per_approach: dict[str, dict],
    cascade_order: list[str],
    prob_floor: float,
    margin_floor: float,
    temperature: float,
) -> str | None:
    """
    Use the first approach in cascade_order that is confident.
    If all are uncertain, use the first approach's answer.
    """
    for name in cascade_order:
        q = per_approach.get(name)
        if q is None:
            continue
        if not _is_uncertain(q, prob_floor, margin_floor, temperature):
            return _top1_cve(q)
    # all uncertain → fall back to first
    first = per_approach.get(cascade_order[0])
    return _top1_cve(first) if first else None


# ── main analysis ────────────────────────────────────────────────────

def analyze_crossing(results_path: str | Path) -> dict:
    results_path = Path(results_path)
    data = json.loads(results_path.read_text())
    cells = data.get("cells", [])

    if not cells:
        return {"error": "no cells in results"}

    # Gather per-approach raw queries, keyed by embedder name
    approach_queries: dict[str, list[dict]] = {}
    for cell in cells:
        name = cell["embedder"]
        raw = cell.get("self_retrieval", {}).get("raw_queries", [])
        if raw:
            approach_queries[name] = raw

    approach_names = list(approach_queries.keys())
    n_queries = len(next(iter(approach_queries.values())))

    # Verify all have the same queries in the same order
    ref_cves = [q["query_cve"] for q in approach_queries[approach_names[0]]]
    for name in approach_names[1:]:
        these = [q["query_cve"] for q in approach_queries[name]]
        assert these == ref_cves, f"Query mismatch between {approach_names[0]} and {name}"

    # Uncertainty thresholds (same as miss analysis defaults)
    prob_floor = 0.12
    margin_floor = 0.005
    temperature = 1.0

    # Determine cascade order: combined first, then by descending hit@1
    hit1_rates = {}
    for name, queries in approach_queries.items():
        hit1_rates[name] = sum(1 for q in queries if _hit_at_1(q)) / len(queries)
    cascade_order = sorted(approach_names, key=lambda n: hit1_rates[n], reverse=True)

    # ── per-query evaluation ──────────────────────────────────────
    strategies = [
        "individual_best",
        "oracle",
        "majority_vote",
        "confidence_weighted_vote",
        "reciprocal_rank_fusion",
        "fallback_cascade",
    ]
    strategy_hits: dict[str, int] = {s: 0 for s in strategies}
    per_query_detail: list[dict] = []

    for i in range(n_queries):
        query_cve = ref_cves[i]
        query_cwe = approach_queries[approach_names[0]][i].get("query_cwe")
        per_approach = {name: approach_queries[name][i] for name in approach_names}

        # Individual best (combined)
        best_name = cascade_order[0]
        individual_pred = _top1_cve(per_approach[best_name])
        individual_hit = individual_pred == query_cve

        oracle_pred = oracle_fusion(per_approach)
        majority_pred = majority_vote(per_approach)
        cw_pred = confidence_weighted_vote(per_approach, prob_floor, margin_floor, temperature)
        rrf_pred = reciprocal_rank_fusion(per_approach)
        cascade_pred = fallback_cascade(
            per_approach, cascade_order, prob_floor, margin_floor, temperature,
        )

        preds = {
            "individual_best": individual_pred,
            "oracle": oracle_pred,
            "majority_vote": majority_pred,
            "confidence_weighted_vote": cw_pred,
            "reciprocal_rank_fusion": rrf_pred,
            "fallback_cascade": cascade_pred,
        }

        row = {
            "query_idx": i,
            "query_cve": query_cve,
            "query_cwe": query_cwe,
        }

        # Per-approach details for this query
        approach_detail = {}
        for name in approach_names:
            q = per_approach[name]
            approach_detail[name] = {
                "top1_cve": _top1_cve(q),
                "hit_at_1": _hit_at_1(q),
                "true_cve_rank": _cve_rank(q, query_cve),
                "uncertain": _is_uncertain(q, prob_floor, margin_floor, temperature),
            }
        row["approaches"] = approach_detail

        # Strategy results
        strategy_results = {}
        for s in strategies:
            pred = preds[s]
            hit = pred == query_cve
            strategy_hits[s] += int(hit)
            strategy_results[s] = {"prediction": pred, "hit": hit}
        row["strategies"] = strategy_results

        per_query_detail.append(row)

    # ── aggregate ─────────────────────────────────────────────────
    summary_rows = []
    for s in strategies:
        h = strategy_hits[s]
        summary_rows.append({
            "strategy": s,
            "hit_at_1": h,
            "total": n_queries,
            "rate": h / n_queries if n_queries else 0,
        })

    # Individual per-approach rates for comparison
    approach_summary = []
    for name in approach_names:
        h = sum(1 for q in approach_queries[name] if _hit_at_1(q))
        approach_summary.append({
            "approach": name,
            "hit_at_1": h,
            "total": n_queries,
            "rate": h / n_queries if n_queries else 0,
        })

    # ── combined miss deep-dive ───────────────────────────────────
    combined_name = "combined" if "combined" in approach_names else cascade_order[0]
    combined_misses = []
    for row in per_query_detail:
        if not row["approaches"].get(combined_name, {}).get("hit_at_1", True):
            miss_info = {
                "query_cve": row["query_cve"],
                "query_cwe": row["query_cwe"],
                "combined_uncertain": row["approaches"][combined_name]["uncertain"],
                "combined_true_rank": row["approaches"][combined_name]["true_cve_rank"],
                "rescued_by_approaches": [
                    name for name in approach_names
                    if name != combined_name and row["approaches"][name]["hit_at_1"]
                ],
                "strategy_hits": {
                    s: row["strategies"][s]["hit"]
                    for s in strategies
                },
            }
            combined_misses.append(miss_info)

    # ── pairwise complementarity ──────────────────────────────────
    pairwise = {}
    for a in approach_names:
        for b in approach_names:
            key = f"{a}+{b}"
            union_hit = sum(
                1 for i in range(n_queries)
                if _hit_at_1(approach_queries[a][i]) or _hit_at_1(approach_queries[b][i])
            )
            both_hit = sum(
                1 for i in range(n_queries)
                if _hit_at_1(approach_queries[a][i]) and _hit_at_1(approach_queries[b][i])
            )
            only_a = sum(
                1 for i in range(n_queries)
                if _hit_at_1(approach_queries[a][i]) and not _hit_at_1(approach_queries[b][i])
            )
            only_b = sum(
                1 for i in range(n_queries)
                if not _hit_at_1(approach_queries[a][i]) and _hit_at_1(approach_queries[b][i])
            )
            pairwise[key] = {
                "union_hit": union_hit,
                "union_rate": union_hit / n_queries if n_queries else 0,
                "both_hit": both_hit,
                "only_a": only_a,
                "only_b": only_b,
                "neither": n_queries - union_hit,
            }

    report = {
        "run_id": data.get("run_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "prob_floor": prob_floor,
            "margin_floor": margin_floor,
            "temperature": temperature,
            "cascade_order": cascade_order,
        },
        "n_queries": n_queries,
        "n_approaches": len(approach_names),
        "approaches": approach_names,
        "approach_summary": approach_summary,
        "strategy_summary": summary_rows,
        "per_query_detail": per_query_detail,
        "pairwise_complementarity": pairwise,
        "combined_miss_deep_dive": {
            "n_misses": len(combined_misses),
            "n_rescued_by_any_approach": sum(
                1 for m in combined_misses if m["rescued_by_approaches"]
            ),
            "n_rescued_by_rrf": sum(
                1 for m in combined_misses if m["strategy_hits"]["reciprocal_rank_fusion"]
            ),
            "n_rescued_by_cascade": sum(
                1 for m in combined_misses if m["strategy_hits"]["fallback_cascade"]
            ),
            "n_all_uncertain": sum(
                1 for m in combined_misses if m["combined_uncertain"]
            ),
            "misses": combined_misses,
        },
    }

    return report


def _print_report(report: dict) -> None:
    print(f"\n{'='*65}")
    print("  CROSSING STRATEGY VERIFICATION")
    print(f"{'='*65}")
    print(f"  Run: {report['run_id']}")
    print(f"  Queries: {report['n_queries']}  |  Approaches: {report['n_approaches']}")
    print(f"  Cascade order: {' → '.join(report['settings']['cascade_order'])}")

    print(f"\n{'─'*65}")
    print("  Individual approach hit@1:")
    print(f"{'─'*65}")
    for row in report["approach_summary"]:
        bar = "█" * int(row["rate"] * 40)
        print(f"    {row['approach']:25s}  {row['hit_at_1']:3d}/{row['total']}  "
              f"({row['rate']:.1%})  {bar}")

    print(f"\n{'─'*65}")
    print("  Fusion strategy hit@1:")
    print(f"{'─'*65}")
    for row in report["strategy_summary"]:
        bar = "█" * int(row["rate"] * 40)
        delta = ""
        best_individual = max(r["rate"] for r in report["approach_summary"])
        diff = row["rate"] - best_individual
        if diff > 0:
            delta = f"  (+{diff:.1%})"
        elif diff < 0:
            delta = f"  ({diff:.1%})"
        print(f"    {row['strategy']:30s}  {row['hit_at_1']:3d}/{row['total']}  "
              f"({row['rate']:.1%}){delta}  {bar}")

    dd = report["combined_miss_deep_dive"]
    print(f"\n{'─'*65}")
    print(f"  Combined approach miss deep-dive ({dd['n_misses']} misses):")
    print(f"{'─'*65}")
    print(f"    All misses are uncertain:     {dd['n_all_uncertain']}/{dd['n_misses']}")
    print(f"    Rescued by any other approach: {dd['n_rescued_by_any_approach']}/{dd['n_misses']}")
    print(f"    Rescued by RRF:                {dd['n_rescued_by_rrf']}/{dd['n_misses']}")
    print(f"    Rescued by fallback cascade:   {dd['n_rescued_by_cascade']}/{dd['n_misses']}")

    print(f"\n    Per-miss detail:")
    for m in dd["misses"]:
        rank_str = f"rank={m['combined_true_rank']}" if m["combined_true_rank"] else "not in top-k"
        rescuers = ", ".join(m["rescued_by_approaches"]) if m["rescued_by_approaches"] else "none"
        strats = [s for s, hit in m["strategy_hits"].items() if hit and s != "individual_best"]
        strat_str = ", ".join(strats) if strats else "none"
        print(f"      {m['query_cve']:20s}  {rank_str:16s}  "
              f"rescued_by=[{rescuers}]  strategies=[{strat_str}]")

    # Pairwise: show only cross-pairs (skip self)
    approach_names = report["approaches"]
    print(f"\n{'─'*65}")
    print("  Pairwise union hit@1 (complementarity):")
    print(f"{'─'*65}")
    header = f"    {'':25s}" + "".join(f"{b:>14s}" for b in approach_names)
    print(header)
    for a in approach_names:
        cells = []
        for b in approach_names:
            key = f"{a}+{b}"
            info = report["pairwise_complementarity"][key]
            cells.append(f"{info['union_rate']:.1%} ({info['only_b']}+)")
        print(f"    {a:25s}" + "".join(f"{c:>14s}" for c in cells))

    print(f"\n{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify crossing strategy performance against combined approach misses."
    )
    parser.add_argument(
        "results",
        nargs="?",
        help="Path to results.json (default: latest run)",
    )
    args = parser.parse_args()

    if args.results:
        results_path = Path(args.results)
    else:
        output_dir = Path("experiments/output")
        runs = sorted(output_dir.iterdir(), key=lambda p: p.name)
        runs = [r for r in runs if (r / "results.json").exists()]
        if not runs:
            raise SystemExit("No experiment runs found in experiments/output/")
        results_path = runs[-1] / "results.json"

    print(f"Reading: {results_path}")
    report = analyze_crossing(results_path)

    out_path = results_path.parent / "crossing_analysis.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Written: {out_path}")

    _print_report(report)

    # Regenerate unified dashboard
    try:
        from experiments.dashboard import generate_html_dashboard
        print(results_path.parent)
        generate_html_dashboard(results_path.parent)
    except Exception as e:
        print(f"[warning] unified dashboard: {e}")


if __name__ == "__main__":
    main()
