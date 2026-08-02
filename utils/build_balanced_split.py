#!/usr/bin/env python3
"""Build a balanced, CVE-aware split_info.json complementing the existing one.

Constraints:
  - Each CVE contributes between 3 and 7 examples (functions).
  - Each CWE class has at least 20 examples before balancing.
  - All CWE classes are then capped to the size of the smallest class, so
    the final dataset is class-balanced (no CVE/CWE overweighting like the
    old CVE-2011-2918 with 16 examples in a single class).

Source pool: cvefixes_filtered_by_cwe.json (full CVEfixes method-pair export).
Existing split_info entries are prioritized when refilling each class so the
new split stays as close as possible to the original one.

Usage:
    uv run python utils/build_balanced_split.py \
        --existing-split cvefixes_experiments/output/method_vs_file_level/split_info_current_run.json \
        --pool cvefixes_filtered_by_cwe.json \
        --out cvefixes_experiments/output/method_vs_file_level/
    # writes <out>/split_info_balanced.json and <out>/split_info_stratified.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

MIN_PER_CVE = 3
MAX_PER_CVE = 12
MIN_PER_CWE = 20
SEED = 42
QUERY_RATIO = 0.2  # fraction of CVEs (per class) held out as query
CAP_MULTIPLIER = 2
CAP_SLACK = 0


def build_stratified_split(
    index_size: int, query_size: int, pool_cwe
) -> tuple[dict[str, int], dict[str, int]]:
    """Build a stratified split of the index and query sets."""
    total_size = index_size + query_size
    index_ratio = index_size / total_size
    query_ratio = query_size / total_size

    # Calculate the number of samples for each class in the stratified split
    index_class_sizes = {
        cwe: int(index_ratio * len(entries)) for cwe, entries in pool_cwe.items()
    }
    query_class_sizes = {
        cwe: int(query_ratio * len(entries)) for cwe, entries in pool_cwe.items()
    }

    # Adjust sizes to ensure they sum up to the desired total sizes
    index_adjustment = index_size - sum(index_class_sizes.values())
    query_adjustment = query_size - sum(query_class_sizes.values())

    # Distribute the adjustments across classes
    for cwe in sorted(pool_cwe.keys()):
        if index_adjustment > 0:
            index_class_sizes[cwe] += 1
            index_adjustment -= 1
        if query_adjustment > 0:
            query_class_sizes[cwe] += 1
            query_adjustment -= 1

    return index_class_sizes, query_class_sizes


def load_pool_by_cwe_cve(
    pool_path: Path, target_cwes: set[str]
) -> dict[str, dict[str, list[str]]]:
    """Return {cwe_id: {cve_id: [func_name, ...]}} restricted to target_cwes."""
    entries = json.loads(pool_path.read_text(encoding="utf-8"))["entries"]

    by_cwe_cve: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for e in entries:
        cwe_ids = {c["cwe_id"] for c in e.get("cwe", [])} & target_cwes
        func_name = e.get("method_name")
        if not func_name:
            continue
        for cwe in cwe_ids:
            by_cwe_cve[cwe][e["cve_id"]].add(func_name)

    return {
        cwe: {cve: sorted(funcs) for cve, funcs in cves.items()}
        for cwe, cves in by_cwe_cve.items()
    }


def build_class(
    cwe: str,
    pool_cves: dict[str, list[str]],
    existing_cves: set[str],
    rng: random.Random,
    target_size: int,
) -> list[tuple[str, str]]:
    """Fill one CWE class with whole CVEs (3-7 funcs each) until size >= target_size.

    Returns a list of (cve_id, func_name) pairs, in priority order (existing
    CVEs first, then the rest shuffled deterministically). Used both to probe
    the achievable raw size (target_size = MIN_PER_CWE) and to greedily pack a
    class down to an exact cap (see fill_to_cap below).
    """
    eligible = {
        cve: funcs for cve, funcs in pool_cves.items() if len(funcs) >= MIN_PER_CVE
    }

    existing_order = sorted(c for c in eligible if c in existing_cves)
    other = [c for c in eligible if c not in existing_cves]
    rng.shuffle(other)
    order = existing_order + other

    entries: list[tuple[str, str]] = []
    for cve in order:
        funcs = eligible[cve]
        take = funcs if len(funcs) <= MAX_PER_CVE else rng.sample(funcs, MAX_PER_CVE)
        entries.extend((cve, f) for f in sorted(take))
        if target_size is not None and len(entries) >= target_size:
            break

    return entries


def fill_to_cap(
    cwe: str,
    pool_cves: dict[str, list[str]],
    existing_cves: set[str],
    rng: random.Random,
    cap: int,
) -> list[tuple[str, str]]:
    """Greedily pack whole CVEs (3-7 funcs each) to hit `cap` exactly.

    Iterates candidate CVEs in priority order (existing-in-split first),
    skipping any CVE that would overshoot the remaining budget (rather than
    stopping), so later, smaller CVEs can close the gap exactly.
    """
    eligible = {
        cve: funcs for cve, funcs in pool_cves.items() if len(funcs) >= MIN_PER_CVE
    }

    existing_order = sorted(c for c in eligible if c in existing_cves)
    other = [c for c in eligible if c not in existing_cves]
    rng.shuffle(other)
    order = existing_order + other

    kept: list[tuple[str, str]] = []
    total = 0
    for cve in order:
        if total == cap:
            break
        funcs = eligible[cve]
        take = funcs if len(funcs) <= MAX_PER_CVE else rng.sample(funcs, MAX_PER_CVE)
        remaining = cap - total
        if len(take) <= remaining:
            kept.extend((cve, f) for f in sorted(take))
            total += len(take)
        elif remaining >= MIN_PER_CVE:
            kept.extend((cve, f) for f in sorted(take)[:remaining])
            total = cap
        # else: doesn't fit without violating MIN_PER_CVE; skip and try next CVE

    if total != cap:
        print(
            f"WARNING: {cwe}: could not pack exactly to cap={cap}, best effort got {total}"
        )

    return kept


def split_index_query(
    entries: list[tuple[str, str]], rng: random.Random
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """CVE-aware split: whole CVEs go to either index or query (no leakage)."""
    by_cve: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    for cve, func in entries:
        if cve not in by_cve:
            order.append(cve)
        by_cve[cve].append(func)

    cves = order[:]
    rng.shuffle(cves)
    n_query_cves = max(1, round(len(cves) * QUERY_RATIO))
    query_cves = set(cves[:n_query_cves])

    index, query = [], []
    for cve in order:
        bucket = query if cve in query_cves else index
        bucket.extend((cve, f) for f in by_cve[cve])
    return index, query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-split", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory; balanced/stratified split files are generated inside it",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    balanced_out = args.out / "split_info_balanced.json"
    stratified_out = args.out / "split_info_stratified.json"

    existing = json.loads(args.existing_split.read_text(encoding="utf-8"))
    existing_entries = existing["index"] + existing["query"]
    cwe_of_cve: dict[str, set[str]] = defaultdict(set)
    existing_by_cwe: dict[str, set[str]] = defaultdict(set)
    for e in existing_entries:
        existing_by_cwe[e["cwe_id"]].add(e["cve_id"])

    target_cwes = set(existing_by_cwe.keys())
    pool = load_pool_by_cwe_cve(args.pool, target_cwes)

    rng = random.Random(SEED)

    # Phase 1: probe how large each class can get with >= MIN_PER_CWE examples.
    raw_classes: dict[str, list[tuple[str, str]]] = {}
    for cwe in sorted(target_cwes):
        raw_classes[cwe] = build_class(
            cwe,
            pool.get(cwe, {}),
            existing_by_cwe[cwe],
            random.Random(SEED),
            target_size=None,
        )

    sizes = {cwe: len(v) for cwe, v in raw_classes.items()}
    min_size = min(sizes.values())
    cap = round(min_size * CAP_MULTIPLIER) + CAP_SLACK

    if min_size < MIN_PER_CWE:
        print(
            f"Warning: smallest classe has < {min_size} examples required: < {MIN_PER_CWE} "
        )

    # Phase 2: pack every class to exactly `min_size`, using a fresh RNG per
    # class (same seed) so the CVE ordering/priority is reproducible.
    balanced_classes = {
        cwe: (
            raw_classes[cwe]
            if sizes[cwe] <= cap
            else fill_to_cap(
                cwe, pool.get(cwe, {}), existing_by_cwe[cwe], random.Random(SEED), cap
            )
        )
        for cwe in sorted(target_cwes)
    }

    # Split each CVE's functions ~half/half between index and query.
    # This is a retrieval pipeline: we WANT the query's CVE to already be
    # present in the index (oracle-style), so a CVE is never wholly
    # confined to only index or only query. For a CVE with n functions,
    # floor(n/2) go to query and the rest to index (at least 1 each side
    # since MIN_PER_CVE=3).
    index_list: list[dict[str, Any]] = []
    query_list: list[dict[str, Any]] = []
    for cwe, entries in balanced_classes.items():
        by_cve: dict[str, list[str]] = defaultdict(list)
        cve_order: list[str] = []
        for cve, func in entries:
            if cve not in by_cve:
                cve_order.append(cve)
            by_cve[cve].append(func)

        for cve in cve_order:
            funcs = sorted(by_cve[cve])
            local_rng = random.Random(f"{SEED}:{cwe}:{cve}")
            local_rng.shuffle(funcs)
            # Keep only 1 function per CVE as query; the rest (up to
            # MAX_PER_CVE - 1) stay in the index as retrievable siblings —
            # maximizes index-side coverage per CVE instead of a ~50/50 split.
            n_query = 1
            n_query = min(n_query, len(funcs) - 1)  # keep at least 1 in index
            query_funcs = funcs[:n_query]
            index_funcs = funcs[n_query:]
            index_list.extend(
                {"cve_id": cve, "func_name": f, "variant": "original", "cwe_id": cwe}
                for f in index_funcs
            )
            query_list.extend(
                {"cve_id": cve, "func_name": f, "variant": "original", "cwe_id": cwe}
                for f in query_funcs
            )

    index_list.sort(key=lambda x: (x["cwe_id"], x["cve_id"], x["func_name"]))
    query_list.sort(key=lambda x: (x["cwe_id"], x["cve_id"], x["func_name"]))

    index_stratified_list, query_stratified_list = build_stratified_split(
        index_size=len(index_list), query_size=len(query_list), pool_cwe=pool
    )

    # Sanity checks
    all_entries = index_list + query_list
    cve_counts: dict[str, int] = defaultdict(int)
    for e in all_entries:
        cve_counts[e["cve_id"]] += 1

    cwe_counts: dict[str, int] = defaultdict(int)
    for e in all_entries:
        cwe_counts[e["cwe_id"]] += 1
    assert all(
        c <= cap for c in cwe_counts.values()
    ), f"CWE class exceeds cap={cap}: {cwe_counts}"

    index_keys = {(e["cve_id"], e["func_name"], e["variant"]) for e in index_list}
    query_keys = {(e["cve_id"], e["func_name"], e["variant"]) for e in query_list}
    assert not (index_keys & query_keys), "index/query overlap (same function in both)"
    index_cves = {e["cve_id"] for e in index_list}
    query_cves = {e["cve_id"] for e in query_list}
    assert query_cves <= index_cves, (
        "every query CVE must also have functions in the index "
        f"(retrieval oracle requirement); missing: {query_cves - index_cves}"
    )

    # balanced split_info.json output
    out = {
        "source_run_id": existing.get("source_run_id"),
        "derived_from": str(args.existing_split),
        "seed": SEED,
        "constraints": {
            "min_per_cve": MIN_PER_CVE,
            "max_per_cve": MAX_PER_CVE,
            "min_per_cwe_before_cap": MIN_PER_CWE,
            "smallest_class_size": min_size,
            "cap_multiplier": CAP_MULTIPLIER,
            "cap_slack": CAP_SLACK,
            "max_class_size_cap": cap,
        },
        "n_index": len(index_list),
        "n_query": len(query_list),
        "n_paired_instances": len(index_list) + len(query_list),
        "index": index_list,
        "query": query_list,
    }

    balanced_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # stratified split info
    out = {
        "source_run_id": existing.get("source_run_id"),
        "derived_from": str(args.existing_split),
        "seed": SEED,
        "constraints": {},
        "n_index": len(index_stratified_list),
        "n_query": len(query_stratified_list),
        "n_paired_instances": len(index_stratified_list) + len(query_stratified_list),
        "index": index_stratified_list,
        "query": query_stratified_list,
    }

    stratified_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Balanced class size (per CWE): {min_size}")
    print(f"Raw class sizes before cap: {sizes}")
    print(f"Final CWE counts: {dict(cwe_counts)}")
    print(
        f"n_index={len(index_list)} n_query={len(query_list)} total={len(index_list) + len(query_list)}"
    )
    print(
        f"CVE count range: min={min(cve_counts.values())} max={max(cve_counts.values())}"
    )
    print(f"Wrote {balanced_out}")
    print(f"Wrote {stratified_out}")


if __name__ == "__main__":
    main()
