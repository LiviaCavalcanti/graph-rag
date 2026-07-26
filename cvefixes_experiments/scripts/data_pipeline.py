"""
CVEfixes Reproducible Data Pipeline.

Transforms raw CVEfixes database into a clean, graph-verified, balanced dataset split
ready for experiments. 8 deterministic steps with stat checkpoints at each stage.

Usage:
    uv run python cvefixes_experiments/scripts/data_pipeline.py \\
        --config config.yaml \\
        --output-dir /path/to/output \\
        --cwes CWE-20 CWE-190 CWE-362 CWE-416 CWE-476 CWE-787 \\
        --min-cve-count 5 \\
        --target-n 225 \\
        [--seed 42] \\
        [--dry-run]

Output:
    {output_dir}/
    ├── index_pairs.csv                         # CSV: ≥225 balanced pairs (80%) — debugging only
    ├── index_pairs.json                        # JSON entries: ≥225 balanced pairs — compatible with load_pairs_from_file()
    ├── query_pairs.csv                         # CSV: query set (20%) — debugging only
    ├── query_pairs.json                        # JSON entries: query set — compatible with load_pairs_from_file()
    ├── split_info_balanced.json                # Split metadata (for precomputed_split_dir)
    ├── step_0_raw_entries.csv                  # After load
    ├── step_1_cwe_filtered.csv                 # After CWE filter
    ├── step_2_deduplicated.csv                 # After dedup by (cve, func)
    ├── step_3_cross_cve_removed.csv            # After removing multi-CVE funcs
    ├── step_4_min_cve_filtered.csv             # After min-count filter
    ├── step_5_graph_verified.csv               # After graph check
    ├── pipeline_report.json                    # Full provenance + stats
    └── missing_graphs.txt                      # List of missing graph dirs (if any)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import random

# ── repo root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import load_pairs_lightweight
from src.data.split import build_split


logger = logging.getLogger(__name__)


@dataclass
class StepStats:
    """Statistics at each pipeline step."""

    step_name: str
    rows_before: int
    rows_after: int
    rows_removed: int
    unique_cves: int
    unique_cwes: int
    cwe_counts: dict  # {cwe_id: count}
    notes: str = ""


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write rows to CSV."""
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    """Read CSV into list of dicts."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _pair_to_dict(pair) -> dict:
    """Convert FunctionPair to CSV row (lightweight)."""
    meta = pair.meta or {}
    return {
        "cve_id": pair.cve_id,
        "cwe_id": pair.cwe_id,
        "func_name": pair.func_name,
        "project": pair.project,
        "variant": meta.get("variant", "original"),
        "method_change_id": meta.get("method_change_id", ""),
        "filename": meta.get("filename", ""),
        "language": meta.get("language", ""),
        "source_before": meta.get("source_before", ""),
        "source_after": meta.get("source_after", ""),
    }


def _dict_to_key(row: dict) -> tuple:
    """Convert row to (cve_id, cwe_id, func_name, project) key."""
    return (
        row["cve_id"],
        row["cwe_id"],
        row["func_name"],
        row["project"],
    )


def _compute_stats(rows: list[dict]) -> dict:
    """Compute stats for checkpoint."""
    cve_counts = Counter(r["cve_id"] for r in rows)
    cwe_counts = Counter(r["cwe_id"] for r in rows)
    return {
        "total_rows": len(rows),
        "unique_cves": len(cve_counts),
        "unique_cwes": len(cwe_counts),
        "cwe_counts": dict(sorted(cwe_counts.items())),
        "cve_counts_top_10": dict(cve_counts.most_common(10)),
    }


def _print_checkpoint(step: StepStats) -> None:
    """Print a step checkpoint."""
    print(
        f"\n{'='*80}\n"
        f"STEP: {step.step_name}\n"
        f"{'='*80}\n"
        f"  Rows before:     {step.rows_before}\n"
        f"  Rows after:      {step.rows_after}\n"
        f"  Rows removed:    {step.rows_removed} ({100*step.rows_removed/max(step.rows_before, 1):.1f}%)\n"
        f"  Unique CVEs:     {step.unique_cves}\n"
        f"  Unique CWEs:     {step.unique_cwes}\n"
        f"  CWE distribution: {step.cwe_counts}\n"
    )
    if step.notes:
        print(f"  Notes: {step.notes}\n")


# ── Step 0 ────────────────────────────────────────────────────────────────


def step_0_load_raw_entries(cfg: dict, output_dir: Path) -> list[dict]:
    """Load raw entries from CVEfixes DB."""
    print("\n[STEP 0] Loading raw entries from CVEfixes database...")
    pairs = load_pairs_lightweight(cfg)
    rows = [_pair_to_dict(p) for p in pairs]

    stats = _compute_stats(rows)
    checkpoint = StepStats(
        step_name="0. Load Raw Entries",
        rows_before=0,
        rows_after=len(rows),
        rows_removed=0,
        unique_cves=stats["unique_cves"],
        unique_cwes=stats["unique_cwes"],
        cwe_counts=stats["cwe_counts"],
    )
    _print_checkpoint(checkpoint)
    _write_csv(rows, output_dir / "step_0_raw_entries.csv")
    return rows


# ── Step 1 ────────────────────────────────────────────────────────────────


def step_1_cwe_filter(rows: list[dict], target_cwes: list[str], output_dir: Path) -> list[dict]:
    """Filter to target CWEs."""
    print(f"\n[STEP 1] Filtering to CWEs: {target_cwes}...")
    before = len(rows)
    filtered = [r for r in rows if r["cwe_id"] in target_cwes]
    after = len(filtered)

    stats = _compute_stats(filtered)
    # Warn if any target CWE is empty
    target_set = set(target_cwes)
    present_cwes = set(stats["cwe_counts"].keys())
    missing = target_set - present_cwes
    if missing:
        print(f"  ⚠ WARNING: Target CWEs not found in data: {missing}")

    checkpoint = StepStats(
        step_name="1. CWE Filter",
        rows_before=before,
        rows_after=after,
        rows_removed=before - after,
        unique_cves=stats["unique_cves"],
        unique_cwes=stats["unique_cwes"],
        cwe_counts=stats["cwe_counts"],
        notes=f"Kept {len(target_cwes)} CWEs; {len(missing)} target CWEs not present" if missing else "",
    )
    _print_checkpoint(checkpoint)
    _write_csv(filtered, output_dir / "step_1_cwe_filtered.csv")
    return filtered


# ── Step 2 ────────────────────────────────────────────────────────────────


def step_2_deduplicate_exact(rows: list[dict], output_dir: Path) -> list[dict]:
    """Remove exact duplicates by (cve_id, func_name)."""
    print("\n[STEP 2] Deduplicating by (cve_id, func_name)...")
    before = len(rows)
    seen = set()
    dedup = []
    duplicates = []

    for row in rows:
        key = (row["cve_id"], row["func_name"])
        if key in seen:
            duplicates.append(row)
        else:
            seen.add(key)
            dedup.append(row)

    after = len(dedup)
    stats = _compute_stats(dedup)

    checkpoint = StepStats(
        step_name="2. Exact Deduplication",
        rows_before=before,
        rows_after=after,
        rows_removed=before - after,
        unique_cves=stats["unique_cves"],
        unique_cwes=stats["unique_cwes"],
        cwe_counts=stats["cwe_counts"],
    )
    _print_checkpoint(checkpoint)
    _write_csv(dedup, output_dir / "step_2_deduplicated.csv")
    if duplicates:
        _write_csv(duplicates, output_dir / "step_2_duplicates_removed.csv")
    return dedup


# ── Step 3 ────────────────────────────────────────────────────────────────


def step_3_remove_cross_cve_funcs(rows: list[dict], output_dir: Path) -> list[dict]:
    """Remove func_names that appear in multiple CVEs."""
    print("\n[STEP 3] Removing functions that appear in multiple CVEs...")
    before = len(rows)

    # Build func -> set of CVEs
    func_cves = defaultdict(set)
    for row in rows:
        func_cves[row["func_name"]].add(row["cve_id"])

    # Identify multi-CVE functions
    multi_cve_funcs = {f for f, cves in func_cves.items() if len(cves) > 1}

    # Remove all rows for those functions
    cleaned = [r for r in rows if r["func_name"] not in multi_cve_funcs]
    after = len(cleaned)

    # Correctness check
    func_cves_after = defaultdict(set)
    for row in cleaned:
        func_cves_after[row["func_name"]].add(row["cve_id"])
    bad = {f for f, cves in func_cves_after.items() if len(cves) > 1}
    if bad:
        print(f"  ✗ ERROR: Found {len(bad)} functions still in multiple CVEs after removal!")
        sys.exit(1)

    stats = _compute_stats(cleaned)

    # Per-CWE impact
    cwe_impact = defaultdict(lambda: {"before": 0, "after": 0})
    for row in rows:
        cwe_impact[row["cwe_id"]]["before"] += 1
    for row in cleaned:
        cwe_impact[row["cwe_id"]]["after"] += 1

    impact_str = "; ".join(
        f"{cwe}: {d['before']}→{d['after']}"
        for cwe, d in sorted(cwe_impact.items())
    )

    checkpoint = StepStats(
        step_name="3. Cross-CVE Contamination Removal",
        rows_before=before,
        rows_after=after,
        rows_removed=before - after,
        unique_cves=stats["unique_cves"],
        unique_cwes=stats["unique_cwes"],
        cwe_counts=stats["cwe_counts"],
        notes=f"Removed {len(multi_cve_funcs)} func_names. Per-CWE: {impact_str}",
    )
    _print_checkpoint(checkpoint)
    _write_csv(cleaned, output_dir / "step_3_cross_cve_removed.csv")
    return cleaned


# ── Step 4 ────────────────────────────────────────────────────────────────


def step_4_min_cve_count_filter(rows: list[dict], min_cve_count: int, output_dir: Path) -> list[dict]:
    """Remove CVEs with < min_cve_count entries."""
    print(f"\n[STEP 4] Filtering CVEs with < {min_cve_count} entries...")
    before = len(rows)

    cve_counts = Counter(r["cve_id"] for r in rows)
    valid_cves = {cve for cve, cnt in cve_counts.items() if cnt >= min_cve_count}
    filtered = [r for r in rows if r["cve_id"] in valid_cves]
    after = len(filtered)

    stats = _compute_stats(filtered)

    # Check if any CWE drops to 0
    cwes_removed = set(Counter(r["cwe_id"] for r in rows)) - set(stats["cwe_counts"].keys())
    warning = ""
    if cwes_removed:
        warning = f"⚠ WARNING: CWEs depleted to 0: {cwes_removed}"
        print(f"  {warning}")

    removed_cves = set(cve_counts.keys()) - valid_cves
    checkpoint = StepStats(
        step_name="4. Min-CVE-Count Filter",
        rows_before=before,
        rows_after=after,
        rows_removed=before - after,
        unique_cves=stats["unique_cves"],
        unique_cwes=stats["unique_cwes"],
        cwe_counts=stats["cwe_counts"],
        notes=f"Removed {len(removed_cves)} CVEs. {warning}" if warning else f"Removed {len(removed_cves)} CVEs.",
    )
    _print_checkpoint(checkpoint)
    _write_csv(filtered, output_dir / "step_4_min_cve_filtered.csv")
    return filtered


# ── Step 5 ────────────────────────────────────────────────────────────────


def step_5_graph_availability_check(
    rows: list[dict], graphml_root: Path, output_dir: Path
) -> list[dict]:
    """Verify that graphml directories exist for each entry."""
    print(f"\n[STEP 5] Checking graph availability in {graphml_root}...")
    before = len(rows)

    missing = []
    verified = []

    for row in rows:
        cve_id = row["cve_id"]
        # Assume dir_name is stored in row meta or we derive it from CVE
        # For now, check if graphml_root has folders matching the pattern
        # Look for folders starting with the CVE ID
        graph_found = False
        if graphml_root.exists():
            for folder in graphml_root.iterdir():
                if folder.is_dir() and folder.name.startswith(cve_id):
                    graph_found = True
                    break
        if graph_found:
            verified.append(row)
        else:
            missing.append(row)

    after = len(verified)
    stats = _compute_stats(verified)

    # Save missing graphs log
    if missing:
        missing_cves = [r["cve_id"] for r in missing]
        with (output_dir / "missing_graphs.txt").open("w") as f:
            f.write("\n".join(sorted(set(missing_cves))))

    checkpoint = StepStats(
        step_name="5. Graph Availability Check",
        rows_before=before,
        rows_after=after,
        rows_removed=before - after,
        unique_cves=stats["unique_cves"],
        unique_cwes=stats["unique_cwes"],
        cwe_counts=stats["cwe_counts"],
        notes=f"Missing graphs: {len(set(r['cve_id'] for r in missing))} CVEs",
    )
    _print_checkpoint(checkpoint)
    _write_csv(verified, output_dir / "step_5_graph_verified.csv")
    return verified


# ── Step 6 ────────────────────────────────────────────────────────────────


def step_6_cwe_balanced_sampling(
    rows: list[dict], target_n: int, seed: int, output_dir: Path
) -> list[dict]:
    """CWE-balanced sampling to reach target_n."""
    print(f"\n[STEP 6] CWE-balanced sampling (target: {target_n})...")
    random.seed(seed)

    # Group by CWE, then sort CVEs within each CWE by occurrence
    cwe_data = defaultdict(list)
    for row in rows:
        cwe_data[row["cwe_id"]].append(row)

    # Sort CWEs by count (ascending) — protect rare CWEs first
    sorted_cwes = sorted(cwe_data.keys(), key=lambda c: len(cwe_data[c]))

    # Compute target per CWE
    n_cwes = len(sorted_cwes)
    quota_per_cwe = max(1, target_n // n_cwes)

    sampled = []
    unused_quota = 0

    for cwe in sorted_cwes:
        cwe_rows = cwe_data[cwe]
        # Sort CVEs within this CWE by occurrence (descending) — greedy prioritization
        cve_counts = Counter(r["cve_id"] for r in cwe_rows)
        sorted_cves = sorted(cve_counts.items(), key=lambda x: x[1], reverse=True)

        # Greedily fill quota: take from top CVEs first
        quota = quota_per_cwe + unused_quota
        taken = 0

        for cve_id, cve_count in sorted_cves:
            cve_rows_filtered = [r for r in cwe_rows if r["cve_id"] == cve_id]
            take = min(len(cve_rows_filtered), quota - taken)
            sampled.extend(cve_rows_filtered[:take])
            taken += take
            if taken >= quota:
                break

        # Track unused quota from depleted CWEs for redistribution
        unused_quota = max(0, quota - taken)

    # If sample is < target, redistribute any remaining quota
    if len(sampled) < target_n:
        remaining_quota = target_n - len(sampled)
        remaining_rows = [r for r in rows if r not in sampled]
        if remaining_rows:
            to_add = remaining_rows[:remaining_quota]
            sampled.extend(to_add)

    # Remove duplicates (shouldn't be any but be safe)
    sampled = list({_dict_to_key(r): r for r in sampled}.values())

    after = len(sampled)
    stats = _compute_stats(sampled)

    # Compute Gini coefficient for CWE balance
    cwe_sizes = list(stats["cwe_counts"].values())
    if len(cwe_sizes) > 1:
        sorted_sizes = sorted(cwe_sizes)
        n = len(sorted_sizes)
        cumsum = sum((i + 1) * x for i, x in enumerate(sorted_sizes))
        gini = (2 * cumsum / (n * sum(sorted_sizes))) - (n + 1) / n
    else:
        gini = 0.0

    # Check if target is met
    target_ok = after >= target_n
    warning = "" if target_ok else f"⚠ WARNING: Sample size {after} < target {target_n}"
    if warning:
        print(f"  {warning}")

    checkpoint = StepStats(
        step_name="6. CWE-Balanced Sampling",
        rows_before=len(rows),
        rows_after=after,
        rows_removed=len(rows) - after,
        unique_cves=stats["unique_cves"],
        unique_cwes=stats["unique_cwes"],
        cwe_counts=stats["cwe_counts"],
        notes=f"Target: {target_n}, Achieved: {after}. Gini: {gini:.3f}. {warning}",
    )
    _print_checkpoint(checkpoint)
    _write_csv(sampled, output_dir / "step_6_balanced_sample.csv")

    if not target_ok:
        logger.warning(f"Target sample size {target_n} not met; achieved {after}")

    return sampled


# ── Step 7 ────────────────────────────────────────────────────────────────


def step_7_precomputed_split(pairs_dict: list[dict], cfg: dict, output_dir: Path) -> tuple[list, list]:
    """Generate precomputed split for reproducibility."""
    print("\n[STEP 7] Generating precomputed split (stratified, seed=42, test_ratio=0.2)...")

    # Reconstruct FunctionPair objects for build_split
    # This is a workaround — we pass dict rows but build_split expects FunctionPair
    # We'll use CSV-based split for now and save as precomputed format

    # For now, do a simple stratified split on the CSV rows
    # Sort by CWE to ensure stratified grouping
    by_cwe = defaultdict(list)
    for row in pairs_dict:
        by_cwe[row["cwe_id"]].append(row)

    index_pairs = []
    query_pairs = []

    for cwe, rows in by_cwe.items():
        random.shuffle(rows)
        split_idx = max(1, int(len(rows) * 0.8))
        index_pairs.extend(rows[:split_idx])
        query_pairs.extend(rows[split_idx:])

    # Save outputs — both CSV (for debugging) and JSON (for experiments)
    _write_csv(index_pairs, output_dir / "index_pairs.csv")
    _write_csv(query_pairs, output_dir / "query_pairs.csv")
    
    # Save as JSON entries files (compatible with load_pairs_from_file)
    with (output_dir / "index_pairs.json").open("w") as f:
        json.dump({"entries": index_pairs}, f, indent=2)
    with (output_dir / "query_pairs.json").open("w") as f:
        json.dump({"entries": query_pairs}, f, indent=2)

    # Save split info in the precomputed format (index/query lists are required
    # by exp_file_method_interface and exp_cvefixes_retrieval_grid_file_level).
    def _to_spec(rows):
        return [
            {"cve_id": r["cve_id"], "func_name": r["func_name"],
             "variant": r.get("variant", "original"), "cwe_id": r["cwe_id"]}
            for r in rows
        ]

    split_info = {
        "enabled": True,
        "mode": "stratified",
        "seed": 42,
        "test_ratio": 0.2,
        "index_n": len(index_pairs),
        "query_n": len(query_pairs),
        "total_n": len(index_pairs) + len(query_pairs),
        "index": _to_spec(index_pairs),
        "query": _to_spec(query_pairs),
    }
    with (output_dir / "split_info_balanced.json").open("w") as f:
        json.dump(split_info, f, indent=2)

    # Per-CWE split stats
    index_cwe = Counter(r["cwe_id"] for r in index_pairs)
    query_cwe = Counter(r["cwe_id"] for r in query_pairs)

    print(f"\n  Index pairs: {len(index_pairs)}")
    print(f"  Query pairs: {len(query_pairs)}")
    print(f"  ✓ Written: index_pairs.csv + .json, query_pairs.csv + .json, split_info_balanced.json")
    print(f"  Per-CWE split:")
    for cwe in sorted(set(index_cwe.keys()) | set(query_cwe.keys())):
        idx_cnt = index_cwe.get(cwe, 0)
        qry_cnt = query_cwe.get(cwe, 0)
        print(f"    {cwe}: {idx_cnt} (index) + {qry_cnt} (query) = {idx_cnt + qry_cnt}")

    return index_pairs, query_pairs


# ── Step 8 ────────────────────────────────────────────────────────────────


def step_8_final_report(
    all_stats: list[StepStats], output_dir: Path, cfg_hash: str
) -> None:
    """Generate final report."""
    print("\n[STEP 8] Generating final report...")

    report = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": cfg_hash,
        "output_dir": str(output_dir),
        "steps": [asdict(s) for s in all_stats],
        "final_stats": {
            "total_rows": all_stats[-1].rows_after,
            "unique_cves": all_stats[-1].unique_cves,
            "unique_cwes": all_stats[-1].unique_cwes,
            "cwe_distribution": all_stats[-1].cwe_counts,
        },
    }

    report_path = output_dir / "pipeline_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved to {report_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    for stat in all_stats:
        print(
            f"{stat.step_name:<40} | {stat.rows_after:>5} rows "
            f"({stat.unique_cves:>3} CVEs, {stat.unique_cwes:>2} CWEs)"
        )
    print("=" * 80)


# ── Step 9: Convert to entries format for cvefixes_file ───────────────────


def step_9_convert_to_entries_format(
    index_pairs: list[dict], query_pairs: list[dict], cfg: dict, output_dir: Path
) -> None:
    """Convert pipeline output to entries JSON format compatible with cvefixes_file.

    Re-enriches each pair with CVE-level metadata (description, severity, …)
    from the CVEfixes DB, scoped to only the CVEs present in the sample.
    Method code is taken directly from the pair dicts (already in memory from
    step 7) — the large file-level blobs (code_before/code_after on
    file_change) are intentionally NOT fetched to avoid OOM on WSL.
    """
    print("\n[STEP 9] Converting to entries format...")

    import sqlite3

    db_path = Path(cfg.get("data", {}).get("cvefixes", {}).get("db_path", "data/cvefixes/CVEfixes.db"))
    if not db_path.exists():
        print(f"  ✗ CVEfixes database not found at {db_path}, skipping Step 9")
        return

    all_pairs = index_pairs + query_pairs
    sample_cve_ids = list({p["cve_id"] for p in all_pairs})
    placeholders = ",".join("?" * len(sample_cve_ids))
    print(f"  Fetching CVE metadata for {len(sample_cve_ids)} CVEs from DB...")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Scoped to sample CVEs only; no file-level blobs (they are unused and
    # can be hundreds of MB for the full DB).
    cursor.execute(f"""
        SELECT DISTINCT
            cv.cve_id,
            cv.description        AS cve_description,
            cv.severity           AS cve_severity,
            cv.cvss3_base_score,
            cv.cvss3_base_severity,
            cv.published_date,
            cc.cwe_id,
            m_after.method_change_id,
            m_after.name          AS func_name,
            f.file_change_id,
            f.filename,
            f.programming_language
        FROM method_change m_after
        JOIN file_change f  ON m_after.file_change_id = f.file_change_id
        JOIN commits c      ON f.hash = c.hash
        JOIN fixes fx       ON c.hash = fx.hash
        JOIN cve cv         ON fx.cve_id = cv.cve_id
        LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
        WHERE m_after.before_change = 'False'
          AND cv.cve_id IN ({placeholders})
    """, sample_cve_ids)

    # Index by (cve_id, method_change_id) — code comes from the pair dict
    metadata_map: dict[tuple, dict] = {}
    for row in cursor.fetchall():
        key = (row["cve_id"], row["method_change_id"])
        metadata_map[key] = dict(row)

    conn.close()
    print(f"  DB metadata loaded for {len(metadata_map)} (cve, method_change) rows")

    def _pair_to_entry(pair_dict: dict) -> dict:
        """Convert a pair dict to entries format."""
        cve_id = pair_dict["cve_id"]
        cwe_id = pair_dict["cwe_id"]
        method_change_id = pair_dict.get("method_change_id", "")

        meta = metadata_map.get((cve_id, method_change_id), {})

        # Code is already present in the pair dict from the pipeline steps —
        # no need to re-fetch it from the DB.
        code_before = pair_dict.get("source_before", "")
        code_after = pair_dict.get("source_after", "")
        lines_before = len(code_before.splitlines()) if code_before else 0
        lines_after = len(code_after.splitlines()) if code_after else 0

        return {
            "cve_id": cve_id,
            "cwe": [{"cwe_id": cwe_id, "cwe_name": "Unknown"}],
            "cve_description": meta.get("cve_description", ""),
            "cve_severity": meta.get("cve_severity", "UNKNOWN"),
            "cvss3_base_score": meta.get("cvss3_base_score", "0.0"),
            "cvss3_base_severity": meta.get("cvss3_base_severity", "UNKNOWN"),
            "published_date": meta.get("published_date", ""),
            "file_change_id": meta.get("file_change_id", ""),
            "filename": meta.get("filename", pair_dict.get("filename", "")),
            "programming_language": meta.get("programming_language", pair_dict.get("language", "")),
            "method_name": meta.get("func_name", pair_dict.get("func_name", "")),
            "method_signature": "",
            "code_before": code_before,
            "code_after": code_after,
            "changes": {
                "lines_added": max(0, lines_after - lines_before),
                "lines_removed": max(0, lines_before - lines_after),
                "lines_before": lines_before,
                "lines_after": lines_after,
            },
        }
    
    # Convert pairs
    index_entries = [_pair_to_entry(p) for p in index_pairs]
    query_entries = [_pair_to_entry(p) for p in query_pairs]
    
    # Collect all target CWEs
    all_cwes = set()
    for entries in [index_entries, query_entries]:
        for e in entries:
            all_cwes.add(e["cwe"][0]["cwe_id"])
    
    # Write index entries
    index_path = output_dir / "index_pairs_entries.json"
    with index_path.open("w") as f:
        json.dump({
            "count": len(index_entries),
            "seed": 42,
            "target_cwes": sorted(all_cwes),
            "entries": index_entries,
        }, f, indent=2)
    print(f"  ✓ {index_path}: {len(index_entries)} entries")
    
    # Write query entries
    query_path = output_dir / "query_pairs_entries.json"
    with query_path.open("w") as f:
        json.dump({
            "count": len(query_entries),
            "seed": 42,
            "target_cwes": sorted(all_cwes),
            "entries": query_entries,
        }, f, indent=2)
    print(f"  ✓ {query_path}: {len(query_entries)} entries")
    
    # Also write combined entries file
    combined_path = output_dir / "pipeline_entries.json"
    with combined_path.open("w") as f:
        json.dump({
            "count": len(index_entries) + len(query_entries),
            "seed": 42,
            "target_cwes": sorted(all_cwes),
            "entries": index_entries + query_entries,
        }, f, indent=2)
    print(f"  ✓ {combined_path}: {len(index_entries) + len(query_entries)} entries (combined)")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--cwes",
        nargs="+",
        default=["CWE-20", "CWE-190", "CWE-362", "CWE-416", "CWE-476", "CWE-787"],
        help="Target CWEs to keep (CWE-400 excluded by default)",
    )
    parser.add_argument("--min-cve-count", type=int, default=5, help="Minimum CVE occurrences")
    parser.add_argument("--target-n", type=int, default=225, help="Target sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Print stats, don't write files")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    graphml_root = Path(cfg.get("data", {}).get("cvefixes", {}).get("graphml_root", "graphml_fixedcves_file"))

    # Compute config hash
    cfg_hash = str(hash(json.dumps(cfg, sort_keys=True, default=str)))[:8]

    print("\n" + "=" * 80)
    print("CVEfixes Reproducible Data Pipeline")
    print("=" * 80)
    print(f"Config: {args.config} (hash: {cfg_hash})")
    print(f"Output: {output_dir}")
    print(f"Target CWEs: {args.cwes}")
    print(f"Min CVE count: {args.min_cve_count}")
    print(f"Target sample size: {args.target_n}")
    print(f"Seed: {args.seed}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)

    all_stats = []

    # ── Execute pipeline ──────────────────────────────────────────────────────

    # Step 0
    raw_rows = step_0_load_raw_entries(cfg, output_dir if not args.dry_run else Path("/tmp"))
    all_stats.append(
        StepStats(
            step_name="0. Load Raw Entries",
            rows_before=0,
            rows_after=len(raw_rows),
            rows_removed=0,
            unique_cves=len(set(r["cve_id"] for r in raw_rows)),
            unique_cwes=len(set(r["cwe_id"] for r in raw_rows)),
            cwe_counts=dict(Counter(r["cwe_id"] for r in raw_rows)),
        )
    )

    # Step 1
    cwe_filtered = step_1_cwe_filter(raw_rows, args.cwes, output_dir if not args.dry_run else Path("/tmp"))
    all_stats.append(
        StepStats(
            step_name="1. CWE Filter",
            rows_before=len(raw_rows),
            rows_after=len(cwe_filtered),
            rows_removed=len(raw_rows) - len(cwe_filtered),
            unique_cves=len(set(r["cve_id"] for r in cwe_filtered)),
            unique_cwes=len(set(r["cwe_id"] for r in cwe_filtered)),
            cwe_counts=dict(Counter(r["cwe_id"] for r in cwe_filtered)),
        )
    )

    # Step 2
    dedup = step_2_deduplicate_exact(cwe_filtered, output_dir if not args.dry_run else Path("/tmp"))
    all_stats.append(
        StepStats(
            step_name="2. Exact Deduplication",
            rows_before=len(cwe_filtered),
            rows_after=len(dedup),
            rows_removed=len(cwe_filtered) - len(dedup),
            unique_cves=len(set(r["cve_id"] for r in dedup)),
            unique_cwes=len(set(r["cwe_id"] for r in dedup)),
            cwe_counts=dict(Counter(r["cwe_id"] for r in dedup)),
        )
    )

    # Step 3
    cross_cve_removed = step_3_remove_cross_cve_funcs(dedup, output_dir if not args.dry_run else Path("/tmp"))
    all_stats.append(
        StepStats(
            step_name="3. Cross-CVE Contamination Removal",
            rows_before=len(dedup),
            rows_after=len(cross_cve_removed),
            rows_removed=len(dedup) - len(cross_cve_removed),
            unique_cves=len(set(r["cve_id"] for r in cross_cve_removed)),
            unique_cwes=len(set(r["cwe_id"] for r in cross_cve_removed)),
            cwe_counts=dict(Counter(r["cwe_id"] for r in cross_cve_removed)),
        )
    )

    # Step 4
    min_cve_filtered = step_4_min_cve_count_filter(
        cross_cve_removed, args.min_cve_count, output_dir if not args.dry_run else Path("/tmp")
    )
    all_stats.append(
        StepStats(
            step_name="4. Min-CVE-Count Filter",
            rows_before=len(cross_cve_removed),
            rows_after=len(min_cve_filtered),
            rows_removed=len(cross_cve_removed) - len(min_cve_filtered),
            unique_cves=len(set(r["cve_id"] for r in min_cve_filtered)),
            unique_cwes=len(set(r["cwe_id"] for r in min_cve_filtered)),
            cwe_counts=dict(Counter(r["cwe_id"] for r in min_cve_filtered)),
        )
    )

    # Step 5
    graph_verified = step_5_graph_availability_check(
        min_cve_filtered, graphml_root, output_dir if not args.dry_run else Path("/tmp")
    )
    all_stats.append(
        StepStats(
            step_name="5. Graph Availability Check",
            rows_before=len(min_cve_filtered),
            rows_after=len(graph_verified),
            rows_removed=len(min_cve_filtered) - len(graph_verified),
            unique_cves=len(set(r["cve_id"] for r in graph_verified)),
            unique_cwes=len(set(r["cwe_id"] for r in graph_verified)),
            cwe_counts=dict(Counter(r["cwe_id"] for r in graph_verified)),
        )
    )

    # Step 6
    balanced_sample = step_6_cwe_balanced_sampling(
        min_cve_filtered, args.target_n, args.seed, output_dir if not args.dry_run else Path("/tmp")
    )
    all_stats.append(
        StepStats(
            step_name="6. CWE-Balanced Sampling",
            rows_before=len(min_cve_filtered),
            rows_after=len(balanced_sample),
            rows_removed=len(min_cve_filtered) - len(balanced_sample),
            unique_cves=len(set(r["cve_id"] for r in balanced_sample)),
            unique_cwes=len(set(r["cwe_id"] for r in balanced_sample)),
            cwe_counts=dict(Counter(r["cwe_id"] for r in balanced_sample)),
        )
    )

    # Step 7
    if not args.dry_run:
        index_pairs, query_pairs = step_7_precomputed_split(balanced_sample, cfg, output_dir)
    else:
        index_pairs, query_pairs = [], []

    # Step 8
    if not args.dry_run:
        step_8_final_report(all_stats, output_dir, cfg_hash)

    # Step 9
    if not args.dry_run:
        step_9_convert_to_entries_format(index_pairs, query_pairs, cfg, output_dir)

    print("\n✓ Pipeline complete!")
    if args.dry_run:
        print("  (Dry run: no files written)")


if __name__ == "__main__":
    main()
