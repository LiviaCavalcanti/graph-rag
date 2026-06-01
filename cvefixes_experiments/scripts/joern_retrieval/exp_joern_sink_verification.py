"""
Experiment: Joern Sink Query Verification

Uses the vulnerability-pattern keywords from vuln_pattern.py to build Joern
queries that locate "sinks" (vulnerable code patterns) in CVEfixes methods.
Then verifies whether the Joern-returned nodes match the actual changed lines
(the vulnerability fix) shown in the dataset.

Steps:
  1. Create a stratified-by-CWE random subset of 100 entries from cvefixes_filtered_by_cwe.json
  2. For each entry, write the code_before to a temp file, run Joern to build a CPG
  3. Walk the CPG graph to find nodes matching sink keywords (per CWE category)
  4. Compute the diff (changed lines) from code_before → code_after
  5. Check if any Joern-identified sink node's code overlaps with the changed lines
  6. Report precision/recall of the sink identification approach
"""

import json
import tempfile
from collections import defaultdict
from pathlib import Path

from src.data.pipeline import load_cpg_dir, run_joern_export, write_c_file

from cvefixes_experiments.scripts.common import (
    JOERN_BIN_DIR,
    DATA_FILE,
    SUBSET_FILE,
    RESULTS_FILE,
    check_overlap,
    create_stratified_subset,
    find_sink_nodes,
    get_changed_lines,
)

# ── Configuration ──
SEED = 42
SUBSET_SIZE = 100


# ── Main pipeline ──


def run_experiment():
    """Run the full Joern sink verification experiment."""
    print("=" * 70)
    print("EXPERIMENT: Joern Sink Query Verification")
    print("=" * 70)

    # Step 1: Create or load subset
    if Path(SUBSET_FILE).exists():
        print(f"\n[1/4] Loading existing subset from {SUBSET_FILE}")
        with open(SUBSET_FILE) as f:
            subset_data = json.load(f)
        sample = subset_data["entries"]
        print(f"  Loaded {len(sample)} entries")
    else:
        print(f"\n[1/4] Creating stratified subset of {SUBSET_SIZE} entries...")
        sample = create_stratified_subset(DATA_FILE, SUBSET_FILE, SUBSET_SIZE, SEED)

    # Step 2-4: Process each entry
    print(f"\n[2/4] Processing {len(sample)} entries with Joern...")
    results = []
    successes = 0
    failures = 0
    hits = 0
    misses = 0

    for i, entry in enumerate(sample):
        cve_id = entry["cve_id"]
        cwe_id = entry["cwe"][0]["cwe_id"]
        method_name = entry.get("method_name", "unknown")
        code_before = entry["code_before"]
        code_after = entry["code_after"]

        print(f"\n  [{i+1}/{len(sample)}] {cve_id} ({cwe_id}) — {method_name}")

        # Compute changed lines
        removed, added = get_changed_lines(code_before, code_after)
        if not removed and not added:
            print("    ⚠ No diff detected, skipping")
            results.append(
                {
                    "cve_id": cve_id,
                    "cwe_id": cwe_id,
                    "method_name": method_name,
                    "status": "no_diff",
                }
            )
            continue

        # Write code to temp file and run Joern
        with tempfile.TemporaryDirectory(prefix=f"sink_{cve_id}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_path = write_c_file(code_before, tmpdir_path / "vuln.c")
            cpg_dir = str(tmpdir_path / "cpg")
            graph_dir = str(tmpdir_path / "graph")

            ok = run_joern_export(JOERN_BIN_DIR, str(src_path), cpg_dir, graph_dir)
            if not ok:
                print("    ✗ Joern export failed")
                failures += 1
                results.append(
                    {
                        "cve_id": cve_id,
                        "cwe_id": cwe_id,
                        "method_name": method_name,
                        "status": "joern_failed",
                    }
                )
                continue

            try:
                G = load_cpg_dir(graph_dir)
            except Exception as e:
                print(f"    ✗ Graph load failed: {e}")
                failures += 1
                results.append(
                    {
                        "cve_id": cve_id,
                        "cwe_id": cwe_id,
                        "method_name": method_name,
                        "status": "graph_load_failed",
                        "error": str(e),
                    }
                )
                continue

            successes += 1
            print(f"    ✓ CPG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            # Find sink nodes
            sink_nodes = find_sink_nodes(G, cwe_id)
            print(f"    Sink nodes found: {len(sink_nodes)}")

            # Check overlap with changed lines
            hit_nodes, total_sinks, total_changed = check_overlap(
                sink_nodes, removed, added,
                cwe_id=cwe_id,
                code_before=code_before,
                code_after=code_after,
            )

            if hit_nodes:
                hits += 1
                print(f"    ✓ HIT — {len(hit_nodes)}/{total_sinks} sinks match changed lines")
            else:
                misses += 1
                print(f"    ✗ MISS — 0/{total_sinks} sinks match changed lines")

            results.append(
                {
                    "cve_id": cve_id,
                    "cwe_id": cwe_id,
                    "method_name": method_name,
                    "status": "success",
                    "graph_nodes": G.number_of_nodes(),
                    "graph_edges": G.number_of_edges(),
                    "total_sinks": total_sinks,
                    "total_changed_lines": total_changed,
                    "hit_count": len(hit_nodes),
                    "hit": bool(hit_nodes),
                    "removed_lines": list(removed)[:10],  # sample for inspection
                    "added_lines": list(added)[:10],
                    "sample_hits": [
                        {"code": h["code"], "pattern": h["matched_pattern"]}
                        for h in hit_nodes[:5]
                    ],
                }
            )

    # Step 5: Report results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    total_processed = successes
    print(f"  Total entries:          {len(sample)}")
    print(f"  Successfully processed: {successes}")
    print(f"  Joern failures:         {failures}")
    print(f"  Hits (sink ∩ changed):  {hits}")
    print(f"  Misses:                 {misses}")
    if total_processed > 0:
        hit_rate = hits / total_processed * 100
        print(f"  Hit rate:               {hit_rate:.1f}%")

    # Breakdown by CWE
    print("\n  Per-CWE breakdown:")
    by_cwe_results = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in results:
        if r["status"] == "success":
            cwe = r["cwe_id"]
            by_cwe_results[cwe]["total"] += 1
            if r["hit"]:
                by_cwe_results[cwe]["hits"] += 1

    for cwe, stats in sorted(by_cwe_results.items(), key=lambda x: -x[1]["total"]):
        rate = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"    {cwe}: {stats['hits']}/{stats['total']} ({rate:.0f}%)")

    # Save results
    output_path = Path(RESULTS_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_entries": len(sample),
        "successfully_processed": successes,
        "joern_failures": failures,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total_processed * 100 if total_processed > 0 else 0,
        "per_cwe": dict(by_cwe_results),
        "details": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment()
