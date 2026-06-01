"""
Experiment: Joern Native Query Sink Verification

Instead of exporting the full CPG as GraphML and walking it with regex
patterns in Python, this experiment uses Joern's built-in CPGQL query
language to find sinks directly. Joern natively supports:
  - Dataflow/taint analysis (reachableBy, flows)
  - Pattern matching on AST/CFG/PDG structure
  - Sink/source definitions via CPGQL

This tests whether Joern-native queries find sinks more precisely than
the regex-over-exported-graph approach.

Protocol:
  1. Load the same stratified subset used by exp_joern_sink_verification
  2. For each entry, build a CPG (joern-parse)
  3. Run CWE-specific CPGQL scripts via `joern --script` to find sinks
  4. Compare results with the regex-based approach (same ground truth)
  5. Report precision/recall improvement
"""

import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from src.data.pipeline import write_c_file

from cvefixes_experiments.scripts.common import (
    JOERN_BIN_DIR,
    SUBSET_FILE,
    RESULTS_FILE,
    DATA_FILE,
    create_stratified_subset,
    get_changed_lines,
)

# ── Configuration ──
SEED = 42
SUBSET_SIZE = 100
OUTPUT_FILE = "cvefixes_experiments/output/joern_native_query_results.json"
SCRIPTS_DIR = Path("cvefixes_experiments/scripts/joern_queries")

# ── CPGQL sink queries per CWE ──
# Each returns JSON lines: [{lineNumber, code}]
JOERN_QUERIES = {
    "CWE-416": """\
// Use-After-Free: find free calls and pointer dereferences after them
val frees = cpg.call.name("(k)?free|vfree|kvfree|delete|release").l
val derefs = cpg.call.filter(_.code.matches(".*->.*|.*\\\\*\\\\w.*")).l
val sinks = (frees ++ derefs).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-476": """\
// Null Pointer Dereference: find dereferences without null checks
val derefs = cpg.call.filter(_.code.matches(".*->.*|.*\\\\*\\\\w.*")).l
val nullChecks = cpg.call.name("IS_ERR|PTR_ERR|ERR_PTR").l
val memOps = cpg.call.name("memcpy|memmove|memset").l
val sinks = (derefs ++ nullChecks ++ memOps).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-190": """\
// Integer Overflow: find arithmetic ops and integer type casts
val arith = cpg.call.name("<operator>.(addition|subtraction|multiplication|division|shiftLeft|shiftRight)").l
val casts = cpg.call.filter(_.code.matches(".*\\\\(\\\\s*(unsigned|signed|int|long|short|size_t|u?int\\\\d+_t)\\\\s*\\\\).*")).l
val sinks = (arith ++ casts).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-191": """\
// Integer Underflow: arithmetic operations
val arith = cpg.call.name("<operator>.(addition|subtraction|multiplication|division|shiftLeft|shiftRight)").l
val sinks = arith.map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-20": """\
// Improper Input Validation: find operations that should be validated
val checks = cpg.call.name("assert|BUG_ON|WARN_ON|IS_ERR").l
val fieldAccess = cpg.call.filter(_.code.matches(".*->.*")).l
val memOps = cpg.call.name("memcpy|copy_from_user|strlen").l
val sinks = (checks ++ fieldAccess ++ memOps).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-362": """\
// Race Condition: find shared-state accesses (should be locked)
val fieldAccess = cpg.call.filter(_.code.matches(".*->.*")).l
val ptrDeref = cpg.call.filter(_.code.matches(".*\\\\*\\\\w.*")).l
val memOps = cpg.call.name("memcpy|memmove|copy_from_user|copy_to_user|memset").l
val sinks = (fieldAccess ++ ptrDeref ++ memOps).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-400": """\
// Resource Consumption: find allocation/deallocation calls
val allocs = cpg.call.name("malloc|calloc|kmalloc|kzalloc|alloc|realloc|krealloc").l
val frees = cpg.call.name("(k)?free|vfree|kvfree|delete|release").l
val sinks = (allocs ++ frees).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-401": """\
// Memory Leak: find allocations without matching frees
val allocs = cpg.call.name("malloc|calloc|kmalloc|kzalloc|alloc|realloc|krealloc").l
val frees = cpg.call.name("(k)?free|vfree|kvfree|delete|release|kfree_skb").l
val sinks = (allocs ++ frees).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-252": """\
// Unchecked Return Value: find calls whose return value should be checked
val checks = cpg.call.name("IS_ERR|PTR_ERR").l
val ifStmts = cpg.controlStructure.controlStructureType("IF").l
val sinks = checks.map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-787": """\
// Out-of-bounds Write: arithmetic + pointer ops
val arith = cpg.call.name("<operator>.(addition|subtraction|multiplication|division|shiftLeft|shiftRight)").l
val ptrOps = cpg.call.filter(_.code.matches(".*->.*|.*\\\\*\\\\w.*")).l
val sinks = (arith ++ ptrOps).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-667": """\
// Improper Locking: find lock/unlock calls and shared-state accesses
val locks = cpg.call.name(".*lock.*|.*unlock.*|down_read|down_write|up_read|up_write").l
val fieldAccess = cpg.call.filter(_.code.matches(".*->.*")).l
val sinks = (locks ++ fieldAccess).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
    "CWE-129": """\
// Improper Array Index: arithmetic and bounds checks
val arith = cpg.call.name("<operator>.(addition|subtraction|multiplication|division|shiftLeft|shiftRight)").l
val sinks = arith.map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
""",
}

# Default query for CWEs not explicitly listed
DEFAULT_JOERN_QUERY = """\
// Generic: find all call nodes (potential sinks)
val allocs = cpg.call.name("malloc|calloc|kmalloc|kzalloc|alloc|realloc|krealloc").l
val frees = cpg.call.name("(k)?free|vfree|kvfree|delete|release").l
val arith = cpg.call.name("<operator>.(addition|subtraction|multiplication|division)").l
val fieldAccess = cpg.call.filter(_.code.matches(".*->.*")).l
val sinks = (allocs ++ frees ++ arith ++ fieldAccess).map(n => Map("lineNumber" -> n.lineNumber.getOrElse(-1), "code" -> n.code))
println(sinks.toJson)
"""


def run_joern_query(cpg_file: str, query: str, timeout: int = 60) -> list[dict] | None:
    """
    Run a CPGQL query against a CPG binary using `joern --script`.
    Returns list of {lineNumber, code} dicts, or None on failure.
    """
    joern_bin = Path(JOERN_BIN_DIR) / "joern"

    # Write query to a temp .sc file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sc", delete=False) as f:
        # Wrap the query: import CPG, run query, close
        script = f"""\
importCpg("{cpg_file}")
{query}
"""
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [str(joern_bin), "--script", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None

        # Parse JSON output from the last line
        output = result.stdout.strip()
        if not output:
            return []

        # The query prints JSON; find the last JSON array in output
        for line in reversed(output.split("\n")):
            line = line.strip()
            if line.startswith("["):
                return json.loads(line)

        return []
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None
    finally:
        Path(script_path).unlink(missing_ok=True)


def parse_cpg(code: str, prefix: str = "native_") -> str | None:
    """
    Run joern-parse to create a CPG binary. Returns path to cpg.bin or None.
    Note: caller must clean up the temp directory.
    """
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    tmpdir_path = Path(tmpdir)
    src_path = write_c_file(code, tmpdir_path / "vuln.c")
    cpg_file = str(tmpdir_path / "cpg.bin")

    joern_parse = Path(JOERN_BIN_DIR) / "joern-parse"
    result = subprocess.run(
        [str(joern_parse), str(src_path), "--output", cpg_file, "--language", "newc"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return None
    return cpg_file


def check_sink_overlap(sink_results: list[dict], removed: set, added: set) -> list[dict]:
    """Check which Joern-found sinks overlap with changed lines."""
    all_changed = removed | added
    hits = []
    for sink in sink_results:
        sink_code = sink.get("code", "").strip()
        if not sink_code:
            continue
        for changed_line in all_changed:
            if not changed_line:
                continue
            if sink_code in changed_line or changed_line in sink_code:
                hits.append(sink)
                break
    return hits


def run_experiment():
    """Run the Joern native query experiment."""
    print("=" * 70)
    print("EXPERIMENT: Joern Native Query Sink Verification")
    print("=" * 70)

    # Load subset
    if Path(SUBSET_FILE).exists():
        print(f"\n[1/3] Loading existing subset from {SUBSET_FILE}")
        with open(SUBSET_FILE) as f:
            subset_data = json.load(f)
        sample = subset_data["entries"]
        print(f"  Loaded {len(sample)} entries")
    else:
        print(f"\n[1/3] Creating stratified subset of {SUBSET_SIZE} entries...")
        sample = create_stratified_subset(DATA_FILE, SUBSET_FILE, SUBSET_SIZE, SEED)

    # Process entries
    print(f"\n[2/3] Running Joern native queries on {len(sample)} entries...")
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

        # Compute ground truth
        removed, added = get_changed_lines(code_before, code_after)
        if not removed and not added:
            print("    ⚠ No diff detected, skipping")
            results.append({
                "cve_id": cve_id, "cwe_id": cwe_id,
                "method_name": method_name, "status": "no_diff",
            })
            continue

        # Parse CPG
        cpg_file = parse_cpg(code_before, prefix=f"native_{cve_id}_")
        if cpg_file is None:
            print("    ✗ Joern parse failed")
            failures += 1
            results.append({
                "cve_id": cve_id, "cwe_id": cwe_id,
                "method_name": method_name, "status": "joern_failed",
            })
            continue

        # Run CWE-specific query
        query = JOERN_QUERIES.get(cwe_id, DEFAULT_JOERN_QUERY)
        sink_results = run_joern_query(cpg_file, query)

        # Clean up CPG temp dir
        cpg_dir = Path(cpg_file).parent
        import shutil
        shutil.rmtree(cpg_dir, ignore_errors=True)

        if sink_results is None:
            print("    ✗ Joern query failed")
            failures += 1
            results.append({
                "cve_id": cve_id, "cwe_id": cwe_id,
                "method_name": method_name, "status": "query_failed",
            })
            continue

        successes += 1
        print(f"    ✓ Joern found {len(sink_results)} sink candidates")

        # Check overlap
        hit_nodes = check_sink_overlap(sink_results, removed, added)
        if hit_nodes:
            hits += 1
            print(f"    ✓ HIT — {len(hit_nodes)}/{len(sink_results)} sinks match changed lines")
        else:
            misses += 1
            print(f"    ✗ MISS — 0/{len(sink_results)} sinks match changed lines")

        results.append({
            "cve_id": cve_id,
            "cwe_id": cwe_id,
            "method_name": method_name,
            "status": "success",
            "total_sinks": len(sink_results),
            "hit_count": len(hit_nodes),
            "hit": bool(hit_nodes),
            "removed_lines": list(removed)[:10],
            "added_lines": list(added)[:10],
            "sample_hits": [{"code": h["code"], "line": h.get("lineNumber")} for h in hit_nodes[:5]],
        })

    # Report
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

    # Per-CWE breakdown
    print("\n  Per-CWE breakdown:")
    by_cwe = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in results:
        if r["status"] == "success":
            cwe = r["cwe_id"]
            by_cwe[cwe]["total"] += 1
            if r["hit"]:
                by_cwe[cwe]["hits"] += 1

    for cwe, stats in sorted(by_cwe.items(), key=lambda x: -x[1]["total"]):
        rate = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"    {cwe}: {stats['hits']}/{stats['total']} ({rate:.0f}%)")

    # Compare with regex-based results if available
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE) as f:
            regex_results = json.load(f)
        regex_hit_rate = regex_results.get("hit_rate", 0)
        print(f"\n  --- Comparison with regex-based approach ---")
        print(f"  Regex hit rate:  {regex_hit_rate:.1f}%")
        if total_processed > 0:
            print(f"  Native hit rate: {hit_rate:.1f}%")
            diff = hit_rate - regex_hit_rate
            print(f"  Difference:      {diff:+.1f}%")

    # Save
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_entries": len(sample),
        "successfully_processed": successes,
        "joern_failures": failures,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total_processed * 100 if total_processed > 0 else 0,
        "per_cwe": dict(by_cwe),
        "details": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    run_experiment()
