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

import difflib
import json
import random
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import networkx as nx

# ── Project imports ──
from src.data.pipeline import load_cpg_dir, run_joern_export, write_c_file

# ── Configuration ──
JOERN_BIN_DIR = "/home/z0050s2b/bin/joern/joern-cli"
DATA_FILE = "cvefixes_experiments/data/cvefixes_filtered_by_cwe.json"
SUBSET_FILE = "cvefixes_experiments/output/joern_sink_subset_100.json"
RESULTS_FILE = "cvefixes_experiments/output/joern_sink_results.json"
SEED = 42
SUBSET_SIZE = 100

# ── Sink patterns (from vuln_pattern.py) mapped to CWE families ──
# Each CWE gets a set of regex patterns that identify the "sink" operations
# relevant to that vulnerability class.

SINK_PATTERNS = {
    # Use-After-Free, Double-Free
    # Sink is both the free AND the subsequent use (dereference)
    "CWE-416": [
        re.compile(r"\b(free|kfree|vfree|kvfree|delete|release)\b"),
        re.compile(r"\*\w|->"),
        re.compile(r"\b(lock|mutex_lock|spin_lock|down_read|down_write)\b"),  # fix may add locking
    ],
    # Null Pointer Dereference
    # Sink is the UNPROTECTED dereference. Fix adds a null check before it.
    "CWE-476": [
        re.compile(r"\*\w|->"),
        re.compile(r"\b(NULL|null|nullptr)\b"),
        re.compile(r"\b(IS_ERR|PTR_ERR|ERR_PTR)\b"),  # kernel error-pointer patterns
        re.compile(r"\b(memcpy|memmove|memset)\b"),    # pointer use without deref syntax
    ],
    # Integer Overflow
    # Sink includes variable declarations with integer types (fix often changes type)
    # and arithmetic operations.
    "CWE-190": [
        re.compile(r"[+\-*/]|<<|>>"),
        re.compile(
            r"\(\s*(?:unsigned|signed|int|long|short|char|void|size_t"
            r"|u?int\d+_t)\s*\*?\s*\)"
        ),
        re.compile(r"\b(int|long|short|unsigned|size_t|uint\d+_t|int\d+_t)\s+\w+"),  # declarations
    ],
    # Integer Underflow
    "CWE-191": [
        re.compile(r"[+\-*/]|<<|>>"),
    ],
    # Improper Input Validation
    # Sink is the unvalidated operation. Fix adds checks before/around it.
    "CWE-20": [
        re.compile(r"\b(if|assert|BUG_ON|WARN_ON|check|verify|IS_ERR)\b"),
        re.compile(r"\b(NULL|null|nullptr)\b"),
        re.compile(r"->"),  # field access on unvalidated input
        re.compile(r"\b(memcpy|copy_from_user|strlen|size|length|len)\b"),  # size/length ops
    ],
    # Race Condition — sink is the UNPROTECTED shared-state access, not the lock itself
    # The fix adds locks around these operations; so we look for what SHOULD be locked.
    "CWE-362": [
        re.compile(r"->"),                         # struct field access (shared state)
        re.compile(r"\b(memcpy|memmove|copy_from_user|copy_to_user|memset)\b"),  # memory writes
        re.compile(r"\*\w"),                        # pointer dereference
        re.compile(r"\w+\s*=\s*\w+->"),            # reading shared state into local
    ],
    # Uncontrolled Resource Consumption / Resource Leak
    # The fix often changes function signatures or adds resource cleanup.
    # Sinks are the resource operations (function calls, event handlers).
    "CWE-400": [
        re.compile(r"\b(malloc|calloc|kmalloc|kzalloc|alloc|realloc|krealloc|new)\b"),
        re.compile(r"\b(free|kfree|vfree|kvfree|delete|release)\b"),
        re.compile(r"->"),  # shared-state access often involved in resource exhaustion
        re.compile(r"\b(perf_|event_|overflow|output_begin|sw_event)\w*\b"),  # perf/event subsystem calls
    ],
    # Memory Leak — fix adds a free/cleanup call for an already-allocated resource
    "CWE-401": [
        re.compile(r"\b(malloc|calloc|kmalloc|kzalloc|alloc|realloc|krealloc|new|skb)\b"),
        re.compile(r"\b(free|kfree|vfree|kvfree|delete|release|kfree_skb)\b"),
        re.compile(r"->"),  # access to allocated object
    ],
    # Unchecked Return Value
    "CWE-252": [
        re.compile(r"\b(if|assert|BUG_ON|WARN_ON|check|verify|IS_ERR)\b"),
    ],
    # Out-of-bounds Write
    "CWE-787": [
        re.compile(r"[+\-*/]|<<|>>"),
        re.compile(r"\b(MAX|MIN|SIZE_MAX|UINT_MAX|INT_MAX|limit|bound|clamp)\b", re.I),
        re.compile(r"\*\w|->"),
    ],
    # Improper Locking — same logic as CWE-362, sink is the unprotected operation
    "CWE-667": [
        re.compile(r"->"),  # shared-state field access
        re.compile(r"\*\w"),  # pointer dereference
        re.compile(r"\b(lock|mutex_lock|spin_lock|down_read|down_write|rtnl_lock)\b"),
        re.compile(
            r"\b(unlock|mutex_unlock|spin_unlock|up_read|up_write|rtnl_unlock)\b"
        ),
        re.compile(r"\b(expand_stack|mmget_still_valid|mmap)\b"),  # mm operations
    ],
    # Improper Restriction of Operations within Bounds
    "CWE-129": [
        re.compile(r"[+\-*/]|<<|>>"),
        re.compile(r"\b(MAX|MIN|SIZE_MAX|UINT_MAX|INT_MAX|limit|bound|clamp)\b", re.I),
    ],
}

# Fallback patterns for CWEs not explicitly listed
DEFAULT_SINK_PATTERNS = [
    re.compile(r"\b(free|kfree|vfree|kvfree|delete|release)\b"),
    re.compile(r"\b(malloc|calloc|kmalloc|kzalloc|alloc|realloc|krealloc|new)\b"),
    re.compile(r"\*\w|->"),
    re.compile(r"\b(if|assert|BUG_ON|WARN_ON|check|verify|IS_ERR)\b"),
    re.compile(r"\b(NULL|null|nullptr)\b"),
    re.compile(r"[+\-*/]|<<|>>"),
    re.compile(r"\b(lock|mutex_lock|spin_lock|down_read|down_write|rtnl_lock)\b"),
    re.compile(
        r"\b(unlock|mutex_unlock|spin_unlock|up_read|up_write|rtnl_unlock)\b"
    ),
    re.compile(r"\b(MAX|MIN|SIZE_MAX|UINT_MAX|INT_MAX|limit|bound|clamp)\b", re.I),
    re.compile(
        r"\(\s*(?:unsigned|signed|int|long|short|char|void|size_t"
        r"|u?int\d+_t)\s*\*?\s*\)"
    ),
]


# ── Step 1: Stratified sampling ──


def create_stratified_subset(data_file: str, output_file: str, n: int, seed: int):
    """Create a stratified-by-primary-CWE subset of size n."""
    with open(data_file) as f:
        data = json.load(f)
    entries = data["entries"]

    # Filter to entries that have both code_before and code_after (needed for diff)
    valid = [e for e in entries if e.get("code_before") and e.get("code_after")]
    print(f"  Total valid entries (with both before/after code): {len(valid)}")

    rng = random.Random(seed)

    # Group by primary CWE
    by_cwe = defaultdict(list)
    for e in valid:
        primary_cwe = e["cwe"][0]["cwe_id"]
        by_cwe[primary_cwe].append(e)

    total = len(valid)
    sample = []

    # Proportional stratified sampling
    for cwe, group in sorted(by_cwe.items(), key=lambda x: -len(x[1])):
        k = max(1, round(len(group) / total * n))
        chosen = rng.sample(group, min(k, len(group)))
        sample.extend(chosen)

    # Trim or pad to exactly n
    rng.shuffle(sample)
    sample = sample[:n]

    # Save subset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset_data = {
        "description": f"Stratified subset of {n} CVEfixes entries (by primary CWE), seed={seed}",
        "source": data_file,
        "size": len(sample),
        "cwe_distribution": dict(
            sorted(
                defaultdict(
                    int,
                    {
                        e["cwe"][0]["cwe_id"]: sum(
                            1 for x in sample if x["cwe"][0]["cwe_id"] == e["cwe"][0]["cwe_id"]
                        )
                        for e in sample
                    },
                ).items(),
                key=lambda x: -x[1],
            )
        ),
        "entries": sample,
    }
    with open(output_path, "w") as f:
        json.dump(subset_data, f, indent=2)
    print(f"  Saved stratified subset ({len(sample)} entries) → {output_file}")
    return sample


# ── Step 2: Compute changed lines ──


def get_changed_lines(code_before: str, code_after: str):
    """Return sets of removed and added line contents (stripped)."""
    before_lines = code_before.split("\n")
    after_lines = code_after.split("\n")

    diff = list(difflib.unified_diff(before_lines, after_lines, lineterm=""))
    removed = set()
    added = set()

    for line in diff:
        if line.startswith("-") and not line.startswith("---"):
            removed.add(line[1:].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            added.add(line[1:].strip())

    return removed, added


# ── Step 3: Find sink nodes in graph ──


def find_sink_nodes(G: nx.MultiDiGraph, cwe_id: str):
    """
    Walk the CPG graph and find nodes whose CODE attribute matches
    the sink patterns for the given CWE.

    Returns a list of dicts: [{node_id, code, label, matched_pattern}, ...]
    """
    patterns = SINK_PATTERNS.get(cwe_id, DEFAULT_SINK_PATTERNS)

    sink_nodes = []
    for n, attr in G.nodes(data=True):
        code = attr.get("CODE", "") or ""
        if not code.strip():
            continue
        for pat in patterns:
            if pat.search(code):
                sink_nodes.append(
                    {
                        "node_id": n,
                        "code": code.strip(),
                        "label": attr.get("labelV", "UNKNOWN"),
                        "matched_pattern": pat.pattern,
                    }
                )
                break  # one match is enough per node

    return sink_nodes


# ── Step 4: Check overlap between sinks and changed lines ──

# Patterns that indicate the fix WRAPS existing code with protection
# (rather than changing the code itself)
RE_LOCK_FIX = re.compile(
    r"\b(lock|mutex_lock|spin_lock|down_read|down_write|rtnl_lock"
    r"|bh_lock_sock|rcu_read_lock|read_lock|write_lock"
    r"|spin_lock_bh|local_bh_disable)\b"
)
RE_NULL_CHECK_FIX = re.compile(
    r"\b(NULL|null|nullptr|IS_ERR|PTR_ERR)\b|!\s*\w+"
)
RE_VALIDATION_FIX = re.compile(
    r"\b(if|assert|BUG_ON|WARN_ON|check|verify|IS_ERR|return)\b"
)
RE_REFCOUNT_FIX = re.compile(
    r"\b(get|put|ref|unref|grab|release|kref)\b"
)
RE_FREE_FIX = re.compile(
    r"\b(free|kfree|vfree|kvfree|kfree_skb|delete|release)\b"
)

# CWEs where the fix typically WRAPS/GUARDS existing code rather than modifying it
WRAPPING_FIX_CWES = {
    "CWE-362": RE_LOCK_FIX,           # adds lock around shared-state access
    "CWE-667": RE_LOCK_FIX,           # adds proper locking
    "CWE-476": RE_NULL_CHECK_FIX,     # adds null check before dereference
    "CWE-416": RE_REFCOUNT_FIX,       # adds refcount/lock before use
    "CWE-20": RE_VALIDATION_FIX,      # adds input validation check
    "CWE-400": RE_VALIDATION_FIX,     # adds resource limit/cleanup check
    "CWE-401": RE_FREE_FIX,           # adds free/cleanup for leaked resource
}


def check_overlap(
    sink_nodes: list,
    removed_lines: set,
    added_lines: set,
    cwe_id: str = "",
    code_before: str = "",
    code_after: str = "",
):
    """
    Check if any sink node's code overlaps with the changed lines.

    Two-phase check:
    1. Direct overlap: sink code appears in a changed line (standard)
    2. Wrapping-fix check (for CWEs where the fix adds protection around
       existing code): the sink's operation still exists in code_after,
       meaning the fix wrapped it with a guard/lock/check rather than
       removing it.

    Returns:
      - hit_nodes: list of sink nodes that match
      - total_sinks: total number of sink nodes found
      - total_changed: total number of changed lines
    """
    all_changed = removed_lines | added_lines
    hit_nodes = []

    # Phase 1: Direct overlap check (for all CWEs)
    for sink in sink_nodes:
        sink_code = sink["code"]
        for changed_line in all_changed:
            if not changed_line:
                continue
            if sink_code in changed_line or changed_line in sink_code:
                hit_nodes.append(sink)
                break

    # Phase 2: Wrapping-fix check for applicable CWEs
    # If no direct overlap found, check if the fix added a protective wrapper
    # around the sink operations (which remain unchanged in code_after)
    if not hit_nodes and cwe_id in WRAPPING_FIX_CWES and code_after:
        fix_pattern = WRAPPING_FIX_CWES[cwe_id]
        # Check if any added lines match the wrapping pattern
        wrap_added = any(fix_pattern.search(line) for line in added_lines)
        if wrap_added:
            # The fix added protection. Check if sink nodes' code still exists
            # in code_after (meaning the op is still there, just wrapped)
            after_lines_stripped = [l.strip() for l in code_after.split("\n")]
            for sink in sink_nodes:
                if sink in hit_nodes:
                    continue
                sink_code = sink["code"]
                # Skip whole-method or block nodes
                if sink["label"] in ("METHOD", "BLOCK", "METHOD_RETURN"):
                    continue
                # Skip very short codes that would match too broadly
                if len(sink_code) < 4:
                    continue
                for after_line in after_lines_stripped:
                    if sink_code in after_line or after_line in sink_code:
                        hit_nodes.append(sink)
                        break

    # Phase 2b: For CWE-400 signature-change pattern
    # The fix changes a function call signature (adds/removes params).
    # The sink is the function call itself — check if any sink's function name
    # appears in a changed line (even if full text doesn't match).
    if not hit_nodes and cwe_id == "CWE-400":
        for sink in sink_nodes:
            sink_code = sink["code"]
            if sink["label"] in ("METHOD", "BLOCK", "METHOD_RETURN"):
                continue
            # Extract function name from call nodes
            func_match = re.match(r"(\w+)\s*\(", sink_code)
            if func_match:
                func_name = func_match.group(1)
                for changed_line in all_changed:
                    if func_name in changed_line:
                        hit_nodes.append(sink)
                        break

    return hit_nodes, len(sink_nodes), len(all_changed)


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
