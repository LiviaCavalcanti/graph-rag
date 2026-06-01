"""
Shared utilities for CVEfixes sink-identification experiments.

Contains:
  - SINK_PATTERNS: CWE → regex pattern mapping
  - DEFAULT_SINK_PATTERNS: union of all per-CWE patterns
  - get_changed_lines(): compute diff between before/after code
  - find_sink_nodes(): find CPG nodes matching sink patterns
  - label_sinks(): label sinks as TRUE/FALSE vs ground-truth
  - extract_node_subgraph(): k-hop neighbourhood extraction
  - create_stratified_subset(): proportional CWE-stratified sampling
  - build_cpg(): write code + run Joern + load graph (full pipeline)
"""

import difflib
import json
import random
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import networkx as nx

from src.data.pipeline import load_cpg_dir, run_joern_export, write_c_file

# ── Configuration ──
JOERN_BIN_DIR = "/home/z0050s2b/bin/joern/joern-cli"
DATA_FILE = "cvefixes_experiments/data/cvefixes_filtered_by_cwe.json"
SUBSET_FILE = "cvefixes_experiments/output/joern_sink_subset_100.json"
RESULTS_FILE = "cvefixes_experiments/output/joern_sink_results.json"

# ── Sink patterns mapped to CWE families ──
SINK_PATTERNS = {
    "CWE-416": [
        re.compile(r"\b(free|kfree|vfree|kvfree|delete|release)\b"),
        re.compile(r"\*\w|->"),
        re.compile(r"\b(lock|mutex_lock|spin_lock|down_read|down_write)\b"),
    ],
    "CWE-476": [
        re.compile(r"\*\w|->"),
        re.compile(r"\b(NULL|null|nullptr)\b"),
        re.compile(r"\b(IS_ERR|PTR_ERR|ERR_PTR)\b"),
        re.compile(r"\b(memcpy|memmove|memset)\b"),
    ],
    "CWE-190": [
        re.compile(r"[+\-*/]|<<|>>"),
        re.compile(
            r"\(\s*(?:unsigned|signed|int|long|short|char|void|size_t"
            r"|u?int\d+_t)\s*\*?\s*\)"
        ),
        re.compile(r"\b(int|long|short|unsigned|size_t|uint\d+_t|int\d+_t)\s+\w+"),
    ],
    "CWE-191": [
        re.compile(r"[+\-*/]|<<|>>"),
    ],
    "CWE-20": [
        re.compile(r"\b(if|assert|BUG_ON|WARN_ON|check|verify|IS_ERR)\b"),
        re.compile(r"\b(NULL|null|nullptr)\b"),
        re.compile(r"->"),
        re.compile(r"\b(memcpy|copy_from_user|strlen|size|length|len)\b"),
    ],
    "CWE-362": [
        re.compile(r"->"),
        re.compile(r"\b(memcpy|memmove|copy_from_user|copy_to_user|memset)\b"),
        re.compile(r"\*\w"),
        re.compile(r"\w+\s*=\s*\w+->"),
    ],
    "CWE-400": [
        re.compile(r"\b(malloc|calloc|kmalloc|kzalloc|alloc|realloc|krealloc|new)\b"),
        re.compile(r"\b(free|kfree|vfree|kvfree|delete|release)\b"),
        re.compile(r"->"),
        re.compile(r"\b(perf_|event_|overflow|output_begin|sw_event)\w*\b"),
    ],
    "CWE-401": [
        re.compile(r"\b(malloc|calloc|kmalloc|kzalloc|alloc|realloc|krealloc|new|skb)\b"),
        re.compile(r"\b(free|kfree|vfree|kvfree|delete|release|kfree_skb)\b"),
        re.compile(r"->"),
    ],
    "CWE-252": [
        re.compile(r"\b(if|assert|BUG_ON|WARN_ON|check|verify|IS_ERR)\b"),
    ],
    "CWE-787": [
        re.compile(r"[+\-*/]|<<|>>"),
        re.compile(r"\b(MAX|MIN|SIZE_MAX|UINT_MAX|INT_MAX|limit|bound|clamp)\b", re.I),
        re.compile(r"\*\w|->"),
    ],
    "CWE-667": [
        re.compile(r"->"),
        re.compile(r"\*\w"),
        re.compile(r"\b(lock|mutex_lock|spin_lock|down_read|down_write|rtnl_lock)\b"),
        re.compile(
            r"\b(unlock|mutex_unlock|spin_unlock|up_read|up_write|rtnl_unlock)\b"
        ),
        re.compile(r"\b(expand_stack|mmget_still_valid|mmap)\b"),
    ],
    "CWE-129": [
        re.compile(r"[+\-*/]|<<|>>"),
        re.compile(r"\b(MAX|MIN|SIZE_MAX|UINT_MAX|INT_MAX|limit|bound|clamp)\b", re.I),
    ],
}

# Deduplicated union of all per-CWE patterns
DEFAULT_SINK_PATTERNS = list(
    {p.pattern: p for pats in SINK_PATTERNS.values() for p in pats}.values()
)

# ── Wrapping-fix patterns ──
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

WRAPPING_FIX_CWES = {
    "CWE-362": RE_LOCK_FIX,
    "CWE-667": RE_LOCK_FIX,
    "CWE-476": RE_NULL_CHECK_FIX,
    "CWE-416": RE_REFCOUNT_FIX,
    "CWE-20": RE_VALIDATION_FIX,
    "CWE-400": RE_VALIDATION_FIX,
    "CWE-401": RE_FREE_FIX,
}


# ── Diff utilities ──


def get_changed_lines(code_before: str, code_after: str) -> tuple[set[str], set[str]]:
    """
    Return (removed_lines, added_lines) as sets of stripped line contents.
    """
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
    removed.discard("")
    added.discard("")
    return removed, added


# ── Sink finding ──


def find_sink_nodes(G: nx.MultiDiGraph, cwe_id: str) -> list[dict]:
    """
    Walk the CPG graph and find nodes whose CODE attribute matches
    the sink patterns for the given CWE.

    Returns list of dicts: [{node_id, code, label, matched_pattern}, ...]
    """
    patterns = SINK_PATTERNS.get(cwe_id, DEFAULT_SINK_PATTERNS)
    sink_nodes = []
    for n, attr in G.nodes(data=True):
        code = (attr.get("CODE") or "").strip()
        if not code:
            continue
        label = attr.get("labelV", "UNKNOWN")
        if label in ("METHOD", "BLOCK", "METHOD_RETURN"):
            continue
        for pat in patterns:
            if pat.search(code):
                sink_nodes.append({
                    "node_id": n,
                    "code": code,
                    "label": label,
                    "matched_pattern": pat.pattern,
                })
                break
    return sink_nodes


def label_sinks(sinks: list[dict], changed_lines: set[str]) -> list[dict]:
    """
    Label each sink as TRUE (its code overlaps a changed line) or FALSE.
    """
    for sink in sinks:
        sink_code = sink["code"]
        sink["is_true"] = any(
            sink_code in cl or cl in sink_code
            for cl in changed_lines
        )
    return sinks


# ── Graph utilities ──


def extract_node_subgraph(G: nx.MultiDiGraph, center_node, k_hop: int) -> nx.MultiDiGraph:
    """Extract k-hop neighbourhood around a single node."""
    neighbourhood = {center_node}
    frontier = {center_node}
    for _ in range(k_hop):
        next_frontier = set()
        for node in frontier:
            for _, succ in G.out_edges(node):
                if succ not in neighbourhood:
                    next_frontier.add(succ)
            for pred, _ in G.in_edges(node):
                if pred not in neighbourhood:
                    next_frontier.add(pred)
        neighbourhood |= next_frontier
        frontier = next_frontier
    return G.subgraph(neighbourhood).copy()


def build_cpg(code: str, prefix: str = "cpg_") -> nx.MultiDiGraph | None:
    """
    Write code to a temp file, run Joern, load the resulting CPG graph.
    Returns the graph or None on failure.
    """
    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        tmpdir_path = Path(tmpdir)
        src_path = write_c_file(code, tmpdir_path / "vuln.c")
        cpg_dir = str(tmpdir_path / "cpg")
        graph_dir = str(tmpdir_path / "graph")

        ok = run_joern_export(JOERN_BIN_DIR, str(src_path), cpg_dir, graph_dir)
        if not ok:
            return None
        try:
            return load_cpg_dir(graph_dir)
        except Exception:
            return None


# ── Sampling ──


def create_stratified_subset(data_file: str, output_file: str, n: int, seed: int):
    """Create a stratified-by-primary-CWE subset of size n."""
    with open(data_file) as f:
        data = json.load(f)
    entries = data["entries"]

    valid = [e for e in entries if e.get("code_before") and e.get("code_after")]
    print(f"  Total valid entries (with both before/after code): {len(valid)}")

    rng = random.Random(seed)

    by_cwe = defaultdict(list)
    for e in valid:
        primary_cwe = e["cwe"][0]["cwe_id"]
        by_cwe[primary_cwe].append(e)

    total = len(valid)
    sample = []

    for cwe, group in sorted(by_cwe.items(), key=lambda x: -len(x[1])):
        k = max(1, round(len(group) / total * n))
        chosen = rng.sample(group, min(k, len(group)))
        sample.extend(chosen)

    rng.shuffle(sample)
    sample = sample[:n]

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


# ── Overlap checking ──


def check_overlap(
    sink_nodes: list,
    removed_lines: set,
    added_lines: set,
    cwe_id: str = "",
    code_before: str = "",
    code_after: str = "",
) -> tuple[list, int, int]:
    """
    Check if any sink node's code overlaps with the changed lines.

    Two-phase check:
    1. Direct overlap: sink code appears in a changed line
    2. Wrapping-fix check: fix added protection around sink operations

    Returns: (hit_nodes, total_sinks, total_changed)
    """
    all_changed = removed_lines | added_lines
    hit_nodes = []

    # Phase 1: Direct overlap
    for sink in sink_nodes:
        sink_code = sink["code"]
        for changed_line in all_changed:
            if not changed_line:
                continue
            if sink_code in changed_line or changed_line in sink_code:
                hit_nodes.append(sink)
                break

    # Phase 2: Wrapping-fix check
    if not hit_nodes and cwe_id in WRAPPING_FIX_CWES and code_after:
        fix_pattern = WRAPPING_FIX_CWES[cwe_id]
        wrap_added = any(fix_pattern.search(line) for line in added_lines)
        if wrap_added:
            after_lines_stripped = [l.strip() for l in code_after.split("\n")]
            for sink in sink_nodes:
                if sink in hit_nodes:
                    continue
                sink_code = sink["code"]
                if sink["label"] in ("METHOD", "BLOCK", "METHOD_RETURN"):
                    continue
                if len(sink_code) < 4:
                    continue
                for after_line in after_lines_stripped:
                    if sink_code in after_line or after_line in sink_code:
                        hit_nodes.append(sink)
                        break

    # Phase 2b: CWE-400 signature-change pattern
    if not hit_nodes and cwe_id == "CWE-400":
        for sink in sink_nodes:
            sink_code = sink["code"]
            if sink["label"] in ("METHOD", "BLOCK", "METHOD_RETURN"):
                continue
            func_match = re.match(r"(\w+)\s*\(", sink_code)
            if func_match:
                func_name = func_match.group(1)
                for changed_line in all_changed:
                    if func_name in changed_line:
                        hit_nodes.append(sink)
                        break

    return hit_nodes, len(sink_nodes), len(all_changed)
