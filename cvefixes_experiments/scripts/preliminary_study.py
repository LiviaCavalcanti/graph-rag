"""Preliminary study: generate CPGs + diffs for 10 CVEfixes examples and compare with ground truth."""

import json
import shutil
import tempfile
from pathlib import Path

from src.data.pipeline import write_c_file, run_joern_export, load_cpg_dir, compute_graph_diff

JOERN_BIN_DIR = "/usr/local/bin"
JSON_PATH = "cvefixes_experiments/data/cvefixes_code_extraction.json"
OUTPUT_DIR = Path("cvefixes_experiments/output/preliminary_study")


def select_examples(entries, n=10):
    """Pick n examples with diverse CWEs, manageable code size."""
    seen_cwes = set()
    picks = []
    for i, e in enumerate(entries):
        cwes = e.get("cwe", [])
        if not cwes:
            continue
        cwe_id = cwes[0]["cwe_id"]
        if cwe_id in seen_cwes or cwe_id.startswith("NVD"):
            continue
        cb = e.get("code_before") or ""
        ca = e.get("code_after") or ""
        if not cb or not ca:
            continue
        bl = len(cb.split("\n"))
        al = len(ca.split("\n"))
        if 10 < bl < 50 and 10 < al < 50:
            seen_cwes.add(cwe_id)
            picks.append(e)
        if len(picks) >= n:
            break
    return picks


def generate_cpg(source_code: str, func_name: str, work_dir: Path) -> Path:
    """Write source, run Joern, return graph dir path."""
    src_file = write_c_file(source_code, work_dir / f"{func_name}.cpp")
    graph_dir = work_dir / "graph"
    success = run_joern_export(JOERN_BIN_DIR, str(src_file), str(work_dir), str(graph_dir))
    if not success:
        raise RuntimeError(f"Joern export failed for {func_name} in {work_dir}")
    return graph_dir


def compute_ground_truth_diff(code_before: str, code_after: str):
    """Simple line-level diff as ground truth reference."""
    before_lines = set(code_before.strip().splitlines())
    after_lines = set(code_after.strip().splitlines())
    removed = before_lines - after_lines
    added = after_lines - before_lines
    return {"removed_lines": len(removed), "added_lines": len(added), "total_changed": len(removed) + len(added)}


def run_study():
    with open(JSON_PATH) as f:
        data = json.load(f)

    examples = select_examples(data["entries"])
    print(f"Selected {len(examples)} examples from different CWEs\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []

    for i, e in enumerate(examples):
        cve = e["cve_id"]
        cwe = e["cwe"][0]["cwe_id"]
        func = e["method_name"]
        code_before = e["code_before"]
        code_after = e["code_after"]

        print(f"[{i+1:2d}/10] {cve} | {cwe} | {func}")
        print(f"        Code: {len(code_before.splitlines())} lines (before) → {len(code_after.splitlines())} lines (after)")

        example_dir = OUTPUT_DIR / f"{i+1:02d}_{cve}_{func}"
        example_dir.mkdir(exist_ok=True)

        # Ground truth: line-level diff
        gt = compute_ground_truth_diff(code_before, code_after)
        print(f"        Ground truth: {gt['removed_lines']} lines removed, {gt['added_lines']} lines added")

        # Generate CPGs
        before_dir = example_dir / "before"
        after_dir = example_dir / "after"

        try:
            # Clean previous runs
            if before_dir.exists():
                shutil.rmtree(before_dir)
            if after_dir.exists():
                shutil.rmtree(after_dir)

            graph_before = generate_cpg(code_before, func, before_dir)
            graph_after = generate_cpg(code_after, func, after_dir)

            # Load graphs
            G_before = load_cpg_dir(str(graph_before))
            G_after = load_cpg_dir(str(graph_after))

            # Compute semantic diff
            G_vuln = compute_graph_diff(G_before, G_after)

            # Collect stats
            diff_labels = {}
            for n in G_vuln.nodes():
                label = G_vuln.nodes[n].get("diff", "context")
                diff_labels[label] = diff_labels.get(label, 0) + 1

            result = {
                "example": i + 1,
                "cve": cve,
                "cwe": cwe,
                "func": func,
                "before_nodes": G_before.number_of_nodes(),
                "before_edges": G_before.number_of_edges(),
                "after_nodes": G_after.number_of_nodes(),
                "after_edges": G_after.number_of_edges(),
                "vuln_slice_nodes": G_vuln.number_of_nodes(),
                "vuln_slice_edges": G_vuln.number_of_edges(),
                "diff_labels": diff_labels,
                "ground_truth": gt,
                "status": "success",
            }
            print(f"        CPG before: {G_before.number_of_nodes()} nodes, {G_before.number_of_edges()} edges")
            print(f"        CPG after:  {G_after.number_of_nodes()} nodes, {G_after.number_of_edges()} edges")
            print(f"        Vuln slice: {G_vuln.number_of_nodes()} nodes, {G_vuln.number_of_edges()} edges")
            print(f"        Diff labels: {diff_labels}")

        except Exception as ex:
            result = {
                "example": i + 1,
                "cve": cve,
                "cwe": cwe,
                "func": func,
                "status": "error",
                "error": str(ex),
                "ground_truth": gt,
            }
            print(f"        ERROR: {ex}")

        results.append(result)
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] != "success"]
    print(f"Success: {len(successes)}/10, Failures: {len(failures)}/10")

    if successes:
        print(f"\n{'#':<3} {'CVE':<20} {'CWE':<10} {'Function':<25} {'B_nodes':>7} {'A_nodes':>7} {'Slice':>6} {'Removed':>8} {'Adj':>5} {'GT_chg':>6}")
        print("-" * 100)
        for r in successes:
            dl = r["diff_labels"]
            print(f"{r['example']:<3} {r['cve']:<20} {r['cwe']:<10} {r['func']:<25} "
                  f"{r['before_nodes']:>7} {r['after_nodes']:>7} {r['vuln_slice_nodes']:>6} "
                  f"{dl.get('removed', 0):>8} {dl.get('fix_adjacent', 0):>5} "
                  f"{r['ground_truth']['total_changed']:>6}")

    if failures:
        print("\nFailed examples:")
        for r in failures:
            print(f"  {r['example']}. {r['cve']} ({r['cwe']}): {r['error']}")

    # Save detailed results
    output_file = OUTPUT_DIR / "study_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    run_study()
