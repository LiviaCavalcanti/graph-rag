"""Extract line-based metrics from preliminary study CPGs for 1:1 GT comparison."""
import json
from pathlib import Path
from src.data.pipeline import load_cpg_dir, compute_graph_diff

OUTPUT_DIR = Path("preliminary_study_output")
results = json.loads((OUTPUT_DIR / "study_results.json").read_text())

print(f"{'#':<3} {'CVE':<20} {'Slice(L)':<9} {'Removed(L)':<11} {'Fix-adj(L)':<11} {'Context(L)':<11} {'GT(L)':<6}")
print("-" * 80)

for r in results:
    idx = r["example"]
    cve = r["cve"]
    func = r["func"]
    
    # Find the directory
    dirs = sorted(OUTPUT_DIR.glob(f"{idx:02d}_*"))
    if not dirs:
        print(f"{idx:<3} {cve:<20} -- directory not found --")
        continue
    
    example_dir = dirs[0]
    before_dir = str(example_dir / "before")
    after_dir = str(example_dir / "after")
    
    try:
        G_before = load_cpg_dir(before_dir)
        G_after = load_cpg_dir(after_dir)
        G_vuln = compute_graph_diff(G_before, G_after)
        
        # Extract unique lines per diff category
        lines_by_cat = {"removed": set(), "fix_adjacent": set(), "context": set(), "edge_changed": set()}
        all_slice_lines = set()
        
        for n in G_vuln.nodes():
            line = G_vuln.nodes[n].get("LINE_NUMBER")
            if line is None or str(line) == "":
                continue
            line = int(line)
            cat = G_vuln.nodes[n].get("diff", "context")
            if cat in lines_by_cat:
                lines_by_cat[cat].add(line)
            all_slice_lines.add(line)
        
        gt = r["ground_truth"]["total_changed"]
        removed_lines = len(lines_by_cat["removed"])
        fixadj_lines = len(lines_by_cat["fix_adjacent"])
        context_lines = len(lines_by_cat["context"]) + len(lines_by_cat["edge_changed"])
        slice_lines = len(all_slice_lines)
        
        print(f"{idx:<3} {cve:<20} {slice_lines:<9} {removed_lines:<11} {fixadj_lines:<11} {context_lines:<11} {gt:<6}")
    except Exception as e:
        print(f"{idx:<3} {cve:<20} ERROR: {e}")
