#!/usr/bin/env python3
"""
Verify correctness of CVEfixes data loading for the retrieval experiment.

Checks:
  1. Database connectivity and query results
  2. GraphML directory structure (symlinks resolved, files exist)
  3. CPG loading success/failure per entry
  4. Graph diff computation validity
  5. Summary of errors with actionable diagnostics

Usage:
    python -m cvefixes_experiments.scripts.pipeline_verification.verify_cvefixes_loading \
      --config config.yaml \
      --graphml-root graphml_selected_cves \
      --db-path data/cvefixes/CVEfixes.db
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def check_symlinks(graphml_root: str) -> dict:
    """Check if graphml_root is a symlink and validate its target."""
    root = Path(graphml_root)
    info = {
        "path": str(root.absolute()),
        "exists": root.exists(),
        "is_symlink": root.is_symlink(),
        "symlink_target": None,
        "target_exists": None,
    }
    if root.is_symlink():
        target = root.resolve()
        info["symlink_target"] = str(target)
        info["target_exists"] = target.exists()
    return info


def check_database(db_path: str, languages: list[str], target_cwes: list[str] | None) -> dict:
    """Verify database access and count available rows."""
    db = Path(db_path)
    if not db.exists():
        return {"error": f"Database not found: {db_path}"}

    conn = sqlite3.connect(str(db), timeout=30)
    conn.row_factory = sqlite3.Row

    placeholders = ", ".join("?" for _ in languages)
    query = f"""\
SELECT
    m_after.method_change_id,
    m_after.name AS func_name,
    m_after.code AS code_after,
    m_before.code AS code_before,
    f.programming_language,
    cv.cve_id,
    cc.cwe_id,
    f.filename
FROM method_change m_after
JOIN method_change m_before
    ON m_before.file_change_id = m_after.file_change_id
    AND m_before.name = m_after.name
    AND m_before.before_change = 'True'
JOIN file_change f ON m_after.file_change_id = f.file_change_id
JOIN commits c ON f.hash = c.hash
JOIN fixes fx ON c.hash = fx.hash
JOIN cve cv ON fx.cve_id = cv.cve_id
LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
WHERE m_after.before_change = 'False'
  AND f.programming_language IN ({placeholders})
  AND m_before.code IS NOT NULL AND m_before.code != ''
  AND m_after.code IS NOT NULL AND m_after.code != ''
"""

    cursor = conn.execute(query, languages)
    rows = cursor.fetchall()
    conn.close()

    total = len(rows)
    cwe_dist = Counter(r["cwe_id"] for r in rows)

    if target_cwes:
        rows = [r for r in rows if r["cwe_id"] in target_cwes]

    return {
        "total_rows": total,
        "after_cwe_filter": len(rows),
        "cwe_distribution": dict(cwe_dist.most_common(20)),
        "unique_cves": len({r["cve_id"] for r in rows}),
        "target_cwes": target_cwes,
        "sample_row_ids": [
            f"{r['cve_id']}_m{r['method_change_id']}" for r in rows[:5]
        ],
    }


def check_graphml_directories(
    graphml_root: str,
    max_check: int = 0,
) -> dict:
    """Check graphml directories by scanning the filesystem directly (no DB query)."""
    root = Path(graphml_root)
    if not root.exists():
        return {"error": f"graphml_root does not exist: {graphml_root}"}

    entries = sorted(e for e in os.listdir(root) if (root / e).is_dir() or (root / e).is_symlink())

    if max_check > 0:
        entries = entries[:max_check]

    results = {
        "checked": len(entries),
        "dir_exists": 0,
        "before_exists": 0,
        "after_exists": 0,
        "before_has_export_xml": 0,
        "after_has_export_xml": 0,
        "both_valid": 0,
        "broken_symlinks": [],
        "missing_export_xml": [],
        "empty_dirs": [],
    }

    for row_id in entries:
        base = root / row_id

        # Check if broken symlink
        if base.is_symlink() and not base.exists():
            results["broken_symlinks"].append(row_id)
            continue

        results["dir_exists"] += 1

        # Check before/after subdirs
        before_dir = base / "original" / "before" / "graph"
        after_dir = base / "original" / "after" / "graph"

        before_ok = False
        after_ok = False

        if before_dir.exists():
            results["before_exists"] += 1
            export_files = glob.glob(str(before_dir / "**" / "export.xml"), recursive=True)
            if export_files:
                results["before_has_export_xml"] += 1
                before_ok = True
            else:
                results["missing_export_xml"].append(f"{row_id}/original/before")
        elif before_dir.is_symlink():
            results["broken_symlinks"].append(f"{row_id}/original/before")

        if after_dir.exists():
            results["after_exists"] += 1
            export_files = glob.glob(str(after_dir / "**" / "export.xml"), recursive=True)
            if export_files:
                results["after_has_export_xml"] += 1
                after_ok = True
            else:
                results["missing_export_xml"].append(f"{row_id}/original/after")
        elif after_dir.is_symlink():
            results["broken_symlinks"].append(f"{row_id}/original/after")

        if not before_ok and not after_ok:
            results["empty_dirs"].append(row_id)

        if before_ok and after_ok:
            results["both_valid"] += 1

    # Truncate lists for readability
    for key in ("broken_symlinks", "missing_export_xml", "empty_dirs"):
        total = len(results[key])
        if total > 20:
            results[key] = results[key][:20] + [f"... and {total - 20} more"]

    return results


def check_cpg_loading(
    graphml_root: str,
    max_load: int = 10,
) -> dict:
    """Actually attempt to load CPGs and compute graph diffs for a sample of existing dirs."""
    try:
        import networkx as nx
        from src.data.pipeline import cpg_dir_for, load_cpg_dir, compute_graph_diff
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    root = Path(graphml_root)
    if not root.exists():
        return {"error": f"graphml_root does not exist: {graphml_root}"}

    # Scan filesystem for entries that look like they have both before+after
    entries = sorted(e for e in os.listdir(root) if (root / e).is_dir())

    loaded = 0
    errors = []
    successes = []

    for row_id in entries:
        if loaded >= max_load:
            break

        try:
            before_dir = cpg_dir_for(graphml_root, cve_id=row_id, variant="original", version="before")
            after_dir = cpg_dir_for(graphml_root, cve_id=row_id, variant="original", version="after")

            G_before = load_cpg_dir(before_dir)
            G_after = load_cpg_dir(after_dir)

            if G_before.number_of_nodes() == 0:
                errors.append({"row_id": row_id, "error": "G_before has 0 nodes"})
                continue
            if G_after.number_of_nodes() == 0:
                errors.append({"row_id": row_id, "error": "G_after has 0 nodes"})
                continue

            G_vuln = compute_graph_diff(G_before, G_after)

            successes.append({
                "row_id": row_id,
                "before_nodes": G_before.number_of_nodes(),
                "after_nodes": G_after.number_of_nodes(),
                "vuln_nodes": G_vuln.number_of_nodes(),
            })
            loaded += 1

        except FileNotFoundError as e:
            errors.append({"row_id": row_id, "error": f"FileNotFoundError: {e}"})
        except Exception as e:
            errors.append({"row_id": row_id, "error": f"{type(e).__name__}: {e}"})

    return {
        "attempted": loaded + len(errors),
        "successful_loads": len(successes),
        "failed_loads": len(errors),
        "successes": successes[:5],
        "errors": errors[:20],
    }


def check_graphml_root_contents(graphml_root: str) -> dict:
    """List what's actually in the graphml root directory."""
    root = Path(graphml_root)
    if not root.exists():
        return {"error": f"Directory does not exist: {graphml_root}"}

    entries = sorted(os.listdir(root))
    subdirs = [e for e in entries if (root / e).is_dir()]
    symlinks = [e for e in entries if (root / e).is_symlink()]
    broken = [e for e in symlinks if not (root / e).exists()]

    return {
        "total_entries": len(entries),
        "directories": len(subdirs),
        "symlinks": len(symlinks),
        "broken_symlinks": len(broken),
        "broken_symlink_samples": broken[:10],
        "sample_entries": subdirs[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Verify CVEfixes data loading pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--graphml-root", default="graphml_selected_cves")
    parser.add_argument("--db-path", default="data/cvefixes/CVEfixes.db")
    parser.add_argument("--languages", nargs="+", default=["C", "C++"])
    parser.add_argument("--target-cwes", nargs="*", default=None)
    parser.add_argument("--max-check", type=int, default=0, help="Max entries to check dirs (0=all)")
    parser.add_argument("--max-load", type=int, default=10, help="Max entries to actually load CPGs")
    parser.add_argument("--skip-load", action="store_true", help="Skip actual CPG loading test")
    args = parser.parse_args()

    print("=" * 70)
    print("CVEfixes Data Loading Verification")
    print("=" * 70)

    # 1. Check symlinks
    print("\n[1/5] Checking graphml_root path...")
    symlink_info = check_symlinks(args.graphml_root)
    for k, v in symlink_info.items():
        print(f"  {k}: {v}")

    # 2. Check directory contents
    print("\n[2/5] Checking graphml_root contents...")
    contents = check_graphml_root_contents(args.graphml_root)
    for k, v in contents.items():
        if isinstance(v, list):
            print(f"  {k}: {v[:5]}{'...' if len(v) > 5 else ''}")
        else:
            print(f"  {k}: {v}")

    # 3. Check database
    # print("\n[3/5] Checking database...")
    # db_info = check_database(args.db_path, args.languages, args.target_cwes)
    # for k, v in db_info.items():
    #     print(f"  {k}: {v}")

    # 4. Check directory structure
    print("\n[4/5] Checking graphml directory structure (filesystem scan)...")
    dir_info = check_graphml_directories(
        args.graphml_root,
        max_check=args.max_check,
    )
    for k, v in dir_info.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [{len(v)} items] first 5: {v[:5]}")
        else:
            print(f"  {k}: {v}")

    # 5. Actual CPG loading
    if not args.skip_load:
        print(f"\n[5/5] Attempting to load {args.max_load} CPGs...")
        load_info = check_cpg_loading(
            args.graphml_root,
            max_load=args.max_load,
        )
        for k, v in load_info.items():
            if isinstance(v, list):
                print(f"  {k}:")
                for item in v:
                    print(f"    {item}")
            else:
                print(f"  {k}: {v}")
    else:
        print("\n[5/5] Skipped CPG loading (--skip-load)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    issues = []
    if symlink_info.get("is_symlink") and not symlink_info.get("target_exists"):
        issues.append("CRITICAL: graphml_root symlink target does not exist!")
    if contents.get("broken_symlinks", 0) > 0:
        issues.append(f"WARNING: {contents['broken_symlinks']} broken symlinks in graphml_root")
    if "dir_exists" in dir_info:
        missing = dir_info["checked"] - dir_info["dir_exists"]
        if missing > 0:
            issues.append(f"WARNING: {missing}/{dir_info['checked']} entries have no directory")
        both_valid = dir_info["both_valid"]
        issues.append(f"INFO: {both_valid}/{dir_info['checked']} entries have valid before+after CPGs")

    if not issues:
        print("  All checks passed!")
    else:
        for issue in issues:
            print(f"  {issue}")


if __name__ == "__main__":
    main()
