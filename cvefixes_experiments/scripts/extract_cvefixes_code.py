"""
Extract before/after code from CVEfixes database for entries present in graphml_cvefixes.
Outputs a JSON with CVE, CWE, code before/after, and change statistics.

Includes filtering to match AutoPatch dataset by CWE or CVE.
"""

import json
import os
import sqlite3
import difflib
from pathlib import Path
from collections import defaultdict


DB_PATH = "data/cvefixes/CVEfixes.db"
GRAPHML_DIR = "graphml_cvefixes"
AUTOPATCH_DIR = "CVE-list"
OUTPUT_PATH = "cvefixes_experiments/data/cvefixes_code_extraction.json"

# Mapping from AutoPatch CWE names to CWE IDs in CVEfixes
AUTOPATCH_CWE_NAME_TO_ID = {
    "Use After Free": "CWE-416",
    "Race Condition": "CWE-362",
    "Access of Resource Using Incompatible Type ('Type Confusion')": "CWE-843",
    "Out-of-bounds Write": "CWE-787",
    "Integer Overflow or Wraparound": "CWE-190",
    "NULL Pointer Dereference": "CWE-476",
    "Unchecked Return Value": "CWE-252",
    "Incorrect Check of Function Return Value": "CWE-253",
    "Expired Pointer Dereference": "CWE-825",
    "Operation on a Resource after Expiration or Release": "CWE-672",
    "Assignment of a Fixed Address to a Pointer": "CWE-587",
    "Improper Locking": "CWE-667",
    "Return of Stack Variable Address After Scope": "CWE-562",
    "Deadlock": "CWE-833",
    "Integer Underflow": "CWE-191",
    "Improper Control of a Resource Through Its Lifetime": "CWE-664",
    "Memory Leak": "CWE-401",
    "Incorrect Type Conversion or Cast": "CWE-704",
    "Use of Uninitialized Variable": "CWE-457",
    "Improper Input Validation": "CWE-20",
    "Business Logic Errors": "CWE-840",
    "Improper Validation of Array Index": "CWE-129",
    "Uncontrolled Resource Consumption ('Resource Exhaustion')": "CWE-400",
}


def get_graphml_entries(graphml_dir: str) -> list[dict]:
    """Parse directory names to extract CVE IDs and method_change_ids."""
    entries = []
    for dirname in os.listdir(graphml_dir):
        if not os.path.isdir(os.path.join(graphml_dir, dirname)):
            continue
        # Format: CVE-XXXX-XXXXX_mXXXXXXXXXXXXXX
        parts = dirname.rsplit("_m", 1)
        if len(parts) == 2:
            cve_id = parts[0]
            method_change_id = parts[1]
            entries.append({
                "cve_id": cve_id,
                "method_change_id": method_change_id,
                "dirname": dirname,
            })
    return entries


def compute_diff_stats(code_before: str, code_after: str) -> dict:
    """Compute statistics about the changes between before and after code."""
    before_lines = code_before.splitlines(keepends=True) if code_before else []
    after_lines = code_after.splitlines(keepends=True) if code_after else []

    diff = list(difflib.unified_diff(before_lines, after_lines, lineterm=""))

    lines_added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    lines_removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    # Compute similarity ratio
    matcher = difflib.SequenceMatcher(None, code_before or "", code_after or "")
    similarity = matcher.ratio()

    return {
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "lines_before": len(before_lines),
        "lines_after": len(after_lines),
        "similarity_ratio": round(similarity, 4),
        "diff": "".join(diff[:200]) if diff else "",  # first 200 lines of diff
    }


def main():
    # 1. Get all method_change_ids from graphml_cvefixes directories
    print(f"Scanning {GRAPHML_DIR} for entries...")
    entries = get_graphml_entries(GRAPHML_DIR)
    print(f"Found {len(entries)} directory entries")

    # Collect unique method_change_ids
    method_ids = set(e["method_change_id"] for e in entries)
    print(f"Unique method_change_ids: {len(method_ids)}")

    # 2. Query the database
    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build lookup of method_change data
    # Process in batches to avoid too-long SQL
    method_data = {}
    method_id_list = list(method_ids)
    batch_size = 500

    print("Querying method_change table...")
    for i in range(0, len(method_id_list), batch_size):
        batch = method_id_list[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cursor.execute(
            f"""
            SELECT method_change_id, file_change_id, name, signature, parameters,
                   start_line, end_line, code, nloc, complexity, token_count,
                   top_nesting_level, before_change
            FROM method_change
            WHERE method_change_id IN ({placeholders})
            """,
            batch,
        )
        for row in cursor.fetchall():
            method_data[row["method_change_id"]] = dict(row)

    print(f"Retrieved {len(method_data)} method_change records from DB")

    # 3. For each method, find its pair (same file_change_id, same name, opposite before_change)
    # Group by (file_change_id, name) to find before/after pairs
    file_change_groups = defaultdict(list)
    for mid, mdata in method_data.items():
        key = (mdata["file_change_id"], mdata["name"])
        file_change_groups[key].append(mdata)

    # Also query methods not in our set but that are pairs of those in our set
    print("Finding paired methods (before/after)...")
    file_change_ids = set(m["file_change_id"] for m in method_data.values())
    paired_methods = {}

    for i in range(0, len(list(file_change_ids)), batch_size):
        batch = list(file_change_ids)[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cursor.execute(
            f"""
            SELECT method_change_id, file_change_id, name, signature, parameters,
                   start_line, end_line, code, nloc, complexity, token_count,
                   top_nesting_level, before_change
            FROM method_change
            WHERE file_change_id IN ({placeholders})
            """,
            batch,
        )
        for row in cursor.fetchall():
            paired_methods[row["method_change_id"]] = dict(row)

    # Rebuild groups with all paired methods
    file_change_groups = defaultdict(list)
    for mid, mdata in paired_methods.items():
        key = (mdata["file_change_id"], mdata["name"])
        file_change_groups[key].append(mdata)

    # 4. Get CVE info: file_change -> fixes -> cve -> cwe_classification
    print("Querying CVE and CWE data...")
    # file_change -> hash -> fixes.cve_id
    file_change_to_cve = {}
    for i in range(0, len(list(file_change_ids)), batch_size):
        batch = list(file_change_ids)[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cursor.execute(
            f"""
            SELECT fc.file_change_id, fc.filename, fc.old_path, fc.new_path,
                   fc.programming_language, fc.num_lines_added, fc.num_lines_deleted,
                   fixes.cve_id
            FROM file_change fc
            JOIN fixes ON fc.hash = fixes.hash
            WHERE fc.file_change_id IN ({placeholders})
            """,
            batch,
        )
        for row in cursor.fetchall():
            file_change_to_cve[row["file_change_id"]] = dict(row)

    # Get CWE classifications for all CVEs
    cve_ids = set(v["cve_id"] for v in file_change_to_cve.values() if v.get("cve_id"))
    cve_to_cwe = defaultdict(list)
    for i in range(0, len(list(cve_ids)), batch_size):
        batch = list(cve_ids)[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cursor.execute(
            f"""
            SELECT cc.cve_id, cc.cwe_id, cwe.cwe_name
            FROM cwe_classification cc
            LEFT JOIN cwe ON cc.cwe_id = cwe.cwe_id
            WHERE cc.cve_id IN ({placeholders})
            """,
            batch,
        )
        for row in cursor.fetchall():
            cve_to_cwe[row["cve_id"]].append({
                "cwe_id": row["cwe_id"],
                "cwe_name": row["cwe_name"],
            })

    # Get CVE descriptions and severity
    cve_info = {}
    for i in range(0, len(list(cve_ids)), batch_size):
        batch = list(cve_ids)[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cursor.execute(
            f"""
            SELECT cve_id, published_date, severity, cvss3_base_score,
                   cvss3_base_severity, description
            FROM cve
            WHERE cve_id IN ({placeholders})
            """,
            batch,
        )
        for row in cursor.fetchall():
            cve_info[row["cve_id"]] = dict(row)

    conn.close()

    # 5. Build the output JSON
    print("Building output JSON...")
    results = []
    processed_pairs = set()

    for entry in entries:
        mid = entry["method_change_id"]
        if mid not in method_data:
            continue

        mdata = method_data[mid]
        key = (mdata["file_change_id"], mdata["name"])

        # Avoid processing the same pair twice
        if key in processed_pairs:
            continue
        processed_pairs.add(key)

        # Find before and after
        group = file_change_groups.get(key, [])
        code_before = None
        code_after = None
        method_before = None
        method_after = None

        for m in group:
            if m["before_change"] == "True":
                code_before = m["code"]
                method_before = m
            else:
                code_after = m["code"]
                method_after = m

        # Get CVE/CWE info
        fc_info = file_change_to_cve.get(mdata["file_change_id"], {})
        cve_id = fc_info.get("cve_id", entry["cve_id"])
        cwes = cve_to_cwe.get(cve_id, [])
        cve_detail = cve_info.get(cve_id, {})

        # Compute diff stats
        diff_stats = compute_diff_stats(code_before or "", code_after or "")

        record = {
            "cve_id": cve_id,
            "cwe": cwes,
            "cve_description": cve_detail.get("description", ""),
            "cve_severity": cve_detail.get("severity", ""),
            "cvss3_base_score": cve_detail.get("cvss3_base_score", ""),
            "cvss3_base_severity": cve_detail.get("cvss3_base_severity", ""),
            "published_date": cve_detail.get("published_date", ""),
            "file_change_id": mdata["file_change_id"],
            "filename": fc_info.get("filename", ""),
            "programming_language": fc_info.get("programming_language", ""),
            "method_name": mdata["name"],
            "method_signature": mdata.get("signature", ""),
            "code_before": code_before,
            "code_after": code_after,
            "changes": {
                "lines_added": diff_stats["lines_added"],
                "lines_removed": diff_stats["lines_removed"],
                "lines_before": diff_stats["lines_before"],
                "lines_after": diff_stats["lines_after"],
                "similarity_ratio": diff_stats["similarity_ratio"],
            },
        }
        results.append(record)

    # 6. Compute summary statistics
    print("Computing statistics...")
    total_entries = len(results)
    unique_cves = set(r["cve_id"] for r in results)
    unique_cwes = set()
    for r in results:
        for cwe in r["cwe"]:
            unique_cwes.add(cwe["cwe_id"])

    cwe_distribution = defaultdict(int)
    for r in results:
        for cwe in r["cwe"]:
            cwe_distribution[cwe["cwe_id"]] += 1

    severity_distribution = defaultdict(int)
    for r in results:
        sev = r.get("cvss3_base_severity") or r.get("cve_severity") or "unknown"
        severity_distribution[sev] += 1

    avg_lines_added = sum(r["changes"]["lines_added"] for r in results) / max(total_entries, 1)
    avg_lines_removed = sum(r["changes"]["lines_removed"] for r in results) / max(total_entries, 1)
    avg_similarity = sum(r["changes"]["similarity_ratio"] for r in results) / max(total_entries, 1)

    lang_distribution = defaultdict(int)
    for r in results:
        lang_distribution[r.get("programming_language") or "unknown"] += 1

    summary = {
        "total_method_pairs": total_entries,
        "unique_cves": len(unique_cves),
        "unique_cwes": len(unique_cwes),
        "cwe_distribution": dict(sorted(cwe_distribution.items(), key=lambda x: -x[1])),
        "severity_distribution": dict(severity_distribution),
        "language_distribution": dict(sorted(lang_distribution.items(), key=lambda x: -x[1])),
        "avg_lines_added": round(avg_lines_added, 2),
        "avg_lines_removed": round(avg_lines_removed, 2),
        "avg_similarity_ratio": round(avg_similarity, 4),
    }

    output = {
        "summary": summary,
        "entries": results,
    }

    # 7. Write output
    print(f"Writing output to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total method before/after pairs: {total_entries}")
    print(f"Unique CVEs: {len(unique_cves)}")
    print(f"Unique CWEs: {len(unique_cwes)}")
    print(f"\nTop 10 CWEs:")
    for cwe_id, count in sorted(cwe_distribution.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cwe_id}: {count}")
    print(f"\nSeverity distribution:")
    for sev, count in sorted(severity_distribution.items(), key=lambda x: -x[1]):
        print(f"  {sev}: {count}")
    print(f"\nLanguage distribution:")
    for lang, count in sorted(lang_distribution.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")
    print(f"\nChange statistics (averages):")
    print(f"  Lines added: {avg_lines_added:.2f}")
    print(f"  Lines removed: {avg_lines_removed:.2f}")
    print(f"  Similarity ratio: {avg_similarity:.4f}")
    print(f"\nOutput written to: {OUTPUT_PATH}")


def get_autopatch_info() -> tuple[set, set]:
    """
    Read AutoPatch CVE-list directory to get unique CVE IDs and CWE IDs.
    Returns (autopatch_cve_ids, autopatch_cwe_ids).
    """
    autopatch_cves = set()
    autopatch_cwes = set()
    autopatch_cwe_names = set()

    for dirname in os.listdir(AUTOPATCH_DIR):
        dirpath = os.path.join(AUTOPATCH_DIR, dirname)
        if not os.path.isdir(dirpath):
            continue
        info_path = os.path.join(dirpath, "info.json")
        if not os.path.exists(info_path):
            continue
        with open(info_path) as f:
            info = json.load(f)

        cve_id = info.get("cve_id", "")
        cwe_name = info.get("cwe_id", "")  # field is named cwe_id but contains name

        if cve_id:
            autopatch_cves.add(cve_id)
        if cwe_name:
            autopatch_cwe_names.add(cwe_name)
            cwe_id = AUTOPATCH_CWE_NAME_TO_ID.get(cwe_name)
            if cwe_id:
                autopatch_cwes.add(cwe_id)

    return autopatch_cves, autopatch_cwes, autopatch_cwe_names


def filter_by_cwe(entries: list[dict], autopatch_cwes: set) -> list[dict]:
    """Filter cvefixes entries to those sharing a CWE with AutoPatch."""
    filtered = []
    for entry in entries:
        entry_cwes = set(cwe["cwe_id"] for cwe in entry.get("cwe", []))
        if entry_cwes & autopatch_cwes:
            filtered.append(entry)
    return filtered


def filter_by_cve(entries: list[dict], autopatch_cves: set) -> list[dict]:
    """Filter cvefixes entries to those sharing a CVE with AutoPatch."""
    filtered = []
    for entry in entries:
        if entry.get("cve_id") in autopatch_cves:
            filtered.append(entry)
    return filtered


def run_filtering_analysis():
    """
    Load the extracted JSON data, filter by AutoPatch CWEs and CVEs,
    and report sample sizes.
    """
    # Load existing extraction
    if not os.path.exists(OUTPUT_PATH):
        print(f"ERROR: {OUTPUT_PATH} not found. Run main() first.")
        return

    print(f"Loading {OUTPUT_PATH}...")
    with open(OUTPUT_PATH) as f:
        data = json.load(f)

    entries = data["entries"]
    total = len(entries)
    print(f"Total CVEfixes entries (in graphml_cvefixes): {total}")

    # Get AutoPatch info
    autopatch_cves, autopatch_cwes, autopatch_cwe_names = get_autopatch_info()
    print(f"\n=== AutoPatch Dataset ===")
    print(f"Unique AutoPatch CVEs: {len(autopatch_cves)}")
    print(f"Unique AutoPatch CWE names: {len(autopatch_cwe_names)}")
    print(f"Mapped to CWE IDs: {sorted(autopatch_cwes)}")

    # Filter by CWE
    cwe_filtered = filter_by_cwe(entries, autopatch_cwes)
    print(f"\n=== Filter by CWE (same CWEs as AutoPatch) ===")
    print(f"Sample size: {len(cwe_filtered)} / {total} "
          f"({100*len(cwe_filtered)/total:.1f}%)")
    cwe_filtered_cves = set(e["cve_id"] for e in cwe_filtered)
    print(f"Unique CVEs in filtered set: {len(cwe_filtered_cves)}")

    # CWE breakdown in filtered set
    cwe_counts = defaultdict(int)
    for e in cwe_filtered:
        for cwe in e.get("cwe", []):
            if cwe["cwe_id"] in autopatch_cwes:
                cwe_counts[cwe["cwe_id"]] += 1
    print(f"CWE breakdown:")
    for cwe_id, count in sorted(cwe_counts.items(), key=lambda x: -x[1]):
        name = next(
            (n for n, i in AUTOPATCH_CWE_NAME_TO_ID.items() if i == cwe_id), cwe_id
        )
        print(f"  {cwe_id} ({name}): {count}")

    # Filter by CVE
    cve_filtered = filter_by_cve(entries, autopatch_cves)
    print(f"\n=== Filter by CVE (same CVEs as AutoPatch) ===")
    print(f"Sample size: {len(cve_filtered)} / {total} "
          f"({100*len(cve_filtered)/total:.1f}%)")
    matched_cves = set(e["cve_id"] for e in cve_filtered)
    print(f"Matched CVEs: {sorted(matched_cves)}")
    unmatched = autopatch_cves - matched_cves
    print(f"AutoPatch CVEs NOT in CVEfixes: {len(unmatched)}")
    if unmatched:
        print(f"  {sorted(unmatched)}")

    # Assessment
    print(f"\n=== Assessment ===")
    print(f"AutoPatch entries: {len(autopatch_cves)} unique CVEs")
    print(f"CVEfixes entries matching same CVEs: {len(cve_filtered)} method pairs")
    print(f"CVEfixes entries matching same CWEs: {len(cwe_filtered)} method pairs")
    if len(cve_filtered) < 30:
        print(f"\n⚠ Only {len(cve_filtered)} entries match by CVE — likely NOT enough "
              f"for meaningful analysis using only same CVEs.")
        print(f"  Recommendation: Use CWE-based filtering ({len(cwe_filtered)} entries) "
              f"for a larger, representative sample.")
    else:
        print(f"\n✓ {len(cve_filtered)} entries match by CVE — may be sufficient "
              f"depending on the analysis requirements.")

    # Save filtered outputs
    cwe_output_path = "cvefixes_experiments/data/cvefixes_filtered_by_cwe.json"
    cve_output_path = "cvefixes_experiments/data/cvefixes_filtered_by_cve.json"

    with open(cwe_output_path, "w") as f:
        json.dump({
            "filter": "same_cwe_as_autopatch",
            "autopatch_cwes": sorted(autopatch_cwes),
            "sample_size": len(cwe_filtered),
            "entries": cwe_filtered,
        }, f, indent=2)
    print(f"\nCWE-filtered output: {cwe_output_path}")

    with open(cve_output_path, "w") as f:
        json.dump({
            "filter": "same_cve_as_autopatch",
            "autopatch_cves": sorted(autopatch_cves),
            "sample_size": len(cve_filtered),
            "entries": cve_filtered,
        }, f, indent=2)
    print(f"CVE-filtered output: {cve_output_path}")


if __name__ == "__main__":
    import sys
    if "--filter" in sys.argv:
        run_filtering_analysis()
    else:
        main()
        print("\n\nRun with --filter to see AutoPatch filtering analysis.")
