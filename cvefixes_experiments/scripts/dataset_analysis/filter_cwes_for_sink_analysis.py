"""
Filter CVEfixes dataset for CWEs suited to sink-based vulnerability detection.

Target CWEs (chosen because the vulnerability is locatable by code structure):
  - CWE-190: Integer Overflow or Wraparound
  - CWE-121: Stack-based Buffer Overflow
  - CWE-122: Heap-based Buffer Overflow
  - CWE-415: Double Free
  - CWE-416: Use After Free
  - CWE-787: Out-of-bounds Write
  - CWE-843: Type Confusion
  - CWE-129: Improper Validation of Array Index
  - CWE-125: Out-of-bounds Read
  - CWE-200: Exposure of Sensitive Information
  - CWE-284: Improper Access Control
  - CWE-264: Permissions, Privileges, and Access Controls
  - CWE-476: NULL Pointer Dereference

Strategy:
  1. First, filter from the existing cvefixes_code_extraction.json (entries
     already have paired code_before/code_after from graphml_cvefixes).
  2. Then, query the CVEfixes SQLite DB directly to find additional paired
     methods for CWEs that are underrepresented in the extraction (CWE-121,
     CWE-122).
  3. Write results split per-CWE into separate files for fast loading and
     easy resumption of downstream processing.

Output:
  cvefixes_experiments/data/sink_cwes/
    metadata.json        — summary + reproducibility info
    CWE-190.json         — all entries for CWE-190
    CWE-121.json         — all entries for CWE-121
    CWE-122.json         — all entries for CWE-122
    CWE-415.json         — all entries for CWE-415
    CWE-416.json         — all entries for CWE-416
"""

import difflib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ──
DB_PATH = Path("data/cvefixes/CVEfixes.db")
EXTRACTION_PATH = Path("cvefixes_experiments/data/cvefixes_code_extraction.json")
OUTPUT_DIR = Path("cvefixes_experiments/data/sink_cwes")

TARGET_CWES = {
    "CWE-190": "Integer Overflow or Wraparound",
    "CWE-121": "Stack-based Buffer Overflow",
    "CWE-122": "Heap-based Buffer Overflow",
    "CWE-415": "Double Free",
    "CWE-416": "Use After Free",
    "CWE-787": "Out-of-bounds Write",
    "CWE-843": "Access of Resource Using Incompatible Type (Type Confusion)",
    "CWE-129": "Improper Validation of Array Index",
    "CWE-125": "Out-of-bounds Read",
    "CWE-200": "Exposure of Sensitive Information to an Unauthorized Actor",
    "CWE-284": "Improper Access Control",
    "CWE-264": "Permissions, Privileges, and Access Controls",
    "CWE-476": "NULL Pointer Dereference",
}

# Only C/C++ code (Joern-compatible)
LANGUAGES = {"C", "C++"}


def compute_changes(code_before: str, code_after: str) -> dict:
    """Compute diff statistics between before and after code."""
    before_lines = code_before.splitlines() if code_before else []
    after_lines = code_after.splitlines() if code_after else []
    diff = list(difflib.unified_diff(before_lines, after_lines, lineterm=""))

    lines_added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    lines_removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    matcher = difflib.SequenceMatcher(None, code_before or "", code_after or "")

    return {
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "lines_before": len(before_lines),
        "lines_after": len(after_lines),
        "similarity_ratio": round(matcher.ratio(), 4),
    }


def filter_from_extraction() -> dict[str, list[dict]]:
    """Filter existing extraction JSON for target CWEs."""
    print(f"[1] Loading existing extraction: {EXTRACTION_PATH}")
    with open(EXTRACTION_PATH) as f:
        data = json.load(f)

    by_cwe: dict[str, list[dict]] = defaultdict(list)
    seen_keys: set[tuple] = set()

    for entry in data["entries"]:
        # Must have both code sides and be C/C++
        if not entry.get("code_before") or not entry.get("code_after"):
            continue
        if entry.get("programming_language") not in LANGUAGES:
            continue

        entry_cwes = {c["cwe_id"] for c in entry["cwe"]}
        matching = entry_cwes & set(TARGET_CWES.keys())
        if not matching:
            continue

        # Dedup key: (cve_id, file_change_id, method_name)
        key = (entry["cve_id"], entry.get("file_change_id", ""), entry.get("method_name", ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Assign to primary matching CWE (first in our priority order)
        for cwe_id in TARGET_CWES:
            if cwe_id in matching:
                by_cwe[cwe_id].append(entry)
                break

    for cwe_id in TARGET_CWES:
        print(f"    {cwe_id}: {len(by_cwe[cwe_id])} entries from extraction")

    return by_cwe, seen_keys


def extract_from_db(existing_keys: set[tuple]) -> dict[str, list[dict]]:
    """
    Query DB directly for target CWEs to find entries not in the extraction.
    Pairs methods by (file_change_id, name) with before_change True/False.

    Uses batched lookups to avoid per-entry queries on the 277K-row method_change table.
    """
    print(f"\n[2] Querying database for additional entries: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Pre-load all CWE classifications and CVE info for target CWEs
    print("    Loading CWE classifications...")
    cwe_placeholders = ",".join("?" * len(TARGET_CWES))
    target_cve_ids = set()
    cve_to_target_cwe: dict[str, str] = {}  # cve_id -> primary target CWE

    rows = conn.execute(
        f"SELECT cve_id, cwe_id FROM cwe_classification WHERE cwe_id IN ({cwe_placeholders})",
        list(TARGET_CWES.keys()),
    ).fetchall()
    for row in rows:
        target_cve_ids.add(row["cve_id"])
        # Store primary CWE (first match in priority order)
        if row["cve_id"] not in cve_to_target_cwe:
            cve_to_target_cwe[row["cve_id"]] = row["cwe_id"]

    print(f"    Found {len(target_cve_ids)} CVEs with target CWEs")

    # Load all CWE info per CVE (for the output)
    cve_to_cwes: dict[str, list[dict]] = defaultdict(list)
    all_cwe_rows = conn.execute(
        "SELECT cc.cve_id, cc.cwe_id, cwe.cwe_name "
        "FROM cwe_classification cc LEFT JOIN cwe ON cc.cwe_id = cwe.cwe_id"
    ).fetchall()
    for row in all_cwe_rows:
        if row["cve_id"] in target_cve_ids:
            cve_to_cwes[row["cve_id"]].append({"cwe_id": row["cwe_id"], "cwe_name": row["cwe_name"]})

    # Load CVE metadata
    print("    Loading CVE metadata...")
    cve_info: dict[str, dict] = {}
    cve_rows = conn.execute(
        "SELECT cve_id, published_date, severity, cvss3_base_score, cvss3_base_severity "
        "FROM cve"
    ).fetchall()
    for row in cve_rows:
        if row["cve_id"] in target_cve_ids:
            cve_info[row["cve_id"]] = dict(row)

    # Find file_change_ids for target CVEs (C/C++ only)
    print("    Finding file changes...")
    fcid_to_meta: dict[str, dict] = {}
    # Query in batches
    cve_list = list(target_cve_ids)
    batch_size = 500
    for i in range(0, len(cve_list), batch_size):
        batch = cve_list[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))
        fc_rows = conn.execute(
            f"""
            SELECT fc.file_change_id, fc.filename, fc.programming_language, f.cve_id
            FROM file_change fc
            JOIN fixes f ON fc.hash = f.hash
            WHERE f.cve_id IN ({placeholders})
            AND fc.programming_language IN ('C', 'C++')
            """,
            batch,
        ).fetchall()
        for row in fc_rows:
            fcid_to_meta[row["file_change_id"]] = dict(row)

    print(f"    Found {len(fcid_to_meta)} file changes")

    # Load method_change entries for those file_change_ids
    print("    Loading method changes (this may take a moment)...")
    fcid_list = list(fcid_to_meta.keys())
    # Group methods by (file_change_id, name)
    groups: dict[tuple, dict] = defaultdict(lambda: {"before": None, "after": None})

    for i in range(0, len(fcid_list), batch_size):
        batch = fcid_list[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))
        mc_rows = conn.execute(
            f"""
            SELECT method_change_id, file_change_id, name, signature, code, before_change
            FROM method_change
            WHERE file_change_id IN ({placeholders})
            """,
            batch,
        ).fetchall()
        for row in mc_rows:
            key = (row["file_change_id"], row["name"])
            if row["before_change"] == "True":
                groups[key]["before"] = dict(row)
            else:
                groups[key]["after"] = dict(row)

    conn.close()
    print(f"    Grouped into {len(groups)} method pairs")

    # Build entries from complete pairs
    by_cwe: dict[str, list[dict]] = defaultdict(list)

    for (fcid, name), group in groups.items():
        if group["before"] is None or group["after"] is None:
            continue

        code_before = group["before"]["code"]
        code_after = group["after"]["code"]
        if not code_before or not code_after:
            continue

        meta = fcid_to_meta.get(fcid)
        if meta is None:
            continue

        cve_id = meta["cve_id"]
        dedup_key = (cve_id, fcid, name)
        if dedup_key in existing_keys:
            continue
        existing_keys.add(dedup_key)

        # Determine which target CWE this belongs to
        cwe_id = cve_to_target_cwe.get(cve_id)
        if cwe_id is None:
            continue

        entry = {
            "cve_id": cve_id,
            "cwe": cve_to_cwes.get(cve_id, []),
            "cve_severity": cve_info.get(cve_id, {}).get("severity"),
            "cvss3_base_score": cve_info.get(cve_id, {}).get("cvss3_base_score"),
            "published_date": cve_info.get(cve_id, {}).get("published_date"),
            "file_change_id": fcid,
            "filename": meta["filename"],
            "programming_language": meta["programming_language"],
            "method_name": name,
            "method_signature": group["before"]["signature"],
            "code_before": code_before,
            "code_after": code_after,
            "changes": compute_changes(code_before, code_after),
            "source": "db_extraction",
        }
        by_cwe[cwe_id].append(entry)

    for cwe_id in TARGET_CWES:
        print(f"    {cwe_id}: {len(by_cwe[cwe_id])} new entries from DB")

    return by_cwe


def write_output(by_cwe: dict[str, list[dict]]):
    """Write per-CWE JSON files and metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    cwe_summary = {}

    for cwe_id in TARGET_CWES:
        entries = by_cwe.get(cwe_id, [])
        total += len(entries)

        # Sort by CVE ID for reproducibility
        entries.sort(key=lambda e: (e["cve_id"], e.get("method_name", "")))

        output_file = OUTPUT_DIR / f"{cwe_id}.json"
        cwe_data = {
            "cwe_id": cwe_id,
            "cwe_name": TARGET_CWES[cwe_id],
            "entry_count": len(entries),
            "entries": entries,
        }
        with open(output_file, "w") as f:
            json.dump(cwe_data, f, indent=2)

        cwe_summary[cwe_id] = {
            "cwe_name": TARGET_CWES[cwe_id],
            "entry_count": len(entries),
            "unique_cves": len(set(e["cve_id"] for e in entries)),
            "file": str(output_file),
        }
        print(f"    {cwe_id} ({TARGET_CWES[cwe_id]}): {len(entries)} entries → {output_file}")

    # Write metadata
    metadata = {
        "description": "CVEfixes entries filtered for sink-based vulnerability detection",
        "rationale": (
            "These CWEs have structurally identifiable sinks (buffer writes, "
            "free calls, arithmetic ops) where the vulnerability fix is typically "
            "at or near the dangerous operation, making them suitable for "
            "Joern CPG-based sink detection."
        ),
        "target_cwes": TARGET_CWES,
        "total_entries": total,
        "per_cwe": cwe_summary,
        "filters_applied": {
            "languages": list(LANGUAGES),
            "requires_code_before": True,
            "requires_code_after": True,
            "deduplicated_by": ["cve_id", "file_change_id", "method_name"],
        },
        "sources": {
            "primary": str(EXTRACTION_PATH),
            "secondary": str(DB_PATH),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
    }
    metadata_file = OUTPUT_DIR / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n    Total: {total} entries across {len(TARGET_CWES)} CWEs")
    print(f"    Metadata: {metadata_file}")


def main():
    print("=" * 70)
    print("Filter CVEfixes for Sink-Based CWEs")
    print("=" * 70)

    # Step 1: Filter from existing extraction
    by_cwe, seen_keys = filter_from_extraction()

    # Step 2: Augment from DB (especially for CWE-121, CWE-122)
    db_additions = extract_from_db(seen_keys)

    # Merge
    for cwe_id, entries in db_additions.items():
        by_cwe[cwe_id].extend(entries)

    # Step 3: Write output
    print(f"\n[3] Writing output to {OUTPUT_DIR}/")
    write_output(by_cwe)

    print("\nDone.")


if __name__ == "__main__":
    main()
