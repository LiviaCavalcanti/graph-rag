# Sink-Based CWE Dataset Report

**Dataset:** CVEfixes Filtered for Structural Vulnerability Detection  
**Date:** 2026-05-28  
**Script:** `cvefixes_experiments/scripts/dataset_analysis/filter_cwes_for_sink_analysis.py`  
**Output:** `cvefixes_experiments/data/sink_cwes/`

---

## 1. Rationale: Why These CWEs?

### 1.1 The Problem with Prior CWE Selection

The previous experiment (`exp_joern_native_query`) tested sink detection across a broad CWE set and achieved only **51% hit rate** (50/98). Miss analysis revealed that the majority of failures were **structural mismatches** between our detection approach and the vulnerability class:

| CWE | Hit Rate | Failure Mode |
|-----|----------|--------------|
| CWE-400 (Resource Consumption) | 0/9 = 0% | Fixes are API signature changes — no identifiable "sink" |
| CWE-362 (Race Condition) | 8/15 = 53% | Fixes add locking primitives, not at the data access point |
| CWE-20 (Input Validation) | 9/17 = 53% | Fixes are guards/conditions upstream of any operation |

The core issue: our architecture uses **Joern CPG queries to locate dangerous operations (sinks)** and **graph-diff embeddings to capture vulnerability structure**. This works best when:

1. The vulnerability manifests at a **syntactically identifiable dangerous operation**
2. The fix modifies code **at or immediately adjacent to** that operation
3. The structural pattern is **consistent** across instances of the same CWE class

### 1.2 Selection Criteria

We select CWEs where the vulnerability is **locatable by code structure** — specifically:

- **Concrete, syntactically identifiable sinks** (e.g., `free()`, `memcpy()`, arithmetic ops, pointer dereferences)
- **Fixes at or near the sink** (bounds checks, NULL guards, size validations, type corrections)
- **Consistent structural patterns** across instances (same CWE → similar graph diff topology)
- **Amenability to CPG analysis** (Joern can trace dataflow, control flow, or pointer relationships)
- **Proven strong or promising hit rates** from prior experiments (CWE-787 at 87.5%, CWE-476 at 64%, CWE-416 at 58%)

### 1.3 Selected CWEs

| CWE | Name | Sink / Pattern Type | Prior Evidence |
|-----|------|---------------------|----------------|
| **CWE-787** | Out-of-bounds Write | Buffer write ops (`memcpy`, `strcpy`, array index) | 87.5% hit rate (7/8) |
| **CWE-125** | Out-of-bounds Read | Buffer read ops, array access without bounds check | Parent of strong performers |
| **CWE-476** | NULL Pointer Dereference | Pointer dereference (`->`, `*ptr`) without NULL check | 64.3% hit rate (9/14) |
| **CWE-416** | Use After Free | `free()` + subsequent pointer dereference | 58.3% hit rate (7/12) |
| **CWE-190** | Integer Overflow | Arithmetic operations, type casts, assignments | Consistent graph diff patterns |
| **CWE-415** | Double Free | Multiple `free()` calls on same pointer | Clear control-flow sink |
| **CWE-122** | Heap-based Buffer Overflow | `malloc`+write, `memcpy`, `realloc` | Child of CWE-787 |
| **CWE-121** | Stack-based Buffer Overflow | `memcpy`, `strcpy`, `sprintf` into stack buffers | Child of CWE-787 |
| **CWE-843** | Type Confusion | Type cast operations, incompatible access | 100% hit rate (1/1 sample) |
| **CWE-129** | Array Index Validation | Array subscript without bounds check | 100% hit rate (1/1 sample) |
| **CWE-264** | Permissions/Privileges | Access control checks, capability validation | 66.7% hit rate (2/3) |
| **CWE-284** | Improper Access Control | Authorization checks, permission gates | 100% hit rate (1/1 sample) |
| **CWE-200** | Information Exposure | Data leak through buffer reads, uninitialized memory | 100% hit rate (1/1 sample) |

### 1.4 Why Not Other CWEs?

| Rejected CWE | Reason |
|--------------|--------|
| CWE-362 (Race Condition) | Fix is locking/synchronization — no single "sink"; requires whole-program analysis |
| CWE-400 (Resource Consumption) | Fixes are API-level changes; no consistent sink pattern; 0% hit rate |
| CWE-20 (Input Validation) | Fix is a validation check *anywhere* in the function; too diffuse for structural detection |
| CWE-667 (Improper Locking) | Similar to CWE-362; fix is adding lock/unlock pairs, not at data operations |

### 1.5 CWE Hierarchy Relationships

Several selected CWEs are related in the MITRE CWE hierarchy:

```
CWE-119 (Buffer Errors)
├── CWE-787 (Out-of-bounds Write)
│   ├── CWE-121 (Stack Buffer Overflow)
│   └── CWE-122 (Heap Buffer Overflow)
└── CWE-125 (Out-of-bounds Read)

CWE-416 (Use After Free)
CWE-415 (Double Free)
    └── Both involve memory lifecycle management

CWE-264 (Permissions) → CWE-284 (Access Control)
    └── Parent/child; both about authorization structure
```

This overlap is intentional — related CWEs often share structural patterns, which means:
- Similar graph diff topologies → better embedding clustering
- More training signal per vulnerability family
- Enables cross-CWE retrieval within the same family

---

## 2. Dataset Description

### 2.1 Source

| Source | Description | Priority |
|--------|-------------|----------|
| `cvefixes_code_extraction.json` | Pre-extracted method pairs from CVEfixes DB (linked to graphml_cvefixes/) | Primary (1,056 entries) |
| `data/cvefixes/CVEfixes.db` | Direct DB query for additional pairs | Secondary (2,607 entries) |

**Extraction date:** 2026-05-28  
**CVEfixes version:** v1.0.8  
**Languages:** C (2,611) and C++ (1,052)

### 2.2 Filtering Pipeline

```
CVEfixes DB (277,948 method_change rows)
    │
    ├─ Filter: CWE ∈ {13 target CWEs}
    ├─ Filter: programming_language ∈ {C, C++}
    ├─ Filter: has both code_before AND code_after (non-empty)
    ├─ Pair: match by (file_change_id, method_name, before_change=True/False)
    ├─ Dedup: unique by (cve_id, file_change_id, method_name)
    │
    └─► 3,663 entries across 1,378 unique CVEs
```

### 2.3 Output Structure

```
cvefixes_experiments/data/sink_cwes/
├── metadata.json        — reproducibility metadata, filters, timestamps
├── CWE-125.json         — 1,081 entries (Out-of-bounds Read)
├── CWE-787.json         — 652 entries (Out-of-bounds Write)
├── CWE-190.json         — 582 entries (Integer Overflow)
├── CWE-476.json         — 412 entries (NULL Pointer Dereference)
├── CWE-416.json         — 369 entries (Use After Free)
├── CWE-200.json         — 136 entries (Information Exposure)
├── CWE-415.json         — 129 entries (Double Free)
├── CWE-264.json         — 100 entries (Permissions)
├── CWE-122.json         — 71 entries (Heap Buffer Overflow)
├── CWE-284.json         — 65 entries (Access Control)
├── CWE-843.json         — 39 entries (Type Confusion)
├── CWE-121.json         — 15 entries (Stack Buffer Overflow)
└── CWE-129.json         — 12 entries (Array Index Validation)
```

---

## 3. Dataset Composition

### 3.1 Summary

| Metric | Value |
|--------|-------|
| Total entries | 3,663 |
| Unique CVEs | 1,378 |
| CWE classes | 13 |
| Language: C | 2,611 (71.3%) |
| Language: C++ | 1,052 (28.7%) |
| Date range | 2012-05-17 to 2024-06-17 |

### 3.2 Per-CWE Class Size

| CWE | Name | Entries | CVEs | Entries/CVE (mean) | Entries/CVE (max) |
|-----|------|---------|------|-------------------|-------------------|
| CWE-125 | Out-of-bounds Read | 1,081 | 334 | 3.24 | 250 |
| CWE-787 | Out-of-bounds Write | 652 | 263 | 2.48 | 64 |
| CWE-190 | Integer Overflow | 582 | 120 | 4.85 | 220 |
| CWE-476 | NULL Pointer Dereference | 412 | 247 | 1.67 | 14 |
| CWE-416 | Use After Free | 369 | 175 | 2.11 | 19 |
| CWE-200 | Information Exposure | 136 | 71 | 1.92 | 22 |
| CWE-415 | Double Free | 129 | 37 | 3.49 | 38 |
| CWE-264 | Permissions/Privileges | 100 | 44 | 2.27 | 12 |
| CWE-122 | Heap Buffer Overflow | 71 | 47 | 1.51 | 8 |
| CWE-284 | Improper Access Control | 65 | 14 | 4.64 | 41 |
| CWE-843 | Type Confusion | 39 | 10 | 3.90 | 24 |
| CWE-121 | Stack Buffer Overflow | 15 | 10 | 1.50 | 3 |
| CWE-129 | Array Index Validation | 12 | 7 | 1.71 | 3 |

### 3.3 Multi-Function CVEs (for Retrieval Tasks)

| CWE | CVEs with ≥2 functions | Entries in those CVEs | % of class |
|-----|------------------------|----------------------|------------|
| CWE-190 | 43 | 505 | 86.8% |
| CWE-284 | 7 | 58 | 89.2% |
| CWE-843 | 5 | 34 | 87.2% |
| CWE-415 | 16 | 108 | 83.7% |
| CWE-125 | 112 | 859 | 79.5% |
| CWE-264 | 22 | 78 | 78.0% |
| CWE-787 | 100 | 489 | 75.0% |
| CWE-416 | 65 | 259 | 70.2% |
| CWE-129 | 3 | 8 | 66.7% |
| CWE-200 | 19 | 84 | 61.8% |
| CWE-121 | 4 | 9 | 60.0% |
| CWE-476 | 64 | 229 | 55.6% |
| CWE-122 | 11 | 35 | 49.3% |
| **ALL** | **471** | **2,756** | **75.2%** |

75.2% of entries belong to CVEs where same-CVE retrieval is feasible (multiple affected functions exist).

---

## 4. Function Size (Lines of Code)

### 4.1 Per-CWE Statistics (code_before)

| CWE | n | Min | Max | Mean | Median | Std |
|-----|---|-----|-----|------|--------|-----|
| CWE-125 | 1,081 | 1 | 2,529 | 104.1 | 45.0 | 198.1 |
| CWE-787 | 652 | 1 | 2,914 | 124.3 | 51.0 | 239.9 |
| CWE-190 | 582 | 1 | 2,029 | 71.8 | 42.0 | 129.1 |
| CWE-476 | 412 | 3 | 3,573 | 126.2 | 55.0 | 279.3 |
| CWE-416 | 369 | 1 | 3,019 | 134.6 | 47.0 | 313.0 |
| CWE-200 | 136 | 5 | 1,505 | 112.6 | 40.0 | 234.7 |
| CWE-415 | 129 | 4 | 315 | 54.5 | 30.0 | 60.0 |
| CWE-264 | 100 | 3 | 734 | 69.9 | 40.0 | 97.8 |
| CWE-122 | 71 | 4 | 1,863 | 166.0 | 81.0 | 272.5 |
| CWE-284 | 65 | 1 | 335 | 64.8 | 46.0 | 68.7 |
| CWE-843 | 39 | 3 | 1,120 | 81.3 | 49.0 | 174.7 |
| CWE-121 | 15 | 18 | 1,414 | 205.6 | 125.0 | 346.7 |
| CWE-129 | 12 | 11 | 185 | 58.6 | 29.5 | 57.4 |
| **ALL** | **3,663** | **1** | **3,573** | **106.3** | **46.0** | **220.0** |

### 4.2 CPG-Viable Entries (10–120 lines)

| CWE | In sweet spot | Total | % |
|-----|---------------|-------|---|
| CWE-129 | 10 | 12 | 83.3% |
| CWE-843 | 32 | 39 | 82.1% |
| CWE-264 | 78 | 100 | 78.0% |
| CWE-190 | 441 | 582 | 75.8% |
| CWE-200 | 99 | 136 | 72.8% |
| CWE-125 | 776 | 1,081 | 71.8% |
| CWE-416 | 261 | 369 | 70.7% |
| CWE-415 | 90 | 129 | 69.8% |
| CWE-122 | 46 | 71 | 64.8% |
| CWE-284 | 42 | 65 | 64.6% |
| CWE-476 | 262 | 412 | 63.6% |
| CWE-787 | 399 | 652 | 61.2% |
| CWE-121 | 7 | 15 | 46.7% |
| **ALL** | **2,543** | **3,663** | **69.4%** |

Functions in the 10–120 line range are ideal for CPG generation: large enough to have meaningful structure, small enough to avoid Joern timeouts and degenerate graph sizes.

---

## 5. Diff Characteristics (Fix Size)

### 5.1 Change Magnitude

| CWE | Lines Added (mean) | Lines Removed (mean) | Total Changed (mean) | Similarity (mean) |
|-----|-------------------|---------------------|---------------------|-------------------|
| CWE-121 | 23.1 | 21.2 | 44.3 | 0.972 |
| CWE-284 | 8.5 | 11.5 | 20.0 | 0.775 |
| CWE-125 | 11.3 | 7.0 | 18.3 | 0.866 |
| CWE-416 | 7.7 | 6.0 | 13.6 | 0.895 |
| CWE-200 | 7.4 | 4.9 | 12.3 | 0.896 |
| CWE-264 | 6.0 | 5.9 | 11.9 | 0.848 |
| CWE-415 | 4.4 | 6.8 | 11.2 | 0.885 |
| CWE-843 | 6.8 | 3.2 | 10.0 | 0.862 |
| CWE-787 | 5.8 | 3.8 | 9.7 | 0.914 |
| CWE-129 | 6.3 | 2.9 | 9.2 | 0.916 |
| CWE-476 | 5.3 | 3.7 | 8.9 | 0.919 |
| CWE-122 | 4.5 | 3.0 | 7.5 | 0.942 |
| CWE-190 | 3.9 | 2.7 | 6.6 | 0.953 |

**Key insight:** Most fixes are **small and localized** — mean similarity > 0.85 for all CWEs except CWE-284 (0.775). CWE-190 has the smallest fixes (mean 6.6 lines changed, 95.3% similarity), making it ideal for precise sink detection. CWE-284 has larger structural changes, reflecting access control refactoring.

---

## 6. Temporal Coverage

| CWE | Earliest CVE | Latest CVE | Span |
|-----|-------------|------------|------|
| CWE-264 | 2012-06-13 | 2018-04-05 | 5.8 years |
| CWE-190 | 2012-05-17 | 2024-01-27 | 11.7 years |
| CWE-476 | 2012-05-17 | 2024-02-09 | 11.7 years |
| CWE-787 | 2012-10-03 | 2024-02-05 | 11.3 years |
| CWE-125 | 2014-05-11 | 2024-06-08 | 10.1 years |
| CWE-416 | 2014-07-03 | 2024-06-17 | 10.0 years |
| CWE-200 | 2013-02-18 | 2023-03-07 | 10.0 years |
| CWE-284 | 2015-02-19 | 2024-02-19 | 9.0 years |
| CWE-415 | 2016-08-07 | 2023-09-15 | 7.1 years |
| CWE-843 | 2017-09-21 | 2023-05-18 | 5.6 years |
| CWE-121 | 2018-07-27 | 2023-06-14 | 4.9 years |
| CWE-129 | 2017-07-02 | 2023-04-03 | 5.8 years |
| CWE-122 | 2019-11-25 | 2023-10-02 | 3.8 years |

---

## 7. Dataset Quality Considerations

### 7.1 Strengths

1. **Structural detectability:** All 13 CWEs have either proven hit rates (>58%) or structural patterns identifiable by CPG queries
2. **Scale:** 3,663 entries across 1,378 CVEs — sufficient for statistically meaningful experiments
3. **Fix locality:** Most entries have similarity > 0.85, meaning fixes are targeted and localized
4. **Retrieval feasibility:** 75.2% of entries belong to multi-function CVEs (same-CVE retrieval possible)
5. **CPG viability:** 69.4% of functions are in the 10–120 line sweet spot
6. **No CWE overlap:** Each entry appears in exactly one CWE file (deduplicated by primary CWE)
7. **Hierarchical coverage:** Includes parent/child CWEs (787→121/122, 264→284) enabling family-level analysis

### 7.2 Limitations

1. **Class imbalance:** CWE-125 (1,081) has 90× more entries than CWE-129 (12)
2. **Sparse classes:** CWE-121 (15), CWE-129 (12) have insufficient entries for standalone statistical analysis
3. **Outlier CVEs:** CVE-2023-36326 (CWE-190, 220 functions) and CVE-2020-22628 (CWE-125, 250 functions) dominate their classes
4. **CWE labeling noise:** NVD classifications can be inconsistent (CWE-787 vs CWE-122 vs CWE-121)
5. **Large functions:** Some entries have >1000 lines (may cause Joern timeouts)

### 7.3 Recommendations for Downstream Experiments

- **Cap function size** at 300 lines for CPG experiments (retains ~85% of data)
- **Downsample outlier CVEs** to ≤20 functions each to reduce class skew
- **Merge sparse classes:** CWE-121 + CWE-122 → "Buffer Overflow" (86 entries); CWE-264 + CWE-284 → "Access Control" (165 entries)
- **Stratified sampling** by CWE when constructing experiment subsets
- **For retrieval tasks:** Filter to multi-function CVEs only (2,756 entries)

---

## 8. Entry Schema

Each entry in the per-CWE JSON files:

```json
{
  "cve_id": "CVE-2019-6978",
  "cwe": [{"cwe_id": "CWE-415", "cwe_name": "Double Free"}],
  "cve_severity": "MEDIUM",
  "cvss3_base_score": "9.8",
  "published_date": "2019-01-28T08:29Z",
  "file_change_id": "123456789012345",
  "filename": "gd.c",
  "programming_language": "C",
  "method_name": "gdImageBmp",
  "method_signature": "gdImageBmp(gdImagePtr im, ...)",
  "code_before": "void gdImageBmp(...) { ... }",
  "code_after": "void gdImageBmp(...) { ... }",
  "changes": {
    "lines_added": 3,
    "lines_removed": 1,
    "lines_before": 42,
    "lines_after": 44,
    "similarity_ratio": 0.9532
  },
  "source": "db_extraction"
}
```

---

## 9. Reproduction

```bash
# Generate the dataset
python -m cvefixes_experiments.scripts.filter_cwes_for_sink_analysis

# Requires:
#   - cvefixes_experiments/data/cvefixes_code_extraction.json
#   - data/cvefixes/CVEfixes.db

# Runtime: ~15 seconds
```

---

## 10. Usage

```python
import json
from pathlib import Path

# Load a single CWE
with open("cvefixes_experiments/data/sink_cwes/CWE-416.json") as f:
    data = json.load(f)
entries = data["entries"]  # list of dicts
print(f"CWE-416: {data['entry_count']} entries")

# Load all CWEs
all_entries = []
for cwe_file in Path("cvefixes_experiments/data/sink_cwes").glob("CWE-*.json"):
    with open(cwe_file) as f:
        all_entries.extend(json.load(f)["entries"])

# Filter to sweet-spot size
viable = [e for e in all_entries
          if 10 <= len(e["code_before"].splitlines()) <= 120]
```

## 11. Previous Issues

**Source:** `cvefixes_experiments/output/joern_native_query_results.json`  
**Overall:** 48 misses out of 98 processed (49.0% miss rate)

### 11.1 Category 1: Zero Sinks Found — 29/48 (60.4% of misses)

The CPGQL query returned no nodes at all. The CWE-specific patterns simply don't match the code constructs involved in the fix.

| Fix Pattern | Count | Example |
|-------------|-------|---------|
| API signature change | 8 | `perf_event_overflow(event, 0, &data, regs)` → removed an argument. CWE-400 query looks for malloc/free, not API call changes. |
| Control flow / logic | 8 | goto target changes, added if conditions, return additions. Not targeted by sink queries. |
| Type change | 3 | `int` → `int64_t` / `size_t`. CWE-190 query looks for arithmetic operators, not type declarations. |
| Lock/sync change | 3 | Locking primitives added/changed. CWE-362 query looks for `->` dereferences, not locking calls. |
| Added validation | 2 | NULL checks added as extra conditions. |
| Other | 5 | Format strings, enum renames, etc. |

**Root cause:** The queries are sink-focused (looking for dangerous operations) but many fixes happen at non-sink locations — they change control flow, type widths, function signatures, or add guards before the sink.

### 11.2 Category 2: Sinks Found, No Overlap — 19/48 (39.6% of misses)

Joern found sinks (avg ~21 per method), but none matched the actual changed lines.

| Sub-pattern | Example |
|-------------|---------|
| Fix is a guard/condition, not the sink itself | `if(dest != src)` → `if(dest != src && src != NULL)` — the fix is the if condition, not the dereference |
| Fix is a type/signature change | `int hours` → `int64_t hours` — declaration, not a call node |
| Fix changes control flow | `goto out` → `goto out2` — a jump, not a call |
| Fix adds an argument | `btrfs_find_device(..., NULL)` → `..., NULL, true)` — substring matching fails |
| Fix replaces locking mechanism | `local_bh_enable()` → `spin_unlock_bh(...)` — different API name |

**Root cause:** The overlap check uses substring matching between sink code text and changed lines. This misses when:

- The fix is at a guard protecting the sink (not the sink itself)
- The fix changes a declaration or signature (not a call node)
- The fix slightly modifies a call (added arg) but substring containment fails

### 11.3 Takeaways

The current dataset's worst performers (CWE-400, CWE-362) fail because their "sinks" are semantically diffuse — the fix is often a completely different code construct than what any pattern-based query would find.

**Caveat on CWE-190:** It was already 2/7 = 29% in the prior run. The issue is that overflow fixes are often type changes (`int` → `size_t`), not at the arithmetic itself. The query would need to expand to include variable declarations and assignment targets — or focus on cases where the fix adds an explicit overflow check (e.g., `if (a + b < a)`).