import json
import difflib
from pathlib import Path
from src.data.pipeline import load_cpg_dir
import re
from ..pipeline_verification.exp_pipeline_verification import select_entries, stratified_split, stratified_split, WORK_DIR, build_pairs, EMBEDDER_NAMES, EMBEDDER_REGISTRY
from collections import Counter
import yaml
import time
import networkx as nx

# file = "cvefixes_filtered_by_cwe.json"


# _CHANGED_THRESH = 2

# CWE_FAMILIES = {
#         "memory_corruption": ["CWE-119","CWE-120","CWE-121","CWE-122","CWE-125","CWE-787","CWE-805"],
#         "use_after_free": ["CWE-416","CWE-415","CWE-763"],
#         "integer_issues": ["CWE-190","CWE-191","CWE-189","CWE-681","CWE-369"],
#         "null_ptr_deref": ["CWE-476","CWE-824"],
#         "race_condition": ["CWE-362","CWE-367","CWE-667"],
# }

CWE = ['CWE-20', 'CWE-416', 'CWE-476', 'CWE-362', 'CWE-787'	, 'CWE-190','CWE-400']
def load_cvefixes_graphs(cwes: list) -> list[dict]:
    """Load CVEfixes graphs from cached Joern-generated CPGs."""

    CVEFIXES_CACHED = Path("workspace/graphfdg_cwe_train")
    # Load CWE mapping from filtered JSON
    cwe_map = {}
    for json_path in [
        Path("cvefixes_filtered_by_cwe.json"),
        Path("cvefixes_experiments/data/cvefixes_filtered_by_cwe.json"),
    ]:
        if json_path.exists():
            data = json.load(open(json_path))
            for e in data["entries"]:
                cve_id = e["cve_id"]
                cwes = [c["cwe_id"] for c in e.get("cwe", [])]
                if cwes:
                    cwe_map[cve_id] = cwes[0]
            break

    results = []
    # Use cached Joern-generated graphs (from the training experiment)
    if CVEFIXES_CACHED.exists():
        for entry_dir in sorted(CVEFIXES_CACHED.iterdir()):
            if not entry_dir.is_dir():
                continue
            before_dir = entry_dir / "before" / "graph"
            if not before_dir.exists():
                continue
            try:
                G = load_cpg_dir(str(before_dir))
            except Exception:
                continue
            # if G.number_of_nodes() < MIN_NODES:
            #     continue

            # Parse dir name: 0383_CVE-2023-3863_llcp_sock_connect
            parts = entry_dir.name.split("_", 2)
            cve_id = parts[1] if len(parts) > 1 else entry_dir.name
            # Try to find CVE-YYYY-NNNNN pattern
            m = re.search(r"(CVE-\d{4}-\d+)", entry_dir.name)
            if m:
                cve_id = m.group(1)
            cwe_id = cwe_map.get(cve_id, "UNKNOWN")

            results.append({
                "cve_id": cve_id,
                "cwe_id": cwe_id,
                "graph": G,
                "dataset": "CVEfixes",
            })
    return results

def get_entries(cfg_path, DATA_FILE, SEED = 42):
    DATA_FILE = Path("cvefixes_experiments/data/cvefixes_filtered_by_cwe.json")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    entries = select_entries(DATA_FILE, SEED)

# entry format
# {
#     'cve_id': 'CVE-2018-8788',
#     'cwe': [
#         {
#             'cwe_id': 'CWE-787', 
#             'cwe_name': 'Out-of-bounds Write'
#         }
#     ], 
#     'cve_description': "["
#         "{"
#             "'lang': 'en', "
#             "'value': 'FreeRDP prior to version 2.0.0-rc4 contains an Out-Of-Bounds Write of up to 4 bytes in function nsc_rle_decode() that results in a memory corruption and possibly even a remote code execution."
#         "'}"
#     "]", 
#     'cve_severity': 'HIGH', 
#     'cvss3_base_score': '9.8', 
#     'cvss3_base_severity': 'CRITICAL', 
#     'published_date': '2018-11-29T18:29Z', 
#     'file_change_id': '278964544778933', 
#     'filename': 'nsc.c', 
#     'programming_language': 'C', 
#     'method_name': 'nsc_rle_decompress_data', 
#     'method_signature': 'nsc_rle_decompress_data( NSC_CONTEXT * context)', 
#     'code_before': 'static void nsc_rle_decompress_data(NSC_CONTEXT* context)\n{\n\tUINT16 i;\n\tBYTE* rle;\n\tUINT32 planeSize;\n\tUINT32 originalSize;\n\trle = context->Planes;\n\n\tfor (i = 0; i < 4; i++)\n\t{\n\t\toriginalSize = context->OrgByteCount[i];\n\t\tplaneSize = context->PlaneByteCount[i];\n\n\t\tif (planeSize == 0)\n\t\t\tFillMemory(context->priv->PlaneBuffers[i], originalSize, 0xFF);\n\t\telse if (planeSize < originalSize)\n\t\t\tnsc_rle_decode(rle, context->priv->PlaneBuffers[i], originalSize);\n\t\telse\n\t\t\tCopyMemory(context->priv->PlaneBuffers[i], rle, originalSize);\n\n\t\trle += planeSize;\n\t}\n}', 
#     'code_after': 'static BOOL nsc_rle_decompress_data(NSC_CONTEXT* context)\n{\n\tUINT16 i;\n\tBYTE* rle;\n\tUINT32 planeSize;\n\tUINT32 originalSize;\n\n\tif (!context)\n\t\treturn FALSE;\n\n\trle = context->Planes;\n\n\tfor (i = 0; i < 4; i++)\n\t{\n\t\toriginalSize = context->OrgByteCount[i];\n\t\tplaneSize = context->PlaneByteCount[i];\n\n\t\tif (planeSize == 0)\n\t\t{\n\t\t\tif (context->priv->PlaneBuffersLength < originalSize)\n\t\t\t\treturn FALSE;\n\n\t\t\tFillMemory(context->priv->PlaneBuffers[i], originalSize, 0xFF);\n\t\t}\n\t\telse if (planeSize < originalSize)\n\t\t{\n\t\t\tif (!nsc_rle_decode(rle, context->priv->PlaneBuffers[i], context->priv->PlaneBuffersLength,\n\t\t\t                    originalSize))\n\t\t\t\treturn FALSE;\n\t\t}\n\t\telse\n\t\t{\n\t\t\tif (context->priv->PlaneBuffersLength < originalSize)\n\t\t\t\treturn FALSE;\n\n\t\t\tCopyMemory(context->priv->PlaneBuffers[i], rle, originalSize);\n\t\t}\n\n\t\trle += planeSize;\n\t}\n\n\treturn TRUE;\n}', 
#     'changes': {
#         'lines_added': 22, 
#         'lines_removed': 2, 
#         'lines_before': 23, 
#         'lines_after': 43, 
#         'similarity_ratio': 0.7484
#     }
# }


def extract_added_removed_lines(code_before: str, code_after: str) -> dict:
    """Return line-level additions and removals between two code strings."""
    before_lines = code_before.splitlines()
    after_lines = code_after.splitlines()

    added = []
    removed = []
    for line in difflib.ndiff(before_lines, after_lines):
        if line.startswith("+ "):
            added.append(line[2:])
        elif line.startswith("- "):
            removed.append(line[2:])

    return {
        "added": added,
        "removed": removed,
        "lines_added": len(added),
        "lines_removed": len(removed),
    }

def collect_changed_code(
    G: nx.MultiDiGraph,
    max_tokens: int = 400,
    dw_thresh: float = 0.2,
) -> str:
    """
    Concatenate CODE from changed nodes (diff_weight > threshold)
    into one string for CodeBERT.  Ordered by importance then line.
    """
    changed = []
    for nd in G.nodes():
        attr = G.nodes[nd]
        dw = float(attr.get("diff_weight", 0.2))
        code = (attr.get("CODE", "") or "").strip()
        if not code:
            continue
        line = int(attr.get("LINE_NUMBER", 9999) or 9999)
        changed.append((dw, line, code))

    # Fallback: no nodes passed the diff threshold → use ALL code nodes
    if not changed:
        for nd in G.nodes():
            code = (G.nodes[nd].get("CODE", "") or "").strip()
            if not code:
                continue
            line = int(G.nodes[nd].get("LINE_NUMBER", 9999) or 9999)
            changed.append((0.0, line, code))

    if not changed:
        return ""

    changed.sort(key=lambda t: (-t[0], t[1]))
    parts, tok_count = [], 0
    for _, _, code in changed:
        words = code.split()
        if tok_count + len(words) > max_tokens:
            remaining = max_tokens - tok_count
            if remaining > 0:
                parts.append(" ".join(words[:remaining]))
            break
        parts.append(code)
        tok_count += len(words)

    return " ".join(parts) # or join with \n ?


# for e in entries:
#     diff_result = extract_added_removed_lines(e['code_before'], e['code_after'])
#     print(f"CVE: {e['cve_id']}")
#     print(f"Added: {diff_result['lines_added']}, Removed: {diff_result['lines_removed']}")
#     print("\nAdded lines:")
#     for line in diff_result["added"]:
#         print(f"+ {line}")
#     print("\nRemoved lines:")
#     for line in diff_result["removed"]:
#         print(f"- {line}")
#     break
# pairs = build_pairs(entries, WORK_DIR)


# # 3. Split
# print(f"\n[3/5] Splitting into index/query (stratified by CWE)...")
# index_pairs, query_pairs = stratified_split(pairs, test_ratio=0.2, seed=SEED)

# # Verify query entries have support in index
# index_cves = set(p.cve_id for p in index_pairs)
# query_pairs = [p for p in query_pairs if p.cve_id in index_cves]

# print(f"  Index: {len(index_pairs)} entries")
# print(f"  Query: {len(query_pairs)} entries (all have same-CVE support in index)")
# print(f"  Index CWE dist: {dict(Counter(p.cwe_id for p in index_pairs))}")
# print(f"  Query CWE dist: {dict(Counter(p.cwe_id for p in query_pairs))}")

# split_info = {
#     "seed": SEED,
#     "n_index": len(index_pairs),
#     "n_query": len(query_pairs),
#     "index_cwe_dist": dict(Counter(p.cwe_id for p in index_pairs)),
#     "query_cwe_dist": dict(Counter(p.cwe_id for p in query_pairs)),
#     "index_cve_unique": len(set(p.cve_id for p in index_pairs)),
#     "query_cve_unique": len(set(p.cve_id for p in query_pairs)),
# }

# # 4. Embed + retrieve for each embedder
# print(f"\n[4/5] Running retrieval for {len(EMBEDDER_NAMES)} embedders...")
# emb_cfg = cfg.get("embeddings", {})
# cells = []

# for emb_name in EMBEDDER_NAMES:
#     embedder = EMBEDDER_REGISTRY[emb_name](emb_cfg)
#     print(f"\n  ── {embedder.name} ──")

#     # Embed index
#     t0 = time.perf_counter()
#     index_graphs = [p.G_vuln for p in index_pairs]
#     index_embeddings = embedder.embed_many(index_graphs)
#     embed_time = time.perf_counter() - t0
#     print(f"    Embedded {len(index_graphs)} graphs in {embed_time:.1f}s")


