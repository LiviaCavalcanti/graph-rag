"""Load FunctionPairs from a pinned JSON entries file (CVEfixes subset mode).

Restricts the pipeline (``--mode query`` / ``batch`` / ``full``) to a fixed
subset of CVE instances — e.g. ``experiments_cves/selected_entries.json`` —
instead of streaming the entire CVEfixes DB. Reuses already-generated CPG
caches when present and falls back to Joern for missing pairs.

Mirrors the cache-resolution convention used by
``cvefixes_experiments/scripts/performance/exp_method_vs_file_level.py``:
cache directories are named ``{idx:04d}_{cve_id}_{func_safe}``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from tqdm import tqdm

from .base import FunctionPair
from .pipeline import (
    compute_graph_diff,
    graph_diff_params,
    load_cpg_dir,
    run_joern_export,
    write_c_file,
)

_CACHE_DIR_RE = re.compile(r"^\d+_(CVE-\d+-\d+)_(.*)$")

DEFAULT_ENTRIES_FILE = "experiments_cves/selected_entries.json"
DEFAULT_ENTRIES_FILE = "testing_te_dataset/queries_pairs.json"
DEFAULT_CACHE_DIRS = [
    "cvefixes_experiments/output/pipeline_verification_method_ctxt/cpg_cache",
]
DEFAULT_BUILD_DIR = "cvefixes_experiments/output/_input_file_cpg_cache"
MIN_GRAPH_NODES = 10


def func_safe(name: str | None) -> str:
    """Sanitize a function name for use in cache directory names."""
    src = name or "func"
    return "".join(c if c.isalnum() or c == "_" else "_" for c in src)


def entry_key(entry: dict) -> tuple[str, str]:
    """Stable identity for a vulnerability instance: (cve_id, safe func)."""
    return (entry["cve_id"], func_safe(entry.get("method_name")))


def cwe_of(entry: dict) -> str:
    cwe = entry.get("cwe")
    if isinstance(cwe, list) and cwe:
        return cwe[0].get("cwe_id", "UNKNOWN")
    return entry.get("cwe_id", "UNKNOWN")


def load_entries(entries_path: str | Path) -> list[dict]:
    """Read a pinned-entries JSON file.

    Accepts either ``{"entries": [...]}`` (as produced by the CVEfixes
    sampling scripts) or a bare JSON array.
    """
    data = json.loads(Path(entries_path).read_text())
    if isinstance(data, dict):
        return data.get("entries", [])
    return data


def build_cache_map(cache_dir: str | Path) -> dict[tuple[str, str], Path]:
    """Map (cve_id, func_safe) -> entry dir for an existing CPG cache."""
    cache_dir = Path(cache_dir)
    mapping: dict[tuple[str, str], Path] = {}
    if not cache_dir.exists():
        return mapping
    for child in cache_dir.iterdir():
        if not child.is_dir():
            continue
        m = _CACHE_DIR_RE.match(child.name)
        if not m:
            continue
        mapping.setdefault((m.group(1), m.group(2)), child)
    return mapping


def build_cache_maps(cache_dirs: list[str | Path]) -> dict[tuple[str, str], Path]:
    """Merge cache maps from several cache directories (first hit wins)."""
    merged: dict[tuple[str, str], Path] = {}
    for d in cache_dirs:
        for k, v in build_cache_map(d).items():
            merged.setdefault(k, v)
    return merged


def load_pair_graphs(entry_dir: Path, min_nodes: int = MIN_GRAPH_NODES):
    """Load (G_before, G_after) from a cached entry dir, or None on failure."""
    before = entry_dir / "before" / "graph"
    after = entry_dir / "after" / "graph"
    if not before.exists() or not after.exists():
        return None
    try:
        g_before = load_cpg_dir(str(before))
        g_after = load_cpg_dir(str(after))
    except Exception:
        return None
    if g_before.number_of_nodes() < min_nodes or g_after.number_of_nodes() < min_nodes:
        return None
    return g_before, g_after


def joern_build_pair(
    code_before: str,
    code_after: str,
    out_dir: Path,
    func_safe_name: str,
    joern_bin: str,
    min_nodes: int = MIN_GRAPH_NODES,
):
    """Generate before/after CPGs with Joern into out_dir; return graphs or None."""
    before_dir = out_dir / "before"
    after_dir = out_dir / "after"
    try:
        src_before = write_c_file(code_before, before_dir / f"{func_safe_name}.cpp")
        if not run_joern_export(
            joern_bin, str(src_before), str(before_dir), str(before_dir / "graph")
        ):
            return None
        src_after = write_c_file(code_after, after_dir / f"{func_safe_name}.cpp")
        if not run_joern_export(
            joern_bin, str(src_after), str(after_dir), str(after_dir / "graph")
        ):
            return None
        return load_pair_graphs(out_dir, min_nodes=min_nodes)
    except Exception:
        return None


def resolve_pairs_from_entries(
    entries_path: str | Path,
    cfg: dict,
    *,
    cache_dirs: list[str] | None = None,
    build_dir: str | Path | None = None,
    reuse_only: bool = False,
) -> list[FunctionPair]:
    """Build FunctionPair objects for a pinned subset of CVEfixes entries.

    Reuses cached CPGs (``{idx}_{cve}_{func}`` naming convention) and falls
    back to Joern for missing pairs when ``joern.bin_dir`` is configured and
    ``reuse_only`` is False.
    """
    entries = load_entries(entries_path)
    cache_dirs = cache_dirs if cache_dirs is not None else DEFAULT_CACHE_DIRS
    cache_map = build_cache_maps(cache_dirs)

    build_dir = Path(build_dir if build_dir is not None else DEFAULT_BUILD_DIR)
    build_dir.mkdir(parents=True, exist_ok=True)

    joern_bin = (cfg.get("joern", {}) or {}).get("bin_dir") or None
    diff_params = graph_diff_params(cfg)

    pairs: list[FunctionPair] = []
    for entry in tqdm(entries, desc="Loading pinned CVEfixes entries"):
        cve_id, func_safe_name = entry_key(entry)

        graphs = None
        cached = cache_map.get((cve_id, func_safe_name))
        if cached is not None:
            graphs = load_pair_graphs(cached)

        if graphs is None:
            fresh_dir = build_dir / f"{cve_id}_{func_safe_name}"
            if (fresh_dir / "before" / "graph").exists():
                graphs = load_pair_graphs(fresh_dir)
            if graphs is None and not reuse_only and joern_bin:
                code_before = entry.get("code_before") or ""
                code_after = entry.get("code_after") or ""
                if code_before and code_after:
                    if fresh_dir.exists():
                        shutil.rmtree(fresh_dir, ignore_errors=True)
                    graphs = joern_build_pair(
                        code_before, code_after, fresh_dir, func_safe_name, joern_bin
                    )

        if graphs is None:
            continue

        g_before, g_after = graphs
        try:
            g_vuln = compute_graph_diff(g_before, g_after, **diff_params)
        except Exception:
            continue
        if g_vuln.number_of_nodes() == 0:
            continue

        pairs.append(
            FunctionPair(
                cve_id=cve_id,
                cwe_id=cwe_of(entry),
                func_name=entry.get("method_name") or "func",
                project=entry.get("project", ""),
                G_before=g_before,
                G_after=g_after,
                G_vuln=g_vuln,
                meta={
                    "dataset": "CVEfixes",
                    "variant": "original",
                    "filename": entry.get("filename", ""),
                    "language": entry.get("programming_language", "C"),
                    "file_change_id": entry.get("file_change_id"),
                    "dir_name": f"{cve_id}_{func_safe_name}",
                },
            )
        )

    if not pairs:
        raise RuntimeError(f"No valid pairs loaded from entries file {entries_path}")

    return pairs
