"""CVEfixes dataset variant that loads FunctionPairs from a pinned JSON
entries file instead of streaming the live CVEfixes.db.

Wraps ``src.data.entries_cache.resolve_pairs_from_entries`` (file-based
loading + CPG cache reuse + optional Joern fallback) with the CVE-aware
CWE sampling from ``src.data.sampling`` (original / proportional /
balanced), so both concerns are available as one first-class dataset that
plugs into ``REGISTRY`` like ``CVEFixesDataset``/``AutoPatchDataset``.
"""

from __future__ import annotations

from typing import Iterator

import networkx as nx

from src.data.base import BaseDataset, ExportJob, FunctionPair
from src.data.entries_cache import (
    DEFAULT_ENTRIES_FILE,
    cwe_of,
    entry_key,
    load_entries,
    resolve_pairs_from_entries,
)
from src.data.sampling import sample_cve_aware


class CVEFixesFileDataset(BaseDataset):
    """Loads a pinned CVEfixes subset from a JSON entries file, with sampling.

    Config keys (under ``data.cvefixes_file``):
      input_file: path to the entries JSON (default: entries_cache.DEFAULT_ENTRIES_FILE)
      sample_mode: "original" (default, no resampling) | "proportional" | "balanced"
      sample_total: int | None — target pair count for proportional/balanced
      seed: int (default 42)
      min_cves_per_cwe: int (default 1)
      cache_dirs: list[str] | None — CPG cache dirs to reuse
        (entries_cache.DEFAULT_CACHE_DIRS if omitted)
      build_dir: str | None — where to build missing CPGs via Joern
        (entries_cache.DEFAULT_BUILD_DIR if omitted)
      reuse_only: bool (default False) — if True, never falls back to Joern
        for entries missing from the CPG cache
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.entries_path = cfg.get("input_file", DEFAULT_ENTRIES_FILE)
        self.sample_mode = cfg.get("sample_mode", "original")
        self.sample_total = cfg.get("sample_total")
        self.seed = cfg.get("seed", 42)
        self.min_cves_per_cwe = cfg.get("min_cves_per_cwe", 1)
        self.cache_dirs = cfg.get("cache_dirs")
        self.build_dir = cfg.get("build_dir")
        self.reuse_only = cfg.get("reuse_only", False)

    def name(self) -> str:
        return "CVEFixesFile"

    def stream(self) -> Iterator[FunctionPair]:
        pairs = resolve_pairs_from_entries(
            self.entries_path,
            self.cfg,
            cache_dirs=self.cache_dirs,
            build_dir=self.build_dir,
            reuse_only=self.reuse_only,
        )
        if self.sample_mode == "original":
            yield from pairs
            return

        sampled, info = sample_cve_aware(
            pairs,
            mode=self.sample_mode,
            seed=self.seed,
            min_cves_per_cwe=self.min_cves_per_cwe,
            total=self.sample_total,
        )
        print(
            f"  [CVEFixesFile] sample_mode={self.sample_mode} -> "
            f"{info['result_total_pairs']} pairs, {info['result_total_cves']} CVEs "
            f"across {len(info['per_cwe'])} CWEs"
        )
        yield from sampled

    def load_lightweight(self) -> list[FunctionPair]:
        """Load pairs with metadata only — no CPG/graph loading.

        Reads the pinned entries JSON directly (code_before/code_after +
        cve/cwe/func metadata) instead of going through
        :func:`resolve_pairs_from_entries` (which builds/loads CPGs), then
        applies the same CVE-aware sampling as :meth:`stream`. Suitable for
        patch generation, which only needs the code text + CVE/CWE metadata,
        not the graph.
        """
        empty = nx.MultiDiGraph()
        entries = load_entries(self.entries_path)
        pairs: list[FunctionPair] = []
        for entry in entries:
            cve_id, func_safe_name = entry_key(entry)
            pairs.append(
                FunctionPair(
                    cve_id=cve_id,
                    cwe_id=cwe_of(entry),
                    func_name=entry.get("method_name") or "func",
                    project=entry.get("project", ""),
                    G_before=empty,
                    G_after=empty,
                    G_vuln=empty,
                    meta={
                        "dataset": "CVEfixes",
                        "variant": "original",
                        "filename": entry.get("filename", ""),
                        "language": entry.get("programming_language", "C"),
                        "file_change_id": entry.get("file_change_id"),
                        "dir_name": f"{cve_id}_{func_safe_name}",
                        "source_before": entry.get("code_before", ""),
                        "source_after": entry.get("code_after", ""),
                    },
                )
            )

        if self.sample_mode == "original":
            return pairs

        sampled, info = sample_cve_aware(
            pairs,
            mode=self.sample_mode,
            seed=self.seed,
            min_cves_per_cwe=self.min_cves_per_cwe,
            total=self.sample_total,
        )
        print(
            f"  [CVEFixesFile] sample_mode={self.sample_mode} -> "
            f"{info['result_total_pairs']} pairs, {info['result_total_cves']} CVEs "
            f"across {len(info['per_cwe'])} CWEs"
        )
        return sampled

    def export_jobs(self, graphml_root: str) -> Iterator[ExportJob]:
        raise NotImplementedError(
            "CVEFixesFileDataset builds CPGs lazily (with an optional Joern "
            "fallback inside resolve_pairs_from_entries) during stream()/"
            "load_all() — there is no separate export step. Use the "
            "'cvefixes' dataset with --mode export for bulk CPG "
            "pre-generation instead."
        )
