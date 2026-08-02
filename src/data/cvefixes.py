"""CVEFixes dataset integration.

Reads the CVEFixes SQLite database (from Zenodo DOI: 10.5281/zenodo.4476563)
and provides method-level vulnerable/patched function pairs compatible with
the graph-rag pipeline.
"""

import json
import sqlite3
from pathlib import Path
from typing import Iterator

import networkx as nx
from tqdm import tqdm

from .base import BaseDataset, ExportJob, FunctionPair
from .pipeline import compute_graph_diff, cpg_dir_for, graph_diff_params, load_cpg_dir

_QUERY_METHODS = """\
SELECT
    m_after.method_change_id,
    m_after.name AS func_name,
    m_after.code AS method_code_after,
    m_before.code AS method_code_before,
    f.code_before AS file_code_before,
    f.code_after AS file_code_after,
    f.programming_language,
    cv.cve_id,
    cc.cwe_id,
    f.filename,
    c.repo_url
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
  AND f.programming_language IN ({lang_placeholders})
  AND m_before.code IS NOT NULL AND m_before.code != ''
  AND m_after.code IS NOT NULL AND m_after.code != ''
    AND f.code_before IS NOT NULL AND f.code_before != ''
    AND f.code_after IS NOT NULL AND f.code_after != ''
"""

_QUERY_FILES = """\
SELECT DISTINCT
    f.file_change_id,
    f.filename,
    f.programming_language,
    cv.cve_id,
    cc.cwe_id,
    c.repo_url
FROM file_change f
JOIN commits c ON f.hash = c.hash
JOIN fixes fx ON c.hash = fx.hash
JOIN cve cv ON fx.cve_id = cv.cve_id
LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
WHERE f.programming_language IN ({lang_placeholders})
  AND EXISTS (
    SELECT 1 FROM method_change m_after
    WHERE m_after.file_change_id = f.file_change_id
      AND m_after.before_change = 'False'
  )
"""

_QUERY_FILES_FULL = """\
SELECT
    f.file_change_id,
    f.filename,
    f.code_before,
    f.code_after,
    f.programming_language,
    cv.cve_id,
    cc.cwe_id,
    c.repo_url,
    GROUP_CONCAT(m_after.name, '|||') AS method_names,
    GROUP_CONCAT(m_after.code, '|||') AS methods_code_after,
    GROUP_CONCAT(m_before.code, '|||') AS methods_code_before
FROM file_change f
JOIN commits c ON f.hash = c.hash
JOIN fixes fx ON c.hash = fx.hash
JOIN cve cv ON fx.cve_id = cv.cve_id
LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
JOIN method_change m_after
    ON m_after.file_change_id = f.file_change_id
    AND m_after.before_change = 'False'
JOIN method_change m_before
    ON m_before.file_change_id = f.file_change_id
    AND m_before.name = m_after.name
    AND m_before.before_change = 'True'
WHERE f.programming_language IN ({lang_placeholders})
  AND f.code_before IS NOT NULL AND f.code_before != ''
  AND f.code_after IS NOT NULL AND f.code_after != ''
  AND m_before.code IS NOT NULL AND m_before.code != ''
  AND m_after.code IS NOT NULL AND m_after.code != ''
GROUP BY f.file_change_id, cv.cve_id
"""


class CVEFixesDataset(BaseDataset):
    """
    Dataset backed by the CVEFixes SQLite database.

    Config keys (under data.cvefixes):
        db_path:       path to CVEfixes.db
        graphml_root:  directory for Joern CPG outputs
        languages:     list of languages to include (default: [C, C++])
        level:         'method' (default) or 'file' — 'file' uses
                       load_from_filesystem() + parallel graph loading
        max_lines:     skip functions longer than this (default: 500)
        sample_limit:  max rows to process, 0 = unlimited (default: 0)
    """

    def name(self) -> str:
        return "CVEFixes"

    def _db_path(self) -> Path:
        return Path(self.cfg["db_path"])

    def _languages(self) -> list[str]:
        return self.cfg.get("languages", ["C", "C++"])

    def _max_lines(self) -> int:
        return int(self.cfg.get("max_lines", 500))

    def _sample_limit(self) -> int:
        return int(self.cfg.get("sample_limit", 0))

    def _target_cwes(self) -> list[str] | None:
        """Get target CWEs to filter on, or None if filtering disabled."""
        target_cwes = self.cfg.get("target_cwes", [])
        return target_cwes if target_cwes else None

    def _connect(self) -> sqlite3.Connection:
        db = self._db_path()
        if not db.exists():
            raise FileNotFoundError(
                f"CVEFixes database not found at {db}. "
                "Download from https://doi.org/10.5281/zenodo.4476563 and run: "
                "gzcat CVEfixes.sql.gz | sqlite3 CVEfixes.db"
            )
        conn = sqlite3.connect(str(db), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _build_query(self) -> tuple[str, list[str]]:
        langs = self._languages()
        placeholders = ", ".join("?" for _ in langs)
        query = _QUERY_METHODS.format(lang_placeholders=placeholders)

        limit = self._sample_limit()
        if limit > 0:
            query += f"\nLIMIT {int(limit)}"

        return query, langs

    def _row_id(self, row: sqlite3.Row) -> str:
        """Unique key for a method change — used for directory naming."""
        return f"{row['cve_id']}_m{row['method_change_id']}"

    def _file_row_id(self, file_id: int, cve_id: str) -> str:
        """Unique key for a file change — used for directory naming."""
        return f"{cve_id}_f{file_id}"

    def _passes_filter(self, code: str) -> bool:
        max_lines = self._max_lines()
        if max_lines > 0 and code.count("\n") > max_lines:
            return False
        return True

    def _get_methods_for_file(
        self, conn: sqlite3.Connection, file_change_id: int
    ) -> list[dict]:
        """Get all methods changed in a given file."""
        query = """\
        SELECT
            m_after.method_change_id,
            m_after.name AS func_name,
            m_after.code AS code_after,
            m_before.code AS code_before
        FROM method_change m_after
        JOIN method_change m_before
            ON m_before.file_change_id = m_after.file_change_id
            AND m_before.name = m_after.name
            AND m_before.before_change = 'True'
        WHERE m_after.file_change_id = ?
          AND m_after.before_change = 'False'
          AND m_before.code IS NOT NULL AND m_before.code != ''
          AND m_after.code IS NOT NULL AND m_after.code != ''
        """
        cursor = conn.execute(query, (file_change_id,))
        return [dict(row) for row in cursor.fetchall()]

    def _iter_rows(self):
        """Yield rows from the database, applying filters."""
        conn = self._connect()
        target_cwes = self._target_cwes()

        try:
            query, params = self._build_query()
            cursor = conn.execute(query, params)
            for row in cursor:
                # Filter by target CWEs if configured
                if target_cwes is not None:
                    cwe_id = row["cwe_id"]
                    if cwe_id not in target_cwes:
                        continue

                # NOTE: max_lines is documented/intended as a per-method cutoff,
                # so filter on the method's own code, not the whole file's code
                # (a short method inside a large file must still pass).
                method_code_before = row["method_code_before"]
                method_code_after = row["method_code_after"]
                if not self._passes_filter(method_code_before):
                    continue
                if not self._passes_filter(method_code_after):
                    continue
                yield row
        finally:
            conn.close()

    def _load_file_level_with_graphs(self) -> list:
        """Load file-level pairs from filesystem metadata.json, populating
        graphs in parallel using cpg_dir_for(graphml_root, dir_name, ...)."""
        import time
        from .pipeline import load_cpg_dirs_parallel

        graphml_root = self.cfg["graphml_root"]
        pairs = self.load_from_filesystem()

        graph_dirs_before, graph_dirs_after = [], []
        for p in pairs:
            dir_name = p.meta["dir_name"]
            graph_dirs_before.append(
                cpg_dir_for(graphml_root, cve_id=dir_name, variant="original", version="before")
            )
            graph_dirs_after.append(
                cpg_dir_for(graphml_root, cve_id=dir_name, variant="original", version="after")
            )

        all_dirs = graph_dirs_before + graph_dirs_after
        t0 = time.perf_counter()
        loaded = load_cpg_dirs_parallel(all_dirs, max_workers=8)
        print(
            f"  [CVEfixes/file] loaded {len(loaded)}/{len(all_dirs)} graphs "
            f"in {time.perf_counter() - t0:.1f}s"
        )

        result = []
        for i, p in enumerate(pairs):
            dir_b, dir_a = graph_dirs_before[i], graph_dirs_after[i]
            if dir_b in loaded and dir_a in loaded:
                g_before, g_after = loaded[dir_b], loaded[dir_a]
                if g_before.number_of_nodes() > 0 and g_after.number_of_nodes() > 0:
                    p.G_before, p.G_after = g_before, g_after
                    p.G_vuln = compute_graph_diff(
                        g_before, g_after, **graph_diff_params(self.cfg)
                    )
                    result.append(p)

        print(f"  [CVEfixes/file] {len(result)}/{len(pairs)} pairs have valid graphs")
        return result

    def stream(self) -> Iterator[FunctionPair]:
        if self.cfg.get("level") == "file":
            yield from self._load_file_level_with_graphs()
            return

        graphml_root = self.cfg["graphml_root"]

        for row in tqdm(self._iter_rows(), desc="Loading CVEFixes pairs"):
            row_id = self._row_id(row)
            cve_id = row["cve_id"]
            cwe_id = row["cwe_id"] or "UNKNOWN"
            func_name = row["func_name"] or ""

            try:
                G_before = load_cpg_dir(
                    cpg_dir_for(
                        graphml_root,
                        cve_id=row_id,
                        variant="original",
                        version="before",
                    )
                )
                G_after = load_cpg_dir(
                    cpg_dir_for(
                        graphml_root, cve_id=row_id, variant="original", version="after"
                    )
                )
            except FileNotFoundError:
                continue

            if G_before.number_of_nodes() == 0 or G_after.number_of_nodes() == 0:
                continue

            G_vuln = compute_graph_diff(
                G_before, G_after, **graph_diff_params(self.cfg)
            )

            yield FunctionPair(
                cve_id=cve_id,
                cwe_id=cwe_id,
                func_name=func_name,
                project=row["repo_url"] or "",
                G_before=G_before,
                G_after=G_after,
                G_vuln=G_vuln,
                meta={
                    "dataset": self.name(),
                    "variant": "original",
                    "method_change_id": row["method_change_id"],
                    "filename": row["filename"] or "",
                    "language": row["programming_language"],
                    "dir_name": row_id,
                    "source_before": row["file_code_before"],
                    "source_after": row["file_code_after"],
                },
            )

    def load_lightweight(self) -> list[FunctionPair]:
        """Load pairs with metadata only — no CPG/graph loading.

        Mirrors :meth:`stream` (method-level query) but skips Joern graph
        loading, so it works without a ``graphml_root`` export and without
        the (slow) CPG parse — suitable for patch generation, which only
        needs the code text + CVE/CWE metadata, not the graph.
        """
        _empty = nx.MultiDiGraph()
        pairs: list[FunctionPair] = []
        for row in tqdm(self._iter_rows(), desc="Loading CVEFixes pairs (lightweight)"):
            row_id = self._row_id(row)
            pairs.append(
                FunctionPair(
                    cve_id=row["cve_id"],
                    cwe_id=row["cwe_id"] or "UNKNOWN",
                    func_name=row["func_name"] or "",
                    project=row["repo_url"] or "",
                    G_before=_empty,
                    G_after=_empty,
                    G_vuln=_empty,
                    meta={
                        "dataset": self.name(),
                        "variant": "original",
                        "method_change_id": row["method_change_id"],
                        "filename": row["filename"] or "",
                        "language": row["programming_language"],
                        "dir_name": row_id,
                        "source_before": row["file_code_before"],
                        "source_after": row["file_code_after"],
                    },
                )
            )
        return pairs

    def _iter_file_rows(self):
        """Yield file_change rows from the database, applying filters."""
        conn = self._connect()
        target_cwes = self._target_cwes()

        try:
            query, params = self._build_query_files()
            cursor = conn.execute(query, params)
            for row in cursor:
                # Filter by target CWEs if configured
                if target_cwes is not None:
                    cwe_id = row["cwe_id"]
                    if cwe_id not in target_cwes:
                        continue
                yield row
        finally:
            conn.close()

    def _build_query_files(self) -> tuple[str, list[str]]:
        langs = self._languages()
        placeholders = ", ".join("?" for _ in langs)
        query = _QUERY_FILES.format(lang_placeholders=placeholders)

        limit = self._sample_limit()
        if limit > 0:
            query += f"\nLIMIT {int(limit)}"

        return query, langs

    def stream_file_level(self) -> Iterator[FunctionPair]:
        """Stream file-level pairs by aggregating all methods in each file."""
        graphml_root = self.cfg["graphml_root"]
        conn = self._connect()

        try:
            for file_row in tqdm(
                self._iter_file_rows(), desc="Loading CVEFixes file-level pairs"
            ):
                file_id = file_row["file_change_id"]
                cve_id = file_row["cve_id"]
                cwe_id = file_row["cwe_id"] or "UNKNOWN"
                filename = file_row["filename"] or "unknown"

                # Get all methods for this file
                methods = self._get_methods_for_file(conn, file_id)
                if not methods:
                    continue

                # Aggregate graphs from all methods
                G_before_merged = nx.MultiDiGraph()
                G_after_merged = nx.MultiDiGraph()
                G_vuln_merged = nx.MultiDiGraph()
                method_names = []
                total_lines_before = 0
                total_lines_after = 0

                for method_row in methods:
                    method_name = method_row["func_name"] or ""
                    method_names.append(method_name)
                    total_lines_before += method_row["code_before"].count("\n")
                    total_lines_after += method_row["code_after"].count("\n")

                # Skip if total file is too large
                if not self._passes_filter(("\n" * total_lines_before)):
                    continue
                if not self._passes_filter(("\n" * total_lines_after)):
                    continue

                # Load CPGs for each method and merge
                for method_row in methods:
                    method_id = method_row["method_change_id"]
                    method_row_id = f"{cve_id}_m{method_id}"

                    try:
                        G_before = load_cpg_dir(
                            cpg_dir_for(
                                graphml_root,
                                cve_id=method_row_id,
                                variant="original",
                                version="before",
                            )
                        )
                        G_after = load_cpg_dir(
                            cpg_dir_for(
                                graphml_root,
                                cve_id=method_row_id,
                                variant="original",
                                version="after",
                            )
                        )

                        if G_before.number_of_nodes() > 0:
                            G_before_merged = nx.union(G_before_merged, G_before)
                        if G_after.number_of_nodes() > 0:
                            G_after_merged = nx.union(G_after_merged, G_after)
                    except FileNotFoundError:
                        continue

                if (
                    G_before_merged.number_of_nodes() == 0
                    or G_after_merged.number_of_nodes() == 0
                ):
                    continue

                G_vuln_merged = compute_graph_diff(
                    G_before_merged, G_after_merged, **graph_diff_params(self.cfg)
                )
                file_row_id = self._file_row_id(file_id, cve_id)

                yield FunctionPair(
                    cve_id=cve_id,
                    cwe_id=cwe_id,
                    func_name=filename,  # Use filename as the identifier
                    project=file_row["repo_url"] or "",
                    G_before=G_before_merged,
                    G_after=G_after_merged,
                    G_vuln=G_vuln_merged,
                    meta={
                        "dataset": self.name(),
                        "variant": "original",
                        "file_change_id": file_id,
                        "filename": filename,
                        "language": file_row["programming_language"],
                        "dir_name": file_row_id,
                        "num_methods": len(method_names),
                        "method_names": method_names,
                    },
                )
        finally:
            conn.close()

    def export_jobs(self, graphml_root: str) -> Iterator[ExportJob]:
        for row in tqdm(self._iter_rows(), desc="Generating export jobs"):
            row_id = self._row_id(row)
            func_name = row["func_name"] or "function"
            base = Path(graphml_root) / row_id

            yield ExportJob(
                cve_id=row_id,
                func_name=func_name,
                variant="original",
                version="before",
                source_code=row["file_code_before"],
                out_dir=str(base / "original" / "before"),
            )

            yield ExportJob(
                cve_id=row_id,
                func_name=func_name,
                variant="original",
                version="after",
                source_code=row["file_code_after"],
                out_dir=str(base / "original" / "after"),
            )

    def load_lightweight(self) -> list[FunctionPair]:
        """Load pairs with metadata only — no CPG/graph loading."""
        _empty = nx.MultiDiGraph()
        pairs: list[FunctionPair] = []

        for row in tqdm(self._iter_rows(), desc="Loading lightweight pairs"):
            row_id = self._row_id(row)
            pairs.append(
                FunctionPair(
                    cve_id=row["cve_id"],
                    cwe_id=row["cwe_id"] or "UNKNOWN",
                    func_name=row["func_name"] or "",
                    project=row["repo_url"] or "",
                    G_before=_empty,
                    G_after=_empty,
                    G_vuln=_empty,
                    meta={
                        "dataset": self.name(),
                        "variant": "original",
                        "method_change_id": row["method_change_id"],
                        "filename": row["filename"] or "",
                        "language": row["programming_language"],
                        "dir_name": row_id,
                        "source_before": row["file_code_before"],
                        "source_after": row["file_code_after"],
                    },
                )
            )

        return pairs

    def load_lightweight_file_level(self) -> list[FunctionPair]:
        """Load file-level pairs with metadata only — no CPG/graph loading.

        Uses a single efficient query instead of N+1 pattern.
        Graphs can be loaded lazily later if needed.
        """
        _empty = nx.MultiDiGraph()
        pairs: list[FunctionPair] = []
        conn = self._connect()
        target_cwes = self._target_cwes()
        max_lines = self._max_lines()

        try:
            # Single query: get all files with their methods in one shot
            langs = self._languages()
            lang_placeholders = ", ".join("?" for _ in langs)

            query = f"""\
            SELECT
                f.file_change_id,
                f.filename,
                f.programming_language,
                cv.cve_id,
                cc.cwe_id,
                c.repo_url,
                m_after.method_change_id,
                m_after.name AS func_name,
                m_after.code AS code_after,
                m_before.code AS code_before
            FROM file_change f
            JOIN commits c ON f.hash = c.hash
            JOIN fixes fx ON c.hash = fx.hash
            JOIN cve cv ON fx.cve_id = cv.cve_id
            LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
            JOIN method_change m_after
                ON m_after.file_change_id = f.file_change_id
                AND m_after.before_change = 'False'
            JOIN method_change m_before
                ON m_before.file_change_id = f.file_change_id
                AND m_before.name = m_after.name
                AND m_before.before_change = 'True'
            WHERE f.programming_language IN ({lang_placeholders})
              AND m_before.code IS NOT NULL AND m_before.code != ''
              AND m_after.code IS NOT NULL AND m_after.code != ''
            ORDER BY f.file_change_id
            """

            params = langs
            cursor = conn.execute(query, params)

            # Group methods by file_change_id
            from itertools import groupby
            from operator import itemgetter

            rows = [dict(r) for r in cursor.fetchall()]
            print(f"  [CVEfixes] fetched {len(rows)} method rows from DB")

            # Group by file
            for file_id, file_methods in groupby(
                rows, key=itemgetter("file_change_id")
            ):
                methods = list(file_methods)
                first = methods[0]
                cve_id = first["cve_id"]
                cwe_id = first["cwe_id"] or "UNKNOWN"
                filename = first["filename"] or "unknown"

                # Filter by target CWEs
                if target_cwes and cwe_id not in target_cwes:
                    continue

                # Check total size
                total_lines = sum(m["code_before"].count("\n") for m in methods)
                if max_lines > 0 and total_lines > max_lines:
                    continue
                total_lines_after = sum(m["code_after"].count("\n") for m in methods)
                if max_lines > 0 and total_lines_after > max_lines:
                    continue

                method_names = [m["func_name"] or "" for m in methods]
                file_row_id = self._file_row_id(file_id, cve_id)

                pair = FunctionPair(
                    cve_id=cve_id,
                    cwe_id=cwe_id,
                    func_name=filename,
                    project=first["repo_url"] or "",
                    G_before=_empty,
                    G_after=_empty,
                    G_vuln=_empty,
                    meta={
                        "dataset": self.name(),
                        "variant": "original",
                        "file_change_id": file_id,
                        "filename": filename,
                        "language": first["programming_language"],
                        "dir_name": file_row_id,
                        "num_methods": len(method_names),
                        "method_names": method_names,
                        "method_ids": [m["method_change_id"] for m in methods],
                        "cve_id_value": cve_id,
                    },
                )
                pairs.append(pair)
        finally:
            conn.close()

        print(f"Lightweight loader returning {len(pairs)} pairs")
        return pairs

    def load_graphs_for_pair(self, pair: FunctionPair) -> FunctionPair:
        """Load graphs for a lightweight file-level pair (lazy loading).

        Takes a pair loaded with load_lightweight_file_level() and loads the actual graphs.
        """
        graphml_root = self.cfg["graphml_root"]
        meta = pair.meta

        if meta.get("_graphs_loaded"):
            return pair  # Already loaded

        cve_id = meta["cve_id_value"]
        method_ids = meta.get("method_ids", [])

        # Aggregate graphs from all methods
        G_before_merged = nx.MultiDiGraph()
        G_after_merged = nx.MultiDiGraph()

        for method_id in method_ids:
            method_row_id = f"{cve_id}_m{method_id}"

            try:
                G_before = load_cpg_dir(
                    cpg_dir_for(
                        graphml_root,
                        cve_id=method_row_id,
                        variant="original",
                        version="before",
                    )
                )
                G_after = load_cpg_dir(
                    cpg_dir_for(
                        graphml_root,
                        cve_id=method_row_id,
                        variant="original",
                        version="after",
                    )
                )

                if G_before.number_of_nodes() > 0:
                    G_before_merged = nx.union(G_before_merged, G_before)
                if G_after.number_of_nodes() > 0:
                    G_after_merged = nx.union(G_after_merged, G_after)
            except FileNotFoundError:
                continue

        G_vuln_merged = compute_graph_diff(
            G_before_merged, G_after_merged, **graph_diff_params(self.cfg)
        )

        # Return new pair with loaded graphs
        return FunctionPair(
            cve_id=pair.cve_id,
            cwe_id=pair.cwe_id,
            func_name=pair.func_name,
            project=pair.project,
            G_before=G_before_merged,
            G_after=G_after_merged,
            G_vuln=G_vuln_merged,
            meta={**pair.meta, "_graphs_loaded": True},
        )

    # ── File-level export and filesystem loader ──────────────────────

    def _iter_file_rows_full(self):
        """Yield file rows with full code and aggregated method info.

        One row per unique (file_change_id, cve_id) — no duplicates.
        """
        conn = self._connect()
        target_cwes = self._target_cwes()
        langs = self._languages()
        placeholders = ", ".join("?" for _ in langs)
        query = _QUERY_FILES_FULL.format(lang_placeholders=placeholders)

        limit = self._sample_limit()
        if limit > 0:
            query += f"\nLIMIT {int(limit)}"

        try:
            cursor = conn.execute(query, langs)
            for row in cursor:
                if target_cwes is not None:
                    cwe_id = row["cwe_id"]
                    if cwe_id not in target_cwes:
                        continue
                code_before = row["code_before"]
                code_after = row["code_after"]
                if not self._passes_filter(code_before):
                    continue
                if not self._passes_filter(code_after):
                    continue
                yield row
        finally:
            conn.close()

    def export_jobs_file_level(self, graphml_root: str) -> Iterator[ExportJob]:
        """Generate file-level export jobs.

        One CPG per file (before + after). Writes metadata.json alongside.
        No method-level duplication.
        """
        for row in tqdm(
            self._iter_file_rows_full(), desc="Generating file-level export jobs"
        ):
            file_id = row["file_change_id"]
            cve_id = row["cve_id"]
            cwe_id = row["cwe_id"] or "UNKNOWN"
            filename = row["filename"] or "file"
            file_row_id = self._file_row_id(file_id, cve_id)
            base = Path(graphml_root) / file_row_id

            # Parse aggregated method info
            method_names = (row["method_names"] or "").split("|||")
            methods_code_before = (row["methods_code_before"] or "").split("|||")
            methods_code_after = (row["methods_code_after"] or "").split("|||")

            # Write metadata.json for this file pair
            metadata = {
                "cve_id": cve_id,
                "cwe_id": cwe_id,
                "file_change_id": file_id,
                "filename": filename,
                "language": row["programming_language"],
                "repo_url": row["repo_url"] or "",
                "method_names": method_names,
                "methods_code_before": methods_code_before,
                "methods_code_after": methods_code_after,
            }
            base.mkdir(parents=True, exist_ok=True)
            (base / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            # Use stem of filename (without extension) as the .cpp file name
            stem = Path(filename).stem or "file"

            yield ExportJob(
                cve_id=file_row_id,
                func_name=stem,
                variant="original",
                version="before",
                source_code=row["code_before"],
                out_dir=str(base / "original" / "before"),
            )

            yield ExportJob(
                cve_id=file_row_id,
                func_name=stem,
                variant="original",
                version="after",
                source_code=row["code_after"],
                out_dir=str(base / "original" / "after"),
            )

    def load_from_filesystem(self) -> list[FunctionPair]:
        """Load file-level pairs from filesystem metadata.json files.

        Scans graphml_root for directories containing metadata.json,
        returns FunctionPair objects with empty graphs (load graphs on demand).

        This is instant — no database query needed.
        """
        graphml_root = Path(self.cfg["graphml_root"])
        target_cwes = self._target_cwes()
        max_lines = self._max_lines()
        _empty = nx.MultiDiGraph()
        pairs: list[FunctionPair] = []

        if not graphml_root.exists():
            raise FileNotFoundError(
                f"GraphML root not found: {graphml_root}. "
                f"Run: python main.py --mode export --dataset cvefixes --level file"
            )

        metadata_files = sorted(graphml_root.glob("*/metadata.json"))
        print(
            f"  [CVEfixes] scanning {graphml_root}: found {len(metadata_files)} metadata.json files"
        )

        for meta_path in metadata_files:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Warning: skipping {meta_path}: {e}")
                continue

            cve_id = meta["cve_id"]
            cwe_id = meta.get("cwe_id", "UNKNOWN")
            filename = meta.get("filename", "unknown")

            # Filter by target CWEs
            if target_cwes and cwe_id not in target_cwes:
                continue

            # dir_name is the folder name (e.g. CVE-2024-4741_f12345)
            dir_name = meta_path.parent.name

            pairs.append(
                FunctionPair(
                    cve_id=cve_id,
                    cwe_id=cwe_id,
                    func_name=filename,
                    project=meta.get("repo_url", ""),
                    G_before=_empty,
                    G_after=_empty,
                    G_vuln=_empty,
                    meta={
                        "dataset": self.name(),
                        "variant": "original",
                        "file_change_id": meta.get("file_change_id"),
                        "filename": filename,
                        "language": meta.get("language", ""),
                        "dir_name": dir_name,
                        "method_names": meta.get("method_names", []),
                        "methods_code_before": meta.get("methods_code_before", []),
                        "methods_code_after": meta.get("methods_code_after", []),
                        "cve_id_value": cve_id,
                    },
                )
            )

        print(f"  [CVEfixes] loaded {len(pairs)} file-level pairs from filesystem")
        return pairs
