"""CVEFixes dataset integration.

Reads the CVEFixes SQLite database (from Zenodo DOI: 10.5281/zenodo.4476563)
and provides method-level vulnerable/patched function pairs compatible with
the graph-rag pipeline.
"""

import sqlite3
from pathlib import Path
from typing import Iterator

import networkx as nx

from .base import BaseDataset, ExportJob, FunctionPair
from .pipeline import compute_graph_diff, cpg_dir_for, load_cpg_dir


_QUERY_METHODS = """\
SELECT
    m_after.method_change_id,
    m_after.name AS func_name,
    m_after.code AS code_after,
    m_before.code AS code_before,
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
"""


class CVEFixesDataset(BaseDataset):
    """
    Dataset backed by the CVEFixes SQLite database.

    Config keys (under data.cvefixes):
        db_path:       path to CVEfixes.db
        graphml_root:  directory for Joern CPG outputs
        languages:     list of languages to include (default: [C, C++])
        level:         'method' (only method supported currently)
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

    def _passes_filter(self, code: str) -> bool:
        max_lines = self._max_lines()
        if max_lines > 0 and code.count("\n") > max_lines:
            return False
        return True

    def _iter_rows(self):
        """Yield rows from the database, applying filters."""
        conn = self._connect()
        try:
            query, params = self._build_query()
            cursor = conn.execute(query, params)
            for row in cursor:
                code_before = row["code_before"]
                code_after = row["code_after"]
                if not self._passes_filter(code_before):
                    continue
                if not self._passes_filter(code_after):
                    continue
                yield row
        finally:
            conn.close()

    def stream(self) -> Iterator[FunctionPair]:
        graphml_root = self.cfg["graphml_root"]

        for row in self._iter_rows():
            row_id = self._row_id(row)
            cve_id = row["cve_id"]
            cwe_id = row["cwe_id"] or "UNKNOWN"
            func_name = row["func_name"] or ""

            try:
                G_before = load_cpg_dir(
                    cpg_dir_for(graphml_root, cve_id=row_id, variant="original", version="before")
                )
                G_after = load_cpg_dir(
                    cpg_dir_for(graphml_root, cve_id=row_id, variant="original", version="after")
                )
            except FileNotFoundError:
                continue

            if G_before.number_of_nodes() == 0 or G_after.number_of_nodes() == 0:
                continue

            G_vuln = compute_graph_diff(G_before, G_after)

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
                },
            )

    def export_jobs(self, graphml_root: str) -> Iterator[ExportJob]:
        for row in self._iter_rows():
            row_id = self._row_id(row)
            func_name = row["func_name"] or "function"
            base = Path(graphml_root) / row_id

            yield ExportJob(
                cve_id=row_id,
                func_name=func_name,
                variant="original",
                version="before",
                source_code=row["code_before"],
                out_dir=str(base / "original" / "before"),
            )

            yield ExportJob(
                cve_id=row_id,
                func_name=func_name,
                variant="original",
                version="after",
                source_code=row["code_after"],
                out_dir=str(base / "original" / "after"),
            )

    def load_lightweight(self) -> list[FunctionPair]:
        """Load pairs with metadata only — no CPG/graph loading."""
        _empty = nx.MultiDiGraph()
        pairs: list[FunctionPair] = []

        for row in self._iter_rows():
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
                        "source_before": row["code_before"],
                        "source_after": row["code_after"],
                    },
                )
            )

        return pairs
