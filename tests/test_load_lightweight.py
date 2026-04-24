"""Tests for AutoPatchDataset.load_lightweight().

These tests verify the lightweight loading path that skips CPG/GraphML
parsing and returns FunctionPair objects with metadata only.
"""

import json
import pytest
from pathlib import Path

import networkx as nx

from data.autopatch import AutoPatchDataset, _VARIANTS
from data.base import FunctionPair


# ── fixtures ──────────────────────────────────────────────────────────

SAMPLE_DB = {
    "cve_id": "CVE-2025-0001",
    "cwe_type": "CWE-119",
    "function_name": "vuln_fn",
    "function_prototype": "void vuln_fn()",
    "root_cause": "buffer overflow",
    "fix_list": ["bounds check"],
    "original_code": "void vuln_fn() { buf[20]=0; }",
    "vuln_patch": "void vuln_fn() { buf[9]=0; }",
    "supplementary_code": "",
}


def _make_cve_dir(
    root: Path,
    dir_name: str,
    db_entry: dict,
    *,
    has_original: bool = True,
    has_supplementary: bool = False,
    variants: list[dict] | None = None,
):
    """Build a minimal CVE directory matching AutoPatch on-disk structure."""
    cve_dir = root / dir_name
    cve_dir.mkdir(parents=True, exist_ok=True)
    out_v2 = cve_dir / "out_v2"
    out_v2.mkdir(exist_ok=True)
    (out_v2 / "db_entry.json").write_text(json.dumps(db_entry))

    if has_original:
        (cve_dir / "original_code.txt").write_text(
            "void vuln_fn() {\n\tchar buf[10];\n\tbuf[20]=0;\n}"
        )
        (cve_dir / "vuln_patch.txt").write_text(
            "void vuln_fn() {\n\tchar buf[10];\n\tbuf[9]=0;\n}"
        )

    if has_supplementary:
        (cve_dir / "supplementary_code.txt").write_text("struct Ctx { int x; };")

    if variants:
        code_dir = out_v2 / "code"
        code_dir.mkdir(exist_ok=True)
        for v in variants:
            (code_dir / v["json_file"]).write_text(
                json.dumps(
                    {
                        "is_vulnerable": v.get("is_vulnerable", True),
                        "re_implemented_code": v.get("code", "void reimpl(){}"),
                        "supplementary_code": v.get("supp", ""),
                    }
                )
            )
            (code_dir / v["fixed_file"]).write_text(v.get("fixed_code", "void reimpl_fixed(){}"))

    return cve_dir


def _make_dataset(root: Path, include_variants: bool = False) -> AutoPatchDataset:
    return AutoPatchDataset({"root": str(root), "include_variants": include_variants})


# ── basic loading ─────────────────────────────────────────────────────


class TestLoadLightweightBasic:

    def test_returns_list_of_function_pairs(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        ds = _make_dataset(tmp_path)
        pairs = ds.load_lightweight()
        assert isinstance(pairs, list)
        assert all(isinstance(p, FunctionPair) for p in pairs)

    def test_loads_correct_count_for_originals_only(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        _make_cve_dir(tmp_path, "CVE-2025-0002", {**SAMPLE_DB, "cve_id": "CVE-2025-0002"})
        ds = _make_dataset(tmp_path)
        pairs = ds.load_lightweight()
        assert len(pairs) == 2

    def test_empty_dataset_returns_empty_list(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert ds.load_lightweight() == []

    def test_skips_dir_without_db_entry(self, tmp_path):
        # dir exists but no out_v2/db_entry.json
        (tmp_path / "CVE-2025-NODB").mkdir()
        ds = _make_dataset(tmp_path)
        assert ds.load_lightweight() == []

    def test_skips_dir_without_original_code_files(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB, has_original=False)
        ds = _make_dataset(tmp_path)
        assert ds.load_lightweight() == []


# ── graphs are empty ──────────────────────────────────────────────────


class TestLoadLightweightGraphs:

    def test_graphs_are_empty_multidigraphs(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        ds = _make_dataset(tmp_path)
        pair = ds.load_lightweight()[0]
        for G in (pair.G_before, pair.G_after, pair.G_vuln):
            assert isinstance(G, nx.MultiDiGraph)
            assert G.number_of_nodes() == 0
            assert G.number_of_edges() == 0

    def test_graphs_are_not_None(self, tmp_path):
        """Even though they're empty, graphs must be present (not None)."""
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert pair.G_before is not None
        assert pair.G_after is not None
        assert pair.G_vuln is not None


# ── metadata correctness ─────────────────────────────────────────────


class TestLoadLightweightMeta:

    def test_cve_id_comes_from_db_entry(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert pair.cve_id == "CVE-2025-0001"

    def test_cwe_id_comes_from_db_entry(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert pair.cwe_id == "CWE-119"

    def test_func_name_comes_from_db_entry(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert pair.func_name == "vuln_fn"

    def test_dir_name_is_directory_name_not_cve_id(self, tmp_path):
        """dir_name must be the physical directory name (e.g. CVE-2025-0001_2),
        NOT the cve_id from db_entry.json (which may be shared across dirs)."""
        db = {**SAMPLE_DB, "cve_id": "CVE-2025-0001"}
        _make_cve_dir(tmp_path, "CVE-2025-0001_2", db)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert pair.meta["dir_name"] == "CVE-2025-0001_2"
        assert pair.cve_id == "CVE-2025-0001"  # cve_id differs from dir_name

    def test_variant_is_original(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert pair.meta["variant"] == "original"

    def test_source_before_is_file_path_for_original(self, tmp_path):
        """For original variants, source_before should be the path to original_code.txt."""
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert Path(pair.meta["source_before"]).name == "original_code.txt"
        assert Path(pair.meta["source_before"]).exists()

    def test_source_after_is_file_path_for_original(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert Path(pair.meta["source_after"]).name == "vuln_patch.txt"
        assert Path(pair.meta["source_after"]).exists()

    def test_root_cause_and_fix_list_propagated(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert pair.meta["root_cause"] == "buffer overflow"
        assert pair.meta["fix_list"] == ["bounds check"]

    def test_supplementary_code_loaded_when_file_exists(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB, has_supplementary=True)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert "struct Ctx" in pair.meta["supplementary_code"]

    def test_supplementary_code_empty_when_file_missing(self, tmp_path):
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB, has_supplementary=False)
        pair = _make_dataset(tmp_path).load_lightweight()[0]
        assert pair.meta["supplementary_code"] == ""


# ── multi-directory same CVE (the _N suffix pattern) ──────────────────


class TestLoadLightweightMultiDir:

    def test_multiple_dirs_same_cve_produce_separate_pairs(self, tmp_path):
        """CVE-2025-0001_1 and CVE-2025-0001_2 share cve_id but represent
        different functions — each MUST produce a separate FunctionPair."""
        db1 = {**SAMPLE_DB, "function_name": "func_a"}
        db2 = {**SAMPLE_DB, "function_name": "func_b"}
        _make_cve_dir(tmp_path, "CVE-2025-0001_1", db1)
        _make_cve_dir(tmp_path, "CVE-2025-0001_2", db2)
        pairs = _make_dataset(tmp_path).load_lightweight()
        assert len(pairs) == 2

    def test_dir_names_are_unique_across_same_cve(self, tmp_path):
        db1 = {**SAMPLE_DB, "function_name": "func_a"}
        db2 = {**SAMPLE_DB, "function_name": "func_b"}
        _make_cve_dir(tmp_path, "CVE-2025-0001_1", db1)
        _make_cve_dir(tmp_path, "CVE-2025-0001_2", db2)
        pairs = _make_dataset(tmp_path).load_lightweight()
        dir_names = [p.meta["dir_name"] for p in pairs]
        assert len(set(dir_names)) == 2
        assert "CVE-2025-0001_1" in dir_names
        assert "CVE-2025-0001_2" in dir_names

    def test_func_names_differ_across_same_cve_dirs(self, tmp_path):
        db1 = {**SAMPLE_DB, "function_name": "func_a"}
        db2 = {**SAMPLE_DB, "function_name": "func_b"}
        _make_cve_dir(tmp_path, "CVE-2025-0001_1", db1)
        _make_cve_dir(tmp_path, "CVE-2025-0001_2", db2)
        pairs = _make_dataset(tmp_path).load_lightweight()
        func_names = {p.func_name for p in pairs}
        assert func_names == {"func_a", "func_b"}


# ── variant loading ───────────────────────────────────────────────────


class TestLoadLightweightVariants:

    def _augmented_variant(self, **overrides):
        base = {
            "json_file": "augmented.json",
            "fixed_file": "augmented_fixed.c",
            "code": "void aug_vuln() { char *p=0; *p=1; }",
            "is_vulnerable": True,
        }
        base.update(overrides)
        return base

    def test_no_variants_without_flag(self, tmp_path):
        _make_cve_dir(
            tmp_path, "CVE-2025-0001", SAMPLE_DB,
            variants=[self._augmented_variant()],
        )
        pairs = _make_dataset(tmp_path, include_variants=False).load_lightweight()
        # only original should be loaded
        assert len(pairs) == 1
        assert pairs[0].meta["variant"] == "original"

    def test_variants_loaded_with_flag(self, tmp_path):
        _make_cve_dir(
            tmp_path, "CVE-2025-0001", SAMPLE_DB,
            variants=[self._augmented_variant()],
        )
        pairs = _make_dataset(tmp_path, include_variants=True).load_lightweight()
        variants = {p.meta["variant"] for p in pairs}
        assert "original" in variants
        assert "augmented" in variants
        assert len(pairs) == 2

    def test_variant_source_before_is_inline_code_not_path(self, tmp_path):
        """For augmented variants, source_before must be the inline code
        from the variant JSON, NOT a file path."""
        _make_cve_dir(
            tmp_path, "CVE-2025-0001", SAMPLE_DB,
            variants=[self._augmented_variant(code="int foo() { return 0; }")],
        )
        pairs = _make_dataset(tmp_path, include_variants=True).load_lightweight()
        aug = [p for p in pairs if p.meta["variant"] == "augmented"][0]
        assert aug.meta["source_before"] == "int foo() { return 0; }"
        assert not Path(aug.meta["source_before"]).exists()

    def test_variant_source_after_is_file_path(self, tmp_path):
        _make_cve_dir(
            tmp_path, "CVE-2025-0001", SAMPLE_DB,
            variants=[self._augmented_variant()],
        )
        pairs = _make_dataset(tmp_path, include_variants=True).load_lightweight()
        aug = [p for p in pairs if p.meta["variant"] == "augmented"][0]
        assert Path(aug.meta["source_after"]).exists()
        assert Path(aug.meta["source_after"]).name == "augmented_fixed.c"

    def test_non_vulnerable_variant_is_skipped(self, tmp_path):
        _make_cve_dir(
            tmp_path, "CVE-2025-0001", SAMPLE_DB,
            variants=[self._augmented_variant(is_vulnerable=False)],
        )
        pairs = _make_dataset(tmp_path, include_variants=True).load_lightweight()
        assert len(pairs) == 1  # only original

    def test_variant_shares_dir_name_with_original(self, tmp_path):
        """All variants from the same directory must have the same dir_name."""
        _make_cve_dir(
            tmp_path, "CVE-2025-0001", SAMPLE_DB,
            variants=[self._augmented_variant()],
        )
        pairs = _make_dataset(tmp_path, include_variants=True).load_lightweight()
        dir_names = {p.meta["dir_name"] for p in pairs}
        assert dir_names == {"CVE-2025-0001"}

    def test_multiple_variants_all_loaded(self, tmp_path):
        variants = [
            {"json_file": "augmented.json", "fixed_file": "augmented_fixed.c",
             "code": "void a(){}", "is_vulnerable": True},
            {"json_file": "re_implemented_deepseek.json",
             "fixed_file": "re_implemented_deepseek_fixed.c",
             "code": "void d(){}", "is_vulnerable": True},
        ]
        _make_cve_dir(tmp_path, "CVE-2025-0001", SAMPLE_DB, variants=variants)
        pairs = _make_dataset(tmp_path, include_variants=True).load_lightweight()
        variant_names = {p.meta["variant"] for p in pairs}
        assert "original" in variant_names
        assert "augmented" in variant_names
        assert "re_implemented_deepseek" in variant_names


# ── sort order ────────────────────────────────────────────────────────


class TestLoadLightweightOrdering:

    def test_pairs_are_sorted_by_directory_name(self, tmp_path):
        for name in ["CVE-2025-0003", "CVE-2025-0001", "CVE-2025-0002"]:
            _make_cve_dir(tmp_path, name, {**SAMPLE_DB, "cve_id": name})
        pairs = _make_dataset(tmp_path).load_lightweight()
        dir_names = [p.meta["dir_name"] for p in pairs]
        assert dir_names == sorted(dir_names)
