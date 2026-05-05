"""Tests for src/io — save_json, read_code_file, make_run_dir, load_config."""

import json

import pytest
from pathlib import Path

from src.io import save_json, read_code_file, make_run_dir, load_config


# ── save_json ────────────────────────────────────────────────────────


class TestSaveJson:
    def test_writes_dict(self, tmp_path):
        p = tmp_path / "out.json"
        save_json({"key": "value"}, p)
        assert json.loads(p.read_text()) == {"key": "value"}

    def test_writes_list(self, tmp_path):
        p = tmp_path / "out.json"
        save_json([1, 2, 3], p)
        assert json.loads(p.read_text()) == [1, 2, 3]


# ── read_code_file ───────────────────────────────────────────────────


class TestReadCodeFile:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "code.c"
        f.write_text("int main() { return 0; }")
        assert read_code_file(str(f)) == "int main() { return 0; }"

    def test_truncates_long_file(self, tmp_path):
        f = tmp_path / "code.c"
        f.write_text("x" * 5000)
        result = read_code_file(str(f), max_chars=100)
        assert len(result) < 5000
        assert "truncated" in result

    def test_none_path(self):
        assert read_code_file(None) == ""

    def test_empty_string(self):
        assert read_code_file("") == ""

    def test_nonexistent_short_path(self):
        assert read_code_file("/no/file") == ""

    def test_inline_string_fallback(self):
        # long string that isn't a real file → returned as-is
        code = "int foo() { " * 10
        assert read_code_file(code) == code


# ── make_run_dir ─────────────────────────────────────────────────────


class TestMakeRunDir:
    def test_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.io.results.OUTPUT_DIR", tmp_path)
        run_id, run_dir = make_run_dir("test")
        assert run_dir.exists()
        assert "test" in run_id

    def test_unique_ids(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.io.results.OUTPUT_DIR", tmp_path)
        id1, _ = make_run_dir()
        id2, _ = make_run_dir()
        assert id1 != id2

    def test_default_tag_no_double_underscore(self, tmp_path, monkeypatch):
        """Empty tag should not produce a double underscore in run_id."""
        monkeypatch.setattr("src.io.results.OUTPUT_DIR", tmp_path)
        run_id, _ = make_run_dir()
        assert "__" not in run_id.split("_", 2)[0:2]  # no adjacent empty segment

    def test_custom_tag_in_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.io.results.OUTPUT_DIR", tmp_path)
        run_id, _ = make_run_dir("myexp")
        assert "myexp" in run_id

    def test_return_types(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.io.results.OUTPUT_DIR", tmp_path)
        run_id, run_dir = make_run_dir("t")
        assert isinstance(run_id, str)
        assert isinstance(run_dir, Path)

    def test_directory_is_real(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.io.results.OUTPUT_DIR", tmp_path)
        _, run_dir = make_run_dir("x")
        assert run_dir.is_dir()

    def test_uniqueness_rapid(self, tmp_path, monkeypatch):
        """10 rapid calls produce 10 distinct run IDs and directories."""
        monkeypatch.setattr("src.io.results.OUTPUT_DIR", tmp_path)
        results = [make_run_dir("r") for _ in range(10)]
        ids = [r[0] for r in results]
        dirs = [r[1] for r in results]
        assert len(set(ids)) == 10
        assert len(set(dirs)) == 10


# ── load_config ──────────────────────────────────────────────────────


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("data:\n  autopatch:\n    root: /tmp\n")
        cfg = load_config(str(f))
        assert cfg["data"]["autopatch"]["root"] == "/tmp"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/cfg.yaml")

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        assert load_config(str(f)) is None
