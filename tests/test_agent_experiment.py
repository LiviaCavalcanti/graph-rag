"""Tests for experiments/agent_experiment.run_experiment.

Verifies result structure (required fields in output JSONL) and resumability
(skipping already-completed queries when resumed).
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import networkx as nx
import numpy as np
import pytest

from data.base import FunctionPair


# ── helpers ───────────────────────────────────────────────────────────


def _make_pair(cve_id="CVE-2025-0001", cwe_id="CWE-119", variant="original", dir_name="CVE-2025-0001"):
    G = nx.MultiDiGraph()
    G.add_node("n1", labelV="METHOD")
    return FunctionPair(
        cve_id=cve_id,
        cwe_id=cwe_id,
        func_name="vuln_fn",
        project="autopatch",
        G_before=G,
        G_after=G,
        G_vuln=G,
        meta={"variant": variant, "dir_name": dir_name},
    )


def _minimal_cfg():
    return {
        "rag": {"top_k": 3},
        "embeddings": {"dim": 128},
        "experiment": {"split": {"enabled": False}},
        "data": {"autopatch": {"root": "/tmp/fake"}},
    }


def _fake_retriever_result(cve_id="CVE-2025-9999", cwe_id="CWE-787", variant="original"):
    return {
        "cve_id": cve_id,
        "cwe_id": cwe_id,
        "variant": variant,
        "dir_name": cve_id,
        "func_name": "helper_fn",
        "score": 0.95,
    }


# ── test result structure ─────────────────────────────────────────────


@patch("experiments.agent_experiment.get_ground_truth_patch", return_value="patch code")
@patch("experiments.agent_experiment._build_index_and_retriever")
@patch("experiments.agent_experiment.build_split")
@patch("experiments.agent_experiment.load_pairs")
def test_result_structure(mock_load_pairs, mock_build_split, mock_build_idx, mock_gt, tmp_path):
    """run_experiment writes JSONL records with required fields."""
    pairs = [_make_pair(), _make_pair(cve_id="CVE-2025-0002", dir_name="CVE-2025-0002")]

    mock_load_pairs.return_value = pairs
    mock_build_split.return_value = (pairs[:1], pairs[1:], {"test_ratio": 0.5})

    # Mock embedder and retriever
    mock_embedder = MagicMock()
    mock_embedder.embed_one.return_value = np.ones(128, dtype=np.float32)
    mock_retriever = MagicMock()
    mock_retriever.query.return_value = [_fake_retriever_result()]
    mock_build_idx.return_value = (mock_embedder, mock_retriever)

    # Patch make_run_dir to use tmp_path
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    with patch("src.io.batch.make_run_dir", return_value=("run_001", run_dir)):
        from experiments.agent_experiment import run_experiment

        result_dir = run_experiment(_minimal_cfg(), max_queries=2, batch_size=5)

    assert result_dir == run_dir

    # Verify JSONL output
    jsonl_path = run_dir / "results.jsonl"
    assert jsonl_path.exists()

    records = [json.loads(line) for line in jsonl_path.read_text().strip().splitlines()]
    assert len(records) >= 1

    rec = records[0]
    # Required top-level fields
    assert "query_cve" in rec
    assert "query_cwe" in rec
    assert "query_variant" in rec
    assert "status" in rec
    assert rec["status"] == "retrieved"
    # Retrieval sub-structure
    assert "retrieval" in rec
    assert "top_k" in rec["retrieval"]
    assert isinstance(rec["retrieval"]["top_k"], list)
    assert len(rec["retrieval"]["top_k"]) >= 1
    top_entry = rec["retrieval"]["top_k"][0]
    assert "rank" in top_entry
    assert "cve_id" in top_entry
    assert "score" in top_entry

    # Verify run metadata
    meta_path = run_dir / "run_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["mode"] == "retriever"
    assert meta["top_k"] == 3


# ── test resumability ─────────────────────────────────────────────────


@patch("experiments.agent_experiment.get_ground_truth_patch", return_value="patch code")
@patch("experiments.agent_experiment._build_index_and_retriever")
@patch("experiments.agent_experiment.build_split")
@patch("experiments.agent_experiment.load_pairs")
def test_resumability(mock_load_pairs, mock_build_split, mock_build_idx, mock_gt, tmp_path):
    """run_experiment skips already-completed queries when resuming."""
    pair_done = _make_pair(cve_id="CVE-2025-0001", variant="original", dir_name="CVE-2025-0001")
    pair_pending = _make_pair(cve_id="CVE-2025-0002", variant="original", dir_name="CVE-2025-0002")

    all_pairs = [pair_done, pair_pending]
    mock_load_pairs.return_value = all_pairs
    # All pairs are query pairs (no index/query distinction for test)
    mock_build_split.return_value = ([], all_pairs, {"test_ratio": 1.0})

    mock_embedder = MagicMock()
    mock_embedder.embed_one.return_value = np.ones(128, dtype=np.float32)
    mock_retriever = MagicMock()
    mock_retriever.query.return_value = [_fake_retriever_result()]
    mock_build_idx.return_value = (mock_embedder, mock_retriever)

    # Set up a run directory with one already-completed record
    run_dir = tmp_path / "resumed_run"
    run_dir.mkdir()
    jsonl_path = run_dir / "results.jsonl"
    existing_record = {
        "query_cve": "CVE-2025-0001",
        "query_cwe": "CWE-119",
        "query_variant": "original",
        "status": "retrieved",
    }
    jsonl_path.write_text(json.dumps(existing_record) + "\n")

    from experiments.agent_experiment import run_experiment

    result_dir = run_experiment(_minimal_cfg(), resume=str(run_dir), batch_size=5)

    assert result_dir == run_dir

    # Should have appended only the pending query
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2  # 1 pre-existing + 1 new

    new_record = json.loads(lines[1])
    assert new_record["query_cve"] == "CVE-2025-0002"
    assert new_record["status"] == "retrieved"

    # The first (already-done) pair should NOT have been re-processed
    # embedder.embed_one should only have been called once (for the pending pair)
    assert mock_embedder.embed_one.call_count == 1
