"""
Shared experiment primitives — data loading, index building, evaluation, I/O.

Every experiment script should use these instead of reimplementing.
"""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.data.autopatch import AutoPatchDataset
from src.rag.hnsw import HNSWIndex
from src.rag.retriever import Retriever

OUTPUT_DIR = Path("experiments/output")


# ── config ───────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── run directory ────────────────────────────────────────────────────

def make_run_dir(tag: str = "") -> tuple[str, Path]:
    """Create a timestamped run directory.  Returns (run_id, run_dir)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    run_id = f"{ts}{suffix}_{uuid.uuid4().hex[:6]}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


# ── data loading & splitting ─────────────────────────────────────────

def load_pairs(cfg: dict) -> list:
    """Load all FunctionPair objects from the configured dataset."""
    ds = AutoPatchDataset(cfg["data"]["autopatch"])
    return ds.load_all()


def load_pairs_lightweight(cfg: dict) -> list:
    """Load pairs with metadata only — no CPG/graph loading (fast)."""
    ds = AutoPatchDataset(cfg["data"]["autopatch"])
    return ds.load_lightweight()


def _is_original(pair) -> bool:
    return pair.meta.get("variant") == "original"


def _split_by_variant(pairs):
    original = [p for p in pairs if _is_original(p)]
    augmented = [p for p in pairs if not _is_original(p)]
    return original, augmented


def _stratified_split(pairs, test_ratio, seed):
    if not pairs:
        return [], []
    if len(pairs) == 1:
        return pairs[:], []
    test_ratio = max(0.0, min(0.9, test_ratio))
    rng = random.Random(seed)

    by_cwe = defaultdict(list)
    for p in pairs:
        cwe = p.cwe_id if p.cwe_id and p.cwe_id != "UNKNOWN" else "__UNKNOWN__"
        by_cwe[cwe].append(p)

    train, test = [], []
    for cwe_pairs in by_cwe.values():
        items = cwe_pairs[:]
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio))
        if n > 1:
            n_test = max(1, min(n - 1, n_test))
        else:
            n_test = 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    if not test and len(pairs) > 1:
        rng.shuffle(train)
        test.append(train.pop())
    if not train and len(pairs) > 1:
        rng.shuffle(test)
        train.append(test.pop())

    return train, test


def _sample_pairs(pairs, keep_ratio, seed):
    if not pairs or keep_ratio >= 1.0:
        return pairs[:]
    if keep_ratio <= 0.0:
        return []
    rng = random.Random(seed)
    items = pairs[:]
    rng.shuffle(items)
    k = max(1, int(round(len(items) * keep_ratio)))
    return items[:k]


def build_split(pairs: list, cfg: dict, seed_override: int | None = None) -> tuple[list, list, dict]:
    """
    Split pairs into index / query sets.

    Backwards-compatible: returns (pairs, pairs, info) when split is disabled.
    """
    split_cfg = (cfg or {}).get("experiment", {}).get("split", {})
    enabled = bool(split_cfg.get("enabled", False))

    if not enabled:
        return pairs[:], pairs[:], {
            "enabled": False,
            "index_n": len(pairs),
            "query_n": len(pairs),
            "mode": "all_vs_all",
        }

    seed = seed_override if seed_override is not None else int(split_cfg.get("seed", 42))
    test_ratio = float(split_cfg.get("test_ratio", 0.2))
    stratified = bool(split_cfg.get("stratified", True))
    include_real = bool(split_cfg.get("include_real_in_index", True))
    aug_train_ratio = float(split_cfg.get("augmented_train_ratio", 1.0))
    query_source = str(split_cfg.get("query_source", "augmented_test"))

    real, aug = _split_by_variant(pairs)

    if stratified:
        aug_train, aug_test = _stratified_split(aug, test_ratio, seed)
    else:
        rng = random.Random(seed)
        shuffled = aug[:]
        rng.shuffle(shuffled)
        cut = int(round(len(shuffled) * (1.0 - max(0.0, min(0.9, test_ratio)))))
        aug_train, aug_test = shuffled[:cut], shuffled[cut:]

    aug_train_kept = _sample_pairs(aug_train, aug_train_ratio, seed + 13)

    index_pairs = []
    if include_real:
        index_pairs.extend(real)
    index_pairs.extend(aug_train_kept)

    if query_source == "augmented_test":
        query_pairs = aug_test
    elif query_source == "all_test":
        _, real_test = _stratified_split(real, test_ratio, seed + 31)
        query_pairs = aug_test + real_test
    elif query_source == "augmented_train":
        query_pairs = aug_train_kept
    else:
        query_pairs = aug_test

    if not index_pairs:
        index_pairs = real[:] if real else aug_train_kept[:]
    if not query_pairs:
        query_pairs = aug_test[:] if aug_test else index_pairs[:]

    info = {
        "enabled": True,
        "seed": seed,
        "stratified": stratified,
        "test_ratio": test_ratio,
        "query_source": query_source,
        "include_real_in_index": include_real,
        "augmented_train_ratio": aug_train_ratio,
        "counts": {
            "total": len(pairs),
            "real_total": len(real),
            "aug_total": len(aug),
            "aug_train_total": len(aug_train),
            "aug_train_used": len(aug_train_kept),
            "aug_test_total": len(aug_test),
            "index_total": len(index_pairs),
            "query_total": len(query_pairs),
        },
    }
    return index_pairs, query_pairs, info


# ── index building ───────────────────────────────────────────────────

from src.rag.utils import populate_index


def build_hnsw(
    pairs: list,
    embeddings: np.ndarray,
    embedder_name: str,
    dim: int,
    run_dir: Path,
    tag: str = "",
) -> tuple[HNSWIndex, Retriever]:
    """Build, save, reload an HNSW index.  Returns (index, retriever)."""
    idx_dir = run_dir / "indices"
    idx_dir.mkdir(exist_ok=True)
    stem = f"{embedder_name}__{tag}" if tag else embedder_name
    index = HNSWIndex(
        dim=dim,
        index_path=str(idx_dir / f"{stem}__hnsw.index"),
        metadata_path=str(idx_dir / f"{stem}__hnsw_meta.json"),
    )
    retriever = populate_index(index, pairs, embeddings, embedder_name)
    return index, retriever


# ── evaluation primitives (canonical home: src/metrics/retrieval_eval) ─
from src.metrics.retrieval_eval import evaluate_cwe_recall, evaluate_retrieval

# ── uncertainty helpers (used by analyze_misses & verify_crossing) ───

def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over retrieval scores."""
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float64) / temperature
    arr -= arr.max()
    exp = np.exp(arr)
    return (exp / exp.sum()).tolist()


def is_uncertain(prob: float, margin: float, prob_floor: float = 0.12, margin_floor: float = 0.005) -> bool:
    return prob < prob_floor or margin < margin_floor


# ── I/O helpers (canonical home: src/io) ─────────────────────────────
from src.io import read_code_file, save_json
