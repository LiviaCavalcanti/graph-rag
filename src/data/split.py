"""Dataset splitting utilities — stratified splits, sampling.

Generic and dataset-agnostic: ``build_split`` produces an index/query split
by CWE-stratified sampling. AutoPatch-specific "real vs. LLM-augmented
variant" splitting lives in ``AutoPatchDataset.build_split``
(``src/data/autopatch.py``); ``build_split`` here transparently delegates to
it whenever any of the given pairs came from AutoPatch
(``pair.project == "autopatch"``), so callers never need to know or care.
"""

from __future__ import annotations

import random
from collections import defaultdict


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

    Dataset-agnostic: produces a plain CWE-stratified (or shuffled, if
    ``stratified: false``) train/test split over all given pairs. If any of
    the pairs came from ``AutoPatchDataset`` (``pair.project ==
    "autopatch"``), delegates transparently to
    ``AutoPatchDataset.build_split`` instead, which understands the
    AutoPatch-specific real/augmented-variant split. Callers never need to
    branch on dataset type.
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

    if pairs and any(getattr(p, "project", None) == "autopatch" for p in pairs):
        from .autopatch import AutoPatchDataset

        return AutoPatchDataset.build_split(pairs, cfg, seed_override=seed_override)

    seed = seed_override if seed_override is not None else int(split_cfg.get("seed", 42))
    test_ratio = float(split_cfg.get("test_ratio", 0.2))
    stratified = bool(split_cfg.get("stratified", True))

    if stratified:
        index_pairs, query_pairs = _stratified_split(pairs, test_ratio, seed)
    else:
        rng = random.Random(seed)
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        cut = int(round(len(shuffled) * (1.0 - max(0.0, min(0.9, test_ratio)))))
        index_pairs, query_pairs = shuffled[:cut], shuffled[cut:]

    info = {
        "enabled": True,
        "seed": seed,
        "stratified": stratified,
        "test_ratio": test_ratio,
        "counts": {
            "total": len(pairs),
            "index_total": len(index_pairs),
            "query_total": len(query_pairs),
        },
    }
    return index_pairs, query_pairs, info

