"""Dataset splitting utilities — variant filtering, stratified splits, sampling."""

from __future__ import annotations

import random
from collections import defaultdict


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
