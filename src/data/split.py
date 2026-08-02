"""Dataset splitting utilities — stratified splits, sampling.

Generic and dataset-agnostic: ``build_split`` produces an index/query split
by CWE-stratified sampling. AutoPatch-specific "real vs. LLM-augmented
variant" splitting lives in ``AutoPatchDataset.build_split``
(``src/data/autopatch.py``); ``build_split`` here transparently delegates to
it whenever any of the given pairs came from AutoPatch
(``pair.project == "autopatch"``), so callers never need to know or care.

A split can also be *anchored* to a previously-generated ``split_info.json``
(see ``experiment.split.anchor_split_path`` in ``config.yaml``) so that a run
reproduces the exact index/query partition of an earlier experiment instead
of resampling. See ``_anchor_split`` below.

A split can also be pinned to a *precomputed* ``split_info_<variant>.json``
directory (see ``experiment.split.precomputed_split_dir`` /
``precomputed_split_variant``) — the format written by
``utils/build_balanced_split.py`` (``index``/``query`` lists of
``{cve_id, func_name, ...}``), also used standalone by
``cvefixes_experiments/scripts/performance/exp_file_method_interface.py``.
This is the mechanism to use when the SAME fixed split must be reused across
multiple independent runs/variants (e.g. an embedding sweep). See
``_precomputed_split`` below. Precedence in ``build_split``:
``precomputed_split_dir`` > ``anchor_split_path`` > fresh stratified/random
split.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


# Precomputed split_info filenames keyed by `precomputed_split_variant`
# (matches utils/build_balanced_split.py's output naming).
_SPLIT_VARIANT_FILENAMES = {
    "balanced": "split_info_balanced.json",
    "stratified": "split_info_stratified.json",
}


def _entry_key(cve_id, func_name):
    return (cve_id or "", func_name or "")


def _entry_key_from_spec(entry: dict) -> tuple[str, str]:
    """Normalize split-entry schemas to a (cve_id, func_name) key.

    Supports both the modern split format (func_name) and older pipeline
    entry files that store method_name/function_name instead.
    """
    func_name = (
        entry.get("func_name")
        or entry.get("method_name")
        or entry.get("function_name")
        or ""
    )
    return _entry_key(entry.get("cve_id"), func_name)


def _pair_keys(p) -> set[tuple[str, str]]:
    """All (cve_id, func_name)-style keys a pair could match against a
    precomputed split spec. Method-level pairs match on their own
    ``func_name``; file-level pairs (merged from several methods) also match
    via ``meta["method_names"]`` so a precomputed method-level split can
    still resolve file-level pairs (see exp_file_method_interface.py).
    """
    method_names = p.meta.get("method_names") if getattr(p, "meta", None) else None
    if method_names:
        return {_entry_key(p.cve_id, m) for m in method_names}
    return {_entry_key(p.cve_id, p.func_name)}


def _precomputed_split(
    pairs: list, split_dir: str, variant: str
) -> tuple[list, list, dict]:
    """Partition ``pairs`` using a precomputed ``split_info_<variant>.json``
    (as written by ``utils/build_balanced_split.py``): ``{"index": [...],
    "query": [...]}``, each entry a ``{cve_id, func_name, ...}`` dict.

    Unlike ``_anchor_split``, this fails fast (raises) if the file is
    missing — the caller asked for THIS exact split explicitly, so silently
    falling back to a fresh/different split would be surprising.
    """
    variant = (variant or "balanced").lower()
    fname = _SPLIT_VARIANT_FILENAMES.get(variant)
    if fname is None:
        raise ValueError(
            f"Unknown precomputed_split_variant '{variant}', expected one "
            f"of {sorted(_SPLIT_VARIANT_FILENAMES)}"
        )

    split_path = Path(split_dir) / fname
    if not split_path.is_file():
        raise FileNotFoundError(f"Precomputed split file not found: {split_path}")

    spec = json.loads(split_path.read_text(encoding="utf-8"))

    index_entries = spec.get("index")
    query_entries = spec.get("query")

    # Backward compatibility: some precomputed split dirs store only split
    # counts in split_info_*.json and keep actual entries in companion files.
    if index_entries is None or query_entries is None:
        index_entries_path = Path(split_dir) / "index_pairs_entries.json"
        query_entries_path = Path(split_dir) / "query_pairs_entries.json"
        if index_entries_path.is_file() and query_entries_path.is_file():
            try:
                index_entries = json.loads(index_entries_path.read_text(encoding="utf-8")).get("entries", [])
                query_entries = json.loads(query_entries_path.read_text(encoding="utf-8")).get("entries", [])
                print(
                    "  [precomputed_split] Using companion entry files "
                    "index_pairs_entries.json/query_pairs_entries.json"
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Precomputed split file lacks index/query lists and companion "
                    f"entry files failed to load from {split_dir}: {exc}"
                ) from exc
        else:
            raise ValueError(
                "Precomputed split file must include 'index' and 'query' lists, "
                "or provide companion files index_pairs_entries.json and "
                "query_pairs_entries.json in the same directory. "
                f"Missing in: {split_dir}"
            )

    index_keys = {_entry_key_from_spec(e) for e in index_entries}
    query_keys = {_entry_key_from_spec(e) for e in query_entries}

    overlap = index_keys & query_keys
    if overlap:
        print(
            f"  [precomputed_split] WARNING: {len(overlap)} (cve, func) keys "
            "appear in both index and query specs; treating as index-only."
        )
        query_keys -= overlap

    index_pairs = [p for p in pairs if _pair_keys(p) & index_keys]
    query_pairs = [p for p in pairs if _pair_keys(p) & query_keys]

    matched_index = {k for p in index_pairs for k in _pair_keys(p)} & index_keys
    matched_query = {k for p in query_pairs for k in _pair_keys(p)} & query_keys
    missing_index = len(index_keys) - len(matched_index)
    missing_query = len(query_keys) - len(matched_query)
    if missing_index or missing_query:
        print(
            f"  [precomputed_split] split_dir={split_dir} variant={variant} "
            f"unresolved_split_entries(index={missing_index}, query={missing_query}) "
            "— pairs not present in the loaded pool"
        )

    info = {
        "enabled": True,
        "mode": "precomputed",
        "source": "precomputed",
        "precomputed_split_dir": str(split_dir),
        "precomputed_split_variant": variant,
        "counts": {
            "total": len(index_pairs) + len(query_pairs),
            "index_total": len(index_pairs),
            "query_total": len(query_pairs),
        },
        "unresolved": {
            "split_index_entries": missing_index,
            "split_query_entries": missing_query,
        },
    }
    return index_pairs, query_pairs, info


def _anchor_split(pairs: list, anchor_path: str) -> tuple[list, list, dict] | None:
    """Partition ``pairs`` to reproduce a baseline split from ``split_info.json``.

    Reads ``index_entries``/``query_entries`` (each a ``{cve_id, cwe_id,
    func_name}`` triple) from the anchor file and matches them against the
    given pairs by ``(cve_id, func_name)``. Pairs not found in either list are
    dropped; anchor entries not resolved against ``pairs`` are reported but
    otherwise ignored (e.g. a different dataset subset was loaded).

    Returns ``None`` if the anchor file is missing/unreadable so callers can
    fall back to a fresh stratified split.
    """
    path = Path(anchor_path)
    if not path.is_file():
        print(f"  [anchor_split] file not found, falling back to fresh split: {anchor_path}")
        return None

    try:
        si = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [anchor_split] failed to read/parse {anchor_path}: {exc}")
        return None

    index_keys = {_entry_key(e.get("cve_id"), e.get("func_name")) for e in si.get("index_entries", [])}
    query_keys = {_entry_key(e.get("cve_id"), e.get("func_name")) for e in si.get("query_entries", [])}

    index_pairs, query_pairs = [], []
    seen_index, seen_query = set(), set()
    for p in pairs:
        key = _entry_key(p.cve_id, p.func_name)
        if key in index_keys:
            index_pairs.append(p)
            seen_index.add(key)
        elif key in query_keys:
            query_pairs.append(p)
            seen_query.add(key)

    missing_index = len(index_keys) - len(seen_index)
    missing_query = len(query_keys) - len(seen_query)
    unmatched = len(pairs) - len(index_pairs) - len(query_pairs)
    if missing_index or missing_query or unmatched:
        print(
            f"  [anchor_split] anchor={anchor_path} "
            f"unresolved_anchor_entries(index={missing_index}, query={missing_query}) "
            f"pairs_not_in_anchor={unmatched}"
        )

    info = {
        "enabled": True,
        "mode": "anchored",
        "anchor_split_path": str(anchor_path),
        "counts": {
            "total": len(index_pairs) + len(query_pairs),
            "index_total": len(index_pairs),
            "query_total": len(query_pairs),
        },
        "unresolved": {
            "anchor_index_entries": missing_index,
            "anchor_query_entries": missing_query,
            "pairs_not_in_anchor": unmatched,
        },
    }
    return index_pairs, query_pairs, info


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

    If ``experiment.split.precomputed_split_dir`` is set, reproduces that
    fixed ``split_info_<precomputed_split_variant>.json`` partition (matched
    by ``(cve_id, func_name)``, with a ``method_names`` fallback for
    file-level pairs) instead of resampling — see ``_precomputed_split``.
    This takes priority over ``anchor_split_path`` and raises if the file is
    missing (fail fast — the caller asked for THIS exact split).

    If ``experiment.split.anchor_split_path`` is set, reproduces that
    baseline's exact index/query partition (matched by ``(cve_id,
    func_name)``) instead of resampling — see ``_anchor_split``. Falls back to
    the normal stratified/random split if the anchor file can't be read.
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

    precomputed_dir = split_cfg.get("precomputed_split_dir")
    if precomputed_dir:
        return _precomputed_split(
            pairs, precomputed_dir, split_cfg.get("precomputed_split_variant", "balanced")
        )

    anchor_path = split_cfg.get("anchor_split_path")
    if anchor_path:
        anchored = _anchor_split(pairs, anchor_path)
        if anchored is not None:
            return anchored

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

