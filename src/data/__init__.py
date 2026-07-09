"""Dataset registry and unified loader."""

from __future__ import annotations

from .autopatch import AutoPatchDataset
from .base import BaseDataset, FunctionPair
from .cvefixes import CVEFixesDataset

REGISTRY: dict[str, type[BaseDataset]] = {
    "autopatch": AutoPatchDataset,
    "cvefixes": CVEFixesDataset,
}


def _resolve_datasets(cfg: dict) -> list[tuple[str, "BaseDataset"]]:
    """Instantiate active datasets from config."""
    data_cfg = cfg["data"]
    graph_cfg = cfg.get("graph", {})
    active = data_cfg.get("active", [k for k in data_cfg if k in REGISTRY])
    datasets = []
    for name in active:
        if name not in REGISTRY:
            continue
        # Inject the shared graph-processing section so compute_graph_diff
        # parameters (slice_depth, ...) flow into the dataset stream.
        ds_cfg = {**data_cfg[name], "graph": graph_cfg} if graph_cfg else data_cfg[name]
        datasets.append((name, REGISTRY[name](ds_cfg)))
    return datasets


def load_pairs(cfg: dict) -> list[FunctionPair]:
    """Load FunctionPair objects from all active datasets.

    Reads ``cfg["data"]["active"]`` to determine which datasets to load.
    If the key is absent, loads all datasets that have a config section
    and are present in the registry.
    """
    pairs: list[FunctionPair] = []
    for name, ds in _resolve_datasets(cfg):
        loaded = ds.load_all()
        print(f"  [{ds.name()}] loaded {len(loaded)} pairs")
        pairs.extend(loaded)

    if not pairs:
        raise RuntimeError(
            f"No pairs loaded. "
            f"available={list(REGISTRY.keys())}"
        )
    return pairs


def load_pairs_lightweight(cfg: dict) -> list[FunctionPair]:
    """Load pairs with metadata only — no CPG/graph loading.

    Same dispatch as load_pairs but calls load_lightweight() on each
    dataset, which skips expensive Joern graph parsing.
    """
    pairs: list[FunctionPair] = []
    for name, ds in _resolve_datasets(cfg):
        loaded = ds.load_lightweight()
        print(f"  [{ds.name()}] loaded {len(loaded)} pairs (lightweight)")
        pairs.extend(loaded)

    if not pairs:
        raise RuntimeError(
            f"No pairs loaded (lightweight). "
            f"available={list(REGISTRY.keys())}"
        )
    return pairs
