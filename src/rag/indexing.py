"""Index-building pipeline: embed CPGs from active datasets → FAISS index.

Extracted from ``main.run_pipeline`` — the CLI's ``--mode index`` now just
calls :func:`build_index`.
"""

from __future__ import annotations

from pathlib import Path

from src.data import _resolve_datasets
from src.embeddings import build_embedders
from src.io.read_write import save_json
from src.rag.faiss_index import FAISSIndex
from src.schema_config import DatasetBatch, EmbeddedBatch, IndexUpdateResult

BATCH_SIZE = 32  # Embed in batches for efficiency


def build_index(cfg: dict) -> None:
    """Embed CPGs from ``cfg["data"]["active"]`` datasets and build the FAISS index."""
    rag_cfg = cfg["rag"]
    variant = rag_cfg["embedding_variant"]
    embedders = build_embedders(cfg)

    indexer = next((e for e in embedders if e.name == variant), None)
    if indexer is None:
        available = [e.name for e in embedders]
        raise ValueError(
            f"Embedding variant '{variant}' not found. Available embedders: {available}"
        )

    index = None

    total = 0
    contract_batches: list[DatasetBatch] = []
    contract_embedded: list[EmbeddedBatch] = []

    # Determine whether a train/test split should restrict the index to the
    # support set only.  When enabled, query pairs are held out so retrieval
    # metrics are measured on truly unseen examples.
    split_cfg = cfg.get("experiment", {}).get("split", {})
    split_enabled = bool(
        split_cfg.get("enabled") or split_cfg.get("precomputed_split_dir")
    )
    if split_enabled:
        from src.data.split import build_split

    datasets = _resolve_datasets(cfg)
    active_datasets = [ds_name for ds_name, _ds in datasets]
    for ds_name, dataset in datasets:
        print(f"-----------{dataset.name()}-----------")

        # Always load all pairs upfront so we can (a) apply the split and
        # (b) pre-fit PCA-based embedders.  For non-PCA embedders without a
        # split this adds a small memory cost but keeps the code uniform.
        all_pairs = list(dataset.stream())

        if split_enabled:
            all_pairs, _, split_info = build_split(all_pairs, cfg)
            print(
                f"  [split] Indexing {len(all_pairs)} support-set pairs "
                f"({split_info.get('mode', 'split')} mode — query pairs excluded to prevent leakage)"
            )

        # Pre-fit PCA-based embedders on the support-set corpus (index pairs
        # only when a split is active; full corpus otherwise).
        if indexer.requires_fitting:
            print(f"  [pre-fit] Loading all graphs to fit PCA embedder '{variant}'...")
            all_graphs = [p.G_vuln for p in all_pairs]
            indexer.fit(all_graphs)
            print(f"  [pre-fit] PCA fitted on {len(all_graphs)} graphs")

        ds_total = 0
        batch_pairs = []

        stream = iter(all_pairs)
        for pair in stream:
            try:
                batch_pairs.append(pair)

                # Process batch when full
                if len(batch_pairs) >= BATCH_SIZE:
                    try:
                        graphs = [p.G_vuln for p in batch_pairs]
                        embeddings = indexer.embed_many(graphs)

                        if index is None:
                            index = FAISSIndex(
                                dim=int(embeddings.shape[1]),
                                index_path=rag_cfg["index_path"],
                                metadata_path=rag_cfg["metadata_path"],
                            )
                        elif embeddings.shape[1] != index.dim:
                            raise ValueError(
                                f"Embedding dimension changed within index build: "
                                f"expected {index.dim}, got {embeddings.shape[1]}"
                            )

                        for p, emb in zip(batch_pairs, embeddings):
                            index.add(p, emb, variant)
                            total += 1
                            ds_total += 1

                        if total % 50 == 0:
                            print(f" indexed {total} pairs.. ")
                        batch_pairs = []
                    except Exception as e:
                        print(f"   batch error: {e}")
                        batch_pairs = []

            except Exception as e:
                print(f"   skip {pair.cve_id} / {pair.func_name}:  {e}")

        # Process remaining pairs in final batch
        if batch_pairs:
            try:
                graphs = [p.G_vuln for p in batch_pairs]
                embeddings = indexer.embed_many(graphs)

                if index is None:
                    index = FAISSIndex(
                        dim=int(embeddings.shape[1]),
                        index_path=rag_cfg["index_path"],
                        metadata_path=rag_cfg["metadata_path"],
                    )
                elif embeddings.shape[1] != index.dim:
                    raise ValueError(
                        f"Embedding dimension changed within index build: "
                        f"expected {index.dim}, got {embeddings.shape[1]}"
                    )

                for p, emb in zip(batch_pairs, embeddings):
                    index.add(p, emb, variant)
                    total += 1
                    ds_total += 1

                print(f" indexed {total} pairs.. (final batch)")
            except Exception as e:
                print(f"   final batch error: {e}")

        contract_batches.append(
            DatasetBatch(
                batch_id=f"{ds_name}-index",
                run_id="index",
                pairs=[],
                metadata={
                    "dataset": ds_name,
                    "streaming": True,
                    "indexed_count": ds_total,
                },
            )
        )
        contract_embedded.append(
            EmbeddedBatch(
                batch_id=f"{ds_name}-embed",
                run_id="index",
                embedder_name=variant,
                embedder_version=None,
                dim=index.dim if index is not None else cfg["embeddings"]["dim"],
                pairs=[],
                embeddings=[],
                metadata={
                    "dataset": ds_name,
                    "streaming": True,
                    "embedded_count": ds_total,
                },
            )
        )

    if index is None or total == 0:
        raise RuntimeError(
            "No vectors were added to the FAISS index. "
            "Check the active dataset, split settings, or embedder output dimension."
        )

    index.save()

    # Persist the fitted PCA alongside the index so query/batch runs can
    # load it and transform-only without re-fitting on the wrong data.
    pca_state = indexer.get_pca_state()
    if pca_state is not None:
        index.save_embedder_state(variant, pca_state)

    index_contract = IndexUpdateResult(
        run_id="index",
        index_backend="faiss",
        index_path=Path(rag_cfg["index_path"]),
        index_version=variant,
        added_count=total,
        total_count=total,
        metadata_path=Path(rag_cfg["metadata_path"]),
        metadata={"active_datasets": active_datasets},
    )

    contracts_path = Path(rag_cfg["metadata_path"]).with_name(
        f"{Path(rag_cfg['metadata_path']).stem}_contracts.json"
    )
    contracts_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "dataset_batches": [b.__dict__ for b in contract_batches],
            "embedded_batches": [b.__dict__ for b in contract_embedded],
            "index_update": index_contract.__dict__,
        },
        contracts_path,
    )
    print(f"\nDone. \nTotal indexed: {total}")
    print(f"Contracts snapshot: {contracts_path}")
