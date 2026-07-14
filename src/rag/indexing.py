"""Index-building pipeline: embed CPGs from active datasets → FAISS index.

Extracted from ``main.run_pipeline`` — the CLI's ``--mode index`` now just
calls :func:`build_index`.
"""

from __future__ import annotations

from pathlib import Path

from src.data import REGISTRY
from src.embeddings import build_embedders
from src.io.read_write import save_json
from src.rag.faiss_index import FAISSIndex
from src.schema_config import DatasetBatch, EmbeddedBatch, IndexUpdateResult

BATCH_SIZE = 32  # Embed in batches for efficiency


def build_index(cfg: dict) -> None:
    """Embed CPGs from ``cfg["data"]["active"]`` datasets and build the FAISS index."""
    active_datasets = [name for name in REGISTRY if name in cfg["data"]["active"]]
    rag_cfg = cfg["rag"]
    variant = rag_cfg["embedding_variant"]
    embedders = build_embedders(cfg)

    indexer = next((e for e in embedders if e.name == variant), None)
    if indexer is None:
        available = [e.name for e in embedders]
        raise ValueError(
            f"Embedding variant '{variant}' not found. Available embedders: {available}"
        )

    index = FAISSIndex(
        dim=cfg["embeddings"]["dim"],
        index_path=rag_cfg["index_path"],
        metadata_path=rag_cfg["metadata_path"],
    )

    total = 0
    contract_batches: list[DatasetBatch] = []
    contract_embedded: list[EmbeddedBatch] = []

    graph_cfg = cfg.get("graph", {})
    for ds_name in active_datasets:
        # Inject the shared graph-processing section so compute_graph_diff
        # parameters (slice_depth, ...) flow into the dataset stream.
        ds_cfg = {**cfg["data"][ds_name], "graph": graph_cfg}
        # instantiate the dataset class with the config params
        dataset = REGISTRY[ds_name](ds_cfg)
        print(f"-----------{dataset.name()}-----------")

        # Pre-fit PCA-based embedders on full corpus before batched indexing
        if hasattr(indexer, "fit") and not getattr(indexer, "_fitted", True):
            print(f"  [pre-fit] Loading all graphs to fit PCA embedder '{variant}'...")
            all_pairs = list(dataset.stream())
            all_graphs = [p.G_vuln for p in all_pairs]
            indexer.fit(all_graphs)
            print(f"  [pre-fit] PCA fitted on {len(all_graphs)} graphs")
        else:
            all_pairs = None

        ds_total = 0
        batch_pairs = []

        stream = iter(all_pairs) if all_pairs is not None else dataset.stream()
        for pair in stream:
            try:
                batch_pairs.append(pair)

                # Process batch when full
                if len(batch_pairs) >= BATCH_SIZE:
                    try:
                        graphs = [p.G_vuln for p in batch_pairs]
                        embeddings = indexer.embed_many(graphs)

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
                dim=cfg["embeddings"]["dim"],
                pairs=[],
                embeddings=[],
                metadata={
                    "dataset": ds_name,
                    "streaming": True,
                    "embedded_count": ds_total,
                },
            )
        )

    index.save()

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
