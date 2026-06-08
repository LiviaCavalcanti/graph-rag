import argparse
import json
from enum import Enum, auto
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.data.autopatch import AutoPatchDataset
from src.data.base import ExportJob
from src.data.cvefixes import CVEFixesDataset
from src.data.pipeline import run_joern_export, write_c_file
from src.embeddings import build_embedders
from src.rag.faiss_index import FAISSIndex
from src.schema_config import (
    AppConfig,
    DatasetBatch,
    EmbeddedBatch,
    EmbeddingConfig,
    GraphProcessingConfig,
    IndexUpdateResult,
    PathsConfig,
    RetrievalResult,
    VariantConfig,
)

DATASETS = {"autopatch": AutoPatchDataset, "cvefixes": CVEFixesDataset}


class JobStatus(Enum):
    OK = auto()
    SKIPPED = auto()
    FAILED = auto()


def _validate_cfg(cfg: dict) -> None:
    """Fail fast on missing top-level config sections used by main modes."""
    required = ["data", "rag", "embeddings"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    if "joern" not in cfg and "paths" not in cfg:
        raise KeyError("Missing joern/path configuration. Need cfg['joern'] or cfg['paths']")


def _instantiate_app_config(cfg: dict) -> AppConfig:
    """Instantiate typed AppConfig from current (legacy-compatible) YAML shape."""
    paths_raw = cfg.get("paths", {})
    joern_bin = paths_raw.get("joern_bin_dir") or cfg.get("joern", {}).get("bin_dir")
    if not joern_bin:
        raise KeyError("Missing Joern binary path. Set joern.bin_dir or paths.joern_bin_dir")

    rag_cfg = cfg.get("rag", {})
    index_path = rag_cfg.get("index_path", "indexes/faiss.index")
    inferred_index_dir = str(Path(index_path).parent) if index_path else "indexes"

    paths_cfg = PathsConfig(
        joern_bin_dir=Path(joern_bin),
        output_dir=Path(paths_raw.get("output_dir", "experiments/output")),
        models_cache_dir=Path(paths_raw.get("models_cache_dir", "models")),
        index_dir=Path(paths_raw.get("index_dir", inferred_index_dir)),
    )

    graph_raw = cfg.get("graph_processing", {})
    graph_cfg = GraphProcessingConfig(
        slice_depth=graph_raw.get("slice_depth", 3),
        change_weight=graph_raw.get(
            "change_weight",
            {
                "function_added": 1.0,
                "function_deleted": 1.0,
                "parameter_changed": 0.5,
            },
        ),
        noise_types=graph_raw.get("noise_types", ["add_noise", "drop_noise"]),
    )

    emb_root = cfg.get("embeddings", {})
    active_embedders = emb_root.get("active", [])
    if not active_embedders and cfg.get("rag", {}).get("embedding_variant"):
        active_embedders = [cfg["rag"]["embedding_variant"]]

    embeddings_cfg: dict[str, EmbeddingConfig] = {}
    for name in active_embedders:
        sub = emb_root.get(name, {}) if isinstance(emb_root.get(name, {}), dict) else {}
        model_checkpoint = sub.get("checkpoint_path") or sub.get("model_checkpoint")
        model_name = sub.get("model_name") or sub.get("model_path") or name

        embeddings_cfg[name] = EmbeddingConfig(
            variant=name,
            dim=emb_root.get("dim", 128),
            model_name=str(model_name),
            model_checkpoint=Path(model_checkpoint) if model_checkpoint else None,
            wl_iterations=emb_root.get("wl", {}).get("num_iterations", 4),
            wl_color_space=emb_root.get("wl", {}).get("color_space", 8192),
            hidden_dim=sub.get("hidden_dim", emb_root.get("wl", {}).get("hidden_dim", 64)),
        )

    variants_cfg: list[VariantConfig] = []
    for raw in cfg.get("variants", []):
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        model = raw.get("model")
        if not name or not model:
            continue
        variants_cfg.append(
            VariantConfig(
                name=name,
                model=model,
                llm_output_file=raw.get("llm_output_file", f"{name}_response.json"),
                patch_file=raw.get("patch_file", f"{name}_patch.py"),
            )
        )

    return AppConfig(
        paths=paths_cfg,
        graph=graph_cfg,
        embeddings=embeddings_cfg,
        variants=variants_cfg,
        rag=cfg.get("rag", {}),
        data=cfg.get("data", {}),
    )


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON with safe fallback for non-serializable objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def apply_split_overrides(cfg: dict, args) -> None:
    """Apply CLI split/augmentation overrides to the experiment config."""
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("split", {})
    split_cfg = cfg["experiment"]["split"]
    if args.split:
        split_cfg["enabled"] = True
    if args.no_split:
        split_cfg["enabled"] = False
    if args.split_test_ratio is not None:
        split_cfg["test_ratio"] = args.split_test_ratio
    if args.aug_train_ratio is not None:
        split_cfg["augmented_train_ratio"] = args.aug_train_ratio


def _process_job(job: ExportJob, joern_bin_dir: str) -> tuple[JobStatus, str]:
    out_dir = Path(job.out_dir)
    label = f"{job.cve_id}/{job.variant}/{job.version}"
    existing = list(out_dir.glob("**/export.xml"))
    if existing:
        return JobStatus.SKIPPED, f"skip {label} already exists"

    c_file = out_dir / f"{job.func_name or 'function'}.cpp"
    graph_folder = out_dir / "graph"

    try:
        write_c_file(job.source_code, c_file, supplementary_code=job.supplementary_code)
    except Exception as e:
        return JobStatus.FAILED, f"write failed {job.cve_id}: {e}"

    success = run_joern_export(joern_bin_dir, c_file, str(out_dir), str(graph_folder))
    if success:
        return JobStatus.OK, f"ok {label}"
    return JobStatus.FAILED, f"FAIL {label}"


def run_export(cfg: dict, dataset_name: str | None = None):
    joern_bin_dir = str(
        cfg.paths.joern_bin_dir
    )
    if not joern_bin_dir:
        raise KeyError("Joern path not found. Set paths.joern_bin_dir or joern.bin_dir")
    # workers = cfg["joern"].get("workers", max(1, cpu_count() - 1))
    workers = max(1, cpu_count() - 1)
    active = [dataset_name] if dataset_name else list(DATASETS.keys())
    active = [n for n in active if cfg.data.get(n)]

    for ds_name in active:
        ds_cfg = cfg.data[ds_name]
        dataset = DATASETS[ds_name](ds_cfg)
        graphml_root = ds_cfg["graphml_root"]

        print(f"\n -------------- exporting {dataset.name()}--------------")
        jobs = list(dataset.export_jobs(graphml_root))
        print(f" {len(jobs)} jobs & {workers} workers")

        worker_fn = partial(_process_job, joern_bin_dir=joern_bin_dir)

        ok = fail = skipped = 0
        with Pool(processes=workers) as pool:
            with tqdm(total=len(jobs), desc=ds_name, unit="job") as pbar:
                for status, msg in pool.imap_unordered(worker_fn, jobs, chunksize=4):
                    if status == JobStatus.SKIPPED:
                        skipped += 1
                    elif status == JobStatus.FAILED:
                        fail += 1
                        tqdm.write(f" {msg}")
                    else:
                        ok += 1
                    pbar.set_postfix(ok=ok, skip=skipped, fail=fail)
                    pbar.update(1)
        print(f"Done \n    ok: {ok}  -  skipped: {skipped}  -  fail: {fail}")


def run_pipeline(cfg):
    active_datasets = [name for name in DATASETS if name in cfg["data"]["active"]]
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

    for ds_name in active_datasets:
        ds_cfg = cfg["data"][ds_name]
        # instantiate the dataset class with the config params
        dataset = DATASETS[ds_name](ds_cfg)
        print(f"-----------{dataset.name()}-----------")

        ds_total = 0

        for pair in dataset.stream():
            try:
                emb = indexer.embed_one(pair.G_vuln)
                index.add(pair, emb, variant)  # index to RAG
                total += 1
                ds_total += 1
                if total % 5 == 0:
                    print(f" indexed {total} pairs.. ")
            except Exception as e:
                print(f"   skip {pair.cve_id} / {pair.func_name}:  {e}")

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
    _write_json(
        contracts_path,
        {
            "dataset_batches": [b.__dict__ for b in contract_batches],
            "embedded_batches": [b.__dict__ for b in contract_embedded],
            "index_update": index_contract.__dict__,
        },
    )
    print(f"\nDone. \nTotal indexed: {total}")
    print(f"Contracts snapshot: {contracts_path}")


# def run_query(cfg: dict, cve_id: str):
#     from src.rag.retriever import Retriever

#     rag_cfg = cfg["rag"]
#     index = FAISSIndex(
#         dim=cfg["embeddings"]["dim"],
#         index_path=rag_cfg["index_path"],
#         metadata_path=rag_cfg["metadata_path"],
#     )
#     index.load()
#     retriever = Retriever(index, top_k=rag_cfg["top_k"])
#     raw_results = retriever.query_by_cve(cve_id)
#     for r in raw_results:
#         print(r)

#     retrieval_contract = RetrievalResult(
#         run_id="query",
#         query_id=cve_id,
#         query_cve=cve_id,
#         retriever_name="metadata_lookup",
#         top_k=len(raw_results),
#         hit_ids=[str(r.get("_idx", i)) for i, r in enumerate(raw_results)],
#         hit_scores=[float(r.get("score", 1.0)) for r in raw_results],
#         hit_metadata=raw_results,
#         metadata={"result_count": len(raw_results)},
#     )

#     query_contract_path = Path(rag_cfg["metadata_path"]).with_name(
#         f"query_{cve_id}_retrieval_contract.json"
#     )
#     _write_json(query_contract_path, retrieval_contract.__dict__)
#     print(f"Retrieval contract: {query_contract_path}")


# def run_batch_query(cfg: dict, args):
#     """Batch query: thin wrapper around retrieval experiment."""
#     from experiments.exp.retrieval_experiment import run_experiment
#     from src.data import load_pairs

#     pairs = load_pairs(cfg)
#     if args.max_queries:
#         pairs = pairs[: args.max_queries]
#     return run_experiment(pairs, cfg)


def run_full_pipeline(cfg: dict, args):
    """End-to-end: retrieval → LLM patching → evaluation."""
    from experiments.exp.retrieval_experiment import run_experiment as run_retrieval_exp
    from experiments.exp.prompt.patching_experiment import run_patching_experiment
    from src.io.read_write import make_run_dir
    from src.data import load_pairs

    run_id, run_dir = make_run_dir("full")
    print(f"\n{'━'*60}")
    print(f"  FULL PIPELINE — unified output: {run_dir}")
    print(f"{'━'*60}")

    # Step 1: Retrieval
    print(f"\n{'━'*60}")
    print(f"  STEP 1/3 — Retrieval (embed + FAISS top-k)")
    print(f"{'━'*60}")
    full_pairs = load_pairs(cfg)
    if args.max_queries:
        full_pairs = full_pairs[: args.max_queries]
    run_retrieval_exp(full_pairs, cfg, output_dir=run_dir)
    print(f"\n  ✓ Retrieval complete: {run_dir / 'retrieval_results.jsonl'}")

    # Step 2: LLM Patching
    print(f"\n{'━'*60}")
    print(f"  STEP 2/3 — LLM Patching (using retrieval results)")
    print(f"{'━'*60}")
    run_patching_experiment(
        cfg,
        retriever_mode="precomputed",
        model_name=args.model,
        query_run=str(run_dir),
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        output_dir=run_dir,
    )
    print(f"\n  ✓ Patching complete: {run_dir / 'results.jsonl'}")

    # Step 3: Evaluation + Dashboards
    print(f"\n{'━'*60}")
    print(f"  STEP 3/3 — Evaluation & Dashboards")
    print(f"{'━'*60}")
    from src.evaluate.__main__ import run_all

    results_jsonl = run_dir / "results.jsonl"
    run_all(
        results_path=results_jsonl,
        config_path=args.config,
        strip_comments=args.strip_comments,
    )

    print(f"\n{'━'*60}")
    print(f"  ALL DONE")
    print(f"{'━'*60}")
    print(f"  Output folder:  {run_dir}")
    print(f"  Patch analysis: {run_dir / 'patch_analysis.html'}")
    print(f"{'━'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["index", "query", "export", "experiment", "diagnostics", "batch", "full"],
        default="export",
    )
    parser.add_argument("--dataset", choices=["autopatch", "cvefixes"], default="autopatch")
    parser.add_argument("--cve")
    parser.add_argument(
        "--loo",
        action="store_true",
        help="run leave-one-out eval (slow, max 1000 samples)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="enable experiment split mode (overrides config)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="disable experiment split mode (overrides config)",
    )
    parser.add_argument(
        "--split-test-ratio", type=float, help="test ratio for split mode, e.g. 0.2"
    )
    parser.add_argument(
        "--aug-train-ratio",
        type=float,
        help="fraction of augmented train pairs to keep in index, e.g. 0.5",
    )
    # batch-mode arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="queries per batch flush (batch mode, default: 10)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="limit total queries for testing (batch mode)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Azure model/deployment name (batch mode, default: MODEL_NAME from .env)",
    )
    parser.add_argument(
        "--resume", default=None, help="path to run dir to resume (batch mode)"
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="use oracle retriever (perfect same-CVE lookup) instead of FAISS embedding retriever (batch mode)",
    )
    parser.add_argument(
        "--query-run",
        default=None,
        help="path to a --mode query run dir whose results.jsonl provides pre-computed retrieval (batch mode, replaces FAISS)",
    )
    parser.add_argument(
        "--strip-comments",
        action="store_true",
        default=True,
        help="remove C/C++ comments before patch comparison (full/batch mode)",
    )

    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    app_cfg = _instantiate_app_config(cfg)
    cfg.setdefault("paths", {})
    cfg["paths"].update(
        {
            "joern_bin_dir": str(app_cfg.paths.joern_bin_dir),
            "output_dir": str(app_cfg.paths.output_dir),
            "models_cache_dir": str(app_cfg.paths.models_cache_dir),
            "index_dir": str(app_cfg.paths.index_dir),
        }
    )

    _validate_cfg(cfg)

    # Apply split overrides for modes that use them
    if args.mode in ("query", "experiment", "batch", "full"):
        apply_split_overrides(cfg, args)

    if args.mode == "export":
        run_export(app_cfg, args.dataset)
    elif args.mode == "index":
        run_pipeline(cfg)
    elif args.mode == "query":
        if args.cve:
            run_query(cfg, args.cve)
        else:
            run_batch_query(cfg, args)
    elif args.mode == "experiment":
        from src.data import load_pairs
        from experiments.exp.retrieval_experiment import RetrievalGridExperiment

        all_pairs = load_pairs(cfg)
        exp = RetrievalGridExperiment(
            run_leave_one_out=args.loo,
            preloaded_pairs=all_pairs,
        )
        exp.run(cfg)
    elif args.mode == "diagnostics":
        from src.data import load_pairs
        from src.diagnostics import run_diagnostics

        run_diagnostics(load_pairs(cfg))

    elif args.mode == "batch":
        from experiments.exp.prompt.patching_experiment import run_patching_experiment

        run_patching_experiment(
            cfg,
            retriever_mode="oracle" if args.oracle else "precomputed",
            model_name=args.model,
            query_run=args.query_run,
            max_queries=args.max_queries,
            batch_size=args.batch_size,
            resume=args.resume,
        )

    elif args.mode == "full":
        run_full_pipeline(cfg, args)
