import argparse
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import yaml
from tqdm import tqdm

from src.data.autopatch import AutoPatchDataset
from src.data.base import ExportJob
from src.data.pipeline import run_joern_export, write_c_file
from src.embeddings import build_embedders
from src.rag.faiss_index import FAISSIndex

DATASETS = {"autopatch": AutoPatchDataset, "cvefixes": None}


def _process_job(job: ExportJob, joern_bin_dir: str):
    out_dir = Path(job.out_dir)
    existing = list(out_dir.glob("**/export.xml"))
    if existing:
        print(f"skip {job.cve_id}/{job.variant}/{job.version} already exists")
        return True, f"skip {job.cve_id}/{job.variant}/{job.version} already exists"

    c_file = out_dir / f"{job.func_name or 'function'}.cpp"
    graph_folder = out_dir / "graph"

    try:
        write_c_file(job.source_code, c_file, supplementary_code=job.supplementary_code)
    except Exception as e:
        return False, f"write failed {job.cve_id}: {e}"

    success = run_joern_export(joern_bin_dir, c_file, str(out_dir), str(graph_folder))
    label = f"{job.cve_id}/{job.variant}/{job.version}"

    return success, f"{'ok' if success else 'FAIL'} {label}"


def run_export(cfg: str, dataset_name: str | None = None):
    joern_bin_dir = cfg["joern"]["bin_dir"]
    workers = cfg["joern"].get("workers", max(1, cpu_count() - 1))
    active = [dataset_name] if dataset_name else list(DATASETS.keys())
    active = [n for n in active if cfg["data"].get(n)]

    for ds_name in active:
        ds_cfg = cfg["data"][ds_name]
        dataset = DATASETS[ds_name](ds_cfg)
        graphml_root = ds_cfg["graphml_root"]

        print(f"\n -------------- exporting {dataset.name()}--------------")
        jobs = list(dataset.export_jobs(graphml_root))
        print(f" {len(jobs)} jobs & {workers} workers")

        worker_fn = partial(_process_job, joern_bin_dir=joern_bin_dir)

        ok = fail = skipped = 0
        with Pool(processes=workers) as pool:
            with tqdm(total=len(jobs), desc=ds_name, unit="job") as pbar:
                for success, msg in pool.imap_unordered(worker_fn, jobs, chunksize=4):
                    if "skip" in msg:
                        skipped += 1
                    elif "fail" in msg.lower():
                        fail += 1
                        tqdm.write(f" {msg}")
                    else:
                        ok += 1
                    pbar.set_postfix(ok=ok, skip=skipped, fail=fail)
                    pbar.update(1)
        print(f"Done \n    ok: {ok}  -  skipped: {skipped}  -  fail: {fail}")


def run_pipeline(cfg):
    active_datasets = [name for name in DATASETS if cfg["data"].get(name)]
    rag_cfg = cfg["rag"]
    variant = rag_cfg["embedding_variant"]
    embedders = build_embedders(cfg)
    indexer = next(e for e in embedders if e.name == variant)
    index = FAISSIndex(
        dim=cfg["embeddings"]["dim"],
        index_path=rag_cfg["index_path"],
        metadata_path=rag_cfg["metadata_path"],
    )

    total = 0

    for ds_name in active_datasets:
        ds_cfg = cfg["data"][ds_name]
        # instantiate the dataset class with the config params
        dataset = DATASETS[ds_name](ds_cfg)
        print(f"-----------{dataset.name()}-----------")

        for pair in dataset.stream():
            try:
                emb = indexer.embed_one(pair.G_vuln)
                index.add(pair, emb, variant)  # index to RAG
                total += 1
                if total % 5 == 0:
                    print(f" indexed {total} pairs.. ")
            except Exception as e:
                print(f"   skip {pair.cve_id} / {pair.func_name}:  {e}")

    index.save()
    print(f"\nDone. \nTotal indexed: {total}")


def run_query(cfg: dict, cve_id: str):
    from src.rag.retriever import Retriever

    rag_cfg = cfg["rag"]
    index = FAISSIndex(
        dim=cfg["embeddings"]["dim"],
        index_path=rag_cfg["index_path"],
        metadata_path=rag_cfg["metadata_path"],
    )
    index.load()
    retriever = Retriever(index, top_k=rag_cfg["top_k"])
    for r in retriever.query_by_cve(cve_id):
        print(r)


def run_batch_query(cfg: dict, args):
    """Batch query: thin wrapper around retrieval experiment."""
    from experiments.exp.retrieval_experiment import run_experiment
    from src.data.autopatch import load_pairs

    pairs = load_pairs(cfg)
    if args.max_queries:
        pairs = pairs[: args.max_queries]
    return run_experiment(pairs, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["index", "query", "export", "experiment", "diagnostics", "batch", "full"],
        default="export",
    )
    parser.add_argument("--dataset", choices=["autopatch"], default="autopatch")
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

    if args.mode == "export":
        run_export(cfg, args.dataset)
    elif args.mode == "index":
        run_pipeline(cfg)
    elif args.mode == "query":
        if args.cve:
            run_query(cfg, args.cve)
        else:
            # apply split overrides before delegating to agent_experiment
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
            run_batch_query(cfg, args)
    elif args.mode == "experiment":
        from src.data.autopatch import load_pairs
        from experiments.exp.retrieval_experiment import RetrievalGridExperiment

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

        all_pairs = load_pairs(cfg)

        exp = RetrievalGridExperiment(
            run_leave_one_out=args.loo,
            preloaded_pairs=all_pairs,
        )
        exp.run(cfg)
    elif args.mode == "diagnostics":
        from src.data.autopatch import load_pairs
        from src.diagnostics import run_diagnostics

        all_pairs = load_pairs(cfg)

        run_diagnostics(all_pairs)

    elif args.mode == "batch":
        from experiments.exp.prompt.patching_experiment import run_patching_experiment

        # apply split overrides (same as experiment mode)
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

        retriever_mode = "oracle" if args.oracle else "precomputed"

        run_patching_experiment(
            cfg,
            retriever_mode=retriever_mode,
            model_name=args.model,
            query_run=args.query_run,
            max_queries=args.max_queries,
            batch_size=args.batch_size,
            resume=args.resume,
        )

    elif args.mode == "full":
        from experiments.exp.retrieval_experiment import run_experiment as run_retrieval_exp
        from experiments.exp.prompt.patching_experiment import run_patching_experiment
        from src.io.read_write import make_run_dir
        from src.data.autopatch import load_pairs as load_pairs_full

        # apply split overrides
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

        # ── Create unified output directory ──────────────────────
        run_id, run_dir = make_run_dir("full")
        print(f"\n{'━'*60}")
        print(f"  FULL PIPELINE — unified output: {run_dir}")
        print(f"{'━'*60}")

        # ── Step 1: Retrieval ────────────────────────────────────
        print(f"\n{'━'*60}")
        print(f"  STEP 1/3 — Retrieval (embed + FAISS top-k)")
        print(f"{'━'*60}")
        full_pairs = load_pairs_full(cfg)
        if args.max_queries:
            full_pairs = full_pairs[: args.max_queries]
        run_retrieval_exp(
            full_pairs,
            cfg,
            output_dir=run_dir,
        )
        print(f"\n  ✓ Retrieval complete: {run_dir / 'retrieval_results.jsonl'}")

        # ── Step 2: LLM Patching ─────────────────────────────────
        print(f"\n{'━'*60}")
        print(f"  STEP 2/3 — LLM Patching (using retrieval results)")
        print(f"{'━'*60}")
        retrieval_jsonl = run_dir / "retrieval_results.jsonl"
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

        # ── Step 3: Evaluation + Dashboards ──────────────────────
        print(f"\n{'━'*60}")
        print(f"  STEP 3/3 — Evaluation & Dashboards")
        print(f"{'━'*60}")
        from src.evaluate.__main__ import run_all

        # 3a. Patching evaluation + patch analysis dashboard
        results_jsonl = run_dir / "results.jsonl"
        dashboard_path = run_all(
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
