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
    """Batch query: embed each query pair, retrieve top-k from FAISS, write results.

    Follows the same patterns as batch_inference (BackgroundWriter, resumability,
    batch flushing) but performs retrieval only — no LLM calls.
    """
    import json as _json
    import time

    from experiments.common import build_split, load_pairs, make_run_dir
    from src.agents.utils import get_ground_truth_patch
    from src.evaluate.retrieval_eval import _build_index_and_retriever
    from src.io import BackgroundWriter, load_completed

    # ── apply split overrides ────────────────────────────────────────
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

    # ── load pairs and split ─────────────────────────────────────────
    pairs = load_pairs(cfg)
    index_pairs, query_pairs, split_info = build_split(pairs, cfg)

    if args.max_queries:
        query_pairs = query_pairs[: args.max_queries]

    # ── build embedder + reuse existing FAISS index ──────────────────
    rag_cfg = cfg["rag"]
    top_k = rag_cfg.get("top_k", 5)
    embedder, retriever = _build_index_and_retriever(index_pairs, cfg, top_k)

    # ── resolve run directory ────────────────────────────────────────
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            print(f"ERROR: resume directory does not exist: {run_dir}")
            sys.exit(1)
        run_id = run_dir.name
        print(f"Resuming run: {run_id}")
    else:
        run_id, run_dir = make_run_dir("batch_query")
        print(f"New run: {run_id}")

    jsonl_path = run_dir / "results.jsonl"
    meta_path = run_dir / "run_meta.json"

    # ── load completed queries ───────────────────────────────────────
    completed = load_completed(jsonl_path)
    if completed:
        print(f"Found {len(completed)} completed queries — will skip them")

    pending = [
        p for p in query_pairs if (p.cve_id, p.meta.get("variant", "")) not in completed
    ]
    total = len(query_pairs)
    print(f"Total: {total} | Done: {total - len(pending)} | Pending: {len(pending)}")

    if not pending:
        print("All queries already completed.")
        return run_dir

    # ── save run metadata ────────────────────────────────────────────
    meta = {
        "run_id": run_id,
        "mode": "batch_query",
        "total_queries": total,
        "top_k": top_k,
        "resumed": args.resume is not None,
        "split_info": split_info,
    }
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2, default=str)

    # ── process in batches ───────────────────────────────────────────
    batch_size = args.batch_size
    writer = BackgroundWriter(jsonl_path)
    n_done = len(completed)
    t_start = time.perf_counter()

    try:
        for batch_idx in range(0, len(pending), batch_size):
            batch = pending[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(pending) + batch_size - 1) // batch_size

            print(f"\n{'─'*60}")
            print(
                f"Batch {batch_num}/{total_batches}  "
                f"(queries {n_done+1}–{n_done+len(batch)} of {total})"
            )
            print(f"{'─'*60}")

            batch_results = []
            for i, qp in enumerate(batch):
                cve_id = qp.cve_id
                cwe_id = qp.cwe_id
                variant = qp.meta.get("variant", "")

                base = {
                    "query_cve": cve_id,
                    "query_cwe": cwe_id,
                    "query_variant": variant,
                    "query_dir": qp.meta.get("dir_name", ""),
                }

                try:
                    q_emb = embedder.embed_one(qp.G_vuln)
                    results = retriever.query(q_emb, top_k=top_k)
                except Exception as e:
                    batch_results.append(
                        {
                            **base,
                            "status": "error",
                            "error": str(e),
                        }
                    )
                    print(f"  [{n_done+i+1}/{total}] {cve_id}/{variant}  ERROR: {e}")
                    continue

                if not results:
                    batch_results.append({**base, "status": "no_results"})
                    print(f"  [{n_done+i+1}/{total}] {cve_id}/{variant}  no results")
                    continue

                top = results[0]
                top_cve = top.get("cve_id", "?")
                cve_match = top_cve == cve_id
                cwe_match = top.get("cwe_id") == cwe_id
                ground_truth = get_ground_truth_patch(qp)

                record = {
                    **base,
                    "example_cve": top_cve,
                    "example_cwe": top.get("cwe_id"),
                    "example_variant": top.get("variant", ""),
                    "example_dir": top.get("dir_name", ""),
                    "status": "retrieved",
                    "cve_match": cve_match,
                    "cwe_match": cwe_match,
                    "retrieval": {
                        "cve_match": cve_match,
                        "cwe_match": cwe_match,
                        "retrieved_variant": top.get("variant", ""),
                        "score": round(top.get("score", 0.0), 6),
                        "top_k": [
                            {
                                "rank": j + 1,
                                "cve_id": r.get("cve_id"),
                                "cwe_id": r.get("cwe_id"),
                                "variant": r.get("variant"),
                                "func_name": r.get("func_name"),
                                "score": round(r.get("score", 0), 6),
                            }
                            for j, r in enumerate(results)
                        ],
                    },
                    "ground_truth_patch": ground_truth[:500],
                }
                batch_results.append(record)

                print(
                    f"  [{n_done+i+1}/{total}] {cve_id}/{variant}  "
                    f"→ {top_cve}/{top.get('variant','?')}  "
                    f"score={top.get('score',0):.4f}  "
                    f"cve_match={cve_match}"
                )

            n_done += len(batch_results)
            writer.write(batch_results)
            elapsed = time.perf_counter() - t_start
            print(
                f"\n  Progress: {n_done}/{total} ({n_done/total:.0%})  |  {elapsed:.0f}s elapsed"
            )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted — flushing writes...")
        writer.flush()
        writer.close()
        print(f"Run directory: {run_dir}")
        print("Re-run with --resume to continue.")
        sys.exit(1)

    writer.flush()
    writer.close()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. {n_done} queries written in {elapsed:.0f}s.")
    print(f"Run directory: {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["index", "query", "export", "experiment", "diagnostics", "batch"],
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
            run_batch_query(cfg, args)
    elif args.mode == "experiment":
        from experiments.common import load_pairs
        from experiments.runner import run_experiment

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

        run_experiment(
            pairs=all_pairs,
            cfg=cfg,
            run_leave_one_out=args.loo,
        )
    elif args.mode == "diagnostics":
        from experiments.common import load_pairs
        from src.diagnostics import run_diagnostics

        all_pairs = load_pairs(cfg)

        run_diagnostics(all_pairs)

    elif args.mode == "batch":
        import os as _os

        from dotenv import load_dotenv

        from experiments.common import build_split
        from src.agents.batch_inference import run_batch_inference
        from src.rag.oracle import OracleRetriever
        from src.rag.precomputed import PrecomputedRetriever

        load_dotenv()

        if not _os.getenv("AZURE_API_KEY") or not _os.getenv("AZURE_API_BASEURL"):
            print("ERROR: Set AZURE_API_KEY and AZURE_API_BASEURL in .env")
            sys.exit(1)

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

        # load pairs WITHOUT CPGs (metadata only — fast)
        from experiments.common import load_pairs_lightweight

        pairs = load_pairs_lightweight(cfg)
        print(f"Loaded {len(pairs)} lightweight pairs (no CPGs)")
        index_pairs, query_pairs, split_info = build_split(pairs, cfg)

        if args.max_queries:
            query_pairs = query_pairs[: args.max_queries]

        # build retriever
        if args.oracle:
            retriever = OracleRetriever(index_pairs)
            print(f"Oracle retriever built from {len(index_pairs)} index pairs")
            retriever_mode = "oracle"
        else:
            # use pre-computed query results (from --mode query)
            if not args.query_run:
                print("ERROR: --query-run <run_dir> required for non-oracle batch mode")
                print(
                    "Run  python main.py --mode query  first, then pass its output dir."
                )
                sys.exit(1)

            query_results = Path(args.query_run) / "results.jsonl"
            if not query_results.exists():
                print(f"ERROR: {query_results} not found")
                sys.exit(1)

            retriever = PrecomputedRetriever(query_results)
            retriever_mode = "embedding"

        # preload db_entry.json for all CVEs, keyed by dir_name
        from src.io import load_db_cache

        cve_root = Path(cfg["data"]["autopatch"]["root"])
        db_cache = load_db_cache(cve_root)
        print(f"Cached {len(db_cache)} db_entries")

        run_batch_inference(
            query_pairs=query_pairs,
            retriever=retriever,
            db_cache=db_cache,
            model_name=args.model,
            batch_size=args.batch_size,
            run_tag=f"batch_{retriever_mode}",
            resume_dir=args.resume,
            meta_extra={"mode": retriever_mode, "split_info": split_info},
        )
