import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import yaml

from data.autopatch import AutoPatchDataset
from data.base import ExportJob
from data.pipeline import run_joern_export, write_c_file
from src.embeddings import build_embedders

DATASETS = {"autopatch": AutoPatchDataset, "cvefixes": None}


def _process_job(job: ExportJob, joern_bin_dir: str):
    out_dir = Path(job.out_dir)
    existing = list(out_dir.glob("**/export.xml"))
    if existing:
        print(f"skip {job.cve_id}/{job.variant}/{job.version} already exists")
        return True, f"skip {job.cve_id}/{job.variant}/{job.version} already exists"

    c_file = out_dir / f"{job.func_name or 'function'}.c"
    graph_folder = out_dir / "graph"

    try:
        write_c_file(job.source_code, c_file, supplementary_code=job.supplementary_code)
    except Exception as e:
        return False, f"write failed {job.cve_id}: {e}"

    success = run_joern_export(
        joern_bin_dir, str(c_file), str(out_dir), str(graph_folder)
    )
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
            for success, msg in pool.imap_unordered(worker_fn, jobs, chunksize=4):
                if "skip" in msg:
                    skipped += 1
                elif "fail" in msg.lower():
                    fail += 1
                    print(f" {msg} \n")
                else:
                    ok += 1
        print(f"Done \n    ok: {ok}  -  skipped: {skipped}  -  fail: {fail}")


def main(cfg):
    active_datasets = [name for name in DATASETS if cfg["data"].get(name)]
    rag_cfg = cfg["rag"]
    variant = rag_cfg["embedding_variant"]
    embedders = build_embedders(cfg)
    total = 0
    indexer = next(e for e in embedders if e.name == variant)

    for ds_name in active_datasets:
        ds_cfg = cfg["data"][ds_name]
        # instantiate the dataset class with the config params
        dataset = DATASETS[ds_name](ds_cfg)
        print(f"-----------{dataset.name()}-----------")

        for pair in dataset.stream():
            try:
                emb = indexer.embed_one(pair.G_vuln)
                # index.add(pair, emb, variant) index to RAG
                total += 1
                if total % 5 == 0:
                    print(f" indexed {total} pairs.. ")
            except Exception as e:
                print(f"   skip {pair.cve_id} / {pair.func_name}:  {e}")

    print(f"\nDone. \nTotal indexed: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", choices=["index", "query", "export"], default="export"
    )
    parser.add_argument("--dataset", choices=["autopatch"], default="autopatch")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "export":
        run_export(cfg, args.dataset)
    elif args.mode == "index":
        main(cfg)
    elif args.mode == "query":
        # to be verified
        # if not args.cve:
        #     raise ValueError("--cve required for query mode")
        raise NotImplementedError()
