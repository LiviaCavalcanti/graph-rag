import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import yaml

from data.autopatch import AutoPatchDataset
from data.base import ExportJob
from src.embeddings import build_embedders

DATASETS = {"autopatch": AutoPatchDataset, "cvefixes": None}


def _process_job(job: ExportJob, joern_bin_dir: str): ...
def run_export(cfg: str, dataset_name: str | None = None):
    joern_bin_dir = cfg["joern"]["bin_dir"]
    workers = cfg["joern"].get("workers", max(1, cpu_count() - 1))
    active = [dataset_name] if dataset_name else list(DATASETS.keys())
    active = [n for n in active if cfg["data"].get(n)]

    for ds_name in active:
        ds_cfg = cfg["data"][ds_name]
        dataset = DATASETS[ds_name](ds_name)
        graphml_root = ds_cfg["graphml_root"]

        print(f"\n -------------- exporting {dataset.name()}--------------")
        jobs = list(dataset.export_jobs(graphml_root))
        print(f" {len(jobs)} jobs & {workers} workers")

        worker_fn = partial(_process_job, joern_bin_dir=joern_bin_dir)


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
            print(pair)
            try:
                emb = indexer.embed_one(pair.G_vuln)
                print("Returned the embedding: ", emb)
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
    parser.add_argument("--mode", choices=["index", "query", "export"], default="index")
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
