import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import yaml
from tqdm import tqdm

from src.data.autopatch import AutoPatchDataset
from src.data.base import ExportJob
from src.data.pipeline import run_joern_export, write_c_file
from src.embeddings import build_embedders
from src.rag.index import FAISSIndex

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

    success = run_joern_export(
        joern_bin_dir, c_file, str(out_dir), str(graph_folder)
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
            with tqdm(total=len(jobs), desc=ds_name, unit='job') as pbar:
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
        dim = cfg['embeddings']['dim'],
        index_path = rag_cfg['index_path'],
        metadata_path = rag_cfg['metadata_path'],
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
                index.add(pair, emb, variant) # index to RAG
                total += 1
                if total % 5 == 0:
                    print(f" indexed {total} pairs.. ")
            except Exception as e:
                print(f"   skip {pair.cve_id} / {pair.func_name}:  {e}")

    index.save()
    print(f"\nDone. \nTotal indexed: {total}")

def run_query(cfg: dict, cve_id: str):
    from src.rag.retriever import Retriever

    rag_cfg = cfg['rag']
    index = FAISSIndex(
        dim=cfg['embeddings']['dim'],
        index_path = rag_cfg['index_path'],
        metadata_path=rag_cfg['metadata_path'],
    )
    index.load()
    retriever = Retriever(index, top_k=rag_cfg['top_k'])
    for r in retriever.query_by_cve(cve_id):
        print(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", choices=["index", "query", "export", "experiment", "diagnostics"], default="export"
    )
    parser.add_argument("--dataset", choices=["autopatch"], default="autopatch")
    parser.add_argument("--cve")
    parser.add_argument('--loo', action='store_true',
                    help='run leave-one-out eval (slow, max 1000 samples)')
    parser.add_argument('--split', action='store_true',
                    help='enable experiment split mode (overrides config)')
    parser.add_argument('--no-split', action='store_true',
                    help='disable experiment split mode (overrides config)')
    parser.add_argument('--split-test-ratio', type=float,
                    help='test ratio for split mode, e.g. 0.2')
    parser.add_argument('--aug-train-ratio', type=float,
                    help='fraction of augmented train pairs to keep in index, e.g. 0.5')

    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    active = [n for n in DATASETS if cfg['data'].get(n)]
    if args.mode == "export":
        run_export(cfg, args.dataset)
    elif args.mode == "index":
        run_pipeline(cfg)
    elif args.mode == "query":
        # change it to query by code snippet.
        if not args.cve:
            raise ValueError("--cve required for query mode")
        run_query(cfg, args.cve)
    elif args.mode == 'experiment':
        from experiments.runner import run_experiment

        cfg.setdefault('experiment', {})
        cfg['experiment'].setdefault('split', {})
        split_cfg = cfg['experiment']['split']
        if args.split:
            split_cfg['enabled'] = True
        if args.no_split:
            split_cfg['enabled'] = False
        if args.split_test_ratio is not None:
            split_cfg['test_ratio'] = args.split_test_ratio
        if args.aug_train_ratio is not None:
            split_cfg['augmented_train_ratio'] = args.aug_train_ratio

        # load all pairs from all active datasets
        all_pairs = []
        for ds_name in active:
            ds_cfg  = cfg['data'][ds_name]
            dataset = DATASETS[ds_name](ds_cfg)
            print(f"Loading {dataset.name()}...")
            all_pairs.extend(dataset.load_all())

        run_experiment(
            pairs             = all_pairs,
            cfg               = cfg,
            run_leave_one_out = args.loo,
        )
    elif args.mode == 'diagnostics':
        from src.diagnostics import run_diagnostics

        # load all pairs from all active datasets
        all_pairs = []
        for ds_name in active:
            ds_cfg  = cfg['data'][ds_name]
            dataset = DATASETS[ds_name](ds_cfg)
            print(f"Loading {dataset.name()}...")
            all_pairs.extend(dataset.load_all())

        run_diagnostics(all_pairs)
