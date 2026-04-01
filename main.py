import yaml
import argparse
from data.autopatch import AutoPatchDataset
from src.embeddings import build_embedders

DATASETS = {"autopatch": AutoPatchDataset, "cvefixes": None}


def run_export(cfg: str, dataset_name: str):

def main(cfg):
    active_datasets = [name for name in DATASETS if cfg["data"].get(name)]
    rag_cfg = cfg['rag']
    variant = rag_cfg['embedding_variant']
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
