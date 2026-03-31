import yaml
import argparse
from data.autopatch import AutoPatchDataset

DATASETS = {"autopatch": AutoPatchDataset, "cvefixes": None}


def main(cfg):
    active_datasets = [name for name in DATASETS if cfg["data"].get(name)]
    total = 0
    for ds_name in active_datasets:
        ds_cfg = cfg["data"][ds_name]
        # instantiate the dataset class with the config params
        dataset = DATASETS[ds_name](ds_cfg)
        print(f"-----------{dataset.name()}-----------")

        for pair in dataset.stream():
            print(pair)
            total += 1

    print(f"\nDone. \nTotal indexed: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["index", "query"], default="index")
    parser.add_argument("--dataset", choices=["autopatch"], default="autopatch")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
