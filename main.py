import yaml
import argparse

DATASETS = ["autopatch", 'cvefixes']

def main(cfg):
    active_datasets = [name for name in DATASETS if cfg['data'].get(name)]
    for ds_name in active_datasets:
        ds_cfg = cfg['data'][ds_name]
        # instantiate the dataset class with the config params
        dataset = DATASETS[ds_name](ds_cfg)
    print("Hello from graph-rag!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--mode', choices=['index', 'query'], default='index')

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
