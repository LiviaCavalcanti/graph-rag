import yaml
import argparse

def main():
    print("Hello from graph-rag!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main()
