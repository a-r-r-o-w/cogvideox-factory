import OmegaConf
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")
    parser.add_argument(
        "config_path",
        type=str,
        default=None,
        required=True
    )
    args = parser.parse_args()
    return OmegaConf.load(args.config_path)
