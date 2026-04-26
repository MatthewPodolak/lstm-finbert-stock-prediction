import argparse

from configs import config_5m, config_15m
from utils import line_gen
from training.trainer import train_model
from inference.inference import inference

TIMEFRAME_CONFIGS = [config_5m, config_15m]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_inf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.only_inf:
        line_gen("TRAINING PIPELINE")
        for cfg in TIMEFRAME_CONFIGS:
            train_model(cfg)
        line_gen("TRAINING COMPLETED!")

    line_gen("STARTING INFERENCE")
    inference()


if __name__ == "__main__":
    main()
