from deployment.config import Config
from deployment.controller import BaseController
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Robot Control')
    parser.add_argument('--is_real', action='store_true', help='Real Robot or Not (default: False)')
    return parser.parse_args()


def update_config_from_args(config, args):
    config.is_real = args.is_real
    return config


if __name__ == '__main__':
    cfg = Config()
    args = parse_args()

    config = update_config_from_args(cfg, args)
    controller = BaseController(cfg=cfg)
    controller.run()
