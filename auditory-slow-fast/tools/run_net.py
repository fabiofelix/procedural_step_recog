#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test an audio classification model."""

import os, sys

model_dir = os.path.dirname(__file__) + '/../'
sys.path.insert(0, model_dir)


from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from extract_net import extract
from test_net import test
from train_net import train


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    if cfg.EXTRACT.ENABLE:
#        extract(cfg)
        launch_job(cfg=cfg, init_method=args.init_method, func=extract)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)



if __name__ == "__main__":
    main()
