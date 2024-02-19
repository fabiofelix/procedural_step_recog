#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

_C = CfgNode()
_C.NUM_GPUS = 1
_C.BATCH_SIZE = 32

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.ARCH = "slowfast"
_C.MODEL.MODEL_NAME = "SlowFast"
_C.MODEL.WIN_LENGTH = 1
_C.MODEL.HOP_SIZE = 0.5
_C.MODEL.NFRAMES = 32
_C.MODEL.IN_SIZE = 224
_C.MODEL.MEAN = []
_C.MODEL.STD = []

# -----------------------------------------------------------------------------
# Dataset options
# -----------------------------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.NAME = ''
_C.DATASET.LOCATION = ''
_C.DATASET.FPS = 30

# -----------------------------------------------------------------------------
# Dataloader options
# -----------------------------------------------------------------------------
_C.DATALOADER = CfgNode()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.PIN_MEMORY = True

# -----------------------------------------------------------------------------
# output options
# -----------------------------------------------------------------------------
_C.OUTPUT = CfgNode()
_C.OUTPUT.LOCATION = ''

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C

def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    return cfg
