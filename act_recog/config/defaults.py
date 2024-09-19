#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
import os
import pathlib
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
_C.MODEL.WIN_LENGTH = 2
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
    return _C.clone()

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
    if isinstance(args, (str, pathlib.Path)):
        args = args_hook(args)
    if args.cfg_file is not None:
        cfg.merge_from_file(find_config_file(args.cfg_file))
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    return cfg

# get built-in configs from the step_recog/config directory
CONFIG_DIR = pathlib.Path(__file__).parent.parent.parent / 'config'

def find_config_file(cfg_file):
    cfg_files = [
        cfg_file,  # you passed a valid config file path
        CONFIG_DIR / cfg_file,  # a path relative to the config directory
        CONFIG_DIR / f'{cfg_file}.yaml',  # the name without the extension
        CONFIG_DIR / f'{cfg_file}.yml',
    ]
    for f in cfg_files:
        if os.path.isfile(f):
            return f
    raise FileNotFoundError(cfg_file)


def args_hook(cfg_file):
  args = lambda: None
  args.cfg_file = cfg_file
  args.opts = None
  return args
