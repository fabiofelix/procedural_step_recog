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
