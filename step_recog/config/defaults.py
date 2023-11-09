#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

_C = CfgNode()

# -----------------------------------------------------------------------------
# TRAIN options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True
_C.TRAIN.NUM_GPUS = 1
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR = 0.001
_C.TRAIN.EPOCHS = 5

# -----------------------------------------------------------------------------
# EVAL options
# -----------------------------------------------------------------------------
_C.EVAL = CfgNode()
_C.EVAL.ENABLE = True

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.HIDDEN_SIZE = 1024
_C.MODEL.CONTEXT_LENGTH = 'all'
_C.MODEL.OUTPUT_DIM = 33
_C.MODEL.DRIVE_ID = ""
_C.MODEL.SKILLS = []
_C.MODEL.USE_AUDIO = False
_C.MODEL.USE_OBJECTS = False
_C.MODEL.USE_BN = False
_C.MODEL.CHECKPOINT_FILE_PATH = ''

# -----------------------------------------------------------------------------
# Dataset options
# -----------------------------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.NAME = ''
_C.DATASET.LOCATION = ''
_C.DATASET.AUDIO_LOCATION = ''
_C.DATASET.VIDEO_LAYER = ''
_C.DATASET.OBJECT_FRAME_LOCATION = ''
_C.DATASET.TR_ANNOTATIONS_FILE = ''
_C.DATASET.VL_ANNOTATIONS_FILE = ''
_C.DATASET.TS_ANNOTATIONS_FILE = ''
_C.DATASET.HOP_SIZE = 0.5
_C.DATASET.FPS = 30
_C.DATASET.WIN_LENGTH = 2
_C.DATASET.INCLUDE_IMAGE_AUGMENTATIONS = False
_C.DATASET.INCLUDE_TIME_AUGMENTATIONS = False

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
