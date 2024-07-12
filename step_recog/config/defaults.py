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
_C.TRAIN.EPOCHS = 5
_C.TRAIN.OPT = "adam"
_C.TRAIN.LR = 0.001
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.SCHEDULER = None
_C.TRAIN.RETURN_METRICS = False
_C.TRAIN.USE_CROSS_VALIDATION = True
_C.TRAIN.CV_TEST_TYPE = None
_C.TRAIN.USE_CLASS_WEIGHT = True

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
_C.MODEL.PROJECTION_SIZE = 512
##_C.MODEL.CONTEXT_LENGTH = 'all'
_C.MODEL.OUTPUT_DIM = 33
##_C.MODEL.DRIVE_ID = ""
_C.MODEL.SKILLS = []
_C.MODEL.USE_ACTION = True
_C.MODEL.USE_OBJECTS = True
_C.MODEL.USE_AUDIO = False
_C.MODEL.USE_BN = False
##_C.MODEL.CHECKPOINT_FILE_PATH = ''
_C.MODEL.DROP_OUT = 0.2

_C.MODEL.YOLO_CHECKPOINT_URL = ''
_C.MODEL.OMNIGRU_CHECKPOINT_URL = ''
_C.MODEL.OMNIVORE_CONFIG = 'OMNIVORE'
_C.MODEL.SLOWFAST_CONFIG = 'SLOWFAST'

# -----------------------------------------------------------------------------
# Dataset options
# -----------------------------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.CLASS = 'Milly_multifeature_v4'
##_C.DATASET.NAME = ''
_C.DATASET.LOCATION = ''
_C.DATASET.AUDIO_LOCATION = ''
##_C.DATASET.VIDEO_LAYER = ''
##_C.DATASET.OBJECT_FRAME_LOCATION = ''
_C.DATASET.TR_ANNOTATIONS_FILE = ''
_C.DATASET.VL_ANNOTATIONS_FILE = ''
_C.DATASET.TS_ANNOTATIONS_FILE = ''
##_C.DATASET.HOP_SIZE = 0.5
##_C.DATASET.FPS = 30
##_C.DATASET.WIN_LENGTH = 2
_C.DATASET.INCLUDE_IMAGE_AUGMENTATIONS = False
_C.DATASET.INCLUDE_TIME_AUGMENTATIONS = False
_C.DATASET.IMAGE_AUGMENTATION_PERCENTAGE = 0.5 #probability of applying image augmentation

_C.SKILLS = []

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