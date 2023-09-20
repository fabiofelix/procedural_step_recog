#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from .build import build_dataset


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            config/defaults.py
    """
    dataset_name = cfg.DATASET.NAME
    batch_size = int(cfg.BATCH_SIZE / cfg.NUM_GPUS)
    shuffle = False
    drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn= None,
    )
    return loader
