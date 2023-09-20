#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import re
import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torch.nn.init import constant_
from torch.nn.init import normal_
from torch.utils import model_zoo
from copy import deepcopy

from .build import MODEL_REGISTRY

has_gpu = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@MODEL_REGISTRY.register()
class Omnivore(nn.Module):
    def __init__(self, cfg, device=device):
        super().__init__()

        # model
        self.device = device = torch.device(device)
        self.model = model = torch.hub.load("facebookresearch/omnivore:main", model="omnivore_swinB_epic")#.to(device)
        model.eval()

        self.heads = self.model.heads
        self.model.heads = nn.Identity()

    def forward(self, x, return_embedding=False):
        shoulder = self.model(x, input_type="video")
        y = self.heads(shoulder)
        if return_embedding:
            return y, shoulder
        return y
