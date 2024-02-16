#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import re
import torch
import torch.nn as nn
import torchvision
import numpy as np
# import pandas as pd
from torch.nn.init import constant_
from torch.nn.init import normal_
from torch.utils import model_zoo
from copy import deepcopy
import cv2

from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class Omnivore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.SIZE = cfg.MODEL.IN_SIZE 
        self.MEAN = cfg.MODEL.MEAN
        self.STD = cfg.MODEL.STD

        # model
        self._device = nn.Parameter(torch.empty(0))
        self.model = torch.hub.load("facebookresearch/omnivore:main", model="omnivore_swinB_epic")
        self.model.eval()

        self.heads = self.model.heads
        self.model.heads = nn.Identity()

    def forward(self, x, return_embedding=False):  # C T H W
        shoulder = self.model(x, input_type="video")
        y = self.heads(shoulder)
        if return_embedding:
            return y, shoulder
        return y

    def prepare_frame(self, frame):
        scale = self.SIZE / frame.shape[0]
        frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = crop_center(frame, self.SIZE, self.SIZE)
        frame = (frame.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        frame = frame.transpose(2, 0, 1)  # C H W
        frame = torch.from_numpy(np.ascontiguousarray(frame)).to(self._device.device)
        return frame.float()


def crop_center(img,cropx,cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]
