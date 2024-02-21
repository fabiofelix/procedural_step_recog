#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import re
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import cv2
import numpy as np
from torch.nn.init import constant_
from torch.nn.init import normal_
from torch.utils import model_zoo
from copy import deepcopy
import pdb

from .build import MODEL_REGISTRY
from act_recog.datasets.transform import uniform_crop

@MODEL_REGISTRY.register()
class Omnivore(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # model
        self.cfg = cfg
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
    
    def prepare_image(self, im):
        # 1,C,H,W
        im = prepare_image(im, self.cfg.MODEL.MEAN, self.cfg.MODEL.STD, self.cfg.MODEL.IN_SIZE)
        return im    

##Similar to act_recog.datasets.milly.py:__getitem__
def prepare_image(im, mean, std, expected_size=224):
    '''[H, W, 3] => [3, 224, 224]'''
    im = cv2.resize(im, (456, 256)) #TODO: review the code act_recog.datasets.milly.py: retry_load_images and __getitem__. There are two resizes
    scale = max(expected_size/im.shape[0], expected_size/im.shape[1])
    im = cv2.resize(im, (0,0), fx=scale, fy=scale)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255
    im = (im - np.asarray(mean)) / np.asarray(std)
    im = im.transpose(2, 0, 1)    
    im, _ = uniform_crop(im, expected_size, 1)
    return torch.Tensor(im)
