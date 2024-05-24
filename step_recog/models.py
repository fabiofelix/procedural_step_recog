#=========================================================================#
#Code from https://github.com/VIDA-NYU/ptg-server-ml/tree/main/ptgprocess #
#=========================================================================#

import os
import sys
import glob
import numpy as np
import torch
import math
import pdb
from torch import nn

import gdown
import yaml
from collections import OrderedDict

##mod_path = os.path.join(os.path.dirname(__file__), 'procedural_step_recog')
mod_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0,  mod_path)

from step_recog.config.defaults import get_cfg
#from step_recog.resnet import ResNet
from step_recog.full.download import cached_download_file
from step_recog.full.clip_patches import ClipPatches 

from act_recog.models import Omnivore
from act_recog.config import load_config as act_load_config

from ultralytics import YOLO

#from slowfast.utils.parser import load_config as slowfast_load_config
#from slowfast.models.audio_model_builder import SlowFast
#from slowfast.utils import checkpoint

MAX_OBJECTS = 25
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def custom_weights(layer):
  if isinstance(layer, nn.Linear):
    nn.init.xavier_normal_(layer.weight)  
    nn.init.zeros_(layer.bias)

def args_hook(cfg_file):
  args = lambda: None
  args.cfg_file = cfg_file
  args.opts = None   
  return args

class OmniGRU(nn.Module):
    def __init__(self, cfg, load = False):
        super().__init__()
        n_layers = 2
        drop_prob = 0.2
        action_size = 1024
        audio_size = 2304
        input_dim = action_size
        hidden_dim = cfg.MODEL.HIDDEN_SIZE
        output_dim = cfg.MODEL.OUTPUT_DIM
        self.use_action = cfg.MODEL.USE_ACTION
        self.use_objects = cfg.MODEL.USE_OBJECTS
        self.use_audio = cfg.MODEL.USE_AUDIO
        self.use_bn = cfg.MODEL.USE_BN
        self.skills = cfg.MODEL.SKILLS

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        gru_input_dim = 0

        if self.use_action:
          gru_input_dim += int(input_dim/2) if self.use_audio or self.use_objects else action_size
          self.action_fc = nn.Linear(action_size, int(input_dim/2))
          if self.use_bn: 
            self.action_bn = nn.BatchNorm1d(int(input_dim/2))

        if self.use_audio:
          gru_input_dim += int(input_dim/2)
          self.audio_fc = nn.Linear(audio_size, int(input_dim/2))
          if self.use_bn: 
              self.aud_bn = nn.BatchNorm1d(int(input_dim/2))

        if self.use_objects:
          gru_input_dim += int(input_dim/2)
          self.obj_fc = nn.Linear(512, int(input_dim/2))
          self.obj_proj = nn.Linear(517, int(input_dim/2))    ## clip space (512) + bouding box (4) + prediction (1)
          self.frame_proj = nn.Linear(517, int(input_dim/2))  ## clip space (512) + bouding box (4) + prediction (1)
          if self.use_bn: 
            self.obj_bn = nn.BatchNorm1d(int(input_dim/2))            

        if gru_input_dim == 0:
           raise Exception("GRU has to use at least one input (action, object/frame, or audio)")             

        self.gru = nn.GRU(gru_input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim + cfg.MODEL.APPEND_OUT_POSITIONS)  ## adding no step, begin, and end positions to the output
        self.relu = nn.ReLU()

        if load:
          f = cfg.MODEL.OMNIGRU_CHECKPOINT_URL or cached_download_file(cfg.MODEL.PRETRAINED_CHECKPOINT_URL)
          self.load_state_dict(self.update_version(torch.load(f)))
        else:
          self.apply(custom_weights)

    def forward(self, action, h=None, aud=None, objs=None, frame=None, return_last_step=True):
        x = []

        if self.use_action and (self.use_audio or self.use_objects):
            action = self.action_fc(action)
            if self.use_bn:
                action = self.action_bn(action.transpose(1, 2)).transpose(1, 2)
            action = self.relu(action)
            x.append(action)

        if self.use_audio:
            aud = self.audio_fc(aud)
            if self.use_bn:
                aud = self.aud_bn(aud.transpose(1, 2)).transpose(1, 2)
            aud = self.relu(aud)
            x.append(aud)

        if self.use_objects:
            obj_proj = self.relu(self.obj_proj(objs))
            frame_proj = self.relu(self.frame_proj(frame))

            #=============================== Original tests ===============================#
#            values = torch.softmax(torch.sum(frame_proj * obj_proj, dim=-1, keepdims=True), dim=-2)
#            obj_in = torch.sum(obj_proj * values, dim=-2)

            #=============================== Conceptual Fusion ===============================#
            theta = nn.functional.cosine_similarity(frame_proj, obj_proj, dim = -1)

            phi   = nn.functional.cosine_similarity(obj_proj[:, :, None, :, :], obj_proj[:, :, :, None, :], dim = -1)
            phi   = torch.mean(phi, dim = -1)

            w     = torch.softmax(theta + phi, dim = -1)
            w     = w[:, :, :, None]

            obj_in = w * frame_proj + (1 - w) * obj_proj
            obj_in = torch.mean(obj_in, dim = -2)
            #=================================================================================#
            obj_in = self.obj_fc(obj_in)
            if self.use_bn:
              obj_in = self.obj_bn(obj_in.transpose(1, 2)).transpose(1, 2)
            obj_in = self.relu(obj_in)            
            x.append(obj_in)

        x = torch.concat(x, -1) if len(x) > 1 else x[0]            
        out, h = self.gru(x, h)
        out = self.relu(out[:, -1]) if return_last_step else self.relu(out)
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

    def update_version(self, state_dict):
      new_dict = OrderedDict()

      for key, value in state_dict.items():
        if "rgb" in key:
          key = key.replace("rgb", "action")  

        new_dict[key] = value
          
      return new_dict    


