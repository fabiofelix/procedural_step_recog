#=========================================================================#
#Code from https://github.com/VIDA-NYU/ptg-server-ml/tree/main/ptgprocess #
#=========================================================================#

import os
import sys
import glob
import numpy as np
import torch
import math
from torch import nn

import gdown
import yaml
from collections import OrderedDict

##mod_path = os.path.join(os.path.dirname(__file__), 'procedural_step_recog')
mod_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0,  mod_path)


def custom_weights(layer):
  if isinstance(layer, nn.Linear):
    nn.init.xavier_normal_(layer.weight)  
    nn.init.zeros_(layer.bias)

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
          self.load_state_dict( self.update_version(torch.load( cfg.MODEL.CHECKPOINT_FILE_PATH )))
        else:
          self.apply(custom_weights)

    def forward(self, action, h=None, aud=None, objs=None, frame=None):
        x = action

        if self.use_action and (self.use_audio or self.use_objects):
            action = self.action_fc(action)
            if self.use_bn:
                action = self.action_bn(action.transpose(1, 2)).transpose(1, 2)
            action = self.relu(action)

        if self.use_audio:
            aud = self.audio_fc(aud)
            if self.use_bn:
                aud = self.aud_bn(aud.transpose(1, 2)).transpose(1, 2)
            aud = self.relu(aud)
            x = aud

            if self.use_action:
               x = torch.concat((action, aud), -1)

        if self.use_objects:
            obj_proj = self.relu(self.obj_proj(objs))
            frame_proj = self.relu(self.frame_proj(frame))

            values = torch.softmax(torch.sum(frame_proj * obj_proj, dim=-1, keepdims=True), dim=-2)
            obj_in = torch.sum(obj_proj * values, dim=-2)
            obj_in = self.relu(self.obj_fc(obj_in))
            if self.use_bn:
                obj_in = self.obj_bn(obj_in.transpose(1, 2)).transpose(1, 2)            
            x = obj_in
            
            if self.use_action:
              if self.use_audio:
                x = torch.concat((action, aud, obj_in), -1)
              else:    
                x = torch.concat((action, obj_in), -1)
            elif self.use_audio:
                x = torch.concat((aud, obj_in), -1)

        out, h = self.gru(x, h)
        out = self.relu(out)
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


