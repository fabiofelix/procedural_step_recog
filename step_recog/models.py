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

from step_recog.config.defaults import get_cfg

# from .omnivore import Omnivore
# from .audio_slowfast import AudioSlowFast

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_DIR = os.getenv('MODEL_DIR') or 'models'
# DEFAULT_CONFIG = os.path.join(mod_path, 'config/STEPGRU.yaml')
# DEFAULT_CHECKPOINT = os.path.join(MODEL_DIR, 'model_best.pt')
# DEFAULT_CHECKPOINT2 = os.path.join(MODEL_DIR, 'model_best_multimodal.pt')

# if not os.path.isfile(DEFAULT_CHECKPOINT2):
#     gdown.download(id="1ArZFX4LuuB4SbmmWSDit4S8gPhkc2DRq", output=DEFAULT_CHECKPOINT2)

# MULTI_ID = "1ArZFX4LuuB4SbmmWSDit4S8gPhkc2DRq"
# MULTI_CONFIG = os.path.join(mod_path, 'config/STEPGRU.yaml')
CHECKPOINTS = {

}

CFG_FILES = glob.glob(os.path.join(mod_path, 'config/*.yaml'))
for f in CFG_FILES:
    cfg = yaml.safe_load(open(f))
    if 'SKILLS' in cfg['MODEL']:
      for skill in cfg['MODEL']['SKILLS']:
          CHECKPOINTS[skill] = (cfg['MODEL']['DRIVE_ID'], f)
print(CHECKPOINTS)

# class Omnimix(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.action = Omnivore(return_embedding=True)
#         self.aud = AudioSlowFast(return_embedding=True)
#         self.mix = OmniGRU()

#     def forward(self, im, aud):
#         action_rgb, emb_rgb = self.action(im)
#         verb_rgb, noun_rgb = self.action.project_verb_noun(action_rgb)

#         (verb_aud, noun_aud), emb_aud = self.aud(aud)
#         step = self.mix(emb_rgb, emb_aud)
#         return step, action_rgb, (verb_rgb, noun_rgb), (verb_aud, noun_aud)

# class GRUNet3(nn.Module):
#     def __init__(self, action_size, audio_size, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
#         super(GRUNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers

#         self.action_fc = nn.Linear(action_size, int(input_dim/2))
#         #self.obj_fc = nn.Linear(512, int(input_dim/2))
#         #self.obj_proj = nn.Linear(517, int(input_dim/2))
#         #self.frame_proj = nn.Linear(517, int(input_dim/2))
#         self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
#         self.fc = nn.Linear(hidden_dim, output_dim + 2)
#         self.relu = nn.ReLU()

#     def forward(self, x, h):

#         #omni,objs,frame = x
#         omni = x
#         omni_in = self.relu(self.action_fc(x[0]))

#         #obj_proj = self.relu(self.obj_proj(x[1]))
#         #frame_proj = self.relu(self.frame_proj(x[2]))
#         #values = torch.softmax(torch.sum(frame_proj*obj_proj,dim=-1,keepdims=True),dim=-2)
#         #obj_in = torch.sum(obj_proj*values,dim=-2)
#         #obj_in = self.relu(self.obj_fc(obj_in))

#         #omni_in = torch.zeros_like(omni_in)
#         #x = torch.concat((omni_in,obj_in),-1)
#         #x = torch.concat((omni_in),-1)
#         out, h = self.gru(x, h)
#         out = self.fc(self.relu(out))
#         return out, h

#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
#         return hidden


class OmniGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.cfg = cfg = get_cfg()
        
        drive_id, cfg_file =  CHECKPOINTS[cfg.MODEL.SKILLS[0]]
        checkpoint = os.path.basename(cfg_file).split('.')[0]
        checkpoint = os.path.join(MODEL_DIR, f'{checkpoint}.pt')
        if not os.path.isfile(checkpoint):
            gdown.download(id=drive_id, output=checkpoint)

        # cfg.merge_from_file(cfg_file)
        n_layers = 2
        drop_prob = 0.2
        action_size = 1024
        audio_size = 2304
        input_dim = action_size
        hidden_dim = cfg.MODEL.HIDDEN_SIZE
        output_dim = cfg.MODEL.OUTPUT_DIM
        self.use_audio = cfg.MODEL.USE_AUDIO
        self.use_objects = cfg.MODEL.USE_OBJECTS
        self.use_bn = cfg.MODEL.USE_BN
        self.skills = cfg.MODEL.SKILLS

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.action_fc = nn.Linear(action_size, int(input_dim/2))
        if self.use_bn: 
            self.action_bn = nn.BatchNorm1d(int(input_dim/2))

        if self.use_audio:
            self.audio_fc = nn.Linear(audio_size, int(input_dim/2))
            if self.use_bn: 
                self.aud_bn = nn.BatchNorm1d(int(input_dim/2))

        if self.use_objects:
            self.obj_fc = nn.Linear(512, int(input_dim/2))
            self.obj_proj = nn.Linear(517, int(input_dim/2))    ## clip space (512) + bouding box (4) + prediction (1)
            self.frame_proj = nn.Linear(517, int(input_dim/2))  ## clip space (512) + bouding box (4) + prediction (1)

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim + 2)  ## adding begin and end positions to the output
        self.relu = nn.ReLU()

        #self.load_state_dict(torch.load(checkpoint))

    def forward(self, action, h=None, aud=None, objs=None, frame=None):
        x = action

        if self.use_audio or self.use_objects:
            action = self.action_fc(action)
            if self.use_bn:
                action = self.action_bn(action.transpose(1, 2)).transpose(1, 2)
            action = self.relu(action)

        if self.use_audio:
            aud = self.audio_fc(aud)
            if self.use_bn:
                aud = self.aud_bn(aud.transpose(1, 2)).transpose(1, 2)
            aud = self.relu(aud)

            x = torch.concat((action, aud), -1)

        if self.use_objects:
            obj_proj = self.relu(self.obj_proj(objs))
            frame_proj = self.relu(self.frame_proj(frame))

            values = torch.softmax(torch.sum(frame_proj * obj_proj, dim=-1, keepdims=True), dim=-2)
            obj_in = torch.sum(obj_proj * values, dim=-2)
            obj_in = self.relu(self.obj_fc(obj_in))
            
            if self.use_audio:
              x = torch.concat((action, aud, obj_in), -1)
            else:    
              x = torch.concat((action, obj_in), -1)

        out, h = self.gru(x, h)
        # print(1, out.shape, flush=True)
        out = self.relu(out[:, -1])
        # print(2, out.shape, flush=True)
        out = self.fc(out)
        # print(3, out.shape, flush=True)
        # print(out, flush=True)
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



# class OmniGRU(nn.Module):
#     def __init__(self, checkpoint=DEFAULT_CHECKPOINT, cfg_file=DEFAULT_CONFIG):
#         super().__init__()
#         self.cfg = cfg = get_cfg()
#         cfg.merge_from_file(cfg_file)
#         action_size = 1024
#         audio_size = 2304
#         input_dim = action_size
#         hidden_dim = cfg.MODEL.HIDDEN_SIZE
#         output_dim = cfg.MODEL.OUTPUT_DIM
#         n_layers = 2
#         drop_prob = 0.2

#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
        
#         self.action_fc = nn.Linear(action_size, int(input_dim/2))
#         self.audio_fc = nn.Linear(audio_size, int(input_dim/2))
#         self.action_bn = nn.BatchNorm1d(int(input_dim/2))
#         self.aud_bn = nn.BatchNorm1d(int(input_dim/2))
#         self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
#         self.init_hidden(32)

#         self.load_state_dict(torch.load(checkpoint))
        
#     def forward(self, action, aud, h=None):
#         action = self.action_fc(action)
#         action = self.action_bn(action.transpose(1, 2)).transpose(1, 2)
#         action = self.relu(action)

#         aud = self.audio_fc(aud)
#         aud = self.aud_bn(aud.transpose(1, 2)).transpose(1, 2)
#         aud = self.relu(aud)
        
#         x = torch.concat((action, aud), -1)
#         x, h = self.gru(x, h)
#         x = self.relu(x[:, -1])
#         x = self.fc(x)
#         return x, h
    
#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
#         return hidden
