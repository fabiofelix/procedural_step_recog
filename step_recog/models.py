#=========================================================================#
#Code from https://github.com/VIDA-NYU/ptg-server-ml/tree/main/ptgprocess #
#=========================================================================#

import torch
import ipdb
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def custom_weights(layer):
  if isinstance(layer, torch.nn.Linear):
    torch.nn.init.xavier_normal_(layer.weight)  
    torch.nn.init.zeros_(layer.bias)

class OmniGRU(torch.nn.Module):
    def __init__(self, cfg, load = False):
        super().__init__()

        action_size = 1024 #default Omnivore output
        audio_size  = 2304 #default Slowfast output
        img_size    = 517  #default Clip output (512) + Yolo bouding box (4) + Yolo confidence (1)
        project_dim = 512  #default space for each feature

        self.use_action = cfg.MODEL.USE_ACTION
        self.use_objects = cfg.MODEL.USE_OBJECTS
        self.use_audio = cfg.MODEL.USE_AUDIO
        self.use_bn = cfg.MODEL.USE_BN
        self.skills = cfg.MODEL.SKILLS
        self.hidden_dim = cfg.MODEL.HIDDEN_SIZE
        self.number_classes = cfg.MODEL.OUTPUT_DIM + 1 #adding no step
        self.number_position = 2 # adding window position in a step to the output

        self.n_gru_layers = 2
        gru_input_dim = 0

        if self.use_action:
          gru_input_dim += project_dim
          self.action_fc = torch.nn.Linear(action_size, project_dim)
          self.action_drop_out = torch.nn.Dropout(cfg.MODEL.DROP_OUT)
          if self.use_bn: 
            self.action_bn = torch.nn.BatchNorm1d(project_dim)

        if self.use_audio:
          gru_input_dim += project_dim
          self.audio_fc = torch.nn.Linear(audio_size, project_dim)
          self.audio_drop_out = torch.nn.Dropout(cfg.MODEL.DROP_OUT)
          if self.use_bn: 
              self.aud_bn = torch.nn.BatchNorm1d(project_dim)

        if self.use_objects:
          gru_input_dim += project_dim
          self.obj_proj   = torch.nn.Linear(img_size, project_dim)    
          self.frame_proj = torch.nn.Linear(img_size, project_dim)  
          self.obj_fc     = torch.nn.Linear(project_dim, project_dim)
          self.obj_drop_out = torch.nn.Dropout(cfg.MODEL.DROP_OUT)
          if self.use_bn: 
            self.obj_bn = torch.nn.BatchNorm1d(project_dim)            

        if gru_input_dim == 0:
           raise Exception("GRU has to use at least one input (action, object/frame, or audio)")             

        self.gru = torch.nn.GRU(gru_input_dim, cfg.MODEL.HIDDEN_SIZE, self.n_gru_layers, batch_first=True, dropout=cfg.MODEL.DROP_OUT)
        self.fc = torch.nn.Linear(cfg.MODEL.HIDDEN_SIZE, self.number_classes + self.number_position)
        self.relu = torch.nn.ReLU()

        if load:
          self.load_state_dict( self.update_version(torch.load( cfg.MODEL.OMNIGRU_CHECKPOINT_URL )))
        else:
          self.apply(custom_weights)

    def forward(self, action, h=None, aud=None, objs=None, frame=None, return_last_step=True):
        x = []

        if self.use_action:
            action = self.action_fc(action)
            if self.use_bn:
                action = self.action_bn(action.transpose(1, 2)).transpose(1, 2)
            action = self.relu(action)
            action = self.action_drop_out(action)
            x.append(action)

        if self.use_audio:
            aud = self.audio_fc(aud)
            if self.use_bn:
                aud = self.aud_bn(aud.transpose(1, 2)).transpose(1, 2)
            aud = self.relu(aud)
            aud = self.audio_drop_out(aud)            
            x.append(aud)

        if self.use_objects:
            obj_proj = self.relu(self.obj_proj(objs))
            frame_proj = self.relu(self.frame_proj(frame))

            #=============================== Original tests ===============================#
#            values = torch.softmax(torch.sum(frame_proj * obj_proj, dim=-1, keepdims=True), dim=-2)
#            obj_in = torch.sum(obj_proj * values, dim=-2)

            #=============================== Conceptual Fusion ===============================#
            theta = torch.nn.functional.cosine_similarity(frame_proj, obj_proj, dim = -1)

            phi   = torch.nn.functional.cosine_similarity(obj_proj[:, :, None, :, :], obj_proj[:, :, :, None, :], dim = -1)
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
            obj_in = self.obj_drop_out(obj_in)                        
            x.append(obj_in)

        x = torch.concat(x, -1) if len(x) > 1 else x[0]            
        out, h = self.gru(x, h)
        out = self.relu(out[:, -1]) if return_last_step else self.relu(out)
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_gru_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

    def update_version(self, state_dict):
      new_dict = OrderedDict()

      for key, value in state_dict.items():
        if "rgb" in key:
          key = key.replace("rgb", "action")  

        new_dict[key] = value
          
      return new_dict    
