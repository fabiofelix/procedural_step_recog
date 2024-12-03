
import torch
import torch.nn as nn
import torchvision
import clip
import math
import ipdb
import copy
from step_recog import utils
import os
from enum import IntEnum
from collections import OrderedDict

class _TextEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder, _ = clip.load("ViT-B/16", jit=False, download_root=utils.clip_download_root)
    self.encoder.cuda()
    self.encoder.eval()

  @torch.no_grad()
  def forward(self, text):
    text = clip.tokenize(text).cuda()

    return self.encoder.encode_text(text).detach().cpu().float()
  
__textencoder = None

def prepare_txt(text_batch):
  if text_batch is not None:
    global __textencoder
    if __textencoder is None:
      __textencoder = _TextEncoder()

    text_batch = __textencoder(text_batch)

  torch.cuda.empty_cache()
  return text_batch

#batch has (B, T, H, W, C) or (T, H, W, C)
def prepare_img(img_batch, input_channels_last=False):
  if isinstance(img_batch, np.ndarray):
    img_batch = torch.from_numpy(img_batch)

  #Adds a batch axis if img_batch has just (T, H, W, C)
  img_batch  = img_batch[None] if len(img_batch.shape) == 4 else img_batch
  #If necessary put channel in the right position
  img_batch  = img_batch.permute(0, 1, 4, 2, 3) if input_channels_last else img_batch
  preprocess = torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1.transforms()
  #32 frames per clip described in the paper
  frame_idx  = np.linspace(0, img_batch.shape[1] - 1, 32).astype('long')
  img_batch  = img_batch[:, frame_idx, :, :, :]

  torch.cuda.empty_cache()
  return preprocess(img_batch.cpu()).cpu().float()

class PTGPerceptionBases(nn.Module):
  def __init__(self):
    super().__init__()
    self._device = nn.Parameter(torch.empty(0))

  #Equal to SwinTransformer3d 
  def _init_layer(self, layer):
    if isinstance(layer, nn.Linear):
      nn.init.trunc_normal_(layer.weight, std=0.02)

      if layer.bias is not None:
        nn.init.zeros_(layer.bias)   

  def update_version(self, state_dict):
    new_dict = OrderedDict()

    for key in state_dict:
      if "encoder." not in key:
        new_dict[key] = state_dict[key]

    return new_dict

  def summary(self):
    trainable_params = 0
    non_trainable_params = 0
    total_params = 0

    for param in self.parameters():
      total_params += param.numel()

      if param.requires_grad:
        trainable_params += param.numel()
      else:
        non_trainable_params += param.numel()

    print("{:24s}".format("|- Trainable params:"), "{:,}".format(trainable_params))
    print("{:24s}".format("|- Non-trainable params:"), "{:,}".format(non_trainable_params))
    print("{:24s}".format("|- Total params:"), "{:,}".format(total_params))      



import numpy as np
from functools import partial
from torchvision.models.video import Swin3D_T_Weights
from .swin_transformer import SwinTransformerBlock_CrossAttention
from .video_swin_transformer import ShiftedWindowAttention3d_CrossAttention

##Basic model 
class OmniTransformer_v3(PTGPerceptionBases):
  def __init__(self, cfg, load = False, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.number_classes = cfg.MODEL.OUTPUT_DIM + 1 #adding no step
    self.image_branch = torchvision.models.video.swin3d_t(weights = Swin3D_T_Weights.KINETICS400_V1)  #params = 28,158,070
    self.__prepare_branches()

    if load:
      self.load_state_dict( self.update_version(torch.load( cfg.MODEL.OMNIGRU_CHECKPOINT_URL )))    

  def __prepare_branches(self):
    ##Configure IMAGE head
    self.image_branch.head = nn.Linear(in_features=self.image_branch.head.in_features, out_features=self.number_classes, bias=self.image_branch.head.bias is not None)    
    self._init_layer(self.image_branch.head)

  def forward(self, img, text = None):
    return self.image_branch(img)

class CombinationType(IntEnum):
  CONCAT  = 0       
  ATTN    = 1
  CONCEPT = 2  

##Concat text features with image tokens
class OmniTransformer_v4(PTGPerceptionBases):
  def __init__(self, cfg, load = False, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.number_classes = cfg.MODEL.OUTPUT_DIM + 1 #adding no step
    self.image_branch = torchvision.models.video.swin3d_t(weights = Swin3D_T_Weights.KINETICS400_V1)  #params = 28,158,070
    self.comb_type = CombinationType.CONCAT
    clip_embedding = 512
    hidden_state = 512

    self.proj_X = nn.Sequential(
             nn.LayerNorm(self.image_branch.patch_embed.norm.normalized_shape, eps=self.image_branch.patch_embed.norm.eps),
             nn.Linear(self.image_branch.patch_embed.norm.normalized_shape[0], hidden_state),
             nn.GELU(),
             nn.Dropout(self.image_branch.pos_drop.p)
            )
    self.proj_Z = nn.Sequential(
              nn.LayerNorm(clip_embedding, eps=self.image_branch.patch_embed.norm.eps),
              nn.Linear(clip_embedding, hidden_state),
              nn.GELU(),
              nn.Dropout(self.image_branch.pos_drop.p)
            )    
    proj_XZ_input_shape = 2 * hidden_state if self.comb_type == CombinationType.CONCAT else hidden_state
    self.proj_XZ = nn.Sequential(
              nn.LayerNorm(proj_XZ_input_shape, eps=self.image_branch.patch_embed.norm.eps),
              nn.Linear(proj_XZ_input_shape, self.image_branch.patch_embed.norm.normalized_shape[0]),
              nn.GELU(),
              nn.Dropout(self.image_branch.pos_drop.p)
            )    

    for layer in self.proj_X:
      self._init_layer(layer)
    for layer in self.proj_Z:
      self._init_layer(layer)
    for layer in self.proj_XZ:
      self._init_layer(layer)

    self.__prepare_branches()

    if load:
      self.load_state_dict( self.update_version(torch.load( cfg.MODEL.OMNIGRU_CHECKPOINT_URL )))

  def __prepare_branches(self):
    ##Configure IMAGE head
    self.image_branch.head = nn.Linear(in_features=self.image_branch.head.in_features, out_features=self.number_classes, bias=self.image_branch.head.bias is not None)    
    self._init_layer(self.image_branch.head)

  def forward(self, img, text):
    ##Equal to SwinTransformer3d.forward()
    X = self.image_branch.patch_embed(img)  # B _T _H _W _C

    ##Combination of image tokens and text features
    #1. Interpolate text features to match img batch (S, F) => (B, 1, 1, 1, F)
    Z = torch.nn.functional.interpolate(input = text[None, None, None, ...], size = (1, X.shape[0], text.shape[1]), mode = "nearest")
    Z = Z.permute(3, 0, 1, 2, 4)
    #2. Repate text features for each image token and project them (B, 1, 1, 1, F) => (B, _T, _H, _W, F)
    Z = Z.repeat(1, X.shape[1], X.shape[2], X.shape[3], 1)
    Z_proj = self.proj_Z(Z)
    #3. Project image tokens to have the text feature shape
    X_proj = self.proj_X(X)
    #4. Combining image and text
    if self.comb_type == CombinationType.CONCAT: 
      XZ = torch.concat((X_proj, Z_proj), -1)
    elif self.comb_type == CombinationType.ATTN: 
      XZ = torch.nn.functional.scaled_dot_product_attention(Z_proj, X_proj, X_proj, dropout_p=0.0)
    elif self.comb_type == CombinationType.CONCEPT:       
      theta = torch.nn.functional.cosine_similarity(X_proj, Z_proj, dim = -1)
      w     = torch.softmax(theta, dim = -1)
      w     = w[:, :, :, :, None]
      XZ    = w * X_proj + (1 - w) * Z_proj
    #5. Project the concatenation to a space with the original feature shape and do a skip-connection (only way to force the training process to converge)
    X = X + self.proj_XZ(XZ)

    ##Equal to SwinTransformer3d.forward()
    X = self.image_branch.pos_drop(X)
    X = self.image_branch.features(X)
    X = self.image_branch.norm(X)
    X = X.permute(0, 4, 1, 2, 3)
    X = self.image_branch.avgpool(X)
    X = torch.flatten(X, 1)
    
    return self.image_branch.head(X)      

class CrossType(IntEnum):
  ALL_ATTN   = 0       
  FIRST_ATTN = 1
  
##Send text features to Cross-Attention  
class OmniTransformer_v5(PTGPerceptionBases):
  def __init__(self, cfg, load = False, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.cross_type = CrossType.ALL_ATTN
    self.number_classes = cfg.MODEL.OUTPUT_DIM + 1 #adding no step

    if self.cross_type == CrossType.ALL_ATTN:
      self.image_branch = torchvision.models.video.swin3d_t(weights = Swin3D_T_Weights.KINETICS400_V1, block = SwinTransformerBlock_CrossAttention)  #params = 28,158,070
    elif self.cross_type == CrossType.FIRST_ATTN:
      self.image_branch = torchvision.models.video.swin3d_t(weights = Swin3D_T_Weights.KINETICS400_V1)  #params = 28,158,070      

    self.__prepare_branches()
    self.__prepare_q_layers()

    if load:
      self.load_state_dict( self.update_version(torch.load( cfg.MODEL.OMNIGRU_CHECKPOINT_URL )))

  def __prepare_branches(self):
    ##Configure IMAGE features
#    ipdb.set_trace()
    ##Replaces ALL block self-attention with cross-attention
    if self.cross_type == CrossType.ALL_ATTN:
      for stage in self.image_branch.features:
        if isinstance(stage, torch.nn.Sequential):
          for block in stage:
            block.add_norm()
            new_attn = ShiftedWindowAttention3d_CrossAttention(
                            block.attn.qkv.in_features, 
                            block.attn.window_size, 
                            block.attn.shift_size, 
                            block.attn.num_heads, 
                            attention_dropout=block.attn.attention_dropout, 
                            dropout=block.attn.dropout 
                        )
            new_attn.copy_self2cross(block.attn)
            block.attn = new_attn
    ##Replaces only the FRIST block self-attention with cross-attention            
    elif self.cross_type == CrossType.FIRST_ATTN: 
      for idx, stage in enumerate(self.image_branch.features):
        if isinstance(stage, torch.nn.Sequential):
          new_block = SwinTransformerBlock_CrossAttention(
            dim=stage[0].attn.qkv.in_features, 
            num_heads=stage[0].attn.num_heads, 
            window_size=stage[0].attn.window_size, 
            shift_size=stage[0].attn.shift_size, 
            dropout=stage[0].attn.dropout, 
            attention_dropout=stage[0].attn.attention_dropout, 
            stochastic_depth_prob=stage[0].stochastic_depth.p, 
            norm_layer=type(stage[0].norm1), 
            attn_layer=ShiftedWindowAttention3d_CrossAttention
          )
          new_block.add_norm()
          new_block.attn.copy_self2cross(stage[0].attn)
          self.image_branch.features[idx][0] = new_block                 

    ##Configure IMAGE head
    self.image_branch.head = nn.Linear(in_features=self.image_branch.head.in_features, out_features=self.number_classes, bias=self.image_branch.head.bias is not None)    
    self._init_layer(self.image_branch.head)

  def __prepare_q_layers(self):
    self.proj_Z = []
    clip_embedding = 512

    for stage in self.image_branch.features:
      if isinstance(stage, torch.nn.Sequential):
        clip_proj = nn.Sequential(
              nn.LayerNorm(clip_embedding, eps=self.image_branch.patch_embed.norm.eps),
              nn.Linear(clip_embedding, stage[0].attn.kv.in_features),
              nn.GELU(),
              nn.Dropout(self.image_branch.pos_drop.p)
            )
        for layer in clip_proj:
          self._init_layer(layer)
        self.proj_Z.append(clip_proj)
      else:
        self.proj_Z.append(None)         

    self.proj_Z = nn.Sequential(*self.proj_Z)        

  def forward(self, img, text = None):
    X = self.image_branch.patch_embed(img)  # B _T _H _W _C
    X = self.image_branch.pos_drop(X)

    ##Combination of image tokens and text features
    #1. Interpolate text features to match img batch (S, 1, 1, 1, F) => (B, 1, 1, 1, F)
    Z = torch.nn.functional.interpolate(input = text[None, None, None, ...], size = (1, X.shape[0], text.shape[1]), mode = "nearest")
    Z = Z.permute(3, 0, 1, 2, 4)
    #2. Repate text features for each image token and project them (B, 1, 1, 1, F) => (B, _T, _H, _W, F)
    Z_repeat = Z.repeat(1, X.shape[1], X.shape[2], X.shape[3], 1)    

    if self.cross_type == CrossType.ALL_ATTN:
      for image_stage, z_proj in zip(self.image_branch.features, self.proj_Z):
        if isinstance(image_stage, torch.nn.Sequential):
          for image_block in image_stage:
            Z_aux = z_proj(Z_repeat)
            X = image_block(X, Z_aux)
        else: #PatchMerging
          X = image_stage(X)
          #2. Repate text features for each image token and project them (B, 1, 1, 1, F) => (B, _T, _H, _W, F)
          Z_repeat = Z.repeat(1, X.shape[1], X.shape[2], X.shape[3], 1)  
    elif self.cross_type == CrossType.FIRST_ATTN:           
      for image_stage, z_proj in zip(self.image_branch.features, self.proj_Z):
        if isinstance(image_stage, torch.nn.Sequential):
          for image_block in image_stage:
            if isinstance(image_block, SwinTransformerBlock_CrossAttention):
              Z_aux = z_proj(Z_repeat)
              X = image_block(X, Z_aux)
            else:  
              X = image_block(X)
        else: #PatchMerging
          X = image_stage(X)
          #2. Repate text features for each image token and project them (B, 1, 1, 1, F) => (B, _T, _H, _W, F)
          Z_repeat = Z.repeat(1, X.shape[1], X.shape[2], X.shape[3], 1)       

    ##Equal to SwinTransformer3d.forward()
    X = self.image_branch.norm(X)
    X = X.permute(0, 4, 1, 2, 3)
    X = self.image_branch.avgpool(X)
    X = torch.flatten(X, 1)
    
    return self.image_branch.head(X)  

