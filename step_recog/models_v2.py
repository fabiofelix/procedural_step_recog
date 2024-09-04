
import torch
import torch.nn as nn
import torchvision
import clip
import math
import ipdb
from step_recog import utils

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

    self.encoder, _ = clip.load("ViT-B/16", jit=False, download_root=utils.clip_download_root)
    self.encoder.eval()

    for param in self.encoder.parameters():
      param.requires_grad = False

  def to(self, device):
    super().to(device)

    if self.encoder is not None:
      self.encoder = self.encoder.cpu()

    return self  

  #Equal to SwinTransformer3d 
  def _init_layer(self, layer):
    if isinstance(layer, nn.Linear):
      nn.init.trunc_normal_(layer.weight, std=0.02)

      if layer.bias is not None:
        nn.init.zeros_(layer.bias)      

  def update_version(self, state_dict):
    return state_dict

  def prepare_txt(self, text_batch):
    if text_batch is not None:
      encoder_param = next(self.encoder.parameters())
      text_batch = clip.tokenize(text_batch).to(encoder_param.device)
      text_batch = self.encoder.encode_text(text_batch)[:, None, None, None, :].cpu().detach().float()      

    torch.cuda.empty_cache()
    return text_batch  

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
    self.image_branch = torchvision.models.video.swin3d_t(weights = Swin3D_T_Weights.KINETICS400_V1)  #params = 36,610,672
    self.__prepare_branches()

    if load:
      self.load_state_dict( self.update_version(torch.load( cfg.MODEL.OMNIGRU_CHECKPOINT_URL )))    

  def __prepare_branches(self):
    ##Configure IMAGE head
    self.image_branch.head = nn.Linear(in_features=self.image_branch.head.in_features, out_features=self.number_classes, bias=self.image_branch.head.bias is not None)    
    self._init_layer(self.image_branch.head)

  @torch.no_grad()
  def prepare(self, img_batch, text_batch = None, input_channels_last=False):
    return prepare_img(img_batch, input_channels_last=input_channels_last).to(self._device.device).float(), self.prepare_txt(text_batch)

  def forward(self, img, text = None):
    return self.image_branch(img)

##Concat text features with image tokens
class OmniTransformer_v4(PTGPerceptionBases):
  def __init__(self, cfg, load = False, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.number_classes = cfg.MODEL.OUTPUT_DIM + 1 #adding no step
    self.image_branch = torchvision.models.video.swin3d_t(weights = Swin3D_T_Weights.KINETICS400_V1)  #params = 36,610,672

    self.proj_X = torch.nn.Linear(self.image_branch.patch_embed.norm.normalized_shape[0], 512)  #project image tokens to a 512-D space, equal to CLIP
    self.proj_Z = torch.nn.Linear(512, 512)  #project CLIP space
    self.proj_XZ = torch.nn.Linear(1024, self.image_branch.patch_embed.norm.normalized_shape[0]) #project concatenation of image and text to the original patch_embed space 
    self.proj_gelu = torch.nn.GELU()

    self._init_layer(self.proj_X)
    self._init_layer(self.proj_Z)
    self._init_layer(self.proj_XZ)

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
    X = self.image_branch.pos_drop(X)

    ##Combination of image tokens and text features
    #1. Interpolate text features to match img batch (S, 1, 1, 1, F) => (B, 1, 1, 1, F)
    Z = text.permute(1, 2, 3, 0, 4) 
    Z = torch.nn.functional.interpolate(input = Z, size = (Z.shape[-3], X.shape[0], Z.shape[-1]), mode = "nearest")
    Z = Z.permute(3, 0, 1, 2, 4)
    #2. Repate text features for each image token and project them (B, 1, 1, 1, F) => (B, _T, _H, _W, F)
    Z = Z.repeat(1, X.shape[1], X.shape[2], X.shape[3], 1)
    Z = self.proj_gelu(self.proj_Z(Z))
    #3. Project image tokens to have the text feature shape
    X_proj = self.proj_gelu(self.proj_X(X))
    #4. Concatenate
    XZ = torch.concat((X_proj, Z), -1)
    #5. Project the concatenation to a space with the original feature shape
    X = self.proj_gelu(self.proj_XZ(XZ))

    ##Equal to SwinTransformer3d.forward()
    X = self.image_branch.features(X)
    X = self.image_branch.norm(X)
    X = X.permute(0, 4, 1, 2, 3)
    X = self.image_branch.avgpool(X)
    X = torch.flatten(X, 1)
    
    return self.image_branch.head(X)      
  
##Send text features to Cross-Attention  
class OmniTransformer_v5(PTGPerceptionBases):
  def __init__(self, cfg, load = False, *args, **kwargs):
    super().__init__(*args, **kwargs)
#    ipdb.set_trace()

    self.number_classes = cfg.MODEL.OUTPUT_DIM + 1 #adding no step
    self.image_branch = torchvision.models.video.swin3d_t(weights = Swin3D_T_Weights.KINETICS400_V1, block = partial(SwinTransformerBlock_CrossAttention, attn_layer=ShiftedWindowAttention3d_CrossAttention))  #params = 36,610,672
    
    self.__prepare_branches()
    self.proj_Z = []

    for feat in self.image_branch.features:
      if isinstance(feat, torch.nn.Sequential):
        self.proj_Z.append(torch.nn.Linear(512, feat[0].attn.kv.in_features))
      else:
        self.proj_Z.append(None)  

    self.proj_Z = nn.Sequential(*self.proj_Z)
    self.proj_gelu = torch.nn.GELU()

    self._init_layer(self.proj_Z)

    if load:
      self.load_state_dict( self.update_version(torch.load( cfg.MODEL.OMNIGRU_CHECKPOINT_URL )))    


  def __prepare_branches(self):
    ##Configure IMAGE features
    ##Even when passing block = partial(..., attn_layer= ...), block doesn't use the correct class to create attn
    for feat in self.image_branch.features:
      if isinstance(feat, torch.nn.Sequential):
        for block in feat:
          block.add_norm()
          block.attn = ShiftedWindowAttention3d_CrossAttention(
                          block.attn.qkv.in_features, 
                          block.attn.window_size, 
                          block.attn.shift_size, 
                          block.attn.num_heads, 
                          attention_dropout=block.attn.attention_dropout, 
                          dropout=block.attn.attention_dropout 
                      )

    ##Configure IMAGE head
    self.image_branch.head = nn.Linear(in_features=self.image_branch.head.in_features, out_features=self.number_classes, bias=self.image_branch.head.bias is not None)    
    self._init_layer(self.image_branch.head)

  def forward(self, img, text):
    ##Equal to SwinTransformer3d.forward()
    X = self.image_branch.patch_embed(img)  # B _T _H _W _C
    X = self.image_branch.pos_drop(X)

    ##Combination of image tokens and text features
    #1. Interpolate text features to match img batch (S, 1, 1, 1, F) => (B, 1, 1, 1, F)
    Z = text.permute(1, 2, 3, 0, 4) 
    Z = torch.nn.functional.interpolate(input = Z, size = (Z.shape[-3], X.shape[0], Z.shape[-1]), mode = "nearest")
    Z = Z.permute(3, 0, 1, 2, 4)
    #2. Repate text features for each image token and project them (B, 1, 1, 1, F) => (B, _T, _H, _W, F)
    Z = Z.repeat(1, X.shape[1], X.shape[2], X.shape[3], 1)
    
    for image_feat, z_proj in zip(self.image_branch.features, self.proj_Z):
      if isinstance(image_feat, torch.nn.Sequential):
        for image_block in image_feat:
          Z = self.proj_gelu(z_proj(Z))
          X = image_block(Z, X)
      else:
        X = image_feat(X)

    ##Equal to SwinTransformer3d.forward()
    X = self.image_branch.features(X)
    X = self.image_branch.norm(X)
    X = X.permute(0, 4, 1, 2, 3)
    X = self.image_branch.avgpool(X)
    X = torch.flatten(X, 1)
    
    return self.image_branch.head(X)  
  
