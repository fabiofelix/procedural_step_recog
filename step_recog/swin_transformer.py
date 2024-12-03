## https://github.com/pytorch/vision/blob/release/0.17/torchvision/models/swin_transformer.py
## torch                     2.2.2
## torchvision               0.17.2

import copy
from torch import Tensor
from torchvision.models.swin_transformer import SwinTransformerBlock

class SwinTransformerBlock_CrossAttention(SwinTransformerBlock):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def add_norm(self):  
    self.norm3 = copy.deepcopy(self.norm1)

#   #x: V, K - y: Q
  def forward(self, x: Tensor, y: Tensor):
    x = x + self.stochastic_depth(self.attn(self.norm1(x), self.norm3(y)))
    x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
    return x