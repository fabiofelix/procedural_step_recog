## https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.video.swin_transformer import ShiftedWindowAttention3d, _get_window_and_shift_size, _compute_pad_size_3d, _compute_attention_mask_3d
from typing import List, Optional

## input1: V, K - input2: Q
def shifted_window_attention_3d_crossattention(
    input1: Tensor,
    input2: Tensor,
    kv_weight: Tensor,
    q_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    kv_bias: Optional[Tensor] = None,    
    q_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[B, T, H, W, C]): The input tensor, 5-dimensions.
        kv_weight (Tensor[in_dim, out_dim]): The weight tensor of key, value.
        q_weight (Tensor[in_dim, out_dim]): The weight tensor of query.        
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): 3-dimensions window size, T, H, W .
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention (T, H, W).
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        kv_bias (Tensor[out_dim], optional): The bias tensor of key, value. Default: None.
        q_bias (Tensor[out_dim], optional): The bias tensor of query. Default: None.        
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[B, T, H, W, C]: The output tensor after shifted window attention.
    """
    torch._assert(input1.shape == input2.shape, f"input1 and input2 should have the same shape but have {input1.shape} and {input2.shape}!")
    b, t, h, w, c = input1.shape
    # pad feature maps to multiples of window size
    pad_size = _compute_pad_size_3d((t, h, w), (window_size[0], window_size[1], window_size[2]))
    x = F.pad(input1, (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
    x2 = F.pad(input2, (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
    _, tp, hp, wp, _ = x.shape
    padded_size = (tp, hp, wp)

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        x2 = torch.roll(x2, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    # partition windows
    num_windows = (
        (padded_size[0] // window_size[0]) * (padded_size[1] // window_size[1]) * (padded_size[2] // window_size[2])
    )
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        window_size[0],
        padded_size[1] // window_size[1],
        window_size[1],
        padded_size[2] // window_size[2],
        window_size[2],
        c,
    )
    x2 = x2.view(
        b,
        padded_size[0] // window_size[0],
        window_size[0],
        padded_size[1] // window_size[1],
        window_size[1],
        padded_size[2] // window_size[2],
        window_size[2],
        c,
    )    
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        b * num_windows, window_size[0] * window_size[1] * window_size[2], c
    )  # B*nW, Wd*Wh*Ww, C
    x2 = x2.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        b * num_windows, window_size[0] * window_size[1] * window_size[2], c
    )  # B*nW, Wd*Wh*Ww, C    

    # multi-head attention
    kv = F.linear(x, kv_weight, kv_bias)
    q  = F.linear(x2, q_weight, q_bias)
    kv = kv.reshape(x.size(0), x.size(1), 2, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q  = q.reshape(x2.size(0), x2.size(1), 1, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)

    q, k, v = q[0], kv[0], kv[1]
    q = q * (c // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask to handle shifted windows with varying size
        attn_mask = _compute_attention_mask_3d(
            x,
            (padded_size[0], padded_size[1], padded_size[2]),
            (window_size[0], window_size[1], window_size[2]),
            (shift_size[0], shift_size[1], shift_size[2]),
        )
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        padded_size[1] // window_size[1],
        padded_size[2] // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        c,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, tp, hp, wp, c)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

    # unpad features
    x = x[:, :t, :h, :w, :].contiguous()
    return x

class ShiftedWindowAttention3d_CrossAttention(ShiftedWindowAttention3d):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.q  = nn.Linear(in_features=self.qkv.in_features, out_features=self.qkv.in_features, bias=self.qkv.bias is not None) 
    self.kv = nn.Linear(in_features=self.qkv.in_features, out_features=self.qkv.in_features * 2, bias=self.qkv.bias is not None) 

    self.qkv.weight.requires_grad = False
    self.qkv.bias.requires_grad = False

    self.q.weight = nn.parameter.Parameter(copy.deepcopy(self.qkv.weight[:self.qkv.in_features, :]))
    self.q.bias   = None if self.q.bias is None else nn.parameter.Parameter(copy.deepcopy(self.qkv.bias[:self.qkv.in_features]))
    self.q.weight.requires_grad = self.q.bias.requires_grad = True

    self.kv.weight = nn.parameter.Parameter(copy.deepcopy(self.qkv.weight[self.qkv.in_features:, :]))
    self.kv.bias   = None if self.kv.bias is None else nn.parameter.Parameter(copy.deepcopy(self.qkv.bias[self.qkv.in_features:]))
    self.kv.weight.requires_grad = self.qkv.bias.requires_grad = True

    del self.qkv
      
  #x: V, K - y: Q  
  def forward(self, x: Tensor, y: Tensor) -> Tensor:
    _, t, h, w, _ = x.shape
    size_dhw = [t, h, w]
    window_size, shift_size = self.window_size.copy(), self.shift_size.copy()
    # Handle case where window_size is larger than the input tensor
    window_size, shift_size = _get_window_and_shift_size(shift_size, size_dhw, window_size)

    relative_position_bias = self.get_relative_position_bias(window_size)

    return shifted_window_attention_3d_crossattention(
        x,
        y,
        self.kv.weight,  
        self.q.weight,          
        self.proj.weight,
        relative_position_bias,
        window_size,
        self.num_heads,
        shift_size=shift_size,
        attention_dropout=self.attention_dropout,
        dropout=self.dropout,
        kv_bias=self.kv.bias,        
        q_bias=self.q.bias,
        proj_bias=self.proj.bias,
        training=self.training,
    )        