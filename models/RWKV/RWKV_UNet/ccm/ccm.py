import torch
import torch.nn as nn
import einops
from .main_blocks import *
from einops import rearrange, reduce
from timm.layers.activations import *
from timm.layers import DropPath, trunc_normal_
import math
from torch.utils.cpp_extension import load
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import DropPath, create_act_layer
import numpy as np
import torchvision
from typing import Callable, Dict, Optional, Type
import pickle
import os




def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)



class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1/4, shift_pixel=1, hidden_rate=2, 
                 key_norm=True):
        super().__init__()
        self.n_embd = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self):
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(xr)) * kv
        return x

class CCMix(nn.Module):
    def __init__(self, in_dims, target_dim, target_size):
        super(CCMix, self).__init__()
        self.target_dim = target_dim
        self.target_size = target_size
        # Projection layers to unify dimensions
        self.projections = nn.ModuleList([nn.Conv2d(in_dim, target_dim, kernel_size=1) for in_dim in in_dims])
        self.ln1 = nn.LayerNorm(target_dim*3)
        self.drop_path = DropPath(0.05) if drop_path else nn.Identity()
        self.channel = VRWKV_ChannelMix(n_embd=target_dim*3, channel_gamma=1/4, shift_pixel=1, hidden_rate=2)
        self.final_projections = nn.ModuleList([nn.Conv2d(target_dim, in_dim, kernel_size=1) for in_dim in in_dims])
        self.original_sizes= [target_size//4,target_size//2,target_size]
    def forward(self, features):
        # Step 1: Up-sample and project each feature map
        upsampled_features=[]
        output_features=[]
        for i, feature in enumerate(features):
            # Upsample to target size
            feature = F.interpolate(feature, size=self.target_size, mode='bilinear', align_corners=False)
            # Project to target dimension directly
            feature = self.projections[i](feature)
            upsampled_features.append(feature)
        
        # Step 2: Concatenate features along the channel axis
        concatenated = torch.cat(upsampled_features, dim=1)
        # Prepare for MHSA: (B, C, H, W) -> (B, N, C) where N=H*W
        B, C, H, W = concatenated.shape
        concatenated = concatenated.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        # Apply MHSA
        attn_output = concatenated + self.drop_path(self.ln1(self.channel(concatenated, (self.target_size,self.target_size))))
        
        # Step 3: Reshape back to (B, C, H, W)
        B, n_patch, hidden = attn_output.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidde
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        attn_output = attn_output.permute(0, 2, 1)
        attn_output = attn_output.contiguous().view(B, hidden, h, w)

        # Step 4: Split output back into four feature maps with original dimensions
        split_features = torch.split(attn_output, self.target_dim, dim=1)
        for i, split_feature in enumerate(split_features):
            # Project back to original dimension
            split_feature = self.final_projections[i](split_feature)  # (B, H, W, original_dim)
            # Resize to original size
            split_feature = F.interpolate(split_feature, size=self.original_sizes[i], mode='bilinear', align_corners=False)
            output_features.append(split_feature)
        
        return output_features

