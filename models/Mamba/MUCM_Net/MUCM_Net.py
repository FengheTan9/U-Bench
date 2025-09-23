import torch


import torchvision

import torch.nn as nn
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types


import pdb

__all__ = ['MUCM_Net','MUCM_Net_2','MUCM_Net_4','MUCM_Net_8']
'''
        self.fc21 =       MambaLayer(
           d_model=mlp_hidden_dim,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
'''

from mamba_ssm import Mamba


#from mamba_ssm import Mamba

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super(MambaLayer, self).__init__()
        # Initialize the Mamba model as part of this layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # You can add more layers here if needed
        # e.g., self.linear = nn.Linear(d_model, some_other_dimension)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        # Pass the input through the Mamba model
        x = self.mamba(x)
        
        # Additional operations can go here
        # e.g., x = self.linear(x)
        
        return x


import torch


import torchvision

import torch.nn as nn

import math


from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import pdb




class LayerNorm(nn.Module):
    r""" From ConvNeXt ([URL]
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


    

class UCMBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        
        # Original UCMBlock initializations
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # Merged shiftmlp components
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.dwconv = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.dwconv1 = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.act = nn.GELU()
        self.act1 = nn.GELU()  # Assuming Activation is a placeholder for an actual activation like GELU
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        
        self.drop = nn.Dropout(drop)
        
        
        
        self.fc21 =       MambaLayer(
           d_model=mlp_hidden_dim,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        
        
        
       # self.fc3 = nn.Linear(mlp_hidden_dim, dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        
        # Weight initialization for merged components
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Norm and DropPath from original UCMBlock
        x = self.norm2(x)
       
        
        # Begin merged shiftmlp forward logic
        B, N, C = x.shape
        x1 = x.clone().detach()
        x2 = self.fc21(x1)

       # x2= self.norm3(x2)
       # x1 = self.fc3(x1)
       # x2=  self.act(x2)
        

        x1 =x2+x1
        
        x = self.fc1(x)
        #print(x.shape)
        x = self.dwconv(x, H, W)
        #print(x.shape)
        
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = self.act1(xn)
        
        x = self.drop(xn)
        x_s = x.reshape(B, C, H * W).contiguous()
        x = x_s.transpose(1, 2)
        
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.dwconv1(x, H, W)
        x = self.drop(x)
        
        
        x += x1
        
        # Apply DropPath
        x = x + self.drop_path(x)
        
        return x
class UCMBlock_2(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        
        # Original UCMBlock initializations
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # Merged shiftmlp components
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.dwconv = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.dwconv1 = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.act = nn.GELU()
        self.act1 = nn.GELU()  # Assuming Activation is a placeholder for an actual activation like GELU
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        
        self.drop = nn.Dropout(drop)
        
        
        
        self.fc21 =       MambaLayer(
           d_model=mlp_hidden_dim//2,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        self.fc22 =       MambaLayer(
           d_model=mlp_hidden_dim//2,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        
        
        
       # self.fc3 = nn.Linear(mlp_hidden_dim, dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        
        # Weight initialization for merged components
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Norm and DropPath from original UCMBlock
        x = self.norm2(x)
       
        
        # Begin merged shiftmlp forward logic
        B, N, C = x.shape
        x1 = x.clone().detach()
      #  print("Shape of x1 :", x1.shape)
     

        halves = torch.split(x1, x1.shape[2] // 2, dim=2)   
       # print("Shape of halves[0]:", halves[0].shape)
       # print("Shape of halves[1]:", halves[1].shape)
        x2_1 = self.fc21(halves[0])
        x2_2 = self.fc22(halves[1])
        x2 = torch.cat((x2_1, x2_2), dim=2)

       # x2= self.norm3(x2)
       # x1 = self.fc3(x1)
       # x2=  self.act(x2)
        

        x1 =x2+x1
        
        x = self.fc1(x)
        #print(x.shape)
        x = self.dwconv(x, H, W)
        #print(x.shape)
        
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = self.act1(xn)
        
        x = self.drop(xn)
        x_s = x.reshape(B, C, H * W).contiguous()
        x = x_s.transpose(1, 2)
        
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.dwconv1(x, H, W)
        x = self.drop(x)
        
        
        x += x1
        
        # Apply DropPath
        x = x + self.drop_path(x)
        
        return x
class UCMBlock_8(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        
        # Original UCMBlock initializations
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # Merged shiftmlp components
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.dwconv = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.dwconv1 = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.act = nn.GELU()
        self.act1 = nn.GELU()  # Assuming Activation is a placeholder for an actual activation like GELU
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        
        self.drop = nn.Dropout(drop)
        
        
        
        self.fc21 =       MambaLayer(
           d_model=mlp_hidden_dim//8,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        self.fc22 =       MambaLayer(
           d_model=mlp_hidden_dim//8,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        
        self.fc23 =       MambaLayer(
           d_model=mlp_hidden_dim//8,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        self.fc24 =       MambaLayer(
           d_model=mlp_hidden_dim//8,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        
        self.fc25 =       MambaLayer(
           d_model=mlp_hidden_dim//8,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        self.fc26 =       MambaLayer(
           d_model=mlp_hidden_dim//8,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        
        self.fc27 =       MambaLayer(
           d_model=mlp_hidden_dim//8,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        self.fc28 =       MambaLayer(
           d_model=mlp_hidden_dim//8,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        
        
        
       # self.fc3 = nn.Linear(mlp_hidden_dim, dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        
        # Weight initialization for merged components
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Norm and DropPath from original UCMBlock
        x = self.norm2(x)
       
        
        # Begin merged shiftmlp forward logic
        B, N, C = x.shape
        x1 = x.clone().detach()
       # print("Shape of x1 :", x1.shape)
     

        halves = torch.split(x1, x1.shape[2] // 8, dim=2)   
       ## print("Shape of halves[0]:", halves[0].shape)
       # print("Shape of halves[1]:", halves[1].shape)
        x2_1 = self.fc21(halves[0])
        x2_2 = self.fc22(halves[1])
        x2_3 = self.fc21(halves[2])
        x2_4 = self.fc22(halves[3])
        x2_5 = self.fc21(halves[4])
        x2_6 = self.fc22(halves[5])
        x2_7 = self.fc21(halves[6])
        x2_8 = self.fc22(halves[7])
        x2 = torch.cat((x2_1, x2_2,x2_3,x2_4,x2_5, x2_6,x2_7,x2_8), dim=2)

       # x2= self.norm3(x2)
       # x1 = self.fc3(x1)
       # x2=  self.act(x2)
        

        x1 =x2+x1
        
        x = self.fc1(x)
        #print(x.shape)
        x = self.dwconv(x, H, W)
        #print(x.shape)
        
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = self.act1(xn)
        
        x = self.drop(xn)
        x_s = x.reshape(B, C, H * W).contiguous()
        x = x_s.transpose(1, 2)
        
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.dwconv1(x, H, W)
        x = self.drop(x)
        
        
        x += x1
        
        # Apply DropPath
        x = x + self.drop_path(x)
        
        return x
    
class UCMBlock_4(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        
        # Original UCMBlock initializations
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # Merged shiftmlp components
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.dwconv = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.dwconv1 = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.act = nn.GELU()
        self.act1 = nn.GELU()  # Assuming Activation is a placeholder for an actual activation like GELU
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        
        self.drop = nn.Dropout(drop)
        
        
        
        self.fc21 =       MambaLayer(
           d_model=mlp_hidden_dim//4,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        self.fc22 =       MambaLayer(
           d_model=mlp_hidden_dim//4,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        
        self.fc23 =       MambaLayer(
           d_model=mlp_hidden_dim//4,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        self.fc24 =       MambaLayer(
           d_model=mlp_hidden_dim//4,
    d_state=4,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        
        
        
       # self.fc3 = nn.Linear(mlp_hidden_dim, dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        
        # Weight initialization for merged components
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Norm and DropPath from original UCMBlock
        x = self.norm2(x)
       
        
        # Begin merged shiftmlp forward logic
        B, N, C = x.shape
        x1 = x.clone().detach()
      #  print("Shape of x1 :", x1.shape)
     

        halves = torch.split(x1, x1.shape[2] // 4, dim=2)  
        print("Shape of x:", x.shape)
        print("Shape of halves[0]:", halves[0].shape)
       # print("Shape of halves[1]:", halves[1].shape)
        x2_1 = self.fc21(halves[0])
        x2_2 = self.fc22(halves[1])
        x2_3 = self.fc21(halves[2])
        x2_4 = self.fc22(halves[3])
        x2 = torch.cat((x2_1, x2_2,x2_3,x2_4), dim=2)

       # x2= self.norm3(x2)
       # x1 = self.fc3(x1)
       # x2=  self.act(x2)
        

        x1 =x2+x1
        
        x = self.fc1(x)
        #print(x.shape)
        x = self.dwconv(x, H, W)
        #print(x.shape)
        
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = self.act1(xn)
        
        x = self.drop(xn)
        x_s = x.reshape(B, C, H * W).contiguous()
        x = x_s.transpose(1, 2)
        
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.dwconv1(x, H, W)
        x = self.drop(x)
        
        
        x += x1
        
        # Apply DropPath
        x = x + self.drop_path(x)
        
        return x
    
   
class UCMBlock1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        
        # Original UCMBlock initializations
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # Merged shiftmlp components
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.dwconv = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.dwconv1 = DWConv(mlp_hidden_dim)  # Assuming DWConv definition is available
        self.act = nn.GELU()
        self.act1 = nn.GELU()  # Assuming Activation is a placeholder for an actual activation like GELU
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        
        self.drop = nn.Dropout(drop)
        
        
        

        
        
        
       # self.fc3 = nn.Linear(mlp_hidden_dim, dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        
        # Weight initialization for merged components
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Norm and DropPath from original UCMBlock
        x = self.norm2(x)
       
        
        # Begin merged shiftmlp forward logic
        B, N, C = x.shape
        x1 = x.clone().detach()

       # x2= self.norm3(x2)
       # x1 = self.fc3(x1)
       # x2=  self.act(x2)
        


        
        x = self.fc1(x)
        #print(x.shape)
        x = self.dwconv(x, H, W)
        #print(x.shape)
        
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = self.act1(xn)
        
        x = self.drop(xn)
        x_s = x.reshape(B, C, H * W).contiguous()
        x = x_s.transpose(1, 2)
        
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.dwconv1(x, H, W)
        x = self.drop(x)
        
        
        x += x1
        
        # Apply DropPath
        x = x + self.drop_path(x)
        
        return x



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
       # self.norm = nn.LayerNorm(dim+1)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

       

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
       
        x = F.layer_norm(x, [H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                             # padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=stride,)
        self.norm = nn.LayerNorm(embed_dim)

        #self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



class MUCM_Net_2(nn.Module):
    

    ## Conv 3 + MLP 2 + shifted MLP w less parameters
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=256, patch_size=16, in_chans=3,  embed_dims=[ 8,16,24,32,48,64,3],
                 num_heads=[1], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        


        self.encoder1 = nn.Conv2d(embed_dims[-1], embed_dims[0], 3, stride=1, padding=1)  
     #   self.encoder2 = nn.Conv2d(6, 16, 3, stride=1, padding=1)  
      #  self.encoder3 = nn.Conv2d(16, 24, 3, stride=1, padding=1)


        #self.ebn1 = nn.BatchNorm2d(embed_dims[0])
        self.ebn1 = nn.GroupNorm(4,embed_dims[0])
     #   self.ebn2 = nn.BatchNorm2d(12)
       # self.ebn3 = nn.BatchNorm2d(18)
        
       # self.norm0 = norm_layer(embed_dims[0])
        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])

        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])
      #  self.dnorm7 = norm_layer(embed_dims[-1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block_0_1 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])  
        
        self.block0 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])        
        
        self.block1 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        
        self.block3 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[5], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock0 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock3 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock4 = nn.ModuleList([UCMBlock_2(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        

        
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size , patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[4])
        
        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[4],
                                              embed_dim=embed_dims[5])
        
      
        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 1, stride=1,padding=0)  
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 1, stride=1,padding=0)  
      #  self.decoder1_1 =   nn.Conv2d(48, 32, 1, stride=1, padding=0)  
        self.decoder2 =   nn.Conv2d(embed_dims[3], embed_dims[2], 1, stride=1, padding=0)  
     #   self.decoder2_1 =   nn.Conv2d(32, 24, 1, stride=1, padding=0)  
        self.decoder3 =   nn.Conv2d(embed_dims[2], embed_dims[1],  1, stride=1, padding=0) 
    #    self.decoder3_1 =   nn.Conv2d(24, 16, 1, stride=1, padding=0) 
        self.decoder4 =   nn.Conv2d(embed_dims[1], embed_dims[0], 1, stride=1, padding=0)
       # self.decoder4_1 =   nn.Conv2d(16, 6, 1, stride=1, padding=0)
        self.decoder5 =   nn.Conv2d(embed_dims[0], embed_dims[-1], 1, stride=1, padding=0)
 
      #  self.dbn0 = nn.BatchNorm2d(embed_dims[4])
      ##  self.dbn1 = nn.BatchNorm2d(embed_dims[3])
       # self.dbn2 = nn.BatchNorm2d(embed_dims[2])
       # self.dbn3 = nn.BatchNorm2d(embed_dims[1])
       # self.dbn4 = nn.BatchNorm2d(embed_dims[0])
        
        self.dbn0 = nn.GroupNorm(4,embed_dims[4])
        self.dbn1 = nn.GroupNorm(4,embed_dims[3])
        self.dbn2 = nn.GroupNorm(4,embed_dims[2])
        self.dbn3 = nn.GroupNorm(4,embed_dims[1])
        self.dbn4 = nn.GroupNorm(4,embed_dims[0])
    
        
      
        self.finalpre0 = nn.Conv2d(embed_dims[4], num_classes, kernel_size=1)
        self.finalpre1 = nn.Conv2d(embed_dims[3], num_classes, kernel_size=1)
        self.finalpre2 = nn.Conv2d(embed_dims[2], num_classes, kernel_size=1)
        self.finalpre3 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.finalpre4 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        
        self.final = nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)

       

    def forward(self, x,inference_mode=False):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        out = self.encoder1(x)

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(out),2,2))
        t1 = out
       
      #  out,H,W = self.patch_embed5(x)
      #  for i, blk in enumerate(self.block_0_2):
      #      out = blk(out, H, W)
      #  out = self.norm0(out)
      #  out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
      #  t1 = out
      
        ### Stage 2
       # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        #t2 = out
        out,H,W = self.patch_embed1(out)
        for i, blk in enumerate(self.block_0_1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out
        ### Stage 3
       

     #   out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
       # t3 = out
        out,H,W = self.patch_embed2(out)
        for i, blk in enumerate(self.block0):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out
        
        ### Bottleneck
        out ,H,W= self.patch_embed5(out)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
       # outtpre0 = F.interpolate(out, scale_factor=32, mode ='bilinear', align_corners=True)
       # outtpre0 =self.finalpre0(outtpre0)
        ### Stage 4
        out = F.relu(F.interpolate(self.dbn0(self.decoder0(out)),scale_factor=2,mode ='bilinear'))


        out = torch.add(out,t5)

        if not inference_mode:
            outtpre0 = F.interpolate(out, scale_factor=32, mode ='bilinear', align_corners=True)
            outtpre0 =self.finalpre0(outtpre0)
        #print('outtpre1',torch.sigmoid(outtpre1).size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock0):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=2,mode ='bilinear'))
   
        
        out = torch.add(out,t4)
        if not inference_mode:
            outtpre1 = F.interpolate(out, scale_factor=16, mode ='bilinear', align_corners=True)
            outtpre1 =self.finalpre1(outtpre1)
        #print('outtpre1',torch.sigmoid(outtpre1).size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=2,mode ='bilinear'))
      #  t41=self.decoder2_1(F.upsample(t4, scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
     #   out = torch.add(out,t41)
        if not inference_mode:
            outtpre2 = F.interpolate(out, scale_factor=8, mode ='bilinear', align_corners=True)
        
            outtpre2 =self.finalpre2(outtpre2)
        #print('outtpre2',outtpre2.size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
     #   t31=self.decoder3_1(F.upsample(t3, scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=2,mode ='bilinear'))
        out = torch.add(out,t2)
    #    out = torch.add(out,t31)
        #print(out.size())
        if not inference_mode:
            outtpre3 = F.interpolate(out, scale_factor=4, mode ='bilinear', align_corners=True)
        
            outtpre3 =self.finalpre3(outtpre3)
        #print('outtpre3',outtpre3.size())
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock3):
            out = blk(out, H, W)

        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()       
     #   print(out.size())
        
     #   t21=self.decoder4_1(F.upsample(t2, scale_factor=(2,2),mode ='bilinear'))
       
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=2,mode ='bilinear'))
        out = torch.add(out,t1)
    #    out = torch.add(out,t21)
        
        if not inference_mode:
            outtpre4 = F.interpolate(out, scale_factor=2, mode ='bilinear', align_corners=True)
        
            outtpre4 =self.finalpre4(outtpre4)
        #print('outtpre4',outtpre4.size())        
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock4):
            out = blk(out, H, W) 
        out = self.dnorm6(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=2,mode ='bilinear'))

        out =  self.final(out)
        if not inference_mode:
            return ( outtpre0,outtpre1, outtpre2, outtpre3, outtpre4), out
        else:
            return out
        
class MUCM_Net_8(nn.Module):
    

    ## Conv 3 + MLP 2 + shifted MLP w less parameters
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=256, patch_size=16, in_chans=3,  embed_dims=[ 8,16,24,32,48,64,3],
                 num_heads=[1], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        


        self.encoder1 = nn.Conv2d(embed_dims[-1], embed_dims[0], 3, stride=1, padding=1)  
     #   self.encoder2 = nn.Conv2d(6, 16, 3, stride=1, padding=1)  
      #  self.encoder3 = nn.Conv2d(16, 24, 3, stride=1, padding=1)


        #self.ebn1 = nn.BatchNorm2d(embed_dims[0])
        self.ebn1 = nn.GroupNorm(4,embed_dims[0])
     #   self.ebn2 = nn.BatchNorm2d(12)
       # self.ebn3 = nn.BatchNorm2d(18)
        
       # self.norm0 = norm_layer(embed_dims[0])
        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])

        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])
      #  self.dnorm7 = norm_layer(embed_dims[-1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block_0_1 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])  
        
        self.block0 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])        
        
        self.block1 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        
        self.block3 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[5], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock0 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock3 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock4 = nn.ModuleList([UCMBlock_8(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        

        
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size , patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[4])
        
        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[4],
                                              embed_dim=embed_dims[5])
        
      
        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 1, stride=1,padding=0)  
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 1, stride=1,padding=0)  
      #  self.decoder1_1 =   nn.Conv2d(48, 32, 1, stride=1, padding=0)  
        self.decoder2 =   nn.Conv2d(embed_dims[3], embed_dims[2], 1, stride=1, padding=0)  
     #   self.decoder2_1 =   nn.Conv2d(32, 24, 1, stride=1, padding=0)  
        self.decoder3 =   nn.Conv2d(embed_dims[2], embed_dims[1],  1, stride=1, padding=0) 
    #    self.decoder3_1 =   nn.Conv2d(24, 16, 1, stride=1, padding=0) 
        self.decoder4 =   nn.Conv2d(embed_dims[1], embed_dims[0], 1, stride=1, padding=0)
       # self.decoder4_1 =   nn.Conv2d(16, 6, 1, stride=1, padding=0)
        self.decoder5 =   nn.Conv2d(embed_dims[0], embed_dims[-1], 1, stride=1, padding=0)
 
      #  self.dbn0 = nn.BatchNorm2d(embed_dims[4])
      ##  self.dbn1 = nn.BatchNorm2d(embed_dims[3])
       # self.dbn2 = nn.BatchNorm2d(embed_dims[2])
       # self.dbn3 = nn.BatchNorm2d(embed_dims[1])
       # self.dbn4 = nn.BatchNorm2d(embed_dims[0])
        
        self.dbn0 = nn.GroupNorm(4,embed_dims[4])
        self.dbn1 = nn.GroupNorm(4,embed_dims[3])
        self.dbn2 = nn.GroupNorm(4,embed_dims[2])
        self.dbn3 = nn.GroupNorm(4,embed_dims[1])
        self.dbn4 = nn.GroupNorm(4,embed_dims[0])
    
        
      
        self.finalpre0 = nn.Conv2d(embed_dims[4], num_classes, kernel_size=1)
        self.finalpre1 = nn.Conv2d(embed_dims[3], num_classes, kernel_size=1)
        self.finalpre2 = nn.Conv2d(embed_dims[2], num_classes, kernel_size=1)
        self.finalpre3 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.finalpre4 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        
        self.final = nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)

       

    def forward(self, x,inference_mode=False):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        out = self.encoder1(x)

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(out),2,2))
        t1 = out
       
      #  out,H,W = self.patch_embed5(x)
      #  for i, blk in enumerate(self.block_0_2):
      #      out = blk(out, H, W)
      #  out = self.norm0(out)
      #  out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
      #  t1 = out
      
        ### Stage 2
       # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        #t2 = out
        out,H,W = self.patch_embed1(out)
        for i, blk in enumerate(self.block_0_1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out
        ### Stage 3
       

     #   out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
       # t3 = out
        out,H,W = self.patch_embed2(out)
        for i, blk in enumerate(self.block0):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out
        
        ### Bottleneck
        out ,H,W= self.patch_embed5(out)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
       # outtpre0 = F.interpolate(out, scale_factor=32, mode ='bilinear', align_corners=True)
       # outtpre0 =self.finalpre0(outtpre0)
        ### Stage 4
        out = F.relu(F.interpolate(self.dbn0(self.decoder0(out)),scale_factor=2,mode ='bilinear'))


        out = torch.add(out,t5)

        if not inference_mode:
            outtpre0 = F.interpolate(out, scale_factor=32, mode ='bilinear', align_corners=True)
            outtpre0 =self.finalpre0(outtpre0)
        #print('outtpre1',torch.sigmoid(outtpre1).size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock0):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=2,mode ='bilinear'))
   
        
        out = torch.add(out,t4)
        if not inference_mode:
            outtpre1 = F.interpolate(out, scale_factor=16, mode ='bilinear', align_corners=True)
            outtpre1 =self.finalpre1(outtpre1)
        #print('outtpre1',torch.sigmoid(outtpre1).size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=2,mode ='bilinear'))
      #  t41=self.decoder2_1(F.upsample(t4, scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
     #   out = torch.add(out,t41)
        if not inference_mode:
            outtpre2 = F.interpolate(out, scale_factor=8, mode ='bilinear', align_corners=True)
        
            outtpre2 =self.finalpre2(outtpre2)
        #print('outtpre2',outtpre2.size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
     #   t31=self.decoder3_1(F.upsample(t3, scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=2,mode ='bilinear'))
        out = torch.add(out,t2)
    #    out = torch.add(out,t31)
        #print(out.size())
        if not inference_mode:
            outtpre3 = F.interpolate(out, scale_factor=4, mode ='bilinear', align_corners=True)
        
            outtpre3 =self.finalpre3(outtpre3)
        #print('outtpre3',outtpre3.size())
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock3):
            out = blk(out, H, W)

        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()       
     #   print(out.size())
        
     #   t21=self.decoder4_1(F.upsample(t2, scale_factor=(2,2),mode ='bilinear'))
       
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=2,mode ='bilinear'))
        out = torch.add(out,t1)
    #    out = torch.add(out,t21)
        
        if not inference_mode:
            outtpre4 = F.interpolate(out, scale_factor=2, mode ='bilinear', align_corners=True)
        
            outtpre4 =self.finalpre4(outtpre4)
        #print('outtpre4',outtpre4.size())        
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock4):
            out = blk(out, H, W) 
        out = self.dnorm6(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=2,mode ='bilinear'))

        out =  self.final(out)
        if not inference_mode:
            return ( outtpre0,outtpre1, outtpre2, outtpre3, outtpre4), out
        else:
            return out        
class MUCM_Net_4(nn.Module):
    

    ## Conv 3 + MLP 2 + shifted MLP w less parameters
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=256, patch_size=16, in_chans=3,  embed_dims=[ 8,16,24,32,48,64,3],
                 num_heads=[1], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        


        self.encoder1 = nn.Conv2d(embed_dims[-1], embed_dims[0], 3, stride=1, padding=1)  
     #   self.encoder2 = nn.Conv2d(6, 16, 3, stride=1, padding=1)  
      #  self.encoder3 = nn.Conv2d(16, 24, 3, stride=1, padding=1)


        #self.ebn1 = nn.BatchNorm2d(embed_dims[0])
        self.ebn1 = nn.GroupNorm(4,embed_dims[0])
     #   self.ebn2 = nn.BatchNorm2d(12)
       # self.ebn3 = nn.BatchNorm2d(18)
        
       # self.norm0 = norm_layer(embed_dims[0])
        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])

        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])
      #  self.dnorm7 = norm_layer(embed_dims[-1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block_0_1 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])  
        
        self.block0 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])        
        
        self.block1 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        
        self.block3 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[5], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock0 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock3 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock4 = nn.ModuleList([UCMBlock_4(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        

        
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size , patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[4])
        
        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[4],
                                              embed_dim=embed_dims[5])
        
      
        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 1, stride=1,padding=0)  
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 1, stride=1,padding=0)  
      #  self.decoder1_1 =   nn.Conv2d(48, 32, 1, stride=1, padding=0)  
        self.decoder2 =   nn.Conv2d(embed_dims[3], embed_dims[2], 1, stride=1, padding=0)  
     #   self.decoder2_1 =   nn.Conv2d(32, 24, 1, stride=1, padding=0)  
        self.decoder3 =   nn.Conv2d(embed_dims[2], embed_dims[1],  1, stride=1, padding=0) 
    #    self.decoder3_1 =   nn.Conv2d(24, 16, 1, stride=1, padding=0) 
        self.decoder4 =   nn.Conv2d(embed_dims[1], embed_dims[0], 1, stride=1, padding=0)
       # self.decoder4_1 =   nn.Conv2d(16, 6, 1, stride=1, padding=0)
        self.decoder5 =   nn.Conv2d(embed_dims[0], embed_dims[-1], 1, stride=1, padding=0)
 
      #  self.dbn0 = nn.BatchNorm2d(embed_dims[4])
      ##  self.dbn1 = nn.BatchNorm2d(embed_dims[3])
       # self.dbn2 = nn.BatchNorm2d(embed_dims[2])
       # self.dbn3 = nn.BatchNorm2d(embed_dims[1])
       # self.dbn4 = nn.BatchNorm2d(embed_dims[0])
        
        self.dbn0 = nn.GroupNorm(4,embed_dims[4])
        self.dbn1 = nn.GroupNorm(4,embed_dims[3])
        self.dbn2 = nn.GroupNorm(4,embed_dims[2])
        self.dbn3 = nn.GroupNorm(4,embed_dims[1])
        self.dbn4 = nn.GroupNorm(4,embed_dims[0])
    
        
      
        self.finalpre0 = nn.Conv2d(embed_dims[4], num_classes, kernel_size=1)
        self.finalpre1 = nn.Conv2d(embed_dims[3], num_classes, kernel_size=1)
        self.finalpre2 = nn.Conv2d(embed_dims[2], num_classes, kernel_size=1)
        self.finalpre3 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.finalpre4 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        
        self.final = nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)

       

    def forward(self, x,inference_mode=False):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        out = self.encoder1(x)

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(out),2,2))
        t1 = out
       
      #  out,H,W = self.patch_embed5(x)
      #  for i, blk in enumerate(self.block_0_2):
      #      out = blk(out, H, W)
      #  out = self.norm0(out)
      #  out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
      #  t1 = out
      
        ### Stage 2
       # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        #t2 = out
        out,H,W = self.patch_embed1(out)
        for i, blk in enumerate(self.block_0_1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out
        ### Stage 3
       

     #   out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
       # t3 = out
        out,H,W = self.patch_embed2(out)
        for i, blk in enumerate(self.block0):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out
        
        ### Bottleneck
        out ,H,W= self.patch_embed5(out)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
       # outtpre0 = F.interpolate(out, scale_factor=32, mode ='bilinear', align_corners=True)
       # outtpre0 =self.finalpre0(outtpre0)
        ### Stage 4
        out = F.relu(F.interpolate(self.dbn0(self.decoder0(out)),scale_factor=2,mode ='bilinear'))


        out = torch.add(out,t5)

        if not inference_mode:
            outtpre0 = F.interpolate(out, scale_factor=32, mode ='bilinear', align_corners=True)
            outtpre0 =self.finalpre0(outtpre0)
        #print('outtpre1',torch.sigmoid(outtpre1).size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock0):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=2,mode ='bilinear'))
   
        
        out = torch.add(out,t4)
        if not inference_mode:
            outtpre1 = F.interpolate(out, scale_factor=16, mode ='bilinear', align_corners=True)
            outtpre1 =self.finalpre1(outtpre1)
        #print('outtpre1',torch.sigmoid(outtpre1).size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=2,mode ='bilinear'))
      #  t41=self.decoder2_1(F.upsample(t4, scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
     #   out = torch.add(out,t41)
        if not inference_mode:
            outtpre2 = F.interpolate(out, scale_factor=8, mode ='bilinear', align_corners=True)
        
            outtpre2 =self.finalpre2(outtpre2)
        #print('outtpre2',outtpre2.size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
     #   t31=self.decoder3_1(F.upsample(t3, scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=2,mode ='bilinear'))
        out = torch.add(out,t2)
    #    out = torch.add(out,t31)
        #print(out.size())
        if not inference_mode:
            outtpre3 = F.interpolate(out, scale_factor=4, mode ='bilinear', align_corners=True)
        
            outtpre3 =self.finalpre3(outtpre3)
        #print('outtpre3',outtpre3.size())
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock3):
            out = blk(out, H, W)

        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()       
     #   print(out.size())
        
     #   t21=self.decoder4_1(F.upsample(t2, scale_factor=(2,2),mode ='bilinear'))
       
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=2,mode ='bilinear'))
        out = torch.add(out,t1)
    #    out = torch.add(out,t21)
        
        if not inference_mode:
            outtpre4 = F.interpolate(out, scale_factor=2, mode ='bilinear', align_corners=True)
        
            outtpre4 =self.finalpre4(outtpre4)
        #print('outtpre4',outtpre4.size())        
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock4):
            out = blk(out, H, W) 
        out = self.dnorm6(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=2,mode ='bilinear'))

        out =  self.final(out)
        if not inference_mode:
            return ( outtpre0,outtpre1, outtpre2, outtpre3, outtpre4), out
        else:
            return out



class MUCM_Net(nn.Module):
    

    ## Conv 3 + MLP 2 + shifted MLP w less parameters
    
    def __init__(self,  num_classes, input_channel=3, deep_supervision=False,img_size=256, patch_size=16, in_chans=3,  embed_dims=[ 8,16,24,32,48,64,3],
                 num_heads=[1], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        


        self.encoder1 = nn.Conv2d(embed_dims[-1], embed_dims[0], 3, stride=1, padding=1)  
     #   self.encoder2 = nn.Conv2d(6, 16, 3, stride=1, padding=1)  
      #  self.encoder3 = nn.Conv2d(16, 24, 3, stride=1, padding=1)


        #self.ebn1 = nn.BatchNorm2d(embed_dims[0])
        self.ebn1 = nn.GroupNorm(4,embed_dims[0])
     #   self.ebn2 = nn.BatchNorm2d(12)
       # self.ebn3 = nn.BatchNorm2d(18)
        
       # self.norm0 = norm_layer(embed_dims[0])
        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])

        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])
      #  self.dnorm7 = norm_layer(embed_dims[-1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block_0_1 = nn.ModuleList([UCMBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])  
        
        self.block0 = nn.ModuleList([UCMBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])        
        
        self.block1 = nn.ModuleList([UCMBlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([UCMBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        
        self.block3 = nn.ModuleList([UCMBlock(
            dim=embed_dims[5], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock0 = nn.ModuleList([UCMBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([UCMBlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([UCMBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock3 = nn.ModuleList([UCMBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.dblock4 = nn.ModuleList([UCMBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        

        
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size , patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[4])
        
        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[4],
                                              embed_dim=embed_dims[5])
        
      
        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 1, stride=1,padding=0)  
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 1, stride=1,padding=0)  
      #  self.decoder1_1 =   nn.Conv2d(48, 32, 1, stride=1, padding=0)  
        self.decoder2 =   nn.Conv2d(embed_dims[3], embed_dims[2], 1, stride=1, padding=0)  
     #   self.decoder2_1 =   nn.Conv2d(32, 24, 1, stride=1, padding=0)  
        self.decoder3 =   nn.Conv2d(embed_dims[2], embed_dims[1],  1, stride=1, padding=0) 
    #    self.decoder3_1 =   nn.Conv2d(24, 16, 1, stride=1, padding=0) 
        self.decoder4 =   nn.Conv2d(embed_dims[1], embed_dims[0], 1, stride=1, padding=0)
       # self.decoder4_1 =   nn.Conv2d(16, 6, 1, stride=1, padding=0)
        self.decoder5 =   nn.Conv2d(embed_dims[0], embed_dims[-1], 1, stride=1, padding=0)
 
      #  self.dbn0 = nn.BatchNorm2d(embed_dims[4])
      ##  self.dbn1 = nn.BatchNorm2d(embed_dims[3])
       # self.dbn2 = nn.BatchNorm2d(embed_dims[2])
       # self.dbn3 = nn.BatchNorm2d(embed_dims[1])
       # self.dbn4 = nn.BatchNorm2d(embed_dims[0])
        
        self.dbn0 = nn.GroupNorm(4,embed_dims[4])
        self.dbn1 = nn.GroupNorm(4,embed_dims[3])
        self.dbn2 = nn.GroupNorm(4,embed_dims[2])
        self.dbn3 = nn.GroupNorm(4,embed_dims[1])
        self.dbn4 = nn.GroupNorm(4,embed_dims[0])
    
        
      
        self.finalpre0 = nn.Conv2d(embed_dims[4], num_classes, kernel_size=1)
        self.finalpre1 = nn.Conv2d(embed_dims[3], num_classes, kernel_size=1)
        self.finalpre2 = nn.Conv2d(embed_dims[2], num_classes, kernel_size=1)
        self.finalpre3 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.finalpre4 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        
        self.final = nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)

       

    def forward(self, x,inference_mode=True):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        out = self.encoder1(x)

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(out),2,2))
        t1 = out
       
      #  out,H,W = self.patch_embed5(x)
      #  for i, blk in enumerate(self.block_0_2):
      #      out = blk(out, H, W)
      #  out = self.norm0(out)
      #  out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
      #  t1 = out
      
        ### Stage 2
       # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        #t2 = out
        out,H,W = self.patch_embed1(out)
        for i, blk in enumerate(self.block_0_1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out
        ### Stage 3
       

     #   out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
       # t3 = out
        out,H,W = self.patch_embed2(out)
        for i, blk in enumerate(self.block0):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out
        
        ### Bottleneck
        out ,H,W= self.patch_embed5(out)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
       # outtpre0 = F.interpolate(out, scale_factor=32, mode ='bilinear', align_corners=True)
       # outtpre0 =self.finalpre0(outtpre0)
        ### Stage 4
        out = F.relu(F.interpolate(self.dbn0(self.decoder0(out)),scale_factor=2,mode ='bilinear'))


        out = torch.add(out,t5)

        if not inference_mode:
            outtpre0 = F.interpolate(out, scale_factor=32, mode ='bilinear', align_corners=True)
            outtpre0 =self.finalpre0(outtpre0)
        #print('outtpre1',torch.sigmoid(outtpre1).size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock0):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=2,mode ='bilinear'))
   
        
        out = torch.add(out,t4)
        if not inference_mode:
            outtpre1 = F.interpolate(out, scale_factor=16, mode ='bilinear', align_corners=True)
            outtpre1 =self.finalpre1(outtpre1)
        #print('outtpre1',torch.sigmoid(outtpre1).size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=2,mode ='bilinear'))
      #  t41=self.decoder2_1(F.upsample(t4, scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
     #   out = torch.add(out,t41)
        if not inference_mode:
            outtpre2 = F.interpolate(out, scale_factor=8, mode ='bilinear', align_corners=True)
        
            outtpre2 =self.finalpre2(outtpre2)
        #print('outtpre2',outtpre2.size())
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
     #   t31=self.decoder3_1(F.upsample(t3, scale_factor=(2,2),mode ='bilinear'))
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=2,mode ='bilinear'))
        out = torch.add(out,t2)
    #    out = torch.add(out,t31)
        #print(out.size())
        if not inference_mode:
            outtpre3 = F.interpolate(out, scale_factor=4, mode ='bilinear', align_corners=True)
        
            outtpre3 =self.finalpre3(outtpre3)
        #print('outtpre3',outtpre3.size())
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock3):
            out = blk(out, H, W)

        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()       
     #   print(out.size())
        
     #   t21=self.decoder4_1(F.upsample(t2, scale_factor=(2,2),mode ='bilinear'))
       
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=2,mode ='bilinear'))
        out = torch.add(out,t1)
    #    out = torch.add(out,t21)
        
        if not inference_mode:
            outtpre4 = F.interpolate(out, scale_factor=2, mode ='bilinear', align_corners=True)
        
            outtpre4 =self.finalpre4(outtpre4)
        #print('outtpre4',outtpre4.size())        
        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock4):
            out = blk(out, H, W) 
        out = self.dnorm6(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=2,mode ='bilinear'))

        out =  self.final(out)
        if not inference_mode:
            return ( outtpre0,outtpre1, outtpre2, outtpre3, outtpre4), out
        else:
            return out

def mucm_net(num_classes, input_channel=3):
    model = MUCM_Net(input_channel=input_channel, num_classes=num_classes)
    return model  