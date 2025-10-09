import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class DLA(nn.Module):
    def __init__(self, inp, oup, kernel_size = 3, stride=1, expand_ratio = 3, refine_mode='conv_exapnd'):
        super(DLA, self).__init__()
        """
            Distributed Local Attention used for refining the attention map.
        """

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.inp, self.oup = inp, oup
        self.high_dim_id = False
        self.refine_mode = refine_mode
        if refine_mode == 'conv':
            self.conv = Conv2dSamePadding(hidden_dim, hidden_dim, (kernel_size,kernel_size), stride, (1,1), groups=1, bias=False)
        elif refine_mode == 'conv_exapnd':
            if self.expand_ratio != 1:
                self.conv_exp = Conv2dSamePadding(inp, hidden_dim, 1, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(hidden_dim)   
            self.depth_sep_conv = Conv2dSamePadding(hidden_dim, hidden_dim, (kernel_size,kernel_size), stride, (1,1), groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)
            self.conv_pro = Conv2dSamePadding(hidden_dim, oup, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)
            self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        x= input
        if self.refine_mode == 'conv':
            return self.conv(x)
        else:
            if self.expand_ratio !=1:
                x = self.relu(self.bn1(self.conv_exp(x)))
            x = self.relu(self.bn2(self.depth_sep_conv(x)))
            x = self.bn3(self.conv_pro(x))
            if self.identity:
                return x + input
            else:
                return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mode="train"):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots/1.0 # T
        attn = self.attend(dots)
        #attentionheatmap_visual(attn, out_dir="./Visualization/ACDC/TransFuse")
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if mode=="train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn

class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., expansion_ratio = 3, apply_transform=True, transform_scale=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.apply_transform = apply_transform

        self.num_heads = heads
        self.scale = dim_head ** -0.5

        if apply_transform:
            self.reatten_matrix = nn.Conv2d(self.num_heads, self.num_heads, 1, 1)
            self.var_norm = nn.BatchNorm2d(self.num_heads)
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mode="train"):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if mode=="train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn

class Attention_Refiner(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.DLA = DLA(self.heads, self.heads, kernel_size=3, refine_mode='conv_exapnd', expand_ratio=3)
        self.adapt_bn = nn.BatchNorm2d(self.heads)

    def forward(self, x, mode="train"):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = math.sqrt(self.scale)*q
        k = math.sqrt(self.scale)*k
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn.softmax(dim=-1)
        attn = self.adapt_bn(self.DLA(attn))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if mode=="train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn


class Transformer_Vanilla(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    def infere(self, x):
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            x = ax + x
            x = ff(x) + x
            ftokens.append(x)
            attmaps.append(amap)
        return x, ftokens, attmaps


class Transformer_Reattention(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ReAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    def infere(self, x):
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            x = ax + x
            x = ff(x) + x
            ftokens.append(x)
            attmaps.append(amap)
        return x, ftokens, attmaps


class Transformer_Layerscale(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.init_values = 1e-4
        self.gamma_1 = nn.Parameter(self.init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(self.init_values * torch.ones((dim)),requires_grad=True)
    def forward(self, x):
        for attn, ff in self.layers:
            x = self.gamma_1*attn(x) + x
            x = self.gamma_2*ff(x) + x
        return x
    def infere(self, x):
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            x = self.gamma_1*ax + x
            x = self.gamma_2*ff(x) + x
            ftokens.append(x)
            attmaps.append(amap)
        return x, ftokens, attmaps

class Transformer_Refiner(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention_Refiner(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    def infere(self, x):
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            x = ax + x
            x = ff(x) + x
            ftokens.append(x)
            attmaps.append(amap)
        return x, ftokens, attmaps

