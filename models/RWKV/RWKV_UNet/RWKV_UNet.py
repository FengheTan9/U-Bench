import os
import math
from torch.utils.cpp_extension import load
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import DropPath, create_act_layer
import numpy as np
import torchvision
from typing import Callable, Dict, Optional, Type
from einops import rearrange, reduce
from timm.layers.activations import *
from timm.layers import DropPath, trunc_normal_
from .module.basic_modules import get_norm, get_act, ConvNormAct, LayerScale2D
import math
from torch.utils.cpp_extension import load
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import DropPath, create_act_layer
import numpy as np
import torchvision
from typing import Callable, Dict, Optional, Type
from .ccm.ccm import CCMix
import timm.layers.weight_init as weight_init
from torch.hub import load_state_dict_from_url
inplace = True


wkv_op1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda', 'wkv_op.cpp')
wkv_cuda1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda', 'wkv_cuda.cu')
print(wkv_op1)
print(wkv_cuda1)
T_MAX1 = 1024
wkv_cuda1 = load(name="wkv", sources=[wkv_op1, wkv_cuda1],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX1}'])


def num_groups(group_size: Optional[int], channels: int):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class SE(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self,
            in_chs,
            rd_ratio = 0.25,
            rd_channels = None,
            act_layer = nn.ReLU,
            gate_layer = nn.Sigmoid,
            force_act_layer = None,
            rd_round_fn = None,
    ):
        super(SE, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX1
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda1.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX1
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda1.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


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


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1/4, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        attn_sz = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self):
        self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        # Use xk, xv, xr to produce k, v, r
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        sr, k, v = self.jit_func(x, patch_resolution)
        x = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=False, has_skip=False, exp_ratio=1.0, norm_layer='bn_2d',
                 dw_ks=3, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.):
        super().__init__()
        self.has_skip =has_skip
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='bn_2d', act_layer='relu', inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation, groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj(x)
        #x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.upsample(x)
        return x

class iR_RWKV(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', dw_ks=3, stride=1, dilation=1, se_ratio=0.0,
                 attn_s=True, drop_path=0., drop=0.,img_size=224, channel_gamma=1/4, shift_pixel=1):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        if attn_s==True:
                self.att = VRWKV_SpatialMix(dim_mid, channel_gamma, shift_pixel)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.attn_s=attn_s
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation, groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        if self.attn_s:
            B, hidden, H, W = x.size()
            patch_resolution = (H,  W)
            x = x.view(B, hidden, -1)  # (B, hidden, H*W) = (B, C, N)
            x = x.permute(0, 2, 1)
            x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
            B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidde
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            x = x.permute(0, 2, 1)
            x = x.contiguous().view(B, hidden, h, w)
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.proj(x)
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class RWKV_UNet_encoder(nn.Module):
    def __init__(self, dim_in=3, num_classes=1000, img_size=224,
                 depths=[2, 4, 4, 2], stem_dim=16, embed_dims=[64, 128, 256, 512], exp_ratios=[2., 2., 4., 4.],
                 norm_layers=['bn_2d', 'bn_2d', 'bn_2d', 'bn_2d'], act_layers=['relu', 'relu', 'relu', 'relu'],
                 dw_kss=[3, 3, 1, 1], se_ratios=[0.0, 0.0, 0.0, 0.0],attn_ss=[False, False, True, True],  drop=0., drop_path=0., channel_gamma=1/4, shift_pixel=1):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes > 0
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stage0 = nn.ModuleList([
            iR_RWKV(  # ds
                dim_in, stem_dim, norm_in=False, has_skip=False, exp_ratio=1,
                norm_layer=norm_layers[0], act_layer=act_layers[0], dw_ks=dw_kss[0],
                stride=1, dilation=1, se_ratio=1, attn_s=False,
                drop_path=0., drop=0.,img_size=img_size, shift_pixel=shift_pixel
            )
        ])
        img_size=img_size//2
        emb_dim_pre = stem_dim
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride, has_skip, attn_s, exp_ratio = 2, False, False, exp_ratios[i] * 2
                    img_size=img_size//2
                else:
                    stride, has_skip, attn_s, exp_ratio = 1, True, attn_ss[i], exp_ratios[i]
                layers.append(iR_RWKV(
                    emb_dim_pre, embed_dims[i], norm_in=True, has_skip=has_skip, exp_ratio=exp_ratio,
                    norm_layer=norm_layers[i], act_layer=act_layers[i],  dw_ks=dw_kss[i],
                    stride=stride, dilation=1, se_ratio=se_ratios[i],attn_s=attn_s,
                    drop_path=dpr[j],drop=drop,img_size=img_size, shift_pixel=shift_pixel))
                emb_dim_pre = embed_dims[i]
            self.__setattr__(f'stage{i + 1}', nn.ModuleList(layers))
        self.pre_dim = embed_dims[-1]
        self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
        self.head = nn.Linear(self.pre_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm,
                            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'alpha', 'gamma', 'beta'}

    @torch.jit.ignore
    def no_ft_keywords(self):
        # return {'head.weight', 'head.bias'}
        return {}

    @torch.jit.ignore
    def ft_head_keywords(self):
        return {'head.weight', 'head.bias'}, self.num_classes

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.pre_dim, num_classes) if num_classes > 0 else nn.Identity()

    def check_bn(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.modules.batchnorm._NormBase):
                m.running_mean = torch.nan_to_num(m.running_mean, nan=0, posinf=1, neginf=-1)
                m.running_var = torch.nan_to_num(m.running_var, nan=0, posinf=1, neginf=-1)

    def forward_features(self, x):
        for blk in self.stage0:
            x = blk(x)
        for blk in self.stage1:
            x = blk(x)
        for blk in self.stage2:
            x = blk(x)
        for blk in self.stage3:
            x = blk(x)
        for blk in self.stage4:
            x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = reduce(x, 'b c h w -> b c', 'mean').contiguous()
        x = self.head(x)
        return {'out': x, 'out_kd': x}


def RWKV_UNet_encoder_T(pretrained=False, **kwargs):
    model = RWKV_UNet_encoder(
        # dim_in=3, num_classes=1000, img_size=224,
        depths=[2, 2, 4, 2], stem_dim=24, embed_dims=[32, 48, 96, 160], exp_ratios=[2., 2.5, 3.0, 3.5],norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'], dw_kss=[5, 5, 5, 5], attn_ss=[False, False, True, True],drop=0., drop_path=0.05,**kwargs)
    return model
    

def RWKV_UNet_encoder_S(pretrained=False, **kwargs):
    model = RWKV_UNet_encoder(
        # dim_in=3, num_classes=1000, img_size=224,
        depths=[3, 3, 6, 3], stem_dim=24, embed_dims=[32, 64, 128, 192], exp_ratios=[2., 2.5, 3.0, 4.0],norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'], dw_kss=[5, 5, 5, 5], attn_ss=[False, False, True, True],drop=0., drop_path=0.05,**kwargs)
    return model


def RWKV_UNet_encoder_B(pretrained=False, **kwargs):
    model = RWKV_UNet_encoder(
        # dim_in=3, num_classes=1000, img_size=224,
        depths=[3, 3, 6, 3], stem_dim=24, embed_dims=[48, 72, 144, 240], exp_ratios=[2., 2.5, 4.0, 4.0],norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'], dw_kss=[5, 5, 5, 5], attn_ss=[False, False, True, True],drop=0., drop_path=0.05,**kwargs)
    return model



class RWKV_UNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=1, img_size=256, pretrained_path='net_B.pth'):
        super(RWKV_UNet, self).__init__()
        self.encoder = RWKV_UNet_encoder_B(img_size=img_size)
        if pretrained_path:
            pretrained_dict = load_state_dict_from_url("[URL]", progress=True)
            self.encoder.load_state_dict(pretrained_dict, strict=False)
        
        self.embed_dims = [48, 72, 144, 240]
        self.ccm = CCMix([self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]], self.embed_dims[0], img_size//2)
        # Decoder layers using iRWKV and 1x1 Conv with bilinear upsampling
        self.decoder1 = UpBlock(self.embed_dims[3], self.embed_dims[2], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)             
        self.decoder2 =  UpBlock(self.embed_dims[2]*2, self.embed_dims[1], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.decoder3 =  UpBlock(self.embed_dims[1]*2, self.embed_dims[0], norm_in=False, has_skip=False, exp_ratio=1.0,dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.decoder4 =  UpBlock(self.embed_dims[0]*2, 24, norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        for blk in self.encoder.stage0:
            x = blk(x)
        enc0 = x
        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x
        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x
        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x
        for blk in self.encoder.stage4:
            x = blk(x)
        enc3,enc2,enc1=self.ccm([enc3,enc2,enc1])
        # Decoder path with concatenated skip connections
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        # Final output
        out = self.final_conv(dec0)
        return out

class RWKV_UNet_S(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, img_size=224, pretrained_path='net_S.pth'):
        super(RWKV_UNet_S, self).__init__()
        self.encoder = RWKV_UNet_encoder_S(img_size=img_size)
        if pretrained_path:
            model_dict = self.encoder.state_dict()
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            self.encoder.load_state_dict(pretrained_dict, strict=False)
        self.embed_dims = [32, 64, 128, 192]
        self.ccm = CCMix([self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]], self.embed_dims[0], img_size//2)
        # Decoder layers using iRWKV and 1x1 Conv with bilinear upsampling
        self.decoder1 = UpBlock(self.embed_dims[3], self.embed_dims[2], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)             
        self.decoder2 =  UpBlock(self.embed_dims[2]*2, self.embed_dims[1], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.decoder3 =  UpBlock(self.embed_dims[1]*2, self.embed_dims[0], norm_in=False, has_skip=False, exp_ratio=1.0,dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.decoder4 =  UpBlock(self.embed_dims[0]*2, 24, norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        input=x
        x = x.repeat(1, 3, 1, 1)
        for blk in self.encoder.stage0:
            x = blk(x)
        enc0 = x
        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x
        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x
        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x
        for blk in self.encoder.stage4:
            x = blk(x)
        enc3,enc2,enc1=self.ccm([enc3,enc2,enc1])
        # Decoder path with concatenated skip connections
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        # Final output
        out = self.final_conv(dec0)
        return out

class RWKV_UNet_T(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, img_size=224, pretrained_path='net_T.pth'):
        super(RWKV_UNet_T, self).__init__()
        self.encoder = RWKV_UNet_encoder_T(img_size=img_size)
        if pretrained_path:
            model_dict = self.encoder.state_dict()
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            self.encoder.load_state_dict(pretrained_dict, strict=False)
        
        self.embed_dims = [32, 48, 96, 160]
        self.ccm = CCMix([self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]], self.embed_dims[0], img_size//2)
        # Decoder layers using iRWKV and 1x1 Conv with bilinear upsampling
        self.decoder1 = UpBlock(self.embed_dims[3], self.embed_dims[2], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)             
        self.decoder2 =  UpBlock(self.embed_dims[2]*2, self.embed_dims[1], norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.decoder3 =  UpBlock(self.embed_dims[1]*2, self.embed_dims[0], norm_in=False, has_skip=False, exp_ratio=1.0,dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.decoder4 =  UpBlock(self.embed_dims[0]*2, 24, norm_in=False, has_skip=False, exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0,drop_path=0., drop=0.)   
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        input=x
        x = x.repeat(1, 3, 1, 1)
        for blk in self.encoder.stage0:
            x = blk(x)
        enc0 = x
        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x
        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x
        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x
        for blk in self.encoder.stage4:
            x = blk(x)
        enc3,enc2,enc1=self.ccm([enc3,enc2,enc1])
        # Decoder path with concatenated skip connections
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        # Final output
        out = self.final_conv(dec0)
        return out

def rwkv_unet(input_channel=3,num_classes=1):
    return RWKV_UNet(input_channel=input_channel,num_classes=num_classes)