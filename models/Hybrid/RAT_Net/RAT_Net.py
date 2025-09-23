import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from .RASAB_t1 import RASAB as NONLocal_t1
from .RASAB_t3 import RASAB as NONLocal_t3
from .RASAB_t3 import RAC
import math

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.RACBLOCK = RAC(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h_x, w_x = x.shape
        x2 = self.RACBLOCK(x)
        _, _, H, W = x2.shape
        x2 = x2.flatten(2).transpose(1, 2)
        x2 = self.norm(x2)

        return x2, H, W, x, h_x, w_x

class regionreplacement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )

    def forward(self, x, H, W, x2, h_x, w_x):
        B, _, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        out = x2.clone()
        out[:, :, (h_x * 200 // 768):(h_x * 500 // 768), (w_x * 100 // 768):(w_x * 600 // 768)] = x
        out = torch.cat((x2, out), dim=1)
        out = self.conv_out(out)
        out = out.flatten(2).transpose(1, 2)

        return out


class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.fusion1 = regionreplacement(64)
        self.fusion2 = regionreplacement(128)
        self.fusion3 = regionreplacement(320)
        self.fusion4 = regionreplacement(512)


    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W, x2, h_x, w_x = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.fusion1(x, H, W, x2, h_x, w_x)
        x = self.norm1(x)
        x = x.reshape(B, h_x, w_x, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W, x2, h_x, w_x = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.fusion2(x, H, W, x2, h_x, w_x)
        x = self.norm2(x)
        x = x.reshape(B, h_x, w_x, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W, x2, h_x, w_x = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.fusion3(x, H, W, x2, h_x, w_x)
        x = self.norm3(x)
        x = x.reshape(B, h_x, w_x, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W, x2, h_x, w_x = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.fusion4(x, H, W, x2, h_x, w_x)
        x = self.norm4(x)
        x = x.reshape(B, h_x, w_x, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x

class RAT(nn.Module):
    def __init__(self, num_classes=1, input_channel=3, threshold= 't3'):
        super(RAT, self).__init__()
        self.encoder = mymit(input_channel=input_channel)
        if threshold == 't1':
            RASAB_RA = NONLocal_t1
        # if threshold == 't2':
        #     RASAB_RA = NONLocal_t2
        if threshold == 't3':
            RASAB_RA = NONLocal_t3
        # if threshold == 't4':
        #     RASAB_RA = NONLocal_t4
        # if threshold == 't5':
        #     RASAB_RA = NONLocal_t5
        self.AASA_1 = RASAB_RA(32)
        self.AASA_2 = RASAB_RA(64)
        self.AASA_3 = RASAB_RA(128)
        self.AASA_4 = RASAB_RA(320)
        self.AASA_5 = RASAB_RA(512)
        self.up1w = nn.Sequential(nn.Conv2d(512, 320, 3, padding=1),
                                  nn.BatchNorm2d(320),
                                  nn.ReLU(inplace=True),
                                  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                                  )
        self.up2w = nn.Sequential(nn.Conv2d(640, 128, 3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                                  )
        self.up3w = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                                  )
        self.up4w = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                                  )
        self.up5 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(inplace=True),
                                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                                 )
        self.doubleconv = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(16, 32, 3, padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(inplace=True)
                                        )
        self.finalout = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        layer_outputs = self.encoder(x)
        doubleout = self.doubleconv(x)
        x = self.up1w(self.AASA_5(layer_outputs[3]))
        x = torch.cat((x, self.AASA_4(layer_outputs[2])), dim=1)
        x = self.up2w(x)
        x = torch.cat((x, self.AASA_3(layer_outputs[1])), dim=1)
        x = self.up3w(x)
        x = torch.cat((x, self.AASA_2(layer_outputs[0])), dim=1)
        x = self.up4w(x)
        x = self.up5(x)
        x = torch.cat((x, self.AASA_1(doubleout)), dim=1)
        out = self.finalout(x)

        return out


class mymit(MixVisionTransformer):
    def __init__(self, input_channel):
        super(mymit, self).__init__(
                in_chans=input_channel,
                embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[1, 4, 2, 1],
                drop_rate=0.05, drop_path_rate=0.1)



def rat_net(input_channel=3, num_classes=1):
    return RAT(input_channel=3, num_classes=1)