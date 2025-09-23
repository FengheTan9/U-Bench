from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch
from timm.models.layers import to_2tuple, trunc_normal_,DropPath
from torchvision.ops.deform_conv import *
from torchvision.ops.ps_roi_pool import *
import torch.nn.functional as F
# from .nonlocal_block import NONLocalBlock2D
#from carafe import CARAFEPack
from torch.nn.modules.utils import _pair
# from .nattencuda import NEWNeighborhoodAttention
# from .nattencuda import NeighborhoodAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
#from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
#from .involution_cuda import involution
from natten import NeighborhoodAttention2D



class OverlapPatchEmbed(nn.Module):
    def __init__(self, patchsize, img_size, in_channels,embed_dim,stride,model='no nat'):#8,32,128
        super().__init__()
        self.model=model
        patch_size = _pair(patchsize)
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=embed_dim,
                                       kernel_size=patchsize,
                                       stride=stride,
                             padding = (patch_size[0] // 2, patch_size[1] // 2)
        )
        
    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.model=='nat':
            x=x.permute(0, 2, 3, 1)
        else:
            x = x.flatten(2).transpose(1, 2)#+self.position_embeddings
        return x

class Mlp(nn.Module):
    def __init__(self, in_channel, mlp_channel,out_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, out_channel)
        self.act_fn = nn.GELU()#nn.Hardswish(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x




class NoskipViTEncoder(nn.Module):
    def __init__(self, patchsize, img_size, in_channels,stride,kernel_size,head):
        super(NoskipViTEncoder, self).__init__()
        self.img_size=img_size
        self.patchembedding_l=OverlapPatchEmbed(patchsize, img_size, in_channels,in_channels,stride)
        self.patchembedding_s = OverlapPatchEmbed(patchsize, img_size, in_channels,in_channels,stride)
        self.norm_l1=nn.LayerNorm(in_channels)
        self.norm_s1 = nn.LayerNorm(in_channels)
        self.cross=NEWNeighborhoodAttention(in_channels,kernel_size,head,attn_drop=0.1,proj_drop=0.1)
        self.norm = nn.LayerNorm(in_channels)
        self.mlp = Mlp(in_channels, 2*in_channels,in_channels)

    def forward(self, xq, xkv):
        #xq_embedding = xq.permute(0, 2, 3, 1)
        #xkv_embedding=xkv.permute(0, 2, 3, 1)
        xq_embedding ,xkv_embedding=self.patchembedding_s(xq),self.patchembedding_l(xkv)
        xq, xkv = self.norm_s1(xq_embedding), self.norm_l1(xkv_embedding)
        att = self.cross(xq, xkv) + xkv_embedding
        x = self.mlp(self.norm(att)) + att
        x = x.permute(0, 3, 1, 2).contiguous()

        return x
class M3Skip(nn.Module):
    def __init__(self, in_channels=[12,24,48]):
        super(M3Skip, self).__init__()
        self.convl=nn.Sequential(
            nn.Conv2d(in_channels[0],in_channels[1],3,2,1),
        )
        self.convm=nn.Sequential(
            nn.Conv2d(in_channels[1],in_channels[1],3,1,1),
        )

        self.convs=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels[2], in_channels[1], 3, 1, 1),
            )
        self.fuse_conv=nn.Sequential(nn.Conv2d(3*in_channels[1],in_channels[1],3,1,1),
                                     nn.BatchNorm2d(in_channels[1]),
                                     nn.GELU()
                                     )
    def forward(self, xl,xm, xs):
        xl=self.convl(xl)
        xm=self.convm(xm)
        xs=self.convs(xs)
        x=torch.cat([xl,xm,xs],dim=1)
        x=self.fuse_conv(x)
        return x

class M2Skip(nn.Module):
    def __init__(self, in_channels=[12,24],model_type='bottom'):#大,小
        super(M2Skip, self).__init__()
        self.model_type=model_type
        if self.model_type=='bottom':
            self.convl=nn.Sequential(
                nn.Conv2d(in_channels[0],in_channels[1],3,2,1),
            )
            self.convs=nn.Sequential(
                nn.Conv2d(in_channels[1], in_channels[1], 3,1,1),
                )
            self.fuse_conv = nn.Sequential(nn.Conv2d(2 * in_channels[1], in_channels[1], 3,1,1),
                                           nn.BatchNorm2d(in_channels[1]),
                                           nn.GELU()
                                           )
        else:
            self.convl=nn.Sequential(
                nn.Conv2d(in_channels[0],in_channels[0],3,1,1),
            )
            self.convs=nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels[1], in_channels[0], 3, 1, 1),
                )
            self.fuse_conv = nn.Sequential(nn.Conv2d(2*in_channels[0], in_channels[0], 3,1,1),
                                           nn.BatchNorm2d(in_channels[0]),
                                           nn.GELU()
                                           )

    def forward(self, xl, xs):
        xl=self.convl(xl)
        xs=self.convs(xs)
        x=torch.cat([xl,xs],dim=1)
        x=self.fuse_conv(x)

        return x



class PatchEmbed(nn.Module):

    def __init__(self, patch_size=7,img_size=224,in_chans=3, out_channel=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        #embed_dim=patch_size[0]*patch_size[1]*in_chans

        self.proj = nn.Conv2d(in_chans, out_channel, kernel_size=patch_size, stride=patch_size,
                              )
        self.norm = nn.LayerNorm(out_channel)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, out_channel))
        self.proj_linear=nn.Linear(out_channel,out_channel)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(-1, -2)+self.position_embeddings
        #x=self.proj_linear(x)
        #x = self.dropout(self.norm(x))
        x=self.norm(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
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

        self.apply(self._init_weights)

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

class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        #self.q = nn.Linear(dim,  2, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q,k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6]):
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

        self.pool_ratios = pool_ratios

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, d_convs=None):#93312
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            pool = F.adaptive_avg_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
            pool = pool + l(pool)  # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1))

        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))

        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)

        return x
class GFT(nn.Module):
    def __init__(self, patchsize, img_size, in_channels,expand_ratios,out_channel,stride,num_heads):
        super(GFT, self).__init__()
        self.patchembedding=OverlapPatchEmbed(patchsize, img_size, in_channels,in_channels,stride)
        self.norm1=nn.LayerNorm(in_channels)
        self.attention=GlobalAttention(in_channels,num_heads)
        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = Mlp(in_channels, expand_ratios*in_channels,in_channels)
        self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channel,1),
                                )

    def forward(self, x):
        B,C,H,W = x.shape
        x_embedding=self.patchembedding(x)
        att = self.attention(self.norm1(x_embedding)) + x_embedding
        x = self.mlp(self.norm2(att)) + att
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x=self.conv(x)
        return x
class PoolTransformer(nn.Module):
    def __init__(self, patchsize, img_size, in_channels,out_channel,stride,num_heads,pool_ratios=[1, 2, 3, 6]):
        super(PoolTransformer, self).__init__()
        self.patchembedding=OverlapPatchEmbed(patchsize, img_size, in_channels,out_channel,stride)
        self.norm1=nn.LayerNorm(out_channel)
        self.attention=PoolingAttention(out_channel,num_heads,pool_ratios=pool_ratios)
        self.norm2 = nn.LayerNorm(out_channel)
        self.mlp = Mlp(out_channel, 2*out_channel,out_channel)
        self.norm3=nn.LayerNorm(out_channel)
        self.d_conv = nn.ModuleList(
            [nn.Conv2d(out_channel, out_channel,3,1,1, groups=out_channel) for tempin in pool_ratios])
        self.stride=stride
        self.hw=img_size//stride
        self.drop_path = DropPath(0.1)
    def forward(self, x):
        B,_,H,W=x.shape
        x_embedding=self.patchembedding(x)

        att = self.drop_path(self.attention(self.norm1(x_embedding),self.hw,self.hw,self.d_conv)) + x_embedding
        x = self.drop_path(self.mlp(self.norm2(att))) + att
        x=self.norm3(x)
        x = x.reshape(B, self.hw, self.hw, -1).permute(0, 3, 1, 2).contiguous()
        if self.stride>1:
            x=F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
        return x

class NAT_Global_Transformer(nn.Module):
    def __init__(self, patchsize, img_size, in_channels,out_channel,stride,kernel_size=[3,5],num_heads=8,pool_ratios=[1, 2, 3, 6],sr_ratio=1):
        super(NAT_Global_Transformer, self).__init__()
        self.stride=stride
        self.patch_hw=img_size//stride#img_size//patchsize##
        self.patchembedding1=  OverlapPatchEmbed(3, img_size, in_channels,out_channel,1)
        self.patchembedding3 = OverlapPatchEmbed(3, img_size, in_channels, out_channel, 1)
        self.patchembedding2 = OverlapPatchEmbed(patchsize, img_size, in_channels, out_channel,stride)
        self.norm1=nn.LayerNorm(out_channel)
        self.att0=NeighborhoodAttention(out_channel,kernel_size[0],num_heads)
        self.att1 = NeighborhoodAttention(out_channel, kernel_size[1], num_heads)
        self.hatt1 = NEWNeighborhoodAttention(out_channel, kernel_size[0], num_heads)
        self.hatt2 = NEWNeighborhoodAttention(out_channel, kernel_size[1], num_heads)
        self.att2 = Attention(out_channel,num_heads,sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(out_channel)
        self.norm1_0 = nn.LayerNorm(out_channel)
        self.norm1_1 = nn.LayerNorm(out_channel)
        self.norm3 = nn.LayerNorm(out_channel)
        self.mlp = Mlp(out_channel, 2*out_channel,out_channel)
        self.proj = nn.Linear(out_channel, out_channel*patchsize*patchsize)
        self.up_conv =nn.Sequential(  nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channel,kernel_size=3, padding=1),
            nn.Upsample(scale_factor=stride,mode='bilinear'))
        self.fuse=nn.Conv2d(2*out_channel,out_channel,1)
        self.drop_path = DropPath(0.1)


        self.d_conv = nn.ModuleList(
            [nn.Conv2d(out_channel, out_channel,3,1,1, groups=out_channel) for tempin in pool_ratios])

    def forward(self,xq,xkv ):
        # B,C,H,W = xkv.shape
        #
        # x_embedding2=self.patchembedding2(xq)
        # att2=self.drop_path(self.att2(self.norm2(x_embedding2),self.patch_hw,self.patch_hw))+ x_embedding2
        #
        # att2 = att2.reshape(B, self.patch_hw, self.patch_hw, -1).permute(0, 3, 1, 2).contiguous()
        # if self.stride>1:
        #     att2=self.up_conv(att2)

        x_embedding1 = self.patchembedding1(xq)
        x_embedding3 = self.patchembedding3(xkv)
        xq=self.norm1(x_embedding1)
        xkv = self.norm1_0(x_embedding1)

        #att=self.fuse(torch.cat([att1,att2],dim=1)).permute(0, 2, 3, 1)
        att1 = self.drop_path(self.hatt1(xq,xkv))
        #att2 = self.drop_path(self.hatt2(xq, xkv))
        att=att1+x_embedding3


        #att1=att1.permute(0, 3, 1, 2)


        #att2=self.proj(att2).reshape(B, H, W, -1)
        #att=(att1+att2).permute(0, 2, 3, 1)
        x = self.drop_path(self.mlp(self.norm3(att))) + att
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SkipAttention(nn.Module):
    def __init__(self, patchsize, img_size, in_channels,out_channel,stride,sr_ratio):
        super(SkipAttention, self).__init__()
        self.patchembedding=OverlapPatchEmbed(patchsize, img_size, in_channels,out_channel,stride)
        self.norm1=nn.LayerNorm(out_channel)
        self.attention=Attention(out_channel,8,sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(out_channel)
        self.mlp = Mlp(out_channel, 2*out_channel,out_channel)
        self.norm3=nn.LayerNorm(out_channel)
    def forward(self, x):
        B=x.shape[0]
        x_embedding,H,W=self.patchembedding(x)
        att = self.attention(self.norm1(x_embedding),H,W) + x_embedding
        x = self.mlp(self.norm2(att)) + att
        x=self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

class PyramidPool(nn.Module):
    def __init__(self,filters=[16,32, 64, 128, 256]):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(128)
        self.pool2 = nn.AdaptiveAvgPool2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d(32)
        self.pool4 = nn.AdaptiveAvgPool2d(16)
        # self.fuse1 = nn.Sequential(nn.Conv2d(filters[0]+filters[1],filters[0]+filters[1], 3,1,1),
        #                            #nn.BatchNorm2d(filters[0]+filters[1]),
        #                            #nn.ReLU(inplace=True)
        #                            )
        #
        # self.fuse2 = nn.Sequential(nn.Conv2d(filters[0]+filters[1]+filters[2],filters[0]+filters[1]+filters[2], 3,1,1),
        #                            #nn.BatchNorm2d(filters[0]+filters[1]+filters[2]),
        #                            #nn.ReLU(inplace=True)
        #                            )
        #
        # self.fuse3 = nn.Sequential(nn.Conv2d(sum(filters)-filters[4],sum(filters)-filters[4], 3,1,1),
        #                            #nn.BatchNorm2d(sum(filters)-filters[4]),
        #                            #nn.ReLU(inplace=True)
        #                            )
        #
        # self.fuse4 = nn.Sequential(nn.Conv2d(sum(filters),sum(filters), 3,1,1),
        #                            #nn.BatchNorm2d(sum(filters)),
        #                            #nn.ReLU(inplace=True)
        #                            )

    def forward(self, x1,x2,x3,x4,x5):


        # x1 = self.pool1(x1)
        # fuse1 = self.fuse1(torch.cat([x1,x2], dim=1))
        # x2 = self.pool2(fuse1)
        # fuse2 = self.fuse2(torch.cat([x2,x3], dim=1))
        # x3 = self.pool3(fuse2)
        # fuse3 = self.fuse3(torch.cat([x3,x4], dim=1))
        # x4 = self.pool4(fuse3)
        # fuse4 = self.fuse4(torch.cat([x4, x5], dim=1))
        # return fuse4


        B, C, H, W = x5.shape
        x=torch.cat([nn.functional.adaptive_avg_pool2d(i, (H, W)) for i in [x1,x2,x3,x4] ], dim=1)
        #x = torch.cat([nn.functional.adaptive_max_pool2d(i, (H, W)) for i in [x1, x2, x3, x4]], dim=1)
        return torch.cat([x,x5],dim=1)





class NeighborhoodTransformer(nn.Module):
    def __init__(self, patchsize, img_size, in_channels,out_channel,stride,kernel_size=[3,5],num_heads=8):
        super(NeighborhoodTransformer, self).__init__()
        self.patchembedding=  OverlapPatchEmbed(patchsize, img_size, in_channels,out_channel,stride,'nat')
        self.norm1=nn.LayerNorm(out_channel)
        self.att1 = NeighborhoodAttention2D(dim=out_channel,num_heads=num_heads,kernel_size=3)
        # self.att2 = NeighborhoodAttention2D(dim=out_channel,num_heads=num_heads,kernel_size=7,)
        self.norm2 = nn.LayerNorm(out_channel)
        self.mlp = Mlp(out_channel, 2*out_channel,out_channel)

    def forward(self, x):
        x_embedding= self.patchembedding(x)
        x= self.norm1(x_embedding)
        att = self.att1(x)+x_embedding#+self.att2(x)
        x = self.mlp(self.norm2(att)) + att
        x = x.permute(0, 3, 1, 2).contiguous()

        return x



class ReparamConv(nn.Module):

    def __init__(self, in_channels,expand_channels,out_channels, large_kernel_size,kernel_size,stride=1, groups=1,deploy=False):
        super(ReparamConv, self).__init__()
        self.large_kernel_size=large_kernel_size
        self.kernel_size=kernel_size
        self.in_channels=in_channels
        self.expand_channels=expand_channels
        self.stride=stride
        self.deploy = deploy
        self.se = SE(expand_channels)

        self.expand_conv =nn.Sequential(nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(expand_channels),
                                        nn.Hardswish(inplace=True))


        if self.deploy:
            self.fuse_conv = nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                        kernel_size=large_kernel_size, stride=stride,
                                        padding=large_kernel_size//2,  groups=expand_channels, bias=True,
                                        )
        else:
            self.large_conv = nn.Sequential(OrderedDict(
                [('conv',nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                         kernel_size=large_kernel_size, stride=stride,
                                         padding=large_kernel_size//2,  groups=expand_channels,bias=False)),
                 ('bn', nn.BatchNorm2d(expand_channels))
                 ]))
            self.square_conv = nn.Sequential(OrderedDict([
                ('conv',nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=kernel_size//2,  groups=expand_channels,bias=False)),
                ('bn', nn.BatchNorm2d(expand_channels))
            ]))
            self.ver_conv = nn.Sequential(OrderedDict([
                 ('conv',nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                      kernel_size=(kernel_size, 1),stride=stride,
                                      padding=[kernel_size // 2,0 ],  groups=expand_channels, bias=False,)),
                ('bn', nn.BatchNorm2d(expand_channels))
            ]))

            self.hor_conv = nn.Sequential(OrderedDict([
                 ('conv',nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                      kernel_size=(1, kernel_size),stride=stride,
                                      padding=[0,kernel_size // 2 ],  groups=expand_channels, bias=False,)),
                ('bn', nn.BatchNorm2d(expand_channels))
            ]))

        self.active = nn.GELU()

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
           #nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x1 = self.expand_conv(x)
        if self.deploy:
            out = self.fuse_conv(x1)
        else:

            out = self.large_conv(x1)
            out += self.square_conv(x1)
            out += self.ver_conv(x1)
            out += self.hor_conv(x1)

        x1 = self.se(self.active(out))
        x1 = self.pointwise_conv(x1)
        out = x1 + self.shortcut(x)
        return out

    def fuse_bn(self,conv, bn, mobel='no avg'):
        if mobel == 'avg':
            kernel = conv
        else:
            kernel = conv.weight
        gamma = bn.weight
        std = (bn.running_var + bn.eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * gamma / std

    def axial_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
        square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel
        return square_kernel


    def get_equivalent_kernel_bias(self):
        large_k, large_b = self.fuse_bn(self.large_conv.conv, self.large_conv.bn)
        square_k, square_b = self.fuse_bn(self.square_conv.conv, self.square_conv.bn)
        hor_k, hor_b = self.fuse_bn(self.hor_conv.conv, self.hor_conv.bn)
        ver_k, ver_b = self.fuse_bn(self.ver_conv.conv, self.ver_conv.bn)
        square_k=self.axial_to_square_kernel(square_k, hor_k)
        square_k=self.axial_to_square_kernel(square_k, ver_k)



        #singel_k, singel_b = self.fuse_bn(self.singel_conv.conv, self.singel_conv.bn)

        #avg_weight = get_avg_weight(self.in_channels, self.kernel_size,self.expand_channels).cuda()
        #avg_k,avg_b=fuse_bn(avg_weight, self.avg_cov.bn,mobel='avg')


        large_b =large_b+square_b+hor_b+ver_b
        # #   add to the central part
        large_k += nn.functional.pad(square_k, [(self.large_kernel_size - self.kernel_size) // 2] * 4)
        # # large_k += nn.functional.pad(avg_k, [(self.large_kernel_size - self.kernel_size) // 2] * 4)
        return large_k, large_b

    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.fuse_conv = nn.Conv2d(in_channels=self.expand_channels, out_channels=self.expand_channels,
                                    kernel_size=self.large_kernel_size, stride=self.stride,
                                    padding=self.large_kernel_size//2, dilation=1,
                                    groups=self.expand_channels, bias=True,
                                    )
        self.fuse_conv.weight.data = deploy_k
        self.fuse_conv.bias.data = deploy_b
        self.__delattr__('square_conv')
        #self.__delattr__('avg_cov')
        self.__delattr__('hor_conv')
        self.__delattr__('ver_conv')



class MobileBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_channels, expand_channels, out_channels,large_kernel_size,kernel_size,):
        super(MobileBlock, self).__init__()
        self.se = SE(expand_channels)

        self.expand_conv =nn.Sequential(nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(expand_channels),
                                        nn.Hardswish(inplace=True))

        self.depthwise_conv_l  = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels,kernel_size=5, stride=1,padding=2,dilation=1, groups=expand_channels, bias=False),
            #DepthWiseConv2dImplicitGEMM(expand_channels,13,False),
            nn.BatchNorm2d(expand_channels),
           # nn.Hardswish(inplace=True)
        )
        self.depthwise_conv_r  = nn.Sequential(
            #nn.Conv2d(expand_channels, expand_channels,kernel_size=3, stride=1,padding=2,dilation=2, groups=expand_channels, bias=False),
            nn.Conv2d(expand_channels, expand_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                      groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
         #   nn.Hardswish(inplace=True)
        )
        self.reparamconv=ReparamConv(expand_channels,5,3,1,1)
        self.active=nn.Hardswish(inplace=True)



        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels))

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x1 = self.expand_conv(x)
        #x_reparam=self.reparamconv(x1)

        x_l = self.depthwise_conv_l(x1)
        x_r = self.depthwise_conv_r(x1)



        x1 = self.se(self.active(x_l+x_r))
        x1 = self.pointwise_conv(x1)
        out = x1 + self.shortcut(x)
        return out
class SegMLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegHead(nn.Module):
    def __init__(self,in_channels=[16,32,64,128]):
        super(SegHead, self).__init__()
        self.linear1 = SegMLP(input_dim=in_channels[0], embed_dim=in_channels[3])
        self.linear2 = SegMLP(input_dim=in_channels[1], embed_dim=in_channels[3])
        self.linear3 = SegMLP(input_dim=in_channels[2], embed_dim=in_channels[3])
        self.linear4 = SegMLP(input_dim=in_channels[3], embed_dim=in_channels[3])

        self.fuse =nn.Sequential(nn.Conv2d(4*in_channels[3],in_channels[0],1),
                                 nn.BatchNorm2d(in_channels[0])
                                  )
        self.dropout=nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(in_channels[0], 2, kernel_size=1)

    def forward(self, x1,x2,x3,x4):
        n, _, _, _ = x1.shape
        x1=self.linear1(x1).permute(0,2,1).reshape(n, -1, x1.shape[2], x1.shape[3])
        x2 = self.linear2(x2).permute(0, 2, 1).reshape(n, -1, x2.shape[2], x2.shape[3])
        x3 = self.linear3(x3).permute(0, 2, 1).reshape(n, -1, x3.shape[2], x3.shape[3])
        x4 = self.linear4(x4).permute(0, 2, 1).reshape(n, -1, x4.shape[2], x4.shape[3])

        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=x1.size()[2:], mode='bilinear', align_corners=False)
        x=torch.cat([x1,x2,x3,x4], dim=1)
        x  = self.fuse(x)
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x

class SoftPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(SoftPool,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

    def soft_pool2d(self, x, kernel_size=2, stride=None, force_inplace=False):
        kernel_size = _pair(kernel_size)
        if stride is None:
            stride = kernel_size
        else:
            stride = _pair(stride)
        _, c, h, w = x.size()
        e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
        return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),#图像大小减半
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=2,dilation=2),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),#图像大小减半，保证跳跃连接大小一致
            nn.BatchNorm2d(output_dim),
        )
        self.mult_scal = SPBlock(output_dim, output_dim)
        self.pam=PAM_Module(output_dim)
        self.cam=CAM_Module(output_dim)
        self.eca=ECA(output_dim,3)

    def forward(self, x):
        x1=self.conv_block(x)
        x2=self.mult_scal(x1)
        #x2=self.mult_scal(x)
        #x3=self.pam(x1)
        #x3=self.cam(x2)
        x3=self.conv_skip(x)
        #x4=self.eca(x1+x2)
        return x2+x3

class DepthwiseConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,
            kernel_size=kernel_size,stride=stride,padding=padding,groups=in_ch)

        self.pointwise_conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,
            kernel_size=1,stride=1,padding=0,groups=1)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out

class DeformConv_V2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1):
        super(DeformConv_V2, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels,2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,stride=stride,
                                     padding=padding,dilation=dilation
                                     )

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,stride=stride,
                                        padding=padding,dilation=dilation
                                        )

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.decov2d=DeformConv2d(in_channels,out_channels,
                                  kernel_size=kernel_size,stride=stride,
                                  padding=padding,dilation=dilation)


    def forward(self, x):

        offset = self.offset_conv(x)
        modulator = torch.sigmoid(self.modulator_conv(x))
        x=self.decov2d(x,offset,modulator)
        return x

class DeformRoIpoolV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformRoIpoolV2, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels,2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,padding=padding,
                                     )

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,padding=padding,
                                        )

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)


        self.decov2d=DeformConv2d(in_channels,out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,padding=padding)
    def forward(self, x):

        offset = self.offset_conv(x)
        modulator = torch.sigmoid(self.modulator_conv(x))
        x=self.decov2d(x,offset,modulator)

        return x

class DeformConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=[1,1],dilation=[1,1]):
        super(DeformConv, self).__init__()

        self.double_conv_l = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3,stride=stride,padding=padding[0],dilation=dilation[0]),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(inplace=True),
            DeformConv_V2(output_dim, output_dim, kernel_size=3, stride=stride, padding=padding[0],dilation=dilation[0]),

        )
        self.double_conv_r = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3,stride=stride,padding=padding[1],dilation=dilation[1]),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(inplace=True),
            DeformConv_V2(output_dim, output_dim, kernel_size=3, stride=stride, padding=padding[1],dilation=dilation[1]),

        )
        self.combine_cov=nn.Sequential(
            nn.Conv2d(2*output_dim,output_dim,1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1),#图像大小减半，保证跳跃连接大小一致
            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):
        x1= self.double_conv_l(x)
        x2=self.double_conv_r(x)
        x3=self.combine_cov(torch.cat([x1, x2], dim=1))
        x4=self.conv_skip(x)

        return x3+x4

class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_conv =nn.MaxPool2d(2)


    def forward(self, x):
        return self.maxpool_conv(x)


class ECA(nn.Module):

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)#b,c,1,1

        # Two different branches of ECA module///b,c,1->b,1,c->b,c,1->b,c,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)#乘法
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class SE(nn.Module):
    def __init__(self,input_channels,reduction=4):
        super(SE,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, input_channels//reduction, 1)
        self.fc2 = nn.Conv2d(input_channels//reduction, input_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.scale_activation = nn.Hardsigmoid(inplace=True)
        self._init_weights()

    def forward(self, input):
        x=self.avgpool(input)
        x=self.fc1(x)
        x=self.activation(x)
        x=self.fc2(x)
        x=self.scale_activation(x)
        return x*input

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SPBlock(nn.Module):#Strip Pooling
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)

        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x3 = self.relu(x1 + x2)
        x3 = self.conv3(x3).sigmoid()
        return x*x3


class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

class connectionfuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(connectionfuse, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.Hardswish(True),
                                )

    def forward(self, x1,x2):
        x=torch.cat([x1,x2],dim=1)
        x= self.conv(x)
        return x

class My_ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[1, 6, 12, 18]):
        super(My_ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block4 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[3], dilation=rate[3]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),#输出1*1的特征图
                                             nn.Conv2d(in_dims, out_dims, 1, stride=1),#1*1的卷积
                                             nn.BatchNorm2d(out_dims),
                                             nn.ReLU(inplace=True))
        self.output = nn.Sequential(nn.Conv2d((len(rate)+1) * out_dims, out_dims, 1),
                                    nn.BatchNorm2d(out_dims),
                                    nn.ReLU(inplace=True)
                                    )
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        x4 = self.aspp_block4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # 自适应全局池化需要上采样
        out = torch.cat([x1, x2, x3,x4,x5], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Up(nn.Module):
    def __init__(self, input_decoder, output_dim):
        super(Up, self).__init__()
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(input_decoder, output_dim, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        x= self.conv_decoder(x)
        return x

class Carafe_Up(nn.Module):
    def __init__(self, input_decoder, output_dim,compressed_channels=64,scale_factor=2):
        super(Carafe_Up, self).__init__()
        self.carafe_up = nn.Sequential(nn.BatchNorm2d(input_decoder),
                                       nn.ReLU(inplace=True),

                                       CARAFEPack(input_decoder,scale_factor=scale_factor,compressed_channels=compressed_channels),
                                       nn.Conv2d(input_decoder, output_dim, 1),
                                       )
    def forward(self, x):
        x= self.carafe_up(x)
        return x



class MyAttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(MyAttentionBlock, self).__init__()

        # self.conv_encoder = nn.Sequential(
        #     nn.Conv2d(input_encoder, output_dim, 3, padding=1),
        #     nn.BatchNorm2d(output_dim),
        #     nn.ReLU()
        # )

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(input_decoder, output_dim, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

        #self.nonlocal_attention=NONLocalBlock2D(output_dim)

        # self.conv_attn = nn.Sequential(
        #     nn.Conv2d(output_dim, 1, 1),
        #     nn.BatchNorm2d(output_dim),
        #     nn.ReLU(),
        # )

    def forward(self, x1, x2):
        x =x1 + self.conv_decoder(x2)
        out=self.nonlocal_attention(x)
        return out


class PPM(nn.Module):#PSP-Net
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)