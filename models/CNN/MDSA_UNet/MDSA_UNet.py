import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from ..builder import BACKBONES
from .bra_legacy import BiLevelRoutingAttention
from ._common import Attention, AttentionLePE


class Context_GLU(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv3x3 = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        ax = self.act(self.norm1(self.dwconv3x3(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * x))
        out = self.fc2(ax)
        return out


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x


class BiTransformer(nn.Module):
    def __init__(self, depth=2, embed_dim=64,
                 nhead=2, qk_scale=None,
                 n_win=7,
                 kv_downsample_mode='identity',
                 topk=1,
                 dp_rate=0.,
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dim=None,
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 before_attn_dwconv=3,
                 mlp_ratio=4,
                 param_attention='qkvo',
                 mlp_dwconv=False):
        super(BiTransformer, self).__init__()

        self.block = nn.ModuleList([
            Block(dim=embed_dim, drop_path=dp_rate,
                  layer_scale_init_value=layer_scale_init_value,
                  topk=topk,
                  num_heads=nhead,
                  n_win=n_win,
                  qk_dim=qk_dim,
                  qk_scale=qk_scale,
                  kv_downsample_mode=kv_downsample_mode,
                  param_attention=param_attention,
                  param_routing=param_routing,
                  diff_routing=diff_routing,
                  soft_routing=soft_routing,
                  mlp_ratio=mlp_ratio,
                  mlp_dwconv=mlp_dwconv,
                  side_dwconv=side_dwconv,
                  before_attn_dwconv=before_attn_dwconv,
                  pre_norm=pre_norm) for j in range(depth)]
        )

    def forward(self, x):
        for blk in self.block:
            x = blk(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
        #                          DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
        #                          nn.GELU(),
        #                          nn.Linear(int(mlp_ratio * dim), dim)
        #                          )
        self.mlp = Context_GLU(dim, int(mlp_ratio * dim))

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class EDM(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=2,):
        super(EDM, self).__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding=kernel_size // 2),
            nn.BatchNorm2d(out_dim),
            # SeparableConv2d(out_dim, out_dim, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.aux_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 8, kernel_size, 1, padding=kernel_size // 2),
            nn.BatchNorm2d(out_dim // 8),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.Conv2d(out_dim // 8, out_dim, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _x = self.aux_conv(x)
        x = self.patch_embed(x)
        x = x + _x
        return x


class Encoder(nn.Module):
    def __init__(self, dims=[3, 64, 128, 320, 512],
                 patch_kernel_sizes=[7, 3, 3, 3],
                 patch_strides=[4, 2, 2, 2],
                 depths=[2, 2, 2, 2],
                 topks=[1, 4, 16, -2],
                 n_win=7,
                 base_dim=32,
                 mlp_ratios=[2, 2, 2, 2],
                 blocks=4):
        super(Encoder, self).__init__()

        self.blocks = blocks
        num_heads = [dim // base_dim for dim in dims[1:]]

        self.edms = nn.ModuleList([EDM(dims[i], dims[i+1], kernel_size=patch_kernel_sizes[i],
                                                      stride=patch_strides[i])
                                           for i in range(blocks)])

        self.layers = nn.ModuleList([BiTransformer(depth=depths[i], embed_dim=dims[1:][i], nhead=num_heads[i],
                                                   n_win=n_win, topk=topks[i], dp_rate=0.1, mlp_ratio=mlp_ratios[i])
                                     for i in range(blocks)])

    def forward(self, x):

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        outs = []

        for i in range(self.blocks):
            x = self.edms[i](x)
            x = self.layers[i](x)
            outs.append(x)
        return outs


class EUF(nn.Module):
    def __init__(self, x_dim, skip_dim, out_dim):
        super(EUF, self).__init__()

        self.EUF = nn.Sequential(
            nn.Conv2d(x_dim + skip_dim, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            # SeparableConv2d(out_dim, out_dim, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.aux_conv = nn.Sequential(
            nn.Conv2d(x_dim + skip_dim, out_dim // 8, 3, 1, padding=1),
            nn.BatchNorm2d(out_dim // 8),
            nn.Conv2d(out_dim // 8, out_dim, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip, size):

        if x.shape[2] != size[0]:
            _x = F.interpolate(x, size=size, mode='bilinear')
        else:
            _x = x

        if skip.shape[2] != size[0]:
            _skip = F.interpolate(skip, size=size, mode='bilinear')
        else:
            _skip = skip

        x = self.EUF(torch.cat([_x, _skip], dim=1))
        x = x + self.aux_conv(torch.cat([_x, _skip], dim=1))
        return x


class Decoder(nn.Module):
    def __init__(self, dims=[512, 320, 128, 64],
                 depths=[2, 2, 2, 2],
                 topks=[-2, 16, 4, 1],
                 n_win=7,
                 base_dim=32,
                 mlp_ratios=[2, 2, 2, 2],
                 blocks=3,):
        super(Decoder, self).__init__()

        self.blocks = blocks
        num_heads = [dim // base_dim for dim in dims]

        self.eufs = nn.ModuleList([EUF(x_dim=dims[i], skip_dim=dims[i+1], out_dim=dims[i+1])
                                     for i in range(blocks)])
        self.layers = nn.ModuleList([BiTransformer(depth=depths[i+1], embed_dim=dims[i+1], nhead=num_heads[i+1],
                                                   n_win=n_win, topk=topks[i+1], dp_rate=0.1, mlp_ratio=mlp_ratios[i+1])
                                     for i in range(blocks)])

    def forward(self, xs):

        x = xs[-1]
        skip = xs[:-1][::-1]

        outs = [x]
        for i in range(self.blocks):
            _, _, h, w = skip[i].shape
            x = self.layers[i](self.eufs[i](x, skip[i], (h, w)))
            outs.append(x)
        return outs


# @BACKBONES.register_module()
class MDSA_UNet(nn.Module):
    def __init__(self, input_channel=3,num_classes=1,
                 dims=[3, 128, 320, 512,],
                 patch_kernel_sizes=[7, 3, 3],
                 patch_strides=[4, 2, 2],
                 depths=[1, 1, 1, 1, 1],
                 topks=[1, 4, 16],
                 n_win=8,
                 base_dim=32,
                 mlp_ratios=[2, 2, 2],
                 blocks=3,
                 ):
        dims[0]=input_channel
        super(MDSA_UNet, self).__init__()
        self.encoder = Encoder(dims, patch_kernel_sizes, patch_strides,
                               depths, topks, n_win, base_dim, mlp_ratios, blocks)

        self.decoder = Decoder(dims[1:][::-1], depths[2:], topks[::-1], n_win, base_dim,
                               mlp_ratios[::-1], blocks-1)

        self.cls = nn.Conv2d(dims[1], num_classes, 1)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        xs = self.encoder(x)
        outs = self.decoder(xs)

        p = self.cls(outs[-1])
        outs = F.interpolate(p, scale_factor=4, mode='bilinear')

        return outs


def mdsa_unet(input_channel=3,num_classes=1):
    return MDSA_UNet(input_channel=input_channel,num_classes=num_classes)



if __name__ == '__main__':

    inputs = torch.ones(1, 3, 224, 224)
    model = MDSAUNet()
    model.eval()

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")
    print(flop_count_table(flops, max_depth=2))

    from thop import profile
    flops, params = profile(model, inputs=(torch.ones(1, 3, 224, 224), ))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
