import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath
from .blocks import SepConvBN
from .nlb import Nonlocal


def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    """Build normalization layer."""
    assert norm_type in ['BN', 'GN', 'LN2d', 'SyncBN']
    if norm_type == 'GN':
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == 'LN2d':
        return LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == 'SyncBN':
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm2d(embed_dims, eps=1e-5)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"] 
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

    
class SRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pwc = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.dwc = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(1)
    def forward(self, x):
        x_max = x.max(1, keepdim=True)[0] # b, 1, h, w
        x_mean = x.mean(1, keepdim=True) # b, 1, h, w
        x_std = x.std(1, keepdim=True) # b, 1, h, w
        u = torch.cat([x_max, x_mean, x_std], dim=1) # b, 3, h, w
        f = self.act(self.pwc(u)+self.dwc(u))
        f = self.bn(f)
        g = torch.sigmoid(f)
        return x * g.expand_as(x)


class Mlp(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.srm = SRM()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.srm(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """
    def __init__(self,
                 embed_dims,
                 channel_split=[1, 3, 4, 2],
                 rates=[6, 12, 18],
                 flag_useAllChannels=False,
                ):
        super(MultiOrderDWConv, self).__init__()
        
        channel_split=[5, 5, 5, 1]

        self.useAllChannels = flag_useAllChannels
        if flag_useAllChannels:
            channel_indices = [(0, embed_dims)] * len(channel_split)
        else:
            split_ratio = [i / sum(channel_split) for i in channel_split]
            channel_indices = [(0, int(split_ratio[0] * embed_dims))]
            for cr in split_ratio[1:]:
                nci = int(cr * embed_dims)
                assert nci > 0, "Ops. Channel split ratio is not correct"
                channel_indices.append((channel_indices[-1][1], channel_indices[-1][1] + nci))
        self.channel_indices = channel_indices
        
        assert len(rates)+1 == len(channel_split) == 4

        self.dlps = nn.ModuleList()
        for rate, cids in zip(rates, channel_indices):
            kernel_size_effective = 3 + (3 - 1) * (rate - 1)
            padding = (kernel_size_effective - 1) // 2
            self.dlps.append(
                SepConvBN(
                    in_channels=cids[1]-cids[0],
                    filters=cids[1]-cids[0],
                    kernel_size=3,
                    stride=1,
                    rate=rate,
                    depth_activation=True,
                    epsilon=1e-5
                )
            )
            
        ipd = channel_indices[-1][1] - channel_indices[-1][0]
        # image pooling
        self.dlps.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(ipd, ipd, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ipd, eps=1e-5),
            nn.LeakyReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=7)
        ))

        self.embed_dims = embed_dims
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        dls_res = []
        for dlp, csps in zip(self.dlps, self.channel_indices):
            y = dlp(x[:, csps[0]:csps[1], ...])
            if y.shape[2] != x.shape[2] or y.shape[3] != x.shape[3]:
                y = F.interpolate(y, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            dls_res.append(y)
        
        if self.useAllChannels:
            x = torch.sum(dls_res)
        else:
            x = torch.cat(dls_res, dim=1)

        x = self.PW_conv(x)
        return x


class CCU(nn.Module):
    def __init__(self, channel, hidden_scale=3):
        super().__init__()
        self.fc1 = nn.Conv1d(channel, hidden_scale*channel, kernel_size=3, groups=channel, bias=False, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(hidden_scale*channel, channel, kernel_size=1, groups=channel, bias=False, padding=0)
        self.bn = nn.BatchNorm1d(channel)
    def forward(self, x):
        b, c, h, w = x.shape
        x_max = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)[0]
        x_mean = torch.mean(x, dim=(2, 3))
        x_std = torch.std(x, dim=(2, 3), unbiased=False)

        u = torch.stack([x_max, x_mean, x_std], dim=-1)
        # style integration
        z = self.fc2(self.act(self.fc1(u))).view(b, c)
        if z.shape[0] > 1:
            z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)

    
class MCA(nn.Module):
    """Spatial Block with Multi-scale Contextual Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self, embed_dims, attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU', rates=[2,3,4]):
        super().__init__()
        self.attn_force_fp32 = True # whether to force the attention to use fp32
        self.embed_dims = embed_dims
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(embed_dims=embed_dims,
            rates=rates, channel_split=attn_channel_split)
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_gate = build_act_layer(attn_act_type)

        # denoising module (NLB)
        self.denoising_module = Nonlocal(embed_dims)

        self.ccu = CCU(embed_dims) # style-based recalibration

    def forward(self, x):
        shortcut = x.clone()
        x = self.ccu(x)
        g = self.gate(x)
        v = self.value(x)
        x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        x = x + shortcut
        x = self.denoising_module(x)
        return x


class CFAModule(nn.Module):
    """A block of CFAM.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 mca_rates=[6, 12, 18],
                 writer=None
                ):
        super().__init__()
        self.writer = writer
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_type, embed_dims)

        # spatial attention
        self.mca = MCA(embed_dims, attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type, rates=mca_rates,
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_type, embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = Mlp(embed_dims, mlp_hidden_dim, 3, act_type, drop_rate)

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.mca(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x
