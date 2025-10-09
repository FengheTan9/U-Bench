from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm

import numpy as np
import torch.nn.functional as F

import math
from timm.layers import trunc_normal_tf_
from timm.models import named_apply
from contextlib import redirect_stderr
 


def get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError(
            "padding value should not be negative, please change the kernel size and/or stride."
        )
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding( kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError(
            "out_padding value should not be negative, please change the kernel size and/or stride."
        )
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_conv_layer(spatial_dims, in_channels, out_channels,
    kernel_size = 3,
    stride = 1,
    act = Act.PRELU,
    norm = Norm.INSTANCE,
    dropout = None,
    bias = False,
    conv_only = True,
    is_transposed = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

# =================================================================================================

class SepConvBN(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, rate=1, depth_activation=False, epsilon=1e-3):
        super(SepConvBN, self).__init__()

        # Calculate padding
        # if stride == 1:
        #     self.padding = kernel_size // 2
        # else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        self.padding = (kernel_size_effective - 1) // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=rate,
            groups=in_channels,
            bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels, eps=epsilon)

        self.pointwise = nn.Conv2d(
            in_channels,
            filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.pointwise_bn = nn.BatchNorm2d(filters, eps=epsilon)
        self.depth_activation = depth_activation
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        if not self.depth_activation:
            x = F.relu(x, inplace=True)

        x = self.depthwise(x)
        x = self.depthwise_bn(x)

        if self.depth_activation:
            x = F.relu(x, inplace=True)

        x = self.pointwise(x)
        x = self.pointwise_bn(x)

        if self.depth_activation:
            x = F.relu(x, inplace=True)

        return x


class UpRb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        norm_name = 'batch'
        act_name = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        rb_block = partial(UnetResBlock, spatial_dims=2, norm_name=norm_name, act_name=act_name, dropout=0)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            rb_block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1),
        )
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        return self.up(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer(activation, inplace=True)
        )
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        return self.up(x)
    
class UpTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, activation='relu'):
        super().__init__()
        self.up = get_conv_layer(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=0,
            bias=False,
            conv_only=True,
            is_transposed=True,
        )
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        return self.up(x)



# class DownRb(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
#         super().__init__()
#         self.down = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
#             nn.BatchNorm2d(out_channels),
#             act_layer(activation, inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#     def forward(self, x):
#         return self.down(x)


from .unet import UnetResBlock   
class DownRb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super().__init__()

        norm_name = 'batch'
        act_name = (activation, {"inplace": True, "negative_slope": 0.01})
        rb_block = partial(UnetResBlock, spatial_dims=2, norm_name=norm_name, act_name=act_name, dropout=0)

        self.down = nn.Sequential(
            rb_block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        return self.down(x)
    




def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

# Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
	        nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

