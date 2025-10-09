import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import clever_format, profile
from torchsummary import summary
# from network.CMUNeXt_new import CMUNeXtBlock_MK_resiual2,conv_1x1_bn
def autopad(k, p=None, d=1):  
    '''
    k: kernel
    p: padding
    d: dilation
    '''
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] # actual kernel-size
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, paddin g, groups, dilation, activation)."""
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

    
# Lightweight Cascade Multi-Receptive Fields Module
class CMRF(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        
        self.N         = N
        self.c         = int(c2 * e / self.N)
        self.add       = shortcut and c1 == c2
        
        self.pwconv1   = Conv(c1, c2//self.N, 1, 1)
        self.pwconv2   = Conv(c2//2, c2, 1, 1)
        self.m         = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N-1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x
        x          = self.pwconv1(x)

        x          = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0]       = x[0] +  x[1] 
        x.pop(1)
        
        y          = torch.cat(x, dim=1) 
        y          = self.pwconv2(y)
        return x_residual + y if self.add else y



class CMRFclone(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        
        self.N         = N
        self.c         = int(c2 * e / self.N)
        self.add       = shortcut and c1 == c2
        
        self.pwconv1   = Conv(c1, c2//self.N, 1, 1)
        self.pwconv2   = Conv(c2//2, c2, 1, 1)
        self.m         = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N-1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x.clone()
        x          = self.pwconv1(x)

        x          = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0]       = x[0] +  x[1] 
        x.pop(1)
        
        y          = torch.cat(x, dim=1) 
        y          = self.pwconv2(y)
        return x_residual + y if self.add else y


class CMRFnew(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        
        self.N         = N
        self.c         = int(c1 / self.N)
        self.add       = shortcut and c1 == c2
        
        self.pwconv1   = Conv(c1, c1//self.N, 1, 1)
        self.pwconv2   = Conv(c1*2, c2, 1, 1)
        self.m         = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N-1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x.clone()
        x          = self.pwconv1(x)

        x          = [x]
        x.extend(m(x[-1]) for m in self.m)
        x[-1]       = x[0] +  x[-1] 
        # x.pop(0)
        x.append(x_residual)
        y          = torch.cat(x, dim=1)
        # y          = torch.cat(x_residu
        y          = self.pwconv2(y)
        return x_residual + y if self.add else y



'''
U-shape/U-like Model
'''
# Encoder in TinyU-Net
class UNetEncoder(nn.Module):
    def __init__(self, input_channel, out_channels):
        super(UNetEncoder, self).__init__()
        self.cmrf       = CMRF(input_channel, out_channels)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.cmrf(x)
        return self.downsample(x), x
 # Decoder in TinyU-Net
class UNetDecoder(nn.Module):
    def __init__(self, input_channel, out_channels):
        super(UNetDecoder, self).__init__()
        self.cmrf      = CMRF(input_channel, out_channels)
        self.upsample  = F.interpolate
        
    def forward(self, x, skip_connection):
        x = self.upsample(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.cmrf(x)
        return x
  
# TinyU-Net

class Tinyunet(nn.Module):
    """TinyU-Net with args(input_channel, num_classes)."""
    '''
    input_channel: The number of input channels
    num_classes: The number of segmentation classes
    '''
    def __init__(self, input_channel=3, num_classes=2):
        super(Tinyunet, self).__init__()
        in_filters      = [192, 384, 768, 1024]
        out_filters     = [64, 128, 256, 512]
        
        self.encoder1   = UNetEncoder(input_channel, 64)
        self.encoder2   = UNetEncoder(64, 128)
        self.encoder3   = UNetEncoder(128, 256)
        self.encoder4   = UNetEncoder(256, 512)

        self.decoder4   = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3   = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2   = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1   = UNetDecoder(in_filters[0], out_filters[0])
        self.final_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x        = self.decoder4(x, skip4)
        x        = self.decoder3(x, skip3)
        x        = self.decoder2(x, skip2)
        x        = self.decoder1(x, skip1)
        x        = self.final_conv(x)
        return x



