import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class depthwise_separable_conv(nn.Module):
    # [URL]
    def __init__(self, nin, nout, kernels_per_layer=2, stride=1, padding=1, bias=True):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernels_per_layer, stride=stride,
                                   padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def channel_shuffle(x, groups):
    # [URL]
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See [URL]
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class GSConv(nn.Module):
    # GSConv [URL]
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

# class GSConvBottleNeck(nn.Module):
#     # Slim-neck by GSConv
#     def __init__(self, c1, c2, k=1, s=1, p=None, bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(GSConvBottleNeck, self).__init__()
#         self.GSconv0 = GSConv(c1, c2//2, k, s, p)
#         self.GSconv1 = GSConv(c2//2, c2, 3, 1, 1, bias=bias)

#     def forward(self, x):
#         x_res = self.GSconv0(x)
#         x_res =  self.GSconv1(x_res)

#         return x+x_res

# class VoVGSConv(nn.Module):
#     # Slim-neck by GSConv
#     def __init__(self, c1, c2, k=1, s=1, p=None, bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(VoVGSConv, self).__init__()
#         self.conv0 = nn.Conv2d(c1, c1//2, 3, 1, 1, bias=bias)
#         self.GSconv0 = GSConv(c1//2, c1//2, 3, 1, 1, bias=bias)
#         self.GSconv1 = GSConv(c1//2, c1//2, 3, 1, 1, bias=bias)
#         self.conv2 = nn.Conv2d((c1//2)*2, c2, k, s, p, bias=bias)

#     def forward(self, x):
#         x = self.conv0(x)
#         x_1 = self.GSconv0(x)
#         x_1 =  self.GSconv1(x_1)
#         x = torch.cat((x,x_1), 1)
#         x = self.conv2(x)
#         return x