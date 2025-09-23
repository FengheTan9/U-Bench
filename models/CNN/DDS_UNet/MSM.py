import os
import pdb
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap2 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()
        y1 = self.gap1(x).view(b, c)
        y2 = self.gap1(x).view(b, c)
        y = y1+y2
        y = self.fc(y).view(b, c, 1, 1)
        return y * x

class MSM(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(MSM, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth//2, 1, 1)
        self.se = SE_Block(in_channel)
        self.atrous_block3 = nn.Conv2d(in_channel // 4, depth // 4, 3, 1, padding=1)
        self.atrous_block6 = nn.Conv2d(in_channel // 4, depth // 4, 5, 1, padding=2)
        self.atrous_block12 = nn.Conv2d(in_channel // 4, depth // 4, 7, 1, padding=3)
        self.atrous_block18 = nn.Conv2d(in_channel // 4, depth // 4, 9, 1, padding=4)
        self.conv_1x1_output = nn.Conv2d(depth * 3, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        B, C, H, W = x.shape

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = nn.functional.interpolate(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        x1, x2, x3, x4 = torch.chunk(self.se(x), 4, dim=1)

        x1 = self.atrous_block3(x1)
        x2 = self.atrous_block6(x2)
        x3 = self.atrous_block12(x3)
        x4 = self.atrous_block18(x4)

        out1 = x1 * x2
        out2 = x1 * x3
        out3 = x1 * x4
        out4 = x2 * x3
        out5 = x2 * x4
        out6 = x3 * x4

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, out1, out2, out3, out4, out5, out6], dim=1))
        net = net + x
        return net