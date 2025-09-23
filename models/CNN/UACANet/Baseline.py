import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim.losses import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s

class Baseline(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=256, output_stride=16, pretrained=True):
        super(Baseline, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained, output_stride=output_stride)

        self.context2 = simple_context(512, channels)
        self.context3 = simple_context(1024, channels)
        self.context4 = simple_context(2048, channels)

        self.decoder = simple_decoder(channels)

        self.attention2 = simple_attention(512, 64, 2, 3)
        self.attention3 = simple_attention(1024, 64, 2, 3)
        self.attention4 = simple_attention(2048, 256, 3, 5)

        self.loss_fn = bce_iou_loss

    def forward(self, sample):
        x = sample['image']
        if 'gt' in sample.keys():
            y = sample['gt']
        else:
            y = None
            
        base_size = x.shape[-2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        
        x2_context = self.context2(x2)
        x3_context = self.context3(x3)
        x4_context = self.context4(x4)

        _, a5 = self.decoder(x4_context, x3_context, x2_context)
        out5 = F.interpolate(a5, size=base_size, mode='bilinear', align_corners=False)

        _, a4 = self.attention4(x4, a5)
        out4 = F.interpolate(a4, size=base_size, mode='bilinear', align_corners=False)

        _, a3 = self.attention3(x3, a4)
        out3 = F.interpolate(a3, size=base_size, mode='bilinear', align_corners=False)

        _, a2 = self.attention2(x2, a3)
        out2 = F.interpolate(a2, size=base_size, mode='bilinear', align_corners=False)


        if y is not None:
            loss5 = self.loss_fn(out5, y)
            loss4 = self.loss_fn(out4, y)
            loss3 = self.loss_fn(out3, y)
            loss2 = self.loss_fn(out2, y)
            loss = loss2 + loss3 + loss4 + loss5
            debug = [out5, out4, out3]
        else:
            loss = 0
            debug = []

        return {'pred': out2, 'loss': loss, 'debug': debug}