import os
import pdb
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .LDR import LDR
from .MSM import MSM
from .SMM import SMM1, SMM2, SMM3, SMM4

# Conv_Block
class Convblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Convblock, self).__init__()
        self.encoder = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.ebn = nn.BatchNorm2d(output_channels)
    def forward(self, x):
        out = F.relu(F.max_pool2d(self.ebn(self.encoder(x)), 2, 2))
        return out

# Up_Block
class Upblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Upblock, self).__init__()
        self.decoder = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.dbn = nn.BatchNorm2d(output_channels)
    def forward(self, x):
        out = F.relu(F.interpolate(self.dbn(self.decoder(x)), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        return out

# DDS-UNet
class DDS_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channel=3, deep_supervision=False, img_size=384, **kwargs):
        super().__init__()
        # self.filters = [8, 16, 32, 64, 128]        #T
        # self.filters = [16, 32, 128, 160, 256]     #S
        # self.filters = [32, 64, 128, 256, 512]     #B
        self.filters = [64, 128, 256, 512, 1024]     #L

        self.sizes = [img_size//2, img_size//4, img_size//8, img_size//16, img_size//32]

        self.Convstage1 = LDR(input_channel, self.filters[0])
        self.Convstage2 = LDR(self.filters[0], self.filters[1])
        self.Convstage3 = LDR(self.filters[1], self.filters[2])
        self.Convstage4 = LDR(self.filters[2], self.filters[3])
        self.Convstage5 = LDR(self.filters[3], self.filters[4])

        self.Upstage1 = Upblock(self.filters[4], self.filters[3])
        self.Upstage2 = Upblock(self.filters[3], self.filters[2])
        self.Upstage3 = Upblock(self.filters[2], self.filters[1])
        self.Upstage4 = Upblock(self.filters[1], self.filters[0])
        self.Upstage5 = Upblock(self.filters[0], self.filters[0])

        self.Sk1 = SMM4(self.filters[0], self.filters[0])
        self.Sk2 = SMM3(self.filters[1], self.filters[1])
        self.Sk3 = SMM2(self.filters[2], self.filters[2])
        self.Sk4 = SMM1(self.filters[3], self.filters[3])

        self.Att_stage1 = MSM(self.filters[0], self.filters[0])
        self.Att_stage2 = MSM(self.filters[1], self.filters[1])
        self.Att_stage3 = MSM(self.filters[2], self.filters[2])
        self.Att_stage4 = MSM(self.filters[3], self.filters[3])
        self.Att_stage5 = MSM(self.filters[4], self.filters[4])
        self.Att_stageU4 = MSM(self.filters[3], self.filters[3])
        self.Att_stageU3 = MSM(self.filters[2], self.filters[2])
        self.Att_stageU2 = MSM(self.filters[1], self.filters[1])
        self.Att_stageU1 = MSM(self.filters[0], self.filters[0])

        self.final = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        ### Stage 1
        out = self.Convstage1(x)
        out = self.Att_stage1(out)
        t1 = out

        ### Stage 2
        out = self.Convstage2(out)
        out = self.Att_stage2(out)
        t2 = out

        ### Stage 3
        out = self.Convstage3(out)
        out = self.Att_stage3(out)
        t3 = out

        ### Stage 4
        out = self.Convstage4(out)
        out = self.Att_stage4(out)
        t4 = out

        ### Bottleneck(5)
        out = self.Convstage5(out)
        out = self.Att_stage5(out)

        ### Stage 4
        out = self.Upstage1(out)
        out = self.Att_stageU4(out)
        out = torch.add(out, self.Sk4(t4))

        ### Stage 3
        out = self.Upstage2(out)
        out = self.Att_stageU3(out)
        out = torch.add(out, self.Sk3(t3))

        ### Stage 2
        out = self.Upstage3(out)
        out = self.Att_stageU2(out)
        out = torch.add(out, self.Sk2(t2))

        ### Stage 1
        out = self.Upstage4(out)
        out = self.Att_stageU1(out)
        out = torch.add(out, self.Sk1(t1))

        ### Stage 0
        out = self.Upstage5(out)
        return self.final(out)


def dds_unet(input_channel=3,num_classes=1):
    return DDS_UNet(n_channels=input_channel,num_classes=num_classes)

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DDS_UNet(1).to(device)
    flops, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True, print_per_layer_stat=True)

    print('flops: ', flops, 'params: ', params)