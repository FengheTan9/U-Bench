from torch import nn
from torch import cat
import numpy as np


class LFU_Net(nn.Module):
    def __init__(self, input_channel,num_classes):

        super().__init__()

        n = 8
        g = 8
        # x0.0
        self.stage00 = nn.Sequential(
            nn.Conv2d(  input_channel,  n, kernel_size=3, stride=1, padding=1, groups=1), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1, groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x1.0
        self.stage10 = nn.Sequential(
            nn.Conv2d(  n, 2*n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(2*n), nn.ReLU(inplace=True),
            nn.Conv2d(2*n, 2*n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(2*n), nn.ReLU(inplace=True),
        )
        # x2.0
        self.stage20 = nn.Sequential(
            nn.Conv2d(2*n, 4*n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(4*n), nn.ReLU(inplace=True),
            nn.Conv2d(4*n, 4*n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(4*n), nn.ReLU(inplace=True),
        )
        # x3.0
        self.stage30 = nn.Sequential(
            nn.Conv2d(4*n, 8*n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(8*n), nn.ReLU(inplace=True),
            nn.Conv2d(8*n, 8*n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(8*n), nn.ReLU(inplace=True),
        )
        # x4.0
        self.stage40 = nn.Sequential(
            nn.Conv2d( 8*n, 16*n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(16*n), nn.ReLU(inplace=True),
            nn.Conv2d(16*n, 16*n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(16*n), nn.ReLU(inplace=True),
        )

        # x0.1
        self.stage01 = nn.Sequential(
            nn.Conv2d(2*n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x1.1
        self.stage11 = nn.Sequential(
            nn.Conv2d(3*n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x0.2
        self.stage02 = nn.Sequential(
            nn.Conv2d(4*n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x2.1
        self.stage21 = nn.Sequential(
            nn.Conv2d(4*n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x1.2
        self.stage12 = nn.Sequential(
            nn.Conv2d(5*n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x0.3
        self.stage03 = nn.Sequential(
            nn.Conv2d(6*n,   n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,   n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x3.1
        self.stage31 = nn.Sequential(
            nn.Conv2d(5*n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x2.2
        self.stage22 = nn.Sequential(
            nn.Conv2d(6*n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x1.3
        self.stage13 = nn.Sequential(
            nn.Conv2d(7*n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )
        # x0.4
        self.stage04 = nn.Sequential(
            nn.Conv2d(8*n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
            nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g), nn.BatchNorm2d(  n), nn.ReLU(inplace=True),
        )


        # trans
        self.stage0p = nn.Sequential(nn.Conv2d(  n,  n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stage1p = nn.Sequential(nn.Conv2d(2*n,  n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stage2p = nn.Sequential(nn.Conv2d(4*n,  n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stage3p = nn.Sequential(nn.Conv2d(8*n,  n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stage4p = nn.Sequential(nn.Conv2d(16*n, n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stagep = self.stage0p

        self.stage0zp = nn.Sequential(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stage1zp = nn.Sequential(nn.Conv2d(2 * n, n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stage2zp = nn.Sequential(nn.Conv2d(4 * n, n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stage3zp = nn.Sequential(nn.Conv2d(8 * n, n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stage4zp = nn.Sequential(nn.Conv2d(16 * n, n, kernel_size=3, stride=1, padding=1,groups=g))
        self.stagezp = self.stage0zp

        # down-sampling
        self.stage2d = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2zd = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage4zd = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))
        self.stage8zd = nn.Sequential(nn.MaxPool2d(kernel_size=8, stride=8))

        # up-sampling
        self.stage2u = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.stage4u = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        self.stage8u = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
        self.stage16u = nn.Sequential(nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True))

        # output
        self.stageout = nn.Sequential(
            nn.Conv2d(  n, num_classes, 1 ,groups=1)
        )


    def forward(self, x):
        s00 = self.stage00(x)
        s10 = self.stage10(self.stage2d(s00))
        s01 = self.stage01(cat([self.stage0p(s00),
                                self.stage1p(self.stage2u(s10))], 1))

        s20 = self.stage20(self.stage2d(s10))
        s11 = self.stage11(cat([self.stage0zp(self.stage2zd(s00)),
                                self.stage1p(s10),
                                self.stage2p(self.stage2u(s20))], 1))
        s02 = self.stage02(cat([self.stage0p(s00), self.stagezp(s01),
                                self.stage2zp(self.stage4u(s20)), self.stagep(self.stage2u(s11))], 1))

        s30 = self.stage30(self.stage2d(s20))
        s21 = self.stage21(cat([self.stage0zp(self.stage4zd(s00)), self.stage1zp(self.stage2zd(s10)),
                                self.stage2p(s20),
                                self.stage3p(self.stage2u(s30))], 1))
        s12 = self.stage12(cat([self.stage0zp(self.stage2zd(s00)),
                                self.stage1p(s10), self.stagezp(s11),
                                self.stage3zp(self.stage4u(s30)), self.stagep(self.stage2u(s21))], 1))
        s03 = self.stage03(
            cat([self.stage0p(s00), self.stagezp(s01), self.stagezp(s02),
                 self.stage3zp(self.stage8u(s30)), self.stagezp(self.stage4u(s21)), self.stagep(self.stage2u(s12))], 1))

        s40 = self.stage40(self.stage2d(s30))
        s31 = self.stage31(
            cat([self.stage0zp(self.stage8zd(s00)), self.stage1zp(self.stage4zd(s10)),
                 self.stage2zp(self.stage2zd(s20)),
                 self.stage3p(s30),
                 self.stage4p(self.stage2u(s40))], 1))
        s22 = self.stage22(cat([self.stage0zp(self.stage4zd(s00)), self.stage1zp(self.stage2zd(s10)),
                                self.stage2p(s20), self.stagezp(s21),
                                self.stage4zp(self.stage4u(s40)), self.stagep(self.stage2u(s31))], 1))
        s13 = self.stage13(
            cat([self.stage0zp(self.stage2zd(s00)),
                 self.stage1p(s10), self.stagezp(s11), self.stagezp(s12),
                 self.stage4zp(self.stage8u(s40)), self.stagezp(self.stage4u(s31)), self.stagep(self.stage2u(s22))], 1))
        s04 = self.stage04(
            cat([self.stage0p(s00), self.stagezp(s01), self.stagezp(s02), self.stagezp(s03),
                 self.stage4zp(self.stage16u(s40)), self.stagezp(self.stage8u(s31)),
                 self.stagezp(self.stage4u(s22)), self.stagep(self.stage2u(s13))], 1))

        out1 = self.stageout(s01)
        out2 = self.stageout(s02)
        out3 = self.stageout(s03)
        out4 = self.stageout(s04)
        # out = out1
        # out = (out1 + out2) / 2
        # out = (out1 + out2 + out3) / 3
        out = (out1 + out2 + out3 + out4) / 4

        return out


def lfu_net(input_channel,num_classes):
    return LFU_Net(input_channel=input_channel,num_classes=num_classes)

