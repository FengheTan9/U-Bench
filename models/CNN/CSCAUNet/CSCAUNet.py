import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from torchsummary.torchsummary import summary

class Basic_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=1):
        super(Basic_blocks, self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x=self.conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2+x

class DSE(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(DSE, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.gapool=nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.layer1(gp)
        x=x*se
        gap=self.gapool(x)
        se2=self.layer2(gap)
        return x * se2

class Spaceatt(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(Spaceatt, self).__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.BatchNorm2d(in_channel // decay),
            nn.Conv2d(in_channel // decay, 1, 1),
            nn.Sigmoid()
        )
        self.K = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay, in_channel//decay, 3, padding=1),
            DSE(in_channel//decay)
        )
        self.V = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay, in_channel//decay, 3, padding=1),
            DSE(in_channel//decay)
        )
        self.sig = nn.Sequential(
            nn.Conv2d(in_channel // decay, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, high):
        Q = self.Q(low)
        K = self.K(low)
        V = self.V(high)
        att = Q * K
        att = att@V
        return self.sig(att)


class CSCA_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=2):
        super(CSCA_blocks, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channel, out_channel, 2, stride=2)
        self.conv = Basic_blocks(in_channel, out_channel//2)
        self.catt = DSE(out_channel//2, decay)
        self.satt = Spaceatt(out_channel//2, decay)
    def forward(self, high, low):
        up = self.upsample(high)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        catt = self.catt(point)
        satt = self.satt(point, catt)
        plusatt = catt*satt
        return torch.cat([plusatt, catt], dim=1)


class CSCAUNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, decay=2):
        super(CSCAUNet, self).__init__()
        self.pool = nn.MaxPool2d(2)

        self.down_conv1 = Basic_blocks(input_channel, 32, decay)
        self.down_conv2 = Basic_blocks(32, 64, decay)
        self.down_conv3 = Basic_blocks(64, 128, decay)
        self.down_conv4 = Basic_blocks(128, 256, decay)
        self.down_conv5 = Basic_blocks(256, 512, decay)

        self.down_conv6 = nn.Sequential(
            Basic_blocks(512, 1024, decay),
            DSE(1024,decay)
            )

        self.up_conv5 = CSCA_blocks(1024, 512, decay)
        self.up_conv4 = CSCA_blocks(512, 256, decay)
        self.up_conv3 = CSCA_blocks(256, 128, decay)
        self.up_conv2 = CSCA_blocks(128, 64, decay)
        self.up_conv1 = CSCA_blocks(64, 32, decay)

        self.dp6 = nn.Conv2d(1024, num_classes, 1)
        self.dp5 = nn.Conv2d(512, num_classes, 1)
        self.dp4 = nn.Conv2d(256, num_classes, 1)
        self.dp3 = nn.Conv2d(128, num_classes, 1)
        self.dp2 = nn.Conv2d(64, num_classes, 1)
        self.out = nn.Conv2d(32,num_classes,3,padding=1)  #103

        self.center5 = nn.Conv2d(1024, 512, 1)
        self.decodeup4 = nn.Conv2d(512, 256, 1)
        self.decodeup3 = nn.Conv2d(256, 128, 1)
        self.decodeup2 = nn.Conv2d(128, 64, 1)

    def forward(self, inputs):
        b, c, h, w = inputs.size()
        down1 = self.down_conv1(inputs)
        pool1 = self.pool(down1)

        down2 = self.down_conv2(pool1)
        pool2 = self.pool(down2)

        down3 = self.down_conv3(pool2)
        pool3 = self.pool(down3)

        down4 = self.down_conv4(pool3)
        pool4 = self.pool(down4)

        down5 = self.down_conv5(pool4)
        pool5 = self.pool(down5)

        center = self.down_conv6(pool5)

        out6 = self.dp6(center)
        out6 = F.interpolate(
            out6, (h, w), mode='bilinear', align_corners=False)

        deco5 = self.up_conv5(center, down5)
        out5 = self.dp5(deco5)
        out5 = F.interpolate(
            out5, (h, w), mode='bilinear', align_corners=False)
        center5 = self.center5(center)
        center5 = F.interpolate(center5, (h//16, w//16),
                                mode='bilinear', align_corners=False)
        deco5 = deco5+center5

        deco4 = self.up_conv4(deco5, down4)
        out4 = self.dp4(deco4)
        out4 = F.interpolate(
            out4, (h, w), mode='bilinear', align_corners=False)
        decoderup4 = self.decodeup4(deco5)
        decoderup4 = F.interpolate(
            decoderup4, (h//8, w//8), mode='bilinear', align_corners=False)
        deco4 = deco4+decoderup4
        
        deco3 = self.up_conv3(deco4, down3)
        out3 = self.dp3(deco3)
        out3 = F.interpolate(
            out3, (h, w), mode='bilinear', align_corners=False)
        decoderup3 = self.decodeup3(deco4)
        decoderup3 = F.interpolate(
            decoderup3, (h//4, w//4), mode='bilinear', align_corners=False)
        deco3 = deco3+decoderup3

        deco2 = self.up_conv2(deco3, down2)
        out2=self.dp2(deco2)
        out2=F.interpolate(out2,(h,w),mode='bilinear',align_corners=False)
        decoderup2=self.decodeup2(deco3)
        decoderup2=F.interpolate(
            decoderup2,(h//2,w//2),mode='bilinear',align_corners=False
        )
        deco2=deco2+decoderup2

        deco1=self.up_conv1(deco2,down1)
        out = self.out(deco1)
        return out2,out3, out4, out5, out6, out


if __name__ == '__main__':
    model = CSCAUNet(1, 2)
    summary(model, (3, 352, 352),batch_size=1,device='cpu')
    # print('# generator parameters:', sum(param.numel()
        #   for param in model.parameters()))