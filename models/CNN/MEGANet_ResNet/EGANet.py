import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules import *
import torchvision.models as models
from torchvision.transforms.functional import rgb_to_grayscale

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4), 
            Conv(in_channels // 4, out_channels)

        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

#Edge-guided Attention Network
class EGANet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(EGANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        resnet = models.resnet34(pretrained=True)
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        self.up5 = nn.Sequential(
            Conv(512, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ) 
        self.up4 = Up(768, 256)
        self.up3 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(128, 64)

        self.ega1 = EGA(64)
        self.ega2 = EGA(64)
        self.ega3 = EGA(128)
        self.ega4 = EGA(256)
        
        self.out5 = Out(512, n_classes)
        self.out4 = Out(256, n_classes)
        self.out3 = Out(128, n_classes)
        self.out2 = Out(64, n_classes)
        self.out1 = Out(64, n_classes)

    def forward(self, x):
        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

        #Encoder
        e1 = self.encoder1_conv(x)  
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  
        
        e2 = self.encoder2(e1_pool)

        e3 = self.encoder3(e2)  
        
        e4 = self.encoder4(e3)  

        e5 = self.encoder5(e4)  

        #Decoder
        d5 = self.up5(e5)
        out5 = self.out5(d5)
        ega4 = self.ega4(edge_feature, e4, out5)

        d4 = self.up4(d5, ega4)
        out4 = self.out4(d4)    
        ega3 = self.ega3(edge_feature, e3, out4)

        d3 = self.up3(d4, ega3)
        out3 = self.out3(d3)
        ega2 = self.ega2(edge_feature, e2, out3)

        d2 = self.up2(d3, ega2)
        out2 = self.out2(d2)  
        ega1 = self.ega1(edge_feature, e1, out2)

        d1 = self.up1(d2, ega1)
        out1 = self.out1(d1)
        
        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear') 
        out3 = F.interpolate(out3, scale_factor=4, mode='bilinear') 
        out4 = F.interpolate(out4, scale_factor=8, mode='bilinear') 
        out5 = F.interpolate(out5, scale_factor=16, mode='bilinear') 

        return  out2, out3, out4, out5, out1


class EGANetModel(nn.Module): 
    def __init__(self, n_channels=3, n_classes=1):
        super(EGANetModel,self).__init__()
        self.channel = n_channels
        self.num_classes = n_classes
        self.net = EGANet(self.channel, self.num_classes)

    def forward(self, images):
        out1, out2, out3, out4, out5= self.net(images)
        return out1, out2, out3, out4, out5


def eganet(input_channel=3, num_classes=1):
    return EGANetModel(n_channels=input_channel, n_classes=num_classes)