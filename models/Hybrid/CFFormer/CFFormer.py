import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, swin_v2_t, Swin_V2_T_Weights, swin_v2_s, Swin_V2_S_Weights, swin_v2_b, Swin_V2_B_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .Attention import Channel_Exchange_Attention, X_spatial
import numpy as np
import random

class DoubleConv(nn.Module):
    """convolution->BN->Relu"""
    def __init__(self,in_channels,out_channels,mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.double_conv(x)


class encoder(nn.Module):
    def __init__(self, cnn: nn.Module, transformer: nn.Module, cnn_chans = [64, 64, 128, 256, 512], trans_chans = [96, 192, 384, 768], out_chan = [96, 192, 384, 768]): #[128, 256, 512, 1024]):
        super(encoder, self).__init__()
        self.cnn = cnn
        self.transformer = transformer 
        self.cnn_layer0 = nn.Sequential(  
            self.cnn.conv1,
            self.cnn.bn1,
            self.cnn.relu,
            
        )

        self.cnn_layer1 = nn.Sequential(
            self.cnn.maxpool,
            self.cnn.layer1
        )
        self.cnn_layer2 = self.cnn.layer2
        self.cnn_layer3 = self.cnn.layer3
        self.cnn_layer4 = self.cnn.layer4

        self.transformer_layer1 = nn.Sequential(
            self.transformer.features[0],
            self.transformer.features[1] 
        )
        self.cdca1 = Channel_Exchange_Attention(trans_chans[0], cnn_chans[1])
        self.X_spatial_1 = X_spatial(trans_chans[0], cnn_chans[1], out_chan[0])


        self.transformer_layer2 = nn.Sequential(
            self.transformer.features[2],
            self.transformer.features[3]
        )
        self.cdca2 = Channel_Exchange_Attention(trans_chans[1], cnn_chans[2])
        self.X_spatial_2 = X_spatial(trans_chans[1], cnn_chans[2], out_chan[1])


        self.transformer_layer3 = nn.Sequential(
            self.transformer.features[4],
            self.transformer.features[5]
        )
        self.cdca3 = Channel_Exchange_Attention(trans_chans[2], cnn_chans[3])
        self.X_spatial_3 = X_spatial(trans_chans[2], cnn_chans[3], out_chan[2])

        self.transformer_layer4 = nn.Sequential(
            self.transformer.features[6],
            self.transformer.features[7] 
        )
        self.cdca4 = Channel_Exchange_Attention(trans_chans[3], cnn_chans[4])
        self.X_spatial_4 = X_spatial(trans_chans[3], cnn_chans[4], out_chan[3])


    def forward(self, x):
        skip0 = self.cnn_layer0(x)
      

        cnn_layer1 = self.cnn_layer1(skip0)
        transformer_layer1 = self.transformer_layer1(x).permute(0, 3, 1, 2)
        transformer_cnn_1, cnn_transformer_1 = self.cdca1(transformer_layer1, cnn_layer1)
        cnn_input_1 = transformer_cnn_1 + cnn_layer1
        trans_input_1 = cnn_transformer_1 + transformer_layer1
        skip1 = self.X_spatial_1(trans_input_1, cnn_input_1)


        cnn_layer2 = self.cnn_layer2(cnn_input_1)
        transformer_layer2 = self.transformer_layer2(trans_input_1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        transformer_cnn_2, cnn_transformer_2 = self.cdca2(transformer_layer2, cnn_layer2)
        cnn_input_2 = transformer_cnn_2 + cnn_layer2
        trans_input_2 = cnn_transformer_2 + transformer_layer2
        skip2 = self.X_spatial_2(trans_input_2, cnn_input_2)        


        cnn_layer3 = self.cnn_layer3(cnn_input_2)
        transformer_layer3 = self.transformer_layer3(trans_input_2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        transformer_cnn_3, cnn_transformer_3 = self.cdca3(transformer_layer3, cnn_layer3)
        cnn_input_3 = transformer_cnn_3 + cnn_layer3
        trans_input_3 = cnn_transformer_3 + transformer_layer3
        skip3 = self.X_spatial_3(trans_input_3, cnn_input_3)

        cnn_layer4 = self.cnn_layer4(cnn_input_3)
        transformer_layer4 = self.transformer_layer4(trans_input_3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        transformer_cnn_4, cnn_transformer_4 = self.cdca4(transformer_layer4, cnn_layer4)
        cnn_input_4 = transformer_cnn_4 + cnn_layer4
        trans_input_4 = cnn_transformer_4 + transformer_layer4
        skip4 = self.X_spatial_4(trans_input_4, cnn_input_4)


        return [skip0, skip1, skip2, skip3, skip4]
class se_block_without_shorcut(nn.Module):
    def __init__(self, channels = 512, mid_channels = 256) -> None:
        super(se_block_without_shorcut, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(channels, mid_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mid_channels, channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, _, _ = x.shape
        out = self.pool(x).view(B, C)
        out = self.linear1(out)
        out = self.relu(out)
 
        out = self.linear2(out).unsqueeze(-1)
        out = self.sigmoid(out)
        return out

class up_channel_selection(nn.Module):
    def __init__(self, high_level_channel = 512, low_level_channel = 256):
        super(up_channel_selection, self).__init__()
        self.high_level_feature_se_block = se_block_without_shorcut(high_level_channel, low_level_channel)
        self.low_level_feature_se_block = se_block_without_shorcut(low_level_channel, high_level_channel)
    def forward(self, high_level_feature, low_level_feature):
        high_level_channel = self.high_level_feature_se_block(high_level_feature).transpose(1, 2)
        low_level_channel = self.low_level_feature_se_block(low_level_feature)
        attn = low_level_channel@high_level_channel  
        return attn    
class decoder(nn.Module):
    def __init__(self, chans = [64, 64, 128, 256, 512], n_classes = 1) -> None:
        super(decoder, self).__init__()

        self.upconv5 = self.double_block(chans[4], chans[3])  
        self.upconv4 = self.double_block(chans[3]*2, chans[2]) 
        self.upconv3 = self.double_block(chans[2]*2, chans[1])   
        self.upconv2 = self.double_block(chans[1]*2, chans[0])    
        self.upconv1 = self.double_block(chans[0]*2, chans[0])    
        self.out = nn.Sequential(
            nn.Conv2d(chans[0], chans[0], kernel_size=3, padding=1),  # 卷积
            nn.BatchNorm2d(chans[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans[0], chans[0], kernel_size=3, padding=1),  # 再次卷积
            nn.BatchNorm2d(chans[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans[0], n_classes, kernel_size=1, padding=0)
        )

    
    def double_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 卷积
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 再次卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        up5= self.upconv5(x5)
        x4= torch.cat((x4, up5), dim = 1)
        up4 = self.upconv4(x4)
        x3 = torch.cat((x3, up4), dim = 1)
        up3 = self.upconv3(x3)
        x2 = torch.cat((x2, up3), dim = 1)
        up2 = self.upconv2(x2)
        x1 = torch.cat((x1, up2), dim = 1)
        up1 = self.upconv1(x1)
        out = self.out(up1)
        return out
        
        


def CDFormer_small(n_classes = 1):
    cnn = resnet34(ResNet34_Weights.IMAGENET1K_V1)
    transformer = swin_v2_t(weights = Swin_V2_T_Weights.IMAGENET1K_V1)
    enc = encoder(cnn, transformer, cnn_chans=[64, 64, 128, 256, 512], trans_chans = [96, 192, 384, 768], out_chan=[96, 192, 384, 768])
    dec = decoder([64, 96, 192, 384, 768], n_classes = n_classes)
    return enc, dec

def CDFormer_base(n_classes = 1):
    cnn = resnet34(ResNet34_Weights.IMAGENET1K_V1)
    transformer = swin_v2_s(weights = Swin_V2_S_Weights.IMAGENET1K_V1)
    enc = encoder(cnn, transformer, cnn_chans=[64, 64, 128, 256, 512], trans_chans = [96, 192, 384, 768], out_chan=[96, 192, 384, 768])
    dec = decoder([64, 96, 192, 384, 768], n_classes = n_classes)
    return enc, dec

def CDFormer_large(n_classes = 1):
    cnn = resnet34(ResNet34_Weights.IMAGENET1K_V1)
    transformer = swin_v2_b(weights = Swin_V2_B_Weights.IMAGENET1K_V1)
    enc = encoder(cnn, transformer, cnn_chans=[64, 64, 128, 256, 512], trans_chans = [128, 256, 512, 1024], out_chan=[128, 256, 512, 1024])
    dec = decoder([64, 128, 256, 512, 1024], n_classes = n_classes)
    return enc, dec


class CDFormer(nn.Module):
    def __init__(self, version = 'b', input_channel=3, num_classes=1) -> None:
        super(CDFormer, self).__init__()
        if version == 's':
            self.encoder, self.decoder = CDFormer_small(n_classes=num_classes)
        elif version == 'b':
            self.encoder, self.decoder = CDFormer_base(n_classes=num_classes)
        else:
            self.encoder, self.decoder = CDFormer_large(n_classes=num_classes)

        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for selection in [self.encoder.cdca1,
                          self.encoder.cdca2,
                          self.encoder.cdca3,
                          self.encoder.cdca4,]:
            for m in selection.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out


def cfformer(input_channel=3, num_classes=1):
    return CDFormer(input_channel=input_channel, num_classes=num_classes)

