#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:34:55 2022

@author: bougourzi
"""

import torch
import torch.nn as nn
from itertools import repeat

from typing import Union, Sequence

from monai.networks.blocks.transformerblock import TransformerBlock

device = torch.device("cuda:0")


##############################################################################
class PatchEmbeddingBlock(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        img_size =  tuple(repeat(img_size, 2))
        patch_size = tuple(repeat(patch_size, 2))
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

##############################################################################
# Transformer Path
class ViT(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        ):
        super().__init__()

        self.patch_embedding = PatchEmbeddingBlock()
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out
    

##############################################################################
# Attention Block
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

##############################################################################
# ResBlock   
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
                
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) 
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))           

    def forward(self, x):
        return self.conv(x) + self.skip(x) 
    
##############################################################################
# UpResBlock
class UPDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer):
        super(UPDoubleConv, self).__init__()
                
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels, out_channels)
        )            

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(out_channels, out_channels)                    
                )
                for i in range(num_layer)
            ]
        )
    def forward(self, x):
        x = self.deconv(x)
        for blk in self.blocks:
            x = blk(x)
        return x        

##############################################################################    
# D-TrAttUnet Architecture
class DTrAttUnet(nn.Module):
    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12):
        super().__init__()

        self.hidden_size = hidden_size
        self.classification = False
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.vit = ViT(in_channels=3,img_size=256,patch_size=16)
        nb_filter = [32, 64, 128, 256, 512]        
        self.encoder1 = DoubleConv(input_channel, nb_filter[0])
        
        self.conv1 = DoubleConv(input_channel, nb_filter[0])
        self.conv2 = DoubleConv(nb_filter[1], nb_filter[1])
        self.conv3 = DoubleConv(nb_filter[2], nb_filter[2])
        self.conv4 = DoubleConv(nb_filter[3], nb_filter[3])
        self.conv5 = DoubleConv(nb_filter[4], nb_filter[4])
        
        
        
        self.encoder2 = UPDoubleConv(
                    in_channels = hidden_size, out_channels = nb_filter[0], num_layer = 2
        )
        self.encoder3 = UPDoubleConv(
                    in_channels = hidden_size, out_channels = nb_filter[1], num_layer = 1
        ) 
        self.encoder4 = UPDoubleConv(
                    in_channels = hidden_size, out_channels = nb_filter[2], num_layer = 0
        )
        self.encoder5 = DoubleConv(hidden_size, nb_filter[3])
        
        # Decoder1        
        self.Att4 = Attention_block(F_g= nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[3])        
        self.Att3 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2])       
        self.Att2 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1])        
        self.Att1 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0]) 

 
        self.deconv1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.deconv2 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.deconv3 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.deconv4 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
        # Decoder2        
        self.Att41 = Attention_block(F_g= nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[3])        
        self.Att31 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2])       
        self.Att21 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1])        
        self.Att11 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0]) 

 
        self.deconv11 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.deconv21 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.deconv31 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.deconv41 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])

        self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)        
        
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size, feat_size, hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x    
    

    def forward(self, x_in):
        v4, hidden_states_out = self.vit(x_in)
        feat_size = 16
        hidden_size = 768

        x1 = self.conv1(x_in)
        
        v1 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(v1, hidden_size ,feat_size))
        x2 = self.conv2(torch.cat([self.pool(x1), enc2], dim=1))
       
        v2 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(v2, hidden_size, feat_size))
        x3 = self.conv3(torch.cat([self.pool(x2), enc3], dim=1))
        
        v3 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(v3, hidden_size, feat_size))
        x4 = self.conv4(torch.cat([self.pool(x3), enc4], dim=1)) 
        
        enc5 = self.encoder5(self.proj_feat(v4, hidden_size, feat_size))
        x5 = self.conv5(torch.cat([self.pool(x4), enc5], dim=1))        
        
        # Decoder
        # Layer1
        # 1
        x50 = self.up(x5)
        xd4 = self.Att4(g=x50, x=x4) 
        d1 = self.deconv1(torch.cat([xd4, x50], 1))
        # 2
        x51 = self.up(x5)
        xd41 = self.Att41(g=x51, x=x4) 
        d11 = self.deconv11(torch.cat([xd41, x51], 1))        
        # Layer2
        #1
        x40 = self.up(d1)
        xd3 = self.Att3(g=x40, x=x3)        
        d2 = self.deconv2(torch.cat([xd3, x40], 1))
        #2
        x41 = self.up(d11)
        xd31 = self.Att31(g=x41, x=x3)        
        d21 = self.deconv21(torch.cat([xd31, x41], 1))        
        # Layer3
        #1
        x30 = self.up(d2)
        xd2 = self.Att2(g=x30, x=x2)         
        d3 = self.deconv3(torch.cat([xd2, x30], 1))
        #2
        x31 = self.up(d21)
        xd21 = self.Att21(g=x31, x=x2)         
        d31 = self.deconv31(torch.cat([xd21, x31], 1))        
        # Layer4
        #1
        x20 = self.up(d3)
        xd1 = self.Att1(g=x20, x=x1)        
        d4 = self.deconv4(torch.cat([xd1, x20], 1))
        #2
        x21 = self.up(d31)
        xd11 = self.Att11(g=x21, x=x1)        
        d41 = self.deconv41(torch.cat([xd11, x21], 1))        
        output = self.final(d4)
        output2 = self.final1(d41)        
        
        return output, output2


def d_trattunet(num_classes, input_channel=3):
    model = DTrAttUnet(input_channel=input_channel, num_classes=num_classes)
    return model

