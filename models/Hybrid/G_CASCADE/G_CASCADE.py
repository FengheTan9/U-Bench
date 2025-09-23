import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import logging

from scipy import ndimage

from .pvtv2 import pvt_v2_b2, pvt_v2_b5, pvt_v2_b0
from .decoders import CUP, CASCADE, CASCADE_Cat, GCUP, GCUP_Cat, GCASCADE, GCASCADE_Cat
from .pyramid_vig import pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu

from .maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from .maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from .maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from .maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out
from torch.hub import load_state_dict_from_url
logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    
class PVT_CUP(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_CUP, self).__init__()
        
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = CUP(channels=[512, 320, 128, 64])
        print('Model %s created, param count: %d' %
                     ('CUP decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4

class PVT_CASCADE(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_CASCADE, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = CASCADE(channels=[512, 320, 128, 64])
        print('Model %s created, param count: %d' %
                     ('CASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4

class PVT_CASCADE_Cat(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_CASCADE_Cat, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        print('Model %s created, param count: %d' %
                     ('PVT backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        # decoder initialization
        self.decoder = CASCADE_Cat(channels=[512, 320, 128, 64])
        
        print('Model %s created, param count: %d' %
                     ('CASCADE_Cat decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4

                        
class PVT_GCUP(nn.Module):
    def __init__(self, n_class=1, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', skip_aggregation='additive'):
        super(PVT_GCUP, self).__init__()
        
        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.channels = [512, 320, 128, 64]
        
        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCUP(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCUP_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCUP(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
                     ('GCUP_decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4  
                
                
class PVT_GCASCADE(nn.Module):
    def __init__(self, num_classes=1, img_size=256, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', skip_aggregation='additive', input_channel=3):
        super(PVT_GCASCADE, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = num_classes
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        save_model = load_state_dict_from_url('[URL]', progress=True)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.channels = [512, 320, 128, 64]
        
        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCASCADE_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)
        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        return p1, p2, p3, p4

class MERIT_GCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', interpolation='bilinear', skip_aggregation='additive'):
        super(MERIT_GCASCADE, self).__init__()
        
        self.interpolation = interpolation
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone1 = maxxvit_rmlp_small_rw_256_4out()  # [64, 128, 320, 512]
        self.backbone2 = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
  
        print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        state_dict1 = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')        
        self.backbone1.load_state_dict(state_dict1, strict=False)
        
        print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')       
        state_dict2 = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')        
        self.backbone2.load_state_dict(state_dict2, strict=False)
        
        print('Pretrain weights loaded.')
        
        self.channels = [768, 384, 192, 96]
        
        # decoder initialization 
        if self.skip_aggregation == 'additive':
            self.decoder1 = GCASCADE(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.decoder2 = GCASCADE(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder1 = GCASCADE_Cat(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.decoder2 = GCASCADE_Cat(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder1 = GCASCADE(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.decoder2 = GCASCADE(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
               
        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder1.parameters()])))
        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder2.parameters()])))
                         
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

        self.out_head4_in = nn.Conv2d(self.channels[3], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
            
        # transformer backbone as encoder
        f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))                
        #print([f1[3].shape,f1[2].shape,f1[1].shape,f1[0].shape])
        
        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder1(f1[3], [f1[2], f1[1], f1[0]])

        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)

        p14_in = self.out_head4_in(x14_o)
        p14_in = self.sigmoid(p14_in)
        

        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)

        p14_in = F.interpolate(p14_in, scale_factor=4, mode=self.interpolation)        
        x_in = x * p14_in
                
        f2 = self.backbone2(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
                    
        skip1_0 = F.interpolate(f1[0], size=(f2[0].shape[-2:]), mode=self.interpolation)
        skip1_1 = F.interpolate(f1[1], size=(f2[1].shape[-2:]), mode=self.interpolation)
        skip1_2 = F.interpolate(f1[2], size=(f2[2].shape[-2:]), mode=self.interpolation)
        skip1_3 = F.interpolate(f1[3], size=(f2[3].shape[-2:]), mode=self.interpolation)
        
        x21_o, x22_o, x23_o, x24_o = self.decoder2(f2[3]+skip1_3, [f2[2]+skip1_2, f2[1]+skip1_1, f2[0]+skip1_0])

        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)

        #print([p21.shape,p22.shape,p23.shape,p24.shape])
               
        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)
        
        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        #print([p1.shape,p2.shape,p3.shape,p4.shape])
        return p1, p2, p3, p4
                                

def g_cascade(num_classes, input_channel=3):
    model = PVT_GCASCADE(num_classes=num_classes, input_channel=input_channel)
    return model
