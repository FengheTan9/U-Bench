from re import S
from turtle import forward
from xml.dom.minidom import Identified
import torch
import torch.nn as nn
from torchvision.models import resnet34 as resnet
# from .DeiT import deit_small_patch16_224 as deit
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class DRA(nn.Module): # Differential Regional Attention
    def __init__(self, in_ch=64, class_num=2, value=0.5, simple=1, large=0):
        super(DRA, self).__init__()
        self.threshold = value
        self.s_mode = simple
        self.l_mode = large
        self.sigmoid = nn.Sigmoid()
        self.add_learn = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.GELU()
        )
        self.to_class_1 = nn.Conv2d(in_channels=in_ch, out_channels=class_num, kernel_size=1)
        self.to_class_2 = nn.Conv2d(in_channels=in_ch, out_channels=class_num, kernel_size=1)
        self.region_learn = nn.Sequential(
            nn.Conv2d(in_channels=class_num, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.GELU()
        )
    
    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        if self.s_mode == 1:
            x1_in       = 1*(x1 > self.threshold) 
            x2_in       = 1*(x2 > self.threshold) 
            edge_diff   = torch.abs(x1_in - x2_in)          
            x           = (edge_diff) * x1 + x1 
            x           = self.add_learn(x)               
        elif self.l_mode == 1:
            x1_in       = self.to_class_1(x1)
            x2_in       = self.to_class_2(x2)
            x1_in       = 1*(x1_in > self.threshold) 
            x2_in       = 1*(x2_in > self.threshold) 
            edge_diff   = torch.abs(x1_in - x2_in) * 1.000000
            edge_diff   = self.region_learn(edge_diff) 
            x           = self.sigmoid(edge_diff) * x1 + x1
            x           = self.add_learn(x) 
        return x

class IRE(nn.Module):
    def __init__(self, in_ch, rate, only_ch=0, only_sp=0):
        super(IRE, self).__init__()
        self.fc1        = nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch/rate), kernel_size=1)
        self.relu       = nn.ReLU(inplace=True)
        self.fc2        = nn.Conv2d(in_channels=int(in_ch/rate), out_channels=in_ch, kernel_size=1)
        self.sigmoid    = nn.Sigmoid()

        self.compress   = ChannelPool()
        self.spatial    = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        self.fc3        = nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch/rate), kernel_size=1)
        self.fc4        = nn.Conv2d(in_channels=int(in_ch/rate), out_channels=in_ch, kernel_size=1)

        self.ch_use     = only_ch
        self.ch_sp_use  = only_sp 
    
    def forward(self, x):
        x_in = x                # 8 256 12 16
        x = x.mean((2, 3), keepdim=True) # 8 256 1 1
        x = self.fc1(x)         # 8 256 1 1 -> 8 64 1 1
        x = self.relu(x)        # 8 64 1 1 -> 8 64 1 1
        x = self.fc2(x)         # 8 64 1 1 -> 8 256 1 1
        x = self.sigmoid(x)     # 8 256 12 16 add channel attention
        if self.ch_use == 1:
            return x * x_in 
        elif self.ch_use == 0:
            x = x * x_in

        
        s_in = x                    # 8 256 12 16
        s = self.compress(x)        # 8 256 12 16 -> 8 2 12 16
        s = self.spatial(s)         # 8 2 12 16 -> 8 1 12 16
        s = self.sigmoid(s)         # 8 1 12 16
        if self.ch_sp_use == 1:
            return s 
        elif self.ch_sp_use == 0:
            s = s * s_in

        
        c_in = s                    # 8 256 12 16
        c = self.fc3(s)             # 8 256 12 16 -> 8 64 12 16
        c = self.relu(c)
        c = self.fc4(c)             # 8 64 12 16 -> 8 256 12 16
        c = self.sigmoid(c) * c_in  # 8 256 12 16 -> 8 256 12 16
    
        return c

class MRA(nn.Module): # Multiple Receptive-field Aggregation
    def __init__(self, c1_in_channels=64, c2_in_channels=128, c3_in_channels=256, embedding_dim=256, drop_rate=0.2, classes=2):
        super(MRA, self).__init__()
        # embedding_dim 是超参数 统一多尺度feature map的channel维度
        self.conv_c1 = nn.Conv2d(in_channels=c1_in_channels, out_channels=embedding_dim, kernel_size=1)
        self.down_1  = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1)
        self.conv_c2 = nn.Conv2d(in_channels=c2_in_channels, out_channels=embedding_dim, kernel_size=1)
        self.down_2  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 三个不同尺度的数据进行融合
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim*3, out_channels=embedding_dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU()
        )
        self.drop = nn.Dropout2d(drop_rate)
        self.edge = DRA(in_ch=embedding_dim, class_num=classes)
    
    def forward(self, inputs):
        # inputs 包括三个feature map 分别是 x_b_2, x_b_1, x_b
        c1, c2, c3 = inputs

        c1_ = self.conv_c1(c1) # 16 64 48 64 -> 16 256 48 64
        c1_ = self.down_1(c1_) # 16 256 48 64 -> 16 256 12 16

        c2_ = self.conv_c2(c2) # 16 128 24 32 -> 16 256 24 32
        c2_ = self.down_2(c2_) # 16 256 24 32 -> 16 256 12 16

        c3_ = c3 # 16 256 12 16

        c_fuse = self.conv_fuse(torch.cat([c1_, c2_, c3_], dim=1)) # 8 64 48 64 -> 8 64*4 48 64 -> 8 64 48 64
        x = self.drop(c_fuse) + self.edge(c2_, c1_) + self.edge(c3_, c2_)

        return x

class PFC(nn.Module):
    def __init__(self, in_ch, channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.pointwise(x)
        return x

class CEE(nn.Module): # Context Enhanced Encoder
    def __init__(self, patch_size=3, stride=2, in_chans=64, embed_dim=64, smaller=0, use_att=0):
        super().__init__()
        self.att_use = use_att
        self.att = IRE(in_ch=embed_dim, rate=4, only_ch=0, only_sp=0)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2)),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.proj_c = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True, groups=embed_dim),
            nn.GELU()
        )
        self.fc0 = Conv(embed_dim, embed_dim, 3, bn=True, relu=True)
        self.dwconv_1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True , groups=embed_dim),
            nn.GELU()
        )
        self.use_small_conv = smaller
        self.dwconv_2 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, kernel_size=3, padding=1, stride=1, bias=True , groups=embed_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.turn_channel = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        # overlap 编码
        x_pe = self.proj(x) # 16 64 48 64 -> 16 64 24 32 
        if self.att_use == 1:
            x_pe = self.att(x_pe)
        # conv 编码
        x_pe_conv = self.proj_c(x)
        # fc_0
        x_PE = x_pe.flatten(2).transpose(1, 2) # 16 64 24 32 ->16 64 24*32 -> 16 24*32 64
        x_PE = self.norm(x_PE)
        x_po = self.dwconv(x_pe).flatten(2).transpose(1, 2)
        x_0  = torch.transpose((x_PE + x_po), 1, 2).view(b, x_pe.shape[1], int(h/2), int(w/2))
        x_0  = self.fc0(x_0) # 16 24*32 64
        # fc_1
        x_1  = x_0 # torch.transpose(x_0, 1, 2).view(b, x_0.shape[2], int(h/2), int(w/2))
        if self.use_small_conv == 1:
            x_1_ = self.dwconv_2(torch.cat([x_1, x_pe_conv], dim=1))
            x_1_ = self.turn_channel(torch.cat([x_1_, x_pe], dim=1)).flatten(2).transpose(1, 2)
            x_out  = x_1_ + x_PE
            return x_out
        else:
            x_1_ = self.dwconv_1(x_1) # .flatten(2).transpose(1, 2) 
            x_1_ = self.turn_channel(torch.cat([x_1, x_pe], dim=1)).flatten(2).transpose(1, 2)
            x_out  = self.fc1(x_1_) + x_PE
            return x_out

class ERDUnet(nn.Module):
    def __init__(self, input_channel=3,num_classes=1, drop_rate=0.2, in_h=256, in_w=256, normal_init=True, pretrained=False, bound=True, single_object=True):
        super(ERDUnet, self).__init__()
        self.model_h = in_h
        self.model_w = in_w

        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('E:/1_pytorch/TransFuse/pretrained/resnet34-43635321.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_1 = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
            )

        self.final_net_1 = nn.Sequential(
            Conv(32, 16, 3, bn=False, relu=False)
        )

        self.final_net_2 = nn.Sequential(
            Conv(32, 16, 3, bn=False, relu=False)
        )

        self.drop = nn.Dropout2d(drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        #---------------------------------------------------------------------------
        self.head = MRA(c1_in_channels=64, c2_in_channels=128, c3_in_channels=256, embedding_dim=256, classes=num_classes)
        self.att_0 = IRE(in_ch=256, rate=4, only_ch=0, only_sp=0)
        self.att_1 = IRE(in_ch=256, rate=4, only_ch=0, only_sp=0)
        
        self.toshow_p0  = nn.Sequential(nn.Identity()) # 32
        self.toshow_p1  = nn.Sequential(nn.Identity()) # 64
        self.toshow_p2  = nn.Sequential(nn.Identity()) # 128
        self.toshow_p3  = nn.Sequential(nn.Identity()) # 256
        
        self.toshow_bi  = nn.Identity()
        self.toshow_att = nn.Identity()
        self.out_norm_0 = nn.BatchNorm2d(256)
        self.out_norm_1 = nn.BatchNorm2d(256)

        self.lowest_layer_head = PFC(in_ch=input_channel, channels=16, kernel_size=7) # 16 3 192 256 -> 16 16 192 256

        self.out_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.out_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.out_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.out_1_skip = nn.Sequential(
            IRE(in_ch=128, rate=4, only_ch=0, only_sp=0),
            nn.BatchNorm2d(128)
        )
        self.out_2_skip = nn.Sequential(
            IRE(in_ch=64, rate=4, only_ch=0, only_sp=0),
            nn.BatchNorm2d(64)
        )
        self.out_3_skip = nn.Sequential(
            nn.GELU()
        )

        self.skip_0_1 = DRA(in_ch=256, class_num=num_classes, value=0.5, simple=1, large=0)
        self.skip_1_1 = DRA(in_ch=128, class_num=num_classes, value=0.5, simple=0, large=1)
        self.skip_2_1 = DRA(in_ch=64, class_num=num_classes, value=0.5, simple=0, large=1)
        self.skip_3   = DRA(in_ch=32, class_num=num_classes, value=0.5, simple=1, large=0)

        self.out_gelu_0 = nn.GELU()
        self.out_gelu_1 = nn.GELU()
        self.out_gelu_2 = nn.GELU()
        self.out_gelu_3 = nn.GELU()

        self.layer_0 = nn.Sequential(
            PFC(64, 32, 7)# Conv(64, 32, 1, bn=True, relu=True)
        )
        self.layer_0_1 = nn.Sequential(
            CEE(patch_size=3, stride=2, in_chans=32, embed_dim=64, smaller=0, use_att=0), # base_cnn_branch(input_channel=32, out_dim=64, conv_rate=2, simple=0),
            nn.Identity()
        )
        self.layer_1_1 = nn.Sequential(
            CEE(patch_size=3, stride=2, in_chans=64, embed_dim=128, smaller=1, use_att=1), # base_cnn_branch(input_channel=64, out_dim=128, conv_rate=2, simple=1),
            nn.Identity()
        )

        # double net
        self.split_att = IRE(16, 4, 0, 1) # split_attention(in_channels=32, channels=16, groups=1, radix=2, reduction_factor=2)
        self.down_path_0 = CEE(patch_size=3, stride=2, in_chans=16, embed_dim=32, smaller=1) # 16 16 192 256 -> 16 96*128 32
        self.down_path_1 = CEE(patch_size=3, stride=2, in_chans=32, embed_dim=64, smaller=1) # 16 32 96 128 -> 16 48*64 64
        self.down_path_2 = CEE(patch_size=3, stride=2, in_chans=64, embed_dim=128, smaller=1) # 16 64 48 64 -> 16 24*32 128

        self.head_2 = MRA(c1_in_channels=32, c2_in_channels=64, c3_in_channels=128, embedding_dim=128, classes=num_classes)
        self.att_0_2 = IRE(in_ch=128, rate=4, only_ch=0, only_sp=0)
        self.att_1_2 = IRE(in_ch=128, rate=4, only_ch=0, only_sp=0)
        self.out_norm_0_2 = nn.BatchNorm2d(128)
        self.out_norm_1_2 = nn.BatchNorm2d(128)

        self.out_2_2 = nn.Sequential(Conv(128, 64, 1, 1, bn=True, relu=True))
        self.out_3_2 = nn.Sequential(Conv(64, 32, 1, 1, bn=True, relu=True))

        self.out_2_2_skip = nn.Sequential(
            IRE(in_ch=64*2, rate=8, only_ch=1, only_sp=0),
            Conv(64*2, 64, 1, 1, True, True)
        )
        self.out_3_2_skip = nn.Sequential(
            IRE(in_ch=32*2, rate=8, only_ch=1, only_sp=0),
            Conv(32*2, 32, 1, 1, True, True)
            # nn.GELU()
        )

        self.skip_0_1_2 = DRA(in_ch=128, class_num=num_classes, value=0.5, simple=1, large=1) # 虽然这里是失误 写成了 1 1 但是实际的特征流动是large mode

        self.skip_2_1_2 = nn.Sequential(
            IRE(in_ch=64*2, rate=8, only_ch=1, only_sp=0),
            Conv(64*2, 64, 1, 1, True, True)
        )
        self.skip_3_1_2   = nn.Sequential(
            IRE(in_ch=32*2, rate=8, only_ch=1, only_sp=0),
            Conv(32*2, 32, 1, 1, True, True)
        )
        #---------------------------------------------------------------------------

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b_0_before = self.lowest_layer_head(imgs) # 16 3 192 256 -> 16 16 192 256
        input_for_head = []
        # pre_trained resnet
        x_u_0 = self.resnet.conv1(imgs) 
        x_u_0 = self.resnet.bn1(x_u_0)
        x_u_0 = self.resnet.relu(x_u_0) # 8 3 192 256 -> 8 64 96 128

        x_u_2 = self.resnet.maxpool(x_u_0) # 8 64 96 128 -> 8 64 48 64
        x_u_2 = self.resnet.layer1(x_u_2) # 8 64 48 64 -> 8 64 48 64
        x_u_2 = self.drop(x_u_2) # 8 64 48 64 -> 8 64 48 64
        input_for_head.append(x_u_2)

        x_u_1 = self.resnet.layer2(x_u_2) # 8 64 48 64 -> 8 128 24 32
        x_u_1 = self.drop(x_u_1) # 8 128 24 32 -> 8 128 24 32
        input_for_head.append(x_u_1)

        x_u = self.resnet.layer3(x_u_1) # 8 128 24 32 -> 8 256 12 16
        x_u = self.drop(x_u) # 8 256 12 16
        input_for_head.append(x_u)

        # top-down path
        x_skip_0   = self.layer_0(x_u_0) # 16 64 96 128 -> 16 32 96 128
        x_skip_0_1 = self.layer_0_1(x_skip_0) # 16 32 96 128 -> 16 48*64 64
        x_skip_0_1 = torch.transpose(x_skip_0_1, 1, 2).view(x_skip_0_1.shape[0], -1, (self.model_h // 4), (self.model_w // 4)) # 16 48*64 64 -> 16 64 48 64
        x_skip_1_1 = self.layer_1_1(x_skip_0_1) # 16 64 48 64 -> 16 24*32 128
        x_skip_1_1 = torch.transpose(x_skip_1_1, 1, 2).view(x_skip_1_1.shape[0], -1, (self.model_h // 8), (self.model_w // 8)) # 16 24*32 128 -> 16 128 24 32

        # bottleneck path
        x_h = self.head(input_for_head)
        x_h = self.toshow_bi(x_h)

        x_h_a = self.att_0(x_h)
        x_h_a = self.toshow_att(x_h_a)
        x_h_a = self.out_norm_0(x_h_a)
        
        x_c_a = self.att_1(x_h_a + x_u)
        x_c_a = self.out_norm_1(x_c_a)
        x_c   = self.out_gelu_0(self.skip_0_1(x_c_a, x_u) + x_c_a)

        # output
        x_out = self.out_1(x_c)
        x_out = F.interpolate(x_out, scale_factor=2, mode='bilinear') # 16 256 12 16 -> 16 128 24 32
        x_out_a = self.out_1_skip(x_out + x_skip_1_1) # x_skip_1_1 # x_b_1
        x_out = self.out_gelu_1(self.skip_1_1(x_out_a, x_u_1) + x_out_a)

        x_out_1 = self.out_2(x_out)
        x_out_1 = F.interpolate(x_out_1, scale_factor=2, mode='bilinear') # 16 128 24 32 -> 16 64 48 64
        x_out_1_a = self.out_2_skip(x_out_1 + x_skip_0_1) # +  # x_skip_0_1 # x_b_2
        x_out_1 = self.out_gelu_2(self.skip_2_1(x_out_1_a, x_u_2) + x_out_1_a)

        x_out_2 = self.out_3(x_out_1)
        x_out_2 = F.interpolate(x_out_2, scale_factor=2, mode='bilinear') # 16 64 48 64 -> 16 32 96 128
        x_out_2_a = self.out_3_skip(x_out_2 + x_skip_0)
        x_out_2 = self.out_gelu_3(self.skip_3(x_out_2_a, x_skip_0) + x_out_2_a)

        net_1_out = F.interpolate(self.final_net_1(x_out_2), scale_factor=2, mode='bilinear') # 16 32 96 128 -> 16 16 192 256
        #----------------------------------
        input_for_net2_att = self.split_att(net_1_out) # 16 16 192 256 -> 16 1 192 256
        input_for_net2_fea = x_b_0_before # 16 16 192 256 -> 16 16 192 256
        y_b_0_before = input_for_net2_fea * input_for_net2_att + input_for_net2_fea
        # NET2 downsample path
        y_for_head = []
        y_b_0 = self.down_path_0(y_b_0_before) # 16 16 192 256 -> 16 96*128 32
        y_b_0 = self.dropout(y_b_0)
        y_b_0 = torch.transpose(y_b_0, 1, 2).view(y_b_0.shape[0], -1, (self.model_h // 2), (self.model_w // 2)) # 16 96*128 32 -> 16 32 96*128 -> 16 32 96 128
        y_b_0 = self.toshow_p0(y_b_0)
        y_for_head.append(y_b_0)

        y_b_2_ = self.down_path_1(y_b_0) # 16 32 96 128 -> 16 48*64 64
        y_b_2_ = self.dropout(y_b_2_)
        y_b_2  = torch.transpose(y_b_2_, 1, 2).view(y_b_2_.shape[0], -1, (self.model_h // 4), (self.model_w // 4)) # 16 3072 64 -> 16 64 3072 -> 16 64 48 64
        y_b_2  = self.toshow_p1(y_b_2)
        y_for_head.append(y_b_2)

        y_b_1_ = self.down_path_2(y_b_2) # 16 64 48 64 -> 16 24*32 128
        y_b_1_ = self.dropout(y_b_1_) # 16 24*32 128 -> 16 24*32 128 (b n c)
        y_b_1  = torch.transpose(y_b_1_, 1, 2).view(y_b_1_.shape[0], -1, (self.model_h // 8), (self.model_w // 8)) # 16 24*32 128 -> 16 128 24*32 -> 16 128 24 32
        y_b_1  = self.toshow_p2(y_b_1)
        y_for_head.append(y_b_1)
        # NET2 bottleneck path
        y_h = self.head_2(y_for_head)
        y_h_a = self.out_norm_0_2(self.att_0_2(y_h))
        
        y_c_a = self.out_norm_1_2(self.att_1_2(y_h_a + y_b_1))
        y_c   = self.out_gelu_0(self.skip_0_1_2(y_c_a, y_b_1) + y_c_a) # 16 128 24 32
        # NET2 upsample path
        y_out_1 = self.out_2_2(y_c)
        y_out_1 = F.interpolate(y_out_1, scale_factor=2, mode='bilinear') # 16 128 24 32 -> 16 64 48 64
        y_out_1_a = self.out_2_2_skip(torch.cat([y_out_1, x_u_2], dim=1)) # self.out_2_2_skip(y_out_1 + x_u_2) 
        y_out_1 = self.skip_2_1_2(torch.cat([y_out_1_a, y_b_2], dim=1)) # self.out_gelu_2(self.skip_2_1_2(y_out_1_a, y_b_2) + y_out_1_a)

        y_out_2 = self.out_3_2(y_out_1)
        y_out_2 = F.interpolate(y_out_2, scale_factor=2, mode='bilinear') # 16 64 48 64 -> 16 32 96 128
        y_out_2_a = self.out_3_2_skip(torch.cat([y_out_2, x_skip_0], dim=1))
        y_out_2 = self.skip_3_1_2(torch.cat([y_out_2_a, y_b_0], dim=1))# self.out_gelu_3(self.skip_3_1_2(y_out_2_a, y_b_0) + y_out_2_a)

        net_2_out = F.interpolate(self.final_net_2(y_out_2), scale_factor=2, mode='bilinear') # 16 32 96 128 -> 16 16 192 256
        #----------------------------------
        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear') # 8 256 12 16 -> 8 14 192 256 
        map_1 = F.interpolate(self.final_1(x_u), scale_factor=16, mode='bilinear') # 8 384 12 16 -> 8 14 192 256 
        map_2 = self.final_2(torch.cat([net_1_out, net_2_out], dim=1)) # 16 32 192 256 -> 16 2 192 256
        
        return map_2

    def init_weights(self):
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final_net_1.apply(init_weights)
        self.final_net_2.apply(init_weights)
        self.head.apply(init_weights)
        self.head_2.apply(init_weights)
        self.att_0.apply(init_weights)
        self.att_0_2.apply(init_weights)
        self.att_1.apply(init_weights)
        self.att_1_2.apply(init_weights)
        self.lowest_layer_head.apply(init_weights)
        self.out_1.apply(init_weights)
        self.out_2.apply(init_weights)
        self.out_3.apply(init_weights)
        self.out_1_skip.apply(init_weights)
        self.out_2_skip.apply(init_weights)
        self.out_3_skip.apply(init_weights)
        self.out_2_2.apply(init_weights)
        self.out_3_2.apply(init_weights)
        self.out_2_2_skip.apply(init_weights)
        self.out_3_2_skip.apply(init_weights)
        self.skip_0_1.apply(init_weights)
        self.skip_1_1.apply(init_weights)
        self.skip_2_1.apply(init_weights)
        self.skip_3.apply(init_weights)
        self.skip_0_1_2.apply(init_weights)
        self.skip_2_1_2.apply(init_weights)
        self.skip_3_1_2.apply(init_weights)
        self.layer_0.apply(init_weights)
        self.layer_0_1.apply(init_weights)
        self.layer_1_1.apply(init_weights)
        self.split_att.apply(init_weights)
        self.down_path_0.apply(init_weights)
        self.down_path_1.apply(init_weights)
        self.down_path_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.GELU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

