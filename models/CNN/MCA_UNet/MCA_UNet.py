# -*- coding: utf-8 -*-
# @Time    : 2021/3/17 10:47 下午
# @Author  : Haonan Wang
# @File    : SN_msf_updec_sum_net.py
# @Software: PyCharm

import os

import torch
from torch import nn
from .layers import SN_CS_Parallel_Attention_block, SN_PostRes2d
from torch import optim
from torch.backends import cudnn
from torch.nn.functional import interpolate
from .layers import SwitchNorm2d


class CAUNet(nn.Module):
    """
    dsunet2d, 残差块数量为2234，并默认的各少一个
    """

    def __init__(self, input_channel=3,num_classes=1, bn_momentum=0.2):
        super().__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.original_size = 256
        self.preBlock = nn.Sequential(
            nn.Conv2d(input_channel, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.Conv2d(24, 24, kernel_size=3, padding=1),
            # SwitchNorm2d(24, momentum=bn_momentum),
            # nn.ReLU(inplace=True)
        )

        self.outBlock = nn.Sequential(

            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.pred = nn.Sequential(
            # nn.Conv2d(24, 24, kernel_size=3, padding=1),
            # SwitchNorm2d(24, momentum=bn_momentum),
            # nn.ReLU(inplace=True),
            nn.Conv2d(24, 2, kernel_size=1),
            SwitchNorm2d(2, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, num_classes, kernel_size=1),
            # nn.Sigmoid()
        )

        num_blocks_forw = [2, 2, 3, 4]
        self.featureNum_forw = [24, 32, 64, 128, 256]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(SN_PostRes2d(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(SN_PostRes2d(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # num_blocks_back = [3, 2, 2]
        # self.featureNum_back = [256, 128, 64, 32]
        # for i in range(len(num_blocks_back)):
        #     blocks = []
        #     for j in range(num_blocks_back[i]):
        #         if j == 0:
        #             blocks.append(PostRes2d(self.featureNum_back[i], self.featureNum_back[i + 1]))
        #         else:
        #             blocks.append(PostRes2d(self.featureNum_back[i + 1], self.featureNum_back[i + 1]))
        #     setattr(self, 'back' + str(i + 1), nn.Sequential(*blocks))


        self.back1 = nn.Sequential(
            SN_PostRes2d(504, 128),
            # SN_PostRes2d(256, 128),
            SN_PostRes2d(128, 128),
        )
        self.back2 = nn.Sequential(
            SN_PostRes2d(312, 64),
            # PostRes2d(128, 64),
            SN_PostRes2d(64, 64),
        )
        self.back3 = nn.Sequential(
            SN_PostRes2d(216, 32),
            # PostRes2d(96, 32),
            SN_PostRes2d(32, 32),
        )

        self.back4 = nn.Sequential(
            SN_PostRes2d(192, 48),
            SN_PostRes2d(48, 24),
            SN_PostRes2d(24, 24),
        )


        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            SwitchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            SwitchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            SwitchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(64, 64))
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(128, 128))

        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            SwitchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(256, 256))
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=3),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(512, 512)),
        )



        # output layers
        self.out_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, num_classes, kernel_size=8, stride=8),
            # nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1),
            # nn.Sigmoid()
        )
        self.out_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=4),
            # nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            # nn.Sigmoid()
        )
        self.out_conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1),
            # nn.Sigmoid()
        )

        # ================================================================================================
        # Fully Connected Layers
        # ================================================================================================
        self.pool_preblock_gate1 =nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_preblock_gate2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool_preblock_gate3 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool_out1_gate2 =nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_out1_gate3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool_out2_gate3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_out1_gate0 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out2_gate0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            SwitchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out2_gate1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            SwitchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate0 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=8, stride=8),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))


        self.Fint_num = [128, 64, 32, 24]
        self.att3_1 = SN_CS_Parallel_Attention_block(F_g=128, F_x=24, F_int=self.Fint_num[0])
        self.att3_2 = SN_CS_Parallel_Attention_block(F_g=128, F_x=32, F_int=self.Fint_num[0])
        self.att3_3 = SN_CS_Parallel_Attention_block(F_g=128, F_x=64, F_int=self.Fint_num[0])
        self.att3_4 = SN_CS_Parallel_Attention_block(F_g=128, F_x=128, F_int=self.Fint_num[0])

        self.att2_1 = SN_CS_Parallel_Attention_block(F_g=64, F_x=24, F_int=self.Fint_num[1])
        self.att2_2 = SN_CS_Parallel_Attention_block(F_g=64, F_x=32, F_int=self.Fint_num[1])
        self.att2_3 = SN_CS_Parallel_Attention_block(F_g=64, F_x=64, F_int=self.Fint_num[1])
        self.att2_4 = SN_CS_Parallel_Attention_block(F_g=64, F_x=64, F_int=self.Fint_num[1])

        self.att1_1 = SN_CS_Parallel_Attention_block(F_g=32, F_x=24, F_int=self.Fint_num[2])
        self.att1_2 = SN_CS_Parallel_Attention_block(F_g=32, F_x=32, F_int=self.Fint_num[2])
        self.att1_3 = SN_CS_Parallel_Attention_block(F_g=32, F_x=32, F_int=self.Fint_num[2])
        self.att1_4 = SN_CS_Parallel_Attention_block(F_g=32, F_x=64, F_int=self.Fint_num[2])

        self.att0_1 = SN_CS_Parallel_Attention_block(F_g=24, F_x=24, F_int=self.Fint_num[3])
        self.att0_2 = SN_CS_Parallel_Attention_block(F_g=24, F_x=24, F_int=self.Fint_num[3])
        self.att0_3 = SN_CS_Parallel_Attention_block(F_g=24, F_x=32, F_int=self.Fint_num[3])
        self.att0_4 = SN_CS_Parallel_Attention_block(F_g=24, F_x=64, F_int=self.Fint_num[3])

    def forward(self, x):
        preblock = self.preBlock(x)  # 24
        out_pool, indices0 = self.maxpool1(preblock)

        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool2(out1)

        out2 = self.forw2(out1_pool)  # 64
        out2_pool, indices2 = self.maxpool3(out2)

        out3 = self.forw3(out2_pool)  # 128
        out3_pool, indices3 = self.maxpool4(out3)

        out4 = self.forw4(out3_pool)  # 256

        # ============================================================================
        # Fully Connected Modules
        # ============================================================================

        # ---------Gate0--------------
        out1_gate0 = self.up_out1_gate0(out1)  # 24
        out2_gate0 = self.up_out2_gate0(out2)  # 32
        out3_gate0 = self.up_out3_gate0(out3)  # 64
        # out4_gate0 = self.up_out4_gate0(out4)


        # ---------Gate1--------------
        preblock_gate1 = self.pool_preblock_gate1(preblock)  # 24
        out2_gate1 = self.up_out2_gate1(out2)  # 32
        out3_gate1 = self.up_out3_gate1(out3)  # 64


        # ---------Gate2--------------
        preblock_gate2 = self.pool_preblock_gate2(preblock)  # 24
        out1_gate2 = self.pool_out1_gate2(out1)  # 32
        out3_gate2 = self.up_out3_gate2(out3)  # 64
        # out4_gate2 = self.up_out4_gate2(out4)


        # ---------Gate3--------------
        preblock_gate3 = self.pool_preblock_gate3(preblock)  # 24
        out1_gate3 = self.pool_out1_gate3(out1)  # 32
        out2_gate3 = self.pool_out2_gate3(out2)  # 64
        # out4_gate3 = self.up_out4_gate3(out4)


        up4 = self.up4(out4)  # 128
        # up4 = self.up4(out4)
        up4 = nn.functional.interpolate(up4, size=(self.original_size//8, self.original_size//8), mode='nearest')
        deconv4 = self.deconv4(out4)
        up4_sig = up4 + deconv4
        up4 = torch.cat((up4_sig, deconv4), dim=1)
        preblock_gate3 = self.att3_1(g=up4_sig, x=preblock_gate3)
        out1_gate3 = self.att3_2(g=up4_sig, x=out1_gate3)
        out2_gate3 = self.att3_3(g=up4_sig, x=out2_gate3)
        out3 = self.att3_4(g=up4_sig, x=out3)
        # gate3_fusion = torch.cat((preblock_gate3, out1_gate3, out2_gate3, out3), dim=1)  # 24+32+64+128=248
        # gate3_back = self.gate3_back(gate3_fusion)  # 128
        gate3 = torch.cat((up4, preblock_gate3, out1_gate3, out2_gate3, out3), dim=1)  # 128 + 128 = 256
        # gate3 = self.em_att_unit3(comb3)
        comb3 = self.back1(gate3)


        up3 = self.up3(comb3)
        up3 = nn.functional.interpolate(up3, size=(self.original_size//4, self.original_size//4), mode='nearest')
        deconv3 = self.deconv3(comb3)
        up3_sig = up3 + deconv3
        up3 = torch.cat((up3, deconv3), dim=1)
        preblock_gate2 = self.att2_1(g=up3_sig, x=preblock_gate2)
        out1_gate2 = self.att2_2(g=up3_sig, x=out1_gate2)
        out2 = self.att2_3(g=up3_sig, x=out2)
        out3_gate2 = self.att2_4(g=up3_sig, x=out3_gate2)
        # gate2_fusion = torch.cat((preblock_gate2, out1_gate2, out2, out3_gate2), dim=1)  # 24+32+64+64=184
        # gate2_back = self.gate2_back(gate2_fusion)  # 64
        gate2 = torch.cat((up3, preblock_gate2, out1_gate2, out2, out3_gate2), dim=1)  # 64 + 64 = 128
        # gate2 = self.em_att_unit2(comb2)
        comb2 = self.back2(gate2)


        up2 = self.up2(comb2)  # 32
        up2 = nn.functional.interpolate(up2, size=(self.original_size//2, self.original_size//2), mode='nearest')
        deconv2 = self.deconv2(comb2)
        up2_sig = up2 + deconv2
        up2 = torch.cat((up2, deconv2), dim=1)
        preblock_gate1 = self.att1_1(g=up2_sig, x=preblock_gate1)
        out1 = self.att1_2(g=up2_sig, x=out1)
        out2_gate1 = self.att1_3(g=up2_sig, x=out2_gate1)
        out3_gate1 = self.att1_4(g=up2_sig, x=out3_gate1)
        # gate1_fusion = torch.cat((preblock_gate1, out1, out2_gate1, out3_gate1), dim=1)  # 24+32+32+64=152
        # gate1_back = self.gate1_back(gate1_fusion)  # 32
        gate1 = torch.cat((up2, preblock_gate1, out1, out2_gate1, out3_gate1), dim=1)  # 32 + 32 = 64
        # gate1 = self.em_att_unit1(comb1)
        comb1 = self.back3(gate1)  # 32

        up1 = self.up1(comb1)  # 24
        up1 = nn.functional.interpolate(up1, size=(self.original_size, self.original_size), mode='nearest')
        deconv1 = self.deconv1(comb1)
        up1_sig = up1 + deconv1
        up1 = torch.cat((up1, deconv1), dim=1)
        preblock = self.att0_1(g=up1_sig, x=preblock)
        out1_gate0 = self.att0_2(g=up1_sig, x=out1_gate0)
        out2_gate0 = self.att0_3(g=up1_sig, x=out2_gate0)
        out3_gate0 = self.att0_4(g=up1_sig, x=out3_gate0)
        gate0 = torch.cat((up1, preblock, out1_gate0, out2_gate0, out3_gate0), dim=1)  # 24 + 24 = 48
        # gate0 = self.em_att_unit0(out)
        out = self.back4(gate0)  # 48
        # out = self.outBlock(out)  # 24
        out = self.pred(out)
        out3 = self.out_conv1(comb3)
        out2 = self.out_conv2(comb2)
        out1 = self.out_conv3(comb1)
        # return out, out3, out2, out1  # 640, 80, 160, 320
        # return out
        return out3,out2,out1,out


def mca_unet(input_channel=3,num_classes=1):
    return CAUNet(num_classes=num_classes,input_channel=input_channel)