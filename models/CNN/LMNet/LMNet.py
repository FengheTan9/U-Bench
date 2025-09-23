import torch.nn as nn
import torch
from .modules import *

class LM_Net(nn.Module):
    def __init__(self, input_channel,num_calsses=2, filters=[12,24, 48, 96, 192],deep_supervision=False):
        super(LM_Net, self).__init__()
        self.deep_supervision=deep_supervision
        self.filters=filters

        self.conv1=nn.Sequential(ReparamConv(channel,filters[1],filters[0],5,3),
                                 ReparamConv(filters[0], filters[1], filters[0],5,3),
                                 )
        self.down1=nn.Sequential(
            nn.Conv2d(filters[0],filters[1],3,2,1),
        )

        self.conv2=nn.Sequential(ReparamConv(filters[1],filters[2],filters[1],5,3),
                                 ReparamConv(filters[1], filters[2], filters[1],5,3),
                                 )
        self.down2=nn.Sequential(
            nn.Conv2d(filters[1], filters[2], 3, 2,1),
             )

        self.conv3=nn.Sequential(ReparamConv(filters[2],filters[3],filters[2],5,3),
                                 ReparamConv(filters[2], filters[3], filters[2],5,3),

                                 )
        self.down3=nn.Sequential(
            nn.Conv2d(filters[2],filters[3],3,2,1),
        )

        self.conv4=nn.Sequential(ReparamConv(filters[3],filters[4],filters[3],5,3),
                                 ReparamConv(filters[3], filters[4], filters[3],5,3),
                                 )
        self.down4=nn.Sequential(
            nn.Conv2d(filters[3],filters[4],3,2,1),

        )
        self.dconv1=nn.Sequential(ReparamConv(filters[3],filters[4],filters[3],5,3),
                                 ReparamConv(filters[3], filters[4], filters[3],5,3),

                                 )
        self.dconv2=nn.Sequential(ReparamConv(filters[2],filters[3],filters[2],5,3),
                                 ReparamConv(filters[2], filters[3], filters[2],5,3),

                                 )
        self.dconv3=nn.Sequential(ReparamConv(filters[1],filters[2],filters[1],5,3),
                                 ReparamConv(filters[1], filters[2], filters[1],5,3),
                                 )
        self.dconv4=nn.Sequential(ReparamConv(filters[0],filters[1],filters[0],5,3),
                                 ReparamConv(filters[0], filters[1], filters[0],5,3),
                                 )

        self.pyramidpool=PyramidPool()
        self.gft=GFT(3, 14, sum(filters),  2, filters[4],1, 12)#多尺度GFT

        self.up1 = nn.Sequential(
                                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                                nn.Conv2d(filters[4], filters[3], 3,1,1)

        )
        self.up2 =nn.Sequential(
                      nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                      nn.Conv2d(filters[3], filters[2], 3,1,1),
                      )
        self.up3 =nn.Sequential(
                      nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                      nn.Conv2d(filters[2], filters[1], 3,1,1),
                      )
        self.up4 =nn.Sequential(
                      nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                      nn.Conv2d(filters[1], filters[0], 3,1,1),
                      )

        self.skip1 = M2Skip([filters[2],filters[3]],'bottom')
        self.skip2 = M3Skip([filters[1],filters[2],filters[3]])
        self.skip3 = M3Skip([filters[0],filters[1],filters[2]])
        self.skip4 = M2Skip([filters[0], filters[1]],'top')

        self.natt1 = NeighborhoodTransformer(3,32,filters[3],filters[3],1,[3,5],12)
        self.natt2 = NeighborhoodTransformer(3, 64, filters[2], filters[2], 1,[3,5],12)
        self.natt3 = NeighborhoodTransformer(3, 128, filters[1], filters[1], 1,[3,5],12)
        self.natt4 = NeighborhoodTransformer(3, 256, filters[0], filters[0], 1,[3,5],12)


        self.output_layer = nn.Conv2d(filters[0], num_calsses, 1)


    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()

    def forward(self, x):
        x1 = self.conv1(x)
        x_down1=self.down1(x1)
        x2 = self.conv2(x_down1)
        x_down2 = self.down2(x2)
        x3 = self.conv3(x_down2)
        x_down3 = self.down3(x3)
        x4 = self.conv4(x_down3)
        x_down4 = self.down4(x4)

        x5=self.gft(self.pyramidpool(x1,x2,x3,x4,x_down4))

        x_skip1 = self.skip1(x3,x4)
        x_skip2 = self.skip2(x2, x3,x4)
        x_skip3 = self.skip3(x1, x2,x3)
        x_skip4 = self.skip4(x1, x2)

        x46 = self.natt1(x_skip1)
        x37 = self.natt2(x_skip2)
        x28 = self.natt3(x_skip3)
        x19 = self.natt4(x_skip4)

        x6 = self.dconv1(self.up1(x5)+x46)
        x7 = self.dconv2(self.up2(x6) + x37)
        x8 = self.dconv3(self.up3(x7) + x28)
        x9 = self.dconv4(self.up4(x8) + x19)

        out = self.output_layer(x9)
        return out