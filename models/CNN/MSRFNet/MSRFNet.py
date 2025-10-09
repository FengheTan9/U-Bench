import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


# BLOCKS to construct the model
class DSDF_block(nn.Module):  #OK
    def __init__(self, in_ch_x, in_ch_y, nf1=128, nf2=256, gc=64, bias=True):
        super().__init__()

        self.nx1 = nn.Sequential(nn.Conv2d(in_ch_x, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.ny1 = nn.Sequential(nn.Conv2d(in_ch_y, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.nx1c = nn.Sequential(nn.Conv2d(in_ch_x, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),    # ks 3 -> 4, stride 1 -> 2
                                  nn.LeakyReLU(negative_slope=0.25))

        self.ny1t = nn.Sequential(nn.ConvTranspose2d(in_ch_y, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),   # ks 3 -> 4
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx2 = nn.Sequential(nn.Conv2d(in_ch_x+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                  nn.LeakyReLU(negative_slope=0.25))

        self.ny2 = nn.Sequential(nn.Conv2d(in_ch_y+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx2c = nn.Sequential(nn.Conv2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),    # ks 3 -> 4, stride 1 -> 2
                                  nn.LeakyReLU(negative_slope=0.25))

        self.ny2t = nn.Sequential(nn.ConvTranspose2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),   # ks 3 -> 4
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx3 = nn.Sequential(nn.Conv2d(in_ch_x+gc+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.ny3 = nn.Sequential(nn.Conv2d(in_ch_y+gc+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.nx3c = nn.Sequential(nn.Conv2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),    # ks 3 -> 4, stride 1 -> 2
                                  nn.LeakyReLU(negative_slope=0.25))

        self.ny3t = nn.Sequential(nn.ConvTranspose2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),   # ks 3 -> 4
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx4 = nn.Sequential(nn.Conv2d(in_ch_x+gc+gc+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.ny4 = nn.Sequential(nn.Conv2d(in_ch_y+gc+gc+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.nx4c = nn.Sequential(nn.Conv2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),    # ks 3 -> 4, stride 1 -> 2
                                  nn.LeakyReLU(negative_slope=0.25))

        self.ny4t = nn.Sequential(nn.ConvTranspose2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),   # ks 3 -> 4
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx5 = nn.Sequential(nn.Conv2d(in_ch_x+gc+gc+gc+gc+gc, nf1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.ny5 = nn.Sequential(nn.Conv2d(in_ch_y+gc+gc+gc+gc+gc, nf2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

    def forward(self, x, y):

        x1 = self.nx1(x)
        y1 = self.ny1(y)

        x1c = self.nx1c(x)
        y1t = self.ny1t(y)

        x2_input = torch.cat([x, x1, y1t], dim=1)
        x2 = self.nx2(x2_input)

        y2_input = torch.cat([y, y1, x1c], dim=1)
        y2 = self.ny2(y2_input)

        x2c = self.nx2c(x1)
        y2t = self.ny2t(y1)

        x3_input = torch.cat([x, x1, x2, y2t], dim=1)
        x3 = self.nx3(x3_input)

        y3_input = torch.cat([y, y1, y2, x2c], dim=1)
        y3 = self.ny3(y3_input)

        x3c = self.nx3c(x3)
        y3t = self.ny3t(y3)

        x4_input = torch.cat([x, x1, x2, x3, y3t], dim=1)
        x4 = self.nx4(x4_input)

        y4_input = torch.cat([y, y1, y2, y3, x3c], dim=1)
        y4 = self.ny4(y4_input)

        x4c = self.nx4c(x4)
        y4t = self.ny4t(y4)

        x5_input = torch.cat([x, x1, x2, x3, x4, y4t], dim=1)
        x5 = self.nx5(x5_input)

        y5_input = torch.cat([y, y1, y2, y3, y4, x4c], dim=1)
        y5 = self.ny5(y5_input)

        x5 *= 0.4
        y5 *= 0.4

        return x5+x, y5+y


class ATTENTION_block(nn.Module):  #OK
    def __init__(self, in_ch_x, in_ch_g, med_ch):
        super().__init__()
        self.theta = nn.Conv2d(in_ch_x, med_ch, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True)
        self.phi = nn.Conv2d(in_ch_g, med_ch, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.block = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(med_ch, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
                                   nn.Sigmoid(),
                                   nn.ConvTranspose2d(1, 1, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True))
        self.batchnorm = nn.BatchNorm2d(in_ch_x)

    def forward(self, x, g):
        theta = self.theta(x) + self.phi(g)
        out = self.batchnorm(self.block(theta) * x)
        return out


class UP_block(nn.Module):  #OK
    def __init__(self, input_1_ch, input_2_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_2_ch, input_1_ch, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True)

    def forward(self, input_1, input_2):
        x = torch.cat([self.up(input_2), input_1], dim=1)
        return x


class SE_block(nn.Module):   #OK
    def __init__(self, in_ch, ratio=16):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(in_ch, in_ch//ratio),
                                   nn.ReLU(),
                                   nn.Linear(in_ch//ratio, in_ch),
                                   nn.Sigmoid())
    def forward(self, x):
        y = x.mean((-2, -1))
        y = self.block(y).unsqueeze(-1).unsqueeze(-1)
        return x*y


class SPATIALATT_block(nn.Module):    #OK
    def __init__(self, in_ch, med_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, med_ch, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
                                   nn.BatchNorm2d(med_ch),
                                   nn.ReLU(),
                                   nn.Conv2d(med_ch, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
                                   nn.Sigmoid())
    def forward(self, x):
        x = self.block(x)

        return x


class RES_block(nn.Module):    #OK
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
                                   nn.BatchNorm2d(in_ch),
                                   nn.ReLU(),
                                   nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=True),
                                   nn.BatchNorm2d(in_ch))
        self.act = nn.ReLU()

    def forward(self, x):
        res = self.block(x)
        out = self.act(res+x)

        return out


class DUALATT_block(nn.Module):    #OK
    def __init__(self, skip_in_ch, prev_in_ch, out_ch):
        super().__init__()
        self.prev_block = nn.Sequential(nn.ConvTranspose2d(prev_in_ch, out_ch, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU())
        self.block = nn.Sequential(nn.Conv2d(skip_in_ch+out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU())
        self.se_block = SE_block(out_ch, ratio=16)
        self.spatial_att = SPATIALATT_block(out_ch, out_ch)

    def forward(self, skip, prev):

        prev = self.prev_block(prev)
        x = torch.cat([skip, prev], dim=1)
        inpt_layer = self.block(x)
        se_out = self.se_block(inpt_layer)
        sab = self.spatial_att(inpt_layer) + 1

        return sab*se_out


class GSC_block(nn.Module):
    def __init__(self, in_ch_x, in_ch_y):
        super().__init__()
        self.block = nn.Sequential(nn.BatchNorm2d(in_ch_x+in_ch_y),
                                   nn.Conv2d(in_ch_x+in_ch_y, in_ch_x+in_ch_y+1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),  #in_ch->out_ch
                                   nn.ReLU(),
                                   nn.Conv2d(in_ch_x+in_ch_y+1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                   nn.BatchNorm2d(1),
                                   nn.Sigmoid())

    def forward(self, x, y):
        inpt = torch.cat([x, y], dim=1)
        inpt = self.block(inpt)

        return inpt

## SHAPE-STREAM
# https://github.com/leftthomas/GatedSCNN/blob/master/model.py

class GatedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1, bias=False)
        self.attention = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, feat, gate):
        attention = self.attention(torch.cat((feat, gate), dim=1))
        out = F.conv2d(feat * (attention + 1), self.weight)
        return out

class ShapeStream(nn.Module):
    def __init__(self, init_feat, num_classes):
        super().__init__()
        self.res2_conv = nn.Conv2d(init_feat * 2, 1, 1)
        self.res3_conv = nn.Conv2d(init_feat * 4, 1, 1)
        self.res4_conv = nn.Conv2d(init_feat * 8, 1, 1)
        self.res1 = BasicBlock(init_feat, init_feat, 1)
        self.res2 = BasicBlock(32, 32, 1)
        self.res3 = BasicBlock(16, 16, 1)
        self.res1_pre = nn.Conv2d(init_feat, 32, 1)
        self.res2_pre = nn.Conv2d(32, 16, 1)
        self.res3_pre = nn.Conv2d(16, 8, 1)
        self.gate1 = GatedConv(32, 32)
        self.gate2 = GatedConv(16, 16)
        self.gate3 = GatedConv(8, 8)
        self.gate = nn.Conv2d(8, num_classes, 1, bias=False)
        self.fuse = nn.Conv2d(3 + num_classes, 1, 1, bias=False)

    def forward(self, x, res2, res3, res4, grad):  #def forward(self, x, res2, res3, res4, grad):
        size = grad.size()[-2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        res2 = F.interpolate(self.res2_conv(res2), size, mode='bilinear', align_corners=True)
        res3 = F.interpolate(self.res3_conv(res3), size, mode='bilinear', align_corners=True)
        res4 = F.interpolate(self.res4_conv(res4), size, mode='bilinear', align_corners=True)
        gate1 = self.gate1(self.res1_pre(self.res1(x)), res2)
        gate2 = self.gate2(self.res2_pre(self.res2(gate1)), res3)
        gate3 = self.gate3(self.res3_pre(self.res3(gate2)), res4)
        gate = torch.sigmoid(self.gate(gate3))
        #gate = torch.sigmoid(self.gate(gate2))
        feat = torch.sigmoid(self.fuse(torch.cat((gate, grad), dim=1)))
        return gate, feat



# MODEL  (NOT CHECKED)
class MSRF(nn.Module):
    def __init__(self, input_channel, num_classes, init_feat=32):
        super().__init__()

        # ENCODER ----------------------------
        self.n11 = nn.Sequential(nn.Conv2d(input_channel, init_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReLU(),
                                 nn.Conv2d(init_feat, init_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(init_feat),
                                 SE_block(init_feat, ratio=init_feat//2)
                                 )

        self.n21 = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2)),
                                 nn.Dropout(0.2),
                                 nn.Conv2d(init_feat, init_feat*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReLU(),
                                 nn.Conv2d(init_feat*2, init_feat*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(init_feat*2),
                                 SE_block(init_feat*2, ratio=init_feat//2))

        self.n31 = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2)),
                                 nn.Dropout(0.2),
                                 nn.Conv2d(init_feat*2, init_feat*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReLU(),
                                 nn.Conv2d(init_feat*4, init_feat*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(init_feat*4),
                                 SE_block(init_feat*4, ratio=init_feat//2))

        self.n41 = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2)),
                                 nn.Dropout(0.2),
                                 nn.Conv2d(init_feat*4, init_feat*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReLU(),
                                 nn.Conv2d(init_feat*8, init_feat*8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(init_feat*8))
        # MSRF-subnetwork ----------------------------
        self.dsfs_1  = DSDF_block(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_2  = DSDF_block(init_feat*4, init_feat*8, nf1=init_feat*4, nf2=init_feat*8, gc=init_feat*4//2)
        self.dsfs_3  = DSDF_block(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_4  = DSDF_block(init_feat*4, init_feat*8, nf1=init_feat*4, nf2=init_feat*8, gc=init_feat*4//2)
        self.dsfs_5  = DSDF_block(init_feat*2, init_feat*4, nf1=init_feat*2, nf2=init_feat*4, gc=init_feat*2//2)
        self.dsfs_6  = DSDF_block(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_7  = DSDF_block(init_feat*4, init_feat*8, nf1=init_feat*4, nf2=init_feat*8, gc=init_feat*4//2)
        self.dsfs_8  = DSDF_block(init_feat*2, init_feat*4, nf1=init_feat*2, nf2=init_feat*4, gc=init_feat*2//2)
        self.dsfs_9  = DSDF_block(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_10 = DSDF_block(init_feat*4, init_feat*8, nf1=init_feat*4, nf2=init_feat*8, gc=init_feat*4//2)

        # SHAPE STREAM ------------IN PROGRESS-------------------
        self.shape_stream = ShapeStream(init_feat, num_classes)

        # DECODER
        # Stage 1:
        self.att_1 = ATTENTION_block(init_feat*4, init_feat*8, init_feat*8)
        self.up_1 = UP_block(init_feat*4, init_feat*8)
        self.dualatt_1 = DUALATT_block(init_feat*4, init_feat*8, init_feat*4)
        self.n34_t = nn.Conv2d(init_feat * 4 + init_feat * 8, init_feat * 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.dec_block_1 = nn.Sequential(nn.BatchNorm2d(init_feat*4),
                                         nn.ReLU(),
                                         nn.Conv2d(init_feat*4, init_feat*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                         nn.BatchNorm2d(init_feat*4),
                                         nn.ReLU(),
                                         nn.Conv2d(init_feat*4, init_feat*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                         )
        self.head_dec_1 = nn.Sequential(nn.Conv2d(init_feat*4, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                        nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))

        # Stage 2:
        self.att_2 = ATTENTION_block(init_feat * 2, init_feat * 4, init_feat * 2)
        self.up_2 = UP_block(init_feat * 2, init_feat * 4)
        self.dualatt_2 = DUALATT_block(init_feat * 2, init_feat * 4, init_feat * 2)
        self.n24_t = nn.Conv2d(init_feat * 2 + init_feat * 4, init_feat * 2, kernel_size=(1, 1), stride=(1, 1), padding=(0,0))
        self.dec_block_2 = nn.Sequential(nn.BatchNorm2d(init_feat * 2),
                                         nn.ReLU(),
                                         nn.Conv2d(init_feat * 2, init_feat * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                         nn.BatchNorm2d(init_feat * 2),
                                         nn.ReLU(),
                                         nn.Conv2d(init_feat*2, init_feat * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                         )
        self.head_dec_2 = nn.Sequential(nn.Conv2d(init_feat * 2, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        # Stage 3:
        self.up_3 = nn.ConvTranspose2d(init_feat * 2, init_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.n14_input = nn.Sequential(nn.Conv2d(init_feat + init_feat + 1, init_feat, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                       nn.ReLU())
        self.dec_block_3 = nn.Sequential(nn.Conv2d(init_feat, init_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(init_feat))

        self.head_dec_3 = nn.Sequential(nn.Conv2d(init_feat, init_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.ReLU(),
                                        nn.Conv2d(init_feat, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))

    def forward(self, x):
        # ENCODER:
        canny = x
        x11 = self.n11(x)
        x21 = self.n21(x11)
        x31 = self.n31(x21)
        x41 = self.n41(x31)

        # MSRF-subnetwork
        x12, x22 = self.dsfs_1(x11, x21)
        x32, x42 = self.dsfs_2(x31, x41)
        x12, x22 = self.dsfs_3(x12, x22)
        x32, x42 = self.dsfs_4(x32, x42)
        x22, x32 = self.dsfs_5(x22, x32)
        x13, x23 = self.dsfs_6(x12, x22)
        x33, x43 = self.dsfs_7(x32, x42)
        x23, x33 = self.dsfs_8(x23, x33)
        x13, x23 = self.dsfs_9(x13, x23)
        x33, x43 = self.dsfs_10(x33, x43)

        x13 = (x13*0.4) + x11
        x23 = (x23*0.4) + x21
        x33 = (x33*0.4) + x31
        x43 = (x43*0.4) + x41

        # SHAPE STREAM
        # https://github.com/leftthomas/GatedSCNN (https://arxiv.org/pdf/1907.05740.pdf)
        canny_gate, canny_feat = self.shape_stream(x13, x23, x33, x43, canny)

        # DECODER
        # Stage 1:
        x34_preinput = self.att_1(x33, x43)

        x34 = self.up_1(x34_preinput, x43)
        x34_t = self.dualatt_1(x33, x43)
        x34_t = torch.cat([x34, x34_t], dim=1)
        x34_t = self.n34_t(x34_t)
        x34 = self.dec_block_1(x34_t) + x34_t

        pred_1 = self.head_dec_1(x34)

        # Stage 2:
        x24_preinput = self.att_2(x23, x34)
        x24 = self.up_2(x24_preinput, x34)
        x24_t = self.dualatt_2(x23, x34)
        x24_t = torch.cat([x24, x24_t], dim=1)
        x24_t = self.n24_t(x24_t)
        x24 = self.dec_block_2(x24_t) + x24_t

        pred_2 = self.head_dec_2(x24)

        # Stage 3:
        x14_preinput = self.up_3(x24)
        x14_input = torch.cat([x14_preinput, x13, canny_feat], dim=1)
        x14_input = self.n14_input(x14_input)
        x14 = self.dec_block_3(x14_input)
        x14 = x14 + x14_input
        pred_3 = self.head_dec_3(x14)

        return canny_gate, pred_1, pred_2, pred_3

def msrfnet(num_classes, input_channel=3):
    model = MSRF(input_channel=input_channel, num_classes=num_classes)
    return model

