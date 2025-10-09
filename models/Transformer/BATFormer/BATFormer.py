from .unets_parts import *
from .transformer_parts import TransformerDown
from .transformer_parts_mp import Transformer_block_global, Transformer_block_local, Transformer_block
from einops import rearrange, repeat
import torch.nn.functional as F


class C2FTrans(nn.Module):
    def __init__(self, global_block, local_block, layers, input_channel, num_classes, imgsize=256, patch_size=2, bilinear=True):
        super(C2FTrans, self).__init__()
        self.n_channels = input_channel
        self.n_classes = num_classes
        self.bilinear = bilinear
        self.scale = 4  # 1 2 4

        self.inc = DoubleConv(input_channel, 64 // self.scale)
        self.down1 = Down(64 // self.scale, 128 // self.scale)
        self.down2 = Down(128 // self.scale, 256 // self.scale)
        self.down3 = Down(256 // self.scale, 512 // self.scale)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 // self.scale, 1024 // factor // self.scale)

        self.up4 = Up(1024 // self.scale, 512 // factor // self.scale, bilinear)
        self.up3 = Up(512 // self.scale, 256 // factor // self.scale, bilinear)
        self.up2 = Up(256 // self.scale, 128 // factor // self.scale, bilinear)
        self.up1 = Up(128 // self.scale, 64 // self.scale, bilinear)

        self.softmax = nn.Softmax(dim=1)

        for p in self.parameters():
            p.requires_grad = True # set "True" manually in the first 350 epochs, then load the best model and set "False" manually in the following 50 epochs.

        self.trans_local2 = local_block(128 // self.scale // factor, 128 // self.scale // factor * 2, imgsize // 2, 1, heads=6, patch_size=1, n_classes=num_classes, win_size=16)
        self.trans_global = global_block(256 // factor // self.scale, 256 // factor // self.scale * 2, imgsize // 4, 1, heads=4, patch_size=1)

        self.outc1 = OutConv(64 // self.scale * 4, num_classes)
        self.convl1 = nn.Conv2d(64 // self.scale, 64 // self.scale * 4, kernel_size=1, padding=0, bias=False)
        self.outc2 = OutConv(64 // self.scale * 4, num_classes)

        self.convl2 = nn.Conv2d(128//factor//self.scale*2, 64//self.scale*4, kernel_size=1, padding=0, bias=False)
        #self.convl2 = nn.Conv2d(128 // factor // self.scale, 64 // self.scale * 4, kernel_size=1, padding=0, bias=False)

        self.outc3 = OutConv(64 // self.scale * 4, num_classes)
        self.convl3 = nn.Conv2d(256//factor//self.scale*2, 64//self.scale*4, kernel_size=1, padding=0, bias=False)
        #self.convl3 = nn.Conv2d(256 // factor // self.scale, 64 // self.scale * 4, kernel_size=1, padding=0, bias=False)
        self.out = OutConv(3 * 64 // self.scale * 4, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d4 = self.up4(x5, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)

        trans_global = self.trans_global(x5, d4, d3)
        l3 = self.convl3(trans_global)
        #l3 = self.convl3(d3)

        pred3 = self.outc3(l3)  # b c h w
        l3_up = l3[:, :, :, :, None].repeat(1, 1, 1, 1, 16)
        l3_up = rearrange(l3_up, 'b c h w (m n) -> b c (h m) (w n)', m=4, n=4)
        #pred3_up = pred3[:, :, :, :, None].repeat(1, 1, 1, 1, 16)
        #pred3_up = rearrange(pred3_up, 'b c h w (m n) -> b c (h m) (w n)', m=4, n=4)

        pred3_p = self.softmax(pred3)
        trans_local2 = self.trans_local2(d2, pred3_p)
        l2 = self.convl2(trans_local2)
        #l2 = self.convl2(d2)

        pred2 = self.outc2(l2)  # b c h w
        l2_up = l2[:, :, :, :, None].repeat(1, 1, 1, 1, 4)
        l2_up = rearrange(l2_up, 'b c h w (m n) -> b c (h m) (w n)', m=2, n=2)
        #pred2_up = pred2[:, :, :, :, None].repeat(1, 1, 1, 1, 4)
        #pred2_up = rearrange(pred2_up, 'b c h w (m n) -> b c (h m) (w n)', m=2, n=2)

        l1 = self.convl1(d1)
        pred1 = self.outc1(l1)  # b c h w

        predf = torch.cat((l1, l2_up, l3_up), dim=1)
        predf = self.out(predf)
        
        
        pred2 = F.interpolate(pred2, scale_factor=2, mode='bilinear')
        pred3 = F.interpolate(pred3, scale_factor=4, mode='bilinear')
        #print(pred1.shape, pred2.shape, pred3.shape, predf.shape)
        return pred1, pred2, pred3, predf


def batformer(num_classes, input_channel=3):
    model = C2FTrans(Transformer_block_global, Transformer_block_local, [1, 1, 1, 1], num_classes=num_classes, input_channel=input_channel)
    return model
