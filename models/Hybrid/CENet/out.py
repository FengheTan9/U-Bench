import torch
from torch import nn
from .modules.unet import UnetResBlock, UnetOutBlock
import torch.nn.functional as F
from functools import partial
from .decoders import EUCB
from .modules.blocks import UpConv, UpTConv, UpRb


class OutHead(nn.Module):
    # dec->up-2x+w*rb_down2->F.up-2x
    def __init__(self,
        dec_in_channels, 
        x_in_channels,
        out_channels, 
        dec_in_spatial=56,
        x_in_spatial=224,
        merge_mode='cat', # concat or add,
        up_block="upcn", # up or tconv
        up_ks=3,
    ):
        super().__init__()
        self.dec_in_channels = dec_in_channels
        self.x_in_channels = x_in_channels
        self.out_channels = out_channels
        self.dec_in_spatial = dec_in_spatial
        self.x_in_spatial = x_in_spatial
        self.merge_mode = merge_mode

        assert up_block in ["uprb", "eucb", "upcn", "uptc"], f"Invalid up_block: {up_block}"
        assert merge_mode in ["cat", "add"], f"Invalid merge_mode: {merge_mode}"

        norm_name = 'batch'
        act_name = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        rb_block = partial(UnetResBlock, spatial_dims=2, norm_name=norm_name, act_name=act_name, dropout=0)

        
        om_chs = dec_in_channels//2
        self.w = nn.Parameter(torch.randn((1, om_chs, 1, 1))+0.75)

        rb_down2x = lambda x_in, x_out: nn.Sequential(
            rb_block(in_channels=x_in, out_channels=x_out, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        mix_chns = om_chs if merge_mode=="add" else om_chs*2
        self.out = nn.Sequential(
            rb_block(in_channels=mix_chns, out_channels=mix_chns, kernel_size=3, stride=1),
            UnetOutBlock(spatial_dims=2, out_channels=out_channels, kernel_size=1, in_channels=mix_chns)
        )
        
        if up_block == "uprb":
            self.up = UpRb(in_channels=dec_in_channels, out_channels=om_chs, kernel_size=up_ks, scale_factor=2)
        elif up_block == "eucb":
            self.up = EUCB(in_channels=dec_in_channels, out_channels=om_chs, kernel_size=up_ks, stride=up_ks//2, activation='leakyrelu')
        elif up_block == "upcn":
            self.up = UpConv(in_channels=dec_in_channels, out_channels=om_chs, kernel_size=up_ks, stride=1, activation='leakyrelu')
        elif up_block == "uptc":
            self.up = UpTConv(in_channels=dec_in_channels, out_channels=om_chs, kernel_size=up_ks, stride=2, activation='leakyrelu')
        self.rb = rb_down2x(x_in_channels, om_chs)
        
    def merge(self, x, y):
        if self.merge_mode == 'cat':
            return torch.cat([x, y], dim=1)
        elif self.merge_mode == 'add':
            return x + y
        else:
            raise ValueError(f"Invalid merge_mode: {self.merge_mode}")
    
    def forward(self, dec, x):
        rb_x = self.w*self.rb(x)
        dec_ = self.up(dec)
        z = self.merge(dec_, rb_x)
        y = self.out(z)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        return y
