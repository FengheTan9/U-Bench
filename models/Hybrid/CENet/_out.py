import torch
from torch import nn
from .modules.unet import (
    UnetResBlock,
    UnetUpBlock,
    UnetOutBlock,
)
import torch.nn.functional as F
from functools import partial
import math
from .decoders import DSEBlock, EUCB
from .modules.blocks import UpConv, UpTConv, UpRb


class OutHead(nn.Module):
    '''
    mode:
      - 1: dec->F.up-4x
      - 2: dec+w*rb_down4->F.up-4x
      - 3: dec->up-2x+w*rb_down2->F.up-2x
      - 4: dec->up-4x+w*rb->out
    '''
    def __init__(self,
        dec_in_channels, 
        x_in_channels,
        out_channels, 
        dec_in_spatial=56,
        x_in_spatial=224,
        out_mode=1,
        merge_mode='cat', # concat or add,
        up_block="eucb", # up or tconv
        up_ks=3,
        use_dseb=0, # 0: no, 1: yes
    ):
        super().__init__()
        self.out_mode = out_mode
        self.dec_in_channels = dec_in_channels
        self.x_in_channels = x_in_channels
        self.out_channels = out_channels
        self.dec_in_spatial = dec_in_spatial
        self.x_in_spatial = x_in_spatial
        self.out_mode = out_mode
        self.merge_mode = merge_mode
        self.use_dseb = use_dseb

        assert up_block in ["uprb", "eucb", "upcn", "uptc"], f"Invalid up_block: {up_block}"
        assert merge_mode in ["cat", "add"], f"Invalid merge_mode: {merge_mode}"
        assert out_mode in [1, 2, 3, 4], f"Invalid out_mode: {out_mode}"    

        norm_name = 'batch'
        act_name = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        rb_block = partial(UnetResBlock, spatial_dims=2, norm_name=norm_name, act_name=act_name, dropout=0)

        if out_mode in [1, 2]: # dec->F.up-4x | dec+w*rb_down4->F.up-4x
            om_chs = dec_in_channels
        elif out_mode == 3: # dec->up-2x+w*rb_down2->F.up-2x
            om_chs = dec_in_channels//2
        elif out_mode == 4: # dec->up-4x+w*rb->out
            om_chs = dec_in_channels//4
        else:
            raise ValueError(f"Invalid out_mode: {out_mode}")
        if out_mode > 1:
            self.w = nn.Parameter(torch.randn((1, om_chs, 1, 1))+0.75)

        if out_mode > 1:
            rb_down2x = lambda x_in, x_out: nn.Sequential(
                rb_block(in_channels=x_in, out_channels=x_out, kernel_size=5, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            mix_chns = om_chs if merge_mode=="add" else om_chs*2
            self.out = nn.Sequential(
                rb_block(in_channels=mix_chns, out_channels=mix_chns, kernel_size=3, stride=1),
                UnetOutBlock(spatial_dims=2, out_channels=out_channels, kernel_size=1, in_channels=mix_chns)
            )
            # self.out = rb_block(in_channels=mix_chns, out_channels=out_channels, kernel_size=3, stride=1)
            # self.out = nn.Conv2d(mix_chns, out_channels, kernel_size=1, stride=1)
        elif out_mode == 1:
            self.out = UnetOutBlock(spatial_dims=2, out_channels=out_channels, kernel_size=1, in_channels=om_chs)
            # self.out = rb_block(in_channels=om_chs, out_channels=out_channels, kernel_size=3, stride=1)

        if out_mode == 1: # dec->F.up-4x
            pass
        elif out_mode == 2: # dec+w*rb_down4->F.up-4x
            self.rb = nn.Sequential(rb_down2x(x_in_channels, om_chs//2), rb_down2x(om_chs//2, om_chs))
            if self.use_dseb:
                self.skip_enh = DSEBlock(dim=om_chs, scale_factors=[0.9, 0.5], num_heads=out_channels, input_size=56)
        elif out_mode == 3: # dec->up-2x+w*rb_down2->F.up-2x
            if up_block == "uprb":
                self.up = UpRb(in_channels=dec_in_channels, out_channels=om_chs, kernel_size=up_ks, scale_factor=2)
            elif up_block == "eucb":
                self.up = EUCB(in_channels=dec_in_channels, out_channels=om_chs, kernel_size=up_ks, stride=up_ks//2, activation='leakyrelu')
            elif up_block == "upcn":
                self.up = UpConv(in_channels=dec_in_channels, out_channels=om_chs, kernel_size=up_ks, stride=1, activation='leakyrelu')
            elif up_block == "uptc":
                self.up = UpTConv(in_channels=dec_in_channels, out_channels=om_chs, kernel_size=up_ks, stride=2, activation='leakyrelu')
            self.rb = rb_down2x(x_in_channels, om_chs)
            if self.use_dseb:
                self.skip_enh = DSEBlock(dim=om_chs, scale_factors=[0.8, 0.4], num_heads=out_channels, input_size=112)
        elif out_mode == 4: # dec->up-4x+w*rb->out
            if up_block == "uprb":
                self.up = UpRb(in_channels=dec_in_channels, out_channels=om_chs, kernel_size=up_ks+2, scale_factor=4)
            elif up_block == "eucb":
                self.up = nn.Sequential(EUCB(in_channels=dec_in_channels, out_channels=om_chs*2, kernel_size=up_ks, stride=up_ks//2),
                                        EUCB(in_channels=om_chs*2, out_channels=om_chs, kernel_size=up_ks, stride=up_ks//2))
            elif up_block == "upcn":
                self.up = nn.Sequential(UpConv(in_channels=dec_in_channels, out_channels=om_chs*2, kernel_size=up_ks, stride=1, activation='leakyrelu'),
                                        UpConv(in_channels=om_chs*2, out_channels=om_chs, kernel_size=up_ks, stride=1, activation='leakyrelu'))
            elif up_block == "uptc":
                self.up = nn.Sequential(UpTConv(in_channels=dec_in_channels, out_channels=om_chs*2, kernel_size=up_ks, stride=2, activation='leakyrelu'),
                                        UpTConv(in_channels=om_chs*2, out_channels=om_chs, kernel_size=up_ks, stride=2, activation='leakyrelu'))
            
            self.rb = rb_block(in_channels=x_in_channels, out_channels=om_chs, kernel_size=5, stride=1, dropout=0)
            if self.use_dseb:
                self.skip_enh = DSEBlock(dim=om_chs, scale_factors=[0.7, 0.35], num_heads=out_channels, input_size=224)

        if not use_dseb:
            self.skip_enh = None

    def merge(self, x, y):
        if self.merge_mode == 'cat':
            return torch.cat([x, y], dim=1)
        elif self.merge_mode == 'add':
            return x + y
        else:
            raise ValueError(f"Invalid merge_mode: {self.merge_mode}")
    
    def forward(self, dec, x):
        if self.out_mode == 1: # dec->F.up-4x
            y = self.out(dec)
            y = F.interpolate(y, scale_factor=4, mode='bilinear')
        elif self.out_mode == 2: # dec+w*rb_down4->F.up-4x
            rb_x = self.w*self.rb(x)
            if self.use_dseb: rb_x = self.skip_enh(rb_x, dec)
            z = self.merge(dec, rb_x)
            y = self.out(z)
            y = F.interpolate(y, scale_factor=4, mode='bilinear')
        elif self.out_mode == 3: # dec->up-2x+w*rb_down2->F.up-2x
            rb_x = self.w*self.rb(x)
            dec_ = self.up(dec)
            if self.use_dseb: rb_x = self.skip_enh(rb_x, dec_)
            z = self.merge(dec_, rb_x)
            y = self.out(z)
            y = F.interpolate(y, scale_factor=2, mode='bilinear')
        elif self.out_mode == 4: # dec->up-4x+w*rb->out
            rb_x = self.w*self.rb(x)
            dec_ = self.up(dec)
            if self.use_dseb: rb_x = self.skip_enh(rb_x, dec_)
            z = self.merge(dec_, rb_x)
            y = self.out(z)
        return y
