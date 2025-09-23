import torch
import torch.nn as nn
from .decoders import Decoder
from .encoder import get_encoder2d
from .out import OutHead


class CENet(nn.Module):
    def __init__(self, 
        input_channels=1,
        num_classes=1, 
        scale_factors=[0.8, 0.4], 
        diffatt_num_heads=[2,2,2], 
        encoder='pvt_v2_b2', enc_pretrain=False, freeze_bb=False, 
        skip_mode="cat", 
        dec_up_block='eucb',
        out_merge_mode="cat", # cat or add
        out_up_block="eucb", # up or tconv
        out_up_ks=3, # [3 or 1] -> upsample kernel size
        writer=None,
        base_ptdir='.'
    ):
        super().__init__()
        self.writer = writer

        self.backbone, channels = get_encoder2d(
            input_channels=input_channels, 
            encoder=encoder, 
            pretrain=enc_pretrain, 
            freeze_bb=freeze_bb,
            base_ptdir=base_ptdir
        )

        self.decoder = Decoder(
            channels=channels, 
            scale_factors=scale_factors,
            skip_mode=skip_mode,
            num_heads=diffatt_num_heads,
            up_block=dec_up_block,
            writer=writer)
        
        self.out = OutHead(
            dec_in_spatial=64,
            dec_in_channels=channels[-1], 
            x_in_spatial=256,
            x_in_channels=input_channels,
            out_channels=num_classes, 
            merge_mode=out_merge_mode, # cat or add
            up_block=out_up_block, # up or tconv
            up_ks=out_up_ks, # [3 or 1] -> upsample kernel size
        )
 
    def forward(self, x):
        # if grayscale input, convert to 3 channels
        y = torch.cat([x, x, x], dim=1) if x.shape[1]==1 else x

        # encoder
        x1, x2, x3, x4 = self.backbone(y)

        # decoder
        dec = self.decoder(x4, [x3, x2, x1])

        # output
        return self.out(dec, x)


def cenet(input_channel=3, num_classes=1):
    return CENet(input_channels=input_channel, num_classes=num_classes)


