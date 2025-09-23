import torch.nn as nn
from einops.layers.torch import Rearrange

from .module.Encoder import All2Cross
from .module.Decoder import ConvUpsample, SegmentationHead
from .module.configs import get_hiformer_b_configs

class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, input_channel=3, num_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.num_classes = num_classes
        self.All2Cross = All2Cross(config = config, img_size= img_size, in_chans=input_channel)
        
        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128,128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)
    
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=num_classes,
            kernel_size=3,
        )    

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
    
    def forward(self, x):
        
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):

            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size//self.patch_size[i]), w=(self.img_size//self.patch_size[i]))(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)
            
            reshaped_embed.append(embed)

        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        out = self.segmentation_head(C)
        
        return out  


def hiformer(num_classes, input_channel=3):
    model = HiFormer(config=get_hiformer_b_configs(), num_classes=num_classes, input_channel=input_channel)
    return model