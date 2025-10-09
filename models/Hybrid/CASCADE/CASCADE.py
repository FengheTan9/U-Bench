import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from scipy import ndimage

from .pvtv2 import pvt_v2_b2
from .decoders import CASCADE
from .cnn_vit_backbone import Transformer, SegmentationHead
from torch.hub import load_state_dict_from_url
logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)



class PVT_CASCADE(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(PVT_CASCADE, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        #path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        
        
        
        save_model = load_state_dict_from_url('https://huggingface.co/FengheTan9/U-Stone/resolve/main/pvt_v2_b2.pth', progress=True)
        #save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = CASCADE(channels=[512, 320, 128, 64])
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, num_classes, 1)
        self.out_head2 = nn.Conv2d(320, num_classes, 1)
        self.out_head3 = nn.Conv2d(128, num_classes, 1)
        self.out_head4 = nn.Conv2d(64, num_classes, 1)

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

class TransCASCADE(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(TransCASCADE, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.img_size = img_size
        
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # hybrid CNN-transformer backbone
        self.transformer = Transformer(config, self.img_size, vis)
        head_channels = 512
        
        # decoder initialization
        self.decoder = CASCADE(channels=[768,512,256,64])
        
        # prediction heads
        self.segmentation_head1 = SegmentationHead(
            in_channels=768,
            out_channels=config['n_classes'],
            kernel_size=1,
            upsampling=16,
        )
        self.segmentation_head2 = SegmentationHead(
            in_channels=512,
            out_channels=config['n_classes'],
            kernel_size=1,
            upsampling=8,
        )
        self.segmentation_head3 = SegmentationHead(
            in_channels=256,
            out_channels=config['n_classes'],
            kernel_size=1,
            upsampling=4,
        )
        self.segmentation_head4 = SegmentationHead(
            in_channels=64,
            out_channels=config['n_classes'],
            kernel_size=1,
            upsampling=2
        )
        self.config = config
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, im_size=224):
        
        if x.size()[1] == 1:
            x = self.conv(x)
         
        x, attn_weights, features = self.transformer(x, im_size)  # (B, n_patch, hidden)
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)

        x1_o, x2_o, x3_o, x4_o = self.decoder(x, features)

        p1 = self.dropout(self.segmentation_head1(x1_o))
        p2 = self.dropout(self.segmentation_head2(x2_o))
        p3 = self.dropout(self.segmentation_head3(x3_o))
        p4 = self.dropout(self.segmentation_head4(x4_o))
        return p1, p2, p3, p4

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)                        
        

def cascade(num_classes, input_channel=3):
    model = PVT_CASCADE(num_classes=num_classes, input_channel=input_channel)
    return model

