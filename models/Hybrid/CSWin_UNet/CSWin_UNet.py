# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn
from .vision_transformer import CSWinTransformer
from torch.hub import load_state_dict_from_url


logger = logging.getLogger(__name__)

class CSwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False, input_channel=3):
        super(CSwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.cswin_unet = CSWinTransformer(img_size=img_size,
                                patch_size=4,
                                in_chans=input_channel,
                                num_classes=self.num_classes,
                                embed_dim=64,
                                depth=[1, 2, 9, 1],
                                split_size=[1, 2, 8, 8],
                                num_heads=[2, 4, 8, 16],
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.cswin_unet(x)
        return logits

    def load_from(self):
        pretrained_path = "https://github.com/eatbeanss/CSWin-UNet/blob/main/pretrained_ckpt/cswin_tiny_224.pth"
        # print('pretrained_path')
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = load_state_dict_from_url("https://huggingface.co/FengheTan9/U-Stone/resolve/main/TransAttUnet.pth", progress=True)
            model_dict = self.cswin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "stage" in k:
                    current_k = "stage_up" + k[5:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,full_dict[k].shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.cswin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


def cswin_unet(num_classes, input_channel=3):
    model = CSwinUnet(input_channel=input_channel, num_classes=num_classes, img_size=256)
    model.load_from()
    return model
