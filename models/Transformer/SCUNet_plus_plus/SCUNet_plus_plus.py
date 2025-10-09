# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from torch.hub import load_state_dict_from_url
logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False, input_channel=3):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.swin_unet = SwinTransformerSys(img_size=224,
                                            patch_size=4,
                                            in_chans=input_channel,
                                            num_classes=self.num_classes,
                                            embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=7,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.0,
                                            drop_path_rate=0.1,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self):
        pretrained_path = 'https://huggingface.co/FengheTan9/U-Stone/resolve/main/swin_tiny_patch4_window7_224.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = load_state_dict_from_url(pretrained_path, progress=True)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            #print(msg)
        else:
            print("none pretrain")



def scunet_plus_plus(num_classes, input_channel=3):
    model = SwinUnet(input_channel=input_channel, num_classes=num_classes, img_size=224)
    model.load_from()
    return model


