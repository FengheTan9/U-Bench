# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
import torch.nn as nn
from .bra_unet_system import BRAUnetSystem
logger = logging.getLogger(__name__)
from torch.hub import load_state_dict_from_url


class BRAUnet(nn.Module):
    def __init__(self, img_size=256, input_channel=3, num_classes=1, n_win=8):
        super(BRAUnet, self).__init__()
        self.bra_unet = BRAUnetSystem(img_size=img_size,
                                      in_chans=input_channel,
                                      num_classes=num_classes,
                                      head_dim=32,
                                      n_win=n_win,
                                      embed_dim=[96, 192, 384, 768],
                                      depth=[2, 2, 8, 2],
                                      depths_decoder=[2, 8, 2, 2],
                                      mlp_ratios=[3, 3, 3, 3],
                                      drop_path_rate=0.2,
                                      topks=[2, 4, 8, -2],
                                      qk_dims=[96, 192, 384, 768])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.bra_unet(x)
        return logits
    def load_from(self):
        pretrained_path = 'https://huggingface.co/FengheTan9/U-Stone/resolve/main/biformer_base_best.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = load_state_dict_from_url(pretrained_path, progress=True)
            model_dict = self.bra_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict['model'])
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,model_dict[k].shape))
                        del full_dict[k]
            msg = self.bra_unet.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")

def braunet_plus_plus(num_classes, input_channel=3):
    model = BRAUnet(input_channel=input_channel, num_classes=num_classes, img_size=256)
    return model

