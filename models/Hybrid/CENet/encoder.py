import torch
from .pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.hub import load_state_dict_from_url

def get_encoder2d(input_channels=1, encoder='pvt_v2_b2', pretrain=False, freeze_bb=False, base_ptdir='.'):
        # backbone network initialization with pretrained weight
        path = ""
        if encoder == 'pvt_v2_b0':
            backbone = pvt_v2_b0()
            path = f'{base_ptdir}/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            backbone = pvt_v2_b1()
            path = f'{base_ptdir}/pvt/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            backbone = pvt_v2_b2()
            path = '[URL]'
            channels=[512, 320, 128, 64]
        


        if 'pvt_v2' in encoder:
            print(f'Loading pretrained weights from {path}')
            save_model = load_state_dict_from_url(path, progress=True)
            model_dict = backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            backbone.load_state_dict(model_dict)
            if freeze_bb:
                for name, param in backbone.named_parameters():
                    param.requires_grad = False


        return backbone, channels