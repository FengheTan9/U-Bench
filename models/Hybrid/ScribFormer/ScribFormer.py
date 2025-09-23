import numpy as np

np.set_printoptions(threshold=np.inf)

from .scribformer_cam import Net

class ScribFormer(Net):
    def __init__(self, linear_layer=True, bilinear=True, num_classes=4, input_channel=3):
        super(ScribFormer, self).__init__(patch_size=16, in_chans=input_channel, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, num_classes=num_classes,
                                                           linear_layer=linear_layer, bilinear=bilinear)
                                                           
                                                           
                                                           
def scribformer(input_channel=3, num_classes=1):
    return ScribFormer(input_channel=input_channel, num_classes=num_classes)


 