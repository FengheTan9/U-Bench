import ml_collections
import os
import wget




# BEFUnet Configs
def get_BEFUnet_configs():
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9
    #if not os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
    #    print('Downloading Swin-transformer model ...')
    #    wget.download(
    #        "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    #        "./weights/swin_tiny_patch4_window7_224.pth")
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # CNN Configs
    cfg.cnn_backbone = "pidinet_small_converted"
    cfg.pdcs = 'carv4'
    cfg.cnn_pyramid_fm = [30, 60, 120, 120]
    cfg.pidinet_pretrained = False
    cfg.PDC_pretrained_path = './weights/table5_pidinet-small.pth'

    # DLF Configs
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio = (2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg
