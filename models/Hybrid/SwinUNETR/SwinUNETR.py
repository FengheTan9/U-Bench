from monai.networks.nets.swin_unetr import SwinUNETR

def swinunetr(input_channel=3, num_classes=1):
    return SwinUNETR(in_channels=input_channel, out_channels=num_classes, img_size=(256, 256), spatial_dims=2)