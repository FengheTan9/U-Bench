from monai.networks.nets.unetr import UNETR

def unetr(input_channel=3, num_classes=1):
    return UNETR(in_channels=input_channel, out_channels=num_classes, img_size=(256, 256), spatial_dims=2)
