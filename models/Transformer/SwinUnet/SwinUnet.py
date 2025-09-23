from .vision_transformer import SwinUnet
from .config import get_config
import argparse

def swinunet(num_classes, input_channel=3):
    model = SwinUnet(img_size=224,input_channel=input_channel, num_classes=num_classes)
    return model