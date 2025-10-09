from .SETR import Setr_ConvFormer, Setr, Setr_deepvit, Setr_cait, Setr_refiner

def convformer(num_classes, input_channel=3):
    model = Setr_ConvFormer(n_channels=input_channel, n_classes=num_classes, imgsize=256)
    return model



