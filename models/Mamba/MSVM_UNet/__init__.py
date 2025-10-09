from __future__ import annotations
import torch
from torch import Tensor
from torch import nn
from typing import Any
from .encoder import Encoder
from .decoder import Decoder

class MSVMUNet(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        num_classes: int = 9,
        *,
        enc_name: str = "tiny_0230s"  # tiny_0230s, small_0229s
    ) -> None:
        super(MSVMUNet, self).__init__()
        self.encoder = Encoder(enc_name, in_channels=input_channel)
        self.dims = self.encoder.dims
        self.decoder = Decoder(dims=self.dims[::-1], num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.decoder(self.encoder(x)[::-1])

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()

def build_model(**kwargs: Any) -> MSVMUNet:
    return MSVMUNet(**kwargs)
