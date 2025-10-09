from __future__ import annotations
import torch
from torch import nn
from .vmamba import ENCODERS, VSSM
from typing import List, Any

class Encoder(nn.Module):
    def __init__(self, name: str, in_channels: int = 3, **kwargs: Any) -> None:
        super(Encoder, self).__init__()
        vss_encoder: VSSM = ENCODERS[name](in_channels=in_channels, **kwargs)
        self.dims = vss_encoder.dims
        self.channel_first = vss_encoder.channel_first

        self.layer0 = nn.Sequential(
            vss_encoder.patch_embed[0],
            vss_encoder.patch_embed[1],
            vss_encoder.patch_embed[2],
            vss_encoder.patch_embed[3],
            vss_encoder.patch_embed[4],
        )
        self.layer1 = nn.Sequential(
            vss_encoder.patch_embed[5],
            vss_encoder.patch_embed[6],
            vss_encoder.patch_embed[7],
        )
        self.layers = vss_encoder.layers
        self.downsamples = vss_encoder.downsamples

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_channels, 1, 1)

        ret = []
        x = self.layer0(x)
        x = self.layer1(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            ret.append(x if self.channel_first else x.permute(0, 3, 1, 2))
            x = self.downsamples[i](x)
        return ret

    @torch.no_grad()
    def freeze_params(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = False

    @torch.no_grad()
    def unfreeze_params(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = True
