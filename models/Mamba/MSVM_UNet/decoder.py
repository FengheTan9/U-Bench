from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn as nn
from einops import rearrange
from .vmamba.vmamba import VSSBlock, LayerNorm2d, Linear2d
from typing import Sequence, Type, Optional

class MSConv(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super(MSConv, self).__init__()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + sum([conv(x) for conv in self.dw_convs])

class MS_MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        channels_first: bool = False,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MS_MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = MSConv(hidden_features, kernel_sizes=kernel_sizes)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSVSS(nn.Sequential):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path: Sequence[float] | float = 0.0,
        use_checkpoint: bool = False,
        norm_layer: Type[nn.Module] = LayerNorm2d,
        channel_first: bool = True,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: Type[nn.Module] = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v05_noz",
        mlp_ratio: float = 4.0,
        mlp_act_layer: Type[nn.Module] = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
    ) -> None:
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                customized_mlp=MS_MLP
            ))
        super(MSVSS, self).__init__(OrderedDict(
            blocks=nn.Sequential(*blocks),
        ))

class LKPE(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Type[nn.Module] = nn.LayerNorm):
        super(LKPE, self).__init__()
        self.dim = dim
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=True)
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return x

class FLKPE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dim_scale: int = 4,
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super(FLKPE, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, padding=1, groups=dim * 16, bias=True)
        )

        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
        self.out = nn.Conv2d(self.output_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return self.out(x)

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path: Sequence[float] | float,
    ) -> None:
        super(UpBlock, self).__init__()
        self.up = LKPE(in_channels)
        self.concat_layer = Linear2d(2 * out_channels, out_channels)
        self.vss_layer = MSVSS(dim=out_channels, depth=depth, drop_path=drop_path)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.up(input)
        out = torch.cat(tensors=(out, skip), dim=1)
        out = self.concat_layer(out)
        out = self.vss_layer(out)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        num_classes: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        drop_path_rate: float = 0.2,
    ) -> None:
        super(Decoder, self).__init__()
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (len(dims) - 1) * 2)]

        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(
                UpBlock(
                    in_channels=dims[i - 1],
                    out_channels=dims[i],
                    depth=depths[i],
                    drop_path=dpr[sum(depths[: i - 1]): sum(depths[: i])],
                ))

        self.out_layers = nn.Sequential(FLKPE(dims[-1], num_classes))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        out = features[0]
        features = features[1:]
        for i, layer in enumerate(self.layers):
            out = layer(out, features[i])
        return self.out_layers[0](out)
