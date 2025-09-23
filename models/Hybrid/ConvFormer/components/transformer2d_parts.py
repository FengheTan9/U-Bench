import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.single_conv(x)

class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, cnn_num=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(
                SingleConv(in_channels, out_channels//2)
            )
        for _ in range(cnn_num-2):
            self.layers.append(
                SingleConv(out_channels//2, out_channels//2)
            )
        self.layers.append(
                SingleConv(out_channels//2, out_channels)
            )
    def forward(self, x):
        for cnn in self.layers:
            x = cnn(x)
        return x

class DownSingleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

#  ===================================================== encoders for 2d transformers ======================================================

class CNNEncoder1(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder1, self).__init__()
        self.cnn_num = 4 # depend on the patch_height, patch_height -> cnn_num : 1 -> 1, 2->1, 4->2, 8->4, 16 -> 8
        self.multi_cnn = MultiConv(n_channels, out_channels//2, cnn_num=self.cnn_num)
        self.patch_dim = out_channels // 2 * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_height, p2=patch_width),
            nn.Conv2d(self.patch_dim, out_channels, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.multi_cnn(x)
        x = self.to_patch_embedding(x)
        return x

class CNNEncoder2(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder2, self).__init__()
        self.scale = 1
        self.inc = SingleConv(n_channels, 64 // self.scale)
        self.down1 = DownSingleConv(64 // self.scale, 128 // self.scale)
        self.down2 = DownSingleConv(128 // self.scale, 256 // self.scale)
        self.down3 = DownSingleConv(256 // self.scale, out_channels)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x


class CNNEncoder3(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder3, self).__init__()
        self.cnn_num = 4 # depend on the patch_height, patch_height -> cnn_num : 1 -> 1, 2->1, 4->2, 8->4, 16 -> 8
        self.multi_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.patch_dim = 32 * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_height, p2=patch_width),
            nn.Conv2d(self.patch_dim, out_channels, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.multi_cnn(x)
        x = self.to_patch_embedding(x)
        return x

class CNNEncoder4(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder4, self).__init__()
        self.multi_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.patch_dim = 32 * (patch_height//2) * (patch_width//2)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_height//2, p2=patch_width//2),
            nn.Conv2d(self.patch_dim, out_channels, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.multi_cnn(x)
        x = self.to_patch_embedding(x)
        return x

class CNNEncoder5(nn.Module):
    def __init__(self, n_channels, out_channels, patch_height, patch_width):
        super(CNNEncoder5, self).__init__()
        self.scale = 1
        self.inc = SingleConv(n_channels, 64 // self.scale)
        self.down1 = SingleConv(64 // self.scale, 128 // self.scale)
        self.down2 = SingleConv(128 // self.scale, 256 // self.scale)
        self.down3 = SingleConv(256 // self.scale, out_channels)
        self.down = nn.MaxPool2d(8)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.down(x4)
        return x4