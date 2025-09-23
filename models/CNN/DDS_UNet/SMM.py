import os
import pdb
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Convblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Convblock, self).__init__()
        self.encoder = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.ebn = nn.BatchNorm2d(output_channels)
    def forward(self, x):
        out = F.relu(F.max_pool2d(self.ebn(self.encoder(x)), 2, 2))
        return out

class Convblock1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Convblock1, self).__init__()
        self.encoder = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.ebn = nn.BatchNorm2d(output_channels)
    def forward(self, x):
        out = F.relu(self.ebn(self.encoder(x)))
        return out

class Upblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Upblock, self).__init__()
        self.decoder = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.dbn = nn.BatchNorm2d(output_channels)
    def forward(self, x):
        out = F.relu(F.interpolate(self.dbn(self.decoder(x)), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        return out

class SMM3(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.filters = [input_channels, input_channels//2, input_channels//4]     #L

        self.Convstage1 = Convblock(input_channels, self.filters[1])
        self.Convstage2 = Convblock(self.filters[1], self.filters[2])
        self.Convstage3 = Convblock(self.filters[2], self.filters[2])

        self.Upstage3 = Upblock(self.filters[2], self.filters[2])
        self.Upstage4 = Upblock(self.filters[2], self.filters[1])
        self.Upstage5 = Upblock(self.filters[1], self.filters[0])
    def forward(self, x):
        ### Stage 1
        out = self.Convstage1(x)
        t1 = out

        ### Stage 2
        out = self.Convstage2(out)
        t2 = out

        ### Stage 3
        out = (self.Convstage3(out))

        ### Stage 2
        out = self.Upstage3(out)
        out = torch.add(out, (t2))

        ### Stage 1
        out = self.Upstage4(out)
        out = torch.add(out, (t1))

        ### Stage 0
        out = self.Upstage5(out)
        return out

class SMM4(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.filters = [input_channels, input_channels//2, input_channels//4,input_channels//8]

        self.Convstage1 = Convblock(input_channels, self.filters[1])
        self.Convstage2 = Convblock(self.filters[1], self.filters[2])
        self.Convstage3 = Convblock(self.filters[2], self.filters[3])
        self.Convstage4 = Convblock(self.filters[3], self.filters[3])

        self.Upstage4 = Upblock(self.filters[3], self.filters[3])
        self.Upstage3 = Upblock(self.filters[3], self.filters[2])
        self.Upstage2 = Upblock(self.filters[2], self.filters[1])
        self.Upstage1 = Upblock(self.filters[1], self.filters[0])

    def forward(self, x):
        ### Stage 1
        out = self.Convstage1(x)
        t1 = out

        ### Stage 2
        out = self.Convstage2(out)
        t2 = out

        ### Stage 3
        out = self.Convstage3(out)
        t3 = out

        out = self.Convstage4(out)

        out = self.Upstage4(out)
        out = torch.add(out, (t3))

        ### Stage 2
        out = self.Upstage3(out)
        out = torch.add(out, (t2))

        ### Stage 1
        out = self.Upstage2(out)
        out = torch.add(out, (t1))

        ### Stage 0
        out = self.Upstage1(out)
        return out

class SMM2(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.filters = [input_channels, input_channels//2]

        self.Convstage1 = Convblock(input_channels, self.filters[1])
        self.Convstage2 = Convblock(self.filters[1], self.filters[1])

        self.Upstage2 = Upblock(self.filters[1], self.filters[1])
        self.Upstage1 = Upblock(self.filters[1], self.filters[0])

    def forward(self, x):
        ### Stage 1
        out = self.Convstage1(x)
        t1 = out

        ### Stage 2
        out = self.Convstage2(out)

        ### Stage 1
        out = self.Upstage2(out)
        out = torch.add(out, (t1))

        ### Stage 0
        out = self.Upstage1(out)
        return out

class SMM1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.filters = [input_channels, input_channels // 2]  # L

        self.Convstage1 = Convblock1(input_channels, self.filters[0])

    def forward(self, x):
        out = self.Convstage1(x)
        return out