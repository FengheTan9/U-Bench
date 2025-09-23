
import torch
import torch.nn as nn
import torchvision.models as models

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        p = self.avg_pool(x).view(b, c)
        y = self.fc(p).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4

class StridedConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(StridedConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderBlock, self).__init__()

        self.residual_block1 = ResidualBlock(in_c, out_c)
        self.strided_conv = StridedConvBlock(out_c, out_c)
        self.residual_block2 = ResidualBlock(out_c, out_c)
        self.pooling = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x1 = self.residual_block1(x)
        x2 = self.strided_conv(x1)
        x3 = self.residual_block2(x2)
        p = self.pooling(x3)
        return x1, x3, p

class CompNet(nn.Module):
    def __init__(self, input_channel,out_channels):
        super(CompNet, self).__init__()

        """ Encoder """
        self.e1 = EncoderBlock(input_channel, 64)
        self.e2 = EncoderBlock(64, 256)

        """ Decoder 1 """
        self.t1 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=4, padding=0)
        self.r1 = ResidualBlock(192, 128)
        self.t2 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1)
        self.r2 = ResidualBlock(256, 128)

        """ Decoder 2 """
        self.t3 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1)
        self.r3 = ResidualBlock(128, 64)
        self.t4 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2, padding=1)
        self.r4 = ResidualBlock(96, 32)

        """ Output """
        self.output = nn.Conv2d(32, out_channels, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        s11, s12, p1 = self.e1(x)       ## 512, 256, 128
        s21, s22, p2 = self.e2(p1)     ## 128, 64, 32

        t1 = self.t1(s22)
        t1 = torch.cat([t1, s12], axis=1)
        r1 = self.r1(t1)

        t2 = self.t2(s21)
        t2 = torch.cat([r1, t2], axis=1)
        r2 = self.r2(t2)

        t3 = self.t3(r2)
        t3 = torch.cat([t3, s11], axis=1)
        r3 = self.r3(t3)

        t4 = self.t4(s12)
        t4 = torch.cat([r3, t4], axis=1)
        r4 = self.r4(t4)

        output = self.output(r4)
        return output


def colonsegnet(input_channel,num_classes):
    return CompNet(out_channels=num_classes, input_channel=input_channel)




if __name__ == "__main__":
    model = CompNet().cuda()
    from torchsummary import summary
    summary(model, (3, 512, 512))