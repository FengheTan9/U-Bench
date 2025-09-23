import torch
import torch.nn as nn
import torch.nn.functional as F

class SKConvBlock0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SKConvBlock0, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class SKConvBlock1plus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SKConvBlock1plus, self).__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(0.01)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.batch_norm4 = nn.BatchNorm2d(out_channels)
        self.activation4 = nn.LeakyReLU(0.01)

    def forward(self, x):
        x2 = self.activation2(self.batch_norm2(self.conv2(x)))
        x4 = self.activation4(self.batch_norm4(self.conv4(x)))
        return torch.cat([x4, x2], dim=1), x2, x4

class SKConvnblockplus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SKConvnblockplus, self).__init__()
        self.block0 = SKConvBlock0(in_channels, out_channels)
        self.block1plus = SKConvBlock1plus(out_channels, out_channels)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2*out_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)

        self.conv5 = nn.Conv2d(out_channels * 2, 1, kernel_size=1)
        self.conv_final = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.batch_norm_final = nn.BatchNorm2d(out_channels)
        self.activation_final = nn.LeakyReLU(0.01)

    def forward(self, x):
        x1 = self.block0(x)
        x2, x21, x22 = self.block1plus(x1)

        # Channel block
        se = self.global_pool(x2).view(x2.size(0), -1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(x21.size(0), x21.size(1), 1, 1)

        data_a = x21 * se
        data_a1 = x22 * (1 - se)

        # Spatial block
        spatial_att = torch.sigmoid(self.conv5(F.relu(x2)))
        data5 = x21 * spatial_att
        data_a2 = x22 * (1 - spatial_att)

        data_out = torch.cat([data_a, data_a1, data_a2, data5, x1], dim=1)
        data_out = self.activation_final(self.batch_norm_final(self.conv_final(data_out)))
        return data_out

class SKupdataplus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SKupdataplus, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = SKConvnblockplus(in_channels + out_channels, out_channels)
        self.conv2 = SKConvnblockplus(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SoOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SoOut, self).__init__()
        # self.size = size
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=size)

    def forward(self, x):
        # 1x1 卷积
        outconv = self.conv(x)
        # Sigmoid 激活
        # out = torch.sigmoid(outconv)
        # 上采样
        # up = self.upsample(out)
        return outconv

class ESKNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=2, dim=32,  deep=False):
        super(ESKNet, self).__init__()
        self.deep = deep
        self.Conv1 = SKConvnblockplus(input_channel, dim)
        self.pool1 = nn.MaxPool2d(2)

        self.Conv2 = SKConvnblockplus(dim, dim*2)
        self.pool2 = nn.MaxPool2d(2)

        self.Conv3 = SKConvnblockplus(dim*2, dim*4)
        self.pool3 = nn.MaxPool2d(2)

        self.Conv4 = SKConvnblockplus(dim*4, dim*8)
        self.pool4 = nn.MaxPool2d(2)

        self.Conv5 = SKConvnblockplus(dim*8, dim*16)

        self.up1 = SKupdataplus(dim*16, dim*8)
        self.up2 = SKupdataplus(dim*8, dim*4)
        self.up3 = SKupdataplus(dim*4, dim*2)
        self.up4 = SKupdataplus(dim*2, dim)

        self.outconv = nn.Conv2d(dim, num_classes, kernel_size=1)
        # self.aux_outconv1 = nn.Conv2d(64, num_cls, kernel_size=1)
        # self.aux_outconv2 = nn.Conv2d(128, num_cls, kernel_size=1)
        # self.aux_outconv3 = nn.Conv2d(256, num_cls, kernel_size=1)
        # self.aux_outconv4 = nn.Conv2d(512, num_cls, kernel_size=1)

    # def soout(self, x, size):
    #     x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
    #     return torch.sigmoid(x)
        if self.deep:
            self.so5 = SoOut(dim*16, num_classes)
            self.so4 = SoOut(dim*8, num_classes)
            self.so3 = SoOut(dim*4, num_classes)
            self.so2 = SoOut(dim*2, num_classes)
        # self.so4 = SoOut(32, num_classes)

    def forward(self, x):
        x1 = self.Conv1(x)
        x1_pool = self.pool1(x1)

        x2 = self.Conv2(x1_pool)
        x2_pool = self.pool2(x2)

        x3 = self.Conv3(x2_pool)
        x3_pool = self.pool3(x3)

        x4 = self.Conv4(x3_pool)
        x4_pool = self.pool4(x4)

        x5 = self.Conv5(x4_pool)
        # out5 = self.aux_outconv4(x5)
        if self.deep:
            out5 = self.so5(x5)

        up1 = self.up1(x5, x4)
        # out4 = self.aux_outconv3(up1)
        if self.deep:
            out4 = self.so4(up1)

        up2 = self.up2(up1, x3)
        # out3 = self.aux_outconv2(up2)
        if self.deep:
            out3 = self.so3(up2)

        up3 = self.up3(up2, x2)
        # out2 = self.aux_outconv1(up3)
        if self.deep:
            out2 = self.so2(up3)

        up4 = self.up4(up3, x1)
        out1 = self.outconv(up4)

        return out1



def esknet(num_classes, input_channel=3):
    return ESKNet(num_classes=num_classes, input_channel=input_channel) 