import sys

import torch
from torch import nn
# sys.path.append("../")

# 定义一个 EMA 类，继承自 nn.Module。
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        # 调用父类的构造函数。
        super(EMA, self).__init__()
        # 将通道数分成多个组，组的数量为 factor。
        self.groups = factor
        # 确保每组至少有一个通道。
        assert channels // self.groups > 0
        # 定义一个 softmax 层，用于计算权重。
        self.softmax = nn.Softmax(-1)
        # 自适应平均池化，将每个通道缩放到 (1,1) 的输出。
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # 自适应平均池化，将高维度缩放到 1，宽维度保持不变。
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 自适应平均池化，将宽维度缩放到 1，高维度保持不变。
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # 组归一化，每组的通道数为 channels // groups。
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # 1x1 卷积层，用于通道的转换和维度缩减。
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        # 3x3 卷积层，用于提取局部特征。
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 获取输入 x 的形状 (batch_size, channels, height, width)。
        b, c, h, w = x.size()
        # 将输入 x 重新排列为 (batch_size * groups, channels // groups, height, width)。
        group_x = x.reshape(b * self.groups, -1, h, w)
        # 计算沿高度方向的池化，得到大小为 (batch_size * groups, channels // groups, height, 1) 的张量。
        x_h = self.pool_h(group_x)
        # 计算沿宽度方向的池化，得到大小为 (batch_size * groups, channels // groups, 1, width) 的张量，并进行转置。
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        # 将两个池化的张量连接在一起，并通过 1x1 卷积层，得到一个新的特征图。
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        # 将特征图按原来的高度和宽度切分，分别得到 x_h 和 x_w。
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # 使用 sigmoid 激活函数和 x_h, x_w 调整 group_x 的特征值。
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        # 使用 3x3 卷积层对 group_x 进行特征提取。
        x2 = self.conv3x3(group_x)
        # 计算 x1 的平均池化并通过 softmax，得到权重。
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # 将 x2 重新排列为 (batch_size * groups, channels // groups, height * width) 的形状。
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        # 计算 x2 的平均池化并通过 softmax，得到权重。
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # 将 x1 重新排列为 (batch_size * groups, channels // groups, height * width) 的形状。
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        # 计算 x11 和 x12, x21 和 x22 的矩阵乘法，并将结果 reshape 为 (batch_size * groups, 1, height, width)。
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # 使用权重调整 group_x 的特征，并 reshape 为原始的形状 (batch_size, channels, height, width)。
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)