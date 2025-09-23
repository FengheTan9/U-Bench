import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .module.HWD import Down_wt
from transformers import MobileViTConfig, MobileViTModel
from .module.dy import DySample
from .module.dp import GSConv
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


__all__ = ['GH_UNet']


class GroupParallelGatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=4, reduction=16, activation_fn=nn.SiLU):
        super(GroupParallelGatedBlock, self).__init__()
        self.num_groups = num_groups
        self.group_convs = nn.ModuleList([
            nn.Conv2d(in_channels // num_groups, out_channels // num_groups, kernel_size=3, padding=1, bias=False)
            for _ in range(num_groups)
        ])

        # 门控机制，用于控制每个分组的输出权重
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=False),
            activation_fn(),  # 使用自定义激活函数
            nn.Conv2d(out_channels // reduction, num_groups, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        group_inputs = torch.chunk(x, self.num_groups, dim=1)
        group_outputs = [conv(group_inputs[i]) for i, conv in enumerate(self.group_convs)]
        out = torch.cat(group_outputs, dim=1)
        gate_weights = self.gate_fc(out)
        group_outputs = torch.chunk(out, self.num_groups, dim=1)
        gated_outputs = [group_outputs[i] * gate_weights[:, i:i + 1, :, :] for i in range(self.num_groups)]
        out = torch.cat(gated_outputs, dim=1)

        return out


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class PGAM(nn.Module):
    def __init__(self, in_channels):
        super(PGAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算通道和空间门控权重
        channel_attention = self.channel_gate(x)
        spatial_attention = self.spatial_gate(x)
        return x * channel_attention * spatial_attention  # 在卷积之前对输入特征加权


class Lo2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(in_features, hidden_features)
        self.fc4 = nn.Linear(in_features, hidden_features)
        self.fc5 = nn.Linear(in_features * 2, hidden_features)
        self.fc6 = nn.Linear(hidden_features * 2, out_features)
        self.drop = nn.Dropout(drop)
        self.dwconv = DWConv(hidden_features)
        self.act1 = act_layer()
        self.act2 = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_features * 2)
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        ### DOR-MLP
        ### OR-MLP
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc1(x_shift_r)
        x_shift_r = self.act1(x_shift_r)
        x_shift_r = self.drop(x_shift_r)
        xn = x_shift_r.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc2(x_shift_c)
        x_1 = self.drop(x_shift_c)

        ### OR-MLP
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, -shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc3(x_shift_c)
        x_shift_c = self.act1(x_shift_c)
        x_shift_c = self.drop(x_shift_c)
        xn = x_shift_c.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc4(x_shift_r)
        x_2 = self.drop(x_shift_r)

        x_1 = torch.add(x_1, x)
        x_2 = torch.add(x_2, x)
        x1 = torch.cat([x_1, x_2], dim=2)
        x1 = self.norm1(x1)
        x1 = self.fc5(x1)
        x1 = self.drop(x1)
        x1 = torch.add(x1, x)
        x2 = x.transpose(1, 2).view(B, C, H, W)

        ### DSC
        x2 = self.dwconv(x2, H, W)
        x2 = self.act2(x2)
        x2 = self.norm2(x2)
        x2 = x2.flatten(2).transpose(1, 2)

        x3 = torch.cat([x1, x2], dim=2)
        x3 = self.fc6(x3)
        x3 = self.drop(x3)
        return x3


class Lo2Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Lo2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.drop_path(self.mlp(x, H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, H, W):
        identity = x
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x + identity


class Feature_Incentive_Block(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.apply(self._init_weights)
        # self.se = CBAM(embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        # x = self.se(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)

        return x, H, W


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, act_layer=nn.ReLU6, group=4):
        super(DoubleConv, self).__init__()
        self.conv1 = GSConv(in_ch, out_ch)
        self.conv2 = GSConv(in_ch, out_ch, k=5)  # 使用不同的卷积核大小
        self.conv3 = GSConv(in_ch, out_ch, k=7)  # 使用不同的卷积核大小
        # 多尺度卷积

        self.pgam1 = PGAM(in_ch)
        self.pgam2 = PGAM(in_ch)
        self.pgam3 = PGAM(in_ch)
        self.pgam4 = PGAM(out_ch * 3)

        # 批归一化和激活函数
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.act = act_layer()
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = act_layer()
        self.conv4 = nn.Conv2d(out_ch * 3, out_ch, kernel_size=3, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm2d(out_ch)
        self.act4 = act_layer()
        # self.cbam = CBAM(out_ch)   # SE 注意力模块
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6()
        )

    def forward(self, input):
        x1 = self.pgam1(input)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)  # 添加层次残差
        x1 = self.act(x1)

        x2 = self.pgam2(input)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)  # 添加层次残差
        x2 = self.act(x2)

        x3 = self.pgam3(input)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)  # 添加层次残差
        x3 = self.act(x3)

        x = torch.cat((x1, x2, x3), dim=1)  # 在通道维度上拼接

        x = self.pgam4(x)
        x = self.conv4(x)

        # 加上残差连接
        x = x + self.residual(input)

        return x


class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, act_layer=nn.ReLU6, group=4):
        super(D_DoubleConv, self).__init__()
        self.conv1 = GSConv(in_ch, in_ch)
        self.conv2 = GSConv(in_ch, in_ch, k=5)  # 使用不同的卷积核大小
        self.conv3 = GSConv(in_ch, in_ch, k=7)  # 使用不同的卷积核大小
        # 多尺度卷积
        self.pgam1 = PGAM(in_ch)
        self.pgam2 = PGAM(in_ch)
        self.pgam3 = PGAM(in_ch)
        self.pgam4 = PGAM(in_ch * 3)

        # 批归一化和激活函数
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.act = act_layer()

        self.conv4 = nn.Conv2d(in_ch * 3, out_ch, kernel_size=3, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm2d(out_ch)
        self.act4 = act_layer()

        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6()
        )
        # self.caa2=CBAM(out_ch)

    def forward(self, input):
        # x1=input
        # x2=input
        # x3=input
        x1 = self.pgam1(input)
        x2 = self.pgam2(input)
        x3 = self.pgam3(input)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        # 进行批归一化和激活
        x1 = self.bn1(x1)
        x1 = self.act(x1)
        # x1= self.se1(x1)

        x2 = self.bn2(x2)
        x2 = self.act(x2)
        # x2= self.se2(x2)

        x3 = self.bn3(x3)
        x3 = self.act(x3)
        # x3= self.se3(x3)

        # 合并各个路径的输出
        x = torch.cat((x1, x2, x3), dim=1)  # 在通道维度上拼接

        x = self.pgam4(x)
        x = self.conv4(x)

        # 加上残差连接
        x = x + self.residual(input)

        return x


class GH_UNet(nn.Module):
    def __init__(self, num_classes, input_channel=3, deep_supervision=True, img_size=672,
                 embed_dims=[32, 64, 96, 128, 640],  # 32, 64, 128, 256, 512
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.conv1 = DoubleConv(input_channel, embed_dims[0])  # 256

        self.pool1 = Down_wt(embed_dims[0], embed_dims[0])
        self.conv2 = DoubleConv(embed_dims[0], embed_dims[1])  # 128

        self.pool2 = Down_wt(embed_dims[1], embed_dims[1])
        self.conv3 = DoubleConv(embed_dims[1], embed_dims[2])  # 64

        self.pool3 = Down_wt(embed_dims[2], embed_dims[2])

        self.pool4 = Down_wt(embed_dims[3], embed_dims[3])
        config = MobileViTConfig(
            image_size=256,
            patch_size=16,
            num_channels=3,
            hidden_size=640,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=2560,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-6,
            max_position_embeddings=256,
            use_absolute_embeddings=False,
            use_relative_embeddings=True,
            num_labels=1000,  # 对于 ImageNet 分类任务
        )

        
        self.vit = MobileViTModel(config=config).base_model
        self.FIBlock1 = Feature_Incentive_Block(img_size=img_size // 4, patch_size=3, stride=1,
                                                in_chans=embed_dims[2],
                                                embed_dim=embed_dims[3])
        self.FIBlock2 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[3],
                                                embed_dim=embed_dims[4])
        self.FIBlock3 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[4],
                                                embed_dim=embed_dims[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([Lo2Block(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate + 0.2, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])
        self.norm3 = norm_layer(embed_dims[3])
        self.FIBlock4 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])
        self.decoder3 = D_DoubleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_DoubleConv(embed_dims[1], embed_dims[0])
        self.decoder1 = D_DoubleConv(embed_dims[0], 16, group=2)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.se1 = GroupParallelGatedBlock(embed_dims[3], embed_dims[3], 2, 16)
        self.se2 = GroupParallelGatedBlock(embed_dims[1], embed_dims[1], 4, 8)
        self.se3 = GroupParallelGatedBlock(embed_dims[0], embed_dims[0], 8, 8)
        self.seg1 = nn.Conv2d(embed_dims[3], num_classes, kernel_size=1)
        self.seg2 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.seg3 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        self.dy1 = DySample(embed_dims[0], scale=2)
        self.dy2 = DySample(embed_dims[1], scale=2)
        self.dy3 = DySample(embed_dims[2], scale=2)
        self.dy4 = DySample(embed_dims[3], scale=2)
        self.dy5 = DySample(embed_dims[3])
        self.dy6 = DySample(embed_dims[2])
        self.dy7 = DySample(embed_dims[1])
        self.dy8 = DySample(embed_dims[0])
        self.dy9 = DySample(embed_dims[4])
        self.dy10 = DySample(embed_dims[3], scale=8)
        self.dy11 = DySample(embed_dims[1], scale=4)
        self.dy12 = DySample(embed_dims[0], scale=2)

        self.m1 = GroupParallelGatedBlock(embed_dims[0], embed_dims[0], 8, 8)
        self.m2 = GroupParallelGatedBlock(embed_dims[1], embed_dims[1], 4, 8)
        self.m3 = GroupParallelGatedBlock(embed_dims[2], embed_dims[2], 2, 16)
        self.m4 = GroupParallelGatedBlock(embed_dims[3], embed_dims[3], 2, 16)

    def forward(self, x):
        B = x.shape[0]
        outputs = []
        vit_ = self.vit(x, output_hidden_states=True)
        v1, v2, v3, v4, _ = vit_.hidden_states

        v1 = self.dy1(v1)
        v2 = self.dy2(v2)
        v3 = self.dy3(v3)
        v4 = self.dy4(v4)

        vit_out = vit_.last_hidden_state

        vit_out = self.dy9(vit_out)
        ### Conv Stage
        out = self.conv1(x)  # 256
        t1 = out
        out += v1
        t1 += v1

        t1 = self.m1(t1)
        out = self.pool1(out)
        out = self.conv2(out)  # 128

        t2 = out
        # out=self.rong2(v2,out)
        out += v2

        t2 += v2

        t2 = self.m2(t2)
        out = self.pool2(out)
        out = self.conv3(out)  # 64

        t3 = out

        out += v3

        t3 += v3

        t3 = self.m3(t3)
        out = self.pool3(out)

        ### Stage 4
        out, H, W = self.FIBlock1(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        t4 = out

        out += v4

        t4 += v4

        t4 = self.m4(t4)
        out = self.pool4(out)
        ### Bottleneck
        out, H, W = self.FIBlock2(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = out + vit_out
        out, H, W = self.FIBlock3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = self.dy5(out)

        ### Stage 4

        out = torch.add(out, t4)
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            out = blk(out, H * 2, W * 2)
        out = self.norm3(out)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2).contiguous()
        p3 = self.se1(out)
        p3 = self.dy10(p3)

        p3 = self.seg1(p3)

        outputs.append(p3)
        out = F.relu(self.dbn4(self.FIBlock4(out)))
        out = self.dy6(out)

        ### Conv Stage

        out = torch.add(out, t3)  # 64
        out = self.decoder3(out)  # 64
        p2 = self.se2(out)
        p2 = self.dy11(p2)

        p2 = self.seg2(p2)

        outputs.append(p2)

        out = self.dy7(out)

        out = torch.add(out, t2)
        out = self.decoder2(out)  # 128
        p1 = self.se3(out)
        p1 = self.dy12(p1)

        p1 = self.seg3(p1)

        outputs.append(p3)

        out = self.dy8(out)
        out = torch.add(out, t1)
        out = self.decoder1(out)  # 256

        out = self.final(out)
        out = out + p1 + p2 + p3
        outputs.append(out)
        return outputs

def gh_unet(num_classes, input_channel=3):
    model = GH_UNet(input_channel=input_channel, num_classes=num_classes, img_size=256)
    return model