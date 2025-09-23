import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvUtr(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, kernel=3):
        super(ConvUtr, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(kernel, kernel), groups=ch_in, padding=(kernel // 2, kernel // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
            ) for i in range(depth)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, inch=3, dims=[8, 16, 32], depths=[1, 1, 3], kernels=[3, 3, 7]):
        super(Embeddings, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(inch, dims[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ConvUtr(dims[0], dims[0], depth=depths[0], kernel=kernels[0])
        self.layer2 = ConvUtr(dims[0], dims[1], depth=depths[1], kernel=kernels[1])
        self.layer3 = ConvUtr(dims[1], dims[2], depth=depths[2], kernel=kernels[2])
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x0 = self.stem(x)
        x0 = self.layer1(x0)

        x1 = self.down(x0)
        x1 = self.layer2(x1)

        x2 = self.down(x1)
        x2 = self.layer3(x2)

        return x2, (x0, x1, x2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 3, padding=1, groups=in_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 3, padding=1, groups=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAgg(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = x + x * (self.sg(self.pos_embed(x)) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.mlp(self.norm2(x)))) - 0.5)
        return x


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class LGLBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()

        if sr_ratio > 1:
            self.LocalAgg = LocalAgg(dim, mlp_ratio, drop, drop_path, act_layer)
        else:
            self.LocalAgg = nn.Identity()

        self.SelfAttn = SelfAttn(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                                 norm_layer, sr_ratio)

    def forward(self, x):
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MobileUViT(nn.Module):
    def __init__(self,
                 input_channel=3,
                 dims=[16, 32, 64, 128],
                 depths=[1, 1, 3, 3, 3],
                 kernels=[3, 3, 7],
                 embed_dim=256,
                 num_classes=1):
        super(MobileUViT, self).__init__()
        self.patch_embeddings = Embeddings(inch=input_channel, dims=dims, depths=depths, kernels=kernels)

        self.adaptive_lgl_bottleneck = nn.ModuleList([
            LGLBlock(
                dim=dims[2], num_heads=embed_dim // 64, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=2)
            for _ in range(depths[3])])

        self.transformer_bottleneck = nn.ModuleList([
            LGLBlock(
                dim=dims[3], num_heads=embed_dim // 64, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=1)
            for _ in range(depths[4])])

        self.down = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.expend_dims = conv_block(ch_in=dims[2], ch_out=dims[3])
        self.reduce_dims = conv_block(ch_in=dims[3], ch_out=dims[2])

        self.Up_conv4 = conv_block(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = conv_block(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = conv_block(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Up1 = up_conv(ch_in=dims[0], ch_out=dims[0])
        self.head = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, skip = self.patch_embeddings(x)
        x = self.down(x)

        for blk in self.adaptive_lgl_bottleneck:
            x = blk(x)
        x = self.expend_dims(x)

        for blk in self.transformer_bottleneck:
            x = blk(x)
        x = self.reduce_dims(x)

        x = self.Up_conv4(torch.cat((x, self.down(skip[2])), dim=1))
        x = self.Up3(x)
        x = self.Up_conv3(torch.cat((x, self.down(skip[1])), dim=1))
        x = self.Up2(x)
        x = self.Up_conv2(torch.cat((x, self.down(skip[0])), dim=1))
        x = self.Up1(x)
        x = self.head(x)
        return x


def mobileuvit(input_channel=3, dims=[16, 32, 64, 128], depths=[1, 1, 3, 3, 3], kernels=[3, 3, 7], embed_dim=256, num_classes=1):
    return MobileUViT(input_channel=input_channel, dims=dims, depths=depths, kernels=kernels, embed_dim=embed_dim, num_classes=num_classes)


def mobileuvit_l(input_channel=3, dims=[32, 64, 128, 256], depths=[1, 1, 3, 3, 4], kernels=[3, 3, 7], embed_dim=512, num_classes=1):
    return MobileUViT(input_channel=input_channel, dims=[32, 64, 128, 256], depths=[1, 1, 3, 3, 4], kernels=[3, 3, 7], embed_dim=512, num_classes=num_classes)