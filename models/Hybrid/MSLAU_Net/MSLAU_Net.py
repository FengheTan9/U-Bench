# All rights reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import copy
from torch.hub import load_state_dict_from_url
from .msla import MSLA

layer_scale = False
init_value = 1e-6


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
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LFE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GFE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        # in_channels, key_channels, head_count, value_channels
        self.attn = MSLA(
            dim=dim, num_heads=num_heads
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
    
    
class Encoder(nn.Module):

    def __init__(self, depth=[4, 8, 11, 5], img_size=224, in_chans=3, num_classes=1, embed_dim=[64, 128, 256, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            LFE(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            LFE(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            GFE(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            GFE(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
        for i in range(depth[3])])
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        features = []
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        features.append(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        features.append(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        features.append(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        features.append(x)
        return features

class Conv_MLA(nn.Module):
    def __init__(self, embed_dim=[64, 128, 256, 512], mla_channels=64, norm_cfg=None):
        super(Conv_MLA, self).__init__()


        self.mla_p4 = nn.Sequential(nn.Conv2d(embed_dim[1], mla_channels, 1 ,bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                    )
        self.mla_p3 = nn.Sequential(nn.Conv2d(embed_dim[2], embed_dim[1], 1 ,bias=False),
                                    nn.BatchNorm2d(embed_dim[1]), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(embed_dim[1], mla_channels, 1, bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                    )
        self.mla_p2 = nn.Sequential(nn.Conv2d(embed_dim[3], embed_dim[2], 1 ,bias=False),
                                    nn.BatchNorm2d(embed_dim[2]), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(embed_dim[2], embed_dim[1], 1, bias=False),
                                    nn.BatchNorm2d(embed_dim[1]), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(embed_dim[1], mla_channels, 1, bias=False),
                                    nn.BatchNorm2d(mla_channels), nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                    )

        self.mla_p2_3x3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                    bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3_3x3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                    bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4_3x3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                    bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p5_3x3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                    bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())

    def forward(self,features):

        uni5, uni4, uni3, uni2 = features

        uni_mla_p4 = self.mla_p4(uni4)
        uni_mla_p3 = self.mla_p3(uni3)
        uni_mla_p2 = self.mla_p2(uni2)

        mla_p4_plus = uni5 + uni_mla_p4
        mla_p3_plus = mla_p4_plus + uni_mla_p3
        mla_p2_plus = mla_p3_plus + uni_mla_p2

        mla_p5 = self.mla_p5_3x3(uni5)
        mla_p4 = self.mla_p4_3x3(mla_p4_plus)
        mla_p3 = self.mla_p3_3x3(mla_p3_plus)
        mla_p2 = self.mla_p2_3x3(mla_p2_plus)

        return [mla_p5, mla_p4, mla_p3, mla_p2]

class MLAHead(nn.Module):
    def __init__(self, mla_channels=64):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.head3 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.head4 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.head5 = nn.Sequential(
            nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels), nn.ReLU())

    def forward(self, mla_list):
        head5 = F.interpolate(self.head2(
            mla_list[0]), 2*mla_list[0].shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head3(
            mla_list[1]), 2*mla_list[1].shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head4(
            mla_list[2]), 2*mla_list[2].shape[-1], mode='bilinear', align_corners=True)
        head2 = F.interpolate(self.head5(
            mla_list[3]), 2*mla_list[3].shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head5, head4, head3, head2], dim=1)

class MSLAU_net(nn.Module):

    def __init__(self, img_size=224, mla_channels=64,input_channel=3, num_classes=1):
        super(MSLAU_net, self).__init__()
        self.img_size = img_size
        self.norm_cfg = None
        self.mla_channels = mla_channels
        self.BatchNorm = nn.BatchNorm2d
        self.num_classes = num_classes
        self.in_chans = input_channel

        self.encoder = Encoder(
            depth=[4, 8, 11, 5], img_size=img_size, in_chans=input_channel, num_classes=1, embed_dim=[64, 128, 256, 512],
            head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None)
        self.conv_mla = Conv_MLA(embed_dim=[64, 128, 256, 512], mla_channels=64)
        self.mlahead = MLAHead(mla_channels=64)
        self.seg = nn.Conv2d(4 * self.mla_channels, self.num_classes, 3, padding=1)

    def forward(self, inputs):
        if inputs.size()[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        encoder_features = self.encoder(inputs)

        conv_mla_features = self.conv_mla(encoder_features)

        x = self.mlahead(conv_mla_features)
        x = self.seg(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear',
                              align_corners=True)
        return x

    def load_from(self):
        pretrained_path = '[URL]'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = load_state_dict_from_url(pretrained_path, progress=True)
            model_dict = self.encoder.state_dict()
            full_dict = copy.deepcopy(pretrained_dict['model'])
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,model_dict[k].shape))
                        del full_dict[k]
            msg = self.encoder.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")



def mslau_net(num_classes, input_channel=3):
    model = MSLAU_net(input_channel=input_channel, num_classes=num_classes, img_size=256)
    model.load_from()
    return model



