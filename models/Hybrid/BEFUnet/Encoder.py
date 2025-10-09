"""
Author: Omid Nejati Manzari
Date: Jun  2023
"""

import torch
import torch.nn as nn
import torchvision
from timm.models.layers import trunc_normal_
from .utils import *
from einops import rearrange
from einops.layers.torch import Rearrange
from .pidinet import PiDiNet
from .config import config_model, config_model_converted
from torch.hub import load_state_dict_from_url
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, dim, factor, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim * factor),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):
        
        super().__init__()
        
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio


        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None)
            self.layers.append(layer)

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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


class PyramidFeatures(nn.Module):
    def __init__(self, config, img_size = 224, in_channels=3):
        super().__init__()
        
        model_path = config.swin_pretrained_path
        self.swin_transformer = SwinTransformer(img_size,in_chans = 3)
        checkpoint = load_state_dict_from_url('https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth', progress=True)['model']
        unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight", "patch_embed.norm.bias",
                     "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
                     "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight", "layers.1.downsample.norm.bias",
                     "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight", "layers.2.downsample.norm.bias",
                     "layers.2.downsample.reduction.weight", "layers.3.downsample.norm.weight", "layers.3.downsample.norm.bias",
                     "layers.3.downsample.reduction.weight","norm.weight", "norm.bias"]

         
        pidinet = PiDiNet(30, config_model(config.pdcs), dil=12, sa=True).eval()
        
        #load weights
        #checkpoint_PDC = load_state_dict_from_url("https://raw.githubusercontent.com/hellozhuo/pidinet/refs/heads/master/trained_models/table5_pidinet-small.pth", progress=True, weights_only=True)
        checkpoint_PDC = torch.load("./models/Hybrid/BEFUnet/table5_pidinet-small.pth")
        #print(checkpoint_PDC)
        state_dict = checkpoint_PDC['state_dict']
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        
        pidinet.load_state_dict(new_state_dict)
        
        self.pidinet_layers = nn.ModuleList(pidinet.children())[:17]


        self.p1_ch = nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0] , kernel_size = 1, stride=4)
        self.p1_pm = PatchMerging((config.image_size // config.patch_size, config.image_size // config.patch_size), config.swin_pyramid_fm[0])
        self.p1_pm.state_dict()['reduction.weight'][:]= checkpoint["layers.0.downsample.reduction.weight"]
        self.p1_pm.state_dict()['norm.weight'][:]= checkpoint["layers.0.downsample.norm.weight"]
        self.p1_pm.state_dict()['norm.bias'][:]= checkpoint["layers.0.downsample.norm.bias"]        
        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1) 


        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1] , kernel_size = 1, stride=4)
        self.p2_pm = PatchMerging((config.image_size // config.patch_size // 2, config.image_size // config.patch_size // 2), config.swin_pyramid_fm[1])
        self.p2_pm.state_dict()['reduction.weight'][:]= checkpoint["layers.1.downsample.reduction.weight"]
        self.p2_pm.state_dict()['norm.weight'][:]= checkpoint["layers.1.downsample.norm.weight"]
        self.p2_pm.state_dict()['norm.bias'][:]= checkpoint["layers.1.downsample.norm.bias"]           
        

        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2] , config.swin_pyramid_fm[2] , kernel_size =  1, stride=4)
        self.p3_pm = PatchMerging((config.image_size // config.patch_size // 4, config.image_size // config.patch_size // 4), config.swin_pyramid_fm[2])
        self.p3_pm.state_dict()['reduction.weight'][:] = checkpoint["layers.2.downsample.reduction.weight"]
        self.p3_pm.state_dict()['norm.weight'][:] = checkpoint["layers.2.downsample.norm.weight"]
        self.p3_pm.state_dict()['norm.bias'][:] = checkpoint["layers.2.downsample.norm.bias"]

        self.p4_ch = nn.Conv2d(config.cnn_pyramid_fm[3], config.swin_pyramid_fm[3], kernel_size=1, stride=4)
        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[3])
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)    


        for key in list(checkpoint.keys()):
            if key in unexpected :
                del checkpoint[key]
        self.swin_transformer.load_state_dict(checkpoint)


    def forward(self, x):
        


        for i in range(4):
            x = self.pidinet_layers[i](x)


        # Level 1
        fm1 = x
        fm1_ch = self.p1_ch(x)
        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)               
        sw1 = self.swin_transformer.layers[0](fm1_reshaped)
        sw1_skipped = fm1_reshaped  + sw1
        norm1 = self.norm_1(sw1_skipped) 
        sw1_CLS = self.avgpool_1(norm1.transpose(1, 2))
        sw1_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw1_CLS) 
        fm1_sw1 = self.p1_pm(sw1_skipped)
        
        # Level 2
        fm1_sw2 = self.swin_transformer.layers[1](fm1_sw1)
        for i in range(4, 8):
            fm1 = self.pidinet_layers[i](fm1)

        fm2 = fm1
        fm2_ch = self.p2_ch(fm2)
        fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch) 
        fm2_sw2_skipped = fm2_reshaped  + fm1_sw2
        fm2_sw2 = self.p2_pm(fm2_sw2_skipped)
    
        # Level 3
        fm2_sw3 = self.swin_transformer.layers[2](fm2_sw2)
        for i in range(8, 12):
            fm2 = self.pidinet_layers[i](fm2)
            
        fm3 = fm2
        fm3_ch = self.p3_ch(fm3)
        fm3_reshaped = Rearrange('b c h w -> b (h w) c')(fm3_ch) 
        fm3_sw3_skipped = fm3_reshaped  + fm2_sw3
        fm3_sw3 = self.p3_pm(fm3_sw3_skipped)

        # Level 4
        fm3_sw4 = self.swin_transformer.layers[3](fm3_sw3)
        for i in range(12, 16):
            fm3 = self.pidinet_layers[i](fm3)

        fm4 = fm3
        fm4_ch = self.p4_ch(fm4)
        fm4_reshaped = Rearrange('b c h w -> b (h w) c')(fm4_ch)
        fm4_sw4_skipped = fm4_reshaped + fm3_sw4
        norm2 = self.norm_2(fm4_sw4_skipped)
        sw4_CLS = self.avgpool_2(norm2.transpose(1, 2))
        sw4_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw4_CLS)

        return [torch.cat((sw1_CLS_reshaped, sw1_skipped), dim=1), torch.cat((sw4_CLS_reshaped, fm4_sw4_skipped), dim=1)]

# MSF Module
class All2Cross(nn.Module):
    def __init__(self, config, img_size = 224 , in_chans=3, embed_dim=(96, 768), norm_layer=nn.LayerNorm):
        super().__init__()
        self.cross_pos_embed = config.cross_pos_embed
        self.pyramid = PyramidFeatures(config=config, img_size= img_size, in_channels=in_chans)
        
        n_p1 = (config.image_size // config.patch_size     ) ** 2  # default: 3136 
        n_p2 = (config.image_size // config.patch_size // 8) ** 2  # default: 49
        num_patches = (n_p1, n_p2)
        self.num_branches = 2
        
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
        
        total_depth = sum([sum(x[-2:]) for x in config.depth])
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_config in enumerate(config.depth):
            curr_depth = max(block_config[:-1]) + block_config[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_config, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                                  qkv_bias=config.qkv_bias, qk_scale=config.qk_scale, drop=config.drop_rate, 
                                  attn_drop=config.attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)

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
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward(self, x):
        xs = self.pyramid(x)

        if self.cross_pos_embed:
          for i in range(self.num_branches):
            xs[i] += self.pos_embed[i]

        for blk in self.blocks:
            xs = blk(xs)
        xs = [self.norm[i](x) for i, x in enumerate(xs)]

        return xs
