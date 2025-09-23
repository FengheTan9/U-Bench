import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from .bra_block import Block
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x
class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, embed_dim, num_heads, drop_path_rate=0.,
                 layer_scale_init_value=-1, topks=[8, 8, -1, -1], qk_dims=[96, 192, 384, 768], n_win=7,
                 kv_per_wins=[2, 2, -1, -1], kv_downsample_kernels=[4, 2, 1, 1], kv_downsample_ratios=[4, 2, 1, 1],
                 kv_downsample_mode='ada_avgpool', param_attention='qkvo', param_routing=False, diff_routing=False,
                 soft_routing=False, pre_norm=True, mlp_ratios=[4, 4, 4, 4], mlp_dwconv=False, side_dwconv=5,
                 qk_scale=None, before_attn_dwconv=3, auto_pad=False, norm_layer=nn.LayerNorm, upsample=None,
                 use_checkpoint=False
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # stochastic depth 随机深度衰减规则
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum([depth]))]
        cur = 0
        # build blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  input_resolution=input_resolution,
                  drop_path=dp_rates[cur + i],
                  layer_scale_init_value=layer_scale_init_value,
                  num_heads=num_heads,
                  n_win=n_win,
                  qk_dim=qk_dims,
                  qk_scale=qk_scale,
                  kv_per_win=kv_per_wins,
                  kv_downsample_ratio=kv_downsample_ratios,
                  kv_downsample_kernel=kv_downsample_kernels,
                  kv_downsample_mode=kv_downsample_mode,
                  topk=topks,
                  param_attention=param_attention,
                  param_routing=param_routing,
                  diff_routing=diff_routing,
                  soft_routing=soft_routing,
                  mlp_ratio=mlp_ratios,
                  mlp_dwconv=mlp_dwconv,
                  side_dwconv=side_dwconv,
                  before_attn_dwconv=before_attn_dwconv,
                  pre_norm=pre_norm,
                  auto_pad=auto_pad)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x