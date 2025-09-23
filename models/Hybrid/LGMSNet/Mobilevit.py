import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from einops import rearrange

# helpers




class CMUNeXtBlock_MK_resiual2(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
    def __init__(self, ch_in, ch_out,
                group_kernel_sizes= [3, 5, 7, 9],
                ): # ch_in 需要是4的倍数
        super().__init__()
        self.ch_in=ch_in
        assert self.ch_in // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.ch_in // 4
        self.group_kernel_sizes = group_kernel_sizes
        self.norm_act = nn.Sequential(nn.GroupNorm(4, ch_in),nn.GELU())

        self.local_dwc = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0], 
                padding=group_kernel_sizes[0] // 2, groups=self.group_chans)
        self.global_dwc_s = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1],
                padding=group_kernel_sizes[1] // 2, groups=self.group_chans)
        self.global_dwc_m = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[2], 
                padding=group_kernel_sizes[2] // 2, groups=self.group_chans)
        self.global_dwc_l = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[3], 
                padding=group_kernel_sizes[3] // 2, groups=self.group_chans)
        self.conv_1x1_bn=conv_1x1_bn(ch_in,self.group_chans)
        self.conv_nxn_bn=conv_nxn_bn(self.group_chans,ch_in)


    def forward(self, x):
        xclone=x.clone()
        b, c, h_, w_ = x.size()
        l_x, g_x_s, g_x_m, g_x_l = torch.split(x, self.group_chans, dim=1)
        x_attn =self.norm_act(torch.cat((
            self.local_dwc(l_x),
            self.global_dwc_s(g_x_s),
            self.global_dwc_m(g_x_m),
            self.global_dwc_l(g_x_l),
        ), dim=1)+xclone)
        # 然后用1*1交互一下channel，这块也可以用注意力什么的想一想
        x_attn = self.conv_1x1_bn(x_attn)
        x=self.conv_nxn_bn(x_attn)
        return x+xclone



class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    if kernel_size==2:
        return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
        )
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
class conv_nxn_bn_2(nn.Module):
    def __init__(self,inp, oup, kernel_size=2, stride=1,padding_position="left"):
        super().__init__()
        self.conv=nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.SiLU()
            )
        self.padding_position=padding_position
    def forward(self, x):
        if self.padding_position=="left":
            x_padded = F.pad(x, (0, 1, 0, 1))
        else:
            x_padded = F.pad(x, (1, 0, 1, 0))
        x=self.conv(x_padded)

        return x

class CONV_nn_bn(nn.Module):
    def __init__(self,inp, oup, kernel_size=3, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(inp, (inp+oup)//4, 1, 1, 0, bias=False)
        self.conv2=nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False)

    def forward(self, x):
        return self.conv2(x)
# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


## 还可以考虑，q,k 和patch如何
class Attention_with_Vconv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.conv=nn.Conv2d(inner_dim,inner_dim,3,1,1)  # 考虑有或没有feature间的交融，也就是是否用DWConv
        self.height=int(math.sqrt(dim))  # TODO 偷个懒，这个输入维度h=w，所以直接开根
        self.weight=int(math.sqrt(dim))
    def conv_V(self,v):
        v=rearrange(v, 'b p h n d -> b p n (h d)').transpose(2,3)
        b,p,n,num=v.shape
        v=rearrange(v,'b p dim (height weight) -> (b p) dim height weight',b=b,p=p,height=int(math.sqrt(num)), weight=int(math.sqrt(num)))  # TODO 偷个懒，这个输入维度h=w，所以直接开根
        v=self.conv(v)
        v=rearrange(v,'(b p) dim height weight -> b p dim (height weight)',b=b,p=p).transpose(2,3)
        v=rearrange(v, 'b p n (h d)->b p h n d ', h=self.heads)
        return v


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        v=self.conv_V(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Attention_with_VDWconv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.conv=nn.Conv2d(inner_dim,inner_dim,3,1,1,groups=inner_dim)  # 考虑有或没有feature间的交融，也就是是否用DWConv

    def conv_V(self,v):   # TODO: 这里的实现方式，是否合理？
        # print(v.shape)
        v=rearrange(v, 'b p h n d -> b p n (h d)').transpose(2,3)
        b,p,n,num=v.shape
        v=rearrange(v,'b p dim (height weight) -> (b p) dim height weight',b=b,p=p,height=int(math.sqrt(num)), weight=int(math.sqrt(num)))  # TODO 偷个懒，这个输入维度h=w，所以直接开根
        v=self.conv(v)
        v=rearrange(v,'(b p) dim height weight -> b p dim (height weight)',b=b,p=p).transpose(2,3)
        v=rearrange(v, 'b p n (h d)->b p h n d ', h=self.heads)
        return v


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        v=self.conv_V(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


Attention_model={
    "Attention":Attention,
    "Attention_with_Vconv":Attention_with_Vconv,
    "Attention_with_VDWconv":Attention_with_VDWconv,
}

class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: [URL]
    Based on: [URL]
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,attention="Attention"):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_model[attention](dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class MobileViTBlocktem_CT(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, spilt_list=[96,32],dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.conv_spilt = DWConv2d_BN_ReLU(spilt_list[1],spilt_list[1],3)
        self.transformer = Transformer(spilt_list[0], depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+ channel, channel, kernel_size)
        self.spilt_list=spilt_list

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        x1,x2 = torch.split(x,self.spilt_list, dim=1)
        x2 = self.conv_spilt(x2)+x2


        # Global representations
        _, _, h, w = x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x1 = self.transformer(x1)        
        x1 = rearrange(x1, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = torch.cat((x1, x2), 1)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x
