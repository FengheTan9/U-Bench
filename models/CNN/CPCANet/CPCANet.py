from einops import rearrange
from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_


class Mlp(nn.Module):
    """ Multilayer perceptron."""

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

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
    result.add_module('relu', nn.ReLU())
    return result

def fuse_bn(conv_or_fc, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    t = t.reshape(-1, 1, 1, 1)

    if len(t) == conv_or_fc.weight.size(0):
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std
    else:
        repeat_times = conv_or_fc.weight.size(0) // len(t)
        repeated = t.repeat_interleave(repeat_times, 0)
        return conv_or_fc.weight * repeated, (bn.bias - bn.running_mean * bn.weight / std).repeat_interleave(
            repeat_times, 0)

class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class RepBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels,in_channels,kernel_size=5,padding=2,groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,7),padding=(0,3),groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(7,1),padding=(3,0),groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,11),padding=(0,5),groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(11,1),padding=(5,0),groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,21),padding=(0,10),groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(21,1),padding=(10,0),groups=in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1),padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out



#   The common FFN Block used in many Transformer and MLP models.
class FFNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = conv_bn(in_channels, hidden_features, 1, 1, 0)
        self.ffn_fc2 = conv_bn(hidden_features, out_features, 1, 1, 0)
        self.act = act_layer()

    def forward(self, x):
        x = self.ffn_fc1(x)
        x = self.act(x)
        x = self.ffn_fc2(x)
        return x


#   The common FFN Block used in SegneXt models.
class FFNBlock2(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels,hidden_features,kernel_size=(1,1),padding=0)
        self.conv2 = nn.Conv2d(hidden_channels,out_features,kernel_size=(1,1),padding=0)
        self.dconv = nn.Conv2d(hidden_features,hidden_features,kernel_size=(3,3),padding=(1,1),groups=hidden_features)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dconv(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim,dim*2,kernel_size=3,stride=2,padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        x = x.permute(0,2,3,1).contiguous()
        x = F.gelu(x)
        x = self.norm(x)
        x=x.permute(0,3,1,2)
        x=self.reduction(x)
        return x
        
class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up=nn.ConvTranspose2d(dim,dim//2,2,2)
    def forward(self, x, H, W):
        x = x.permute(0,2,3,1).contiguous()
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        x = self.up(x)
        return x
        
class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 channelAttention_reduce=4,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,  
                 ):
        super().__init__()
        self.depth = depth
        self.dim=dim
        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                channelAttention_reduce=channelAttention_reduce
                )
            for i in range(depth)])
       
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        
        for blk in self.blocks:
            x = blk(x)
    
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x,  H, W, x_down, Wh, Ww
        else:
            return x,  H, W, x, H, W

class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 channelAttention_reduce=4,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                ):
        super().__init__()
        self.depth = depth
        self.dim=dim
        
        # build blocks
        self.blocks = nn.ModuleList([
            Block_up(dim=dim)
            for i in range(depth)])
        
        self.Upsample = upsample(dim=2*dim, norm_layer=norm_layer)
    def forward(self, x,skip, H, W):
        x_up = self.Upsample(x, H, W)
        x = x_up + skip
        H, W = H * 2, W * 2
        
        for blk in self.blocks:         
            x = blk(x)
            
        return x, H, W
        
class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x

class project_up(nn.Module):
    def __init__(self,in_dim,out_dim,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.ConvTranspose2d(in_dim,out_dim,kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        

        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x
        
    

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_block=int(np.log2(patch_size[0]))
        self.project_block=[]
        self.dim=[int(embed_dim)//(2**i) for i in range(self.num_block)]
        self.dim.append(in_chans)
        self.dim=self.dim[::-1] # in_ch, embed_dim/2, embed_dim or in_ch, embed_dim/4, embed_dim/2, embed_dim
        
        for i in range(self.num_block)[:-1]:
            self.project_block.append(project(self.dim[i],self.dim[i+1],2,1,nn.GELU,nn.LayerNorm,False))
        self.project_block.append(project(self.dim[-2],self.dim[-1],2,1,nn.GELU,nn.LayerNorm,True))
        self.project_block=nn.ModuleList(self.project_block)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, self.patch_size[0] - W % self.patch_size[0]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        for blk in self.project_block:
            x = blk(x)
       
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x



class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size=(224,224),
                 patch_size=(4,4),
                 in_chans=1  ,
                 embed_dim=96,
                 depths=(3, 3, 3, 3),
                 channelAttention_reduce=4,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer, pretrain_img_size[1] // patch_size[1] // 2 ** i_layer),
                depth=depths[i_layer],
                channelAttention_reduce=channelAttention_reduce,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
                )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

  
    def forward(self, x):
        """Forward function."""
        
        x = self.patch_embed(x)
        down=[]
       
        Wh, Ww = x.size(2), x.size(3)
        
        x = self.pos_drop(x)
        
      
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out=x_out.permute(0,2,3,1)
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0,3, 1, 2).contiguous()
                down.append(out)
        return down


class decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=(4,4),
                 depths=(3,3,3),
                 channelAttention_reduce=4,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        
        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths)-i_layer-1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths)-i_layer-1), pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths)-i_layer-1)),
               
                depth=depths[i_layer],
                channelAttention_reduce=channelAttention_reduce,
                drop_path=dpr[sum(
                    depths[:(len(depths)-i_layer-1)]):sum(depths[:(len(depths)-i_layer)])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
                )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
    def forward(self,x,skips):
        outs=[]
        H, W = x.size(2), x.size(3)     
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:          
            layer = self.layers[i]           
            x, H, W,  = layer(x,skips[i], H, W)
            outs.append(x)
        return outs




        
class final_patch_expanding(nn.Module):
    def __init__(self,dim,num_class,patch_size):
        super().__init__()
        self.num_block=int(np.log2(patch_size[0]))-2
        self.project_block=[]
        self.dim_list=[int(dim)//(2**i) for i in range(self.num_block+1)]
        # dim, dim/2, dim/4
        for i in range(self.num_block):
            self.project_block.append(project_up(self.dim_list[i],self.dim_list[i+1],nn.GELU,nn.LayerNorm,False))
        self.project_block=nn.ModuleList(self.project_block)
        self.up_final=nn.ConvTranspose2d(self.dim_list[-1],num_class,4,4)

    def forward(self,x):
        for blk in self.project_block:
            x = blk(x)
        x = self.up_final(x) 
        return x    

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.,ffn_expand=4, channelAttention_reduce=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.bn = nn.BatchNorm2d(num_features=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_block = FFNBlock2(dim,dim*ffn_expand)
        self.repmlp_block = RepBlock(in_channels=dim, out_channels=dim, channelAttention_reduce=channelAttention_reduce)

    def forward(self, x):
        input = x.clone()

        x = self.bn(x)
        x = self.repmlp_block(x)
        x = input + self.drop_path(x)
        x2 = self.bn(x)
        x2 = self.ffn_block(x2)
        x = x + self.drop_path(x2)

        return x

class Block_up(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1) # conv
        self.bn = nn.BatchNorm2d(num_features=dim)
        self.relu = nn.ReLU()
    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
                                         
class CPCANet(nn.Module):
    def __init__(self, 
                 input_channel=3, 
                 embedding_dim=96, 
                 num_classes=1, 
                 deep_supervision=False, 
                 conv_op=nn.Conv2d):
        super(CPCANet, self).__init__()
        
        # Don't uncomment conv_op
        self.num_input_channels = input_channel
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.do_ds = deep_supervision          
        self.embed_dim = embedding_dim
        self.depths=[3,3,12,3]
        self.crop_size = [256,256]
        self.patch_size=[4, 4]
        self.channelAttention_reduce = 16
        # if window size of the encoder is [7,7,14,7], then decoder's is [14,7,7]. In short, reverse the list and start from the index of 1 
        self.model_down = encoder(
                                  pretrain_img_size=self.crop_size,
                                  embed_dim=self.embed_dim,
                                  patch_size=self.patch_size,
                                  depths=self.depths,
                                  in_chans=self.num_input_channels,
                                  channelAttention_reduce=self.channelAttention_reduce

        )
                                        
        self.decoder = decoder(
                               pretrain_img_size=self.crop_size,
                               embed_dim=self.embed_dim,
                               patch_size=self.patch_size,
                               depths=[2,2,1],
                               channelAttention_reduce=self.channelAttention_reduce
                              )
   
        self.final=[]
        for i in range(len(self.depths)-1):
            self.final.append(final_patch_expanding(self.embed_dim*2**i,self.num_classes,patch_size=self.patch_size))
        self.final=nn.ModuleList(self.final)
        
    def forward(self, x):
        seg_outputs=[]
        skips = self.model_down(x)
        neck=skips[-1]
        out=self.decoder(neck,skips)
        for i in range(len(out)):  
            seg_outputs.append(self.final[-(i+1)](out[i]))
        return seg_outputs[-1]


def cpcanet(num_classes, input_channel=3):
    return CPCANet(num_classes=num_classes, input_channel=input_channel) 

        
        
   

 