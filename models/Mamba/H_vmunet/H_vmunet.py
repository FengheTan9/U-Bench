import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
import os
import sys
import torch.fft
import math
from .vmamba import SS2D
from torch.hub import load_state_dict_from_url
import traceback

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
        # print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())
        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

        # print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:
    def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

    # print('Using PyTorch large kernel dw conv impl')

class H_SS2D(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0,d_state=16):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        num = len(self.dims)
        if num == 2:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16) 
        elif num == 3 :
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16) 
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=16) 
        elif num == 4 :
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16) 
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=16) 
            self.ss2d_3 = SS2D(d_model=self.dims[3], dropout=0, d_state=16) 
        elif num == 5 :
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16) 
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=16) 
            self.ss2d_3 = SS2D(d_model=self.dims[3], dropout=0, d_state=16) 
            self.ss2d_4 = SS2D(d_model=self.dims[4], dropout=0, d_state=16) 

        self.ss2d_in = SS2D(d_model=self.dims[0], dropout=0, d_state=16)

        self.scale = s

        print('[H_SS2D]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)


    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        x = x.permute(0, 2, 3, 1)
        x = self.ss2d_in(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]
            if i == 0 :
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_1(x)
                x = x.permute(0, 3, 1, 2)
            elif i == 1 :
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_2(x)
                x = x.permute(0, 3, 1, 2)
            elif i == 2 :
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_3(x)
                x = x.permute(0, 3, 1, 2)
            elif i == 3 :
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_4(x)
                x = x.permute(0, 3, 1, 2)            
        
        x = self.proj_out(x)

        return x

class Block(nn.Module):
    r""" H_VSS Block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, H_SS2D=H_SS2D):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.H_SS2D = H_SS2D(dim) 
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.H_SS2D(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1), 
                         self.avgpool(t2), 
                         self.avgpool(t3), 
                         self.avgpool(t4), 
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
            
        return att1, att2, att3, att4, att5
    
    
class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                          nn.Sigmoid())
    
    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]

    
class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        
    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_
    
    
class H_vmunet(nn.Module):
    
    def __init__(self, num_classes=1, input_channel=3,layer_scale_init_value=1e-6,H_SS2D=H_SS2D, block=Block,pretrained=None,
                 use_checkpoint=False, c_list=[8,16,32,64,128,256], depths=[2, 2, 2, 2],drop_path_rate=0.,
                split_att='fc', bridge=True):
        super().__init__()
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.bridge = bridge
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channel, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 


        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        if not isinstance(H_SS2D, list):
            H_SS2D = [partial(H_SS2D, order=2, s=1/3, gflayer=Local_SS2D), 
            partial(H_SS2D, order=3, s=1/3, gflayer=Local_SS2D), 
            partial(H_SS2D, order=4, s=1/3, h=24, w=13, gflayer=Local_SS2D),
            partial(H_SS2D, order=5, s=1/3, h=12, w=7, gflayer=Local_SS2D)]
        else:
            H_SS2D = H_SS2D
            assert len(H_SS2D) == 4

        if isinstance(H_SS2D[0], str):
            H_SS2D = [eval(h) for h in H_SS2D]

        if isinstance(block, str):
            block = eval(block)
        

 
        self.encoder3 = nn.Sequential(
            *[block(dim=c_list[1], drop_path=dp_rates[0 + j],
            layer_scale_init_value=layer_scale_init_value, H_SS2D=H_SS2D[0]) for j in range(depths[0])],
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )

        self.encoder4 = nn.Sequential(
            *[block(dim=c_list[2], drop_path=dp_rates[2 + j],
            layer_scale_init_value=layer_scale_init_value, H_SS2D=H_SS2D[1]) for j in range(depths[1])],
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
        )
        

        self.encoder5 = nn.Sequential(
            *[block(dim=c_list[3], drop_path=dp_rates[4 + j],
            layer_scale_init_value=layer_scale_init_value, H_SS2D=H_SS2D[2]) for j in range(depths[2])],
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
            )

        self.encoder6 = nn.Sequential(
            *[block(dim=c_list[4], drop_path=dp_rates[6 + j],
            layer_scale_init_value=layer_scale_init_value, H_SS2D=H_SS2D[3]) for j in range(depths[3])],
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1),
            )


        if bridge: 
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')
        
        self.decoder1 = nn.Sequential(
            *[block(dim=c_list[5], drop_path=dp_rates[6],
            layer_scale_init_value=layer_scale_init_value, H_SS2D=H_SS2D[3]) for j in range(depths[3])],
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1),
            )


        self.decoder2 = nn.Sequential(
            *[block(dim=c_list[4], drop_path=dp_rates[4+j],
            layer_scale_init_value=layer_scale_init_value, H_SS2D=H_SS2D[2]) for j in range(depths[2])],
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1),
            )

        self.decoder3 = nn.Sequential(
            *[block(dim=c_list[3], drop_path=dp_rates[2+j],
            layer_scale_init_value=layer_scale_init_value, H_SS2D=H_SS2D[1]) for j in range(depths[1])],
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
            )

        self.decoder4 = nn.Sequential(
            *[block(dim=c_list[2], drop_path=dp_rates[0+j],
            layer_scale_init_value=layer_scale_init_value, H_SS2D=H_SS2D[0]) for j in range(depths[0])],
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
            )

        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        return out0

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
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
        
class Local_SS2D(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

        self.SS2D = SS2D(d_model=dim // 2, dropout=0, d_state=16)


    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        B, C, a, b = x2.shape

        x2 = x2.permute(0, 2, 3, 1)

        x2 = self.SS2D(x2)

        x2 = x2.permute(0, 3, 1, 2)

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x
        
        
def h_vmunet(num_classes, input_channel=3):
    model = H_vmunet(input_channel=input_channel, num_classes=num_classes)
    return model          
        
        
        
        
        
        