import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.layers import DropPath
# from torch_dwconv import DepthwiseConv2d
# from mmcv.runner.base_module import BaseModule, ModuleList
# from mmcls.models.utils import resize_pos_embed
from einops import rearrange
from .scan import vertical_forward_scan, vertical_forward_scan_inv, vertical_backward_scan,vertical_backward_scan_inv
from .scan import horizontal_forward_scan, horizontal_forward_scan_inv, horizontal_backward_scan,horizontal_backward_scan_inv


T_MAX3 = 4096 #128*128 2048 均不可以 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]

from torch.utils.cpp_extension import load


wkv_cuda3 = load(name="wkv", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda', 'wkv_op.cpp'), os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda', 'wkv_cuda.cu')],
                verbose=True, extra_cuda_cflags=['-res-usage','--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX3}']) #'--maxrregcount 60', 

def q_shift(input, shift_pixel=1, gamma=1/4, resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    if resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
    input = input
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


def in_vertical_scan(input_tensor):
    """
    对输入张量进行垂直向内扫描
    """
    B, C, H, W = input_tensor.shape
    mid = H // 2
    # 将 NCHW 张量转置为 NHWC 格式，以便对倒数第二维（垂直方向）进行操作
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    top_half = input_tensor[:, :mid, :, :]
    bottom_half = torch.flip(input_tensor[:, mid:, :, :], dims=[1])
    # 拼接上下两部分
    # print("top_half",top_half.flatten())
    # print("bottom", bottom_half.flatten())    
    combined = torch.cat([top_half, bottom_half], dim=1)
    # 将张量转置为 (B * W, H, C) 形状
    transposed = rearrange(combined, 'B H W C -> (B W) H C').contiguous()
    return transposed.view(B, W * H, C) #.permute(0, 2, 1)

def q_shift(input, shift_pixel=1, gamma=1 / 4, resolution=None, scan_type='horizontal_forward'):
    """
    对输入张量进行 q_shift 操作，并根据 scan_type 选择不同的扫描方式
    """
    assert gamma <= 1 / 4
    if len(input.shape) == 3:
        B, N, C = input.shape
    N = input.shape[1]
    if resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid resolution for the given input shape.")
    if len(input.shape) == 3:
        input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
    # B, C, H, W = input.shape
    # output = torch.zeros_like(input)

    # if scan_type == 'horizontal_forward':
    #     output = horizontal_forward_scan(input)
    # elif scan_type == 'horizontal_backward':
    #     output = horizontal_backward_scan(input)
    # elif scan_type == 'vertical_forward':
    #     output = vertical_forward_scan(input)
    # elif scan_type == 'vertical_backward':
    #     output = vertical_backward_scan(input)
    # elif scan_type == 'in_horizontal':
    #     output = in_horizontal_scan(input)
    # elif scan_type == 'out_horizontal':
    #     output = out_horizontal_scan(input)
    # elif scan_type == 'in_vertical':
    #     output = in_vertical_scan(input)
    # elif scan_type == 'out_vertical':
    #     output = out_vertical_scan(input)
    # else:
    #     raise ValueError(f"Invalid scan_type: {scan_type}")

    # # output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    # # output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :, shift_pixel:W]
    # # output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3), 0:H - shift_pixel, :]
    # # output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:, int(C * gamma * 3):int(C * gamma * 4), shift_pixel:H, :]
    # # output[:, int(C * gamma * 4):,...] = input[:, int(C * gamma * 4):,...]
    output = in_vertical_scan(input)
    return output #.flatten(2).transpose(1, 2)

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX3
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda3.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX3
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda3.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

class SpatialInteractionMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, key_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        # 初始化 fancy 模式的权重
        with torch.no_grad():
            ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0

            # fancy time_decay
            decay_speed = torch.ones(self.n_embd)
            for h in range(self.n_embd):
                decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.spatial_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        # 设置 shift 相关参数
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        # 定义线性层
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        
        # 可选的 LayerNorm
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        
        # 输出线性层
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        # 初始化线性层的 scale_init
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        self.value.scale_init = 1

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution=None):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)
        rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv

class SpectralMixer(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, key_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        # 初始化 fancy 模式的权重
        with torch.no_grad():
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        # 设置 shift 相关参数
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        # 定义线性层
        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        # 可选的 LayerNorm
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

        # 初始化线性层的 scale_init
        self.value.scale_init = 0
        self.receptance.scale_init = 0
        self.key.scale_init = 1

    def forward(self, x, resolution=None):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        # 计算 key、value 和 receptance
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        # 计算最终输出
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


import torch
import torch.nn as nn
from einops import rearrange
# from .dwt import DWT_2D  # 假设 DWT_2D 已经实现
import pywt

# class DWT_2D(nn.Module):
#     def __init__(self, n_embd, wave='haar'):
#         super(DWT_2D, self).__init__()
#         self.n_embd = n_embd
#         self.groups = n_embd // 4 
#         w = pywt.Wavelet(wave)
#         dec_hi = torch.Tensor(w.dec_hi[::-1])  # 高通滤波器
#         dec_lo = torch.Tensor(w.dec_lo[::-1])  # 低通滤波器

#         # 构造四个滤波器核
#         w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)  # 低频子带
#         w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)  # 水平高频子带
#         w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)  # 垂直高频子带
#         w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # 对角线高频子带

#         # 注册滤波器核为模型的缓冲区
#         self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))
#         # self.conv_xx = nn.Conv2d(in_channels=n_embd,out_channels=self.groups, kernel_size=3,padding=1,groups=self.groups)

#     def forward(self, x):
#         dim = x.shape[1]  # 输入通道数
#         groups = dim // 4  # 每组输入通道数为 4

#         # 对输入通道进行分组卷积
#         x_ll = F.conv2d(x, self.w_ll.expand(groups, 1, 3, 3), stride=1, padding=1, groups=groups)
#         x_lh = F.conv2d(x, self.w_lh.expand(groups, 1, 3, 3), stride=1, padding=1, groups=groups)
#         x_hl = F.conv2d(x, self.w_hl.expand(groups, 1, 3, 3), stride=1, padding=1, groups=groups)
#         x_hh = F.conv2d(x, self.w_hh.expand(groups, 1, 3, 3), stride=1, padding=1, groups=groups)

#         # 将四个子带拼接在一起
#         x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
#         print("x_ll.shape", x_ll.shape, "x.shape", x.shape)
#         return x
    
class DWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])  # 高通滤波器
        dec_lo = torch.Tensor(w.dec_lo[::-1])  # 低通滤波器

        # 构造四个滤波器核
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)  # 低频子带
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)  # 水平高频子带
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)  # 垂直高频子带
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # 对角线高频子带

        # 注册滤波器核为模型的缓冲区
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        dim = x.shape[1]  # 输入通道数
        groups = dim // 4  # 每组输入通道数为 4

        # 对输入通道进行分组卷积
        x_ll = torch.nn.functional.conv2d(x, self.w_ll.expand(groups, 4, -1, -1), stride=2, groups=groups)
        x_lh = torch.nn.functional.conv2d(x, self.w_lh.expand(groups, 4, -1, -1), stride=2, groups=groups)
        x_hl = torch.nn.functional.conv2d(x, self.w_hl.expand(groups, 4, -1, -1), stride=2, groups=groups)
        x_hh = torch.nn.functional.conv2d(x, self.w_hh.expand(groups, 4, -1, -1), stride=2, groups=groups)

        # 将四个子带拼接在一起
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        print("x_ll.shape",x_ll.shape,"x.shape",x.shape)
        return x
    



# class FreqInteractionMixer(nn.Module):
#     def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy',
#                  key_norm=False, wave='haar'):
#         super().__init__()
#         self.layer_id = layer_id
#         self.n_layer = n_layer
#         self.n_embd = n_embd

#         # 初始化 DWT_2D 模块
#         self.dwt = DWT_2D(n_embd=n_embd,wave=wave)

#         # 定义线性层
#         hidden_sz = int(hidden_rate * n_embd)
#         # self.key = nn.Linear(n_embd * 4, hidden_sz, bias=False)  # 输入通道数变为 4 倍
#         # self.receptance = nn.Linear(n_embd * 4, n_embd, bias=False)  # 输入通道数变为 4 倍
#         # self.value = nn.Linear(hidden_sz, n_embd, bias=False)

#         self.key = nn.Linear(n_embd, hidden_sz, bias=False)
#         self.receptance = nn.Linear(n_embd, n_embd, bias=False)
#         self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        
#         # 可选的 LayerNorm
#         if key_norm:
#             self.key_norm = nn.LayerNorm(hidden_sz)
#         else:
#             self.key_norm = None

#     def forward(self, x, resolution=None):
#         # print("x shape",x.shape)
#         if resolution:
#             h, w = resolution
#         else:
#             L= x.shape[1]
#             h=w = int(math.sqrt(int(L)))
#         # 将输入特征图从 (B, L, C) 转换为 (B, C, H, W)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

#         # 使用 DWT_2D 进行多尺度分解
#         x = self.dwt(x)  # 输出形状: (B, C, H/2, W/2)

#         # 将特征图从 (B, C, H/2, W/2) 转换为 (B, L', C) 
#         x = rearrange(x, 'b c h w -> b (h w) c')

#         # 通道混合操作
#         k = self.key(x)  # 线性变换
#         k = torch.square(torch.relu(k))  # 非线性激活
#         if self.key_norm is not None:
#             k = self.key_norm(k)  # 归一化
#         kv = self.value(k)  # 线性变换

#         # 计算最终输出
#         print("kv.shape",kv.shape)
#         x = kv * torch.sigmoid(self.receptance(x)) 

#         return x

# class VEncBlock(nn.Module):
#     def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
#                  init_mode='fancy', key_norm=False,ffn_first=False):
#         super().__init__()
#         self.layer_id = layer_id
#         self.depth = depth  # 新增 depth 参数
#         self.ffn_first = ffn_first    
#         # LayerNorm 层
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.ln2 = nn.LayerNorm(n_embd)

#         # 空间混合模块
        
#         self.att = SpatialInteractionMix(
#             n_embd=n_embd,
#             n_layer=n_layer,
#             layer_id=layer_id,
#             shift_mode='q_shift',
#             key_norm=key_norm
#         )
        
        # # 频域混合模块
        # self.ffn = SpectralMixer(
        #     n_embd=n_embd,
        #     n_layer=n_layer,
        #     layer_id=layer_id,
        #     shift_mode='q_shift',
        #     key_norm=key_norm
        # )
#         # 可学习的缩放参数
#         self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
#         self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

#     def forward(self, x):
#         # print(x.shape)
#         if len(x.shape) == 4:
#             b,c,h,w = x.shape
#             x = rearrange(x, 'b c h w -> b (h w) c')
#         b, n, c= x.shape  #, 
#         h = w = int(n ** 0.5) 
#         resolution = (h, w)

#         # 空间混合
#         if self.ffn_first:
#             x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
#             x = x + self.gamma1 * self.att(self.ln1(x), resolution)
#             # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

#             # # 频域混合
#             # x = rearrange(x, 'b c h w -> b (h w) c')
            
#         else:
#             x = x + self.gamma1 * self.att(self.ln1(x), resolution)
#             # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

#             # # 频域混合
#             # x = rearrange(x, 'b c h w -> b (h w) c')
#             x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)            
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

#         return x

class VEncBlock(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                 init_mode='fancy', key_norm=False,ffn_first=False):
        super().__init__()
        self.layer_id = layer_id
        self.depth = depth  # 新增 depth 参数
        self.ffn_first = ffn_first    
        # LayerNorm 层
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # 空间混合模块
        
        self.att = SpatialInteractionMix(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            shift_mode='q_shift',
            key_norm=key_norm
        )
        
        # 频域混合模块
        self.ffn = SpectralMixer(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            shift_mode='q_shift',
            key_norm=key_norm
        )
        
        # 可学习的缩放参数
        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x):
        # print(x.shape)
        if len(x.shape) == 4:
            b,c,h,w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
        b, n, c= x.shape  #, 
        h = w = int(n ** 0.5) 
        resolution = (h, w)

        # 空间混合
        if self.ffn_first:
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
            x = x + self.gamma1 * self.att(self.ln1(x), resolution)
            # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

            # # 频域混合
            # x = rearrange(x, 'b c h w -> b (h w) c')
            
        else:
            x = x + self.gamma1 * self.att(self.ln1(x), resolution)
            # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

            # # 频域混合
            # x = rearrange(x, 'b c h w -> b (h w) c')
            
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x

class BinaryOrientatedRWKV2D(nn.Module):
    def __init__(self, n_embd, n_layer, shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy', drop_path=0., key_norm=True,ffn_first=False):
        super().__init__()
        self.block_forward_rwkv = VEncBlock(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=0,
            hidden_rate=4,
            init_mode=init_mode,
            key_norm=key_norm,
            ffn_first=ffn_first
        )
        self.block_backward_rwkv = VEncBlock(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=1,
            hidden_rate=4,
            init_mode=init_mode,
            key_norm=key_norm,
            ffn_first=ffn_first
        )

    def forward(self, z):
        B, C, H, W = z.shape
        # 这里需要搞定一下
        z_f = z.flatten(2).transpose(1, 2)
        z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)

        # rwkv_f = self.rwkv_forward(z_f)
        # rwkv_r = self.rwkv_reverse(z_r)
        rwkv_f = self.block_forward_rwkv(z_f)
        rwkv_r = self.block_backward_rwkv(z_r)
        
        fused = rwkv_f + rwkv_r
        if len(fused.shape) == 3:
            fused = fused.transpose(1, 2).view(B, C, H, W)
        return fused


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class LoRDBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = ConvBlock(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x

class LoRDBlockEnc(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = ConvBlock(ch_in, ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        p = self.pool(x)
        return x,p
    

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.up(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x, skip):
#         x = self.up(x)
#         x = torch.cat([x, skip], dim=1)
#         x = self.conv(x)
#         return x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c+out_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x

def v_enc_sym4_256(dims=[16, 32, 128, 160, 256]):
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                
                # self.pool = nn.AdaptiveAvgPool2d(1)
                # self.fc1 = nn.Conv2d(out_channels, self.after_red_out_c, kernel_size=1)
                # self.relu = nn.ReLU(inplace=True)
                # self.fc2 = nn.Conv2d(self.after_red_out_c, out_channels, kernel_size=1)
                # 使用 MultiSE 模块
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                # 直接使用 DWConv + Pwconv
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)

        def forward(self, x):
            x_residual = x
            # MultiSE 模块
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                # import pdb;pdb.set_trace()
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)

                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)

                
                # se_y = self.pool(y)
                # se_y = self.fc1(se_y)
                # se_y = self.relu(se_y)
                # se_y = self.fc2(se_y)
                # se_y = self.sigmoid(se_y)
                # y = y * se_y

            else:
                x = self.dwconv(x)
                y = self.pwconv(x)

            return x_residual + y if self.add else y 
            
            
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DecoderBlock, self).__init__()

            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )

        def forward(self, x, s):
            x = self.upsample(x)
            x = torch.cat([x, s], dim=1)
            x = self.conv(x)
            return x

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=True)
            self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=True)
            self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=True)
            self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            # self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            # self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            # self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            # self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=False)
            # self.e1 = EncoderBlock(dims[0], dims[1])
            # self.e2 = EncoderBlock(dims[1], dims[2])
            # self.e3 = EncoderBlock(dims[2], dims[3])
            # self.e4 = EncoderBlock(dims[3], dims[4])

            """ Decoder """
            self.s1 = DecoderBlock(dims[4], dims[3])
            self.s2 = DecoderBlock(dims[3], dims[2])
            self.s3 = DecoderBlock(dims[2], dims[1])
            self.s4 = DecoderBlock(dims[1], dims[0])

            # self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
            #                                     n_layer=12,
            #                                     shift_mode='q_shift',
            #                                     channel_gamma=1/4,
            #                                     shift_pixel=1,
            #                                     hidden_rate=4,
            #                                     init_mode="fancy",
            #                                     drop_path=0,
            #                                     key_norm=True)

            """ Output """
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            # import pdb;pdb.set_trace()
            p1 = self.stem(x)
            x2 = self.e1(p1)
            p2 = self.pool(x2)
            
            x3 = self.e2(p2)
            p3 = self.pool(x3)
            
            x4 = self.e3(p3)
            p4 = self.pool(x4)
            
            x5 = self.e4(p4)
            p5 = self.pool(x5)
            # p5 = self.Brwkv(p5)

            """ Decoder """
            s1 = self.s1(p5, x5)
            s2 = self.s2(s1, x4)
            s3 = self.s3(s2, x3)
            s4 = self.s4(s3, x2)

            """ Output """
            out = self.outconv(s4)
            return out

    return VEncSym()

def v_enc_256_fffse_dec_simple2(dims=[16,32,128,160,256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                
                # self.pool = nn.AdaptiveAvgPool2d(1)
                # self.fc1 = nn.Conv2d(out_channels, self.after_red_out_c, kernel_size=1)
                # self.relu = nn.ReLU(inplace=True)
                # self.fc2 = nn.Conv2d(self.after_red_out_c, out_channels, kernel_size=1)
                # 使用 MultiSE 模块
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                # 直接使用 DWConv + Pwconv
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            # MultiSE 模块
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                # import pdb;pdb.set_trace()
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)

                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)

                
                # se_y = self.pool(y)
                # se_y = self.fc1(se_y)
                # se_y = self.relu(se_y)
                # se_y = self.fc2(se_y)
                # se_y = self.sigmoid(se_y)
                # y = y * se_y

            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
                # y = self.pwconv(x)

            y =  x_residual + y if self.add else y 
            ypool = self.pool(y)
            
            return y,ypool
            
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DecoderBlock, self).__init__()

            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )

        def forward(self, x, s):
            x = self.upsample(x)
            x = torch.cat([x, s], dim=1)
            x = self.conv(x)
            return x

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            # self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            # self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            # self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            # self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=False)
            # self.e1 = EncoderBlock(dims[0], dims[1])
            # self.e2 = EncoderBlock(dims[1], dims[2])
            # self.e3 = EncoderBlock(dims[2], dims[3])
            # self.e4 = EncoderBlock(dims[3], dims[4])

            """ Decoder """
            self.s1 = DecoderBlock(dims[4], dims[3])
            self.s2 = DecoderBlock(dims[3], dims[2])
            self.s3 = DecoderBlock(dims[2], dims[1])
            self.s4 = DecoderBlock(dims[1], dims[0])

            # self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
            #                                     n_layer=12,
            #                                     shift_mode='q_shift',
            #                                     channel_gamma=1/4,
            #                                     shift_pixel=1,
            #                                     hidden_rate=4,
            #                                     init_mode="fancy",
            #                                     drop_path=0,
            #                                     key_norm=True)

            """ Output """
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            # import pdb;pdb.set_trace()
            p1 = self.stem(x)
            x2, p2 = self.e1(p1)
            x3, p3 = self.e2(p2)
            x4, p4 = self.e3(p3)
            x5, p5 = self.e4(p4)
            # p5 = self.Brwkv(p5)

            """ Decoder """
            s1 = self.s1(p5, x5)
            s2 = self.s2(s1, x4)
            s3 = self.s3(s2, x3)
            s4 = self.s4(s3, x2)

            """ Output """
            out = self.outconv(s4)
            return out

    return VEncSym()

# def v_enc_sym4_256(dims=[16, 32, 128, 160, 256]):
#     class DecoderBlock(nn.Module):
#         def __init__(self, in_channels, out_channels):
#             super(DecoderBlock, self).__init__()

#             self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.GELU(),
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.GELU()
#             )

#         def forward(self, x, s):
#             x = self.upsample(x)
#             x = torch.cat([x, s], dim=1)
#             x = self.conv(x)
#             return x

#     class VEncSym(nn.Module):
#         def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
#             super(VEncSym, self).__init__()
#             self.n_emb = dims[-1]
#             """ Shared Encoder """
#             self.stem = nn.Sequential(
#                 nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
#                 nn.BatchNorm2d(dims[0]),
#                 nn.GELU()
#             )
#             self.e1 = EncoderBlock(dims[0], dims[1])
#             self.e2 = EncoderBlock(dims[1], dims[2])
#             self.e3 = EncoderBlock(dims[2], dims[3])
#             self.e4 = EncoderBlock(dims[3], dims[4])

#             """ Decoder """
#             self.s1 = DecoderBlock(dims[4], dims[3])
#             self.s2 = DecoderBlock(dims[3], dims[2])
#             self.s3 = DecoderBlock(dims[2], dims[1])
#             self.s4 = DecoderBlock(dims[1], dims[0])

#             self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
#                                                 n_layer=12,
#                                                 shift_mode='q_shift',
#                                                 channel_gamma=1/4,
#                                                 shift_pixel=1,
#                                                 hidden_rate=4,
#                                                 init_mode="fancy",
#                                                 drop_path=0,
#                                                 key_norm=True)

#             """ Output """
#             self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

#         def forward(self, x):
#             """ Encoder """
#             p1 = self.stem(x)
            # x2, p2 = self.e1(p1)
            # x3, p3 = self.e2(p2)
            # x4, p4 = self.e3(p3)
            # x5, p5 = self.e4(p4)

#             p5 = self.Brwkv(p5)

#             """ Decoder """
#             s1 = self.s1(p5, x5)
#             s2 = self.s2(s1, x4)
#             s3 = self.s3(s2, x3)
#             s4 = self.s4(s3, x2)

#             """ Output """
#             out = self.outconv(s4)
#             return out

#     return VEncSym()

def v_enc_sym_256_out8(dims=[16,32,128,160,256]):
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DecoderBlock, self).__init__()
            
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            
        def forward(self,x,s):
            x = self.upsample(x)
            
            x = torch.cat([x,s],axis=1)
            
            x = self.conv(x)
            
            return x
            
    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            out_hidden_dim = 8
            """ Shared Encoder """
            # self.e1 = LoRDBlockEnc(input_channel, dims[0])
            # self.e2 = LoRDBlockEnc(dims[0], dims[1])
            # self.e3 = LoRDBlockEnc(dims[1], dims[2])
            # self.e4 = LoRDBlockEnc(dims[2], dims[3])
            # self.e5 = LoRDBlockEnc(dims[3], dims[4])
            self.n_emb = dims[-1]
            self.e1 = EncoderBlock(input_channel, dims[0])
            self.e2 = EncoderBlock(dims[0], dims[1])
            self.e3 = EncoderBlock(dims[1], dims[2])
            self.e4 = EncoderBlock(dims[2], dims[3])
            self.e5 = EncoderBlock(dims[3], dims[4])
            """ Decoder: Segmentation """
            self.s1 = DecoderBlock(dims[4] , dims[3])
            self.s2 = DecoderBlock(dims[3], dims[2])
            self.s3 = DecoderBlock(dims[2], dims[1])
            self.s4 = DecoderBlock(dims[1], dims[0])
            self.s5 = DecoderBlock(dims[0], out_hidden_dim)

            """ Decoder: Autoencoder """
            self.a1 = DecoderBlock(dims[4], dims[3])
            self.a2 = DecoderBlock(dims[3], dims[2])
            self.a3 = DecoderBlock(dims[2], dims[1])
            self.a4 = DecoderBlock(dims[1], dims[0])
            self.a5 = DecoderBlock(dims[0], out_hidden_dim)
            
            self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
                                                n_layer=12, 
                                                shift_mode='q_shift',
                                                channel_gamma=1/4,
                                                shift_pixel=1,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                drop_path=0,
                                                key_norm=True)

            """ Autoencoder attention map """
            self.m1 = nn.Sequential(
                nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m2 = nn.Sequential(
                nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m3 = nn.Sequential(
                nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m4 = nn.Sequential(
                nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m5 = nn.Sequential(
                nn.Conv2d(out_hidden_dim, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )

            """ Output """
            self.output1 = nn.Conv2d(out_hidden_dim, num_classes, kernel_size=1, padding=0)
            self.output2 = nn.Conv2d(out_hidden_dim, num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            x1, p1 = self.e1(x)
            
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)
            x4, p4 = self.e4(p3)
            x5, p5 = self.e5(p4)
            # print(x4.shape, p4.shape)
            # print(x5.shape, p5.shape)
            x5 = self.Brwkv(x5)
            p5 = self.Brwkv(p5)
            
            # import pdb;pdb.set_trace()
            """ Decoder 1 """
            # print(self.s1)
            s1 = self.s1(p5, x5)
            a1 = self.a1(p5, x5)
            m1 = self.m1(a1)
            x6 = s1 * m1

            """ Decoder 2 """
            s2 = self.s2(x6, x4)
            a2 = self.a2(a1, x4)
            m2 = self.m2(a2)
            x7 = s2 * m2

            """ Decoder 3 """
            s3 = self.s3(x7, x3)
            a3 = self.a3(a2, x3)
            m3 = self.m3(a3)
            x8 = s3 * m3

            """ Decoder 4 """
            s4 = self.s4(x8, x2)
            a4 = self.a4(a3, x2)
            m4 = self.m4(a4)
            x9 = s4 * m4

            """ Decoder 5 """
            s5 = self.s5(x9, x1)
            a5 = self.a5(a4, x1)
            m5 = self.m5(a5)
            x10 = s5 * m5

            """ Output """
            out1 = self.output1(x10)
            out2 = self.output2(a5)
            # print("pause")
            return out1 #, out2
    return VEncSym()

def v_enc_sym_l_384(dims=[24, 48, 96, 192, 384]):
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DecoderBlock, self).__init__()
            
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            
        def forward(self,x,s):
            x = self.upsample(x)
            
            x = torch.cat([x,s],axis=1)
            
            x = self.conv(x)
            
            return x
            
    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()

            """ Shared Encoder """
            # self.e1 = LoRDBlockEnc(input_channel, dims[0])
            # self.e2 = LoRDBlockEnc(dims[0], dims[1])
            # self.e3 = LoRDBlockEnc(dims[1], dims[2])
            # self.e4 = LoRDBlockEnc(dims[2], dims[3])
            # self.e5 = LoRDBlockEnc(dims[3], dims[4])
            self.n_emb = dims[-1]
            self.e1 = EncoderBlock(input_channel, dims[0])
            self.e2 = EncoderBlock(dims[0], dims[1])
            self.e3 = EncoderBlock(dims[1], dims[2])
            self.e4 = EncoderBlock(dims[2], dims[3])
            self.e5 = EncoderBlock(dims[3], dims[4])
            """ Decoder: Segmentation """
            self.s1 = DecoderBlock(dims[4] , dims[3])
            self.s2 = DecoderBlock(dims[3], dims[2])
            self.s3 = DecoderBlock(dims[2], dims[1])
            self.s4 = DecoderBlock(dims[1], dims[0])
            self.s5 = DecoderBlock(dims[0], 16)

            """ Decoder: Autoencoder """
            self.a1 = DecoderBlock(dims[4], dims[3])
            self.a2 = DecoderBlock(dims[3], dims[2])
            self.a3 = DecoderBlock(dims[2], dims[1])
            self.a4 = DecoderBlock(dims[1], dims[0])
            self.a5 = DecoderBlock(dims[0], 16)
            
            self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
                                                n_layer=12, 
                                                shift_mode='q_shift',
                                                channel_gamma=1/4,
                                                shift_pixel=1,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                drop_path=0,
                                                key_norm=True)

            """ Autoencoder attention map """
            self.m1 = nn.Sequential(
                nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m2 = nn.Sequential(
                nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m3 = nn.Sequential(
                nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m4 = nn.Sequential(
                nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m5 = nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )

            """ Output """
            self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
            self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            x1, p1 = self.e1(x)
            
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)
            x4, p4 = self.e4(p3)
            x5, p5 = self.e5(p4)
            # print(x4.shape, p4.shape)
            # print(x5.shape, p5.shape)
            x5 = self.Brwkv(x5)
            p5 = self.Brwkv(p5)
            
            # import pdb;pdb.set_trace()
            """ Decoder 1 """
            # print(self.s1)
            s1 = self.s1(p5, x5)
            a1 = self.a1(p5, x5)
            m1 = self.m1(a1)
            x6 = s1 * m1

            """ Decoder 2 """
            s2 = self.s2(x6, x4)
            a2 = self.a2(a1, x4)
            m2 = self.m2(a2)
            x7 = s2 * m2

            """ Decoder 3 """
            s3 = self.s3(x7, x3)
            a3 = self.a3(a2, x3)
            m3 = self.m3(a3)
            x8 = s3 * m3

            """ Decoder 4 """
            s4 = self.s4(x8, x2)
            a4 = self.a4(a3, x2)
            m4 = self.m4(a4)
            x9 = s4 * m4

            """ Decoder 5 """
            s5 = self.s5(x9, x1)
            a5 = self.a5(a4, x1)
            m5 = self.m5(a5)
            x10 = s5 * m5

            """ Output """
            out1 = self.output1(x10)
            out2 = self.output2(a5)
            # print("pause")
            return out1 #, out2
    return VEncSym()

class VEnc(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[32, 64, 128, 256,768], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(VEnc, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        
        self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        x5 = self.Brwkv(x5)
        p5 = self.Brwkv(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1, out2

# def lord_without(is_Brwkv=False):
#     mdl = VEnc(dims=[24, 48, 96, 192, 384], depths=[1, 1, 1, 3, 1])
#     if is_Brwkv==False:
#         mdl.Brwkv = nn.Identity()
        
#     return mdl
# def lord_l(dims=[48, 96, 192, 384 , 768], depths=[12, 12, 12, 3, 1]):
#     model = VEnc(dims=dims, depths=depths)
#     return VEnc


class VEnc__5(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(VEnc__5, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        # x4 = self.Brwkv_4(x4)
        # p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        # print("x5.shape",x5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2


class VEnc__5_Classification(nn.Module):
    def __init__(self, input_channel=3, num_classes=10, dims=[24, 48, 96, 192, 384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(VEnc__5_Classification, self).__init__()

        """ Shared Encoder """
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])

        """ RWKV Blocks """
        self.Brwkv_4 = BinaryOrientatedRWKV2D(
            n_embd=dims[3], 
            n_layer=12, 
            shift_mode='q_shift',
            channel_gamma=1/4,
            shift_pixel=1,
            hidden_rate=4,
            init_mode="fancy",
            drop_path=0,
            key_norm=True
        )
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(
            n_embd=dims[4], 
            n_layer=12, 
            shift_mode='q_shift',
            channel_gamma=1/4,
            shift_pixel=1,
            hidden_rate=4,
            init_mode="fancy",
            drop_path=0,
            key_norm=True
        )

        """ Global Average Pooling """
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化

        """ Output Layer """
        self.fc = nn.Linear(dims[4], num_classes)  # 全连接层，输出类别概率

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)

        """ RWKV Blocks """
        x5 = self.Brwkv_5(x5)

        """ Global Average Pooling """
        x5_pooled = self.global_pool(x5)  # 全局平均池化
        x5_pooled = x5_pooled.view(x5_pooled.size(0), -1)  # 展平

        """ Output """
        out = self.fc(x5_pooled)  # 全连接层，输出类别概率
        return out

class VEnc_4_5_woBrwkv(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(VEnc_4_5_woBrwkv, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        # x4 = self.Brwkv_4(x4)
        # p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # print("x5.shape", x5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2


    
    
class VEnc_4_5(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(VEnc_4_5, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x4 = self.Brwkv_4(x4)
        p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # print("x5.shape", x5.shape)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2
    
class VEnc_4_5_nlayer8(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(VEnc_4_5_nlayer8, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=8, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                            n_layer=8, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x4 = self.Brwkv_4(x4)
        p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # print("x5.shape", x5.shape)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2
    
class VEnc_4_5_nlayer8_woBrwkv(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(VEnc_4_5_nlayer8_woBrwkv, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=8, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                            n_layer=8, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        # x4 = self.Brwkv_4(x4)
        # p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # print("x5.shape", x5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2

class VEnc_4_5_rwkv_ffn_first(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super().__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True,
                                            ffn_first=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True,
                                            ffn_first=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x4 = self.Brwkv_4(x4)
        p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # print("x5.shape", x5.shape)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1

class VEnc_3_4_5(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super().__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        # self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_3 = BinaryOrientatedRWKV2D(n_embd=dims[2], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=dims[-1], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x3 = self.Brwkv_3(x3)
        p3 = self.Brwkv_3(p3)
        x4, p4 = self.e4(p3)
        x4 = self.Brwkv_4(x4)
        p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # print("x5.shape", x5.shape)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2


def v_enc_256_fffse_dec_resi2_rwkv_withbirwkv(dims=[16,32,128,160,256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                
                # self.pool = nn.AdaptiveAvgPool2d(1)
                # self.fc1 = nn.Conv2d(out_channels, self.after_red_out_c, kernel_size=1)
                # self.relu = nn.ReLU(inplace=True)
                # self.fc2 = nn.Conv2d(self.after_red_out_c, out_channels, kernel_size=1)
                # 使用 MultiSE 模块
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                # 直接使用 DWConv + Pwconv
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            # MultiSE 模块
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                # import pdb;pdb.set_trace()
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)

                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)

                
                # se_y = self.pool(y)
                # se_y = self.fc1(se_y)
                # se_y = self.relu(se_y)
                # se_y = self.fc2(se_y)
                # se_y = self.sigmoid(se_y)
                # y = y * se_y

            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
                # y = self.pwconv(x)

            y =  x_residual + y if self.add else y 
            ypool = self.pool(y)
            
            return y,ypool
            
    # class DecoderBlock(nn.Module):
    #     def __init__(self, in_channels, out_channels):
    #         super(DecoderBlock, self).__init__()

    #         self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    #         self.conv = nn.Sequential(
    #             nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(out_channels),
    #             nn.GELU(),
    #             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(out_channels),
    #             nn.GELU()
    #         )

    #     def forward(self, x, s):
    #         x = self.upsample(x)
    #         x = torch.cat([x, s], dim=1)
    #         x = self.conv(x)
    #         return x
    
    class DecoderBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super(DecoderBlock, self).__init__()

            self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            self.r1 = ResidualBlock(in_c+out_c, out_c)
            self.r2 = ResidualBlock(out_c, out_c)

        def forward(self, x, s):
            x = self.upsample(x)
            x = torch.cat([x, s], axis=1)
            x = self.r1(x)
            x = self.r2(x)

            return x    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x
        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=True)

            # self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            # self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            # self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            # self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            # self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            # self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            # self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            # self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            # self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            # self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            # self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            # self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            # self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=False)
            # self.e1 = EncoderBlock(dims[0], dims[1])
            # self.e2 = EncoderBlock(dims[1], dims[2])
            # self.e3 = EncoderBlock(dims[2], dims[3])
            # self.e4 = EncoderBlock(dims[3], dims[4])

            """ Decoder """
            self.s1 = DecoderBlock(dims[4], dims[3])
            self.s2 = DecoderBlock(dims[3], dims[2])
            self.s3 = DecoderBlock(dims[2], dims[1])
            self.s4 = DecoderBlock(dims[1], dims[0])

            # self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
            #                                     n_layer=8, 
            #                                     shift_mode='q_shift',
            #                                     channel_gamma=1/4,
            #                                     shift_pixel=1,
            #                                     hidden_rate=4,
            #                                     init_mode="fancy",
            #                                     drop_path=0,
            #                                     key_norm=True)
            
            self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                                n_layer=8, 
                                                shift_mode='q_shift',
                                                channel_gamma=1/4,
                                                shift_pixel=1,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                drop_path=0,
                                                key_norm=True)
            # self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
            #                                     n_layer=12,
            #                                     shift_mode='q_shift',
            #                                     channel_gamma=1/4,
            #                                     shift_pixel=1,
            #                                     hidden_rate=4,
            #                                     init_mode="fancy",
            #                                     drop_path=0,
            #                                     key_norm=True)

            """ Output """
            # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            # import pdb;pdb.set_trace()
            p1 = self.stem(x)
            x2, p2 = self.e1(p1)
            x3, p3 = self.e2(p2)
            x4, p4 = self.e3(p3)
            x5, p5 = self.e4(p4)
            # p5 = self.Brwkv(p5)
            # import pdb;pdb.set_trace()
            # # x5 = self.encoder5(x5)
            
            # d5 = self.Up5(p5)
            # d5 = torch.cat((x5, d5), dim=1)
            # d5 = self.Up_conv5(d5)

            # d4 = self.Up4(d5)
            # d4 = torch.cat((x4, d4), dim=1)
            # d4 = self.Up_conv4(d4)

            # d3 = self.Up3(d4)
            # d3 = torch.cat((x3, d3), dim=1)
            # d3 = self.Up_conv3(d3)

            # d2 = self.Up2(d3)
            # d2 = torch.cat((x2, d2), dim=1)
            # d2 = self.Up_conv2(d2)
            # d1 = self.Conv_1x1(d2)

            # return d1
            """ Decoder """
            p5 = self.Brwkv_5(p5)
            s1 = self.s1(p5, x5)
            s2 = self.s2(s1, x4)
            s3 = self.s3(s2, x3)
            s4 = self.s4(s3, x2)

            """ Output """
            out = self.outconv(s4)
            return out

    return VEncSym()


def v_enc_256_fffse_dec_resi2_rwkv_with2x4(dims=[16,32,128,160,256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                
                # self.pool = nn.AdaptiveAvgPool2d(1)
                # self.fc1 = nn.Conv2d(out_channels, self.after_red_out_c, kernel_size=1)
                # self.relu = nn.ReLU(inplace=True)
                # self.fc2 = nn.Conv2d(self.after_red_out_c, out_channels, kernel_size=1)
                # 使用 MultiSE 模块
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                # 直接使用 DWConv + Pwconv
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            # MultiSE 模块
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                # import pdb;pdb.set_trace()
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)

                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)

                
                # se_y = self.pool(y)
                # se_y = self.fc1(se_y)
                # se_y = self.relu(se_y)
                # se_y = self.fc2(se_y)
                # se_y = self.sigmoid(se_y)
                # y = y * se_y

            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
                # y = self.pwconv(x)

            y =  x_residual + y if self.add else y 
            ypool = self.pool(y)
            
            return y,ypool
            
    # class DecoderBlock(nn.Module):
    #     def __init__(self, in_channels, out_channels):
    #         super(DecoderBlock, self).__init__()

    #         self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    #         self.conv = nn.Sequential(
    #             nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(out_channels),
    #             nn.GELU(),
    #             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(out_channels),
    #             nn.GELU()
    #         )

    #     def forward(self, x, s):
    #         x = self.upsample(x)
    #         x = torch.cat([x, s], dim=1)
    #         x = self.conv(x)
    #         return x
    
    class DecoderBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super(DecoderBlock, self).__init__()

            self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            self.r1 = ResidualBlock(in_c+out_c, out_c)
            self.r2 = ResidualBlock(out_c, out_c)

        def forward(self, x, s):
            x = self.upsample(x)
            x = torch.cat([x, s], axis=1)
            x = self.r1(x)
            x = self.r2(x)

            return x    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            # 可学习的缩放参数
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            # LayerNorm 层
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            # 第一次 scan
            x1 = horizontal_forward_scan(x_4d)

            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1,x4d_shape)

            # 第二次 scan
            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2,x4d_shape)

            # 第三次 scan
            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3,x4d_shape)

            # 第四次 scan
            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4,x4d_shape)

            # 计算平均值
            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            # 扁平化处理
            x3dout = x4dout.flatten(2).transpose(1,2)
            # print("SpaBlockScan x3dout.shape",x3dout.shape)
            return x3dout
            
            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth  # 新增 depth 参数
            self.ffn_first = ffn_first    
            # LayerNorm 层
            # self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
            # 可学习的缩放参数
            # self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

            # 空间混合模块
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            # self.att = SpatialInteractionMix(
                # n_embd=n_embd,
                # n_layer=n_layer,
                # layer_id=layer_id,
                # shift_mode='q_shift',
                # key_norm=key_norm
            # )
            
            # 频域混合模块
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            # 必为4d
            b,c,h,w = x.shape
            
            x = self.allinone_spa(x)   

            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution = None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=True)

            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=False)
            self.e1 = EncoderBlock(dims[0], dims[1])
            self.e2 = EncoderBlock(dims[1], dims[2])
            self.e3 = EncoderBlock(dims[2], dims[3])
            self.e4 = EncoderBlock(dims[3], dims[4])

            """ Decoder """
            self.s1 = DecoderBlock(dims[4], dims[3])
            self.s2 = DecoderBlock(dims[3], dims[2])
            self.s3 = DecoderBlock(dims[2], dims[1])
            self.s4 = DecoderBlock(dims[1], dims[0])
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # self.bx4rwkv_4 = LoRABlock_f_plus_rev(n_embd=dims[-2],
            #                                     n_layer=8, 
            #                                     layer_id=0,
            #                                     hidden_rate=4,
            #                                     init_mode="fancy",
            #                                     key_norm=True)

            """ Output """
            # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            # import pdb;pdb.set_trace()
            p1 = self.stem(x)
            x2, p2 = self.e1(p1)
            x3, p3 = self.e2(p2)
            x4, p4 = self.e3(p3)
            x5, p5 = self.e4(p4)
            p5 = self.bx4rwkv(p5)
            # import pdb;pdb.set_trace()
            # # x5 = self.encoder5(x5)
            
            # d5 = self.Up5(p5)
            # d5 = torch.cat((x5, d5), dim=1)
            # d5 = self.Up_conv5(d5)

            # d4 = self.Up4(d5)
            # d4 = torch.cat((x4, d4), dim=1)
            # d4 = self.Up_conv4(d4)

            # d3 = self.Up3(d4)
            # d3 = torch.cat((x3, d3), dim=1)
            # d3 = self.Up_conv3(d3)

            # d2 = self.Up2(d3)
            # d2 = torch.cat((x2, d2), dim=1)
            # d2 = self.Up_conv2(d2)
            # d1 = self.Conv_1x1(d2)

            # return d1
            """ Decoder """
            # p5 = self.Brwkv_5(p5)
            s1 = self.s1(p5, x5)
            s2 = self.s2(s1, x4)
            s3 = self.s3(s2, x3)
            s4 = self.s4(s3, x2)

            """ Output """
            out = self.outconv(s4)
            return out

    return VEncSym()


def v_enc_256_fffse_dec_resi2_rwkv_with2x4(dims=[16,32,128,160,256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                
                # self.pool = nn.AdaptiveAvgPool2d(1)
                # self.fc1 = nn.Conv2d(out_channels, self.after_red_out_c, kernel_size=1)
                # self.relu = nn.ReLU(inplace=True)
                # self.fc2 = nn.Conv2d(self.after_red_out_c, out_channels, kernel_size=1)
                # 使用 MultiSE 模块
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                # 直接使用 DWConv + Pwconv
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            # MultiSE 模块
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                # import pdb;pdb.set_trace()
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)

                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)

                
                # se_y = self.pool(y)
                # se_y = self.fc1(se_y)
                # se_y = self.relu(se_y)
                # se_y = self.fc2(se_y)
                # se_y = self.sigmoid(se_y)
                # y = y * se_y

            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
                # y = self.pwconv(x)

            y =  x_residual + y if self.add else y 
            ypool = self.pool(y)
            
            return y,ypool
            
    # class DecoderBlock(nn.Module):
    #     def __init__(self, in_channels, out_channels):
    #         super(DecoderBlock, self).__init__()

    #         self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    #         self.conv = nn.Sequential(
    #             nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(out_channels),
    #             nn.GELU(),
    #             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(out_channels),
    #             nn.GELU()
    #         )

    #     def forward(self, x, s):
    #         x = self.upsample(x)
    #         x = torch.cat([x, s], dim=1)
    #         x = self.conv(x)
    #         return x
    
    class DecoderBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super(DecoderBlock, self).__init__()

            self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            self.r1 = ResidualBlock(in_c+out_c, out_c)
            self.r2 = ResidualBlock(out_c, out_c)

        def forward(self, x, s):
            x = self.upsample(x)
            x = torch.cat([x, s], axis=1)
            x = self.r1(x)
            x = self.r2(x)

            return x    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            # 可学习的缩放参数
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            # LayerNorm 层
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            # 第一次 scan
            x1 = horizontal_forward_scan(x_4d)

            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1,x4d_shape)

            # 第二次 scan
            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2,x4d_shape)

            # 第三次 scan
            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3,x4d_shape)

            # 第四次 scan
            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4,x4d_shape)

            # 计算平均值
            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            # 扁平化处理
            x3dout = x4dout.flatten(2).transpose(1,2)
            # print("SpaBlockScan x3dout.shape",x3dout.shape)
            return x3dout
            
            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth  # 新增 depth 参数
            self.ffn_first = ffn_first    
            # LayerNorm 层
            # self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
            # 可学习的缩放参数
            # self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

            # 空间混合模块
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )

            
            # 频域混合模块
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            # self.att = SpatialInteractionMix(
                # n_embd=n_embd,
                # n_layer=n_layer,
                # layer_id=layer_id,
                # shift_mode='q_shift',
                # key_norm=key_norm
            # )            

        def forward(self, x):
            # 必为4d
            b,c,h,w = x.shape
            
            x = self.allinone_spa(x)   

            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution = None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=True)

            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=False)
            self.e1 = EncoderBlock(dims[0], dims[1])
            self.e2 = EncoderBlock(dims[1], dims[2])
            self.e3 = EncoderBlock(dims[2], dims[3])
            self.e4 = EncoderBlock(dims[3], dims[4])

            """ Decoder """
            self.s1 = DecoderBlock(dims[4], dims[3])
            self.s2 = DecoderBlock(dims[3], dims[2])
            self.s3 = DecoderBlock(dims[2], dims[1])
            self.s4 = DecoderBlock(dims[1], dims[0])
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # self.bx4rwkv_4 = LoRABlock_f_plus_rev(n_embd=dims[-2],
            #                                     n_layer=8, 
            #                                     layer_id=0,
            #                                     hidden_rate=4,
            #                                     init_mode="fancy",
            #                                     key_norm=True)

            """ Output """
            # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            # import pdb;pdb.set_trace()
            p1 = self.stem(x)
            x2, p2 = self.e1(p1)
            x3, p3 = self.e2(p2)
            x4, p4 = self.e3(p3)
            x5, p5 = self.e4(p4)
            p5 = self.bx4rwkv(p5)
            # import pdb;pdb.set_trace()
            # # x5 = self.encoder5(x5)
            
            # d5 = self.Up5(p5)
            # d5 = torch.cat((x5, d5), dim=1)
            # d5 = self.Up_conv5(d5)

            # d4 = self.Up4(d5)
            # d4 = torch.cat((x4, d4), dim=1)
            # d4 = self.Up_conv4(d4)

            # d3 = self.Up3(d4)
            # d3 = torch.cat((x3, d3), dim=1)
            # d3 = self.Up_conv3(d3)

            # d2 = self.Up2(d3)
            # d2 = torch.cat((x2, d2), dim=1)
            # d2 = self.Up_conv2(d2)
            # d1 = self.Conv_1x1(d2)

            # return d1
            """ Decoder """
            # p5 = self.Brwkv_5(p5)
            s1 = self.s1(p5, x5)
            s2 = self.s2(s1, x4)
            s3 = self.s3(s2, x3)
            s4 = self.s4(s3, x2)

            """ Output """
            out = self.outconv(s4)
            return out

    return VEncSym()


def v_enc_128_fffse_dec_resi2_rwkv_withbirwkv(dims=[8, 16, 32, 64, 128]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                
                # self.pool = nn.AdaptiveAvgPool2d(1)
                # self.fc1 = nn.Conv2d(out_channels, self.after_red_out_c, kernel_size=1)
                # self.relu = nn.ReLU(inplace=True)
                # self.fc2 = nn.Conv2d(self.after_red_out_c, out_channels, kernel_size=1)
                # 使用 MultiSE 模块
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                # 直接使用 DWConv + Pwconv
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            # MultiSE 模块
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                # import pdb;pdb.set_trace()
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)

                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)

                
                # se_y = self.pool(y)
                # se_y = self.fc1(se_y)
                # se_y = self.relu(se_y)
                # se_y = self.fc2(se_y)
                # se_y = self.sigmoid(se_y)
                # y = y * se_y

            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
                # y = self.pwconv(x)

            y =  x_residual + y if self.add else y 
            ypool = self.pool(y)
            
            return y,ypool
            
    # class DecoderBlock(nn.Module):
    #     def __init__(self, in_channels, out_channels):
    #         super(DecoderBlock, self).__init__()

    #         self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    #         self.conv = nn.Sequential(
    #             nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(out_channels),
    #             nn.GELU(),
    #             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(out_channels),
    #             nn.GELU()
    #         )

    #     def forward(self, x, s):
    #         x = self.upsample(x)
    #         x = torch.cat([x, s], dim=1)
    #         x = self.conv(x)
    #         return x
    
    class DecoderBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super(DecoderBlock, self).__init__()

            self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            self.r1 = ResidualBlock(in_c+out_c, out_c)
            self.r2 = ResidualBlock(out_c, out_c)

        def forward(self, x, s):
            x = self.upsample(x)
            x = torch.cat([x, s], axis=1)
            x = self.r1(x)
            x = self.r2(x)

            return x    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x
        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=True)

            # self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            # self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            # self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            # self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            # self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            # self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            # self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            # self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            # self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            # self.e1 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            # self.e2 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            # self.e3 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            # self.e4 = MultiSE(dims[3],dims[4],if_deep_then_use=False)
            # self.e1 = EncoderBlock(dims[0], dims[1])
            # self.e2 = EncoderBlock(dims[1], dims[2])
            # self.e3 = EncoderBlock(dims[2], dims[3])
            # self.e4 = EncoderBlock(dims[3], dims[4])

            """ Decoder """
            self.s1 = DecoderBlock(dims[4], dims[3])
            self.s2 = DecoderBlock(dims[3], dims[2])
            self.s3 = DecoderBlock(dims[2], dims[1])
            self.s4 = DecoderBlock(dims[1], dims[0])

            # self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
            #                                     n_layer=8, 
            #                                     shift_mode='q_shift',
            #                                     channel_gamma=1/4,
            #                                     shift_pixel=1,
            #                                     hidden_rate=4,
            #                                     init_mode="fancy",
            #                                     drop_path=0,
            #                                     key_norm=True)
            
            self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                                n_layer=8, 
                                                shift_mode='q_shift',
                                                channel_gamma=1/4,
                                                shift_pixel=1,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                drop_path=0,
                                                key_norm=True)
            # self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
            #                                     n_layer=12,
            #                                     shift_mode='q_shift',
            #                                     channel_gamma=1/4,
            #                                     shift_pixel=1,
            #                                     hidden_rate=4,
            #                                     init_mode="fancy",
            #                                     drop_path=0,
            #                                     key_norm=True)

            """ Output """
            # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            # import pdb;pdb.set_trace()
            p1 = self.stem(x)
            x2, p2 = self.e1(p1)
            x3, p3 = self.e2(p2)
            x4, p4 = self.e3(p3)
            x5, p5 = self.e4(p4)
            # p5 = self.Brwkv(p5)
            # import pdb;pdb.set_trace()
            # # x5 = self.encoder5(x5)
            
            # d5 = self.Up5(p5)
            # d5 = torch.cat((x5, d5), dim=1)
            # d5 = self.Up_conv5(d5)

            # d4 = self.Up4(d5)
            # d4 = torch.cat((x4, d4), dim=1)
            # d4 = self.Up_conv4(d4)

            # d3 = self.Up3(d4)
            # d3 = torch.cat((x3, d3), dim=1)
            # d3 = self.Up_conv3(d3)

            # d2 = self.Up2(d3)
            # d2 = torch.cat((x2, d2), dim=1)
            # d2 = self.Up_conv2(d2)
            # d1 = self.Conv_1x1(d2)

            # return d1
            """ Decoder """
            p5 = self.Brwkv_5(p5)
            s1 = self.s1(p5, x5)
            s2 = self.s2(s1, x4)
            s3 = self.s3(s2, x3)
            s4 = self.s4(s3, x2)

            """ Output """
            out = self.outconv(s4)
            return out

    return VEncSym()


def v_enc_256_fffse_dec_fusion_with2x4(dims=[16, 32, 128, 160, 256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool
        
    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            x1 = horizontal_forward_scan(x_4d)
            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)

            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)

            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3, x4d_shape)

            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4, x4d_shape)

            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=True)

            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) # 这里p2就是没有 经过 enc 的x2
            x2, p3 = self.e2(p2) # p2 --enc x3 --pool p3 于是 p3 没有 经过 enc 的x3
            x3, p4 = self.e3(p3) # p3 --enc x4 --pool p4 于是 p4没有 经过 enc 的x4
            x4, p5 = self.e4(p4) # p4 --enc x5, --pool p5. p5 就是没有经过 enc 的x5 ==== 于是 x5 完全用不到
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()

def v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[16, 32, 128, 160, 256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool
        
    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            x1 = horizontal_forward_scan(x_4d)
            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)

            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)

            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3, x4d_shape)

            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4, x4d_shape)

            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) 
            x2, p3 = self.e2(p2)
            x3, p4 = self.e3(p3) 
            x4, p5 = self.e4(p4) 
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            x5 = self.bx4rwkv(x5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()


def v_enc_256_fffse_dec_sym_rwkv_with2x4(dims=[16, 32, 128, 160, 256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class Shallow(nn.Module):
        def __init__(self, in_channels, out_channels, reduction=8, split=2):
            super(Shallow, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            # self.if_deep_then_use = if_deep_then_use

            # if if_deep_then_use:
            #     self.sigmoid = nn.Sigmoid()
            #     self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
            #     self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
            #     self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            # else:
            self.bn_in_c = nn.BatchNorm2d(in_channels)
            self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
            self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
            self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
            self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
        
            # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            # if self.if_deep_then_use:
            #     x = self.pwconv1(x)
            #     x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
            #     x.extend(m(x[-1]) for m in self.m)
            #     x[0] = x[0] + x[1]
            #     x.pop(1)
            #     y = torch.cat(x, dim=1)
            #     y = self.pwconv2(y)
            # else:
            x = self.dwconv(x)
            x = self.bn_in_c(x)
            x = x_residual + x
            x = self.pwconv_in_in4(x)
            y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            # ypool = self.pool(y)
            return y #, ypool
        
    class Deeper(nn.Module):
        def __init__(self, in_channels, out_channels, reduction=8, split=2):
            super(Deeper, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            # self.if_deep_then_use = if_deep_then_use

            # if if_deep_then_use:
            self.sigmoid = nn.Sigmoid()
            self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
            self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
            self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            
            # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            x = self.pwconv1(x)
            x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
            x.extend(m(x[-1]) for m in self.m)
            x[0] = x[0] + x[1]
            x.pop(1)
            y = torch.cat(x, dim=1)
            y = self.pwconv2(y)
            y = x_residual + y if self.add else y
            # ypool = self.pool(y)
            return y #, ypool
        
    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool
        
    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            x1 = horizontal_forward_scan(x_4d)
            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)

            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)

            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3, x4d_shape)

            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4, x4d_shape)

            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0])
            self.e2 = MultiSE(dims[0],dims[1])
            self.e3 = MultiSE(dims[1],dims[2])
            self.e4 = MultiSE(dims[2],dims[3])
            self.e5 = MultiSE(dims[3],dims[4])
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            # p1 = self.stem(x)
            # x1, p2 = self.e1(p1) 
            # x2, p3 = self.e2(p2)
            # x3, p4 = self.e3(p3) 
            # x4, p5 = self.e4(p4) 
            # # p5 = self.bx4rwkv(p5)
            # x5, _ = self.e5(p5)
            # x5 = self.bx4rwkv(x5)
            import pdb;pdb.set_trace()
            x1 = self.stem(x)
            x1e = self.e1(x1)
            x2 = self.pool(x1e)
            x2e = self.e2(x2)
            x3 = self.pool(x2e)
            x3e = self.e3(x3)
            x4 = self.pool(x3e)
            x4e = self.e4(x4)
            x5  = self.pool(x4e)
            
            
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5], dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()

def u_rwkv(num_classes, input_channel=3, dims=[16, 32, 128, 160, 256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                # x = self.pwconv_in_in4(x)
                # y = self.pwconv_in4_out(x)
                y = self.pwconv(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool
        
    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            x1 = horizontal_forward_scan(x_4d)
            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)

            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)

            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3, x4d_shape)

            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4, x4d_shape)

            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) 
            x2, p3 = self.e2(p2)
            x3, p4 = self.e3(p3) 
            x4, p5 = self.e4(p4) 
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            x5 = self.bx4rwkv(x5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym(input_channel=input_channel, num_classes=num_classes)

def v_enc_256_fffse_dec_fusion_rwkv_with2x4_allx4(dims=[16, 32, 128, 160, 256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool
        
    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            x1 = horizontal_forward_scan(x_4d)
            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)

            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)

            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3, x4d_shape)

            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4, x4d_shape)

            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=False)
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) 
            x2, p3 = self.e2(p2)
            x3, p4 = self.e3(p3) 
            x4, p5 = self.e4(p4) 
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            x5 = self.bx4rwkv(x5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()

def v_enc_256_fffse_dec_fusion_rwkv_with2x4_single(dims=[16, 32, 128, 160, 256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool
        
    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            x1 = horizontal_forward_scan(x_4d)
            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)

            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)

            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3, x4d_shape)

            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4, x4d_shape)

            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            
            self.bx4rwkv = LoRABlock(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) 
            x2, p3 = self.e2(p2)
            x3, p4 = self.e3(p3) 
            x4, p5 = self.e4(p4) 
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            x5 = self.bx4rwkv(x5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()


def v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo(dims=[16, 32, 128, 160, 256]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool
        
    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            x1 = horizontal_forward_scan(x_4d)
            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)

            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)

            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3, x4d_shape)

            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4, x4d_shape)

            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) 
            x2, p3 = self.e2(p2)
            x3, p4 = self.e3(p3) 
            x4, p5 = self.e4(p4) 
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            # x5 = self.bx4rwkv(x5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()


def v_enc_128_fffse_decx2_fusion_rwkv_with2x4(dims=[8, 16, 32, 64, 128]): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*2, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*2, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool
        
    

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module): #HBYe changed
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 2, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 2),
                nn.Conv2d(ch_out * 2, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    # class SpaBlockVFScan(SpatialInteractionMix):
    #     def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
    #                 channel_gamma=1/4, shift_pixel=1, key_norm=True):
    #         super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
    #         self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
    #         self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
    #         self.ln1 = nn.LayerNorm(n_embd)
    #         self.ln2 = nn.LayerNorm(n_embd)

    #     def forward(self, x, resolution=None):
    #         x4d_shape = x.shape

    #         x_4d = x

    #         # x1 = horizontal_forward_scan(x_4d)
    #         # x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
    #         # x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)

    #         # x2 = horizontal_backward_scan(x_4d)
    #         # x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
    #         # x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)

    #         x3 = vertical_forward_scan(x_4d)
    #         x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
    #         x3_4d = vertical_forward_scan_inv(x3, x4d_shape)

    #         # x4 = vertical_backward_scan(x_4d)
    #         # x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
    #         # x4_4d = vertical_backward_scan_inv(x4, x4d_shape)

    #         # x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

    #         # x3dout = x4dout.flatten(2).transpose(1, 2)
    #         # return x3dout       
    #         return x3_4d.flatten(2).transpose(1, 2)    

    # class SpaBlockVFScan(SpatialInteractionMix):
    #     def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
    #                 channel_gamma=1/8, shift_pixel=1, key_norm=True):  # 增大channel_gamma压缩比例
    #         super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
    #         # 共享LayerNorm参数
    #         self.ln_share = nn.LayerNorm(n_embd)
            
    #     def forward(self, x, resolution=None):
    #         # 简化扫描路径，仅保留必要方向
    #         x_4d = x
    #         x3 = vertical_forward_scan(x_4d)
    #         x3 = x3 + self.gamma1 * super().forward(self.ln_share(x3), resolution)
    #         return vertical_forward_scan_inv(x3, x.shape).flatten(2).transpose(1, 2)     


    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x, resolution=None):
            x4d_shape = x.shape

            x_4d = x

            x1 = horizontal_forward_scan(x_4d)
            x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
            x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)

            x2 = horizontal_backward_scan(x_4d)
            x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
            x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)

            x3 = vertical_forward_scan(x_4d)
            x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
            x3_4d = vertical_forward_scan_inv(x3, x4d_shape)

            x4 = vertical_backward_scan(x_4d)
            x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
            x4_4d = vertical_backward_scan_inv(x4, x4d_shape)

            x4dout = torch.mean(torch.stack([x1_4d, x2_4d, x3_4d, x4_4d]), dim=0)

            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout            

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm           
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) 
            x2, p3 = self.e2(p2)
            x3, p4 = self.e3(p3) 
            x4, p5 = self.e4(p4) 
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            x5 = self.bx4rwkv(x5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()

def vscan_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[16, 32, 128, 160, 256],ab_scan='1_3'): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True, ab_scan=ab_scan):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            """
            ab_scan = 'H_V', 'H_HFlip','V_VFlip','HFlip_VFlip', 分别对应着 xi_4d 的开关：
            
            if '1_3', 1 and 3 on,  只跑出x1_4d x3_4d 将二者加和
            if '1_2' 1 and 2 on,  只跑出x1_4d x2_4d 将二者加和
            if '2_4', 2 and 4 on,  只跑出x2_4d x4_4d 将二者加和
            if '3_4', 3 and 4 on,  只跑出x3_4d x4_4d 将二者加和
            """
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ab_scan = ab_scan
            self.scan_on = self.parse_ab_scan(ab_scan)

        def parse_ab_scan(self, ab_scan):
            """
            解析 ab_scan 字符串，生成需要启用的方向索引列表。
            支持单数字（如 "1"）或双数字（如 "1_3"）格式。
            """
            if '_' in ab_scan:
                try:
                    return [int(part) for part in ab_scan.split('_')]
                except ValueError:
                    raise ValueError(f"Invalid ab_scan format: {ab_scan}. Expected digits separated by '_'.")
            else:
                try:
                    return [int(ab_scan)]
                except ValueError:
                    raise ValueError(f"Invalid ab_scan format: {ab_scan}. Expected a single digit or digits separated by '_'.")
        
        def forward(self, x, resolution=None):
            x4d_shape = x.shape
            x_4d = x
            outputs = []  # 存储所有启用方向的输出

            if 1 in self.scan_on:
                x1 = horizontal_forward_scan(x_4d)
                x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
                x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)
                outputs.append(x1_4d)

            if 2 in self.scan_on:
                x2 = horizontal_backward_scan(x_4d)
                x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
                x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)
                outputs.append(x2_4d)

            if 3 in self.scan_on:
                x3 = vertical_forward_scan(x_4d)
                x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
                x3_4d = vertical_forward_scan_inv(x3, x4d_shape)
                outputs.append(x3_4d)

            if 4 in self.scan_on:
                x4 = vertical_backward_scan(x_4d)
                x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
                x4_4d = vertical_backward_scan_inv(x4, x4d_shape)
                outputs.append(x4_4d)

            # 对所有启用方向的输出求平均
            x4dout = torch.mean(torch.stack(outputs), dim=0)
            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False, ab_scan=ab_scan):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm,
                ab_scan=ab_scan  # 将 ab_scan 参数传递给 SpaBlockScan          
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False,ab_scan=ab_scan):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first,
                ab_scan=ab_scan
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) 
            x2, p3 = self.e2(p2)
            x3, p4 = self.e3(p3) 
            x4, p5 = self.e4(p4) 
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            x5 = self.bx4rwkv(x5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()


def vscan_enc_256_fffse_dec_fusion_rwkv2_h_v(dims=[16, 32, 128, 160, 256],ab_scan='1_3'): #_withinpool
    class ConvBnoptinalAct(nn.Module):
        def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=None):
            super(ConvBnoptinalAct, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
            self.with_act = with_act

        def forward(self, x):
            x = self.conv(x)
            if self.with_act == 'GELU':
                x = F.gelu(x)
            return x

    class DWConv(nn.Module):
        def __init__(self, in_channels, out_channels, k=3, act=True):
            super(DWConv, self).__init__()
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)
            if act:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()

        def forward(self, x):
            return self.act(self.dwconv(x))


    class MultiSE(nn.Module):
        def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):
            super(MultiSE, self).__init__()
            self.after_red_out_c = int(out_channels / reduction)
            self.add = (in_channels == out_channels)
            self.if_deep_then_use = if_deep_then_use

            if if_deep_then_use:
                self.sigmoid = nn.Sigmoid()
                self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)
                self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
                self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))
            else:
                self.bn_in_c = nn.BatchNorm2d(in_channels)
                self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
                self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)
                self.pwconv_in_in4 = ConvBnoptinalAct(in_channels, in_channels*4, kernel_size=1, padding=0, with_act='GELU')
                self.pwconv_in4_out = ConvBnoptinalAct(in_channels*4, out_channels, kernel_size=1, padding=0, with_act='GELU')
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x_residual = x
            if self.if_deep_then_use:
                x = self.pwconv1(x)
                x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
                x.extend(m(x[-1]) for m in self.m)
                x[0] = x[0] + x[1]
                x.pop(1)
                y = torch.cat(x, dim=1)
                y = self.pwconv2(y)
            else:
                x = self.dwconv(x)
                x = self.bn_in_c(x)
                x = x_residual + x
                x = self.pwconv_in_in4(x)
                y = self.pwconv_in4_out(x)
            y = x_residual + y if self.add else y 
            ypool = self.pool(y)
            return y, ypool

    class UpsampleConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(UpsampleConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.GELU()
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class ChannelFusionConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(ChannelFusionConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out * 4),
                nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class SpaBlockScan(SpatialInteractionMix):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True, ab_scan=ab_scan):
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            """
            ab_scan = 'H_V', 'H_HFlip','V_VFlip','HFlip_VFlip', 分别对应着 xi_4d 的开关：
            
            if '1_3', 1 and 3 on,  只跑出x1_4d x3_4d 将二者加和
            if '1_2' 1 and 2 on,  只跑出x1_4d x2_4d 将二者加和
            if '2_4', 2 and 4 on,  只跑出x2_4d x4_4d 将二者加和
            if '3_4', 3 and 4 on,  只跑出x3_4d x4_4d 将二者加和
            """
            super().__init__(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, key_norm)
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ab_scan = ab_scan
            self.scan_on = self.parse_ab_scan(ab_scan)

        def parse_ab_scan(self, ab_scan):
            """
            解析 ab_scan 字符串，生成需要启用的方向索引列表。
            支持单数字（如 "1"）或双数字（如 "1_3"）格式。
            """
            if '_' in ab_scan:
                try:
                    return [int(part) for part in ab_scan.split('_')]
                except ValueError:
                    raise ValueError(f"Invalid ab_scan format: {ab_scan}. Expected digits separated by '_'.")
            else:
                try:
                    return [int(ab_scan)]
                except ValueError:
                    raise ValueError(f"Invalid ab_scan format: {ab_scan}. Expected a single digit or digits separated by '_'.")
        
        def forward(self, x, resolution=None):
            x4d_shape = x.shape
            x_4d = x
            outputs = []  # 存储所有启用方向的输出

            if 1 in self.scan_on:
                x1 = horizontal_forward_scan(x_4d)
                x1 = x1 + self.gamma1 * super().forward(self.ln1(x1), resolution)
                x1_4d = horizontal_forward_scan_inv(x1, x4d_shape)
                outputs.append(x1_4d)

            if 2 in self.scan_on:
                x2 = horizontal_backward_scan(x_4d)
                x2 = x2 + self.gamma1 * super().forward(self.ln1(x2), resolution)
                x2_4d = horizontal_backward_scan_inv(x2, x4d_shape)
                outputs.append(x2_4d)

            if 3 in self.scan_on:
                x3 = vertical_forward_scan(x_4d)
                x3 = x3 + self.gamma1 * super().forward(self.ln1(x3), resolution)
                x3_4d = vertical_forward_scan_inv(x3, x4d_shape)
                outputs.append(x3_4d)

            if 4 in self.scan_on:
                x4 = vertical_backward_scan(x_4d)
                x4 = x4 + self.gamma1 * super().forward(self.ln1(x4), resolution)
                x4_4d = vertical_backward_scan_inv(x4, x4d_shape)
                outputs.append(x4_4d)

            # 对所有启用方向的输出求平均
            x4dout = torch.mean(torch.stack(outputs), dim=0)
            x3dout = x4dout.flatten(2).transpose(1, 2)
            return x3dout

    class LoRABlock(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False, ab_scan=ab_scan):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first    
            self.ln2 = nn.LayerNorm(n_embd)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.allinone_spa_h = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm,
                ab_scan="1_2" # 将 ab_scan 参数传递给 SpaBlockScan          
            )
            self.allinone_spa_v = SpaBlockScan(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm,
                ab_scan="3_4" # 将 ab_scan 参数传递给 SpaBlockScan          
            )
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode='q_shift',
                key_norm=key_norm
            )
            

        def forward(self, x):
            b,c,h,w = x.shape
            x = self.allinone_spa_h(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.allinone_spa_v(x)   
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution=None)            
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        
    class LoRABlock_f_plus_rev(LoRABlock):
        def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                    init_mode='fancy', key_norm=False,ffn_first=False,ab_scan=ab_scan):
            super().__init__(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                hidden_rate=4,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first,
                ab_scan=ab_scan
            )
        
        def forward(self, x):
            b,c, h,w = x.shape
            x_r = x.transpose(2, 3).flatten(2).transpose(1, 2)        
            x_r4d = x_r.transpose(1, 2).view(b, c, h, w)
            return super().forward(x) + super().forward(x_r4d)        
    

    class VEncSym(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=dims, depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
            super(VEncSym, self).__init__()
            self.n_emb = dims[-1]
            """ Shared Encoder """
            self.stem = nn.Sequential(
                nn.Conv2d(input_channel, dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
            self.e1 = MultiSE(dims[0],dims[0],if_deep_then_use=False)
            self.e2 = MultiSE(dims[0],dims[1],if_deep_then_use=False)
            self.e3 = MultiSE(dims[1],dims[2],if_deep_then_use=False)
            self.e4 = MultiSE(dims[2],dims[3],if_deep_then_use=False)
            self.e5 = MultiSE(dims[3],dims[4],if_deep_then_use=True)
            
            self.bx4rwkv = LoRABlock_f_plus_rev(n_embd=self.n_emb,
                                                n_layer=8, 
                                                layer_id=0,
                                                hidden_rate=4,
                                                init_mode="fancy",
                                                key_norm=True)
            # Decoder
            self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
            self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
            self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
            self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
            self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
            self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
            self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
            self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
            self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

            """ Output """
            self.outconv = nn.Conv2d(dims[0], num_classes, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            p1 = self.stem(x)
            x1, p2 = self.e1(p1) 
            x2, p3 = self.e2(p2)
            x3, p4 = self.e3(p3) 
            x4, p5 = self.e4(p4) 
            # p5 = self.bx4rwkv(p5)
            x5, _ = self.e5(p5)
            x5 = self.bx4rwkv(x5)
            """ Decoder """
            # d5 = self.Up5(x5) self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3]) 
            d5 = self.Up5(x5)
            # print("x4.shape, d5.shape, x5.shape",x4.shape, d5.shape, x5.shape)
            d5 = torch.cat([x4, d5],  dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)
            d1 = self.Conv_1x1(d2)

            return d1

    return VEncSym()







if __name__ == "__main__":
#     x = torch.rand((1, 3, 256, 256)).cuda()
#     model = VEnc_4_5().cuda()
#     y1, y2 = model(x)
#     print(y1.shape, y2.shape)
#     print(count_params(model))
#     print("END")

    import os 
    os.environ['CUDA_VISIBLE_DEVICES']='7'

    import time 
    from thop import profile, clever_format
    
    '''
    !!!!!!!!
    Caution: Please comment out the code related to reparameterization and retain only the 5x5 convolutional layer in the OmniShift.
    !!!!!!!!
    '''
    
    
    # x=torch.zeros((1, 384, 16, 16)).type(torch.FloatTensor).cuda()

    
    # x = torch.zeros((1, 256, 384)).cuda()
    # n_embd = 384
    # self_gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True).cuda()
    # self_ffn = FreqInteractionMixer(n_embd=384,
    #                          n_layer=12, layer_id=0).cuda()
    # model = FreqInteractionMixer(n_embd=384,
    #                          n_layer=12, layer_id=0).cuda()
    # self_ln2 = nn.LayerNorm(384).cuda()
    # y = self_gamma2 * self_ffn(self_ln2(x),(16,16))
    # print("y.shape",y.shape)
    
    # model = BinaryOrientatedRWKV2D(n_embd=384, n_layer=12)
    
    # x=torch.zeros((1, 768,32,32)).type(torch.FloatTensor).cuda()
    # model = BinaryOrientatedRWKV2D(n_embd=768, n_layer=12)
    x=torch.zeros((1, 3, 256, 256)).type(torch.FloatTensor).cuda()    
    # model = v_enc_256_fffse_dec_resi2_rwkv_with2x4()
    model = v_enc_256_fffse_dec_fusion_rwkv_with2x4_simpchinchout()
    # model = v_enc_sym4_256()
    # x=torch.zeros((2, 256, 16, 16)).type(torch.FloatTensor).cuda()  
    # # model = CMRF(256,512)
    # model = MultiSE(256,512,if_deep_then_use=True)
    # model = v_enc_sym4_256(dims=[8, 16, 32, 64, 128])
    # model = v_enc_sym4_256(dims=[64,128,256,512,1024])
    # model = v_enc_sym4_256(dims=[16,32,128,160,256])
    model.cuda()
    
    since = time.time()
    # y=model(x)
    print("time", time.time()-since)
    
    flops, params = profile(model, inputs=(x, ))
    flops, params = clever_format([flops, params], '%.6f')
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")