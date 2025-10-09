# coding=utf-8
import numpy as np
import torch
from torch import nn
import math
from torch.nn.functional import interpolate
from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
# from bn_lib.nn.modules import SynchronizedBatchNorm2d



class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias



class Post2d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(Post2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        # self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        # self.bn2 = nn.BatchNorm2d(n_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        return out



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class SN_CS_Parallel_Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_x, F_int):
        super(SN_CS_Parallel_Attention_block, self).__init__()
        reduction_ratio = 2
        F_int = F_g
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_x // reduction_ratio, F_x)
        )
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_g // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_g // reduction_ratio, F_x)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))

        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = channel_att_x + channel_att_g
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale

        # spacial-wise attention
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        x_after_spacial = x * psi
        out = self.relu(x_after_spacial + x_after_channel)
        return out

class CS_Parallel_Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_x, F_int):
        super(CS_Parallel_Attention_block, self).__init__()
        reduction_ratio = 2
        F_int = F_g
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_x // reduction_ratio, F_x)
        )
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_g // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_g // reduction_ratio, F_x)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))

        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = channel_att_x + channel_att_g
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale

        # spacial-wise attention
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        x_after_spacial = x * psi
        out = self.relu(x_after_spacial + x_after_channel)
        return out

class CS_Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_x, F_int):
        super(CS_Attention_block, self).__init__()
        reduction_ratio = 2
        # F_int = F_g
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_x // reduction_ratio, F_x)
        )
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_g // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_g // reduction_ratio, F_x)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # print avg_pool_x.Size()
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = channel_att_x + channel_att_g
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale

        # spacial-wise attention
        g1 = self.W_g(g)
        x1 = self.W_x(x_after_channel)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x_after_channel * psi
        return out



class FCAttention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_x1, F_x2, F_x3, F_x4, F_int):
        super(FCAttention_block, self).__init__()

        bn_momentum = 0.1

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )

        self.W_x1 = nn.Sequential(
            nn.Conv2d(F_x1, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )
        self.W_x2 = nn.Sequential(
            nn.Conv2d(F_x2, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )
        self.W_x3 = nn.Sequential(
            nn.Conv2d(F_x3, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )
        self.W_x4 = nn.Sequential(
            nn.Conv2d(F_x4, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1,momentum=bn_momentum),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x1, x2, x3, x4):
        g1 = self.W_g(g)
        x11 = self.W_x1(x1)
        x22 = self.W_x2(x2)
        x33 = self.W_x3(x3)
        x44 = self.W_x4(x4)
        psi = self.relu(g1 + x11 + x22 + x33 + x44)
        psi = self.psi(psi)
        x1  = x1 * psi
        x2  = x2 * psi
        x3  = x3 * psi
        x4  = x4 * psi
        return x1, x2, x3, x4



class Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


# norm_layer = partial(SynchronizedBatchNorm2d, momentum=0.1)



class SN_PostRes2d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(SN_PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = SwitchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = SwitchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                SwitchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class PostRes2d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out
    
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride = 1,bn_momentum=0.2):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class CU(nn.Module):

    def __init__(self, n_in, n_out, stride = 1,bn_momentum=0.2):
        super(CU, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(num_features=n_out,momentum=bn_momentum)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(num_features=n_out,momentum=bn_momentum)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class Upsample_1(nn.Module):
    def __init__(self,n_in,n_out,up_rate,bn_momentum=0.2):
        super(Upsample_1,self).__init__()

        self.conv=nn.Conv2d(n_in,n_out,kernel_size=1)
        self.bn=nn.BatchNorm2d(n_out,momentum=bn_momentum)
        self.relu=nn.ReLU(inplace=True)
        self.scale_factor=up_rate

    def forward(self,x):
        f=self.scale_factor
        out=self.conv(x)
        out=self.bn(out)
        out=self.relu(out)
        out=interpolate(input=out,scale_factor=(f,f),mode="bilinear",align_corners=True)
        return out

class SELayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SELayer,self).__init__()

        self.avg_pool=nn.AdaptiveAvgPool3d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        b,c,_,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1,1)
        return x*y.expand_as(x)
         

class SEResNetLayer(nn.Module):

    def __init__(self, n_in, n_out, stride = 1,bn_momentum=0.2):
        super(SEResNetLayer, self).__init__()

        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum))
        else:
            self.shortcut = None

        self.avg_pool=nn.AdaptiveAvgPool3d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        b,c,_,_,_=out.size()
        y=self.avg_pool(out).view(b,c)
        y=self.fc(y).view(b,c,1,1,1)
        scale=out*y.expand_as(x)

        scale += residual
        scale =self.relu(scale)
        return scale

class Rec3(nn.Module):
    def __init__(self, n0, n1, n2, n3, p = 0.0, integrate = True):
        super(Rec3, self).__init__()
        
        self.block01 = nn.Sequential(
            nn.Conv3d(n0, n1, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))

        self.block11 = nn.Sequential(
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
        
        self.block21 = nn.Sequential(
            nn.ConvTranspose3d(n2, n1, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
 
        self.block12 = nn.Sequential(
            nn.Conv3d(n1, n2, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block22 = nn.Sequential(
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block32 = nn.Sequential(
            nn.ConvTranspose3d(n3, n2, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
 
        self.block23 = nn.Sequential(
            nn.Conv3d(n2, n3, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.block33 = nn.Sequential(
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.relu = nn.ReLU(inplace = True)
        self.p = p
        self.integrate = integrate

    def forward(self, x0, x1, x2, x3):
        if self.p > 0 and self.training:
            coef = torch.bernoulli((1.0 - self.p) * torch.ones(8))
            out1 = coef[0] * self.block01(x0) + coef[1] * self.block11(x1) + coef[2] * self.block21(x2)
            out2 = coef[3] * self.block12(x1) + coef[4] * self.block22(x2) + coef[5] * self.block32(x3)
            out3 = coef[6] * self.block23(x2) + coef[7] * self.block33(x3)
        else:
            out1 = (1 - self.p) * (self.block01(x0) + self.block11(x1) + self.block21(x2))
            out2 = (1 - self.p) * (self.block12(x1) + self.block22(x2) + self.block32(x3))
            out3 = (1 - self.p) * (self.block23(x2) + self.block33(x3))

        if self.integrate:
            out1 += x1
            out2 += x2
            out3 += x3

        return x0, self.relu(out1), self.relu(out2), self.relu(out3)

def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

class Loss(nn.Module):
    def __init__(self, num_hard = 0):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss() # 二分类交叉熵损失函数``
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard # =2

    def forward(self, output, labels, train = True):
        # labels.shape = [batch_size,32,32,32,3,5]
        batch_size = labels.size(0)
        #print('labels.shape = ',labels.shape)
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)
        #print('after reshape,labels.shape = ',labels.shape)
        
        pos_idcs = labels[:, 0] > 0.5
        #print('pos_idcs = ',np.where(pos_idcs))
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5) # 过滤出标签是正例的网络的output（因为只有标签是正例的才参与计算）
        pos_labels = labels[pos_idcs].view(-1, 5) # 过滤出标签是正例的label

        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs] # 只有label是反例的输出的class部分才参与计算,其余地方需要回归的值不参与计算
        neg_labels = labels[:, 0][neg_idcs]
        
        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = self.sigmoid(neg_output)

        #classify_loss = self.classify_loss(
         #   torch.cat((pos_prob, neg_prob), 0),
          #  torch.cat((pos_labels[:, 0], neg_labels + 1), 0))
        if len(pos_output)>0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.data[0] for l in regress_losses]
            classify_loss = 0.5 * self.classify_loss(
            pos_prob, pos_labels[:, 0]) + 0.5 * self.classify_loss(
            neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.5).sum() # tp数
            pos_total = len(pos_prob) # 正例总数，即tp+fn

        else:
            regress_losses = [0,0,0,0]
            classify_loss =  0.5 * self.classify_loss(
            neg_prob, neg_labels + 1)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0,0,0,0]
        classify_loss_data = classify_loss.data[0]

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.5).sum() # tn数
        neg_total = len(neg_prob) # 反例总数，即tn+fp

        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total] # 损失函数返回的格式是[总的损失，分类损失，回归损失1,2,3,4，tp,tp+fn,tn,tn+fp]

class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output,thresh = -3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        
        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh # 因为sigmoid是在Loss中计算的，而测试的时候没有计算Loss，所以这里的thresh是-3，相当于只取得分大于sigmoid(-3)的部分。
        xx,yy,zz,aa = np.where(mask)
        
        output = output[xx,yy,zz,aa]
        if ismask:
            return output,[xx,yy,zz,aa]
        else:
            return output

        #output = output[output[:, 0] >= self.conf_th] 
        #bboxes = nms(output, self.nms_th)

def iou(box0, box1): #计算iou
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

