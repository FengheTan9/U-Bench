import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.layers.weight_init as weight_init
from timm.models.layers import DropPath
from timm.models.registry import register_model
from torchvision import models
# Series informed activation function. Implemented by conv.
class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x), 
                self.weight, self.bias, padding=(self.act_num*2 + 1)//2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False):
        super().__init__()
        self.act_learn = 0
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        self.act = activation(dim_out, act_num, deploy=self.deploy)
 
    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            
            # We use leakyrelu to implement the deep training technique.
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            
            x = self.conv2(x)

        x = self.pool(x)
        x = self.act(x)
        return x
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight_init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1,3), self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True

class UpBlock(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, factor=2, deploy=False, ada_pool=None):
        super().__init__()
        self.act_learn = 0
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear')
        self.act = activation(dim_out, act_num, deploy=self.deploy)
    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            
            # We use leakyrelu to implement the deep training technique.
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            
            x = self.conv2(x)

        x = self.upsample(x)
        x = self.act(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1,3), self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True



class LV_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channel=3,  dims=[20*4,40*4,60*4,120*4], dims2=[80,40,24,16],
                 drop_rate=0, act_num=1, strides=[2,2,2], deploy=False):
        super().__init__()
        self.deploy = deploy
        mobile = models.mobilenet_v3_large(pretrained=True)
        self.firstconv = mobile.features[0]
        self.encoder1 = nn.Sequential(
                mobile.features[1],
                mobile.features[2],
            )
        self.encoder2 = nn.Sequential(
             mobile.features[3],
                mobile.features[4],
                mobile.features[5],
            )
        self.encoder3 =  nn.Sequential(
                mobile.features[6],
                mobile.features[7],
                mobile.features[8],
            mobile.features[9]
            )
        self.act_learn = 0
        self.stages = nn.ModuleList()
        self.up_stages1 = nn.ModuleList()
        self.up_stages2 = nn.ModuleList()
        for i in range(len(strides)):
            stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy)
            self.stages.append(stage)
        for i in range(len(strides)):
            stage = UpBlock(dim=dims[3-i], dim_out=dims[2-i], act_num=act_num, factor=strides[2-i],deploy=deploy)
            self.up_stages1.append(stage)
        for i in range(3):
            stage = UpBlock(dim=dims2[i], dim_out=dims2[i+1], act_num=act_num, factor=2,deploy=deploy)
            self.up_stages2.append(stage)
        self.depth = len(strides)
        self.final= nn.ModuleList()  
        self.final.append(UpBlock(dim=16, dim_out=16, act_num=act_num, factor=2))
        self.final.append(nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0))
    def change_act(self, m):
        for i in range(self.depth):
            self.stages[i].act_learn = m
        for i in range(self.depth):
            self.stages2[i].act_learn = m
        for i in range(3):
            self.stages3[i].act_learn = m
        for i in range(len(self.final)):
            self.final[i].act_learn = m
        self.final.act_learn = m
        self.act_learn = m

    def forward(self, x):
        x = self.firstconv(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e4 = self.encoder3(e2)
        encoder=[]
        for i in range(self.depth):
            encoder.append(e4)
            e4 = self.stages[i](e4)
        for i in range(self.depth):
            e4 = self.up_stages1[i](e4)
            e4=e4+encoder[2-i]
        e4=self.up_stages2[0](e4)+e2
        e4=self.up_stages2[1](e4)+e1
        e4= self.up_stages2[2](e4)
        for i in range(len(self.final)):
             e4=self.final[i](e4)
        return e4

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        for i in range(self.depth):
            self.stages[i].switch_to_deploy()
        for i in range(self.depth):
            self.up_stages1[i].switch_to_deploy()
        for i in range(self.depth):
            self.up_stages2[i].switch_to_deploy()
        self.final[0].switch_to_deploy()
        self.deploy = True



def lv_unet(num_classes, input_channel=3):
    model = LV_UNet(input_channel=input_channel, num_classes=num_classes)
    return model



