from .vmamba import VSSM
# from vmamba import VSSM # debug use
import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):#[f1,f2,f3,f4]
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans





class VMUNetV2(nn.Module):
    def __init__(self, 
                 input_channel=3, 
                 num_classes=1,
                 mid_channel = 48,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 deep_supervision=True
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        # SDI
        self.ca_1 = ChannelAttention(2*mid_channel)
        self.sa_1 = SpatialAttention() 

        self.ca_2 = ChannelAttention(4*mid_channel)
        self.sa_2 = SpatialAttention()
        # TODO 320 or mid_channel * 8?
        self.ca_3 = ChannelAttention(8*mid_channel)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(16*mid_channel)
        self.sa_4 = SpatialAttention()
        
        self.Translayer_1 = BasicConv2d(2*mid_channel, mid_channel, 1)
        self.Translayer_2 = BasicConv2d(4*mid_channel, mid_channel, 1)
        self.Translayer_3 = BasicConv2d(8*mid_channel, mid_channel, 1)
        self.Translayer_4 = BasicConv2d(16*mid_channel, mid_channel, 1)  

        self.sdi_1 = SDI(mid_channel)
        self.sdi_2 = SDI(mid_channel)
        self.sdi_3 = SDI(mid_channel)
        self.sdi_4 = SDI(mid_channel)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(mid_channel, num_classes, 1, 1) for _ in range(4)])
        
        
    
        self.deconv2 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv6 = nn.ConvTranspose2d(num_classes, num_classes, 3, stride=2, padding=1, output_padding=1)
        
        
        
        self.vmunet = VSSM(in_chans=input_channel,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def forward(self, x):
        seg_outs = []
        if x.size()[1] == 1: # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1,3,1,1)
        f1, f2, f3, f4 = self.vmunet(x) #  f1 [2, 64, 64, 96]  f3  [2, 8, 8, 768]  [b h w c]
        # b h w c --> b c h w
        f1 = f1.permute(0, 3, 1, 2) # f1 [2, 96, 64, 64]
        f2 = f2.permute(0, 3, 1, 2)
        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)
        
        # use sdi  
        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1) # f1 [2, 48, 64, 64]
        
        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2) # f2 [2, 48, 32, 32]

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3) # f3 [2, 48, 16, 16]

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4) # f4 [2, 48, 8, 8]
        
        
        f41 = self.sdi_4([f1, f2, f3, f4], f4) # [2, 48, 8, 8]
        f31 = self.sdi_3([f1, f2, f3, f4], f3) # [2, 48, 16, 16]
        f21 = self.sdi_2([f1, f2, f3, f4], f2) # [2, 48, 32, 32]
        f11 = self.sdi_1([f1, f2, f3, f4], f1) # [2, 48, 64, 64]
        
        # 函数seg_outs 输出列表也是 seg_outs 只是名字相同
        seg_outs.append(self.seg_outs[0](f41)) # seg_outs[0] [2, 1, 8, 8]

        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y)) # seg_outs[1] [2, 1, 16, 16]

        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y)) # seg_outs[2] [2, 1, 32, 32]

        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y)) # seg_outs[3] [2, 1, 64, 64]
        
        for i, o in enumerate(seg_outs): # 4 倍上采样
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear')

        if self.deep_supervision:
            
            temp = seg_outs[::-1]  # 0 [2, 1, 256, 256] 1 [2, 1, 128, 128]
            out_0 = temp[0]
            out_1 = temp[1]
            out_1 = self.deconv6(out_1)
            return out_0 + out_1  #  [2, 1, 256, 256]
        else:
            return seg_outs[-1]

    
    def load_from(self):
        
        pretrained_path = "https://huggingface.co/FengheTan9/U-Stone/resolve/main/vmamba_small_e238_ema.pth"
        model_dict = self.vmunet.state_dict()
    
        modelCheckpoint = load_state_dict_from_url(pretrained_path, progress=True)
        pretrained_dict = modelCheckpoint['model']
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数 
        print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
        self.vmunet.load_state_dict(model_dict)

        not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
        #print('Not loaded keys:', not_loaded_keys)
        print("encoder loaded finished!")

            
def vmunetv2(num_classes, input_channel=3):
    model = VMUNetV2(input_channel=input_channel, num_classes=num_classes)
    model.load_from()
    return model 

    
    
