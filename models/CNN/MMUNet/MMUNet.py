from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchsummary import summary



class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, groups=dim // 4)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm2d(dim // 4)
        self.dwconv2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=5, padding=2, groups=dim // 4)
        self.norm2 = nn.BatchNorm2d(dim // 4)
        self.act2 = nn.GELU()
        self.dwconv3 = nn.Conv2d(dim // 4, dim // 4, kernel_size=7, padding=3, groups=dim // 4)
        self.norm3 = nn.BatchNorm2d(dim // 4)
        self.act3 = nn.GELU()

        self.norm4 = nn.BatchNorm2d(dim)

        self.pwconv1 = nn.Linear(dim, int(4 * dim))  # pointwise/1x1 convs, implemented with linear layers
        self.act4 = nn.GELU()
        self.pwconv2 = nn.Linear(int(4 * dim), dim)

        self.width = dim // 4



    #################### nn.GroupNorm
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = torch.split(x, self.width, 1)

        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        x1 = self.dwconv1(x1)
        x1 = self.norm1(x1)
        x1 = self.act1(x1)
        x2 = self.dwconv2(x1 + x2)
        x2 = self.norm2(x2)
        x2 = self.act2(x2)
        x3 = self.dwconv3(x2 + x3)
        x3 = self.norm3(x3)
        x3 = self.act3(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.norm4(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]

        x = self.pwconv1(x)
        #  print("sb3", x.size())
        x = self.act4(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        # x = self.CBAM(x)
        x = shortcut + x


        return x


class Block1(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, groups=dim // 4)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm2d(dim // 4)
        self.dwconv2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=5, padding=2, groups=dim // 4)
        self.norm2 = nn.BatchNorm2d(dim // 4)
        self.act2 = nn.GELU()
        self.dwconv3 = nn.Conv2d(dim // 4, dim // 4, kernel_size=7, padding=3, groups=dim // 4)
        self.norm3 = nn.BatchNorm2d(dim // 4)
        self.act3 = nn.GELU()

        self.norm4 = nn.BatchNorm2d(dim)

        self.pwconv1 = nn.Linear(dim, int(4 * dim))  # pointwise/1x1 convs, implemented with linear layers
        self.act4 = nn.GELU()
        self.pwconv2 = nn.Linear(int(4 * dim), dim)

        self.width = dim // 4
        self.norm_ea = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.k = 64
        self.linear_0 = nn.Conv1d(dim, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, dim, 1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = torch.split(x, self.width, 1)

        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        x1 = self.dwconv1(x1)
        x1 = self.norm1(x1)
        x1 = self.act1(x1)
        x2 = self.dwconv2(x1 + x2)
        x2 = self.norm2(x2)
        x2 = self.act2(x2)
        x3 = self.dwconv3(x2 + x3)
        x3 = self.norm3(x3)
        x3 = self.act3(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.norm4(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]


        x = self.pwconv1(x)
        #  print("sb3", x.size())
        x = self.act4(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = shortcut + x

        shortcut1 = x
        x = self.norm_ea(x)
        x2_conv1 = self.conv1(x)
        b, c, h, w = x2_conv1.size()
        n = h * w
        x2_conv1 = x2_conv1.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x2_conv1)  # b, k, n
        # linear_0是第一个memory unit
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n

        x2_conv1 = self.linear_1(attn)  # b, c, n
        # linear_1是第二个memory unit
        x2_conv1 = x2_conv1.view(b, c, h, w)
        x2_conv2 = self.conv2(x2_conv1)
        x2 = shortcut1 + x2_conv2
        x = F.gelu(x2)
        return x












class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_scale_init_value: float = 1e-6, use_erode=False):
        super(Up, self).__init__()
        # dim=out_channels*4
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2))
            self.conv = nn.Sequential(

                nn.Conv2d(in_channels, out_channels , kernel_size=1),
                nn.BatchNorm2d(out_channels ),
                Block1(dim=out_channels , drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
                Block1(dim=out_channels, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),

            )
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=4)
        self.softmax = nn.Softmax2d()
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.softmax1 = nn.Softmax2d()
        self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        self.linear1=nn.Conv2d(in_channels//2,in_channels//2,kernel_size=1)
        self.mlp = Mlp(in_channels // 2, in_channels // 2 , in_channels // 4)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        #  print("x1", x1.size())


        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x3 = x1 + x2

        x3_short = self.mlp(x3)

        x2_erode = -self.maxpool(self.maxpool(-self.softmax(x2)))
        x2_dilate = self.maxpool1(self.maxpool1(self.softmax1(x2)))
        x2 = torch.sigmoid(self.linear1(x2_erode + x2)) * x2 + torch.sigmoid(x2_erode) * torch.tanh(x2_dilate)
        x = torch.cat([x2,  x1], dim=1)

        x = self.conv(x) + x3_short
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Up1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_scale_init_value: float = 1e-6, use_erode=False):
        super(Up1, self).__init__()
        # dim=out_channels*4
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2))
            self.conv = nn.Sequential(

                nn.Conv2d(in_channels, out_channels , kernel_size=1),
                nn.BatchNorm2d(out_channels ),
                Block(dim=out_channels , drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
                Block(dim=out_channels, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),

            )
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=4)
         #   self.conv = DoubleConv(in_channels, out_channels)
        #    self.conv2_1=nn.Conv2d(3,1,kernel_size=3,padding=1)

        self.softmax = nn.Softmax2d()
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.softmax1 = nn.Softmax2d()
        self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        self.linear1=nn.Conv2d(in_channels//2,in_channels//2,kernel_size=1)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        #  print("x1", x1.size())

        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])




        x2_erode = -self.maxpool(self.maxpool(-self.softmax(x2)))
        x2_dilate = self.maxpool1(self.maxpool1(self.softmax1(x2)))
        x2 = torch.sigmoid(self.linear1(x2_erode + x2)) * x2 + torch.sigmoid(x2_erode) * torch.tanh(x2_dilate)
        x = torch.cat([x2,  x1], dim=1)



        x = self.conv(x)
        #  print("x", x.size())
        return x
class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_scale_init_value: float = 1e-6, use_erode=False):
        super(Up2, self).__init__()
        # dim=out_channels*4
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2))
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.conv = nn.Sequential(


                Block(dim=out_channels , drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),

                Block(dim=out_channels, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),

            )
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=4)
    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)


        x = self.conv(x1)
        return x



class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )




class EFM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EFM, self).__init__()
        self.up_x2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                   nn.Conv2d(in_channels=in_dim, out_channels=out_dim,kernel_size=3,bias=False,padding=1,groups=out_dim),
                                   nn.BatchNorm2d(out_dim),
                                   nn.GELU()
                                   )
        self.linear1 = nn.Conv2d(2 * out_dim, out_dim, kernel_size=1)
        self.maxpool1=nn.MaxPool2d(kernel_size=7,stride=1,padding=3)
        self.softmax=nn.Softmax2d()



    def forward(self, x1,x2,x3):
         # shortcut=x3
         x2=self.up_x2(x2)

         #
         x1_dilate = self.maxpool1(self.softmax(x1))
        # print(x1_dilate.size())
         x1_erode = -self.maxpool1(-self.softmax(x1))
       #  print(x1_erode.size())
         x2_dilate = self.maxpool1(self.softmax(x2))
       #  print(x2_dilate.size())
         x2_erode = -self.maxpool1(-self.softmax(x2))
        # print(x2_erode.size())

         x2_edge = x2_dilate - x2_erode
         x1_edge = x1_dilate - x1_erode

         new_edge=self.linear1(torch.cat((x2_edge,x1_edge),dim=1))

         x3=x3+new_edge


         return x3





class MMUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, bilinear: bool = True, base_channels=96,
                 layer_scale_init_value: float = 1e-6, se_ratio=0.25):
        super(MMUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.first_down = nn.Sequential(
            nn.Conv2d(in_channels, int(base_channels), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(base_channels)),
            Block(dim=int(base_channels), drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(base_channels),
            Block(dim=base_channels, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.GELU()
        )

        self.down0 = nn.Sequential(
            nn.Conv2d(base_channels, int(base_channels * 2), kernel_size=2, stride=2),
            nn.BatchNorm2d(int(base_channels * 2)),
            Block(dim=int(base_channels * 2), drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(2 * base_channels),
            Block(dim=int(base_channels * 2), drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.GELU()

        )
        self.down0_1 = nn.Sequential(
            nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(2 * base_channels),
            Block(dim=int(base_channels * 2), drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(2 * base_channels),
            Block(dim=base_channels * 2, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.GELU()
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=2, stride=2),  ##384
            nn.BatchNorm2d(base_channels * 4),
            Block(dim=base_channels * 4, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(4 * base_channels),
            Block(dim=base_channels * 4, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),

            nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=2, stride=2),  ##384
            nn.BatchNorm2d(base_channels * 8),
            Block1(dim=base_channels * 8, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(8 * base_channels),
            Block1(dim=base_channels * 8, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),

           nn.GELU()
        )
        factor = 2 if bilinear else 1
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16 // factor, kernel_size=2, stride=2),  ##384
            nn.BatchNorm2d(base_channels * 16 // factor),
            Block1(dim=base_channels * 16 // factor, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(base_channels * 16 // factor),
            Block1(dim=base_channels * 16 // factor, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.GELU()

        )

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up1(base_channels * 4, base_channels * 2, bilinear)
        self.up4 = Up1(base_channels * 4, base_channels, bilinear)
        self.up5 = Up2(base_channels , base_channels, bilinear)
        self.eam=EFM(base_channels*2,base_channels)
        self.out_conv = OutConv(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        #    print("x", x.size())
        x1 = self.first_down(x)  ##480 480 32
        #  print("x1", x1.size())
        x2 = self.down0(x1)  ##120 120 64
        # x2_SE=
        #   print("x2", x2.size())
        x3 = self.down0_1(x2)
        #   print("x3", x3.size())
        x4 = self.down1(x3)  ##60 60 128
        #   print("x3", x4.size())
        x5 = self.down2(x4)  ##30 3 0 256
        #  print("x5", x5.size())
        x6 = self.down3(x5)  ##15 15 512
        # print(x6.size())
        #x6 = self.aspp(x6)

        #   print(x6.size())
        #  print("x6", x6.size())
        # x6 = self.down4(x5)##7 7 512
        # print("x6", x6.size())
        x = self.up1(x6, x5)  ##30 30 256
        #   print("up1", x.size())
        x = self.up2(x, x4)  ##6060 128
        #  print("up2", x.size())
        x = self.up3(x, x3)  ## 120 64
        #  print("up3", x.size())
        x = self.up4(x, x2)  ##120 120 32
        #  print("up4", x.size())
        x = self.up5(x)  ##120 120 32

        x=self.eam(x1,x2,x)

        logits = self.out_conv(x)

        # return {"out": logits}
        return logits


def mmunet(input_channel=3, num_classes=2, base_channels=64):
    return  MMUNet(in_channels=input_channel, num_classes=num_classes, base_channels=base_channels)

if __name__ == "__main__":
    model = MMUNet(in_channels=3, num_classes=2, base_channels=64).to('cuda')
    # model = UNet(in_channels=3, num_classes=2, base_c=32).to('cuda')
    summary(model, input_size=(3, 224, 224))
    input = torch.randn(1, 3, 224, 224).to('cuda')
    flops, params = profile(model, inputs=(input,))
    print(flops/1e9,params/1e6)
