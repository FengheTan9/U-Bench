import torch
from torch import nn
from torch.nn import functional as F


class RAC(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(RAC, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)
        b_x, c_x, h_x, w_x = x.size()
        x_attmap = x[:, :, (h_x * (100) // 768):(h_x * (550) // 768), (w_x * 26 // 768):(w_x * 767 // 768)].clone()
        x_attmap_1 = x[:, :, (h_x * (200) // 768):(h_x * (500) // 768), (w_x * 100 // 768):(w_x * 600 // 768)].clone()

        theta_x_1 = self.theta(x_attmap_1).view(batch_size, self.inter_channels, -1)
        theta_x_1 = theta_x_1.permute(0, 2, 1)

        g_x = self.g(x_attmap).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        phi_x = self.phi(x_attmap).view(batch_size, self.inter_channels, -1)

        f_1 = torch.matmul(theta_x_1, phi_x)
        f_div_C_1 = F.softmax(f_1, dim=-1)
        y_1 = torch.matmul(f_div_C_1, g_x)
        y_1 = y_1.permute(0, 2, 1).contiguous()
        y_1 = y_1.view(batch_size, self.inter_channels, *x_attmap_1.size()[2:])
        W_y_1 = self.W(y_1)

        return W_y_1

class RASAB(nn.Module):
    def __init__(self, in_channels):
        super(RASAB, self).__init__()
        self.in_channels = in_channels
        self.conv_z = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.RACBLOCK = RAC(in_channels)

    def forward(self, x):
        W_y_1 = self.RACBLOCK(x)
        x_mask = x.clone()
        b_x, c_x, h_x, w_x = x.size()
        x_mask[:, :, (h_x * (200) // 768):(h_x * (500) // 768), (w_x * 100 // 768):(w_x * 600 // 768)] = W_y_1
        z = torch.cat((x, x_mask), dim=1)
        z = self.conv_z(z)

        return z


if __name__ == '__main__':
    pass