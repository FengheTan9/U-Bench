import torch
import torch.nn as nn
import torch.nn.functional as F

class se_block_without_shorcut(nn.Module):
    def __init__(self, channels = 512, mid_channels = 256) -> None:
        super(se_block_without_shorcut, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(channels, mid_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mid_channels, channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, _, _ = x.shape
        out = self.pool(x).view(B, C)
        out = self.linear1(out)
        out = self.relu(out)
 
        out = self.linear2(out).unsqueeze(-1)
        out = self.sigmoid(out)
        return out
    
class X_spatial(nn.Module):
    def __init__(self, transformer_chans = 128, cnn_chans = 64 , out_chan = 64):
        super(X_spatial, self).__init__()
        self.cnn_conv = nn.Conv2d(in_channels = cnn_chans, out_channels = transformer_chans, kernel_size=(5, 5), stride = 1, padding = 2)
        self.trans_conv = nn.Conv2d(in_channels = transformer_chans, out_channels = cnn_chans, kernel_size=(3, 3), stride = 1, padding = 1)
        self.out = nn.Conv2d(in_channels=cnn_chans + transformer_chans, out_channels=out_chan, kernel_size=(3, 3), stride=1, padding = 1)
    def forward(self, trans, cnn):
        cnn_branch_fuse = cnn + self.trans_conv(trans)
        trans_branch_fuse = trans + self.cnn_conv(cnn)
        combine = torch.cat((cnn_branch_fuse, trans_branch_fuse), dim = 1)
        out = self.out(combine)
        return out

# class X_spatial(nn.Module):
#     def __init__(self, transformer_chans = 128, cnn_chans = 64 , out_chan = 64):
#         super(X_spatial, self).__init__()
#         self.cnn_conv = nn.Conv2d(in_channels = cnn_chans, out_channels = transformer_chans, kernel_size=(5, 5), stride = 1, padding = 2)
#         self.trans_conv = nn.Conv2d(in_channels = transformer_chans, out_channels = cnn_chans, kernel_size=(3, 3), stride = 1, padding = 1)
#         self.cnn_series = nn.Sequential(
#             nn.Conv2d(cnn_chans, cnn_chans, kernel_size=(5, 5), stride= 1, padding=2),
#             nn.BatchNorm2d(cnn_chans),
#             nn.ReLU()
#         )
#         self.transformer_series = nn.Sequential(
#             nn.Conv2d(transformer_chans, transformer_chans, kernel_size=(3, 3), stride= 1, padding=1),
#             nn.BatchNorm2d(transformer_chans),
#             nn.ReLU()
#         )
#         self.out = nn.Conv2d(in_channels=(cnn_chans + transformer_chans)*2, out_channels=out_chan, kernel_size=(3, 3), stride=1, padding = 1)
#     def forward(self, trans, cnn):
#         cnn_branch_fuse = self.cnn_series(cnn + self.trans_conv(trans))
#         trans_branch_fuse = self.transformer_series(trans + self.cnn_conv(cnn))
#         combine = torch.cat((cnn_branch_fuse, trans_branch_fuse, trans, cnn), dim = 1)
#         out = self.out(combine)
#         return out

class Channel_Exchange_Attention(nn.Module):
    def __init__(self, high_level_channel = 512, low_level_channel = 256):
        super(Channel_Exchange_Attention, self).__init__()
        self.channel_self_attn1 = se_block_without_shorcut(high_level_channel, low_level_channel) # compress -> excitation
        self.channel_self_attn2 = se_block_without_shorcut(low_level_channel, high_level_channel) # excitation -> compress
    def forward(self, x1, x2):
        B, C1, H1, W1 = x1.shape
        _, C2, _, _ = x2.shape

        x1_attn = self.channel_self_attn1(x1)
        x1_attn = x1_attn.transpose(1, 2)

        x2_attn = self.channel_self_attn2(x2)

        
        attn = x2_attn@x1_attn

        attn1 = F.softmax(attn, dim = 2)
        output1 = torch.einsum('bxy,byhw->bxhw', attn1, x1)

        attn2 = F.softmax(attn, dim = 1)
        output2 = torch.einsum('bxy,bxhw->byhw', attn2, x2)
        return output1, output2

# if __name__  == "__main__":
#     x1 = torch.randn(1, 2, 2, 2)
#     x2 = torch.randn(1, 4, 2, 2)
#     cea = Channel_Exchange_Attention(4, 2)
#     out1, out2 = cea(x2, x1)
#     print(out1.shape, out2.shape)