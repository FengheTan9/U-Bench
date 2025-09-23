import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import load_cnn_backbone_model

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq


        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

class MFMSAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 groups=32):
        super(MFMSAttentionBlock, self).__init__()

        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution

        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1 + scale_idx, dilation=1 + scale_idx, groups=groups, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True)
            ))

        c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            if frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        MultiFrequencyChannelAttention(inter_channel, c2wh[in_channels], c2wh[in_channels], frequency_branches, frequency_selection)))
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Sigmoid()))
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)))

    def forward(self, x):
        feature_aggregation = 0
        for scale_idx in range(self.scale_branches):
            feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0) if int(x.shape[2] // 2 ** scale_idx) >= self.min_resolution else x
            feature = self.multi_scale_branches[scale_idx](feature)
            if self.frequency_branches > 0:
                feature = self.multi_frequency_branches[scale_idx](feature)
            spatial_attention_map = self.multi_frequency_branches_conv1[scale_idx](feature)
            feature = self.multi_frequency_branches_conv2[scale_idx](feature * (1 - spatial_attention_map) * self.alpha_list[scale_idx] + feature * spatial_attention_map * self.beta_list[scale_idx])
            feature_aggregation += F.interpolate(feature, size=None, scale_factor=2**scale_idx, mode='bilinear', align_corners=None) if (x.shape[2] != feature.shape[2]) or (x.shape[3] != feature.shape[3]) else feature
        feature_aggregation /= self.scale_branches
        feature_aggregation += x

        return feature_aggregation

class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_connection_channels,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8):
        super(UpsampleBlock, self).__init__()

        in_channels = in_channels + skip_connection_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.attention_layer = MFMSAttentionBlock(out_channels, scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, skip_connection=None):
        x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=None)

        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.attention_layer(x)
        x = self.conv2(x)

        return x

class CascadedSubDecoderBinary(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 scale_factor,
                 interpolation_mode='bilinear'):
        super(CascadedSubDecoderBinary, self).__init__()

        self.output_map_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.output_distance_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.output_boundary_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=interpolation_mode, align_corners=True)
        self.count = 0

    def forward(self, x):
        map = self.output_map_conv(x) # B, 1, H, W
        distance = self.output_distance_conv(x) * torch.sigmoid(map)
        boundary = self.output_boundary_conv(x) * torch.sigmoid(distance)

        boundary = self.upsample(boundary)
        distance = self.upsample(distance) + torch.sigmoid(boundary)
        map = self.upsample(map) + torch.sigmoid(distance)

        return map, distance, boundary

class MFMSNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 input_channel=1,
                 scale_branches=3,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 cnn_backbone='resnet50'):
        super(MFMSNet, self).__init__()


        self.num_classes = num_classes

        self.feature_encoding = load_cnn_backbone_model(backbone_name=cnn_backbone, pretrained=True)
        if cnn_backbone not in ['resnet50', 'res2net50_v1b_26w_4s', 'resnest50']:
            print('Wrong CNN Backbone model')
            sys.exit()

        if cnn_backbone in ['resnet50', 'res2net50_v1b_26w_4s', 'resnest50']:
            self.in_channels = 2048
            self.skip_channel_list = [1024, 512, 256, 64]
            self.decoder_channel_list = [256, 128, 64, 32]
        else:
            print("Wrong CNN Backbone...")
            sys.exit()

        self.feature_encoding.fc = nn.Identity()

        self.skip_channel_down_list = [64, 64, 64, 64]

        self.skip_connection1 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[0], self.skip_channel_down_list[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[0]), nn.ReLU(inplace=True))
        self.skip_connection2 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[1], self.skip_channel_down_list[1], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[1]), nn.ReLU(inplace=True))
        self.skip_connection3 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[2], self.skip_channel_down_list[2], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[2]), nn.ReLU(inplace=True))
        self.skip_connection4 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[3], self.skip_channel_down_list[3], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[3]), nn.ReLU(inplace=True))

        self.decoder_stage1 = UpsampleBlock(self.in_channels, self.decoder_channel_list[0], self.skip_channel_down_list[0], scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)
        self.decoder_stage2 = UpsampleBlock(self.decoder_channel_list[0], self.decoder_channel_list[1], self.skip_channel_down_list[1], scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)
        self.decoder_stage3 = UpsampleBlock(self.decoder_channel_list[1], self.decoder_channel_list[2], self.skip_channel_down_list[2], scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)
        self.decoder_stage4 = UpsampleBlock(self.decoder_channel_list[2], self.decoder_channel_list[3], self.skip_channel_down_list[3], scale_branches, frequency_branches, frequency_selection, block_repetition, min_channel, min_resolution)

        # Sub-Decoder
        self.sub_decoder_stage1 = CascadedSubDecoderBinary(self.decoder_channel_list[0], num_classes, scale_factor=16)
        self.sub_decoder_stage2 = CascadedSubDecoderBinary(self.decoder_channel_list[1], num_classes, scale_factor=8)
        self.sub_decoder_stage3 = CascadedSubDecoderBinary(self.decoder_channel_list[2], num_classes, scale_factor=4)
        self.sub_decoder_stage4 = CascadedSubDecoderBinary(self.decoder_channel_list[3], num_classes, scale_factor=2)

    def forward(self, x, mode='train'):
        if x.size()[1] == 1: x = x.repeat(1, 3, 1, 1)
        _, _, H, W = x.shape

        features, x = self.feature_encoding.forward_feature(x, out_block_stage=4)

        x1 = self.decoder_stage1(x, self.skip_connection1(features[0]))
        x2 = self.decoder_stage2(x1, self.skip_connection2(features[1]))
        x3 = self.decoder_stage3(x2, self.skip_connection3(features[2]))
        x4 = self.decoder_stage4(x3, self.skip_connection4(features[3]))
        if mode == 'train':
            map_output1, distance_output1, boundary_output1 = self.sub_decoder_stage1(x1)
            map_output2, distance_output2, boundary_output2 = self.sub_decoder_stage2(x2)
            map_output3, distance_output3, boundary_output3 = self.sub_decoder_stage3(x3)
            map_output4, distance_output4, boundary_output4 = self.sub_decoder_stage4(x4)


            # return [map_output1, distance_output1, boundary_output1], \
            #        [map_output2, distance_output2, boundary_output2], \
            #        [map_output3, distance_output3, boundary_output3], \
            #        [map_output4, distance_output4, boundary_output4]
            return map_output1,map_output2,map_output3,map_output4
        else:
            map, _, _ = self.sub_decoder_stage4(x4)

            return map

    def _calculate_criterion(self, y_pred, y_true):
        loss = self.structure_loss(y_pred, y_true)

        return loss

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()