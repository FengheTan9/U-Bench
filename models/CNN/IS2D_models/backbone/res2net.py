import math

import torch
import torch.nn as nn

class Bottle2Neck(nn.Module) :
    expansion = 4
    def __init__(self, inplanes, features, stride=(1, 1), downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2Neck, self).__init__()

        width = int(math.floor(features * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1 : self.nums = 1
        else : self.nums = scale - 1

        if stype == 'stage' : self.pool = nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1))

        convs, bns = [], []

        for i in range(self.nums) :
            convs.append(nn.Conv2d(width, width, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False))
            bns.append(nn.BatchNorm2d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, features * self.expansion, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(features * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module) :
    def __init__(self, block, layers, baseWidth, scale, num_classes=1000):
        super(Res2Net, self).__init__()

        self.baseWidth = baseWidth
        self.scale = scale
        self.init_features = 32
        self.inplanes = 64

        ######### Stem layer #########
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.init_features * 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.init_features * 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.init_features * 1, self.init_features * 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.init_features * 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.init_features * 1, self.init_features * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )
        self.bn1 = nn.BatchNorm2d(self.init_features * 2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ######### Stem layer #########

        self.layer1 = self._make_layer(block, self.init_features * 2, layers[0])
        self.layer2 = self._make_layer(block, self.init_features * 4, layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, self.init_features * 8, layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, self.init_features * 16, layers[3], stride=(2, 2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, features, blocks, stride=(1, 1)):
        downsample = None

        if stride != (1, 1) or self.inplanes != features * block.expansion :
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, features * block.expansion, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(features * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, features, stride, downsample=downsample, stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = features * block.expansion
        for i in range(1, blocks) :
            layers.append(block(self.inplanes, features, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        ######### Stem layer #########
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        ######### Stem layer #########

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_feature(self, x, out_block_stage):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if out_block_stage == 1: return [x], x1
        elif out_block_stage == 2: return [x1, x], x2
        elif out_block_stage == 3: return [x2, x1, x], x3
        elif out_block_stage == 4: return [x3, x2, x1, x], x4

    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x0 = self.maxpool(x)
    #
    #     x1 = self.layer1(x0)
    #     x2 = self.layer2(x1)
    #     x3 = self.layer3(x2)
    #     x4 = self.layer4(x3)
    #
    #     return [x, x1, x2, x3], x4
    #