"""
Modified from https://github.com/xternalz/WideResNet-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from dassl.modeling.ops import AFH
from dassl.modeling.ops import style_insert
from dassl.modeling.ops import MixStyle, OMA, DSU

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut) and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            ) or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(
        self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0
    ):
        super().__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(Backbone):

    def __init__(self, depth, widen_factor, dropRate=0.0):
        super().__init__()
        nChannels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        assert (depth-4) % 6 == 0
        n = (depth-4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate
        )
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(0.01, inplace=True)

        self._out_features = nChannels[3]

        ######################################################################################################
        self.cad3 = AFH(in_channels=64, out_channels=30, kernel_size=1, dk=30, dv=30, Nh=1,
                                 relative=False)
        self.style = style_insert
        self.OMA = OMA(p=0.5, alpha=0.1)
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)
        self.dsu = DSU(p=0.5, alpha=0.1)
        self.ms_layers = ["layer1", "layer2", "layer3"]
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        ######################################################################################################


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, domain, label, mixup_sample=None, layer_mix=None, style=None):
        out = self.conv1(x)
        out = self.block1(out)
        feat = []

        if "layer111" in self.ms_layers:
            x, label = self.OMA(x, domain, label, mixup_sample, layer_mix=1)


        out = self.block2(out)
        if "layer211" in self.ms_layers:
            x, label = self.OMA(x, domain, label, mixup_sample, layer_mix=2)


        out = self.block3(out)
        if "layer311" in self.ms_layers:
            x, label = self.OMA(x, domain, label, mixup_sample, layer_mix=3)

        out = self.relu(self.bn1(out))

        # x = self.cad3(x, domain, label, mixup_sample, layer=4, tmp_=False)

        out = F.adaptive_avg_pool2d(out, 1)
        # x = torch.sum(x.view(x.size(0), x.size(1), -1), dim=2)
        return out.view(out.size(0), -1), label, feat

    def forward_cb(self, x, domain, label, mixup_sample=None, layer_mix=None, style=None):
        out = self.conv1(x)
        out = self.block1(out)
        feat = []
        feat_attn = []

        out = self.block2(out)
        feat.append(x)

        out = self.block3(out)
        feat.append(x)

        out = self.relu(self.bn1(out))
        feat.append(x)

        # x = self.cad3(x, domain, label, mixup_sample, layer=4, tmp_=False)

        # feat_attn.append(flat_feat)
        return out.view(out.size(0), -1), feat, feat_attn

@BACKBONE_REGISTRY.register()
def wide_resnet_28_2(**kwargs):
    return WideResNet(28, 2)


@BACKBONE_REGISTRY.register()
def wide_resnet_16_4(**kwargs):
    return WideResNet(16, 4)
