import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from dassl.modeling.ops import AFH
from dassl.modeling.ops import style_insert

import torch


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_a=0.1,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion
        self.num_blocks = sum(layers)

        if self.num_blocks == 8: # ResNet18
            self.cad3 = AFH(in_channels=512, out_channels=30, kernel_size=1, dk=30, dv=30, Nh=1, relative=False) # 64 128 512
        else: # ResNet50
            self.cad3 = AFH(in_channels=2048, out_channels=30, kernel_size=1, dk=30, dv=30, Nh=1, relative=False) # 64 128 512

        # New Style from other clients
        self.style = style_insert

        self.mixstyle = None

        if ms_layers:
            if isinstance(ms_class, tuple):
                self.OMA = ms_class[0](p=ms_p, alpha=ms_a)
                self.mixstyle = ms_class[1](p=ms_p, alpha=ms_a)

            else:
                self.mixstyle = ms_class(p=ms_p, alpha=ms_a)
                for layer_name in ms_layers:
                    assert layer_name in ["layer1", "layer2", "layer3"]
                print(
                    f"Insert {self.mixstyle.__class__.__name__} after {ms_layers}"
                )

        self.ms_layers = ms_layers
        self.vis = 0
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x, domain, label, supplemental_samples, style, StableFDG_param):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # Style sharing
        x = self.style(x, style, self.num_blocks)

        self.OMA.indicator = 0

        #Style exploration
        if "layer1" in self.ms_layers:
            x, label = self.OMA(x, domain, label, supplemental_samples, param=StableFDG_param, layer_mix=1)

        x = self.layer2(x)

        # Style exploration
        if "layer2" in self.ms_layers:
            x, label = self.OMA(x, domain, label, supplemental_samples, param=StableFDG_param, layer_mix=2)

        x = self.layer3(x)

        # Style exploration
        if "layer3" in self.ms_layers:
            x, label = self.OMA(x, domain, label, supplemental_samples, param=StableFDG_param, layer_mix=3)

        x = self.layer4(x)

        # 3-3 Attention module
        x = self.cad3(x, domain, label, supplemental_samples=supplemental_samples, layer=4, tmp_=False)

        return x, label

    def forward(self, x, domain, label, supplemental_samples=None, style=None, StableFDG_param=None):
        f, label = self.featuremaps(x, domain, label, supplemental_samples, style, StableFDG_param)

        # Normal
        # v = self.global_avgpool(f)

        # 3-3 Attention Module
        v = torch.sum(f.view(f.size(0), f.size(1), -1), dim=2)
        return v.view(v.size(0), -1), label


    def featuremaps_cb(self, x, domain, label):
        feat = []
        feat_attn = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        feat.append(x)
        x = self.layer2(x)

        feat.append(x)
        x = self.layer3(x)
        feat.append(x)

        x = self.layer4(x)

        # Feature extraction
        x, flat_feat = self.cad3(x, domain, label, tmp_=True)
        feat_attn.append(flat_feat)

        return x, feat, feat_attn

    def forward_cb(self, x, domain, label):
        f, feat, flat_feat = self.featuremaps_cb(x, domain, label)
        v = self.global_avgpool(f)

        return v.view(v.size(0), -1), feat, flat_feat


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)

    model.load_state_dict(pretrain_dict, strict=False)



"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


@BACKBONE_REGISTRY.register()
def resnet18_OMA_ms_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle, OMA

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=(OMA, MixStyle),
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model

@BACKBONE_REGISTRY.register()
def resnet50_OMA_ms_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle, OMA

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=(OMA, MixStyle),
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model



############################################################################################################
@BACKBONE_REGISTRY.register()
def resnet18(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet34(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet34"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet152(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet152"])

    return model


"""
Residual networks with mixstyle
"""


@BACKBONE_REGISTRY.register()
def resnet18_ms_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_ms_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_ms_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_ms_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_ms_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_ms_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_ms_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_ms_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_ms_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


"""
Residual networks with efdmix
"""


@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_efdmix_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_efdmix_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_efdmix_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=EFDMix,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model

#######################################################################################