import torch.nn as nn
from torch.nn import functional as F
import torch
from dassl.utils import init_network_weights
import pdb
from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from dassl.modeling.ops import AFH
from dassl.modeling.ops import style_insert
from dassl.modeling.ops import MixStyle, OMA, DSU

class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvNet(Backbone):

    def __init__(self, c_hidden=64):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)



        self._out_features = 1 ** 1 * c_hidden

        ######################################################################################################
        self.cad3 = AFH(in_channels=64, out_channels=30, kernel_size=1, dk=30, dv=30, Nh=1,
                                 relative=False)  # 64 128 512
        self.style = style_insert
        self.OMA = OMA(p=0.5, alpha=0.1)
        self.ms_layers = ["layer1","layer2", "layer3"]
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        ######################################################################################################

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x, domain, label, mixup_sample=None, layer_mix=None, style=None):
        self._check_input(x)
        feat = []
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)

        x = self.style(x, style)

        if "layer1" in self.ms_layers:
            x, label = self.OMA(x, domain, label, mixup_sample, layer_mix=1)

        x = self.conv2(x)

        x = F.max_pool2d(x, 2)
        if "layer2" in self.ms_layers:
            x, label = self.OMA(x, domain, label, mixup_sample, layer_mix=2)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        if "layer3" in self.ms_layers:
            x, label = self.OMA(x, domain, label, mixup_sample, layer_mix=3)

        x = self.conv4(x)

        x = F.max_pool2d(x, 2)

        x = self.cad3(x, domain, label, mixup_sample, layer=4, tmp_=False)
        pdb.set_trace()
        # x = self.global_avgpool(x)
        x = torch.sum(x.view(x.size(0), x.size(1), -1), dim=2)
        return x.view(x.size(0), -1), label, feat

    def forward_cb(self, x, domain, label, mixup_sample=None, layer_mix=None, style=None):
        self._check_input(x)
        feat = []
        feat_attn = []
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        feat.append(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        feat.append(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        feat.append(x)

        x = self.conv4(x)

        x = F.max_pool2d(x, 2)

        x, flat_feat = self.cad3(x, domain, label, tmp_=True)
        feat_attn.append(flat_feat)

        return x.view(x.size(0), -1), feat, feat_attn

@BACKBONE_REGISTRY.register()
def cnn_digitsdg(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    model = ConvNet(c_hidden=64)
    init_network_weights(model, init_type="kaiming")
    return model
