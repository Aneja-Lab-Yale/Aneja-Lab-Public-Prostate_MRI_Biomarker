"""EPE-Net / SVI-Net BoxNet (lab GitHub architecture).

Copied from Aneja-Lab-Public-Prostate_MRI_Biomarker/models/EPE_Net_final.py
so torch.load of the saved full model can find models.EPE_Net_final.BoxNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm1d,
    BatchNorm3d,
    Dropout,
    Dropout3d,
    Flatten,
    LeakyReLU,
    Linear,
    MaxPool3d,
    Module,
    Sigmoid,
)


class BoxNet(Module):
    def __init__(
        self,
        n_dims=60,
        n_channels=1,
        n_classes=1,
        feats=None,
        filt=None,
        stride=None,
        p_filt=None,
        p_stride=None,
    ):
        super().__init__()
        _ = n_dims
        feats = feats or [16, 32, 64, 128, 256, 2304, 4208]
        filt = filt or [3]
        stride = stride or [1]
        p_filt = p_filt or [2]
        p_stride = p_stride or [2]
        self.mp1 = MaxPool3d((1, 2, 2), (1, 2, 2))
        self.mp2 = MaxPool3d(p_filt[0], p_stride[0])
        self.conv_layer1 = self._conv_layer_set(n_channels, feats[0], filt[0], stride[0])
        self.conv_layer2 = self._conv_layer_set(feats[0], feats[1], filt[0], stride[0])
        self.conv_layer3 = self._conv_layer_set(feats[1], feats[2], filt[0], stride[0])
        self.conv_layer4 = self._conv_layer_set(feats[2], feats[2], filt[0], stride[0])
        self.conv_layer5 = self._conv_layer_set(feats[2], feats[3], filt[0], stride[0])
        self.conv_layer6 = self._conv_layer_set(feats[3], feats[3], filt[0], stride[0])
        self.conv_layer7 = self._conv_layer_set(feats[3], feats[4], filt[0], stride[0])
        self.conv_layer8 = self._conv_layer_set(feats[4], feats[4], filt[0], stride[0])
        self.s_dp = Dropout3d(p=0)
        self.dp = Dropout(p=0.6)
        self.fl1 = Flatten()
        self.fc1 = self._full_connect_set(feats[5], feats[5])
        self.fc2 = self._full_connect_set(feats[5], feats[6])
        self.fc3 = Linear(feats[5], n_classes, bias=True)
        self.act = Sigmoid()

    def _conv_layer_set(self, in_channels, out_channels, k, s):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, k, s, 1, bias=False),
            BatchNorm3d(out_channels),
            LeakyReLU(negative_slope=3e-2, inplace=True),
        )

    def _full_connect_set(self, in_channels, out_channels):
        return nn.Sequential(
            Linear(in_channels, out_channels, bias=False),
            BatchNorm1d(out_channels),
            LeakyReLU(negative_slope=3e-2, inplace=True),
        )

    def forward(self, inp):
        out = inp.float()
        out = self.conv_layer1(out)
        out = self.mp1(out)
        out = self.conv_layer2(out)
        out = self.mp2(out)
        out = self.conv_layer3(out)
        out = self.mp2(out)
        out = self.conv_layer4(out)
        out = self.mp2(out)
        out = self.conv_layer5(out)
        out = self.conv_layer6(out)
        out = self.mp2(out)
        out = self.conv_layer7(out)
        out = self.conv_layer8(out)
        out = self.mp2(out)
        out = self.fl1(out)
        out = self.fc1(out)
        out = self.dp(out)
        out = self.fc3(out)
        return out


def sigmoid_acc(y_pred):
    y_pred_tag = Sigmoid()(y_pred)
    return torch.round(y_pred_tag)
