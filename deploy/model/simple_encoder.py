"""
Simple Encoder — ResNet backbone with expanded first conv.

Supports arbitrary input channels: first 3 copy ImageNet weights,
extra channels are Kaiming-initialized.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple


def _expand_first_conv(old_conv: nn.Conv2d, new_in_channels: int,
                       pretrained: bool) -> nn.Conv2d:
    """
    Expand 3-channel first conv to N channels.
    First 3 channels copy ImageNet weights; extra channels Kaiming-initialized.
    """
    out_ch = old_conv.out_channels
    k = old_conv.kernel_size[0]
    s = old_conv.stride[0]
    p = old_conv.padding[0]
    new_conv = nn.Conv2d(new_in_channels, out_ch, kernel_size=k,
                         stride=s, padding=p, bias=old_conv.bias is not None)

    if pretrained:
        with torch.no_grad():
            new_conv.weight[:, :3].copy_(old_conv.weight)
            for c in range(3, new_in_channels):
                nn.init.kaiming_normal_(
                    new_conv.weight[:, c:c + 1],
                    mode='fan_out', nonlinearity='relu',
                )

    return new_conv


class SimpleEncoder(nn.Module):
    """
    Multi-scale encoder. Outputs features at stride [4, 8, 16, 32].

    Args:
        backbone:   'resnet34' | 'resnet50'
        pretrained: load ImageNet weights
        in_channels: input channel count (9 for D435+geo features)

    | Backbone  | Channels                  | Params |
    |-----------|---------------------------|--------|
    | resnet34  | [64, 128, 256, 512]      | 21M    |
    | resnet50  | [256, 512, 1024, 2048]   | 26M    |
    """

    BACKBONE_CONFIG = {
        'resnet34': {'builder': models.resnet34, 'channels': [64, 128, 256, 512]},
        'resnet50': {'builder': models.resnet50, 'channels': [256, 512, 1024, 2048]},
    }

    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True,
                 in_channels: int = 9):
        super().__init__()

        if backbone not in self.BACKBONE_CONFIG:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Choose from: {list(self.BACKBONE_CONFIG.keys())}"
            )

        cfg = self.BACKBONE_CONFIG[backbone]
        self.backbone_name = backbone
        self.channels = cfg['channels']

        weights = 'DEFAULT' if pretrained else None
        resnet = cfg['builder'](weights=weights)

        # Expand first conv: 3 → N channels
        old_conv = resnet.conv1
        new_conv = _expand_first_conv(old_conv, in_channels, pretrained)

        self.stem = nn.Sequential(new_conv, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # stride=4
        self.layer2 = resnet.layer2   # stride=8
        self.layer3 = resnet.layer3   # stride=16
        self.layer4 = resnet.layer4   # stride=32

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (f1, f2, f3, f4) at strides [4, 8, 16, 32]
        """
        x = self.stem(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)      # stride=4
        f2 = self.layer2(f1)     # stride=8
        f3 = self.layer3(f2)     # stride=16
        f4 = self.layer4(f3)     # stride=32

        return f1, f2, f3, f4
