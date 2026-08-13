"""
SimpleModel — multi-modal carpet segmentation network.

Architecture:
    Input (9ch: RGB 3ch + Depth 1ch + Normal 3ch + Edge 1ch + Curvature 1ch)
        │
    SimpleEncoder (ResNet50, first conv expanded 3→9ch)
        │
    [f1, f2, f3, f4]  at strides [4, 8, 16, 32]
        │
    SimpleDecoder (U-Net skip connections + RefinementHead)
        │
    Output (2ch logits at original resolution → argmax → 0/1 mask)

Inference: use `predict()` for direct 0/1 mask output.
"""

import torch
import torch.nn as nn
from typing import Dict
from .simple_encoder import SimpleEncoder
from .simple_decoder import SimpleDecoder

class SimpleModel(nn.Module):
    """
    Multi-modal segmentation model.

    Args:
        backbone:       'resnet34' | 'resnet50'
        pretrained:     use ImageNet pretrained weights
        num_classes:    output classes (default 2: background, carpet)
        use_refinement: stride=2 refinement head (doubles prediction resolution)
        in_channels:    input channel count (9 for D435+geo features)
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        num_classes: int = 2,
        use_refinement: bool = True,
        in_channels: int = 9,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.encoder = SimpleEncoder(
            backbone=backbone,
            pretrained=pretrained,
            in_channels=in_channels,
        )
        self.decoder = SimpleDecoder(
            encoder_channels=self.encoder.channels,
            num_classes=num_classes,
            use_refinement=use_refinement,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9, H, W) — RGB(3) + Depth(1) + Normal(3) + Edge(1) + Curvature(1)
        Returns:
            logits: (B, num_classes, H, W)
        """
        feats = self.encoder(x)
        return self.decoder(feats)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: return 0/1 mask (B, H, W)."""
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: return softmax probability (B, 2, H, W)."""
        logits = self.forward(x)
        return logits.float().softmax(dim=1)


def build_model(backbone: str = 'resnet50', in_channels: int = 9, **kwargs) -> SimpleModel:
    """Factory function for creating model instances."""
    return SimpleModel(backbone=backbone, in_channels=in_channels, **kwargs)
