"""
Simple U-Net Decoder — top-down decoder with skip connections + RefinementHead.

Design:
    f4 → [1×1 reduce] → Up+Skip(f3) → ConvBlock →
    Up+Skip(f2) → ConvBlock → Up+Skip(f1) → ConvBlock →
    [RefinementHead (stride=4→2→1) | Legacy (4× upsample)] → output

RefinementHead: two-stage upsampling that doubles effective prediction
resolution from stride=4 (120×160) to stride=2 (240×320) for sharper boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class ConvBlock(nn.Module):
    """Double 3×3 convolution: Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RefinementHead(nn.Module):
    """
    Two-stage upsampling head: stride=4 → stride=2 → stride=1.

    Heavy convs at stride=4; light conv at stride=2 — keeps overhead low.

    Args:
        in_dim:      input feature channels (d0 from decoder, 64 for ResNet50)
        num_classes: output classes
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        refine_dim = max(in_dim // 2, 16)

        # Stage 1: stride=4 heavy convs
        self.heavy_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(in_dim, refine_dim, 1)

        # Stage 2: stride=2 light refine
        self.refine_conv = nn.Sequential(
            nn.Conv2d(refine_dim, refine_dim, 3, padding=1),
            nn.BatchNorm2d(refine_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 3: project to classes
        self.head = nn.Conv2d(refine_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H/4, W/4) decoder features at stride=4
        Returns:
            out: (B, num_classes, H, W) logits at full resolution
        """
        x = self.heavy_conv(x)
        x = self.proj(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=False)
        x = self.refine_conv(x)

        x = self.head(x)
        out = F.interpolate(x, scale_factor=2, mode='bilinear',
                            align_corners=False)
        return out


class SimpleDecoder(nn.Module):
    """
    U-Net style decoder with lateral skip connections + RefinementHead.

    For ResNet50 enc=[256, 512, 1024, 2048]:
        d3=512, d2=256, d1=128, d0=64

    For ResNet34 enc=[64, 128, 256, 512]:
        d3=256, d2=128, d1=64, d0=32

    Args:
        encoder_channels: 4-scale encoder output channels
        num_classes:      output classes
        use_refinement:   True=RefinementHead, False=legacy 4× upsample
    """

    def __init__(self, encoder_channels: List[int], num_classes: int = 2,
                 use_refinement: bool = True):
        super().__init__()
        e1, e2, e3, e4 = encoder_channels
        self.use_refinement = use_refinement

        # Auto-scale decoder channels
        d3 = min(e4 // 4, 512)
        d2 = min(d3, 256)
        d1 = min(d2, 128)
        d0 = min(d1, 64)

        # 1×1 lateral reductions
        self.lat4 = nn.Conv2d(e4, d3, 1)
        self.lat3 = nn.Conv2d(e3, d3 // 2, 1)

        # Decoder blocks
        self.dec3 = ConvBlock(d3 + d3 // 2, d2)

        self.lat2 = nn.Conv2d(e2, d2 // 2, 1)
        self.dec2 = ConvBlock(d2 + d2 // 2, d1)

        self.dec1 = ConvBlock(d1 + e1, d0)

        # Output head
        if use_refinement:
            self.refinement = RefinementHead(d0, num_classes)
            self.final = None
            self.head = None
        else:
            self.final = ConvBlock(d0, d0)
            self.head = nn.Conv2d(d0, num_classes, 1)
            self.refinement = None

        self._channels = [d3, d2, d1, d0]

    @property
    def decoder_channels(self) -> List[int]:
        return self._channels

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            feats: (f1, f2, f3, f4) at strides [4, 8, 16, 32]
        Returns:
            logits: (B, num_classes, H, W) at original resolution
        """
        f1, f2, f3, f4 = feats

        # Stage 3: f4 (stride=32) → f3 (stride=16)
        x = self.lat4(f4)
        skip = self.lat3(f3)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.dec3(x)

        # Stage 2: → f2 (stride=8)
        skip = self.lat2(f2)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.dec2(x)

        # Stage 1: → f1 (stride=4)
        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear',
                          align_corners=False)
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)

        # Output
        if self.use_refinement:
            return self.refinement(x)
        else:
            x = self.final(x)
            x = self.head(x)
            x = F.interpolate(x, scale_factor=4, mode='bilinear',
                              align_corners=False)
            return x
