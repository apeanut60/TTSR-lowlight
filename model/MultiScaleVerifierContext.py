"""Minimal SSEN-inspired multi-scale verifier context for V3-A.5D2.

This is intentionally NOT a copy of SSEN and does not use deformable convolution.
It tests only one hypothesis:

    does a larger multi-scale receptive field make q*_G64 more learnable?

Input:
    common H/4 verifier feature [B,C,H/4,W/4]

Output:
    enriched H/4 feature of the same shape/channel count.

No proposal evidence is used here; D2 must remain a clean RF/context ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAct(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class MultiScaleContext(nn.Module):
    """H/4 -> H/8 -> H/16 context, then residual fusion back at H/4."""

    def __init__(self, channels=160, bottleneck=64):
        super().__init__()
        c = int(channels)
        m = int(bottleneck)

        self.in_proj = ConvAct(c, m, 1)

        self.to_h8 = ConvAct(m, m, 2)
        self.h8_refine = ConvAct(m, m, 1)

        self.to_h16 = ConvAct(m, m, 2)
        self.h16_refine = nn.Sequential(
            ConvAct(m, m, 1),
            ConvAct(m, m, 1),
        )

        self.fuse_h8 = ConvAct(m * 2, m, 1)
        self.fuse_h4 = nn.Sequential(
            nn.Conv2d(m * 2, c, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(c, c, 1, 1, 0, bias=True),
        )

        # Residual branch begins as zero so a D2-A1 model can be made step-0
        # identical to the current verifier if desired.
        with torch.no_grad():
            self.fuse_h4[-1].weight.zero_()
            self.fuse_h4[-1].bias.zero_()

    def forward(self, h4):
        x4 = self.in_proj(h4)

        x8 = self.h8_refine(self.to_h8(x4))
        x16 = self.h16_refine(self.to_h16(x8))

        u16 = F.interpolate(
            x16, size=x8.shape[-2:], mode="bilinear", align_corners=False
        )
        x8 = self.fuse_h8(torch.cat([x8, u16], dim=1))

        u8 = F.interpolate(
            x8, size=x4.shape[-2:], mode="bilinear", align_corners=False
        )
        residual = self.fuse_h4(torch.cat([x4, u8], dim=1))
        return h4 + residual
