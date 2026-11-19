"""IlluminationEstimator for the CSFI-U-Net light-up base (block-amend).

Moved verbatim out of ``model/igab.py`` (layers, init and forward unchanged) so
the backbone no longer imports the historical IGAB module. ``model.igab`` keeps
re-exporting this class for old imports.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IlluminationEstimator(nn.Module):
    """Predict illumination features + log_gain from Low in [0, 1]."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, channels, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels, bias=True)
        self.conv2 = nn.Conv2d(channels, 3, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, low01: torch.Tensor):
        mean_c = low01.mean(dim=1, keepdim=True)
        f = self.depth_conv(self.conv1(torch.cat([low01, mean_c], dim=1)))
        log_gain = self.conv2(f)
        return f, log_gain
