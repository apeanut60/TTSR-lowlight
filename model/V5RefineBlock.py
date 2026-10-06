"""Base-aware RefineBlock. Zero-init out so ΔD=0 at step0. No MHA/RDB/gate."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from model.V5TextureEncoder import ResidualBlock


class V5RefineBlock(nn.Module):
    """u=[D, P(A), D-P(A)] → ΔD; D* = D+ΔD."""

    def __init__(self, base_ch: int, ref_ch: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or base_ch
        self.ref_proj = nn.Conv2d(ref_ch, base_ch, 1)
        self.body = nn.Sequential(
            nn.Conv2d(base_ch * 3, hidden, 3, 1, 1),
            nn.GELU(),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        self.out = nn.Conv2d(hidden, base_ch, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, base_feat: Tensor, aligned_ref: Tensor) -> Tuple[Tensor, Tensor]:
        r = self.ref_proj(aligned_ref)
        u = torch.cat([base_feat, r, base_feat - r], dim=1)
        delta = self.out(self.body(u))
        return base_feat + delta, delta
