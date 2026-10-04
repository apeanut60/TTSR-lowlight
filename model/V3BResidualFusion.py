"""V3-B.0 minimal implicit RGB residual. Clean-room; no explicit q/D/gate."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class V3B0ResidualFusion(nn.Module):
    """u=[F0,T,F0-T] at H/2 → ΔY at Y0 resolution. Last conv is zero-init."""

    def __init__(self, in_ch=96, hid=64, mid=32, rgb=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hid, mid, 3, 1, 1, bias=True),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.Conv2d(mid, rgb, 3, 1, 1, bias=True),
            nn.GELU(),
        )
        self.out = nn.Conv2d(rgb, 3, 3, 1, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, F0, T, out_hw):
        if F0.shape != T.shape or int(F0.shape[1]) != 32:
            raise SystemExit('expected F0/T [B,32,H/2,W/2], got %s %s'
                             % (tuple(F0.shape), tuple(T.shape)))
        u = torch.cat([F0, T, F0 - T], dim=1)
        h = self.stem(u)
        h = F.interpolate(h, size=tuple(out_hw), mode='bilinear',
                          align_corners=False)
        return self.out(self.up(h))


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))
