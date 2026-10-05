"""V3-B.4 H/2 feature residual adapter. Last conv zero-init. No gate/attention."""

import torch
import torch.nn as nn

from model.V3BFeatureBridge import BASE_H2_CH


class FeatureResidualAdapter(nn.Module):
    """u=[F0,T,F0-T] 96ch → ΔF_h2 80ch. Last layer zero-init ⇒ ΔF=0 at step0."""

    def __init__(self, in_ch=96, hid=64, out_ch=BASE_H2_CH):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_ch, hid, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hid, self.out_ch, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, F0, T):
        if F0.shape != T.shape or int(F0.shape[1]) != 32:
            raise SystemExit('expected F0/T [B,32,H/2,W/2], got %s %s'
                             % (tuple(F0.shape), tuple(T.shape)))
        u = torch.cat([F0, T, F0 - T], dim=1)
        return self.net(u)


def delta_fn_from_adapter(adapter, F0, T):
    """Build tiled delta_fn that crops full-image F0/T onto each tile."""
    def delta_fn(y0, x0, th, tw, f_h2):
        from model.V3BFeatureBridge import crop_ft
        f0c, tc = crop_ft(F0, T, y0, x0, th, tw)
        if f0c.shape[-2:] != f_h2.shape[-2:]:
            raise SystemExit('F0 crop %s != F_h2 %s (tile y0=%d x0=%d th=%d tw=%d)'
                             % (tuple(f0c.shape), tuple(f_h2.shape), y0, x0, th, tw))
        return adapter(f0c, tc)
    return delta_fn
