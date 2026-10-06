"""V4.1a RefGrounder: reverse-match T_low grounds FR, then frozen B0 reuses FR*."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.V4RefCanvas import encode_pair, match_qv

GROUND_DIR = 'FR_query_F0_source'
FORWARD_DIR = 'F0_query_FRstar_source'


class RefGrounder(nn.Module):
    """u=[FR,T_low,FR-T_low] 96ch → ΔFR 32ch. Last conv zero-init."""

    def __init__(self, in_ch=96, hid=64, out_ch=32):
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

    def forward(self, FR, T_low):
        if FR.shape != T_low.shape or int(FR.shape[1]) != 32:
            raise SystemExit('expected FR/T_low [B,32,H/2,W/2], got %s %s'
                             % (tuple(FR.shape), tuple(T_low.shape)))
        u = torch.cat([FR, T_low, FR - T_low], dim=1)
        if int(u.shape[1]) != self.in_ch:
            raise SystemExit('concat ch %d != %d' % (int(u.shape[1]), self.in_ch))
        return self.net(u)


@torch.no_grad()
def frozen_pair_and_tlow(wrapper, y0, reference, y0_donor=None, tlow_mode='normal'):
    """Frozen encode + reverse T_low. Safe to cache. Returns F0, FR, T_low, T_raw."""
    core, f0, fr = encode_pair(wrapper, y0, reference)
    t_raw = match_qv(core.match, f0, fr)
    mode = str(tlow_mode)
    if mode == 'normal':
        t_low = match_qv(core.match, fr, f0)
    elif mode == 'self':
        t_low = match_qv(core.match, fr, fr)
    elif mode == 'zero':
        t_low = torch.zeros_like(fr)
    elif mode in ('shuffled', 'shuffled_target'):
        if y0_donor is None:
            raise SystemExit('shuffled T_low needs y0_donor')
        from model.V4RefCanvas import encode_m11
        f0d = encode_m11(core.encoder, y0_donor)
        if f0d.shape != fr.shape:
            raise SystemExit('F0_donor %s != FR %s' % (tuple(f0d.shape), tuple(fr.shape)))
        t_low = match_qv(core.match, fr, f0d)
    else:
        raise SystemExit('unknown tlow_mode %r' % mode)
    return f0, fr, t_low, t_raw


def grounded_forward(match, b0_head, f0, fr, t_low, y0, grounder):
    """Differentiable: FR* → Match(F0,FR*) → frozen B0 head → Y.

    Matcher/B0/encoder params must have requires_grad=False, but this forward
    must NOT be under torch.no_grad() so grads reach the grounder.
    """
    delta = grounder(fr, t_low)
    fr_star = fr + delta
    t_star = match_qv(match, f0, fr_star)
    y = y0 + b0_head(f0, t_star, y0.shape[-2:], E=None)
    return y, delta, fr_star, t_star
