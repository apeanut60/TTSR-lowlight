"""V3-B.3 RGB global-statistics prior + A0→A1 shared init.

6 dims: mean_R/G/B, std_R/G/B on R01=(R+1)/2, std unbiased=False.
MLP 6→32 GELU →32→64, broadcast to H/2. No AdaIN/FiLM/evidence.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from model.V3BResidualFusion import V3B0ResidualFusion

GLOBAL_STAT_NAMES = (
    'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
)
EPS = 1e-6
CLIP = 5.0
MLP_HID = 32
G_CH = 64
A0_IN = 96
A1_IN = 160


def rgb01_mean_std(x_m11: torch.Tensor) -> torch.Tensor:
    """x in [-1,1] → [B,6] RGB mean/std on [0,1], std unbiased=False."""
    if x_m11.dim() != 4 or int(x_m11.shape[1]) != 3:
        raise SystemExit('expected RGB [B,3,H,W], got %s' % (tuple(x_m11.shape),))
    x = (x_m11 + 1.0) * 0.5
    mu = x.mean(dim=(2, 3))
    std = x.std(dim=(2, 3), unbiased=False)
    return torch.cat([mu, std], dim=1)


def normalize_global_stats(s_raw: torch.Tensor, stats: Dict,
                           names: Sequence[str] = GLOBAL_STAT_NAMES) -> torch.Tensor:
    if s_raw.shape[-1] != len(names):
        raise SystemExit('stats dim %d != names %d'
                         % (int(s_raw.shape[-1]), len(names)))
    outs = []
    for i, name in enumerate(names):
        st = stats[name]
        mu = float(st['mean'])
        std = float(st['std'])
        x = (s_raw[..., i] - mu) / (std + EPS)
        outs.append(x.clamp(-CLIP, CLIP))
    return torch.stack(outs, dim=-1)


class GlobalStatMLP(nn.Module):
    def __init__(self, in_dim=6, hid=MLP_HID, out_dim=G_CH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, out_dim),
        )

    def forward(self, s_norm):
        return self.net(s_norm)


def broadcast_g(g: torch.Tensor, hw) -> torch.Tensor:
    """[B,64] → [B,64,H,W]."""
    h, w = int(hw[0]), int(hw[1])
    return g[:, :, None, None].expand(-1, -1, h, w).contiguous()


class V3B3A1(nn.Module):
    """Residual head 160ch + global MLP. Extra 64 input channels start at 0."""

    def __init__(self):
        super().__init__()
        self.mlp = GlobalStatMLP()
        self.head = V3B0ResidualFusion(in_ch=A1_IN)

    def forward(self, F0, T, out_hw, s_norm):
        g = self.mlp(s_norm)
        G = broadcast_g(g, F0.shape[-2:])
        return self.head(F0, T, out_hw, E=G)


def copy_a0_to_a1_head(a0: V3B0ResidualFusion, a1_head: V3B0ResidualFusion) -> None:
    if a0.in_ch != A0_IN or a1_head.in_ch != A1_IN:
        raise SystemExit('expected A0 in_ch=%d A1 in_ch=%d, got %d/%d'
                         % (A0_IN, A1_IN, a0.in_ch, a1_head.in_ch))
    with torch.no_grad():
        w0 = a0.stem[0].weight
        w1 = a1_head.stem[0].weight
        w1.zero_()
        w1[:, :A0_IN].copy_(w0)
        a1_head.stem[0].bias.copy_(a0.stem[0].bias)
        sd0 = a0.state_dict()
        sd1 = a1_head.state_dict()
        for k, v in sd0.items():
            if k == 'stem.0.weight':
                continue
            if sd1[k].shape != v.shape:
                raise SystemExit('shape mismatch %s' % k)
            sd1[k].copy_(v)
        a1_head.load_state_dict(sd1, strict=True)


@torch.no_grad()
def assert_shared_init(a0: V3B0ResidualFusion, a1: V3B3A1,
                       f0, t, s_norm, out_hw, tol=1e-7) -> float:
    d0 = a0(f0, t, out_hw, E=None)
    d1 = a1(f0, t, out_hw, s_norm)
    diff = float((d0 - d1).abs().max())
    if diff > tol:
        raise SystemExit('A0/A1 step0 diverge: max_abs=%.3e' % diff)
    w = a1.head.stem[0].weight
    if float(w[:, A0_IN:].abs().max()) > 0:
        raise SystemExit('A1 global input slice not zero')
    return diff


def global_first_layer_norms(a1: V3B3A1) -> dict:
    w = a1.head.stem[0].weight[:, A0_IN:].detach().float()
    mlp_l1 = float(sum(p.detach().float().abs().sum() for p in a1.mlp.parameters()))
    mlp_l2 = float(sum(p.detach().float().pow(2).sum()
                       for p in a1.mlp.parameters()).sqrt())
    return dict(
        l1=float(w.abs().sum()),
        l2=float(w.pow(2).sum().sqrt()),
        mlp_l1=mlp_l1,
        mlp_l2=mlp_l2,
    )
