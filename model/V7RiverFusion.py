"""ReBaIR-inspired single-scale AttnFusion drop-in for V5RefineBlock.

Matches RIVER AttnFusion spatial layout: at each (h,w), Q=main attends over
N aligned refs at that site (here N=1). Zero-init residual so ΔD=0 at step0.

Not a full MultiScaleAttnFusion / MSFE port. Avoids global HW×HW attention.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.V5TextureEncoder import ResidualBlock


class V7RiverFusion(nn.Module):
    """(D, A) → (D+ΔD, ΔD). A treated as N=1 ref stack."""

    def __init__(self, base_ch: int, ref_ch: int,
                 n_heads: int | None = None, head_dim: int = 16):
        super().__init__()
        if n_heads is None:
            if base_ch % head_dim != 0:
                raise ValueError('base_ch %d not divisible by head_dim %d'
                                 % (base_ch, head_dim))
            n_heads = base_ch // head_dim
        emb = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.emb = emb
        self.base_ch = base_ch

        self.in_base = (nn.Identity() if base_ch == emb
                        else nn.Conv2d(base_ch, emb, 1))
        self.in_ref = (nn.Identity() if ref_ch == emb
                       else nn.Conv2d(ref_ch, emb, 1))
        self.q_proj = ResidualBlock(emb)
        self.k_proj = ResidualBlock(emb)
        self.v_proj = ResidualBlock(emb)
        self.fusion = nn.Sequential(
            nn.Conv2d(emb * 2, emb, 3, 1, 1),
            nn.GELU(),
            ResidualBlock(emb),
        )
        self.out_conv = nn.Conv2d(emb, emb, 3, 1, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        if base_ch == emb:
            self.out_to_base = nn.Identity()
        else:
            self.out_to_base = nn.Conv2d(emb, base_ch, 1)
            nn.init.zeros_(self.out_to_base.weight)
            nn.init.zeros_(self.out_to_base.bias)

    def forward(self, base_feat: Tensor, aligned_ref: Tensor
                ) -> Tuple[Tensor, Tensor]:
        if base_feat.shape[-2:] != aligned_ref.shape[-2:]:
            raise ValueError('spatial mismatch D %s vs A %s'
                             % (tuple(base_feat.shape), tuple(aligned_ref.shape)))
        b, _, h, w = base_feat.shape
        nh, hd = self.n_heads, self.head_dim
        x = self.in_base(base_feat)
        # ys: [B, N=1, C, H, W]
        ys = self.in_ref(aligned_ref).unsqueeze(1)
        n_ref = ys.shape[1]

        q = self.q_proj(x)
        # (B*H*W, Nh, 1, Hd)
        q = q.reshape(b, nh, hd, h * w).permute(0, 3, 1, 2).reshape(
            b * h * w, nh, 1, hd)
        k = self.k_proj(ys.reshape(b * n_ref, self.emb, h, w))
        k = k.reshape(b, n_ref, nh, hd, h * w).permute(0, 4, 2, 1, 3).reshape(
            b * h * w, nh, n_ref, hd)
        v = self.v_proj(ys.reshape(b * n_ref, self.emb, h, w))
        v = v.reshape(b, n_ref, nh, hd, h * w).permute(0, 4, 2, 1, 3).reshape(
            b * h * w, nh, n_ref, hd)

        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = hd ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            y = torch.matmul(scores.softmax(dim=-1), v)
        # (B*H*W, Nh, 1, Hd) → [B, C, H, W]
        y = y.reshape(b, h * w, nh, hd).permute(0, 2, 3, 1).reshape(
            b, self.emb, h, w)

        y = self.fusion(torch.cat([x, y], dim=1))
        y = self.out_conv(y)
        delta = self.out_to_base(y)
        return base_feat + delta, delta


def v7_fusion_step0_zero(module: V7RiverFusion) -> bool:
    if float(module.out_conv.weight.abs().max()) > 0:
        return False
    if float(module.out_conv.bias.abs().max()) > 0:
        return False
    if isinstance(module.out_to_base, nn.Conv2d):
        if float(module.out_to_base.weight.abs().max()) > 0:
            return False
        if float(module.out_to_base.bias.abs().max()) > 0:
            return False
    return True
