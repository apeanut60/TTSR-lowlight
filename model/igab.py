"""Historical IGAB (ASRT / RetinexFormer-style illumination-guided channel attention).

Kept for reference and for parameter/behaviour comparisons only. The block-amend
CSFI-U-Net (model_variant ``csfi_lfca_v1``) no longer instantiates or calls
``IGAB`` / ``IG_MSA``; see ``model/frequency_mixer.py``. Does not import ASRT.

``IlluminationEstimator`` moved to ``model/illumination.py`` (unchanged) and is
re-exported here for old imports.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.illumination import IlluminationEstimator  # noqa: F401  (compat re-export)


class IG_MSA(nn.Module):
    """Channel-transposed illumination-guided attention (not HW x HW)."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 1):
        super().__init__()
        if dim != dim_head * heads:
            raise ValueError(
                'IG_MSA needs dim == heads*dim_head; got %d vs %d'
                % (dim, heads * dim_head))
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x, illu):
        b, h, w, c = x.shape
        if illu.shape != x.shape:
            raise ValueError('IG_MSA: x and illu must share BHWC shape')
        n = h * w

        def reshape(t):
            return t.reshape(b, n, self.num_heads, self.dim_head).permute(0, 2, 3, 1)

        q = reshape(self.to_q(x.reshape(b, n, c)))
        k = reshape(self.to_k(x.reshape(b, n, c)))
        v_inp = self.to_v(x.reshape(b, n, c))
        v = reshape(v_inp) * reshape(illu.reshape(b, n, c))
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = torch.softmax(
            torch.matmul(k, q.transpose(-2, -1)) * self.rescale, dim=-1)
        out = torch.matmul(attn, v).permute(0, 3, 1, 2).reshape(b, n, c)
        out_c = self.proj(out).reshape(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2))
        return out_c + out_p.permute(0, 2, 3, 1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, padding=1, groups=dim * mult, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class IGAB(nn.Module):
    """Illumination-guided attention block."""

    def __init__(self, dim: int, num_blocks: int = 1, heads: int = 1):
        super().__init__()
        if dim % heads:
            raise ValueError('channels must be divisible by heads')
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim // heads, heads=heads),
                nn.LayerNorm(dim),
                FeedForward(dim),
            ]) for _ in range(num_blocks)
        ])

    def forward(self, feat, illu):
        x = feat.permute(0, 2, 3, 1)
        illu = illu.permute(0, 2, 3, 1)
        for attn, norm, ff in self.blocks:
            x = x + attn(x, illu)
            x = x + ff(norm(x))
        return x.permute(0, 3, 1, 2)
