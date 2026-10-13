"""ReBaIR-inspired fusion drop-in for V5RefineBlock.

Modes:
  - spatial_window=1: legacy N=1 co-located attn (Q/K grads structurally 0).
  - spatial_window>=3: each query attends to a KxK neighborhood of Ref
    tokens (N=K*K >= 9) so Softmax is non-trivial and Q/K get gradients.

Zero-init residual out keeps step0 identity when out_conv is zero.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.V5TextureEncoder import ResidualBlock


class V7RiverFusion(nn.Module):
    """(D, A) → (D+ΔD, ΔD)."""

    def __init__(self, base_ch: int, ref_ch: int,
                 n_heads: int | None = None, head_dim: int = 16,
                 spatial_window: int = 1):
        super().__init__()
        if n_heads is None:
            if base_ch % head_dim != 0:
                raise ValueError('base_ch %d not divisible by head_dim %d'
                                 % (base_ch, head_dim))
            n_heads = base_ch // head_dim
        if spatial_window < 1 or spatial_window % 2 == 0:
            raise ValueError('spatial_window must be odd positive, got %s'
                             % spatial_window)
        emb = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.emb = emb
        self.base_ch = base_ch
        self.spatial_window = int(spatial_window)
        self.n_keys = self.spatial_window ** 2

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

    def _window_kv(self, feat: Tensor) -> Tensor:
        """feat [B,C,H,W] → [B, H*W, K*K, C] neighborhood tokens.

        F.unfold returns [B, C*K*K, HW] with layout (C, kh, kw) along dim=1.
        Must reshape to [B,C,K*K,HW] before permuting to [B,HW,K*K,C].
        The buggy path transpose(1,2).reshape(B,HW,K*K,C) mixes C with K*K.
        """
        k = self.spatial_window
        pad = k // 2
        # unfold: [B, C*K*K, H*W]
        patches = F.unfold(feat, kernel_size=k, padding=pad)
        b, _, hw = patches.shape
        patches = (patches.reshape(b, self.emb, self.n_keys, hw)
                   .permute(0, 3, 2, 1).contiguous())
        return patches

    def forward(self, base_feat: Tensor, aligned_ref: Tensor,
                return_attn: bool = False):
        if base_feat.shape[-2:] != aligned_ref.shape[-2:]:
            raise ValueError('spatial mismatch D %s vs A %s'
                             % (tuple(base_feat.shape), tuple(aligned_ref.shape)))
        b, _, h, w = base_feat.shape
        nh, hd = self.n_heads, self.head_dim
        x = self.in_base(base_feat)
        y_ref = self.in_ref(aligned_ref)

        q = self.q_proj(x)
        # [B, Nh, HW, Hd]
        q = q.reshape(b, nh, hd, h * w).permute(0, 1, 3, 2)

        if self.spatial_window == 1:
            # Legacy N=1 (kept for checkpoint compat / ablation)
            k = self.k_proj(y_ref).reshape(b, nh, hd, h * w).permute(0, 1, 3, 2)
            v = self.v_proj(y_ref).reshape(b, nh, hd, h * w).permute(0, 1, 3, 2)
            # treat as N=1 along a new dim → same as before
            k = k.unsqueeze(3)  # [B,Nh,HW,1,Hd]
            v = v.unsqueeze(3)
            q_ = q.unsqueeze(3)
            scale = hd ** -0.5
            scores = (q_ * k).sum(-1) * scale  # [B,Nh,HW,1]
            weights = scores.softmax(-1)
            attn = torch.matmul(weights.unsqueeze(-2), v).squeeze(3)
        else:
            k_tok = self._window_kv(self.k_proj(y_ref))  # B,HW,N,C
            v_tok = self._window_kv(self.v_proj(y_ref))
            k = k_tok.reshape(b, h * w, self.n_keys, nh, hd).permute(0, 3, 1, 2, 4)
            v = v_tok.reshape(b, h * w, self.n_keys, nh, hd).permute(0, 3, 1, 2, 4)
            # q: [B,Nh,HW,Hd] → [B,Nh,HW,1,Hd]
            q_ = q.unsqueeze(3)
            scale = hd ** -0.5
            scores = (q_ * k).sum(-1) * scale  # [B,Nh,HW,N]
            weights = scores.softmax(-1)
            attn = torch.matmul(weights.unsqueeze(-2), v).squeeze(3)
            # attn: [B,Nh,HW,Hd]

        y = attn.permute(0, 1, 3, 2).reshape(b, self.emb, h, w)
        y = self.fusion(torch.cat([x, y], dim=1))
        y = self.out_conv(y)
        delta = self.out_to_base(y)
        out = (base_feat + delta, delta)
        if not return_attn:
            return out
        # scores/weights: [B,Nh,HW,N]
        return out[0], out[1], dict(scores=scores, weights=weights)


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
