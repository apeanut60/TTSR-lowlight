"""LFCA: Low-Frequency Relation Channel Attention (block-amend restoration block).

Replaces the two IGAB positions (coarse H/4, middle H/2) of the CSFI-U-Net.

Per block, with input feature ``F`` of shape ``[B, C, H, W]``::

    F_low  = P(F)                  # fixed 5x5 binomial low-pass (buffer, no grad)
    F_high = F - F_low
    Q = Wq(F_low), K = Wk(F_low)   # channel relations estimated on F_low
    V = Wv(F)                      # values keep the full band (never F_low)
    Qn, Kn = L2-normalise Q, K along the spatial axis N = H*W
    A = softmax(temperature * (Qn @ Kn^T), dim=-1)      # [B, heads, d, d]
    O_attn  = Wo(A @ V)
    O_local = DWConv3x3(GELU(DWConv3x3(F_high)))        # bias-free
    X = F + O_attn + O_local
    Y = X + FFN(LayerNorm_channels(X))

The attention is channel x channel (``d = C // heads``), not spatial. The block
takes the feature only: no illumination feature, log_gain, Ref, T_lv* or GT.

The low-pass is a soft band split, not an ideal frequency cut-off: ``F_low``
still carries attenuated mid/high content and ``F_high`` is its complement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOWPASS_KERNEL_TYPE = 'binomial'
LOWPASS_KERNEL_SIZE = 5
_BINOMIAL_1D = (1., 4., 6., 4., 1.)


class FixedBinomialLowPass(nn.Module):
    """Fixed depthwise 5x5 binomial blur: k = [1,4,6,4,1]/16, K = outer(k, k).

    Stride 1, output has the input's H/W. The kernel is a non-trainable buffer
    (not a Parameter) and is defined by code, so it is kept out of the
    state_dict. Borders use reflect padding; an axis shorter than 3 pixels
    cannot be reflect-padded by 2 and falls back to replicate on that axis.
    """

    def __init__(self):
        super().__init__()
        k = torch.tensor(_BINOMIAL_1D, dtype=torch.float32)
        k = k / k.sum()
        kernel = torch.outer(k, k)
        self.pad = LOWPASS_KERNEL_SIZE // 2
        self.register_buffer(
            'kernel', kernel.view(1, 1, LOWPASS_KERNEL_SIZE, LOWPASS_KERNEL_SIZE),
            persistent=False)

    def _pad(self, x):
        p = self.pad
        h, w = x.shape[-2:]
        # Reflect needs pad < size on that axis; otherwise replicate (explicit).
        x = F.pad(x, (p, p, 0, 0), mode='reflect' if w > p else 'replicate')
        x = F.pad(x, (0, 0, p, p), mode='reflect' if h > p else 'replicate')
        return x

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('FixedBinomialLowPass expects [B,C,H,W], got %s'
                             % (tuple(x.shape),))
        c = x.shape[1]
        kernel = self.kernel.to(dtype=x.dtype, device=x.device)
        kernel = kernel.expand(c, 1, -1, -1)
        return F.conv2d(self._pad(x), kernel, stride=1, padding=0, groups=c)


class LowFreqRelationChannelAttention(nn.Module):
    """Channel attention whose relation matrix is estimated on ``F_low``.

    Q/K come from the low-passed feature, V from the full-band feature. Head
    split matches the historical IGAB layout: channel index ``= head * d + j``.
    """

    def __init__(self, dim: int, heads: int = 1, eps: float = 1e-6):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError(
                'LFCA needs channels divisible by heads; got dim=%d heads=%d'
                % (dim, heads))
        if eps <= 0:
            raise ValueError('eps must be > 0')
        self.dim = dim
        self.num_heads = heads
        self.dim_head = dim // heads
        self.eps = float(eps)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        # One trainable scalar per head, broadcast over [B, heads, d, d].
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)

    @staticmethod
    def _tokens(x):
        # [B, C, H, W] -> [B, N, C]
        return x.flatten(2).transpose(1, 2)

    def _split_heads(self, t):
        # [B, N, C] -> [B, heads, d, N]
        b, n, _ = t.shape
        return t.reshape(b, n, self.num_heads, self.dim_head).permute(0, 2, 3, 1)

    def attention_matrix(self, feat_low):
        """A = softmax(temperature * (Qn @ Kn^T), dim=-1), shape [B, heads, d, d]."""
        tok_low = self._tokens(feat_low)
        q = self._split_heads(self.to_q(tok_low))
        k = self._split_heads(self.to_k(tok_low))
        qn = F.normalize(q, p=2, dim=-1, eps=self.eps)
        kn = F.normalize(k, p=2, dim=-1, eps=self.eps)
        logits = torch.matmul(qn, kn.transpose(-2, -1)) * self.temperature
        return torch.softmax(logits, dim=-1)

    def forward(self, feat, feat_low):
        if feat.shape != feat_low.shape:
            raise ValueError('LFCA: feat and feat_low must share shape')
        b, c, h, w = feat.shape
        attn = self.attention_matrix(feat_low)
        v = self._split_heads(self.to_v(self._tokens(feat)))    # full band
        out = torch.matmul(attn, v)                             # [B, heads, d, N]
        out = out.permute(0, 3, 1, 2).reshape(b, h * w, c)      # [B, N, C]
        out = self.proj(out)
        return out.transpose(1, 2).reshape(b, c, h, w)


class HighFreqLocalBranch(nn.Module):
    """O_local = DWConv3x3(GELU(DWConv3x3(F_high))), bias-free (0 in -> 0 out)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, feat_high):
        return self.net(feat_high)


class ConvFeedForward(nn.Module):
    """1x1 C->mult*C, GELU, DW3x3, GELU, 1x1 mult*C->C (all bias-free)."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class LFCABlock(nn.Module):
    """One LFCA block. Residual coefficients are fixed to 1 (no learnable gates)."""

    def __init__(self, dim: int, heads: int = 1, ffn_expansion: int = 4):
        super().__init__()
        self.lowpass = FixedBinomialLowPass()
        self.attn = LowFreqRelationChannelAttention(dim, heads=heads)
        self.local = HighFreqLocalBranch(dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = ConvFeedForward(dim, mult=ffn_expansion)

    def forward(self, feat):
        feat_low = self.lowpass(feat)
        feat_high = feat - feat_low
        x = feat + self.attn(feat, feat_low) + self.local(feat_high)
        normed = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.ffn(normed)


class LFCA(nn.Module):
    """Stack of LFCA blocks; each block recomputes P(F) from its own input."""

    def __init__(self, dim: int, num_blocks: int = 1, heads: int = 1,
                 ffn_expansion: int = 4):
        super().__init__()
        if num_blocks < 1:
            raise ValueError('LFCA needs num_blocks >= 1, got %d' % num_blocks)
        self.blocks = nn.ModuleList([
            LFCABlock(dim, heads=heads, ffn_expansion=ffn_expansion)
            for _ in range(num_blocks)
        ])

    def forward(self, feat):
        for block in self.blocks:
            feat = block(feat)
        return feat
