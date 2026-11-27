"""Multi-scale spatial gain estimator (csfi_lfca_mgain_topk_v2).

Predicts a per-pixel ``log_gain`` from Low in ``[0, 1]`` only. It does not
read Reference, GT, transferred texture ``T`` or the match map ``S``.

The map is a learned spatial context, not a physically supervised
illumination, and the coarse grid is not a hard frequency constraint.
The final 3-channel convolution is zero-initialized, so a fresh module
emits ``log_gain = 0``. Earlier layers use the default convolution init;
their gradient is zero on the first step while that readout stays zero.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.frequency_mixer import FixedBinomialLowPass

GAIN_CLAMP = 1.5


def choose_token_grid(height, width, cap=400):
    """Deterministic pool grid with ``ph * pw <= cap`` and ``ph <= height``.

    When the native grid already fits, it is returned unchanged (never
    upsampled). Otherwise the grid keeps the aspect ratio as well as a
    bounded search allows: smallest aspect error, then largest area, then
    smaller ``ph``. A 1-pixel axis stays 1 and the other axis absorbs the cap.
    """
    height = int(height)
    width = int(width)
    cap = int(cap)
    if height < 1 or width < 1:
        raise ValueError(
            'token grid axes must be positive, got (%d, %d)' % (height, width))
    if cap < 1:
        raise ValueError('token cap must be positive, got %d' % cap)
    if height * width <= cap:
        return height, width
    target = height / float(width)
    best_key = None
    best = (1, 1)
    max_ph = min(height, cap)
    max_pw = min(width, cap)
    for ph in range(1, max_ph + 1):
        room = cap // ph
        if room < 1:
            continue
        pw_max = min(max_pw, room)
        ideal = ph / target
        cands = {pw_max}
        for near in (int(math.floor(ideal)), int(round(ideal))):
            near = max(1, min(max_pw, near))
            if ph * near <= cap:
                cands.add(near)
        for pw in cands:
            err = abs((ph / float(pw)) - target)
            key = (err, -(ph * pw), ph, pw)
            if best_key is None or key < best_key:
                best_key = key
                best = (ph, pw)
    return best


def summarize_log_gain(log_gain, clamp=GAIN_CLAMP):
    """Small detached stats. Does not keep the map."""
    g = log_gain.detach()
    flat = g.float().reshape(-1)
    qs = torch.quantile(
        flat, torch.tensor([0.01, 0.5, 0.99], device=flat.device))
    sat = ((g <= -float(clamp)) | (g >= float(clamp))).float().mean()
    return {
        'log_gain_q01': float(qs[0]),
        'log_gain_q50': float(qs[1]),
        'log_gain_q99': float(qs[2]),
        'log_gain_clamp_frac': float(sat),
        'log_gain_shape': tuple(int(x) for x in g.shape),
        'log_gain_clamp': float(clamp),
    }


class BlurDown2(nn.Module):
    """Fixed binomial low-pass, then take even pixels. Not trainable.

    Reuses ``FixedBinomialLowPass`` (the LFCA kernel). That module's numerics
    are not modified here. Output size is whatever ``[..., ::2, ::2]`` yields.
    """

    def __init__(self):
        super().__init__()
        self.lowpass = FixedBinomialLowPass()

    def forward(self, x):
        return self.lowpass(x)[..., ::2, ::2]


class ConvResBlock(nn.Module):
    """``Z + Conv3x3(GELU(Conv3x3(Z)))``. No norm and no extra gate."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class SpatialContextBlock(nn.Module):
    """One pre-LN spatial Transformer block. Attention is ``[B, heads, N, N]``.

    ``N = H*W`` of the (already token-capped) map. This is not LFCA's
    channel attention. Dropout is zero. Depth is one block.
    """

    def __init__(self, channels=64, heads=4, ffn_expansion=2):
        super().__init__()
        channels = int(channels)
        heads = int(heads)
        if heads < 1 or channels % heads != 0:
            raise ValueError(
                'channels=%d must be divisible by heads=%d' % (channels, heads))
        if int(ffn_expansion) < 1:
            raise ValueError('ffn_expansion must be >= 1')
        self.channels = channels
        self.heads = heads
        self.dim_head = channels // heads
        self.pos = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels,
            bias=True)
        self.norm1 = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        hidden = channels * int(ffn_expansion)
        self.norm2 = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.act = nn.GELU()
        self.last_attn_shape = None

    def forward(self, x):
        # x: [B, C, H, W]
        z = x + self.pos(x)
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)  # [B, N, C]
        n = tokens.shape[1]
        hdn = self.norm1(tokens)
        def _heads(t):
            return t.view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)
        q = _heads(self.to_q(hdn))
        k = _heads(self.to_k(hdn))
        v = _heads(self.to_v(hdn))
        scale = self.dim_head ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, heads, N, N]
        self.last_attn_shape = tuple(int(s) for s in scores.shape)
        attn = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attn, v)  # [B, heads, N, dim_head]
        mixed = mixed.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        u = tokens + self.proj(mixed)
        ff = self.fc2(self.act(self.fc1(self.norm2(u))))
        out = u + ff
        return out.transpose(1, 2).contiguous().view(b, c, h, w)


class MultiScaleSpatialGainEstimator(nn.Module):
    """``low01 [B,3,H,W] -> (Zf_quarter [B,C4,h4,w4], log_gain [B,3,H,W])``.

    ``Zf_quarter`` is the fused 1/4-scale feature. Callers use ``log_gain``
    only; a full-resolution illumination embedding is not built.

    Defaults are the fixed v2 starting point, not a tuned optimum:
    channels 32/64, res-blocks 2/2/1, 4 heads, depth 1, FFN x2, 400 tokens.
    """

    def __init__(
            self,
            channels=(32, 64),
            res_blocks=(2, 2, 1),
            heads=4,
            attn_depth=1,
            ffn_expansion=2,
            max_tokens=400):
        super().__init__()
        c4, c8 = int(channels[0]), int(channels[1])
        n4, n8, nf = (int(x) for x in res_blocks)
        if len(channels) != 2 or len(res_blocks) != 3:
            raise ValueError('channels must be (c4, c8) and res_blocks (n4, n8, nf)')
        if min(c4, c8, n4, n8, nf) < 1:
            raise ValueError('gain widths and residual counts must be >= 1')
        if int(attn_depth) != 1:
            raise ValueError(
                'gain spatial attention depth is fixed at 1, got %r' % attn_depth)
        self.channels = (c4, c8)
        self.res_blocks = (n4, n8, nf)
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        self.down = BlurDown2()
        self.in_conv = nn.Conv2d(4, c4, 3, padding=1, bias=True)
        self.act = nn.GELU()
        self.blocks4 = nn.ModuleList(ConvResBlock(c4) for _ in range(n4))
        self.to8 = nn.Conv2d(c4, c8, 3, padding=1, bias=True)
        self.blocks8 = nn.ModuleList(ConvResBlock(c8) for _ in range(n8))
        self.context = SpatialContextBlock(
            c8, heads=int(heads), ffn_expansion=int(ffn_expansion))
        self.fuse = nn.Conv2d(c4 + c8, c4, kernel_size=1, bias=True)
        self.blocks_f = nn.ModuleList(ConvResBlock(c4) for _ in range(nf))
        self.readout = nn.Conv2d(c4, 3, 3, padding=1, bias=True)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.last_shapes = None
        self.last_pool = None

    def encode_quarter(self, low01):
        mean_c = low01.mean(dim=1, keepdim=True)
        i4 = torch.cat([low01, mean_c], dim=1)
        i_q = self.down(self.down(i4))
        z4 = self.act(self.in_conv(i_q))
        for block in self.blocks4:
            z4 = block(z4)
        return z4

    def context_delta(self, z8):
        """Return ``Z8_context``. No pool => exactly ``A(Z8)``."""
        h, w = int(z8.shape[-2]), int(z8.shape[-1])
        ph, pw = choose_token_grid(h, w, self.max_tokens)
        if ph > h or pw > w or ph * pw > self.max_tokens:
            raise RuntimeError(
                'token grid (%d,%d) exceeds source (%d,%d) or cap %d'
                % (ph, pw, h, w, self.max_tokens))
        self.last_pool = (ph, pw, h, w)
        if ph == h and pw == w:
            return self.context(z8)
        pooled = F.adaptive_avg_pool2d(z8, (ph, pw))
        delta = self.context(pooled) - pooled
        delta = F.interpolate(
            delta, size=(h, w), mode='bilinear', align_corners=False)
        return z8 + delta

    def forward(self, low01):
        if low01.dim() != 4 or int(low01.shape[1]) != 3:
            raise ValueError(
                'spatial gain expects low01 [B,3,H,W], got %s' % (
                    tuple(low01.shape),))
        z4 = self.encode_quarter(low01)
        z8 = self.act(self.to8(self.down(z4)))
        for block in self.blocks8:
            z8 = block(z8)
        z8_context = self.context_delta(z8)
        z8_up = F.interpolate(
            z8_context, size=z4.shape[-2:], mode='bilinear',
            align_corners=False)
        zf = self.act(self.fuse(torch.cat([z4, z8_up], dim=1)))
        for block in self.blocks_f:
            zf = block(zf)
        g4 = self.readout(zf)
        log_gain = F.interpolate(
            g4, size=low01.shape[-2:], mode='bilinear', align_corners=False)
        self.last_shapes = {
            'low01': tuple(int(x) for x in low01.shape),
            'Z4': tuple(int(x) for x in z4.shape),
            'Z8': tuple(int(x) for x in z8.shape),
            'pool_hw': self.last_pool[:2] if self.last_pool else None,
            'Z8_context': tuple(int(x) for x in z8_context.shape),
            'Zf_quarter': tuple(int(x) for x in zf.shape),
            'g4': tuple(int(x) for x in g4.shape),
            'log_gain': tuple(int(x) for x in log_gain.shape),
        }
        # Zf is the fused quarter-resolution feature. Callers use log_gain.
        return zf, log_gain
