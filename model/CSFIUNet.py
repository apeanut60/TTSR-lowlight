"""CSFI-U-Net Stage-A backbone for TTSREnhance (block-amend).

Cross-scale restore with IlluminationEstimator light-up base / LFCA restoration
blocks / soft reference inject. Skeleton copied and simplified from ASRT
Stage-A; does not import the ASRT package. v3 / v4 / vNext modules are
intentionally omitted.

model_variant ``csfi_lfca_v1``: the two restoration positions (coarse H/4,
middle H/2) use LFCA (``model/frequency_mixer.py``) and take the feature only.
The IlluminationEstimator now feeds only the pixel-domain light-up base; its
feature map is no longer routed into the restoration blocks.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.frequency_mixer import (
    LFCA, LOWPASS_KERNEL_SIZE, LOWPASS_KERNEL_TYPE,
)
from model.illumination import IlluminationEstimator
from model.MainNetEnhance import GlobalIllumHead, RefIllumTransfer
from model.restoration_config import MODEL_VARIANT, parse_restoration_blocks


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)


def conv3x3(in_ch, out_ch, stride=1, dilation=1):
    return nn.Conv2d(
        in_ch, out_ch, 3, stride=stride, padding=dilation, dilation=dilation,
        bias=True)


def resize_like(x, target, mode='bilinear'):
    hw = target.shape[-2:]
    if x.shape[-2:] == hw:
        return x
    return F.interpolate(x, size=hw, mode=mode, align_corners=False)


class ResBlock(nn.Module):
    def __init__(self, channels, res_scale=1.):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)

    def forward(self, x):
        return x + self.res_scale * self.conv2(self.relu(self.conv1(x)))


class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super().__init__()
        self.conv_head = conv3x3(3, n_feats)
        self.RBs = nn.ModuleList([
            ResBlock(n_feats, res_scale=res_scale) for _ in range(num_res_blocks)])
        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x0 = x
        for rb in self.RBs:
            x = rb(x)
        return self.conv_tail(x) + x0


class CrossScaleCSFI2(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, stride=2)
        self.conv_merge1 = conv3x3(2 * n_feats, n_feats)
        self.conv_merge2 = conv3x3(2 * n_feats, n_feats)

    def forward(self, x1, x2):
        x12 = F.relu(self.conv12(resize_like(x1, x2)))
        x21 = resize_like(F.relu(self.conv21(x2)), x1)
        y1 = F.relu(self.conv_merge1(torch.cat([x1, x21], dim=1)))
        y2 = F.relu(self.conv_merge2(torch.cat([x2, x12], dim=1)))
        return y1, y2


class CrossScaleCSFI3(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        c = n_feats
        self.conv12 = conv1x1(c, c)
        self.conv13 = conv1x1(c, c)
        self.conv21 = conv3x3(c, c, stride=2)
        self.conv23 = conv1x1(c, c)
        self.conv31_1 = conv3x3(c, c, stride=2)
        self.conv31_2 = conv3x3(c, c, stride=2)
        self.conv32 = conv3x3(c, c, stride=2)
        self.conv_merge1 = conv3x3(3 * c, c)
        self.conv_merge2 = conv3x3(3 * c, c)
        self.conv_merge3 = conv3x3(3 * c, c)

    def forward(self, x1, x2, x3):
        x12 = F.relu(self.conv12(resize_like(x1, x2)))
        x13 = F.relu(self.conv13(resize_like(x1, x3)))
        x21 = resize_like(F.relu(self.conv21(x2)), x1)
        x23 = F.relu(self.conv23(resize_like(x2, x3)))
        x31 = resize_like(F.relu(self.conv31_2(F.relu(self.conv31_1(x3)))), x1)
        x32 = resize_like(F.relu(self.conv32(x3)), x2)
        y1 = F.relu(self.conv_merge1(torch.cat([x1, x21, x31], dim=1)))
        y2 = F.relu(self.conv_merge2(torch.cat([x2, x12, x32], dim=1)))
        y3 = F.relu(self.conv_merge3(torch.cat([x3, x13, x23], dim=1)))
        return y1, y2, y3


class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv_merge = conv3x3(3 * n_feats, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, 3)
        with torch.no_grad():
            self.conv_tail2.weight.mul_(0.01)
            self.conv_tail2.bias.zero_()

    def forward(self, x1, x2, x3):
        f = F.relu(self.conv_merge(torch.cat([
            resize_like(x1, x3), resize_like(x2, x3), x3], dim=1)))
        return self.conv_tail2(self.conv_tail1(f))


def parse_num_res_blocks(spec) -> list:
    if isinstance(spec, (list, tuple)):
        blocks = [int(x) for x in spec]
    else:
        blocks = [int(x) for x in str(spec).split('+')]
    if len(blocks) != 4:
        raise ValueError(
            'num_res_blocks expected SFE+coarse+middle+fine, got %r' % (spec,))
    return blocks


_ALLOWED_STAGE_MODES = frozenset({
    'normal', 'branch_bypass', 'texture_zero',
})


def _normalize_stage_mode(mode):
    if mode is None:
        return 'normal'
    m = str(mode).strip().lower()
    if m in ('', 'none', 'default'):
        return 'normal'
    aliases = {
        'bypass': 'branch_bypass',
        'zero': 'texture_zero',
        'zero_texture': 'texture_zero',
        'no_inject': 'branch_bypass',
    }
    m = aliases.get(m, m)
    if m not in _ALLOWED_STAGE_MODES:
        raise ValueError(
            'Unsupported stage_mode %r (allowed: %s)'
            % (mode, ', '.join(sorted(_ALLOWED_STAGE_MODES))))
    return m


class CSFIUNet(nn.Module):
    """Cross-scale CSFI U-Net with LFCA blocks. Drop-in for MainNetEnhance."""

    model_variant = MODEL_VARIANT

    def __init__(
        self,
        num_res_blocks='8+8+4+2',
        n_feats=64,
        res_scale=1.,
        restoration_block='lfca',
        restoration_blocks=(1, 1),
        restoration_heads=1,
        restoration_ffn_expansion=4,
        enable_lightup=True,
        use_global_illum=False,
        use_ref_illum=False,
        ref_illum_pool=8,
    ):
        super().__init__()
        blocks = parse_num_res_blocks(num_res_blocks)
        if str(restoration_block).strip().lower() != 'lfca':
            raise ValueError(
                'CSFIUNet (%s) only builds restoration_block="lfca", got %r'
                % (MODEL_VARIANT, restoration_block))
        depth_coarse, depth_middle = parse_restoration_blocks(restoration_blocks)
        restoration_heads = int(restoration_heads)
        restoration_ffn_expansion = int(restoration_ffn_expansion)
        if restoration_heads < 1 or n_feats % restoration_heads:
            raise ValueError(
                'n_feats=%d must be divisible by restoration_heads=%d'
                % (n_feats, restoration_heads))
        self.restoration_block = 'lfca'
        self.restoration_depths = (depth_coarse, depth_middle)
        self.restoration_heads = restoration_heads
        self.restoration_ffn_expansion = restoration_ffn_expansion
        self.num_res_blocks = blocks
        self.n_feats = n_feats
        self.enable_lightup = bool(enable_lightup)
        self.use_global_illum = bool(use_global_illum)
        self.use_ref_illum = bool(use_ref_illum)

        self.SFE = SFE(blocks[0], n_feats, res_scale)
        self.down1 = nn.Sequential(
            conv3x3(n_feats, n_feats, stride=2), nn.LeakyReLU(0.1))
        self.down2 = nn.Sequential(
            conv3x3(n_feats, n_feats, stride=2), nn.LeakyReLU(0.1))
        self.skip_e1 = nn.Sequential(
            conv1x1(n_feats * 2, n_feats), nn.LeakyReLU(0.1))
        self.skip_e0 = nn.Sequential(
            conv1x1(n_feats * 2, n_feats), nn.LeakyReLU(0.1))

        self.conv11_head = conv3x3(n_feats + 256, n_feats)
        self.RB11 = nn.ModuleList(
            [ResBlock(n_feats, res_scale) for _ in range(blocks[1])])
        self.conv11_tail = conv3x3(n_feats, n_feats)

        self.conv22_head = conv3x3(n_feats + 128, n_feats)
        self.ex12 = CrossScaleCSFI2(n_feats)
        self.RB21 = nn.ModuleList(
            [ResBlock(n_feats, res_scale) for _ in range(blocks[2])])
        self.RB22 = nn.ModuleList(
            [ResBlock(n_feats, res_scale) for _ in range(blocks[2])])
        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        self.conv33_head = conv3x3(n_feats + 64, n_feats)
        self.ex123 = CrossScaleCSFI3(n_feats)
        self.RB31 = nn.ModuleList(
            [ResBlock(n_feats, res_scale) for _ in range(blocks[3])])
        self.RB32 = nn.ModuleList(
            [ResBlock(n_feats, res_scale) for _ in range(blocks[3])])
        self.RB33 = nn.ModuleList(
            [ResBlock(n_feats, res_scale) for _ in range(blocks[3])])
        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        # Soft init for reference injectors: near-identity when gate*head≈0.
        for head in (self.conv11_head, self.conv22_head, self.conv33_head):
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.weight.mul_(0.01)

        self.illumination_estimator = IlluminationEstimator(n_feats)
        self.coarse_freq_block = LFCA(
            n_feats, num_blocks=depth_coarse, heads=restoration_heads,
            ffn_expansion=restoration_ffn_expansion)
        self.middle_freq_block = LFCA(
            n_feats, num_blocks=depth_middle, heads=restoration_heads,
            ffn_expansion=restoration_ffn_expansion)
        self.merge_tail = MergeTail(n_feats)
        self.global_illum_head = GlobalIllumHead()
        self.ref_illum = RefIllumTransfer(pool=ref_illum_pool)

        self.last_gate = None
        self.last_used_reference = False
        self.last_gate_stats = None
        self.last_residual_decomp = None

    @staticmethod
    def safe_pad(x, target_hw):
        ph = target_hw[0] - x.shape[-2]
        pw = target_hw[1] - x.shape[-1]
        if ph == 0 and pw == 0:
            return x
        mode = 'reflect' if ph < x.shape[-2] and pw < x.shape[-1] else 'replicate'
        return F.pad(x, (0, pw, 0, ph), mode=mode)

    @staticmethod
    def _align_ref(T, feat, name):
        if T is None:
            raise ValueError(name + ' required when using reference')
        dh = abs(T.shape[-2] - feat.shape[-2])
        dw = abs(T.shape[-1] - feat.shape[-1])
        if dh > 3 or dw > 3:
            raise ValueError(
                '%s spatial mismatch >3px: %s vs %s'
                % (name, T.shape[-2:], feat.shape[-2:]))
        return resize_like(T, feat)

    @staticmethod
    def _gate(S, feat):
        if S is None:
            return None
        return torch.sigmoid(F.interpolate(
            S, size=feat.shape[-2:], mode='bicubic', align_corners=False))

    def _run_rbs(self, feat, rbs, tail):
        start = feat
        for rb in rbs:
            feat = rb(feat)
        return start + tail(feat)

    def restoration_identity(self) -> dict:
        """What sits at the two restoration positions (for run metadata)."""
        return dict(
            model_variant=self.model_variant,
            restoration_block=self.restoration_block,
            coarse_depth=self.restoration_depths[0],
            middle_depth=self.restoration_depths[1],
            heads=self.restoration_heads,
            ffn_expansion=self.restoration_ffn_expansion,
            lowpass_kernel=LOWPASS_KERNEL_TYPE,
            lowpass_kernel_size=LOWPASS_KERNEL_SIZE,
        )

    def encode_low_stem(self, x):
        """Pad + light-up base + SFE + downs.

        The estimator's feature map is only an internal of the log_gain
        readout; it is not part of the stem (LFCA blocks take features only).
        """
        orig_hw = x.shape[-2:]
        th = int(math.ceil(orig_hw[0] / 4.0) * 4)
        tw = int(math.ceil(orig_hw[1] / 4.0) * 4)
        low = self.safe_pad(x, (th, tw))
        low01 = (low + 1.) * 0.5
        _, log_gain = self.illumination_estimator(low01)
        if self.enable_lightup:
            lightup = low01 * torch.exp(log_gain.clamp(-1.5, 1.5))
            base = 2. * lightup - 1.
        else:
            base = low
        e0 = self.SFE(low)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        return dict(
            low=low, low01=low01, base=base, orig_hw=orig_hw,
            e0=e0, e1=e1, e2=e2,
            log_gain=log_gain,
        )

    def _maybe_counterfactual_gate(
            self, S, feat, gate_mode=None, gate_shift=None,
            gate_smooth_kernel=31, collect_gate_stats=False):
        """Optional eval gate transforms (compatible with MainNetEnhance)."""
        if S is None:
            self.last_gate_stats = None
            return None
        g = self._gate(S, feat)
        mode = (gate_mode or 'none')
        if isinstance(mode, str):
            mode = mode.strip().lower()
        if mode in (None, '', 'none', 'default'):
            pass
        elif mode == 'ones':
            g = torch.ones_like(g)
        elif mode == 'zeros':
            g = torch.zeros_like(g)
        elif mode == 'shuffle':
            flat = g.flatten(2)
            idx = torch.randperm(flat.shape[-1], device=flat.device)
            g = flat[..., idx].view_as(g)
        elif mode == 'shift':
            if gate_shift is None:
                gate_shift = (0, 0)
            dy, dx = int(gate_shift[0]), int(gate_shift[1])
            g = torch.roll(g, shifts=(dy, dx), dims=(-2, -1))
        elif mode == 'smooth':
            k = int(gate_smooth_kernel)
            if k < 1:
                k = 1
            if k % 2 == 0:
                k += 1
            pad = k // 2
            g = F.avg_pool2d(
                F.pad(g, (pad, pad, pad, pad), mode='reflect'),
                kernel_size=k, stride=1)
        else:
            raise ValueError('Unknown gate_mode: %r' % gate_mode)

        if collect_gate_stats:
            self.last_gate_stats = {
                'mean': float(g.mean().detach()),
                'std': float(g.std().detach()),
                'min': float(g.min().detach()),
                'max': float(g.max().detach()),
            }
        else:
            self.last_gate_stats = None
        return g

    def _inject(
            self, feat, T, S, head, use_ref: bool, stage_mode='normal',
            gate=None):
        mode = _normalize_stage_mode(stage_mode)
        if mode == 'branch_bypass' or not use_ref or T is None or S is None:
            return feat, None
        T = self._align_ref(T, feat, 'T')
        if mode == 'texture_zero':
            T = torch.zeros_like(T)
        if gate is None:
            g = self._gate(S, feat)
        else:
            g = resize_like(gate, feat) if gate.shape[-2:] != feat.shape[-2:] else gate
            if g.shape[1] != 1 and g.shape[1] != feat.shape[1]:
                g = g[:, :1]
        residual = g * head(torch.cat([feat, T], dim=1))
        return feat + residual, residual

    def apply_illum_head(self, x, low):
        """Match MainNetEnhance photometric API (used post-stitch)."""
        if not self.use_global_illum:
            return x
        gain, bias, gamma = self.global_illum_head(low)
        x = torch.clamp((x + 1.) / 2., 1e-2, 1.)
        x = torch.pow(x, gamma)
        x = x * 2. - 1.
        return torch.clamp(x * gain + bias, -1, 1)

    def apply_ref_illum(self, x, low, ref):
        if not self.use_ref_illum or ref is None:
            return x
        return self.ref_illum(x, low, ref)

    @staticmethod
    def crop_stem(stem, y0, y1, x0, x1):
        """Crop a full-image ``encode_low_stem`` dict to a tile window.

        ``y0/y1/x0/x1`` are in full-resolution (H) coordinates on the padded
        low. Encoder features at H/2 and H/4 are cropped with integer halving.
        """
        def c(t, sy, ey, sx, ex):
            return t[..., sy:ey, sx:ex]

        e0 = c(stem['e0'], y0, y1, x0, x1)
        e1 = c(stem['e1'], y0 // 2, y1 // 2, x0 // 2, x1 // 2)
        e2 = c(stem['e2'], y0 // 4, y1 // 4, x0 // 4, x1 // 4)
        return dict(
            low=c(stem['low'], y0, y1, x0, x1),
            low01=c(stem['low01'], y0, y1, x0, x1),
            base=c(stem['base'], y0, y1, x0, x1),
            orig_hw=(y1 - y0, x1 - x0),
            e0=e0, e1=e1, e2=e2,
            log_gain=c(stem['log_gain'], y0, y1, x0, x1),
        )

    def forward(
        self,
        x,
        S=None,
        T_lv3=None,
        T_lv2=None,
        T_lv1=None,
        apply_illum=True,
        ref=None,
        apply_ref_illum=True,
        gate_mode=None,
        gate_shift=None,
        gate_smooth_kernel=31,
        collect_gate_stats=False,
        disable_T_lv3=False,
        disable_T_lv2=False,
        disable_T_lv1=False,
        stage_mode_lv3=None,
        stage_mode_lv2=None,
        stage_mode_lv1=None,
        collect_residual_decomp=False,
        use_reference: Optional[bool] = None,
        stem: Optional[dict] = None,
    ):
        """
        Args:
            x: Low RGB in [-1, 1]
            S / T_*: SearchTransfer soft-attention and textures
            stage_mode_lv*: ``normal`` | ``branch_bypass`` | ``texture_zero``
              (default ``normal`` — inject T_lv2 at H/2 like ASRT Stage-A)
        """
        if disable_T_lv3:
            stage_mode_lv3 = 'branch_bypass'
        if disable_T_lv2:
            stage_mode_lv2 = 'branch_bypass'
        if disable_T_lv1:
            stage_mode_lv1 = 'branch_bypass'
        stage_mode_lv3 = _normalize_stage_mode(stage_mode_lv3)
        stage_mode_lv2 = _normalize_stage_mode(stage_mode_lv2)
        stage_mode_lv1 = _normalize_stage_mode(stage_mode_lv1)

        if use_reference is None:
            use_reference = (
                T_lv3 is not None and T_lv2 is not None and T_lv1 is not None
                and S is not None)
        self.last_used_reference = bool(use_reference)

        if stem is None:
            stem = self.encode_low_stem(x)
        base = stem['base']
        orig_hw = stem['orig_hw']
        e0, e1, e2 = stem['e0'], stem['e1'], stem['e2']
        low = stem['low']

        gate_e0 = self._maybe_counterfactual_gate(
            S, e0, gate_mode=gate_mode, gate_shift=gate_shift,
            gate_smooth_kernel=gate_smooth_kernel,
            collect_gate_stats=collect_gate_stats)

        # Coarse @ H/4: inject T_lv3 → RB → LFCA
        x11, res11 = self._inject(
            e2, T_lv3, S, self.conv11_head, use_reference,
            stage_mode=stage_mode_lv3,
            gate=None if gate_e0 is None else resize_like(gate_e0, e2))
        x11 = self._run_rbs(x11, self.RB11, self.conv11_tail)
        x11 = self.coarse_freq_block(x11)

        # Middle @ H/2: skip + inject T_lv2 → CSFI2 → RB → LFCA
        x21 = x11
        x22 = self.skip_e1(torch.cat([resize_like(x11, e1), e1], dim=1))
        x22, res22 = self._inject(
            x22, T_lv2, S, self.conv22_head, use_reference,
            stage_mode=stage_mode_lv2,
            gate=None if gate_e0 is None else resize_like(gate_e0, e1))
        x21, x22 = self.ex12(x21, x22)
        x21 = self._run_rbs(x21, self.RB21, self.conv21_tail)
        x22 = self._run_rbs(x22, self.RB22, self.conv22_tail)
        x22 = self.middle_freq_block(x22)

        # Fine @ H: skip + inject T_lv1 → CSFI3 → RB (no attention block)
        x31, x32 = x21, x22
        x33 = self.skip_e0(torch.cat([resize_like(x22, e0), e0], dim=1))
        x33, res33 = self._inject(
            x33, T_lv1, S, self.conv33_head, use_reference,
            stage_mode=stage_mode_lv1,
            gate=gate_e0)
        x31, x32, x33 = self.ex123(x31, x32, x33)
        x31 = self._run_rbs(x31, self.RB31, self.conv31_tail)
        x32 = self._run_rbs(x32, self.RB32, self.conv32_tail)
        x33 = self._run_rbs(x33, self.RB33, self.conv33_tail)

        rgb_residual = self.merge_tail(x31, x32, x33)
        out = (base + rgb_residual)[..., :orig_hw[0], :orig_hw[1]]
        low_crop = low[..., :orig_hw[0], :orig_hw[1]]

        if apply_ref_illum:
            out = self.apply_ref_illum(out, low_crop, ref)
        if apply_illum:
            out = self.apply_illum_head(out, low_crop)

        self.last_gate = None if gate_e0 is None else gate_e0.detach()
        if collect_residual_decomp:
            self.last_residual_decomp = {
                'coarse': None if res11 is None else res11.detach(),
                'middle': None if res22 is None else res22.detach(),
                'fine': None if res33 is None else res33.detach(),
                'base': base[..., :orig_hw[0], :orig_hw[1]].detach(),
                'rgb_residual': rgb_residual[..., :orig_hw[0], :orig_hw[1]].detach(),
            }
        else:
            self.last_residual_decomp = None
        return out
