"""
MainNetEnhance — 1:1 resolution enhancement network for low-light image enhancement.
Adapted from MainNet (4x SR) by removing all PixelShuffle upsampling and keeping
all feature streams at the same spatial resolution.
Cross-stream feature integration (CSFI) uses dilated convolutions instead of
stride/interpolation to capture multi-scale context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=dilation, dilation=dilation, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class SFE(nn.Module):
    """Shallow feature extraction — same as original."""
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(3, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                     res_scale=res_scale))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


# ─── CSFI modules modified for same-resolution ───────────────────────────

class CSFI2_SameRes(nn.Module):
    """Cross-Stream Feature Integration for 2 streams at same resolution.
    Uses dilated convs (d1 + d3) for multi-scale context without resolution change.
    """
    def __init__(self, n_feats):
        super(CSFI2_SameRes, self).__init__()
        # x1 ↔ x2 cross branches with multi-scale dilated convs
        self.conv12_d1 = conv3x3(n_feats, n_feats, dilation=1)
        self.conv12_d3 = conv3x3(n_feats, n_feats, dilation=3)
        self.conv21_d1 = conv3x3(n_feats, n_feats, dilation=1)
        self.conv21_d3 = conv3x3(n_feats, n_feats, dilation=3)

        self.conv_merge1 = conv3x3(n_feats * 3, n_feats)
        self.conv_merge2 = conv3x3(n_feats * 3, n_feats)

    def forward(self, x1, x2):
        # cross-branch features with multi-scale dilated convs
        x12 = self.conv12_d1(x1) + self.conv12_d3(x1)
        x12 = F.relu(x12)
        x21 = self.conv21_d1(x2) + self.conv21_d3(x2)
        x21 = F.relu(x21)

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x1), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x2), dim=1)))

        return x1, x2


class CSFI3_SameRes(nn.Module):
    """Cross-Stream Feature Integration for 3 streams at same resolution.
    Each stream sends to the other two with d1+d3 for multi-scale context.
    """
    def __init__(self, n_feats):
        super(CSFI3_SameRes, self).__init__()
        # cross-branch: each stream sends one d1 and one d3 path to the other two
        self.conv12 = conv3x3(n_feats, n_feats, dilation=1)
        self.conv13 = conv3x3(n_feats, n_feats, dilation=3)
        self.conv21 = conv3x3(n_feats, n_feats, dilation=1)
        self.conv23 = conv3x3(n_feats, n_feats, dilation=3)
        self.conv31 = conv3x3(n_feats, n_feats, dilation=3)
        self.conv32 = conv3x3(n_feats, n_feats, dilation=1)

        self.conv_merge1 = conv3x3(n_feats * 3, n_feats)
        self.conv_merge2 = conv3x3(n_feats * 3, n_feats)
        self.conv_merge3 = conv3x3(n_feats * 3, n_feats)

    def forward(self, x1, x2, x3):
        # cross-stream features
        x12 = F.relu(self.conv12(x1))
        x13 = F.relu(self.conv13(x1))
        x21 = F.relu(self.conv21(x2))
        x23 = F.relu(self.conv23(x2))
        x31 = F.relu(self.conv31(x3))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x31), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x32), dim=1)))
        x3 = F.relu(self.conv_merge3(torch.cat((x3, x13, x23), dim=1)))

        return x1, x2, x3


# ─── MergeTail for same resolution ───────────────────────────────────────

class MergeTail_SameRes(nn.Module):
    """Merge three streams at same spatial resolution. No interpolation needed."""
    def __init__(self, n_feats):
        super(MergeTail_SameRes, self).__init__()
        self.conv_merge = conv3x3(n_feats * 3, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, 3)

    def forward(self, x1, x2, x3):
        # all at same resolution — just concat and process
        x = F.relu(self.conv_merge(torch.cat((x1, x2, x3), dim=1)))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)
        return x


class GlobalIllumHead(nn.Module):
    """Per-image global illumination correction.

    Predicts a per-channel gamma curve plus multiplicative gain and additive
    bias from the global statistics (mean/std) of the low-light input. All
    three are zero initialized, so ``gamma == gain == 1`` and ``bias == 0``
    at load time and old checkpoints keep their exact previous output until
    the head is trained.
    """

    def __init__(self, hidden=32):
        super(GlobalIllumHead, self).__init__()
        self.fc1 = nn.Linear(6, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc_gain = nn.Linear(hidden, 3)
        self.fc_bias = nn.Linear(hidden, 3)
        self.fc_gamma = nn.Linear(hidden, 3)
        for fc in (self.fc_gain, self.fc_bias, self.fc_gamma):
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, low):
        m = low.mean(dim=(2, 3))
        s = low.std(dim=(2, 3))
        h = self.act(self.fc1(torch.cat((m, s), dim=1)))
        log_gain = torch.clamp(self.fc_gain(h), -1.5, 1.5)
        gain = torch.exp(log_gain).view(-1, 3, 1, 1)
        bias = torch.clamp(self.fc_bias(h), -0.5, 0.5).view(-1, 3, 1, 1)
        log_gamma = torch.clamp(self.fc_gamma(h), -0.7, 0.7)
        gamma = torch.exp(log_gamma).view(-1, 3, 1, 1)
        return gain, bias, gamma


class RefIllumTransfer(nn.Module):
    """Add the reference's low-frequency illumination/tone to the output.

    The backbone's only reference path is texture transfer (T_lv1/2/3 + S), so
    the reference never decides how bright/tinted the result should be. This
    module reads the low-pass of the reference, the low input and the *current
    output* (x), predicts a low-frequency residual at 1/pool resolution and adds
    it back at full resolution.

    Two design points come from measurements, not taste:
    - Feeding ``x`` matters. Without it the module must guess how far off the
      current output is; a linear probe reaches +0.51 dB blind vs +1.00 dB with
      ``x``. (A stats-only variant could not condition on the reference at all:
      it predicted gain 0.970 for a dark ref and 0.972 for a bright one.)
    - The readout is *not* zero-initialised. With zeros, dL/dconv1 == 0 and
      conv1/conv2 stay at their random init (measured +1.3% movement over 30
      epochs). Scaling the default init by 0.05 keeps the initial residual below
      one grey level while letting the whole module train from step 1.
    """

    def __init__(self, n_feats=16, pool=8, init_scale=0.05):
        super(RefIllumTransfer, self).__init__()
        self.pool = max(1, int(pool))
        self.conv1 = conv3x3(9, n_feats)
        self.conv2 = conv3x3(n_feats, n_feats)
        self.conv3 = conv3x3(n_feats, 3)
        with torch.no_grad():
            self.conv3.weight.mul_(init_scale)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x, low, ref):
        if ref is None:
            return x
        p = self.pool
        lp_ref = F.avg_pool2d(ref, p) if p > 1 else ref
        lp_low = F.avg_pool2d(low, p) if p > 1 else low
        lp_x = F.avg_pool2d(x, p) if p > 1 else x
        h = F.relu(self.conv1(torch.cat((lp_ref, lp_low, lp_x), dim=1)))
        h = F.relu(self.conv2(h))
        delta = self.conv3(h)
        if delta.shape[-2:] != x.shape[-2:]:
            delta = F.interpolate(delta, size=x.shape[-2:], mode='bilinear',
                                  align_corners=False)
        return torch.clamp(x + delta, -1, 1)


# ─── MainNetEnhance ──────────────────────────────────────────────────────

class MainNetEnhance(nn.Module):
    """1:1 resolution enhancement network for low-light enhancement.
    Three parallel feature streams at the same spatial resolution,
    receiving transferred textures (T_lv1/2/3) from the SearchTransfer module.

    Architecture:
        input (h×w) ── SFE ──┬── Stage1 (h×w) + T_lv3 + S
                              ├── Stage2 (h×w) + T_lv2 + S
                              └── Stage3 (h×w) + T_lv1 + S
                              CSFI modules exchange features across streams.
                              MergeTail produces final output (h×w).
    """
    def __init__(self, num_res_blocks, n_feats, res_scale, ref_illum_pool=8):
        super(MainNetEnhance, self).__init__()
        self.num_res_blocks = num_res_blocks  # list: [sfe, stage1, stage2, stage3]
        self.n_feats = n_feats

        self.SFE = SFE(self.num_res_blocks[0], n_feats, res_scale)

        # ── Stage1 (base stream, h×w) ──
        # input: SFE(x) + T_lv3 (256 ch from VGG relu3_4) * S
        self.conv11_head = conv3x3(256 + n_feats, n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        # ── Stage2 (h×w) ──
        # input: SFE(x) + T_lv2 (128 ch from VGG relu2_2) * S
        self.conv22_head = conv3x3(128 + n_feats, n_feats)

        self.ex12 = CSFI2_SameRes(n_feats)

        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        # ── Stage3 (h×w) ──
        # input: SFE(x) + T_lv1 (64 ch from VGG relu1_2) * S
        self.conv33_head = conv3x3(64 + n_feats, n_feats)

        self.ex123 = CSFI3_SameRes(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        # ── Merge ──
        self.merge_tail = MergeTail_SameRes(n_feats)

        # ── Global illumination correction (per-image gain/bias) ──
        self.global_illum_head = GlobalIllumHead()

        # ── Reference-driven low-frequency illumination transfer ──
        self.ref_illum = RefIllumTransfer(pool=ref_illum_pool)

    def apply_ref_illum(self, x, low, ref):
        """Add the reference-driven low-frequency illumination residual."""
        if ref is None:
            return x
        return self.ref_illum(x, low, ref)

    def apply_illum_head(self, x, low):
        """Apply the global illumination correction as a separate step.

        Keeping this out of ``forward`` lets tiled inference run the backbone
        per tile and apply the (per-image) illumination correction exactly once
        on the stitched result, so neighbouring tiles cannot end up with
        mismatched gamma/gain.
        """
        gain, bias, gamma = self.global_illum_head(low)
        # NOTE: the base must stay strictly positive. pow(0, gamma) has an
        # infinite gradient w.r.t. the base when gamma < 1, which turns the
        # whole network into NaN (observed at ~epoch 10 of P6).
        x = torch.clamp((x + 1.) / 2., 1e-2, 1.)
        x = torch.pow(x, gamma)
        x = x * 2. - 1.
        x = torch.clamp(x * gain + bias, -1, 1)
        return x

    @staticmethod
    def build_gate_from_S(S, size_hw):
        """G = sigmoid(bicubic_upsample(S)) — production gate definition."""
        S_up = F.interpolate(S, size=size_hw, mode='bicubic')
        return torch.sigmoid(S_up)

    @staticmethod
    def apply_gate_transform(G, gate_mode=None, gate_shift=None,
                             smooth_kernel=31):
        """Eval-only counterfactual transforms on the upsampled sigmoid gate.

        Default ``gate_mode is None`` returns G unchanged (production path).
        """
        if gate_mode is None or gate_mode in ('default', 'a0', 'none', ''):
            return G
        if gate_mode == 'constant_mean':
            # Per-tile / per-batch spatial mean of G (already sigmoid+upsample).
            # Not sigmoid(mean(S)).
            return G.mean(dim=(2, 3), keepdim=True).expand_as(G)
        if gate_mode == 'smooth':
            k = int(smooth_kernel)
            if k % 2 == 0:
                raise ValueError('smooth_kernel must be odd, got %d' % k)
            pad = k // 2
            # Fixed avg-pool low-pass; restore per-sample mean; clamp to [0,1].
            g_s = F.avg_pool2d(G, kernel_size=k, stride=1, padding=pad)
            mean_g = G.mean(dim=(2, 3), keepdim=True)
            mean_s = g_s.mean(dim=(2, 3), keepdim=True)
            g_s = g_s + (mean_g - mean_s)
            return g_s.clamp(0.0, 1.0)
        if gate_mode == 'shift':
            if gate_shift is None:
                raise ValueError('gate_mode=shift requires gate_shift=(dy,dx)')
            dy, dx = int(gate_shift[0]), int(gate_shift[1])
            return torch.roll(G, shifts=(dy, dx), dims=(2, 3))
        raise ValueError('Unknown gate_mode %r' % (gate_mode,))

    @staticmethod
    def _resolve_stage_mode(mode, disable_flag):
        """Map stage mode / legacy disable_T flag → normal|texture_zero|branch_bypass.

        Explicit ``stage_mode_*`` always wins. Legacy ``disable_T_lv*`` only
        applies when mode is left unset (None/'').
        """
        valid = ('normal', 'texture_zero', 'branch_bypass')
        if mode is not None and str(mode).strip() != '':
            mode = str(mode).lower().strip()
            if mode not in valid:
                raise ValueError(
                    'stage mode must be one of %s, got %r' % (valid, mode))
            return mode
        return 'texture_zero' if disable_flag else 'normal'

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None,
                apply_illum=True, ref=None, apply_ref_illum=True,
                gate_mode=None, gate_shift=None, gate_smooth_kernel=31,
                collect_gate_stats=False,
                disable_T_lv3=False, disable_T_lv2=False, disable_T_lv1=False,
                stage_mode_lv3=None, stage_mode_lv2=None, stage_mode_lv1=None,
                collect_residual_decomp=False):
        """
        Soft-attention order at each injection stage (confirmed):
          1) concat(stream, T_*_up)
          2) conv*_head → residual
          3) residual *= G   (same G for stages 1–3)
          4) stream += gated residual
          5) subsequent RBs / CSFI / merge / illum heads

        Eval-only stage modes (v1r12):
          - normal: original T
          - texture_zero: T→0 before concat; still run conv_head + G + add
            (≡ v1r11 disable_T_lv*)
          - branch_bypass: skip conv_head residual entirely (stream unchanged)

        Default all-normal / all-disable False = production path.
        """
        low_input = x
        # Shallow feature extraction
        x = self.SFE(x)  # [N, n_feats, H, W]
        H, W = x.size()[-2:]

        mode_lv3 = self._resolve_stage_mode(stage_mode_lv3, disable_T_lv3)
        mode_lv2 = self._resolve_stage_mode(stage_mode_lv2, disable_T_lv2)
        mode_lv1 = self._resolve_stage_mode(stage_mode_lv1, disable_T_lv1)

        # Upsample T tensors and S to match MainNet feature spatial size
        # VGG LTE gives: lv1 at H×W, lv2 at H/2×W/2, lv3 at H/4×W/4
        # SearchTransfer returns T at native VGG scales:
        #   T_lv3 @ H/4×W/4 (256ch), T_lv2 @ H/2×W/2 (128ch),
        #   T_lv1 @ H×W (64ch), S @ H/4×W/4
        T_lv3_up = F.interpolate(T_lv3, size=(H, W), mode='bicubic')
        T_lv2_up = F.interpolate(T_lv2, size=(H, W), mode='bicubic')
        T_lv1_up = F.interpolate(T_lv1, size=(H, W), mode='bicubic')
        G0 = self.build_gate_from_S(S, (H, W))
        S_up = self.apply_gate_transform(
            G0, gate_mode=gate_mode, gate_shift=gate_shift,
            smooth_kernel=gate_smooth_kernel)

        stage_stats = [] if collect_gate_stats else None
        decomp_stats = [] if collect_residual_decomp else None

        def _inject(stream, T_up, head_conv, tag, mode):
            if mode == 'branch_bypass':
                if stage_stats is not None:
                    stage_stats.append({
                        'stage': tag, 'mode': mode,
                        'res_norm_pre_mean': 0.0,
                        'res_norm_post_mean': 0.0,
                        'rel_inject': 0.0,
                        'res_post_spatial_cv': 0.0,
                        'res_post_map': torch.zeros(
                            stream.shape[-2:], device='cpu'),
                        'bypassed': True,
                    })
                if decomp_stats is not None:
                    decomp_stats.append({
                        'stage': tag, 'mode': mode, 'bypassed': True,
                    })
                return stream

            T_in = torch.zeros_like(T_up) if mode == 'texture_zero' else T_up
            res_pre = head_conv(torch.cat((stream, T_in), dim=1))
            res_post = res_pre * S_up

            if decomp_stats is not None and mode == 'normal':
                # Local decomposition on the same incoming F (stream).
                with torch.no_grad():
                    r_n = head_conv(torch.cat((stream, T_up), dim=1)).detach()
                    r_z = head_conv(torch.cat(
                        (stream, torch.zeros_like(T_up)), dim=1)).detach()
                    r_t = r_n - r_z
                    g = S_up.detach()
                    def _rms(t):
                        return float(t.float().pow(2).mean().sqrt().item())
                    def _mae(t):
                        return float(t.float().abs().mean().item())
                    def _l2(t):
                        return float(t.float().norm().item())
                    rn, rz, rt = _l2(r_n), _l2(r_z), _l2(r_t)
                    cos = float(F.cosine_similarity(
                        r_n.flatten(), r_z.flatten(), dim=0).item())
                    decomp_stats.append({
                        'stage': tag, 'mode': mode,
                        'R_normal_mae': _mae(r_n), 'R_normal_rms': _rms(r_n),
                        'R_zero_mae': _mae(r_z), 'R_zero_rms': _rms(r_z),
                        'R_texture_mae': _mae(r_t), 'R_texture_rms': _rms(r_t),
                        'R_zero_over_normal_l2': rz / max(rn, 1e-8),
                        'R_texture_over_normal_l2': rt / max(rn, 1e-8),
                        'cos_R_normal_R_zero': cos,
                        'recon_rel_err': float(
                            (r_n - r_z - r_t).float().norm() /
                            max(rn, 1e-8)),
                        'R_gated_normal_map': (g * r_n)[0].norm(dim=0).cpu(),
                        'R_gated_zero_map': (g * r_z)[0].norm(dim=0).cpu(),
                        'R_gated_texture_map': (g * r_t)[0].norm(dim=0).cpu(),
                    })

            if stage_stats is not None:
                with torch.no_grad():
                    n_pre = res_pre.detach().float().norm(dim=1)
                    n_post = res_post.detach().float().norm(dim=1)
                    stage_stats.append({
                        'stage': tag, 'mode': mode,
                        'res_norm_pre_mean': float(n_pre.mean().item()),
                        'res_norm_post_mean': float(n_post.mean().item()),
                        'rel_inject': float(
                            (n_post.mean() /
                             n_pre.mean().clamp(min=1e-8)).item()),
                        'res_post_spatial_cv': float(
                            (n_post.std(unbiased=False) /
                             n_post.mean().clamp(min=1e-8)).item()),
                        'res_post_map': n_post[0].cpu(),
                        'bypassed': False,
                        't_was_zero': mode == 'texture_zero',
                    })
            return stream + res_post

        # ── Stage1 ──
        x11 = x
        x11 = _inject(x11, T_lv3_up, self.conv11_head, 'stage1_Tlv3', mode_lv3)

        x11_res = x11
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        # ── Stage2 ──
        x21 = x11
        x21_res = x21
        x22 = x

        # Soft-attention: inject T_lv2
        x22 = _inject(x22, T_lv2_up, self.conv22_head, 'stage2_Tlv2', mode_lv2)

        x22_res = x22

        # Cross-stream feature exchange (same res, no up/down-sampling)
        x21_res, x22_res = self.ex12(x21_res, x22_res)

        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)

        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21 = x21 + x21_res
        x22 = x22 + x22_res

        # ── Stage3 ──
        x31 = x21
        x31_res = x31
        x32 = x22
        x32_res = x32
        x33 = x

        # Soft-attention: inject T_lv1
        x33 = _inject(x33, T_lv1_up, self.conv33_head, 'stage3_Tlv1', mode_lv1)

        x33_res = x33

        # Cross-stream feature exchange (same res)
        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)

        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res

        # Merge — all same resolution
        x = self.merge_tail(x31, x32, x33)

        # Reference-driven low-frequency illumination (tone / brightness)
        if apply_ref_illum:
            x = self.apply_ref_illum(x, low_input, ref)

        # Per-image global illumination correction (gamma + gain + bias)
        if apply_illum:
            x = self.apply_illum_head(x, low_input)

        # Diagnostics (not part of checkpoint / not used in training)
        self.last_gate = S_up.detach()
        self.last_gate_pre_transform = G0.detach()
        self.last_stage_modes = {
            'lv3': mode_lv3, 'lv2': mode_lv2, 'lv1': mode_lv1,
        }
        self.last_gate_stats = None
        self.last_residual_decomp = None
        if collect_gate_stats:
            with torch.no_grad():
                g = S_up.detach().float()
                g0 = G0.detach().float()
                self.last_gate_stats = {
                    'G_mean': float(g.mean().item()),
                    'G_std': float(g.std(unbiased=False).item()),
                    'G_cv': float(
                        (g.std(unbiased=False) /
                         g.mean().clamp(min=1e-8)).item()),
                    'G0_mean': float(g0.mean().item()),
                    'G0_std': float(g0.std(unbiased=False).item()),
                    'G_min': float(g.min().item()),
                    'G_max': float(g.max().item()),
                    'stages': [
                        {k: v for k, v in s.items() if k != 'res_post_map'}
                        for s in stage_stats
                    ],
                    'stage_maps': {
                        s['stage']: s['res_post_map'] for s in stage_stats
                    },
                }
        if collect_residual_decomp:
            self.last_residual_decomp = [
                {k: v for k, v in d.items()
                 if not k.endswith('_map')}
                for d in decomp_stats
            ]
            self.last_residual_decomp_maps = {
                d['stage']: {
                    k: d[k] for k in d if k.endswith('_map')
                } for d in decomp_stats if not d.get('bypassed')
            }
        return x
