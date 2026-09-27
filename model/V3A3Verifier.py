"""V3-A.3: action-conditioned verifier.

Only one thing changes relative to V3-A.2: the verifier is shown the correction
the proposal is about to apply.

    D  = g_v2 * delta
    D4 = area_downsample(D, H/4)          # 3 ch
    E4 = mean_rgb(D4^2)                   # 1 ch
    input = [FX, F0, FR, |F0-FR|, |FX-FR|, D4, E4]      # 160 -> 164 ch

The target is unchanged, and the extra channels are zero-initialised so that
C1's step 0 is identical to C0's -- any later difference comes only from what
the action channels learn.
"""

import copy

import torch
import torch.nn.functional as F

from model.V3A1Refiner import LowAnchorVerifier, V3A1Refiner

ACTION_CH = 4


def action_features(d_correction, factor=4):
    """-> (D4 [B,3,h,w], E4 [B,1,h,w]); area downsampling, no normalisation."""
    h, w = d_correction.shape[-2:]
    th, tw = max(1, h // factor), max(1, w // factor)
    d4 = F.interpolate(d_correction, size=(th, tw), mode='area')
    e4 = (d4 ** 2).mean(dim=1, keepdim=True)
    return d4, e4


class V3A3Refiner(V3A1Refiner):
    """V2-stable proposal + verifier, optionally conditioned on the action."""

    def __init__(self, use_action=True):
        super().__init__()
        self.use_action = bool(use_action)
        if self.use_action:
            # rebuild head0 with the extra action channels
            v = self.verifier
            v.extra_ch = ACTION_CH
            old = v.head0
            new = torch.nn.Conv2d(5 * v.out_ch + ACTION_CH, old.out_channels,
                                  1, 1, 0, bias=True).to(old.weight.device)
            with torch.no_grad():
                new.weight[:, :old.in_channels].copy_(old.weight)
                new.weight[:, old.in_channels:].zero_()      # action channels = 0
                new.bias.copy_(old.bias)
            v.head0 = new

    def forward(self, y0, reference=None, low=None, force_qv=None):
        if reference is None:
            return y0, dict(bypass=True, gate_v2=None, q_v=None, gate_final=None,
                            delta=None, q_v4=None)
        if low is None:
            raise ValueError('V3A3Refiner requires the true low-light input')
        _sr_v2, aux = self.proposal(y0, reference)
        g_v2, delta = aux['gate'], aux['delta']
        D = g_v2 * delta
        extra = None
        if self.use_action:
            d4, e4 = action_features(D)
            extra = torch.cat([d4, e4], dim=1)
        q_v4 = self.verifier(low, y0, reference, extra=extra)
        q_v = F.interpolate(q_v4, size=y0.shape[-2:], mode='bilinear',
                            align_corners=False)
        if force_qv is not None:
            q_v = torch.full_like(q_v, float(force_qv))
        gate_final = g_v2 * q_v
        return y0 + gate_final * delta, dict(
            bypass=False, gate_v2=g_v2, q_v=q_v, q_v4=q_v4,
            gate_final=gate_final, delta=delta, D=D, action=extra)


def build_shared_init(seed=42):
    """-> (c0_state_dict, c1_state_dict) whose common weights are identical.

    C1 is built first (with action channels), then C0 takes everything except the
    last four input channels of head0. That makes "C0's 160 channels == C1's
    first 160 channels" true by construction rather than by luck.
    """
    torch.manual_seed(seed)
    c1 = V3A3Refiner(use_action=True)
    c0 = V3A3Refiner(use_action=False)
    s1, s0 = c1.state_dict(), c0.state_dict()
    for k in s0:
        if k == 'verifier.head0.weight':
            s0[k] = s1[k][:, :s0[k].shape[1]].clone()
        else:
            s0[k] = s1[k].clone()
    return s0, s1
