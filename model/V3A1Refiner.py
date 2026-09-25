"""V3-A.1: V2-stable proposal + low-anchor verifier.

The V3-A H/8 RGB proposal was abandoned: even with a GT-derived gate it stayed
0.80 dB below the frozen base, and 37/100 images moved by more than 5 dB (whole
-image level rewrites). This model keeps the V2 proposal/output path, which was
measured stable (+0.374 with the v2 references on the same base and test set),
and adds only a *suppressive* trust factor:

    sr_v2, aux = proposal(Y0, R)          # V2 path, unchanged
    q_v        = verifier(X, Y0, R)       # H/4, independent encoder
    Y_hat      = Y0 + (g_v2 * q_v) * delta

Two structural guarantees:

  * both factors live in [0,1], so ``q_v`` can only ever *shrink* the V2
    correction -- it cannot amplify it;
  * ``force_qv=1`` reproduces the V2 output bit-for-bit, and ``force_qv=0``
    returns ``Y0`` exactly, which is what the plan's gate-safety test checks.

``low`` is REQUIRED whenever a reference is given. An earlier version defaulted
it to ``Y0`` and every caller silently took that path, so the verifier's FX term
collapsed to F0 and it never saw the low-light input at all. Raising here makes
that failure mode impossible instead of merely fixed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LocalRefine import Refiner as V2Proposal


class LowAnchorVerifier(nn.Module):
    """Independent encoder (H/4) + small head -> q_v in [0,1]."""

    def __init__(self, ch=(24, 32), hidden=32, mid=16):
        super().__init__()
        c0, c1 = ch
        self.conv0 = nn.Conv2d(3, c0, 3, 2, 1, bias=True)
        self.conv1 = nn.Conv2d(c0, c1, 3, 2, 1, bias=True)
        self.act = nn.GELU()
        # inputs: FX, F0, FR, |F0-FR|, |FX-FR|
        self.head0 = nn.Conv2d(5 * c1, hidden, 1, 1, 0, bias=True)
        self.head1 = nn.Conv2d(hidden, mid, 3, 1, 1, bias=True)
        self.head2 = nn.Conv2d(mid, 1, 1, 1, 0, bias=True)
        self.out_ch = c1

    def encode(self, x):
        return self.act(self.conv1(self.act(self.conv0(x))))

    def forward(self, x, y0, reference):
        fx = self.encode(x)
        f0 = self.encode(y0)
        fr = self.encode(reference)
        h = torch.cat([fx, f0, fr, (f0 - fr).abs(), (fx - fr).abs()], dim=1)
        h = self.act(self.head0(h))
        h = self.act(self.head1(h))
        return torch.sigmoid(self.head2(h))


class V3A1Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.proposal = V2Proposal()
        self.verifier = LowAnchorVerifier()

    def forward(self, y0, reference=None, low=None, force_qv=None):
        if reference is None:
            return y0, dict(bypass=True, gate_v2=None, q_v=None, gate_final=None,
                            delta=None, q_v4=None)
        if low is None:
            raise ValueError(
                'V3A1Refiner requires the true low-light input: the verifier is '
                'anchored on X, and defaulting it to Y0 collapses FX to F0')

        _sr_v2, aux = self.proposal(y0, reference)
        g_v2, delta = aux['gate'], aux['delta']
        q_v4 = self.verifier(low, y0, reference)
        q_v = F.interpolate(q_v4, size=y0.shape[-2:], mode='bilinear',
                            align_corners=False)
        if force_qv is not None:
            q_v = torch.full_like(q_v, float(force_qv))
        gate_final = g_v2 * q_v
        out = y0 + gate_final * delta
        return out, dict(bypass=False, gate_v2=g_v2, q_v=q_v, q_v4=q_v4,
                         gate_final=gate_final, delta=delta)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
