"""V3-A: low-frequency, hallucination-aware reference refinement.

Design (see V3A_LOLV2_REAL_HALLUCINATION_AWARE_PLAN.md §3):

    FX, F0, FR = E(X), E(Y0), E(R)              shared encoder, H/8
    delta8     = head_proposal([F0, FR, F0-FR]) zero-init last layer
    q8         = sigmoid(head_verifier([FX, F0, FR, |F0-FR|, |FX-FR|]))
    Y_hat      = Y0 + upsample(q8) * upsample(delta8)

Two structural facts that the whole experiment leans on:

  * the correction is produced at H/8, so it can only be low-frequency -- the
    previous round measured that the V2.refiner's gain was already 99% low
    frequency, this makes it explicit rather than accidental;
  * the proposal's last layer is zero-initialised, so at step 0
    ``Y_hat == Y0`` bit-for-bit while the verifier still receives ``L_trust``
    gradients.

``reference=None`` must return ``Y0`` exactly (the bypass condition).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    """Three stride-2 convs: H -> H/8."""

    def __init__(self, ch=(32, 48, 64)):
        super().__init__()
        c0, c1, c2 = ch
        self.conv0 = nn.Conv2d(3, c0, 3, 2, 1, bias=True)
        self.conv1 = nn.Conv2d(c0, c1, 3, 2, 1, bias=True)
        self.conv2 = nn.Conv2d(c1, c2, 3, 2, 1, bias=True)
        self.act = nn.GELU()
        self.out_ch = c2

    def forward(self, x):
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        return self.act(self.conv2(x))


class V3ARefiner(nn.Module):
    def __init__(self, ch=(32, 48, 64), hidden=64, ver_hidden=32):
        super().__init__()
        self.encoder = SharedEncoder(ch)
        c2 = ch[2]

        # proposal: 3*c2 -> hidden -> 3 (zero-init output)
        self.p_conv1 = nn.Conv2d(3 * c2, hidden, 3, 1, 1, bias=True)
        self.p_act = nn.GELU()
        self.p_out = nn.Conv2d(hidden, 3, 3, 1, 1, bias=True)

        # verifier: 5*c2 -> hidden2 -> 32 -> 1
        self.v_conv1 = nn.Conv2d(5 * c2, hidden, 1, 1, 0, bias=True)
        self.v_act1 = nn.GELU()
        self.v_conv2 = nn.Conv2d(hidden, ver_hidden, 3, 1, 1, bias=True)
        self.v_act2 = nn.GELU()
        self.v_out = nn.Conv2d(ver_hidden, 1, 1, 1, 0, bias=True)

        nn.init.zeros_(self.p_out.weight)
        nn.init.zeros_(self.p_out.bias)

    def forward(self, y0, reference=None, low=None):
        """y0, reference, low: [B,3,H,W] in [-1,1].

        ``reference=None`` -> exact bypass (returns y0 unchanged).
        ``low`` defaults to ``y0``; the verifier's FX term needs the low input
        that produced y0, which callers can pass explicitly.
        """
        if reference is None:
            return y0, dict(gate=None, delta=None, bypass=True)

        x = y0 if low is None else low
        fx = self.encoder(x)
        f0 = self.encoder(y0)
        fr = self.encoder(reference)

        d_fr = (f0 - fr).abs()
        d_xr = (fx - fr).abs()
        p = self.p_act(self.p_conv1(torch.cat([f0, fr, f0 - fr], dim=1)))
        delta8 = self.p_out(p)
        v = self.v_act1(self.v_conv1(torch.cat([fx, f0, fr, d_fr, d_xr], dim=1)))
        v = self.v_act2(self.v_conv2(v))
        q8 = torch.sigmoid(self.v_out(v))

        size = y0.shape[-2:]
        delta = F.interpolate(delta8, size=size, mode='bilinear',
                              align_corners=False)
        q = F.interpolate(q8, size=size, mode='bilinear', align_corners=False)
        out = y0 + q * delta
        return out, dict(gate=q, gate8=q8, delta=delta, delta8=delta8,
                         bypass=False)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
