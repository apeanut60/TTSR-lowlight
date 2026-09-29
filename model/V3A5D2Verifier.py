"""V3-A.5D2: G64 verifier RF/context ablation (no evidence).

Arms
----
A0_control   : current G64 verifier
A1_multiscale: MultiScaleContext on F_common H/4, then exact G64 pool + same head

Residual fusion of MultiScaleContext is zero-init so step0 A0 ≡ A1.
"""

import torch
import torch.nn as nn

from model.MultiScaleVerifierContext import MultiScaleContext
from model.V3A1Refiner import LowAnchorVerifier
from v3a5_runtime import pool_feature_by_geom, snapshot_

ARMS = ('A0_control', 'A1_multiscale')


class V3A5D2Verifier(nn.Module):
    def __init__(self, arm='A0_control', bottleneck=64):
        super().__init__()
        if arm not in ARMS:
            raise SystemExit('unknown D2 arm %r (expected %r)' % (arm, ARMS))
        self.arm = arm
        self.use_context = arm == 'A1_multiscale'
        self.net = LowAnchorVerifier()
        with torch.no_grad():
            self.net.head2.weight.zero_()
            self.net.head2.bias.zero_()
        common_ch = 5 * int(self.net.out_ch)  # 160
        if self.use_context:
            self.context = MultiScaleContext(
                channels=common_ch, bottleneck=int(bottleneck))
        else:
            self.context = None

    def common_features(self, low, y0, reference):
        net = self.net
        fx = net.encode(low)
        f0 = net.encode(y0)
        fr = net.encode(reference)
        return torch.cat([fx, f0, fr, (f0 - fr).abs(), (fx - fr).abs()], dim=1)

    def forward_head(self, h):
        h = self.net.act(self.net.head0(h))
        h = self.net.act(self.net.head1(h))
        return torch.sigmoid(self.net.head2(h))

    def prepare_features(self, h, geom):
        native = tuple(int(v) for v in h.shape[-2:])
        want = tuple(int(v) for v in geom['shape'])
        if native == want:
            return h
        return pool_feature_by_geom(h, geom)

    def context_residual_max_abs(self):
        if self.context is None:
            return 0.0
        w = self.context.fuse_h4[-1].weight
        b = self.context.fuse_h4[-1].bias
        return float(max(w.detach().abs().max(), b.detach().abs().max()))

    def forward(self, low, y0, reference, geom=None):
        h = self.common_features(low, y0, reference)
        if self.use_context:
            h = self.context(h)
        if geom is not None:
            h = self.prepare_features(h, geom)
        return self.forward_head(h)


def build_d2_shared_init(seed=42, bottleneck=64):
    """Shared common trunk; A1 context residual projection exactly zero."""
    torch.manual_seed(int(seed))
    a0 = V3A5D2Verifier('A0_control', bottleneck=bottleneck)
    sa0 = snapshot_(a0)

    a1 = V3A5D2Verifier('A1_multiscale', bottleneck=bottleneck)
    a1.net.load_state_dict(a0.net.state_dict(), strict=True)
    # Re-zero residual projection after any RNG advances in MultiScaleContext ctor.
    with torch.no_grad():
        a1.context.fuse_h4[-1].weight.zero_()
        a1.context.fuse_h4[-1].bias.zero_()
    if a1.context_residual_max_abs() != 0.0:
        raise SystemExit('A1 context residual not zero at init')
    return {
        'A0_control': snapshot_(a0),
        'A1_multiscale': snapshot_(a1),
    }
