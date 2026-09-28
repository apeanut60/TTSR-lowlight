"""V3-A.5: one verifier trunk, two target resolutions.

    A0 dense_block_h4 : F_common (H/4 = 100x150)      -> head -> q [100,150]
    A1 g64            : area-pool F_common to 64x64   -> SAME head -> q [64,64]

The head is fully convolutional, so the two arms have **identical parameter
shapes and count**; the only difference is the spatial resolution of the feature
map the head sees (``adaptive_avg_pool2d`` = exact region pooling, plan §7).

Inputs are the V3-A.4 C0 common features only -- no action channels (§6):

    FX, F0, FR, |F0-FR|, |FX-FR|

The final layer is zero-initialised, so at step 0 every location of BOTH arms
returns sigmoid(0) = 0.5 and the arms are output-identical (§7/§15).
"""

import torch
import torch.nn.functional as F

from model.V3A1Refiner import LowAnchorVerifier
from v3a5_runtime import MODES, snapshot_


class V3A5Verifier(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise SystemExit('unknown V3-A.5 verifier mode %r (expected %r)'
                             % (mode, MODES))
        self.mode = mode
        self.net = LowAnchorVerifier()
        with torch.no_grad():                 # q == 0.5 everywhere at step 0
            self.net.head2.weight.zero_()
            self.net.head2.bias.zero_()

    def common_features(self, low, y0, reference):
        net = self.net
        fx = net.encode(low)
        f0 = net.encode(y0)
        fr = net.encode(reference)
        return torch.cat([fx, f0, fr, (f0 - fr).abs(), (fx - fr).abs()], dim=1)

    def forward(self, low, y0, reference, target_shape=None, strict_native=False):
        """-> q [B,1,nby,nbx] in (0,1); ``target_shape`` None = native H/4.

        ``strict_native`` is used by the dense arm: its target grid IS the native
        H/4 feature map, so silently pooling to a different grid would change the
        parameterisation instead of just resizing the input.
        """
        h = self.common_features(low, y0, reference)
        native = tuple(int(v) for v in h.shape[-2:])
        want = None if target_shape is None else tuple(int(v) for v in target_shape)
        if want is not None and native != want:
            if strict_native:
                raise SystemExit(
                    'dense arm expects the native H/4 feature map to equal the '
                    'Block_H4 grid, got %s vs %s -- refuse to silently re-pool '
                    'a different partition' % (native, want))
            h = F.adaptive_avg_pool2d(h, want)
        h = self.net.act(self.net.head0(h))
        h = self.net.act(self.net.head1(h))
        return torch.sigmoid(self.net.head2(h))


def build_shared_init(mode_a='dense_block_h4', mode_b='g64', seed=42):
    """-> dict mode -> state_dict with bit-equal parameters (§15).

    The two arms are parameter-identical by construction, so the second arm
    takes the first arm's tensors directly -- building two models under one seed
    is NOT enough, because each construction advances the RNG.
    """
    torch.manual_seed(int(seed))
    a = V3A5Verifier(mode_a)
    sa = snapshot_(a)
    b = V3A5Verifier(mode_b)
    b.load_state_dict({k: v.clone() for k, v in sa.items()}, strict=True)
    return {mode_a: sa, mode_b: snapshot_(b)}
