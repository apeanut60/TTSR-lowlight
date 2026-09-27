"""V3-A.4: three verifier arms differing only in action conditioning.

    C0-clean     [FX, F0, FR, |F0-FR|, |FX-FR|]                     160 ch
    C1-raw-clean common160 + D4_raw(3) + E4_raw(1)                  164 ch
    C2-norm-clean common160 + D4_norm(3) + E4_norm(1)               164 ch

D4/E4 always come from ``D = g_v2 * delta``; normalization only rescales the
verifier's *input*, never the target (``q_opt`` keeps using the raw physical D).
The extra channels are zero-initialised so step 0 is identical across arms.
"""

import torch
import torch.nn.functional as F

from model.V3A1Refiner import V3A1Refiner
from v3a4_runtime import action_features_norm, action_features_raw

ACTION_CH = 4
MODES = ('none', 'raw', 'norm')


class V3A4Refiner(V3A1Refiner):
    def __init__(self, mode='none', action_rms=None):
        super().__init__()
        if mode not in MODES:
            raise SystemExit('unknown action mode %r' % mode)
        self.mode = mode
        self.use_action = mode in ('raw', 'norm')
        if self.use_action:
            v = self.verifier
            v.extra_ch = ACTION_CH
            old = v.head0
            new = torch.nn.Conv2d(5 * v.out_ch + ACTION_CH, old.out_channels,
                                  1, 1, 0, bias=True)
            with torch.no_grad():
                new.weight[:, :old.in_channels].copy_(old.weight)
                new.weight[:, old.in_channels:].zero_()
                new.bias.copy_(old.bias)
            v.head0 = new
        if action_rms is None:
            self.register_buffer('action_rms', torch.zeros(3))
        else:
            self.register_buffer('action_rms',
                                 torch.as_tensor(action_rms, dtype=torch.float32))

    def action_channels(self, D):
        if self.mode == 'raw':
            d4, e4 = action_features_raw(D)
        elif self.mode == 'norm':
            d4, e4 = action_features_norm(D, self.action_rms)
        else:
            return None, None, None
        return torch.cat([d4, e4], dim=1), d4, e4

    def train(self, mode=True):
        """The proposal is frozen for the whole of V3-A.4.

        ``model.train()`` is the obvious thing to call before a training loop and
        it silently flips every submodule -- including the frozen proposal. The
        plan's ordering (train() first, then proposal.eval()) relies on the
        caller; doing it here makes the invariant structural instead.
        """
        super().train(mode)
        self.proposal.eval()
        return self

    def forward(self, y0, reference=None, low=None, force_qv=None):
        if reference is None:
            return y0, dict(bypass=True, gate_v2=None, q_v=None, gate_final=None,
                            delta=None, q_v4=None, D=None, action=None,
                            d4=None, e4=None)
        if low is None:
            raise ValueError('V3A4Refiner requires the true low-light input')
        _sr_v2, aux = self.proposal(y0, reference)
        g_v2, delta = aux['gate'], aux['delta']
        D = g_v2 * delta
        action, d4, e4 = self.action_channels(D)
        q_v4 = self.verifier(low, y0, reference, extra=action)
        q_v = F.interpolate(q_v4, size=y0.shape[-2:], mode='bilinear',
                            align_corners=False)
        if force_qv is not None:
            q_v = torch.full_like(q_v, float(force_qv))
        return y0 + (g_v2 * q_v) * delta, dict(
            bypass=False, gate_v2=g_v2, q_v=q_v, q_v4=q_v4,
            gate_final=g_v2 * q_v, delta=delta, D=D,
            action=action, d4=d4, e4=e4)


def build_shared_init(seed=42, action_rms=None):
    """-> dict mode -> state_dict whose common weights are identical.

    One model is constructed; the other two take its tensors by construction.
    Building three models under one seed is NOT enough -- each construction
    advances the RNG, so their "common" weights would differ.
    """
    torch.manual_seed(seed)
    c1 = V3A4Refiner('raw', action_rms)
    s1 = {k: v.clone() for k, v in c1.state_dict().items()}

    c0, c2 = V3A4Refiner('none', action_rms), V3A4Refiner('norm', action_rms)
    s0, s2 = {}, {}
    for k, t in c0.state_dict().items():
        if k == 'verifier.head0.weight':
            s0[k] = s1[k][:, :t.shape[1]].clone()      # drop the action channels
        else:
            assert s1[k].shape == t.shape, (k, s1[k].shape, t.shape)
            s0[k] = s1[k].clone()
    for k, t in c2.state_dict().items():
        assert s1[k].shape == t.shape, (k, s1[k].shape, t.shape)
        s2[k] = s1[k].clone()                          # identical incl. action=0
    # every common tensor must be bit-equal across the three arms
    for k in s0:
        if k == 'verifier.head0.weight':
            assert torch.equal(s0[k], s1[k][:, :s0[k].shape[1]])
            assert torch.equal(s2[k], s1[k])
        else:
            assert torch.equal(s0[k], s1[k]) and torch.equal(s2[k], s1[k]), k
    if float(s1['verifier.head0.weight'][:, -ACTION_CH:].abs().max()) != 0.0:
        raise AssertionError('action channels must be zero-initialised')
    return {'none': s0, 'raw': s1, 'norm': s2}
