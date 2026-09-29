"""V3-A.5D1: G64 verifier with optional narrow evidence head.

Arms
----
A0_control  : identical to V3-A.5 G64 verifier (no evidence)
A1_evidence : same common encoder + EvidenceAwareHead with
              [sim_max, f0_minus_t, gate_v2] concat only at G64 head

RF / MultiScaleContext are deliberately absent (that is D2).
"""

import torch
import torch.nn as nn

from model.EvidenceFusion import EvidenceAwareHead, evidence_weight_max_abs
from model.V3A1Refiner import LowAnchorVerifier
from v3a5_runtime import pool_feature_by_geom, snapshot_

ARMS = ('A0_control', 'A1_evidence')
EVIDENCE_NAMES = ('sim_max', 'f0_minus_t', 'gate_v2')
N_EVIDENCE = len(EVIDENCE_NAMES)


class V3A5DVerifier(nn.Module):
    def __init__(self, arm='A0_control'):
        super().__init__()
        if arm not in ARMS:
            raise SystemExit('unknown D1 arm %r (expected %r)' % (arm, ARMS))
        self.arm = arm
        self.use_evidence = arm == 'A1_evidence'
        self.net = LowAnchorVerifier()
        with torch.no_grad():
            self.net.head2.weight.zero_()
            self.net.head2.bias.zero_()
        if self.use_evidence:
            self.ev_head = EvidenceAwareHead(
                self.net.head0, self.net.head1, self.net.head2,
                self.net.act, N_EVIDENCE)
        else:
            self.ev_head = None
        # Norm buffers: z-score for sim_max / f0_minus_t; gate_v2 identity.
        # Defaults are identity until setup loads train575 stats.
        self.register_buffer('ev_mean', torch.zeros(N_EVIDENCE))
        self.register_buffer('ev_std', torch.ones(N_EVIDENCE))
        self.register_buffer('ev_zscore_mask',
                             torch.tensor([1.0, 1.0, 0.0]))  # gate raw

    def set_evidence_norm(self, mean, std, zscore_mask=None):
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(N_EVIDENCE)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(N_EVIDENCE)
        if float(std.min()) <= 0:
            raise SystemExit('evidence std must be positive, got %s' % std.tolist())
        self.ev_mean.copy_(mean)
        self.ev_std.copy_(std)
        if zscore_mask is not None:
            m = torch.as_tensor(zscore_mask, dtype=torch.float32).reshape(N_EVIDENCE)
            self.ev_zscore_mask.copy_(m)

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

    def normalize_evidence(self, evidence_g64):
        """evidence_g64: [B,3,64,64] raw pooled maps in EVIDENCE_NAMES order."""
        if evidence_g64.shape[1] != N_EVIDENCE:
            raise SystemExit('expected %d evidence ch, got %d'
                             % (N_EVIDENCE, evidence_g64.shape[1]))
        mean = self.ev_mean.view(1, -1, 1, 1)
        std = self.ev_std.view(1, -1, 1, 1)
        mask = self.ev_zscore_mask.view(1, -1, 1, 1)
        z = (evidence_g64 - mean) / std.clamp(min=1e-6)
        return evidence_g64 * (1.0 - mask) + z * mask

    def forward(self, low, y0, reference, geom=None, evidence_g64=None):
        h = self.common_features(low, y0, reference)
        if geom is not None:
            h = self.prepare_features(h, geom)
        if not self.use_evidence:
            return self.forward_head(h)
        if evidence_g64 is None:
            raise ValueError('A1_evidence requires evidence_g64 [B,3,nby,nbx]')
        if tuple(evidence_g64.shape[-2:]) != tuple(h.shape[-2:]):
            raise SystemExit('evidence spatial %s != common %s'
                             % (tuple(evidence_g64.shape[-2:]),
                                tuple(h.shape[-2:])))
        ev = self.normalize_evidence(evidence_g64)
        return self.ev_head(h, ev)

    def evidence_weight_max_abs(self):
        if self.ev_head is None:
            return 0.0
        return evidence_weight_max_abs(self.ev_head)


def build_d1_shared_init(seed=42, ev_mean=None, ev_std=None):
    """Bit-equal common trunks; A1 evidence head0 weights exactly zero."""
    torch.manual_seed(int(seed))
    a0 = V3A5DVerifier('A0_control')
    sa0 = snapshot_(a0)

    a1 = V3A5DVerifier('A1_evidence')
    a1.net.load_state_dict(a0.net.state_dict(), strict=True)
    # Rebuild head from the shared zeroed A0 heads so evidence cols are 0.
    a1.ev_head = EvidenceAwareHead(
        a0.net.head0, a0.net.head1, a0.net.head2, a0.net.act, N_EVIDENCE)
    if ev_mean is not None and ev_std is not None:
        a0.set_evidence_norm(ev_mean, ev_std)  # unused on A0 but lock buffers
        a1.set_evidence_norm(ev_mean, ev_std)
    if a1.evidence_weight_max_abs() != 0.0:
        raise SystemExit('A1 evidence weights not zero at init')
    return {
        'A0_control': snapshot_(a0),
        'A1_evidence': snapshot_(a1),
        'ev_mean': a1.ev_mean.detach().cpu().tolist(),
        'ev_std': a1.ev_std.detach().cpu().tolist(),
    }
