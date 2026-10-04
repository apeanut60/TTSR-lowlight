"""V3-B.2 matcher evidence pack + A0→A1 shared initialization.

confidence_entropy := matcher entropy already normalized by log(K).
No polarity flip; train z-score + clip applied by the caller.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model.LocalRefine import LocalSoftMatch, Refiner
from model.ReferenceEvidence import LocalSoftMatchWithEvidence
from model.V3BResidualFusion import V3B0ResidualFusion

EVIDENCE_NAMES = ('sim_max', 'confidence_entropy', 'margin', 'disp_var')
# matcher key for each B2 channel name
_MATCH_KEY = {
    'sim_max': 'sim_max',
    'confidence_entropy': 'entropy',  # already / log(K)
    'margin': 'margin',
    'disp_var': 'disp_var',
}
EPS = 1e-6
CLIP = 5.0


def attach_match_evidence(proposal) -> LocalSoftMatchWithEvidence:
    """Weight-copy LocalSoftMatch → LocalSoftMatchWithEvidence; frozen."""
    if not isinstance(proposal, Refiner):
        # wrapper may own .proposal
        if hasattr(proposal, 'proposal') and isinstance(proposal.proposal, Refiner):
            proposal = proposal.proposal
        else:
            raise TypeError('expected Refiner, got %r' % type(proposal))
    m = proposal.match
    if not isinstance(m, LocalSoftMatch):
        raise TypeError('proposal.match must be LocalSoftMatch, got %r' % type(m))
    match_ev = LocalSoftMatchWithEvidence(
        ch=m.q.out_channels, radius=m.radius, tau=m.tau,
        eps=m.eps, chunk=m.chunk)
    match_ev.load_state_dict(m.state_dict(), strict=True)
    for p in match_ev.parameters():
        p.requires_grad_(False)
    match_ev.eval()
    return match_ev


@torch.no_grad()
def extract_match_maps(encoder, match_ev, y0, reference):
    """Frozen encoder + evidence matcher. y0/R in [-1,1].

    Returns F0, T, E_raw[B,4,H/2,W/2] in EVIDENCE_NAMES order.
    """
    f0 = encoder((y0 + 1.0) * 0.5)
    fr = encoder((reference + 1.0) * 0.5)
    t, ev = match_ev(f0, fr, return_evidence=True)
    chans = []
    for name in EVIDENCE_NAMES:
        key = _MATCH_KEY[name]
        if key not in ev:
            raise SystemExit('missing evidence key %r' % key)
        c = ev[key]
        if c.shape[-2:] != f0.shape[-2:]:
            raise SystemExit('evidence %s shape %s != F0 %s'
                             % (name, tuple(c.shape), tuple(f0.shape)))
        chans.append(c)
    e_raw = torch.cat(chans, dim=1)
    if e_raw.shape[1] != len(EVIDENCE_NAMES):
        raise SystemExit('bad E channels')
    if e_raw.shape[-2:] != f0.shape[-2:] or f0.shape != t.shape:
        raise SystemExit('spatial misalignment F0/T/E')
    return f0, t, e_raw


def normalize_evidence(e_raw: torch.Tensor, stats: Dict,
                       names: Sequence[str] = EVIDENCE_NAMES) -> torch.Tensor:
    """e' = clip((e - mu_train)/(std_train+eps), -5, 5). stats is per-channel."""
    if e_raw.shape[1] != len(names):
        raise SystemExit('E channels %d != names %d'
                         % (int(e_raw.shape[1]), len(names)))
    outs = []
    for i, name in enumerate(names):
        st = stats[name]
        mu = float(st['mean'])
        std = float(st['std'])
        x = (e_raw[:, i:i + 1] - mu) / (std + EPS)
        outs.append(x.clamp(-CLIP, CLIP))
    return torch.cat(outs, dim=1)


def copy_a0_to_a1(a0: V3B0ResidualFusion, a1: V3B0ResidualFusion) -> None:
    """Copy A0 weights into A1; zero the 4 new input-channel slices."""
    if a0.in_ch != 96 or a1.in_ch != 100:
        raise SystemExit('expected A0 in_ch=96 A1 in_ch=100, got %d/%d'
                         % (a0.in_ch, a1.in_ch))
    with torch.no_grad():
        w0 = a0.stem[0].weight  # [64,96,3,3]
        w1 = a1.stem[0].weight  # [64,100,3,3]
        w1.zero_()
        w1[:, :96].copy_(w0)
        a1.stem[0].bias.copy_(a0.stem[0].bias)
        # remaining layers: exact copy by name
        sd0 = a0.state_dict()
        sd1 = a1.state_dict()
        for k, v in sd0.items():
            if k == 'stem.0.weight':
                continue
            if k not in sd1:
                raise SystemExit('A1 missing key %s' % k)
            if sd1[k].shape != v.shape:
                raise SystemExit('shape mismatch %s %s vs %s'
                                 % (k, tuple(sd1[k].shape), tuple(v.shape)))
            sd1[k].copy_(v)
        a1.load_state_dict(sd1, strict=True)


@torch.no_grad()
def assert_shared_init(a0: V3B0ResidualFusion, a1: V3B0ResidualFusion,
                       f0, t, e, out_hw, tol=1e-7) -> float:
    """Even with nonzero E, A0/A1 step0 outputs match within tol."""
    d0 = a0(f0, t, out_hw, E=None)
    d1 = a1(f0, t, out_hw, E=e)
    diff = float((d0 - d1).abs().max())
    if diff > tol:
        raise SystemExit('A0/A1 step0 diverge: max_abs=%.3e' % diff)
    # evidence input slice must be exactly zero
    w = a1.stem[0].weight
    if float(w[:, 96:].abs().max()) > 0:
        raise SystemExit('A1 evidence weight slice not zero')
    return diff


def evidence_first_layer_norms(head: V3B0ResidualFusion) -> Dict[str, float]:
    """L1/L2 of A1 first-layer weights on the 4 evidence input channels."""
    if head.in_ch < 100:
        return dict(l1=0.0, l2=0.0)
    w = head.stem[0].weight[:, 96:].detach().float()
    return dict(
        l1=float(w.abs().sum()),
        l2=float(w.pow(2).sum().sqrt()),
        per_channel_l1=[float(w[:, i].abs().sum()) for i in range(4)],
    )


def common_weight_sha(a0: V3B0ResidualFusion) -> str:
    from v3a5_runtime import state_dict_sha
    return state_dict_sha(a0.state_dict())
