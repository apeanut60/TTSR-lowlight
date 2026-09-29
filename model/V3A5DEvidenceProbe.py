"""Frozen R1 Proposal wrapper that exposes matcher/proposal evidence.

Does NOT modify LocalSoftMatch.forward or Proposal parameters. Evidence comes
from LocalSoftMatchWithEvidence with weights identical to the frozen match.
Hard invariant (checked optionally): T / g_v2 / delta / D / sr match the
original Proposal within 1e-6 max-abs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LocalRefine import LocalSoftMatch, Refiner
from model.ReferenceEvidence import LocalSoftMatchWithEvidence, proposal_evidence


class V3A5DEvidenceProbe(nn.Module):
    """Diagnostic probe around a frozen ``Refiner`` proposal.

    Shares encoder / gate / c_hidden / c_out modules with the source proposal
    (parameter identity). Only the match path is replaced by an evidence
    subclass carrying a *copy* of the match weights.
    """

    def __init__(self, proposal):
        super().__init__()
        if not isinstance(proposal, Refiner):
            raise TypeError('expected model.LocalRefine.Refiner, got %r'
                            % type(proposal))
        if not isinstance(proposal.match, LocalSoftMatch):
            raise TypeError('proposal.match must be LocalSoftMatch, got %r'
                            % type(proposal.match))
        self.proposal = proposal
        m = proposal.match
        self.match_ev = LocalSoftMatchWithEvidence(
            ch=m.q.out_channels, radius=m.radius, tau=m.tau,
            eps=m.eps, chunk=m.chunk)
        self.match_ev.load_state_dict(m.state_dict(), strict=True)
        for p in self.match_ev.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.proposal.eval()
        self.match_ev.eval()
        return self

    def forward(self, y0, r_eff, check_equiv=False, equiv_tol=1e-6):
        """-> (sr, aux, evidence).

        ``evidence`` keys:
            match: dict of H/2 maps (sim_max, pmax, ...)
            proposal: dict(full=..., feature=...) from proposal_evidence
        """
        enc = self.proposal.encoder
        f0 = enc((y0 + 1.) * 0.5)
        fr = enc((r_eff + 1.) * 0.5)
        t, match_ev = self.match_ev(f0, fr, return_evidence=True)
        u = torch.cat((f0, t, f0 - t), dim=1)
        g_small = self.proposal.gate(u)
        delta_small = self.proposal.c_hidden(u)
        delta = self.proposal.c_out(F.interpolate(
            delta_small, size=y0.shape[-2:], mode='bilinear',
            align_corners=False))
        g = F.interpolate(g_small, size=y0.shape[-2:], mode='bilinear',
                          align_corners=False)
        sr = y0 + g * delta
        D = g * delta
        aux = dict(gate=g, delta=delta, t=t, f0=f0, D=D)
        pe = proposal_evidence(g, delta, D, f0, t)
        evidence = dict(match=match_ev, proposal=pe)

        if check_equiv:
            report = self.check_equivalence(y0, r_eff, sr, aux, tol=equiv_tol)
            if not report['ok']:
                raise RuntimeError('evidence probe broke Proposal equivalence: %s'
                                   % report)
            aux['equiv'] = report
        return sr, aux, evidence

    @torch.no_grad()
    def check_equivalence(self, y0, r_eff, sr=None, aux=None, tol=1e-6):
        """Compare evidence path against the untouched proposal.match forward."""
        enc = self.proposal.encoder
        f0 = enc((y0 + 1.) * 0.5)
        fr = enc((r_eff + 1.) * 0.5)
        t_orig = self.proposal.match(f0, fr)
        t_ev, _ = self.match_ev(f0, fr, return_evidence=True)
        t_max = float((t_ev - t_orig).abs().max())
        t_mean = float((t_ev - t_orig).abs().mean())

        sr_ref, aux_ref = self.proposal(y0, r_eff)
        if sr is None or aux is None:
            sr, aux, _ = self.forward(y0, r_eff, check_equiv=False)
        D_ref = aux_ref['gate'] * aux_ref['delta']
        D = aux['gate'] * aux['delta']
        checks = {
            'T_max_abs': t_max,
            'T_mean_abs': t_mean,
            'gate_max_abs': float((aux['gate'] - aux_ref['gate']).abs().max()),
            'delta_max_abs': float((aux['delta'] - aux_ref['delta']).abs().max()),
            'D_max_abs': float((D - D_ref).abs().max()),
            'sr_max_abs': float((sr - sr_ref).abs().max()),
        }
        ok = (t_max <= tol and t_mean <= 1e-7
              and all(checks[k] <= tol for k in (
                  'gate_max_abs', 'delta_max_abs', 'D_max_abs', 'sr_max_abs')))
        return dict(ok=ok, tol=tol, **checks)
