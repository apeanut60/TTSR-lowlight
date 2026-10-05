"""V4.0 reverse-match Ref-as-Canvas. Frozen encoder/matcher; RGB residual head."""

from __future__ import annotations

import torch

from v3b_runtime import proposal_core

DIR_A0 = 'F0_query_FR_source'
DIR_A1 = 'FR_query_F0_source'
CANVAS_A0 = 'Y0'
CANVAS_A1 = 'R'


def encode_m11(encoder, x_m11):
    """Images in [-1,1] → SharedEncoder features at H/2."""
    return encoder((x_m11 + 1.0) * 0.5)


def match_qv(match, query, source):
    """LocalSoftMatch: Q from first arg, K/V from second."""
    if query.shape != source.shape:
        raise SystemExit('match shape %s vs %s' % (tuple(query.shape), tuple(source.shape)))
    return match(query, source)


def encode_pair(wrapper, y0, reference):
    core = proposal_core(wrapper)
    f0 = encode_m11(core.encoder, y0)
    fr = encode_m11(core.encoder, reference)
    return core, f0, fr


@torch.no_grad()
def features_a0(wrapper, y0, reference):
    """A0: Q=F0, K/V=FR → T_ref. Returns (F0, T_ref)."""
    core, f0, fr = encode_pair(wrapper, y0, reference)
    t = match_qv(core.match, f0, fr)
    return f0, t


@torch.no_grad()
def features_a1(wrapper, y0, reference):
    """A1: Q=FR, K/V=F0 → T_low. Returns (FR, T_low)."""
    core, f0, fr = encode_pair(wrapper, y0, reference)
    t = match_qv(core.match, fr, f0)
    return fr, t


def features_a1_mode(wrapper, y0, reference, mode='normal', y0_donor=None):
    """A1 reverse-guidance diagnostics. FR always from ``reference``."""
    mode = str(mode)
    core, f0, fr = encode_pair(wrapper, y0, reference)
    if mode == 'normal':
        return fr, match_qv(core.match, fr, f0)
    if mode == 'self':
        return fr, match_qv(core.match, fr, fr)
    if mode == 'zero':
        t = match_qv(core.match, fr, f0)
        return fr, torch.zeros_like(t)
    if mode in ('shuffled_target', 'shuffled'):
        if y0_donor is None:
            raise SystemExit('shuffled_target needs y0_donor')
        f0d = encode_m11(core.encoder, y0_donor)
        if f0d.shape != fr.shape:
            raise SystemExit('F0_donor %s != FR %s' % (tuple(f0d.shape), tuple(fr.shape)))
        return fr, match_qv(core.match, fr, f0d)
    raise SystemExit('unknown A1 mode %r' % mode)
