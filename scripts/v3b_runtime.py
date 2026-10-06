"""V3-B.0 implicit residual fusion — match features, diagnostics, verdict.

B0 fusion inputs are F0/T only. D / g_v2 / q* / U / H never enter the head.
"""

from __future__ import annotations

import inspect
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from v3a5_runtime import STATES
from v3a71_runtime import json_ready
from v3a72_runtime import safety_from_deltas

CKPT_STEPS = (0, 1000, 3000, 5000, 10000, 20000)
EVAL_STEPS = (0, 3000, 10000, 20000)
DEFAULT_UPDATES = 20000
GRAD_ACCUM = 4
LR = 1e-4
SEED = 42
ARM = 'B0_implicit_residual'
FORMAL_LOCK_KEYS = (
    'repo_commit', 'base_cache_metadata_sha256', 'proposal_sha256',
    'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
    'reference_variant', 'architecture', 'arm', 'n_params', 'init_sha',
    'optimizer', 'lr', 'updates', 'grad_accum', 'seed',
    'official_test_allowed',
)


def proposal_core(wrapper):
    return wrapper.proposal if hasattr(wrapper, 'proposal') else wrapper


@torch.no_grad()
def match_features(wrapper, y0, reference):
    """Frozen V2 encoder + LocalSoftMatch. y0/R in [-1,1]. No D/g_v2/c_out."""
    core = proposal_core(wrapper)
    f0 = core.encoder((y0 + 1.0) * 0.5)
    fr = core.encoder((reference + 1.0) * 0.5)
    t = core.match(f0, fr)
    return f0, t


def match_features_mode(wrapper, y0, reference, mode='normal'):
    """Diagnostic T modes. ``self`` uses the old protocol: R_eff = Y0, rematch."""
    mode = str(mode)
    if mode == 'normal':
        return match_features(wrapper, y0, reference)
    if mode == 'self':
        return match_features(wrapper, y0, y0)
    if mode == 'zero':
        f0, t = match_features(wrapper, y0, reference)
        return f0, torch.zeros_like(t)
    if mode in ('shuffled', 'mismatch'):
        if reference is None:
            raise ValueError('shuffled mode needs mismatch donor reference')
        return match_features(wrapper, y0, reference)
    raise ValueError('unknown T mode %r' % mode)


def b0_forward(head, F0, T, y0):
    """Y = Y0 + ΔY. Gradients only through head if F0/T/Y0 are detached."""
    delta = head(F0, T, y0.shape[-2:])
    return y0 + delta, delta


def b0_loss(y, h):
    return F.mse_loss(y, h)


def residual_magnitude(delta):
    a = delta.detach().abs().reshape(-1).float()
    if a.numel() == 0:
        return dict(mean=float('nan'), p90=float('nan'))
    return dict(mean=float(a.mean()), p90=float(torch.quantile(a, 0.90)))


def residual_freq_stats(delta, sigma=5.0):
    """Gaussian low-pass energy vs high-pass residual energy of ΔY."""
    d = delta.detach().float()
    # separable approx via F.conv2d with gaussian kernel
    k = max(3, int(round(sigma * 4)) | 1)
    x = torch.arange(k, device=d.device, dtype=d.dtype) - (k // 2)
    g1 = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    g1 = g1 / g1.sum()
    kern = (g1[:, None] * g1[None, :]).view(1, 1, k, k)
    kern = kern.expand(d.shape[1], 1, k, k)
    pad = k // 2
    low = F.conv2d(d, kern, padding=pad, groups=d.shape[1])
    high = d - low
    e_lo = float(low.pow(2).mean())
    e_hi = float(high.pow(2).mean())
    tot = e_lo + e_hi
    return dict(e_low=e_lo, e_high=e_hi,
                low_frac=float(e_lo / tot) if tot > 0 else float('nan'))


def assert_no_gt_in_forward(fn):
    names = set(inspect.signature(fn).parameters)
    banned = {'H', 'h', 'q_star', 'qstar', 'U', 'AO', 'q_action', 'D', 'g_v2'}
    hit = names & banned
    if hit:
        raise SystemExit('GT/action leakage in forward args: %s' % sorted(hit))


def verdict_v3b0(psnr_b0: Dict, psnr_base: Dict, psnr_r1: Dict,
                 psnr_a6: Dict, dep: Dict) -> Dict:
    """Case A–D from V3-B plan §15. Inputs are per-state PSNR dicts."""
    def mean(d):
        vals = [float(d[s]) for s in STATES if s in d]
        return float(np.mean(vals)) if vals else float('nan')

    b0, base, r1, a6 = mean(psnr_b0), mean(psnr_base), mean(psnr_r1), mean(psnr_a6)
    delta_a6 = b0 - a6
    correct_ok = bool(psnr_b0.get('correct', -1e9) >= psnr_r1.get('correct', 1e9) - 0.02)
    dark, mis = 'true_dark_g0.5', 'mismatch'
    harm_ok = bool(
        psnr_b0.get(dark, -1e9) >= psnr_base.get(dark, 1e9) - 0.02
        and psnr_b0.get(mis, -1e9) >= psnr_base.get(mis, 1e9) - 0.02)
    nrm = float(dep.get('normal', float('nan')))
    slf = float(dep.get('self', float('nan')))
    zro = float(dep.get('zero', float('nan')))
    ref_ok = bool(math.isfinite(nrm) and math.isfinite(slf) and math.isfinite(zro)
                  and nrm >= slf + 0.03 and nrm >= zro + 0.03)
    ignored = bool(math.isfinite(nrm) and math.isfinite(slf) and math.isfinite(zro)
                   and abs(nrm - slf) < 0.02 and abs(nrm - zro) < 0.02)
    beats = bool(delta_a6 >= 0.05)
    weak = bool(delta_a6 > 0.0 and not beats)

    if beats and correct_ok and harm_ok and ref_ok:
        label, nxt, meaning = (
            'V3B0_CASE_A_STRONG_GO', 'V3B1_masa_adapt',
            'implicit residual beats V3-A.6 with real reference dependence')
    elif weak and ref_ok:
        label, nxt, meaning = (
            'V3B0_CASE_B_WEAK_GO', 'V3B1_masa_adapt',
            'implicit residual has reference signal but < +0.05 vs A1')
    elif ignored:
        label, nxt, meaning = (
            'V3B0_CASE_C_REFERENCE_IGNORED', 'inspect_T_injection',
            'PSNR change is a Base residual corrector, not reference fusion')
    else:
        label, nxt, meaning = (
            'V3B0_CASE_D_FAIL', 'feature_level_injection',
            'simple RGB implicit residual does not beat V3-A.6 with ref use')
    return dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(B0=b0, Base=base, R1=r1, V3A6_A1=a6),
        delta_b0_a6=delta_a6, correct_ok=correct_ok, harm_ok=harm_ok,
        ref_ok=ref_ok, ignored=ignored,
        dep_normal_self=float(nrm - slf) if math.isfinite(nrm) and math.isfinite(slf)
        else float('nan'),
        dep_normal_zero=float(nrm - zro) if math.isfinite(nrm) and math.isfinite(zro)
        else float('nan'),
    )


B1_ARMS = ('B1a_identity_adapt', 'B1b_masa_adapt')
FORMAL_LOCK_KEYS_B1 = FORMAL_LOCK_KEYS + ('b0_init_sha', 'b1_arms')


class MapCache(object):
    def __init__(self, max_gb=8.0):
        self.store = {}
        self.hits = 0
        self.misses = 0
        self.bytes = 0
        self.max_bytes = int(float(max_gb) * (1 << 30))

    def get(self, key, produce, device):
        v = self.store.get(key)
        if v is None:
            self.misses += 1
            f0, t = produce()
            f0 = f0.detach().to('cpu')
            t = t.detach().to('cpu')
            nbytes = (f0.numel() + t.numel()) * f0.element_size()
            if self.bytes + nbytes <= self.max_bytes:
                self.store[key] = (f0.clone(), t.clone())
                self.bytes += nbytes
            return f0.to(device), t.to(device)
        self.hits += 1
        return v[0].to(device), v[1].to(device)


def adapt_aux_stats(aux):
    dg = aux['delta_gamma'].detach().float()
    db = aux['delta_beta'].detach().float()
    dt = aux['delta_t'].detach().float()
    return dict(
        dgamma_mean=float(dg.mean()), dgamma_std=float(dg.std()),
        dbeta_mean=float(db.mean()), dbeta_std=float(db.std()),
        dt_mean_abs=float(dt.abs().mean()),
        dt_rms=float(dt.pow(2).mean().sqrt()),
    )


def verdict_v3b1(psnr_b1: Dict, psnr_b0: Dict, safety_b1: Dict,
                 safety_b0: Dict) -> Dict:
    """B1 vs frozen B0 @20k. Plan §18."""
    def mean(d):
        vals = [float(d[s]) for s in STATES if s in d]
        return float(np.mean(vals)) if vals else float('nan')

    b1, b0 = mean(psnr_b1), mean(psnr_b0)
    delta = b1 - b0
    g_cor = float(psnr_b1.get('correct', float('nan'))
                  - psnr_b0.get('correct', float('nan')))
    g_mis = float(psnr_b1.get('mismatch', float('nan'))
                  - psnr_b0.get('mismatch', float('nan')))
    g_dark = float(psnr_b1.get('true_dark_g0.5', float('nan'))
                   - psnr_b0.get('true_dark_g0.5', float('nan')))
    two_ok = bool(g_cor > 0.0 and g_mis > 0.0)
    mean_ok = bool(delta >= 0.03)

    def lh(safety, st):
        return float((safety.get(st) or {}).get('large_harm_rate', float('nan')))

    safety_ok = True
    for st in STATES:
        a = lh(safety_b1, st)
        b = lh(safety_b0, st)
        if math.isfinite(a) and math.isfinite(b) and a > b + 0.02:
            safety_ok = False
    if mean_ok or two_ok:
        if safety_ok:
            label, nxt, meaning = (
                'V3B1_CASE_A_ADAPT_GO', 'V3B2_evidence_as_feature',
                'adaptation beats B0 on mean or correct+mismatch without worse tails')
        else:
            label, nxt, meaning = (
                'V3B1_CASE_C_UNSAFE', 'drop_adapt_keep_B0',
                'PSNR up but large-harm worse than B0')
    else:
        label, nxt, meaning = (
            'V3B1_CASE_B_NO_GAIN', 'drop_adapt_keep_B0',
            'adaptation does not beat B0; drop B1')
    return dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(B1=b1, B0=b0), delta_b1_b0=delta,
        gain_correct=g_cor, gain_dark=g_dark, gain_mismatch=g_mis,
        mean_ok=mean_ok, two_state_ok=two_ok, safety_ok=safety_ok,
    )
