"""V3-A.6 Decision-Aligned Gate — losses, constant-q, verdict, lock helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from v3a5_runtime import STATES, expand_gate, masked_smooth_l1
from v3a5c_runtime import dump_json
from v3a5e1b_runtime import file_sha256

ARMS = ('A0_qstar', 'A1_decision_mse')
OBJECTIVES = {
    'A0_qstar': 'qstar_regression',
    'A1_decision_mse': 'decision_mse',
}
CKPT_STEPS = (0, 1000, 3000, 5000, 10000, 20000)
EVAL_STEPS = (3000, 10000, 20000)
CONSTANT_Q = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_UPDATES = 20000
GRAD_ACCUM = 4
LR = 1e-4
SEED = 42
BOTTLENECK = 64


def decision_mse_loss(q_g64, y0, H, D, geom, mask_g64):
    """Masked output MSE; q* must NOT appear here."""
    q_full = expand_gate(q_g64, geom)
    y_hat = y0 + q_full * D
    mask_full = expand_gate(mask_g64.float(), geom)
    err2 = (y_hat - H).pow(2).mean(dim=1, keepdim=True)
    loss = (err2 * mask_full).sum() / (mask_full.sum() + 1e-8)
    return loss, y_hat


def qstar_regression_loss(q_g64, q_star, mask_g64):
    """A0: MaskedSmoothL1(q, q*) only (gate-only; no output term)."""
    return masked_smooth_l1(q_g64, q_star, mask_g64, beta=1.0)


def arm_loss(arm, q_g64, y0, H, D, geom, mask_g64, q_star=None):
    if arm == 'A0_qstar':
        if q_star is None:
            raise SystemExit('A0 requires q_star')
        loss = qstar_regression_loss(q_g64, q_star, mask_g64)
        return loss, None
    if arm == 'A1_decision_mse':
        return decision_mse_loss(q_g64, y0, H, D, geom, mask_g64)
    raise SystemExit('unknown arm %r' % arm)


@torch.no_grad()
def constant_q_psnr(y0, H, D, q_scalar, metrics_fn):
    q = float(q_scalar)
    y = y0 + q * D
    return float(metrics_fn(y, H)[0])


def masked_fraction(mask_g64):
    m = mask_g64.float().reshape(-1)
    return float(m.mean()) if m.numel() else float('nan')


def git_head(repo='/root/projects/TTSR-lowlight'):
    try:
        return subprocess.check_output(
            ['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def sha_json(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest()


def hard_verify_lock(lock: Dict, expected: Dict, formal=True):
    """Compare selected keys; formal → SystemExit on mismatch."""
    mismatches = []
    for k, v in expected.items():
        if k not in lock:
            mismatches.append('%s missing' % k)
        elif lock[k] != v:
            mismatches.append('%s: lock=%r expected=%r' % (k, lock[k], v))
    if mismatches:
        msg = 'artifact_lock HARD FAIL:\n  ' + '\n  '.join(mismatches)
        if formal:
            raise SystemExit(msg)
        print('WARN ' + msg, flush=True)
    return mismatches


def require_ckpt(path, formal=True):
    if os.path.isfile(path):
        return path
    msg = 'requested checkpoint missing: %s' % path
    if formal:
        raise SystemExit(msg)
    raise FileNotFoundError(msg)


def nanmean(vals):
    a = np.asarray(list(vals), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float('nan')


def verdict_v3a6(dev_a0: Dict, dev_a1: Dict, base: Dict, r1: Dict,
                 const_best: Dict, gate_stats_a1: Dict) -> Dict:
    """Formal Case A–D from plan §15.

    Inputs are per-state PSNR dicts (and gate_stats_a1 with q_std / state diffs).
    """
    states = list(STATES)
    mean = lambda d: nanmean([d[s] for s in states if s in d])

    a0_m = mean(dev_a0)
    a1_m = mean(dev_a1)
    base_m = mean(base)
    r1_m = mean(r1)
    cq_m = mean(const_best)

    delta_a1_a0 = a1_m - a0_m
    correct_ok = bool(
        dev_a1.get('correct', -1e9) >= r1.get('correct', 1e9) - 0.02)
    dark = 'true_dark_g0.5'
    harm_ok = []
    for st in (dark, 'mismatch'):
        harm_ok.append(bool(dev_a1.get(st, -1e9) >= base.get(st, 1e9) - 1e-9))
    harm_soft = []
    for st in (dark, 'mismatch'):
        harm_soft.append(bool(dev_a1.get(st, -1e9) >= base.get(st, 1e9) - 0.02))
    harmful_pass = bool(
        (harm_ok[0] and harm_soft[1]) or (harm_ok[1] and harm_soft[0])
        or (harm_ok[0] and harm_ok[1]))

    # constant-q: mean +0.03 or ≥2/3 states better by >0
    beat_cq_mean = bool(a1_m >= cq_m + 0.03)
    beat_cq_states = sum(
        1 for s in states
        if s in dev_a1 and s in const_best and dev_a1[s] > const_best[s] + 1e-6)
    beat_cq = bool(beat_cq_mean or beat_cq_states >= 2)

    q_std = float(gate_stats_a1.get('q_std_mean', float('nan')))
    q_collapsed = bool(math.isfinite(q_std) and q_std < 0.02)
    state_diff = float(gate_stats_a1.get('state_qmean_std', float('nan')))
    weak_cond = bool(math.isfinite(state_diff) and state_diff < 0.01)

    beats_a0 = bool(delta_a1_a0 >= 0.05)
    weak_beats_a0 = bool(delta_a1_a0 >= 0.02)

    case_a = bool(
        beats_a0 and correct_ok and harmful_pass and beat_cq and not q_collapsed)
    case_c = bool(
        (weak_beats_a0 or beats_a0)
        and (q_collapsed or (not beat_cq and abs(a1_m - cq_m) < 0.03) or weak_cond)
        and not case_a)
    case_b = bool(weak_beats_a0 and not case_a and not case_c)
    case_d = bool(not weak_beats_a0)

    if case_a:
        label, next_step = 'V3A6_CASE_A_STRONG_GO', 'utility_or_selective'
        meaning = 'decision-aligned objective works'
    elif case_c:
        label, next_step = 'V3A6_CASE_C_CONSTANT_SHRINKAGE', 'advantage_accept_reject'
        meaning = 'output-MSE learned attenuation prior, not conditional gate'
    elif case_b:
        label, next_step = 'V3A6_CASE_B_WEAK_GO', 'utility_selective_formulation'
        meaning = 'objective alignment signal; conditional decision still weak'
    else:
        label, next_step = 'V3A6_CASE_D_FAIL', 'gt_utility_selective_risk'
        meaning = 'naive decision-MSE-only insufficient; do NOT return to q* regression'

    return dict(
        label=label, next_step=next_step, meaning=meaning,
        case_a=case_a, case_b=case_b, case_c=case_c, case_d=case_d,
        delta_a1_a0=delta_a1_a0,
        mean_psnr=dict(A0=a0_m, A1=a1_m, Base=base_m, R1=r1_m, const_best=cq_m),
        correct_ok=correct_ok, harmful_pass=harmful_pass,
        beat_constant_q=beat_cq, beat_cq_states=int(beat_cq_states),
        q_collapsed=q_collapsed, weak_conditionality=weak_cond,
        q_std_mean=q_std, state_qmean_std=state_diff,
    )


__all__ = [
    'ARMS', 'BOTTLENECK', 'CKPT_STEPS', 'CONSTANT_Q', 'DEFAULT_UPDATES',
    'EVAL_STEPS', 'GRAD_ACCUM', 'LR', 'OBJECTIVES', 'SEED',
    'arm_loss', 'constant_q_psnr', 'decision_mse_loss', 'dump_json',
    'file_sha256', 'git_head', 'hard_verify_lock', 'masked_fraction',
    'nanmean', 'qstar_regression_loss', 'require_ckpt', 'sha_json',
    'verdict_v3a6',
]
