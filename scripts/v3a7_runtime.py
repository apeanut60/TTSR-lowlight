"""V3-A.7 GT block-utility accept/reject — losses and verdict wrappers.

Question after V3-A.6 Case D: naive output-MSE cannot teach a transferable
gate. Instead supervise a binary accept/reject of the frozen proposal using
privileged GT utility:

    U_B = mean_B[ ||Y0-H||^2 - ||Y0+D-H||^2 ]
    t_B = 1[U_B > 0]

A0 control is V3-A.6 A1 (decision_mse) at the same update count — not retrained.
Sole new variable: loss = masked BCE(q, t). q* is diagnostic only.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from v3a5_runtime import block_mean
from v3a6_runtime import (CKPT_STEPS, CONSTANT_Q, DEFAULT_UPDATES,  # noqa: F401
                          EVAL_STEPS, GRAD_ACCUM, LR, SEED, dump_json,
                          file_sha256, git_head, hard_verify_lock,
                          masked_fraction, nanmean, require_ckpt,
                          sha_json, verdict_v3a6)

ARMS = ('A1_utility_bce',)
OBJECTIVES = {'A1_utility_bce': 'utility_accept_bce'}
A0_CONTROL = 'A0_decision_mse'
BOTTLENECK = 64
FORMAL_LOCK_KEYS = (
    'repo_commit', 'proposal_sha256', 'cache_metadata_sha256', 'split_sha256',
    'mismatch_train_sha256', 'mismatch_dev_sha256', 'energy_stats_sha256',
    'reference_variant', 'geometry', 'architecture', 'bottleneck', 'init_sha',
    'updates', 'grad_accum', 'lr', 'seed', 'pair_schedule_seed', 'states',
    'mask_mode', 'official_test_allowed',
)


def block_utility(y0, H, D, geom):
    """Per-block proposal advantage vs Base (RGB-mean squared error).

    Positive U means Y0+D is closer to H than Y0. Detach callers' tensors
    before this if they must not receive grad through H/D/Y0.
    """
    e0 = (y0 - H).pow(2).mean(dim=1, keepdim=True)
    e1 = (y0 + D - H).pow(2).mean(dim=1, keepdim=True)
    nby, nbx = geom['shape']
    return block_mean(e0 - e1, geom).reshape(-1, 1, nby, nbx)


def accept_target(U):
    return (U > 0).to(U.dtype)


def utility_bce_loss(q_g64, y0, H, D, geom, mask_g64):
    """Masked BCE(q, 1[U>0]). q* must not appear. H only as target."""
    U = block_utility(y0.detach(), H, D.detach(), geom)
    t = accept_target(U).detach()
    m = mask_g64.to(q_g64.dtype)
    bce = F.binary_cross_entropy(q_g64, t, reduction='none')
    loss = (m * bce).sum() / m.sum().clamp(min=1.0)
    pos = ((t * m).sum() / m.sum().clamp(min=1.0)).detach()
    return loss, dict(U=U.detach(), t=t, pos_frac=float(pos))


def arm_loss(arm, q_g64, y0, H, D, geom, mask_g64, q_star=None):
    if arm == 'A1_utility_bce':
        return utility_bce_loss(q_g64, y0, H, D, geom, mask_g64)
    raise SystemExit('unknown V3-A.7 arm %r' % arm)


def q_raw_unclipped(y0, H, D, geom, eps=1e-8):
    """N/Z at G64 (no clip). U_B>0 iff this > 0.5 (same RGB-sum reduction)."""
    N = ((H - y0) * D).sum(dim=1, keepdim=True)
    Z = (D ** 2).sum(dim=1, keepdim=True)
    nby, nbx = geom['shape']
    n = block_mean(N, geom).reshape(-1, 1, nby, nbx)
    z = block_mean(Z, geom).reshape(-1, 1, nby, nbx)
    return n / (z + eps)


def deployable_global_constant(const_by_state, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """One q shared by all states. const_by_state[state][str(q)] = PSNR."""
    from v3a5_runtime import STATES
    scores = {}
    for q in qs:
        key = str(q)
        scores[q] = nanmean([const_by_state[s][key] for s in STATES
                             if s in const_by_state and key in const_by_state[s]])
    best = max(scores, key=lambda k: scores[k] if scores[k] == scores[k] else -1e9)
    return dict(best_q=float(best), mean_psnr=float(scores[best]),
                per_q={str(k): float(v) for k, v in scores.items()})


def verdict_v3a7(dev_a0, dev_a1, base, r1, const_best_per_state,
                 gate_stats_a1, global_const=None):
    """V3-A.7 labels. A0=decision-MSE, A1=utility-BCE.

    Fair constant baseline is deployable *global* q (one value, all states).
    per-state const_best is diagnostic only.
    """
    import math
    from v3a5_runtime import STATES
    states = list(STATES)
    mean = lambda d: nanmean([d[s] for s in states if s in d])
    a0_m, a1_m = mean(dev_a0), mean(dev_a1)
    base_m, r1_m = mean(base), mean(r1)
    cq_state_m = mean(const_best_per_state)
    gq = global_const or {}
    cq_glob = float(gq.get('mean_psnr', float('nan')))
    delta = a1_m - a0_m
    correct_ok = bool(dev_a1.get('correct', -1e9) >= r1.get('correct', 1e9) - 0.02)
    dark, mis = 'true_dark_g0.5', 'mismatch'
    harmful_pass = bool(
        (dev_a1.get(dark, -1e9) >= base.get(dark, 1e9) - 0.02)
        and (dev_a1.get(mis, -1e9) >= base.get(mis, 1e9) - 0.02)
        and ((dev_a1.get(dark, -1e9) >= base.get(dark, 1e9) - 1e-9)
             or (dev_a1.get(mis, -1e9) >= base.get(mis, 1e9) - 1e-9)))
    beat_gq = bool(math.isfinite(cq_glob) and a1_m >= cq_glob + 0.03)
    q_std = float(gate_stats_a1.get('q_std_mean', float('nan')))
    q_collapsed = bool(math.isfinite(q_std) and q_std < 0.02)
    beats = bool(delta >= 0.05)
    weak = bool(delta >= 0.02)
    if beats and correct_ok and harmful_pass and beat_gq and not q_collapsed:
        label, nxt, meaning = (
            'V3A7_CASE_A_STRONG_GO', 'global_or_regional_utility',
            'utility-BCE beats decision-MSE with deployable constant bar')
    elif q_collapsed and (weak or beats):
        label, nxt, meaning = (
            'V3A7_CASE_C_CONSTANT_SHRINKAGE', 'advantage_l_neg_qU',
            'utility-BCE collapsed toward a constant gate')
    elif weak and not beats:
        label, nxt, meaning = (
            'V3A7_CASE_B_WEAK_GO', 'utility_predictability_audit',
            'weak PSNR gain vs decision-MSE; not a GO')
    else:
        label, nxt, meaning = (
            'V3A7_CASE_D_NO_GAIN', 'utility_predictability_audit',
            'block BCE accept/reject does not beat decision-MSE')
    return dict(
        label=label, next_step=nxt, meaning=meaning,
        delta_a1_a0=delta, correct_ok=correct_ok, harmful_pass=harmful_pass,
        beat_global_constant=beat_gq, q_collapsed=q_collapsed,
        mean_psnr=dict(A0=a0_m, A1=a1_m, Base=base_m, R1=r1_m,
                       const_per_state=cq_state_m, const_global=cq_glob),
        global_constant=gq,
    )
