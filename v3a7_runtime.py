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
