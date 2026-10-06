"""V3-A.5D1 runtime: evidence channel lock, pooling helpers, D1 verdict."""

from __future__ import annotations

from typing import Dict

from model.V3A5DVerifier import EVIDENCE_NAMES, N_EVIDENCE
from v3a5c_runtime import CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM, TRAIN_SEED
from v3a5d_runtime import CHANNEL_SOURCE_SCALE, pool_spatial_map_to_geom

# Re-export schedule constants (gate-only, same as V3-A.5C C1).
OUT_WEIGHT = 0.0
TINY_ROOT_SRC = '/root/data/experiments/v3a5c_tiny_overfit'

# D0-selected narrow set (order locked).
assert EVIDENCE_NAMES == ('sim_max', 'f0_minus_t', 'gate_v2')


def stack_narrow_evidence_g64(evidence, geom):
    """From probe evidence dict -> [B,3,64,64] raw (pre-norm) tensor."""
    import torch
    raw = {
        'sim_max': evidence['match']['sim_max'],
        'f0_minus_t': evidence['proposal']['feature']['f0_minus_t'],
        'gate_v2': evidence['proposal']['full']['gate_v2'],
    }
    chunks = []
    for name in EVIDENCE_NAMES:
        scale = CHANNEL_SOURCE_SCALE[name]
        chunks.append(pool_spatial_map_to_geom(raw[name], scale, geom))
    return torch.cat(chunks, dim=1)


def verdict_d1(a0: Dict, a1: Dict) -> Dict:
    """Compare A1 vs A0 on tiny overfit primary metrics.

    Success (go D1-full):
        (Δcorr >= +0.15 OR ΔMAE <= -0.08) AND acc_A1 >= 0.90

    Strong:
        A1 mMAE<=0.10 AND mCorr>=0.80 AND acc>=0.90

    Weak / insufficient (stop D1, go D2):
        small bumps only (not success)

    Close evidence route:
        Δcorr < +0.08 AND ΔMAE > -0.04  (A1 ≈ A0)
    """
    mae0 = float(a0.get('masked_MAE', float('nan')))
    mae1 = float(a1.get('masked_MAE', float('nan')))
    corr0 = float(a0.get('masked_corr', float('nan')))
    corr1 = float(a1.get('masked_corr', float('nan')))
    acc1 = float(a1.get('decision_accuracy', float('nan')))

    d_corr = corr1 - corr0
    d_mae = mae1 - mae0  # negative = improvement

    strong = bool(
        mae1 <= 0.10 and corr1 >= 0.80 and acc1 >= 0.90)
    success = bool(
        ((d_corr >= 0.15) or (d_mae <= -0.08)) and acc1 >= 0.90)
    close_route = bool(d_corr < 0.08 and d_mae > -0.04)

    if strong:
        label = 'D1_strong_success'
        action = 'run_d1_full'
        next_step = 'D1-full'
    elif success:
        label = 'D1_success'
        action = 'run_d1_full'
        next_step = 'D1-full'
    elif close_route:
        label = 'D1_null_close_evidence'
        action = 'run_d2_rf'
        next_step = 'D2'
    else:
        label = 'D1_insufficient'
        action = 'run_d2_rf'
        next_step = 'D2'

    return dict(
        label=label,
        action=action,
        next_step=next_step,
        strong=strong,
        success=success,
        close_evidence_route=close_route,
        delta_masked_corr=d_corr,
        delta_masked_MAE=d_mae,
        A0=dict(masked_MAE=mae0, masked_corr=corr0,
                decision_accuracy=float(a0.get('decision_accuracy', float('nan')))),
        A1=dict(masked_MAE=mae1, masked_corr=corr1,
                decision_accuracy=acc1),
        thresholds=dict(
            success_d_corr=0.15, success_d_mae=-0.08, success_acc=0.90,
            strong_mae=0.10, strong_corr=0.80, strong_acc=0.90,
            close_d_corr=0.08, close_d_mae=-0.04),
    )


__all__ = [
    'CKPT_STEPS', 'DEFAULT_UPDATES', 'EVIDENCE_NAMES', 'GRAD_ACCUM',
    'N_EVIDENCE', 'OUT_WEIGHT', 'TINY_ROOT_SRC', 'TRAIN_SEED',
    'stack_narrow_evidence_g64', 'verdict_d1',
]
