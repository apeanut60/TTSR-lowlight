"""V3-A.5D2.1: exposure / scale audit on a fixed train64 subset."""

from __future__ import annotations

from typing import Dict

from v3a5c_runtime import CKPT_STEPS, GRAD_ACCUM, TRAIN_SEED, choose_tiny_ids
from v3a5d1_runtime import OUT_WEIGHT

N_SCALE = 64
SCALE_SEED = 20260930
# Checkpoints for the exposure curve (single 20k run).
EXPOSURE_STEPS = (5000, 10000, 20000)
DEFAULT_UPDATES = 20000
# ~417 micro-batches per pair at 20k × accum4 / 192
PAIRS_PER_IMAGE = 3  # correct / dark / mismatch


def classify_d21(train64: Dict, dev64: Dict) -> Dict:
    """Case A–D from user brief (primary: A1 @20k).

    A: train64 near-tiny fit (mCorr>0.9, mMAE<0.1) — structure OK, budget issue
    B: train64 fits clearly, dev64 poor — memorization / weak cross-image
    C: train64 cannot fit (mCorr still <0.4) — target complexity explodes
    D: train64 and dev both clearly improve — MultiScale route real

    Soft Case B: train64 mCorr≥0.80 & mMAE≤0.12 with poor dev (strong fit
    short of the 0.90 line still counts as “can memorize the subset”).
    """
    tr_c = float(train64.get('masked_corr', float('nan')))
    tr_m = float(train64.get('masked_MAE', float('nan')))
    dv_c = float(dev64.get('masked_corr', float('nan')))
    dv_m = float(dev64.get('masked_MAE', float('nan')))

    train_fit = bool(tr_c >= 0.90 and tr_m <= 0.10)
    train_clear = bool(tr_c >= 0.80 and tr_m <= 0.12)
    train_fail = bool(tr_c < 0.40)
    dev_good = bool(dv_c >= 0.40 and dv_m <= 0.30)
    dev_poor = bool(dv_c < 0.30)

    if train_fit and dev_good:
        case, label = 'D', 'D21_case_D_scale_and_transfer'
        next_step = 'plan_full575_schedule'
    elif (train_fit or train_clear) and dev_poor:
        case, label = 'B', 'D21_case_B_memorize_not_transfer'
        next_step = 'target_predictability_audit'
    elif train_fail:
        case, label = 'C', 'D21_case_C_complexity_explosion'
        next_step = 'target_redesign'
    elif train_fit and not dev_good:
        case, label = 'A', 'D21_case_A_structure_ok_budget'
        next_step = 'scale_curve_128_256_575'
    else:
        case, label = 'PARTIAL', 'D21_partial'
        next_step = 'REVIEW'

    return dict(
        case=case, label=label, next_step=next_step,
        train_fit_strict=train_fit, train_fit_clear=train_clear,
        train64=dict(masked_corr=tr_c, masked_MAE=tr_m,
                     decision_accuracy=float(train64.get('decision_accuracy', float('nan')))),
        dev64=dict(masked_corr=dv_c, masked_MAE=dv_m,
                   decision_accuracy=float(dev64.get('decision_accuracy', float('nan')))),
        thresholds=dict(train_fit_corr=0.90, train_fit_mae=0.10,
                        train_clear_corr=0.80, train_clear_mae=0.12,
                        train_fail_corr=0.40, dev_good_corr=0.40,
                        dev_poor_corr=0.30),
    )


__all__ = [
    'DEFAULT_UPDATES', 'EXPOSURE_STEPS', 'GRAD_ACCUM', 'N_SCALE', 'OUT_WEIGHT',
    'PAIRS_PER_IMAGE', 'SCALE_SEED', 'TRAIN_SEED', 'choose_tiny_ids',
    'classify_d21',
]
