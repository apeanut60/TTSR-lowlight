"""V3-A.5D2 runtime: RF/context tiny-overfit verdict (same bars as D1)."""

from __future__ import annotations

from typing import Dict

from v3a5c_runtime import CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM, TRAIN_SEED
from v3a5d1_runtime import OUT_WEIGHT, verdict_d1

TINY_ROOT_SRC = '/root/data/experiments/v3a5c_tiny_overfit'


def verdict_d2(a0: Dict, a1: Dict) -> Dict:
    """Reuse D1 numeric bars; remap labels to D2."""
    v = verdict_d1(a0, a1)
    label_map = {
        'D1_strong_success': 'D2_strong_success',
        'D1_success': 'D2_success',
        'D1_null_close_evidence': 'D2_null_close_rf',
        'D1_insufficient': 'D2_insufficient',
    }
    action_map = {
        'run_d1_full': 'run_d2_full',
        'run_d2_rf': 'stop_architecture_dead_end',
    }
    v['label'] = label_map.get(v['label'], v['label'].replace('D1', 'D2'))
    v['action'] = action_map.get(v['action'], v['action'])
    if v['success'] or v['strong']:
        v['next_step'] = 'D2-full'
    elif v.get('close_evidence_route'):
        # reused flag name from verdict_d1; means A1≈A0
        v['close_rf_route'] = True
        v['next_step'] = 'STOP_or_D2b_review'
    else:
        v['close_rf_route'] = False
        v['next_step'] = 'STOP_or_D2b_review'
    v['close_rf_route'] = bool(v.get('close_evidence_route'))
    return v


__all__ = [
    'CKPT_STEPS', 'DEFAULT_UPDATES', 'GRAD_ACCUM', 'OUT_WEIGHT',
    'TINY_ROOT_SRC', 'TRAIN_SEED', 'verdict_d2',
]
