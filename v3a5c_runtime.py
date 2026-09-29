"""V3-A.5C helpers: overfit metrics, verdict, tiny subset selection."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch

from v3a5_runtime import STATES, gate_metrics

TINY_SEED = 20260928
TRAIN_SEED = 42
N_TINY = 16
N_MICRO = 4
CKPT_STEPS = (0, 500, 1000, 2000, 5000, 10000, 20000)
DEFAULT_UPDATES = 20000
GRAD_ACCUM = 4
OUT_WEIGHT_C0 = 0.1
OUT_WEIGHT_C1 = 0.0


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def choose_tiny_ids(train_ids: Sequence[str], n: int = N_TINY,
                    seed: int = TINY_SEED) -> List[str]:
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(train_ids), size=int(n), replace=False)
    return [train_ids[int(i)] for i in idx]


def slice_mismatch_map(full_map: Dict[str, str],
                       ids: Sequence[str]) -> Dict[str, str]:
    out = {}
    for k in ids:
        if k not in full_map:
            raise KeyError('id %r missing from mismatch map' % k)
        donor = full_map[k]
        if donor not in full_map and donor not in set(full_map.values()):
            # donor must still be a train575 id (values of the full map)
            pass
        out[k] = donor
    # donors must belong to the original train575 key set
    train_keys = set(full_map)
    bad = [d for d in out.values() if d not in train_keys]
    if bad:
        raise SystemExit('mismatch donors not in train575: %s' % bad[:5])
    return out


def polarized_decision_accuracy(q_v: torch.Tensor, q_star: torch.Tensor,
                                lo: float = 0.05, hi: float = 0.95) -> Dict:
    """Plan §9: only on polarized target blocks; predict by q_v </> 0.5."""
    a = q_v.detach().float().reshape(-1)
    b = q_star.detach().float().reshape(-1)
    polar = (b <= lo) | (b >= hi)
    n = int(polar.sum().item())
    if n == 0:
        return dict(decision_accuracy=float('nan'), n_polar=0,
                    n_correct=0)
    pred_accept = a >= 0.5
    tgt_accept = b >= hi
    # reject targets are b<=lo; accept targets b>=hi; middle excluded
    ok = ((b <= lo) & (~pred_accept)) | ((b >= hi) & pred_accept)
    n_ok = int(ok[polar].sum().item())
    return dict(decision_accuracy=float(n_ok) / float(n),
                n_polar=n, n_correct=n_ok)


def within_image_spatial_corr(q_v: torch.Tensor,
                              q_star: torch.Tensor) -> float:
    a = q_v.detach().float().reshape(-1)
    b = q_star.detach().float().reshape(-1)
    if a.numel() < 3 or float(a.std()) == 0 or float(b.std()) == 0:
        return float('nan')
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def aggregate_spatial_corrs(corrs: Sequence[float]) -> Dict:
    arr = np.asarray([c for c in corrs if c == c], dtype=np.float64)
    if arr.size == 0:
        return dict(spatial_corr_mean=float('nan'),
                    spatial_corr_median=float('nan'),
                    spatial_corr_p25=float('nan'),
                    spatial_corr_p75=float('nan'), n=0)
    return dict(
        spatial_corr_mean=float(arr.mean()),
        spatial_corr_median=float(np.median(arr)),
        spatial_corr_p25=float(np.percentile(arr, 25)),
        spatial_corr_p75=float(np.percentile(arr, 75)),
        n=int(arr.size))


def pair_gate_bundle(q_v: torch.Tensor, q_star: torch.Tensor,
                     mask: torch.Tensor) -> Dict:
    m = gate_metrics(q_v, q_star, mask)
    d = polarized_decision_accuracy(q_v, q_star)
    sp = within_image_spatial_corr(q_v, q_star)
    m.update(d)
    m['spatial_corr'] = sp
    qv_std = m['q_v_std']
    qo_std = m['q_opt_std']
    m['std_ratio'] = (float(qv_std) / float(qo_std)
                      if qo_std and qo_std == qo_std and qo_std > 0
                      else float('nan'))
    m['frac_qv_le_05'] = float((q_v.detach().float().reshape(-1) <= 0.05)
                               .float().mean())
    m['frac_qv_ge_95'] = float((q_v.detach().float().reshape(-1) >= 0.95)
                               .float().mean())
    m['frac_qstar_eq0'] = float((q_star.detach().float().reshape(-1) <= 0.05)
                                .float().mean())
    m['frac_qstar_eq1'] = float((q_star.detach().float().reshape(-1) >= 0.95)
                                .float().mean())
    return m


def _nanmean(vals: Sequence[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float('nan')


def aggregate_pair_metrics(rows: Sequence[Dict]) -> Dict:
    """Mean over pairs; decision_accuracy weighted by n_polar."""
    keys = ['MAE', 'masked_MAE', 'RMSE', 'corr', 'masked_corr',
            'q_v_mean', 'q_v_std', 'q_opt_mean', 'q_opt_std', 'std_ratio',
            'frac_qv_le_05', 'frac_qv_ge_95', 'frac_qstar_eq0',
            'frac_qstar_eq1', 'mask_valid_frac', 'spatial_corr',
            'PSNR', 'Recovery64']
    out = {k: _nanmean([r.get(k, float('nan')) for r in rows]) for k in keys}
    n_polar = sum(int(r.get('n_polar', 0)) for r in rows)
    n_ok = sum(int(r.get('n_correct', 0)) for r in rows)
    out['decision_accuracy'] = (float(n_ok) / float(n_polar)
                                if n_polar else float('nan'))
    out['n_polar'] = n_polar
    out['n_pairs'] = len(rows)
    out.update(aggregate_spatial_corrs(
        [r.get('spatial_corr', float('nan')) for r in rows]))
    return out


def verdict_c0(overall: Dict, by_state: Dict[str, Dict]) -> Dict:
    """Plan §13–§15 + §42.1."""
    mae = overall.get('masked_MAE', float('nan'))
    corr = overall.get('masked_corr', float('nan'))
    acc = overall.get('decision_accuracy', float('nan'))
    std_r = overall.get('std_ratio', float('nan'))
    state_ok = all(
        (by_state[s].get('masked_corr', float('nan')) >= 0.65)
        for s in STATES
        if s in by_state and by_state[s].get('masked_corr', float('nan'))
        == by_state[s].get('masked_corr', float('nan')))

    strong = bool(
        mae <= 0.05 and corr >= 0.90 and acc >= 0.95
        and std_r == std_r and std_r >= 0.70 and state_ok)
    primary = bool(
        mae <= 0.10 and corr >= 0.80 and acc >= 0.90 and state_ok)
    hard_fail = bool(
        mae > 0.20 or corr < 0.50 or acc < 0.70
        or (std_r == std_r and std_r < 0.4))

    if strong:
        label = 'C0_strong_success'
        action = 'stop_no_c1_predictability'
    elif primary:
        label = 'C0_success'
        action = 'stop_no_c1_predictability'
    elif hard_fail:
        label = 'C0_fail'
        action = 'run_c1'
    else:
        label = 'C0_partial_fail'
        action = 'run_c1'

    return dict(
        label=label, action=action, partial=(label == 'C0_partial_fail'),
        strong=strong, primary=primary, hard_fail=hard_fail,
        state_corr_ok=state_ok,
        thresholds=dict(masked_MAE=mae, masked_corr=corr,
                        decision_accuracy=acc, std_ratio=std_r))


def verdict_c1(c0_overall: Dict, c1_overall: Dict) -> Dict:
    """Plan §18: all three hard gates required."""
    mae0 = c0_overall.get('masked_MAE', float('nan'))
    mae1 = c1_overall.get('masked_MAE', float('nan'))
    corr0 = c0_overall.get('masked_corr', float('nan'))
    corr1 = c1_overall.get('masked_corr', float('nan'))
    acc1 = c1_overall.get('decision_accuracy', float('nan'))
    d_mae = mae0 - mae1          # positive = improvement
    d_corr = corr1 - corr0
    ok1 = bool(d_mae >= 0.10)
    ok2 = bool(d_corr >= 0.30)
    ok3 = bool(acc1 >= 0.90)
    n_ok = int(ok1) + int(ok2) + int(ok3)
    if ok1 and ok2 and ok3:
        label = 'C1_success'
        action = 'objective_alignment'
    elif n_ok >= 1:
        label = 'C1_partial'
        action = 'architecture_rf_audit'
    else:
        label = 'C1_fail'
        action = 'architecture_rf_audit'
    return dict(
        label=label, action=action, partial=(label == 'C1_partial'),
        ok_mae=ok1, ok_corr=ok2, ok_acc=ok3,
        delta_masked_MAE=d_mae, delta_masked_corr=d_corr,
        decision_accuracy=acc1,
        std_ratio=c1_overall.get('std_ratio', float('nan')))


def make_pair_schedule(n_updates: int, grad_accum: int, n_pairs: int,
                       seed: int) -> np.ndarray:
    """Deterministic cycle of full-set permutations (equal frequency)."""
    rng = np.random.default_rng(int(seed))
    need = int(n_updates) * int(grad_accum)
    buf = []
    while len(buf) < need:
        buf.extend(rng.permutation(int(n_pairs)).tolist())
    return np.asarray(buf[:need], dtype=np.int64)


def early_overfit_success(overall: Dict) -> bool:
    return bool(
        overall.get('masked_MAE', 1) <= 0.05
        and overall.get('masked_corr', 0) >= 0.90
        and overall.get('decision_accuracy', 0) >= 0.95)


def dump_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
