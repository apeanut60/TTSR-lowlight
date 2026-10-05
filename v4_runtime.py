"""V4.0 Ref-as-Canvas — lock, verdict, ckpt hygiene."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model.V4RefCanvas import CANVAS_A0, CANVAS_A1, DIR_A0, DIR_A1
from v3a5_runtime import STATES
from v3b_runtime import CKPT_STEPS, DEFAULT_UPDATES, EVAL_STEPS, GRAD_ACCUM, LR, SEED
from v3b2_runtime import pair_delta_stats_by_state
from v3b3_runtime import safety_plus

ARMS = ('A0_b0_replay', 'A1_ref_canvas')
ARM_A0 = 'A0_b0_replay'
ARM_A1 = 'A1_ref_canvas'

FORMAL_LOCK_KEYS_V4 = (
    'repo_commit', 'base_cache_metadata_sha256', 'proposal_sha256',
    'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
    'reference_variant',
    'direction_A0', 'direction_A1', 'canvas_A0', 'canvas_A1',
    'architecture', 'arms', 'shared_head_init_sha',
    'a0_init_sha', 'a1_init_sha',
    'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
)


def require_ckpt_blob_v4(blob, *, arm, step, init_sha, proposal_sha,
                         repo_commit, direction, canvas, shared_init_sha,
                         formal=True):
    errs = []
    if int(blob.get('step', -1)) != int(step):
        errs.append('step: blob=%r want=%r' % (blob.get('step'), step))
    if blob.get('arm') != arm:
        errs.append('arm: blob=%r want=%r' % (blob.get('arm'), arm))
    if blob.get('objective') != 'mse_reconstruction':
        errs.append('objective: %r' % blob.get('objective'))
    if int(blob.get('seed', -1)) != int(SEED):
        errs.append('seed: %r' % blob.get('seed'))
    if blob.get('init_sha') != init_sha:
        errs.append('init_sha mismatch')
    if blob.get('shared_head_init_sha') != shared_init_sha:
        errs.append('shared_head_init_sha mismatch')
    if blob.get('proposal_sha') != proposal_sha:
        errs.append('proposal_sha mismatch')
    if blob.get('repo_commit') != repo_commit:
        errs.append('ckpt repo_commit mismatch')
    if blob.get('direction') != direction:
        errs.append('direction: blob=%r want=%r' % (blob.get('direction'), direction))
    if blob.get('canvas') != canvas:
        errs.append('canvas: blob=%r want=%r' % (blob.get('canvas'), canvas))
    if errs:
        msg = 'ckpt HARD FAIL:\n  ' + '\n  '.join(errs)
        if formal:
            raise SystemExit(msg)
        raise ValueError(msg)
    return True


def arm_meta(arm):
    if arm == ARM_A0:
        return DIR_A0, CANVAS_A0
    if arm == ARM_A1:
        return DIR_A1, CANVAS_A1
    raise SystemExit('unknown arm %r' % arm)


def gaussian_lowpass(x, sigma=5.0):
    d = x.float()
    k = max(3, int(round(sigma * 4)) | 1)
    t = torch.arange(k, device=d.device, dtype=d.dtype) - (k // 2)
    g1 = torch.exp(-0.5 * (t / float(sigma)) ** 2)
    g1 = g1 / g1.sum()
    kern = (g1[:, None] * g1[None, :]).view(1, 1, k, k)
    kern = kern.expand(d.shape[1], 1, k, k)
    return F.conv2d(d, kern, padding=k // 2, groups=d.shape[1])


def spatial_grad_l1(a, b):
    da_y = a[..., 1:, :] - a[..., :-1, :]
    db_y = b[..., 1:, :] - b[..., :-1, :]
    da_x = a[..., :, 1:] - a[..., :, :-1]
    db_x = b[..., :, 1:] - b[..., :, :-1]
    return float((da_y - db_y).abs().mean() + (da_x - db_x).abs().mean())


def laplacian_l1(a, b):
    ker = a.new_tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]])
    ker = ker.expand(a.shape[1], 1, 3, 3)
    la = F.conv2d(a.float(), ker, padding=1, groups=a.shape[1])
    lb = F.conv2d(b.float(), ker, padding=1, groups=b.shape[1])
    return float((la - lb).abs().mean())


def hf_energy(x, sigma=5.0):
    low = gaussian_lowpass(x, sigma)
    return float((x.float() - low).pow(2).mean())


def canvas_retention(y, r, y0):
    dy_r = float((y - r).abs().mean())
    dy_y0 = float((y - y0).abs().mean())
    r_y0 = float((r - y0).abs().mean())
    return dict(
        dY_R=dy_r, dY_Y0=dy_y0, dR_Y0=r_y0,
        rho_R=float(dy_r / (r_y0 + 1e-8)),
    )


def summarize_list(vals):
    a = np.asarray(list(vals), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(mean=float('nan'), median=float('nan'),
                    p10=float('nan'), p90=float('nan'), n=0)
    return dict(
        mean=float(a.mean()), median=float(np.median(a)),
        p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
        n=int(a.size),
    )


def verdict_v4(psnr_a0: Dict, psnr_a1: Dict, psnr_raw: Dict, psnr_base: Dict,
               safety_a0: Dict, safety_a1: Dict, dep_a1: Dict,
               pair_stats: Optional[Dict] = None) -> Dict:
    def mean(d):
        vals = [float(d[s]) for s in STATES if s in d]
        return float(np.mean(vals)) if vals else float('nan')

    a0, a1 = mean(psnr_a0), mean(psnr_a1)
    raw, base = mean(psnr_raw), mean(psnr_base)
    delta = a1 - a0
    g_cor = float(psnr_a1.get('correct', float('nan'))
                  - psnr_a0.get('correct', float('nan')))
    g_dark = float(psnr_a1.get('true_dark_g0.5', float('nan'))
                   - psnr_a0.get('true_dark_g0.5', float('nan')))
    g_mis = float(psnr_a1.get('mismatch', float('nan'))
                  - psnr_a0.get('mismatch', float('nan')))
    vs_raw = a1 - raw
    vs_raw_cor = float(psnr_a1.get('correct', float('nan'))
                       - psnr_raw.get('correct', float('nan')))

    def lh(safety, st):
        return float((safety.get(st) or {}).get('large_harm_rate', float('nan')))

    lh_a0 = lh(safety_a0, 'mismatch')
    lh_a1 = lh(safety_a1, 'mismatch')
    tail_ok = bool(math.isfinite(lh_a0) and math.isfinite(lh_a1)
                   and lh_a1 <= lh_a0 + 0.02)
    tail_unsafe = bool(math.isfinite(lh_a0) and math.isfinite(lh_a1)
                       and lh_a1 > lh_a0 + 0.02)

    nrm = float(dep_a1.get('normal', float('nan')))
    slf = float(dep_a1.get('self', float('nan')))
    zro = float(dep_a1.get('zero', float('nan')))
    shf = float(dep_a1.get('shuffled_target', dep_a1.get('shuffled', float('nan'))))
    ref_self = bool(math.isfinite(nrm) and math.isfinite(slf) and nrm >= slf + 0.03)
    ref_shuf = bool(math.isfinite(nrm) and math.isfinite(shf) and nrm >= shf + 0.03)
    ref_ok = bool(ref_self and ref_shuf)
    ignored = bool(math.isfinite(nrm) and math.isfinite(slf) and abs(nrm - slf) < 0.02)
    vs_raw_ok = bool(math.isfinite(vs_raw) and (vs_raw >= 0.05 or vs_raw_cor >= 0.05))
    raw_no_adv = bool(math.isfinite(raw) and math.isfinite(base) and raw <= base + 0.02)

    case_a = bool(
        delta >= 0.05 and g_cor >= 0.05 and g_dark >= -0.02 and g_mis >= -0.02
        and tail_ok and ref_ok)
    case_b_gain = bool(g_cor >= 0.10 or (g_cor >= 0.05 and g_dark >= 0.05))
    mismatch_unsafe = bool(g_mis < -0.02 or tail_unsafe)
    case_b = bool(case_b_gain and vs_raw_ok and ref_ok and mismatch_unsafe)
    case_c = bool(abs(delta) < 0.02 and abs(vs_raw) < 0.05 and ignored)
    case_d = bool(g_cor < -0.05 and raw_no_adv)

    if case_a:
        label, nxt, meaning = (
            'V4_CASE_A_STRONG_GO', 'V4.1_source_structural_grounding',
            'Ref-as-canvas beats B0 replay with reverse-guidance dependence')
    elif case_b:
        label, nxt, meaning = (
            'V4_CASE_B_CANVAS_UNSAFE', 'V4.1_base_source_skip',
            'Ref canvas works on correct/dark but mismatch/tail needs source skip')
    elif case_d:
        label, nxt, meaning = (
            'V4_CASE_D_BROAD_HARM', 'stop_v4_ref_canvas',
            'Ref-as-canvas hurts correct and Raw Ref has no PSNR advantage')
    elif case_c:
        label, nxt, meaning = (
            'V4_CASE_C_NULL', 'close_direct_ref_canvas_residual',
            'A1 ≈ A0 and ≈ Raw Ref; reverse residual unused')
    else:
        label, nxt, meaning = (
            'V4_CASE_C_NULL', 'close_direct_ref_canvas_residual',
            'Ref-as-canvas inconclusive vs B0 replay')

    out = dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(A0=a0, A1=a1, raw_ref=raw, Base=base),
        delta_a1_a0=delta, delta_a1_raw=vs_raw,
        gain_correct=g_cor, gain_dark=g_dark, gain_mismatch=g_mis,
        ref_ok=ref_ok, ignored=ignored, vs_raw_ok=vs_raw_ok,
        tail_ok=tail_ok, raw_no_adv=raw_no_adv,
        dep_normal_self=float(nrm - slf) if math.isfinite(nrm) and math.isfinite(slf)
        else float('nan'),
        dep_normal_zero=float(nrm - zro) if math.isfinite(nrm) and math.isfinite(zro)
        else float('nan'),
        dep_normal_shuffled_target=float(nrm - shf) if math.isfinite(nrm) and math.isfinite(shf)
        else float('nan'),
        large_harm_mismatch=dict(A0=lh_a0, A1=lh_a1),
        direction_A0=DIR_A0, direction_A1=DIR_A1,
        canvas_A0=CANVAS_A0, canvas_A1=CANVAS_A1,
    )
    if pair_stats is not None:
        out['pair_a1_minus_a0'] = pair_stats
    return out
