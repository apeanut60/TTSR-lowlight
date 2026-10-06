"""V5.0 aligned reference texture refinement — lock, verdict, diagnostics."""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from model.V5Correspondence import CORRELATION_TYPE, SEARCH_RANGE  # noqa: E402
from model.V5DeformAlign import DCN_KERNEL  # noqa: E402
from model.V5FlowAlign import WARP_TYPE  # noqa: E402
from model.V5MatchEncoder import MATCH_CH  # noqa: E402
from model.V5Model import ARCHITECTURE  # noqa: E402
from model.V5RetinexBridge import INJECTION_POINT, REFINE_SCALES  # noqa: E402
from model.V5TextureEncoder import TEXTURE_CHANNELS  # noqa: E402
from v3a5_runtime import STATES  # noqa: E402
from v3b_runtime import GRAD_ACCUM, LR, SEED  # noqa: E402
from v3b2_runtime import pair_delta_stats_by_state  # noqa: E402
from v3b3_runtime import safety_plus  # noqa: E402

ARM_A0 = 'A0_frozen_b0'
ARM_A1 = 'A1_v5_aligned'
ARMS = (ARM_A0, ARM_A1)

DEFAULT_UPDATES = 30000
CKPT_STEPS = (0, 1000, 3000, 5000, 10000, 20000, 30000)
EVAL_STEPS = (3000, 10000, 20000, 30000)
OPTIMIZER = 'Adam'
LOSS = 'MSE'
ZERO_INIT = True

FORMAL_LOCK_KEYS_V5 = (
    'stage', 'repo_commit',
    'base_ckpt', 'base_ckpt_sha256', 'base_cache_metadata_sha256',
    'proposal_sha256', 'split_sha256', 'mismatch_train_sha256',
    'mismatch_dev_sha256', 'reference_variant',
    'architecture', 'match_encoder', 'texture_encoder',
    'search_range', 'correlation_type', 'warp_type', 'dcn_kernel',
    'refine_scales', 'zero_init', 'injection_point',
    'loss', 'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
    'frozen_base', 'trainable_modules', 'v5_init_sha',
)


def require_ckpt_blob_v5(blob, *, arm, step, init_sha, repo_commit,
                         injection_point=INJECTION_POINT, formal=True):
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
    if blob.get('repo_commit') != repo_commit:
        errs.append('ckpt repo_commit mismatch')
    if blob.get('injection_point') != injection_point:
        errs.append('injection_point: blob=%r want=%r'
                    % (blob.get('injection_point'), injection_point))
    if blob.get('loss') not in (None, LOSS, 'mse_reconstruction', 'MSE'):
        errs.append('loss: %r' % blob.get('loss'))
    if blob.get('frozen_base') is not True:
        errs.append('frozen_base: %r' % blob.get('frozen_base'))
    if errs:
        msg = 'ckpt HARD FAIL:\n  ' + '\n  '.join(errs)
        if formal:
            raise SystemExit(msg)
        raise ValueError(msg)
    return True


def lock_architecture_fields():
    return dict(
        stage='V5.0',
        architecture=ARCHITECTURE,
        match_encoder='V5MatchEncoder_3_32_64s2_64_96s2_96',
        texture_encoder='TexturePyramid_%s' % (TEXTURE_CHANNELS,),
        search_range=SEARCH_RANGE,
        correlation_type=CORRELATION_TYPE,
        warp_type=WARP_TYPE,
        dcn_kernel=DCN_KERNEL,
        refine_scales=list(REFINE_SCALES),
        zero_init=ZERO_INIT,
        injection_point=INJECTION_POINT,
        loss=LOSS,
        match_ch=MATCH_CH,
        official_test_allowed=False,
        frozen_base=True,
        trainable_modules=[
            'MatchEncoder', 'TextureEncoder', 'ChunkedHardMatcher',
            'OffsetNet', 'DCN', 'FlowDeformFuse', 'RefineH4', 'RefineH2',
        ],
    )


def feat_energy(x):
    a = x.detach().float()
    gx = a[:, :, :, 1:] - a[:, :, :, :-1]
    gy = a[:, :, 1:, :] - a[:, :, :-1, :]
    high = F.avg_pool2d(a, 3, 1, 1)
    high = a - high
    return dict(
        var=float(a.var()),
        grad=float(gx.pow(2).mean() + gy.pow(2).mean()),
        hf=float(high.pow(2).mean()),
        mean_abs=float(a.abs().mean()),
    )


def r_delta(delta, base):
    md = float(delta.detach().float().abs().mean())
    mb = float(base.detach().float().abs().mean())
    return md / (mb + 1e-8), md


def flow_stats(flow):
    mag = flow.detach().float().pow(2).sum(1).sqrt()
    flat = mag.reshape(-1)
    h, w = flow.shape[-2:]
    sx = flow[:, 0]
    sy = flow[:, 1]
    # reconstructed source coords
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing='ij',
    )
    srcx = xx[None] + sx
    srcy = yy[None] + sy
    hit = ((srcx <= 0) | (srcx >= w - 1) | (srcy <= 0) | (srcy >= h - 1)).float()
    return dict(
        disp_mean=float(flat.mean()),
        disp_p50=float(torch.quantile(flat, 0.50)),
        disp_p90=float(torch.quantile(flat, 0.90)),
        disp_p95=float(torch.quantile(flat, 0.95)),
        boundary_hit=float(hit.mean()),
    )


def residual_offset_stats(residual_offset):
    # [B,2*kk,H,W] (dy,dx) pairs; take L2 of mean over taps
    b, c, h, w = residual_offset.shape
    kk = c // 2
    yx = residual_offset.reshape(b, kk, 2, h, w)
    mag = yx.float().pow(2).sum(2).sqrt()
    flat = mag.reshape(-1)
    return dict(
        dP_mean=float(flat.mean()),
        dP_p50=float(torch.quantile(flat, 0.50)),
        dP_p90=float(torch.quantile(flat, 0.90)),
    )


def verdict_v5(psnr_b0: Dict, psnr_v5: Dict, safety_b0: Dict, safety_v5: Dict,
               dep_v5: Dict, lpips_b0: Optional[Dict] = None,
               lpips_v5: Optional[Dict] = None,
               pair_stats: Optional[Dict] = None,
               align_ok: Optional[bool] = None) -> Dict:
    def mean(d):
        vals = [float(d[s]) for s in STATES if s in d]
        return float(np.mean(vals)) if vals else float('nan')

    b0, v5 = mean(psnr_b0), mean(psnr_v5)
    delta = v5 - b0
    g_cor = float(psnr_v5.get('correct', float('nan'))
                  - psnr_b0.get('correct', float('nan')))
    g_dark = float(psnr_v5.get('true_dark_g0.5', float('nan'))
                   - psnr_b0.get('true_dark_g0.5', float('nan')))
    g_mis = float(psnr_v5.get('mismatch', float('nan'))
                  - psnr_b0.get('mismatch', float('nan')))

    def lh(safety, st):
        return float((safety.get(st) or {}).get('large_harm_rate', float('nan')))

    lh_b0 = lh(safety_b0, 'mismatch')
    lh_v5 = lh(safety_v5, 'mismatch')
    tail_ok = bool(math.isfinite(lh_b0) and math.isfinite(lh_v5)
                   and lh_v5 <= lh_b0 + 0.02)
    tail_worse = bool(math.isfinite(lh_b0) and math.isfinite(lh_v5)
                      and lh_v5 > lh_b0 + 0.02)

    nrm = float(dep_v5.get('normal', float('nan')))
    slf = float(dep_v5.get('self', float('nan')))
    zro = float(dep_v5.get('zero', float('nan')))
    shf = float(dep_v5.get('shuffled', float('nan')))
    ref_ok = bool(
        math.isfinite(nrm) and math.isfinite(slf) and math.isfinite(zro)
        and nrm >= slf + 0.02 and nrm >= zro + 0.02)
    ref_weak = bool(
        math.isfinite(nrm) and math.isfinite(slf)
        and abs(nrm - slf) < 0.02
        and (not math.isfinite(zro) or abs(nrm - zro) < 0.02))

    lpips_ok = True
    lpips_better = False
    if lpips_b0 is not None and lpips_v5 is not None:
        lb, lv = mean(lpips_b0), mean(lpips_v5)
        lpips_ok = bool(math.isfinite(lb) and math.isfinite(lv) and lv <= lb + 1e-4)
        lpips_better = bool(math.isfinite(lb) and math.isfinite(lv) and lv <= lb - 0.01)
    else:
        lpips_ok = False

    align_works = bool(align_ok) if align_ok is not None else False

    mean_strong = bool(delta >= 0.10)
    mean_mod = bool(delta >= 0.05 and delta < 0.10)
    nullish = bool(delta <= 0.02)
    unsafe = bool(g_cor >= 0.02 and (g_mis < -0.05 or tail_worse))

    if unsafe:
        label, nxt, meaning = (
            'V5_CASE_E_UNSAFE', 'close_no_gate_patch',
            'correct up / mismatch or tail down; do not expand Ref branch')
    elif (mean_strong and g_cor >= 0.0 and g_dark >= -0.02 and g_mis >= -0.02
          and tail_ok and lpips_ok and ref_ok):
        label, nxt, meaning = (
            'V5_CASE_A_STRONG_GO', 'incumbent_candidate',
            'Explicit alignment + base-aware refine beats B0')
    elif mean_mod and lpips_better and tail_ok and g_mis >= -0.02:
        label, nxt, meaning = (
            'V5_CASE_B_MODERATE_GO', 'V5.1_fusion_or_loss',
            'Moderate PSNR + LPIPS; keep V5, richer fusion or loss next')
    elif nullish and ref_ok and align_works:
        label, nxt, meaning = (
            'V5_CASE_C_ALIGN_FUSION_WEAK', 'V5.1_ReBaIR_fusion_only',
            'Alignment used; PSNR not yet; change fusion only')
    elif nullish and ref_weak:
        label, nxt, meaning = (
            'V5_CASE_D_ALIGN_NULL', 'close_explicit_alignment',
            'Correspondence/warp/DCN unused for final PSNR')
    else:
        label, nxt, meaning = (
            'V5_CASE_D_ALIGN_NULL', 'close_explicit_alignment',
            'V5.0 inconclusive vs B0')

    out = dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(B0=b0, V5=v5), delta_v5_b0=delta,
        gain_correct=g_cor, gain_dark=g_dark, gain_mismatch=g_mis,
        tail_ok=tail_ok, ref_ok=ref_ok, ref_weak=ref_weak,
        lpips_ok=lpips_ok, lpips_better=lpips_better,
        align_ok=bool(align_works),
        dep_normal_self=float(nrm - slf) if math.isfinite(nrm) and math.isfinite(slf)
        else float('nan'),
        dep_normal_zero=float(nrm - zro) if math.isfinite(nrm) and math.isfinite(zro)
        else float('nan'),
        dep_normal_shuffled=float(nrm - shf) if math.isfinite(nrm) and math.isfinite(shf)
        else float('nan'),
        large_harm_mismatch=dict(B0=lh_b0, V5=lh_v5),
    )
    if pair_stats is not None:
        out['pair_v5_minus_b0'] = pair_stats
    return out
