"""V6.0 V5-structure Ref on frozen LLFormer Base — lock, verdict, diagnostics."""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Optional

import numpy as np

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from model.LLFormerBridge import D1_CHANNELS, D2_CHANNELS  # noqa: E402
from model.V5Correspondence import CORRELATION_TYPE, SEARCH_RANGE  # noqa: E402
from model.V5DeformAlign import DCN_KERNEL  # noqa: E402
from model.V5FlowAlign import WARP_TYPE  # noqa: E402
from model.V5MatchEncoder import MATCH_CH  # noqa: E402
from model.V6LLFormerRefBridge import INJECTION_POINT, REFINE_SCALES  # noqa: E402
from model.V6Model import ARCHITECTURE  # noqa: E402
from model.V6TextureEncoder import TEXTURE_CHANNELS  # noqa: E402
from v3a5_runtime import STATES  # noqa: E402
from v3b_runtime import GRAD_ACCUM, LR, SEED  # noqa: E402
from v3b3_runtime import safety_plus  # noqa: E402
from v5_runtime import (feat_energy, flow_stats, r_delta,  # noqa: E402
                        residual_offset_stats)

ARM_A0 = 'A0_frozen_base'
ARM_A1 = 'A1_v6_aligned'
ARMS = (ARM_A0, ARM_A1)

DEFAULT_UPDATES = 30000
CKPT_STEPS = (0, 1000, 3000, 5000, 10000, 20000, 30000)
EVAL_STEPS = (3000, 10000, 20000, 30000)
OPTIMIZER = 'Adam'
LOSS = 'MSE'
ZERO_INIT = True

ROOT = '/root/data/experiments/v6_llformer_aligned_ref'
SRC = '/root/data/experiments/v3a1_lolv2real'
SPLIT_JSON = '/root/data/experiments/v3a4_lolv2real/splits/split.json'
MISMATCH_TRAIN = (
    '/root/data/experiments/v3a4_lolv2real/mappings/mismatch_train_575.json')
MISMATCH_DEV = (
    '/root/data/experiments/v3a4_lolv2real/mappings/mismatch_dev_64.json')
CACHE_NAME = 'cache_y0_llformer_e110'
REFERENCE_VARIANT = 'nanobanana_ref_v2'

FORMAL_LOCK_KEYS_V6 = (
    'stage', 'repo_commit',
    'base_ckpt', 'base_ckpt_sha256', 'base_cache_metadata_sha256',
    'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
    'reference_variant',
    'architecture', 'match_encoder', 'texture_encoder',
    'search_range', 'correlation_type', 'warp_type', 'dcn_kernel',
    'refine_scales', 'zero_init', 'injection_point',
    'loss', 'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
    'frozen_base', 'trainable_modules', 'v6_init_sha',
    'd2_channels', 'd1_channels',
)


def require_ckpt_blob_v6(blob, *, arm, step, init_sha, repo_commit,
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
        stage='V6.0',
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
        d2_channels=D2_CHANNELS,
        d1_channels=D1_CHANNELS,
        official_test_allowed=False,
        frozen_base=True,
        trainable_modules=[
            'MatchEncoder', 'TextureEncoder', 'ChunkedHardMatcher',
            'OffsetNet', 'DCN', 'FlowDeformFuse', 'RefineH4', 'RefineH2',
        ],
    )


def verdict_v6(psnr_base: Dict, psnr_v6: Dict, safety_base: Dict, safety_v6: Dict,
               dep_v6: Dict, lpips_base: Optional[Dict] = None,
               lpips_v6: Optional[Dict] = None,
               pair_stats: Optional[Dict] = None,
               align_ok: Optional[bool] = None) -> Dict:
    """Case vs frozen LLFormer Base (A0). Same thresholds as V5 vs B0."""
    def mean(d):
        vals = [float(d[s]) for s in STATES if s in d]
        return float(np.mean(vals)) if vals else float('nan')

    b0, v5 = mean(psnr_base), mean(psnr_v6)
    delta = v5 - b0
    g_cor = float(psnr_v6.get('correct', float('nan'))
                  - psnr_base.get('correct', float('nan')))
    g_dark = float(psnr_v6.get('true_dark_g0.5', float('nan'))
                   - psnr_base.get('true_dark_g0.5', float('nan')))
    g_mis = float(psnr_v6.get('mismatch', float('nan'))
                  - psnr_base.get('mismatch', float('nan')))

    def lh(safety, st):
        return float((safety.get(st) or {}).get('large_harm_rate', float('nan')))

    lh_b0 = lh(safety_base, 'mismatch')
    lh_v5 = lh(safety_v6, 'mismatch')
    tail_ok = bool(math.isfinite(lh_b0) and math.isfinite(lh_v5)
                   and lh_v5 <= lh_b0 + 0.02)
    tail_worse = bool(math.isfinite(lh_b0) and math.isfinite(lh_v5)
                      and lh_v5 > lh_b0 + 0.02)

    nrm = float(dep_v6.get('normal', float('nan')))
    slf = float(dep_v6.get('self', float('nan')))
    zro = float(dep_v6.get('zero', float('nan')))
    shf = float(dep_v6.get('shuffled', float('nan')))
    ref_ok = bool(
        math.isfinite(nrm) and math.isfinite(slf) and math.isfinite(zro)
        and nrm >= slf + 0.02 and nrm >= zro + 0.02)
    ref_weak = bool(
        math.isfinite(nrm) and math.isfinite(slf)
        and abs(nrm - slf) < 0.02
        and (not math.isfinite(zro) or abs(nrm - zro) < 0.02))

    lpips_ok = True
    lpips_better = False
    if lpips_base is not None and lpips_v6 is not None:
        lb, lv = mean(lpips_base), mean(lpips_v6)
        lpips_ok = bool(math.isfinite(lb) and math.isfinite(lv) and lv <= lb + 1e-4)
        lpips_better = bool(math.isfinite(lb) and math.isfinite(lv) and lv <= lb - 0.01)
    else:
        lpips_ok = False

    align_works = bool(align_ok) if align_ok is not None else False
    mean_strong = bool(delta >= 0.10)
    mean_mod = bool(delta >= 0.05 and delta < 0.10)
    nullish = bool(delta <= 0.02)
    unsafe = bool(g_cor >= 0.02 and (g_mis < -0.05 or tail_worse))

    # Also Case E if Ref dependence inverted (normal worse than self/zero/shuf)
    ref_inverted = bool(
        math.isfinite(nrm) and math.isfinite(slf) and math.isfinite(zro)
        and nrm < slf - 0.01 and nrm < zro - 0.01)

    if unsafe or ref_inverted:
        label, nxt, meaning = (
            'V6_CASE_E_UNSAFE', 'close_no_gate_patch',
            'correct up / mismatch-tail or Ref-inverted; do not expand Ref')
    elif (mean_strong and g_cor >= 0.0 and g_dark >= -0.02 and g_mis >= -0.02
          and tail_ok and lpips_ok and ref_ok):
        label, nxt, meaning = (
            'V6_CASE_A_STRONG_GO', 'incumbent_candidate',
            'V5-structure Ref beats frozen LLFormer Base')
    elif mean_mod and lpips_better and tail_ok and g_mis >= -0.02:
        label, nxt, meaning = (
            'V6_CASE_B_MODERATE_GO', 'V6.1_fusion_or_loss',
            'Moderate PSNR + LPIPS; keep V6 structure')
    elif nullish and ref_ok and align_works:
        label, nxt, meaning = (
            'V6_CASE_C_ALIGN_FUSION_WEAK', 'V6.1_fusion_only',
            'Alignment used; PSNR not yet')
    elif nullish and ref_weak:
        label, nxt, meaning = (
            'V6_CASE_D_ALIGN_NULL', 'close_explicit_alignment',
            'Correspondence unused for final PSNR')
    else:
        label, nxt, meaning = (
            'V6_CASE_D_ALIGN_NULL', 'close_explicit_alignment',
            'V6.0 inconclusive vs Base')

    out = dict(
        label=label, next_step=nxt, meaning=meaning,
        mean_psnr=dict(Base=b0, V6=v5), delta_v6_base=delta,
        gain_correct=g_cor, gain_dark=g_dark, gain_mismatch=g_mis,
        tail_ok=tail_ok, ref_ok=ref_ok, ref_weak=ref_weak,
        ref_inverted=ref_inverted,
        lpips_ok=lpips_ok, lpips_better=lpips_better,
        align_ok=bool(align_works),
        dep_normal_self=float(nrm - slf) if math.isfinite(nrm) and math.isfinite(slf)
        else float('nan'),
        dep_normal_zero=float(nrm - zro) if math.isfinite(nrm) and math.isfinite(zro)
        else float('nan'),
        dep_normal_shuffled=float(nrm - shf) if math.isfinite(nrm) and math.isfinite(shf)
        else float('nan'),
        large_harm_mismatch=dict(Base=lh_b0, V6=lh_v5),
    )
    if pair_stats is not None:
        out['pair_v6_minus_base'] = pair_stats
    return out
