#!/usr/bin/env python
"""v2.3 Phase D: ref ablations + residual DC split for best H0/H1/H2 ckpts."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from local_refine_runtime import (  # noqa: E402
    EVAL_QUERY_CHUNK, load_cache, metrics, read_manifest_csv, _read_eval_tensor,
)
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402
from model.v22_dc_loss import mean_hw, split_dc  # noqa: E402

V2_EXP = '/root/data/experiments/TTSR-lowlight-v2'
EXP = '/root/data/experiments/TTSR-lowlight-v23'
FOCUS = ['2053', '2059', '2055', '2058', '2042', '2052', '2057', '2044', '2061', '2056']


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device)
    enc = SharedEncoder(32).to(device).eval()
    mat = LocalSoftMatch(32, 4, 0.1, chunk=EVAL_QUERY_CHUNK).to(device).eval()
    head = V3B0ResidualFusion(96).to(device).eval()
    enc.load_state_dict(ck['encoder'])
    mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    return enc, mat, head


@torch.no_grad()
def predict(enc, mat, head, y0, nano=None, mode='correct', t_override=None):
    f0 = enc((y0 + 1.) * 0.5)
    if mode == 'bypass':
        return y0, torch.zeros_like(y0), f0, None
    if mode == 'self':
        fr = enc((y0 + 1.) * 0.5)
    elif mode == 'zero_t':
        fr = enc((nano + 1.) * 0.5)
        t = torch.zeros_like(f0)
        d = head(f0, t, y0.shape[-2:])
        return y0 + d, d, f0, t
    elif mode == 'mismatch':
        # caller passes mismatched nano already
        fr = enc((nano + 1.) * 0.5)
    else:
        fr = enc((nano + 1.) * 0.5)
    t = mat(f0, fr) if t_override is None else t_override
    d = head(f0, t, y0.shape[-2:])
    return y0 + d, d, f0, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() else 'cpu'
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    cache = os.path.join(V2_EXP, 'cache_p2_eval')
    out_dir = os.path.join(a.out_root, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(a.out_root, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # pick best step by Eval30 PSNR from eval_history
    runs = {
        'H0': 'H0_head_only_s42',
        'H1': 'H1_matcher_head_s42',
        'H2': 'H2_encoder_matcher_head_s42',
        'H3': None,
    }
    ckpts = {}
    for arm, run in runs.items():
        if arm == 'H3':
            ckpts[arm] = ('/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
                          'H3_v3b0_s42/checkpoint_02000.pt')
            continue
        hist_path = os.path.join(a.out_root, 'checkpoints', run, 'eval_history.json')
        with open(hist_path) as f:
            hist = json.load(f)
        best = max(hist, key=lambda x: x['summary']['psnr'])
        st = best['step']
        ckpts[arm] = os.path.join(
            a.out_root, 'checkpoints', run, 'checkpoint_%05d.pt' % st)
        print('best %s @ step %d PSNR=%.4f' % (arm, st, best['summary']['psnr']))

    # mismatch nano: rotate by 1 in eval list
    nano_map = {r['sample_id']: r['nano_path'] for r in ev}
    ids = [r['sample_id'] for r in ev]
    mismatch_nano = {ids[i]: nano_map[ids[(i + 1) % len(ids)]] for i in range(len(ids))}

    abl_modes = ['correct', 'self', 'mismatch', 'zero_t', 'bypass']
    results = {}

    for arm, path in ckpts.items():
        enc, mat, head = load_ckpt(path, device)
        arm_res = {m: {} for m in abl_modes}
        residual = {}
        for row in ev:
            sid = row['sample_id']
            y0 = torch.from_numpy(np.ascontiguousarray(
                load_cache(cache, row)))[None].to(device)
            gt = _read_eval_tensor(row['high_path'], device)
            nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
            nano_mm = _read_eval_tensor(
                mismatch_nano[sid], device, size=y0.shape[-2:])
            for mode in abl_modes:
                n_in = nano_mm if mode == 'mismatch' else nano
                y, d, _, _ = predict(enc, mat, head, y0, n_in, mode=mode)
                p, s, m = metrics(y, gt)
                arm_res[mode][sid] = dict(psnr=p, ssim=s, mse=m)

            # residual split on correct
            y, d, _, _ = predict(enc, mat, head, y0, nano, mode='correct')
            dm, ds = split_dc(d)
            y_mean = y0 + dm
            y_cent = y0 + ds
            p_f, _, _ = metrics(y, gt)
            p_m, _, _ = metrics(y_mean, gt)
            p_c, _, _ = metrics(y_cent, gt)
            e_full = float(d.pow(2).mean())
            e_dc = float(dm.pow(2).mean())
            e_sp = float(ds.pow(2).mean())
            residual[sid] = dict(
                psnr_full=p_f, psnr_mean_only=p_m, psnr_centered=p_c,
                e_full=e_full, e_dc=e_dc, e_sp=e_sp,
                dc_share=e_dc / max(e_full, 1e-12),
                d_mean_rgb=[float(x) for x in mean_hw(d).reshape(-1).cpu()],
            )
            # save focus visuals
            if any(sid.endswith(f) or sid == f or f in sid for f in FOCUS):
                # store numpy for later
                pass
        # summary tables
        summary = {}
        for mode in abl_modes:
            summary[mode] = dict(
                psnr=float(np.mean([v['psnr'] for v in arm_res[mode].values()])),
                ssim=float(np.mean([v['ssim'] for v in arm_res[mode].values()])),
            )
        summary['correct_minus_self'] = (
            summary['correct']['psnr'] - summary['self']['psnr'])
        summary['correct_minus_mismatch'] = (
            summary['correct']['psnr'] - summary['mismatch']['psnr'])
        summary['correct_minus_zero_t'] = (
            summary['correct']['psnr'] - summary['zero_t']['psnr'])
        summary['residual_mean'] = dict(
            psnr_full=float(np.mean([v['psnr_full'] for v in residual.values()])),
            psnr_mean_only=float(np.mean([v['psnr_mean_only'] for v in residual.values()])),
            psnr_centered=float(np.mean([v['psnr_centered'] for v in residual.values()])),
            dc_share=float(np.mean([v['dc_share'] for v in residual.values()])),
        )
        results[arm] = dict(ckpt=path, summary=summary, per_ablation=arm_res,
                            residual=residual)
        print('%s abl correct=%.3f self=%.3f mm=%.3f zeroT=%.3f bypass=%.3f '
              'C-S=%+.3f C-M=%+.3f C-Z=%+.3f | Full=%.3f Mean=%.3f Cent=%.3f DCshare=%.3f' % (
                  arm,
                  summary['correct']['psnr'], summary['self']['psnr'],
                  summary['mismatch']['psnr'], summary['zero_t']['psnr'],
                  summary['bypass']['psnr'],
                  summary['correct_minus_self'],
                  summary['correct_minus_mismatch'],
                  summary['correct_minus_zero_t'],
                  summary['residual_mean']['psnr_full'],
                  summary['residual_mean']['psnr_mean_only'],
                  summary['residual_mean']['psnr_centered'],
                  summary['residual_mean']['dc_share']))
        del enc, mat, head
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, 'ref_ablation_residual.json'), 'w') as f:
        json.dump(results, f)
    print('wrote', os.path.join(out_dir, 'ref_ablation_residual.json'))


if __name__ == '__main__':
    main()
