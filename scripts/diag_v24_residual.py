#!/usr/bin/env python
"""v2.4 Phase D: ref ablations + DC residual split for best A0–A3."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from local_refine_runtime import metrics, read_manifest_csv, _read_eval_tensor  # noqa: E402
from model.v22_dc_loss import mean_hw, split_dc  # noqa: E402
from train_v24_end2end import (  # noqa: E402
    EXP, V2_EXP, configure_p2, h3_forward, load_h3, load_p2,
)
from v2_p2_runtime import tiled_p2  # noqa: E402

FOCUS = ['2053', '2059', '2055', '2058', '2042', '2052', '2057', '2044', '2061', '2056']
ARMS = {
    'A0': 'A0_frozenP2_mse_s42',
    'A1': 'A1_frozenP2_l1ssim_s42',
    'A2': 'A2_jointP2_mse_s42',
    'A3': 'A3_jointP2_l1ssim_s42',
}


def load_best(arm, run, out_root, device):
    hist = json.load(open(os.path.join(out_root, 'checkpoints', run, 'eval_history.json')))
    best = max(hist, key=lambda x: x['summary']['psnr'])
    st = best['step']
    path = os.path.join(out_root, 'checkpoints', run, 'checkpoint_%05d.pt' % st)
    ck = torch.load(path, map_location=device)
    p2, args = load_p2(device, joint=False)
    p2.load_state_dict(ck['p2']); configure_p2(p2, joint=False)
    enc, mat, head, _ = load_h3(device)
    enc.load_state_dict(ck['encoder']); mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    enc.eval(); mat.eval(); head.eval(); p2.eval()
    print('best %s @%d PSNR=%.4f' % (arm, st, best['summary']['psnr']))
    return p2, args, enc, mat, head, path, st


@torch.no_grad()
def h3_ablate(enc, mat, head, y0, nano, nano_mm, mode):
    f0 = enc((y0 + 1.) * 0.5)
    if mode == 'bypass':
        return y0, torch.zeros_like(y0)
    if mode == 'self':
        fr = enc((y0 + 1.) * 0.5)
        t = mat(f0, fr)
    elif mode == 'mismatch':
        fr = enc((nano_mm + 1.) * 0.5)
        t = mat(f0, fr)
    elif mode == 'zero_t':
        t = torch.zeros_like(f0)
    else:
        fr = enc((nano + 1.) * 0.5)
        t = mat(f0, fr)
    d = head(f0, t, y0.shape[-2:])
    return y0 + d, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=EXP)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    ids = [r['sample_id'] for r in ev]
    nano_map = {r['sample_id']: r['nano_path'] for r in ev}
    mismatch = {ids[i]: nano_map[ids[(i + 1) % len(ids)]] for i in range(len(ids))}
    out_dir = os.path.join(a.out_root, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    modes = ['correct', 'self', 'mismatch', 'zero_t', 'bypass']
    results = {}

    for arm, run in ARMS.items():
        p2, args, enc, mat, head, path, st = load_best(arm, run, a.out_root, device)
        arm_res = {m: {} for m in modes}
        residual = {}
        for row in ev:
            sid = row['sample_id']
            low = _read_eval_tensor(row['low_path'], device)
            nano = _read_eval_tensor(row['nano_path'], device, size=low.shape[-2:])
            nano_mm = _read_eval_tensor(mismatch[sid], device, size=low.shape[-2:])
            gt = _read_eval_tensor(row['high_path'], device)
            y0 = tiled_p2(p2, args, low, low, nano, nano)
            for mode in modes:
                y, d = h3_ablate(enc, mat, head, y0, nano, nano_mm, mode)
                p, s, m = metrics(y, gt)
                arm_res[mode][sid] = dict(psnr=p, ssim=s, mse=m)
            y, d = h3_ablate(enc, mat, head, y0, nano, nano_mm, 'correct')
            dm, ds = split_dc(d)
            p_f, _, _ = metrics(y, gt)
            p_m, _, _ = metrics(y0 + dm, gt)
            p_c, _, _ = metrics(y0 + ds, gt)
            e_full = float(d.pow(2).mean())
            e_dc = float(dm.pow(2).mean())
            residual[sid] = dict(
                psnr_full=p_f, psnr_mean_only=p_m, psnr_centered=p_c,
                e_full=e_full, e_dc=e_dc, e_sp=float(ds.pow(2).mean()),
                dc_share=e_dc / max(e_full, 1e-12),
                d_mean_rgb=[float(x) for x in mean_hw(d).reshape(-1).cpu()],
                p2_psnr=metrics(y0, gt)[0],
            )
        summary = {}
        for mode in modes:
            summary[mode] = dict(
                psnr=float(np.mean([v['psnr'] for v in arm_res[mode].values()])),
                ssim=float(np.mean([v['ssim'] for v in arm_res[mode].values()])),
            )
        summary['correct_minus_self'] = summary['correct']['psnr'] - summary['self']['psnr']
        summary['correct_minus_mismatch'] = (
            summary['correct']['psnr'] - summary['mismatch']['psnr'])
        summary['correct_minus_zero_t'] = (
            summary['correct']['psnr'] - summary['zero_t']['psnr'])
        summary['residual_mean'] = dict(
            psnr_full=float(np.mean([v['psnr_full'] for v in residual.values()])),
            psnr_mean_only=float(np.mean([v['psnr_mean_only'] for v in residual.values()])),
            psnr_centered=float(np.mean([v['psnr_centered'] for v in residual.values()])),
            dc_share=float(np.mean([v['dc_share'] for v in residual.values()])),
            p2_only=float(np.mean([v['p2_psnr'] for v in residual.values()])),
        )
        # focus
        focus = {}
        for key in FOCUS:
            sid = next((s for s in residual if key in s), None)
            if sid:
                focus[key] = residual[sid]
        results[arm] = dict(
            ckpt=path, step=st, summary=summary, per_ablation=arm_res,
            residual=residual, focus=focus)
        print('%s C=%.3f S=%.3f M=%.3f Z=%.3f B=%.3f | Full=%.3f Mean=%.3f Cent=%.3f '
              'DCsh=%.3f P2=%.3f' % (
                  arm, summary['correct']['psnr'], summary['self']['psnr'],
                  summary['mismatch']['psnr'], summary['zero_t']['psnr'],
                  summary['bypass']['psnr'],
                  summary['residual_mean']['psnr_full'],
                  summary['residual_mean']['psnr_mean_only'],
                  summary['residual_mean']['psnr_centered'],
                  summary['residual_mean']['dc_share'],
                  summary['residual_mean']['p2_only']))
        del p2, enc, mat, head
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, 'ref_ablation_residual.json'), 'w') as f:
        json.dump(results, f)
    print('wrote', os.path.join(out_dir, 'ref_ablation_residual.json'))


if __name__ == '__main__':
    main()
