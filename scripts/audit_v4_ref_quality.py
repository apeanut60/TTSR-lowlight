#!/usr/bin/env python
"""V4.0 Ref quality audit: Y0 vs GT and raw R vs GT on dev64. No training."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics as _metrics                    # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors       # noqa: E402
from v3a5_runtime import STATES                                         # noqa: E402
from v3a6_runtime import dump_json                                      # noqa: E402
from v3a72_runtime import json_ready                                    # noqa: E402
from v4_runtime import (gaussian_lowpass, hf_energy, laplacian_l1,      # noqa: E402
                        spatial_grad_l1, summarize_list)

SRC = '/root/data/experiments/v3a1_lolv2real'
B0 = '/root/data/experiments/v3b0_implicit_residual'
ROOT = '/root/data/experiments/v4_ref_canvas'


def _psnr_ssim(a, b):
    t = _metrics(a, b)
    return float(t[0]), float(t[1])


def _low_psnr(a, b):
    return float(_metrics(gaussian_lowpass(a), gaussian_lowpass(b))[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--b0_root', default=B0)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(_CLI)

    b0 = json.load(open(os.path.join(a.b0_root, 'artifact_lock.json')))
    os.makedirs(os.path.join(a.root, 'diagnostics'), exist_ok=True)
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = b0['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       b0['split_json'])
    mmap = json.load(open(b0['mismatch_dev']))
    ds = make_dataset(
        ns, splits['dev'],
        os.path.join(a.src_root, b0.get('cache_name', 'cache_y0_lolbase'),
                     'refiner_train'), mmap)
    n = len(splits['dev']) if not a.limit else min(a.limit, len(splits['dev']))

    lpips_net = None
    lpips_ver = None
    try:
        import lpips
        lpips_net = lpips.LPIPS(net='alex').to(a.device).eval()
        lpips_ver = getattr(lpips, '__version__', 'unknown')
        for p in lpips_net.parameters():
            p.requires_grad_(False)
    except Exception as e:
        print('LPIPS skipped: %s' % e, flush=True)

    buckets = {s: dict(y0_psnr=[], r_psnr=[], y0_ssim=[], r_ssim=[],
                       y0_lpips=[], r_lpips=[], y0_low=[], r_low=[],
                       y0_grad=[], r_grad=[], y0_lap=[], r_lap=[],
                       y0_hf=[], r_hf=[], gt_hf=[], win=[])
               for s in STATES}
    print('=== V4 ref quality audit n=%d lpips=%s ===' % (n, lpips_ver), flush=True)
    with torch.no_grad():
        for i in range(n):
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                y0, r, h = t['Y0'], t['R'], t['H']
                py, sy = _psnr_ssim(y0, h)
                pr, sr = _psnr_ssim(r, h)
                b = buckets[state]
                b['y0_psnr'].append(py)
                b['r_psnr'].append(pr)
                b['y0_ssim'].append(sy)
                b['r_ssim'].append(sr)
                b['y0_low'].append(_low_psnr(y0, h))
                b['r_low'].append(_low_psnr(r, h))
                b['y0_grad'].append(spatial_grad_l1(y0, h))
                b['r_grad'].append(spatial_grad_l1(r, h))
                b['y0_lap'].append(laplacian_l1(y0, h))
                b['r_lap'].append(laplacian_l1(r, h))
                b['y0_hf'].append(hf_energy(y0))
                b['r_hf'].append(hf_energy(r))
                b['gt_hf'].append(hf_energy(h))
                b['win'].append(1.0 if pr > py else 0.0)
                if lpips_net is not None:
                    b['y0_lpips'].append(float(lpips_net(y0, h).mean()))
                    b['r_lpips'].append(float(lpips_net(r, h).mean()))
            if (i + 1) % 8 == 0 or i + 1 == n:
                print('  %d/%d' % (i + 1, n), flush=True)

    def pack(state):
        b = buckets[state]
        out = dict(
            y0_psnr=summarize_list(b['y0_psnr']),
            r_psnr=summarize_list(b['r_psnr']),
            y0_ssim=summarize_list(b['y0_ssim']),
            r_ssim=summarize_list(b['r_ssim']),
            y0_lowpass_psnr=summarize_list(b['y0_low']),
            r_lowpass_psnr=summarize_list(b['r_low']),
            y0_grad_l1=summarize_list(b['y0_grad']),
            r_grad_l1=summarize_list(b['r_grad']),
            y0_lap_l1=summarize_list(b['y0_lap']),
            r_lap_l1=summarize_list(b['r_lap']),
            y0_hf_energy=summarize_list(b['y0_hf']),
            r_hf_energy=summarize_list(b['r_hf']),
            gt_hf_energy=summarize_list(b['gt_hf']),
            win_rate_r_gt_y0=float(np.mean(b['win'])),
            delta_psnr_r_minus_y0=float(np.mean(np.asarray(b['r_psnr']) - np.asarray(b['y0_psnr']))),
        )
        if b['y0_lpips']:
            out['y0_lpips'] = summarize_list(b['y0_lpips'])
            out['r_lpips'] = summarize_list(b['r_lpips'])
            out['delta_lpips_r_minus_y0'] = float(
                np.mean(np.asarray(b['r_lpips']) - np.asarray(b['y0_lpips'])))
        return out

    report = dict(
        n=n, split='dev',
        reference_variant=b0['reference_variant'],
        lpips_net='alex' if lpips_net is not None else None,
        lpips_version=lpips_ver,
        by_state={s: pack(s) for s in STATES},
        note='Do not kill V4 only because raw Ref PSNR is low; check LPIPS/HF/low-pass.',
    )
    out_path = os.path.join(a.root, 'diagnostics', 'ref_quality_audit.json')
    dump_json(out_path, json_ready(report))
    print('WROTE', out_path, flush=True)
    for s in STATES:
        p = report['by_state'][s]
        extra = ''
        if 'r_lpips' in p:
            extra = '  lpips Y0/R=%.3f/%.3f' % (p['y0_lpips']['mean'], p['r_lpips']['mean'])
        print('  %s  Y0=%.3f R=%.3f Δ=%.3f win=%.2f  ssim Y0/R=%.3f/%.3f%s' % (
            s, p['y0_psnr']['mean'], p['r_psnr']['mean'],
            p['delta_psnr_r_minus_y0'], p['win_rate_r_gt_y0'],
            p['y0_ssim']['mean'], p['r_ssim']['mean'], extra), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
