#!/usr/bin/env python
"""Where does the refiner's correction live: matched content, or global level?

The external LOL run showed `correct - mismatch = +0.061` (CI crosses zero)
while `mismatch - N0 = +0.233`. If the correction is mostly a low-frequency
(illumination / level) adjustment, then a wrong reference works nearly as well
as the right one -- which is what we measured.

This script separates the correction `residual = sr - Y0` into a low-pass and a
high-pass part and reports the PSNR obtainable from each part alone:

    y0 + lowpass(residual)      global level / colour trend
    y0 + highpass(residual)     local texture / detail

`lowpass + highpass == residual`, so the two parts are exhaustive and the
decomposition is exact, not a model.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import (EVAL_QUERY_CHUNK, _read_eval_tensor,   # noqa: E402
                                  load_cache, read_manifest_csv)
from model.LocalRefine import Refiner                                    # noqa: E402


def gaussian_kernel(sigma, device):
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum())


def lowpass(x, sigma):
    """Separable Gaussian blur with replicate padding (no border darkening)."""
    if sigma <= 0:
        return x
    k = gaussian_kernel(sigma, x.device)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode='replicate')
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def psnr_rgb(sr, hr):
    """Same convention as the main metric: round to uint8, peak 255."""
    a = ((sr + 1.) * 127.5).round()
    b = ((hr + 1.) * 127.5).round()
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')


def load_refiner(ckpt, device):
    m = Refiner().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)['refiner'])
    m.eval()
    m.match.chunk = EVAL_QUERY_CHUNK
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def analyse(manifest, cache, ckpt_nano, ckpt_self, device, sigma, label):
    rows = read_manifest_csv(manifest)
    nano_m = load_refiner(ckpt_nano, device)
    self_m = load_refiner(ckpt_self, device)
    acc = {k: [] for k in ('N0', 'self', 'correct', 'mismatch')}
    stats = {k: dict(gate_mean=[], gate_spatial_std=[], resid_absmean=[],
                     lf_frac=[], psnr_lf=[], psnr_hf=[])
             for k in ('self', 'correct', 'mismatch')}
    for row in rows:
        y0 = torch.from_numpy(
            np.ascontiguousarray(load_cache(cache, row)))[None].to(device)
        hr = _read_eval_tensor(row['high_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        acc['N0'].append(psnr_rgb(y0, hr))
        for name, model, ref in (('self', self_m, y0),
                                 ('correct', nano_m, nano),
                                 ('mismatch', nano_m, None)):
            if name == 'mismatch':
                continue
            sr, aux = model(y0, ref)
            resid = sr - y0
            lo = lowpass(resid, sigma)
            hi = resid - lo
            s = stats[name]
            s['gate_mean'].append(float(aux['gate'].mean()))
            s['gate_spatial_std'].append(float(aux['gate'].std()))
            s['resid_absmean'].append(float(resid.abs().mean()))
            el = float((lo ** 2).sum())
            eh = float((hi ** 2).sum())
            s['lf_frac'].append(el / (el + eh + 1e-12))
            s['psnr_lf'].append(psnr_rgb(y0 + lo, hr))
            s['psnr_hf'].append(psnr_rgb(y0 + hi, hr))
            if name == 'correct':
                acc['correct'].append(psnr_rgb(sr, hr))
            else:
                acc['self'].append(psnr_rgb(sr, hr))

    # mismatch pass (needs the deranged reference)
    perm = list(range(len(rows)))
    for cam in sorted({r['camera'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['camera'] == cam]
        for pos, i in enumerate(idx):
            perm[i] = idx[(pos + 1) % len(idx)]
    for i, row in enumerate(rows):
        y0 = torch.from_numpy(
            np.ascontiguousarray(load_cache(cache, row)))[None].to(device)
        hr = _read_eval_tensor(row['high_path'], device)
        mis = _read_eval_tensor(rows[perm[i]]['nano_path'], device,
                                size=y0.shape[-2:])
        sr, aux = nano_m(y0, mis)
        resid = sr - y0
        lo = lowpass(resid, sigma)
        hi = resid - lo
        s = stats['mismatch']
        s['gate_mean'].append(float(aux['gate'].mean()))
        s['gate_spatial_std'].append(float(aux['gate'].std()))
        s['resid_absmean'].append(float(resid.abs().mean()))
        el = float((lo ** 2).sum())
        eh = float((hi ** 2).sum())
        s['lf_frac'].append(el / (el + eh + 1e-12))
        s['psnr_lf'].append(psnr_rgb(y0 + lo, hr))
        s['psnr_hf'].append(psnr_rgb(y0 + hi, hr))
        acc['mismatch'].append(psnr_rgb(sr, hr))

    def m(v):
        return float(np.mean(v)) if v else None

    out = dict(label=label, n=len(rows), sigma=sigma,
               psnr={k: m(v) for k, v in acc.items()},
               stats={k: {kk: m(vv) for kk, vv in v.items()}
                      for k, v in stats.items()})
    n0, cor = out['psnr']['N0'], out['psnr']['correct']
    out['gains'] = dict(Nano_vs_N0=cor - n0,
                        Nano_vs_Self=cor - out['psnr']['self'],
                        correct_vs_mismatch=cor - out['psnr']['mismatch'])
    out['gains_lf_only'] = {k: v - n0 for k, v in
                            (('correct', out['stats']['correct']['psnr_lf']),
                             ('mismatch', out['stats']['mismatch']['psnr_lf']))}
    out['gains_hf_only'] = {k: v - n0 for k, v in
                            (('correct', out['stats']['correct']['psnr_hf']),
                             ('mismatch', out['stats']['mismatch']['psnr_hf']))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--ckpt_nano', required=True)
    ap.add_argument('--ckpt_self', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--sigma', type=float, default=8.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='')
    a = ap.parse_args(_CLI)
    r = analyse(a.manifest, a.cache, a.ckpt_nano, a.ckpt_self, a.device,
                a.sigma, a.label)
    print('══ %s  (n=%d, low-pass sigma=%.0f px) ══' % (a.label, r['n'], r['sigma']))
    print('  PSNR: ' + '  '.join('%s=%.4f' % (k, v) for k, v in r['psnr'].items()))
    print('  gains: ' + '  '.join('%s=%+.4f' % (k, v) for k, v in r['gains'].items()))
    print()
    print('  %-9s %8s %9s %9s %9s %9s %9s' % ('cond', 'gate', 'gate_sd',
                                              '|resid|', 'LF frac', 'PSNR_LF', 'PSNR_HF'))
    for c, s in r['stats'].items():
        print('  %-9s %8.4f %9.4f %9.5f %9.3f %9.4f %9.4f'
              % (c, s['gate_mean'], s['gate_spatial_std'], s['resid_absmean'],
                 s['lf_frac'], s['psnr_lf'], s['psnr_hf']))
    print()
    print('  gain over N0 from the LOW-pass part only : %s'
          % '  '.join('%s=%+.4f' % (k, v) for k, v in r['gains_lf_only'].items()))
    print('  gain over N0 from the HIGH-pass part only: %s'
          % '  '.join('%s=%+.4f' % (k, v) for k, v in r['gains_hf_only'].items()))
    if a.out:
        json.dump(r, open(a.out, 'w', encoding='utf-8'), indent=2, sort_keys=True)
        print('\n-> %s' % a.out)


if __name__ == '__main__':
    main()
