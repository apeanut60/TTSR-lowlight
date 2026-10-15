"""Shared runtime for TTSR-lowlight v2 LocalRefine experiments.

Adapted from historical v5 ``local_refine_runtime.py`` (Retinex N0). The frozen
base is now TTSR P2; Y0 is produced by ``v2_p2_runtime`` and stored as float32
``.npy`` caches. Manifest / cache / eval helpers are kept so Nano vs Self arms
share identical geometry and base outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import calc_psnr_and_ssim  # noqa: E402


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_manifest_csv(path):
    import csv as _csv
    if not os.path.isfile(path):
        raise SystemExit('manifest CSV not found: %s' % path)
    with open(path, encoding='utf-8') as f:
        return list(_csv.DictReader(f))


def cache_path(cache_dir, row):
    return os.path.join(cache_dir, row['sample_id'].replace('/', '__') + '.npy')


def load_cache(cache_dir, row):
    p = cache_path(cache_dir, row)
    if not os.path.isfile(p):
        raise SystemExit('Y0 cache miss: %s' % p)
    return np.load(p)


def write_metadata(cache_dir, extra):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(extra, f, indent=2, sort_keys=True)


def verify_cache(cache_dir, expect_manifest_sha=None, strict=True):
    path = os.path.join(cache_dir, 'metadata.json')
    if not os.path.isfile(path):
        raise SystemExit('cache metadata missing: %s' % path)
    meta = json.load(open(path, encoding='utf-8'))
    base = meta.get('base_checkpoint_sha256') or meta.get('base_sha256')
    if not base:
        raise SystemExit('%s records no base checkpoint hash' % path)
    meta['_base_sha'] = base
    if not meta.get('complete'):
        msg = 'cache %s has no completion flag' % cache_dir
        if strict:
            raise SystemExit(msg)
        print('[cache] WARNING: %s' % msg)
    if expect_manifest_sha and meta.get('manifest_sha256') \
            and meta['manifest_sha256'] != expect_manifest_sha:
        raise SystemExit('cache %s was built from a different manifest' % cache_dir)
    return meta


def metrics(sr, hr):
    if sr.shape != hr.shape:
        raise ValueError('shape mismatch %s vs %s'
                         % (tuple(sr.shape), tuple(hr.shape)))
    t = calc_psnr_and_ssim(sr.detach(), hr.detach())
    # calc_psnr_and_ssim -> (ssim_map?, ssim, mse, psnr) historically used as
    # _, ssim, mse, psnr in v1 eval scripts.
    return float(t[3]), float(t[1]), float(t[2])


EVAL_QUERY_CHUNK = 1024


def mismatch_permutation(rows):
    """Fixed cyclic shift within each camera block (no fixed points)."""
    perm = list(range(len(rows)))
    for cam in sorted({r['camera'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['camera'] == cam]
        k = len(idx)
        if k > 1:
            for pos, i in enumerate(idx):
                perm[i] = idx[(pos + 1) % k]
    if any(perm[i] == i for i in range(len(rows))):
        raise SystemExit('fixed point in mismatch permutation')
    return perm


def _read_eval_tensor(path, device, size=None):
    from imageio import imread
    im = imread(path)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    t = torch.from_numpy(
        (im.astype(np.float32) / 127.5 - 1.).transpose(2, 0, 1))[None]
    if size is not None and tuple(t.shape[-2:]) != tuple(size):
        t = torch.nn.functional.interpolate(
            t, size=tuple(size), mode='bilinear', align_corners=False)
    return t.to(device)


def _metric_triple(sr, hr):
    p, s, m = metrics(sr, hr)
    return dict(psnr_rgb=p, ssim_y=s, mse=m)


@torch.no_grad()
def evaluate_conditions(refiner, rows, eval_cache_dir, conditions, device,
                        mismatch=None, verbose=True):
    """conditions: list of (label, mode) with mode in bypass|self|nano|mismatch."""
    refiner.eval()
    if hasattr(refiner, 'match'):
        refiner.match.chunk = EVAL_QUERY_CHUNK
    out = {label: {} for label, _ in conditions}
    if mismatch is None:
        mismatch = mismatch_permutation(rows)
    for i, row in enumerate(rows):
        y0 = torch.from_numpy(np.ascontiguousarray(load_cache(eval_cache_dir, row)))
        y0 = y0[None].to(device)
        hr = _read_eval_tensor(row.get('high_path') or row['gt_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        mis = _read_eval_tensor(
            rows[mismatch[i]]['nano_path'], device, size=y0.shape[-2:])
        for label, mode in conditions:
            if mode == 'bypass':
                sr = y0
            else:
                r = {'self': y0, 'nano': nano, 'mismatch': mis}[mode]
                sr = refiner(y0.detach(), r.detach())[0]
            out[label][row['sample_id']] = dict(
                camera=row['camera'], **_metric_triple(sr, hr))
    if verbose:
        _print_table(out, rows)
    return out


def _print_table(out, rows):
    labels = list(out.keys())
    cams = sorted({r['camera'] for r in rows})
    print('  %-10s %s' % ('camera', ' '.join('%12s' % l for l in labels)))
    for cam in cams + ['ALL']:
        vals = {}
        for l in labels:
            ids = [sid for sid, d in out[l].items()
                   if cam == 'ALL' or d['camera'] == cam]
            vals[l] = float(np.mean([out[l][i]['psnr_rgb'] for i in ids]))
        print('  %-10s %s' % (cam, ' '.join('%12.3f' % vals[l] for l in labels)))


@torch.no_grad()
def evaluate_v3b0(head, encoder, match, rows, eval_cache_dir, device,
                  verbose=True):
    """H3 eval: Y = Y0 + V3B0(F0,T) with frozen encoder/match from H1."""
    head.eval()
    encoder.eval()
    match.eval()
    if hasattr(match, 'chunk'):
        match.chunk = EVAL_QUERY_CHUNK
    out = {'P2': {}, 'H3': {}}
    for row in rows:
        y0 = torch.from_numpy(np.ascontiguousarray(load_cache(eval_cache_dir, row)))
        y0 = y0[None].to(device)
        hr = _read_eval_tensor(row.get('high_path') or row['gt_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        f0 = encoder((y0 + 1.) * 0.5)
        fr = encoder((nano + 1.) * 0.5)
        t = match(f0, fr)
        dy = head(f0, t, y0.shape[-2:])
        sr = y0 + dy
        out['P2'][row['sample_id']] = dict(
            camera=row['camera'], **_metric_triple(y0, hr))
        out['H3'][row['sample_id']] = dict(
            camera=row['camera'], **_metric_triple(sr, hr))
    if verbose:
        _print_table(out, rows)
    return out
