#!/usr/bin/env python
"""Frozen-weight external evaluation (plan §E).

Five conditions on one shared Y0 cache. Uses the *same* verified metric and
reference-reading helpers as the source-domain V2.1 run; only the aggregation
is generic (no Huawei/Nikon 0.6/0.4).
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import (EVAL_QUERY_CHUNK, _read_eval_tensor,    # noqa: E402
                                  load_cache, mismatch_permutation,
                                  read_manifest_csv, sha256, verify_cache)
from model.LocalRefine import Refiner, count_params                      # noqa: E402


def load_refiner(ckpt, device):
    m = Refiner().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)['refiner'])
    m.eval()
    if count_params(m) != 78724:
        raise SystemExit('%s: unexpected parameter count' % ckpt)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def run(cfg, device, save_predictions=True):
    out_root = cfg['output_root']
    tdir = os.path.join(out_root, 'target_lolv2real')
    rows = read_manifest_csv(os.path.join(tdir, 'external_manifest.csv'))
    cache = os.path.join(tdir, 'cache_n0_eval')
    meta = verify_cache(cache, strict=True)
    lock = json.load(open(os.path.join(out_root, 'artifact_lock.json'), encoding='utf-8'))
    if lock['manifest_sha256'] != sha256(os.path.join(tdir, 'external_manifest.csv')):
        raise SystemExit('manifest changed since lock')
    print('samples %d, cache base %s, repo %s'
          % (len(rows), meta['_base_sha'][:16], lock['repo_commit'][:12]))

    self_m = load_refiner(cfg['source']['self_checkpoint'], device)
    nano_m = load_refiner(cfg['source']['nano_checkpoint'], device)
    self_m.match.chunk = EVAL_QUERY_CHUNK
    nano_m.match.chunk = EVAL_QUERY_CHUNK
    perm = mismatch_permutation(rows)

    pred_dir = os.path.join(tdir, 'predictions')
    conditions = ['N0', 'Self-fixed', 'Nano-fixed-correct', 'Nano-fixed-bypass',
                  'Nano-fixed-mismatch']
    if save_predictions:
        for c in conditions:
            os.makedirs(os.path.join(pred_dir, c), exist_ok=True)

    recs = []
    for i, row in enumerate(rows):
        y0 = torch.from_numpy(
            np.ascontiguousarray(load_cache(cache, row)))[None].to(device)
        hr = _read_eval_tensor(row['high_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        mis = _read_eval_tensor(rows[perm[i]]['nano_path'], device,
                                size=y0.shape[-2:])
        outs = {
            'N0': y0,
            'Self-fixed': self_m(y0, y0)[0],
            'Nano-fixed-correct': nano_m(y0, nano)[0],
            'Nano-fixed-bypass': y0,
            'Nano-fixed-mismatch': nano_m(y0, mis)[0],
        }
        for c, sr in outs.items():
            if save_predictions:
                np.save(os.path.join(pred_dir, c, row['sample_id'] + '.npy'),
                        sr[0].float().cpu().numpy())
            recs.append(dict(sample_id=row['sample_id'], camera=row['camera'],
                             group=row['group'], condition=c, sr=sr.detach(),
                             hr=hr, nano_path=row['nano_path'],
                             mismatch_path=rows[perm[i]]['nano_path']))
        if (i + 1) % 25 == 0:
            print('  %d/%d' % (i + 1, len(rows)))
    del self_m, nano_m
    return rows, recs, tdir, out_root, lock, meta, cache


def metric_rows(recs):
    from local_refine_runtime import metrics
    out = {}
    for r in recs:
        p, s, m = metrics(r['sr'], r['hr'])
        out[(r['sample_id'], r['condition'])] = dict(
            psnr_rgb=p, ssim_y=s, mse=m, camera=r['camera'], group=r['group'],
            nano_path=r['nano_path'], mismatch_path=r['mismatch_path'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no_predictions', action='store_true')
    a = ap.parse_args(_CLI)
    cfg = json.load(open(a.config, encoding='utf-8'))
    rows, recs, tdir, out_root, lock, meta, cache = run(
        cfg, a.device, save_predictions=not a.no_predictions)
    res = metric_rows(recs)
    for r in recs:
        r['sr'] = None            # release
        r['hr'] = None

    conditions = ['N0', 'Self-fixed', 'Nano-fixed-correct', 'Nano-fixed-bypass',
                  'Nano-fixed-mismatch']
    labels = ['N0', 'V2-Self', 'V2-Nano-correct', 'V2-Nano-bypass', 'V2-Nano-mismatch']
    mapping = dict(zip(conditions, labels))
    inv = {v: k for k, v in mapping.items()}
    sids = [r['sample_id'] for r in rows]

    def mean(label, key='psnr_rgb'):
        return float(np.mean([res[(s, inv[label])][key] for s in sids]))

    print()
    print('══ external: %s %s (%d samples) ══'
          % (cfg['target']['dataset'], cfg['target']['split'], len(rows)))
    print('  %-18s %10s %10s' % ('condition', 'PSNR(rgb)', 'SSIM(Y)'))
    for l in labels:
        print('  %-18s %10.4f %10.4f' % (l, mean(l), mean(l, 'ssim_y')))
    n0 = mean('N0')
    slf = mean('V2-Self')
    nan = mean('V2-Nano-correct')
    byp = mean('V2-Nano-bypass')
    mis = mean('V2-Nano-mismatch')
    print()
    print('  Δscheme   = Nano − N0   = %+.4f' % (nan - n0))
    print('  Δexternal = Nano − Self = %+.4f' % (nan - slf))
    print('  bypass    − N0          = %+.4f  (should be exactly 0)' % (byp - n0))
    print('  correct   − mismatch    = %+.4f' % (nan - mis))
    for l in ('V2-Self', 'V2-Nano-correct'):
        c = inv[l]
        d = np.array([res[(s, c)]['psnr_rgb'] - res[(s, 'N0')]['psnr_rgb'] for s in sids])
        print('  per-image %-16s vs N0: mean %+.4f median %+.4f better %d/%d worst %+.3f'
              % (l, d.mean(), np.median(d), int((d > 0).sum()), len(d), d.min()))

    out_csv = os.path.join(tdir, 'per_image_external.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'camera', 'group', 'condition', 'ref_source',
                    'ref_path', 'base_checkpoint_sha256', 'self_checkpoint_sha256',
                    'nano_checkpoint_sha256', 'cache_metadata_sha256',
                    'manifest_sha256', 'repo_commit', 'psnr_rgb', 'ssim_y',
                    'delta_vs_n0', 'delta_vs_self'])
        cache_sha = sha256(os.path.join(cache, 'metadata.json'))
        for s in sids:
            n0v = res[(s, 'N0')]['psnr_rgb']
            slv = res[(s, 'Self-fixed')]['psnr_rgb']
            for c, l in mapping.items():
                d = res[(s, c)]
                if l in ('N0', 'V2-Nano-bypass'):
                    rs, rp = 'bypass(no_refiner)', ''
                elif l == 'V2-Self':
                    rs, rp = 'self_Y0_cache', ''
                elif l == 'V2-Nano-correct':
                    rs, rp = 'nano_correct', d['nano_path']
                else:
                    rs, rp = 'nano_mismatch', d['mismatch_path']
                w.writerow([s, d['camera'], d['group'], l, rs, rp,
                            lock['base_checkpoint_sha256'],
                            lock['self_checkpoint_sha256'],
                            lock['nano_checkpoint_sha256'], cache_sha,
                            lock['manifest_sha256'], lock['repo_commit'],
                            '%.6f' % d['psnr_rgb'], '%.6f' % d['ssim_y'],
                            '%+.6f' % (d['psnr_rgb'] - n0v),
                            '%+.6f' % (d['psnr_rgb'] - slv)])
    print('\nper-image -> %s' % out_csv)


if __name__ == '__main__':
    main()
