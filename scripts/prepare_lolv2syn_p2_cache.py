#!/usr/bin/env python
"""Cache frozen S1 P2 Y0 for LOLv2-Syn H1/H3 (Huawei v2 recipe)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from local_refine_runtime import (  # noqa: E402
    cache_path, read_manifest_csv, sha256, write_metadata,
)
from train_nikon_domain import load_p2_from  # noqa: E402
from v2_p2_runtime import P2_CONFIG, infer_p2_pair  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-lolv2syn/h3_v2recipe'
# overwritten by --p2_ckpt after S1 finishes
L1_P2 = (
    '/root/data/experiments/TTSR-lowlight-lolv2syn/checkpoints/'
    'S1_p2_only_mse_s42/checkpoint_00100.pt')
PREPROCESS_VERSION = (
    'lolv2syn:S1-best P2 tiled(256/96)->float32[-1,1]; Huawei v2 H1/H3 recipe'
)


def cache_split(rows, model, args, cache_dir, device, tag, ckpt):
    os.makedirs(cache_dir, exist_ok=True)
    for i, row in enumerate(rows):
        out = cache_path(cache_dir, row)
        if os.path.isfile(out):
            continue
        y0 = infer_p2_pair(model, args, row['low_path'], row['nano_path'], device)
        if not torch.isfinite(y0).all():
            raise SystemExit('non-finite Y0 for %s' % row['sample_id'])
        np.save(out, y0[0].float().cpu().numpy())
        if (i + 1) % 25 == 0 or i == 0:
            print('  [%s] %d/%d %s' % (tag, i + 1, len(rows), row['sample_id']), flush=True)
        torch.cuda.empty_cache()
    missing = [r['sample_id'] for r in rows if not os.path.isfile(cache_path(cache_dir, r))]
    if missing:
        raise SystemExit('cache incomplete: %s' % missing[:5])
    man_bytes = json.dumps([r['sample_id'] for r in rows], sort_keys=True).encode()
    write_metadata(cache_dir, dict(
        complete=True, tag=tag, n=len(rows),
        base_checkpoint=ckpt,
        base_checkpoint_sha256=sha256(ckpt),
        p2_config=P2_CONFIG,
        preprocess_version=PREPROCESS_VERSION,
        dtype='float32', range='[-1,1]', layout='CHW',
        manifest_sha256=hashlib.sha256(man_bytes).hexdigest(),
        sample_ids=[r['sample_id'] for r in rows],
    ))
    print('[%s] complete -> %s' % (tag, cache_dir), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_root', default=EXP)
    ap.add_argument('--p2_ckpt', default=L1_P2)
    ap.add_argument('--man_src', default='/root/data/experiments/TTSR-lowlight-lolv2syn/manifests_h3')
    a = ap.parse_args()
    man = os.path.join(a.exp_root, 'manifests')
    os.makedirs(man, exist_ok=True)
    for name in ('manifests_train.csv', 'manifests_eval.csv'):
        src = os.path.join(a.man_src, name)
        dst = os.path.join(man, name)
        if not os.path.isfile(dst):
            os.symlink(os.path.abspath(src), dst)
            print('symlink', dst, '->', src, flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p2, args = load_p2_from(a.p2_ckpt, device, joint=False)
    p2.eval()
    for p in p2.parameters():
        p.requires_grad_(False)
    tr = read_manifest_csv(os.path.join(man, 'manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(man, 'manifests_eval.csv'))
    print('caching train=%d eval=%d from %s' % (len(tr), len(ev), a.p2_ckpt), flush=True)
    cache_split(tr, p2, args, os.path.join(a.exp_root, 'cache_p2_train'), device, 'train', a.p2_ckpt)
    cache_split(ev, p2, args, os.path.join(a.exp_root, 'cache_p2_eval'), device, 'eval', a.p2_ckpt)


if __name__ == '__main__':
    main()
