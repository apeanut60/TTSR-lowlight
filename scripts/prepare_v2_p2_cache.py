#!/usr/bin/env python
"""Generate frozen TTSR-P2 Y0 float32 caches for v2 (train500 + Eval30)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from local_refine_runtime import (  # noqa: E402
    cache_path, read_manifest_csv, sha256, write_metadata,
)
from v2_p2_runtime import (  # noqa: E402
    CKPT, P2_CONFIG, infer_p2_pair, load_frozen_p2,
)

EXP = '/root/data/experiments/TTSR-lowlight-v2'
PREPROCESS_VERSION = (
    'v2:decode->normalise(div127.5-1)->nano_resize(bilinear,ac=False)'
    '->TTSR_P2_tiled(256/96 cosine)->float32[-1,1] no clamp'
)


def _repo_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def _code_sha():
    h = hashlib.sha256()
    for rel in ('v2_p2_runtime.py', 'local_refine_runtime.py',
                'model/TTSREnhance.py', 'model/MainNetEnhance.py',
                'model/query_illum_align.py'):
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                h.update(f.read())
    return h.hexdigest()


def cache_split(rows, model, args, cache_dir, device, tag):
    os.makedirs(cache_dir, exist_ok=True)
    for i, row in enumerate(rows):
        out = cache_path(cache_dir, row)
        if os.path.isfile(out):
            if (i + 1) % 50 == 0:
                print('  [%s] skip existing %d/%d' % (tag, i + 1, len(rows)))
            continue
        y0 = infer_p2_pair(model, args, row['low_path'], row['nano_path'], device)
        if not torch.isfinite(y0).all():
            raise SystemExit('non-finite Y0 for %s' % row['sample_id'])
        np.save(out, y0[0].float().cpu().numpy())
        if (i + 1) % 10 == 0 or i == 0:
            print('  [%s] cached %d/%d %s' % (
                tag, i + 1, len(rows), row['sample_id']))
        torch.cuda.empty_cache()
    # verify complete
    missing = [r['sample_id'] for r in rows
               if not os.path.isfile(cache_path(cache_dir, r))]
    if missing:
        raise SystemExit('cache incomplete, missing %d: %s'
                         % (len(missing), missing[:5]))
    man_bytes = json.dumps(
        [r['sample_id'] for r in rows], sort_keys=True).encode()
    write_metadata(cache_dir, dict(
        complete=True,
        tag=tag,
        n=len(rows),
        base_checkpoint=CKPT,
        base_checkpoint_sha256=sha256(CKPT),
        git_commit=_repo_commit(),
        code_sha256=_code_sha(),
        p2_config=P2_CONFIG,
        preprocess_version=PREPROCESS_VERSION,
        dtype='float32',
        range='[-1,1]',
        layout='CHW',
        manifest_sha256=hashlib.sha256(man_bytes).hexdigest(),
        sample_ids=[r['sample_id'] for r in rows],
    ))
    print('[%s] complete -> %s' % (tag, cache_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_root', default=EXP)
    ap.add_argument('--only', choices=['train', 'eval', 'both'], default='both')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, args = load_frozen_p2(device)
    man = os.path.join(a.exp_root, 'manifests')
    if a.only in ('train', 'both'):
        rows = read_manifest_csv(os.path.join(man, 'manifests_train.csv'))
        if a.limit:
            rows = rows[:a.limit]
        cache_split(rows, model, args,
                    os.path.join(a.exp_root, 'cache_p2_train'), device, 'train')
    if a.only in ('eval', 'both'):
        rows = read_manifest_csv(os.path.join(man, 'manifests_eval.csv'))
        if a.limit:
            rows = rows[:a.limit]
        cache_split(rows, model, args,
                    os.path.join(a.exp_root, 'cache_p2_eval'), device, 'eval')


if __name__ == '__main__':
    main()
