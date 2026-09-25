#!/usr/bin/env python
"""Cache the frozen Base-R output (Y0) for V3-A (plan §2).

Y0 depends only on the low input, so a single cache serves every reference
variant. FP32 .npy, no clamp, no round; metadata binds the base SHA, the
manifests and the inference code version.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.lolv2real_v3a import read_model_image                    # noqa: E402
from local_refine_runtime import (infer_n0, load_frozen_n0, sha256,   # noqa: E402
                                  write_metadata)


def _commit():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/data/experiments/v3a_lolv2real')
    ap.add_argument('--cache_name', default='cache_y0')
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--base_run_dir', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)

    # always use the fixed trainer (outer-border tiling fix), same as V2.1
    model, tr, cfg = load_frozen_n0(a.base_ckpt, a.base_run_dir, a.device)

    jobs = [('refiner_train', os.path.join(a.root, 'manifests', 'refiner_train.csv')),
            ('test', os.path.join(a.root, 'manifests', 'test.csv'))]
    import csv
    metas = {}
    for tag, man in jobs:
        cdir = os.path.join(a.root, a.cache_name, tag)
        os.makedirs(cdir, exist_ok=True)
        rows = list(csv.DictReader(open(man, encoding='utf-8')))
        for i, r in enumerate(rows):
            low = read_model_image(r['low_path'])[None].to(a.device)
            y0 = infer_n0(model, tr, low)
            np.save(os.path.join(cdir, os.path.basename(r['low_path']) + '.npy'),
                    y0[0].float().cpu().numpy())
            if (i + 1) % 100 == 0:
                print('  %s %d/%d' % (tag, i + 1, len(rows)))
        write_metadata(cdir, dict(
            split=tag, n=len(rows), sample_count=len(rows),
            sample_ids=[os.path.basename(r['low_path']) for r in rows],
            base_checkpoint=a.base_ckpt, base_checkpoint_sha256=sha256(a.base_ckpt),
            manifest_sha256=sha256(man),
            preprocess_version='v21:decode->normalise(div127.5-1)->resize->geometry',
            tile_size=getattr(cfg, 'tile_size', 256),
            tile_overlap=getattr(cfg, 'tile_overlap', 96),
            tile_window=getattr(cfg, 'tile_window', 'cosine'),
            repo_commit=_commit(), dtype='float32', layout='CHW',
            clamp=False, round=False, infer_from_low_only=True, complete=True))
        metas[tag] = dict(n=len(rows), sha256=sha256(os.path.join(cdir, 'metadata.json')))
        print('%s: %d cached -> %s' % (tag, len(rows), cdir))
    json.dump(metas, open(os.path.join(a.root, a.cache_name, 'index.json'), 'w',
                          encoding='utf-8'), indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
