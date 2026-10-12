#!/usr/bin/env python
"""Cache frozen LLFormer E110 Y0 in [-1,1] CHW .npy for V6 Ref training.

Uses the same manifests as V3/V5 (refiner_train + test). Does not reuse
Retinex cache_y0_lolbase.
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

from dataset.lolv2real_v3a import read_model_image  # noqa: E402
from local_refine_runtime import sha256, write_metadata  # noqa: E402
from model.V6LLFormerRefBridge import (CANONICAL_BASE, base_forward_m11,  # noqa: E402
                                       load_frozen_llformer_base)
from v3a6_runtime import git_head  # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v6_llformer_aligned_ref'
CACHE_NAME = 'cache_y0_llformer_e110'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--cache_name', default=CACHE_NAME)
    ap.add_argument('--base_ckpt', default=CANONICAL_BASE)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)

    base = load_frozen_llformer_base(a.base_ckpt, a.device)
    jobs = [
        ('refiner_train', os.path.join(a.src_root, 'manifests', 'refiner_train.csv')),
        ('test', os.path.join(a.src_root, 'manifests', 'test.csv')),
    ]
    metas = {}
    for tag, man in jobs:
        cdir = os.path.join(a.root, a.cache_name, tag)
        os.makedirs(cdir, exist_ok=True)
        rows = list(csv.DictReader(open(man, encoding='utf-8')))
        print('=== cache %s n=%d ===' % (tag, len(rows)), flush=True)
        with torch.no_grad():
            for i, r in enumerate(rows):
                low = read_model_image(r['low_path'])[None].to(a.device)  # [-1,1]
                y0 = base_forward_m11(base, low)
                name = os.path.basename(r['low_path'])
                np.save(os.path.join(cdir, name + '.npy'),
                        y0[0].float().cpu().numpy())
                if (i + 1) % 50 == 0 or i + 1 == len(rows):
                    print('  %d/%d' % (i + 1, len(rows)), flush=True)
        write_metadata(cdir, dict(
            split=tag, n=len(rows), sample_count=len(rows),
            sample_ids=[os.path.basename(r['low_path']) for r in rows],
            base_checkpoint=a.base_ckpt,
            base_checkpoint_sha256=sha256(a.base_ckpt),
            manifest_sha256=sha256(man),
            preprocess_version='v6:llformer_e110_m11_pad16_tile256_ov96',
            pad_multiple=16, tile_size=256, tile_overlap=96,
            input_range=[-1, 1], output_range=[-1, 1],
            repo_commit=git_head(), dtype='float32', layout='CHW',
            clamp=False, round=False, infer_from_low_only=True, complete=True,
            base_arch='LLFormerBridge',
            note='Y0 = tiled Base (same tiling as V6 Ref) for step0 identity',
        ))
        metas[tag] = dict(
            n=len(rows),
            sha256=sha256(os.path.join(cdir, 'metadata.json')),
            dir=cdir,
        )
        print('%s: %d -> %s' % (tag, len(rows), cdir), flush=True)
    idx = os.path.join(a.root, a.cache_name, 'index.json')
    json.dump(metas, open(idx, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('wrote', idx, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
