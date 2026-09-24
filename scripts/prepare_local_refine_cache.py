#!/usr/bin/env python
"""Generate the frozen-N0 Y0 float32 cache for V2 (plan section C2).

Full-image tiled inference, no clamp/round, stored as one .npy per sample so
both arms read byte-identical base outputs.
"""

import argparse
import os
import sys

import numpy as np
import torch
from imageio import imread

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import (build_manifest, cache_path, infer_n0,   # noqa: E402
                                  load_frozen_n0, read_manifest_ids,
                                  sha256, write_metadata)

NANO_SUBDIR = {'Huawei': 'nanobanana_ref', 'Nikon': 'nanobanana_ref_Nikon'}


def as_batch(path, device):
    im = imread(path)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    a = im.astype(np.float32) / 127.5 - 1.
    return torch.from_numpy(a.transpose(2, 0, 1))[None].to(device)


def eval_rows(dataset_dir):
    rows = []
    for cam in ('Huawei', 'Nikon'):
        low_dir = os.path.join(dataset_dir, 'Eval', cam, 'low')
        for name in sorted(os.listdir(low_dir)):
            low = os.path.join(low_dir, name)
            high = os.path.join(dataset_dir, 'Eval', cam, 'high', name)
            ref_dir = os.path.join(dataset_dir, 'Eval', cam, NANO_SUBDIR[cam])
            ref = None
            for ext in ('.png', '.jpg'):
                p = os.path.join(ref_dir, os.path.splitext(name)[0] + ext)
                if os.path.isfile(p):
                    ref = p
                    break
            if ref is None:
                raise SystemExit('eval reference missing for %s/%s' % (cam, name))
            rows.append(dict(sample_id='%s/%s' % (cam, name), camera=cam,
                             low_path=low, high_path=high, nano_path=ref,
                             split='eval'))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt',
                    default='/root/data/experiments/retinex_v11/N0_fixed_s42/'
                            'model/model_00040.pt')
    ap.add_argument('--base_run_dir',
                    default='/root/data/experiments/retinex_v11/N0_fixed_s42')
    ap.add_argument('--dataset_dir', default='/root/data/datasets/data1')
    ap.add_argument('--manifest_dir',
                    default='/root/data/datasets/data1/.nanobanana_sample_manifest')
    ap.add_argument('--out_root',
                    default='/root/data/experiments/retinex_v2_localref')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)

    train_ids = (read_manifest_ids(a.manifest_dir, 'Huawei')
                 + read_manifest_ids(a.manifest_dir, 'Nikon'))
    tr_rows = build_manifest(train_ids, a.dataset_dir, 'train')
    ev_rows = eval_rows(a.dataset_dir)
    print('train rows %d (%s), eval rows %d (%s)'
          % (len(tr_rows), {c: sum(1 for r in tr_rows if r['camera'] == c) for c in ('Huawei', 'Nikon')},
             len(ev_rows), {c: sum(1 for r in ev_rows if r['camera'] == c) for c in ('Huawei', 'Nikon')}))

    os.makedirs(a.out_root, exist_ok=True)
    with open(os.path.join(a.out_root, 'manifests_train.csv'), 'w', encoding='utf-8') as f:
        f.write('sample_id,camera,low_path,high_path,nano_path\n')
        for r in tr_rows:
            f.write(','.join([r['sample_id'], r['camera'], r['low_path'],
                              r['high_path'], r['nano_path']]) + '\n')
    with open(os.path.join(a.out_root, 'manifests_eval.csv'), 'w', encoding='utf-8') as f:
        f.write('sample_id,camera,low_path,high_path,nano_path\n')
        for r in ev_rows:
            f.write(','.join([r['sample_id'], r['camera'], r['low_path'],
                              r['high_path'], r['nano_path']]) + '\n')

    model, tr, cfg = load_frozen_n0(a.base_ckpt, a.base_run_dir, a.device)
    base_sha = sha256(a.base_ckpt)
    for tag, rows in (('train', tr_rows), ('eval', ev_rows)):
        cdir = os.path.join(a.out_root, 'cache_n0_' + tag)
        print('caching %s -> %s' % (tag, cdir))
        os.makedirs(cdir, exist_ok=True)
        for i, row in enumerate(rows):
            low = as_batch(row['low_path'], a.device)
            y0 = infer_n0(model, tr, low)
            np.save(cache_path(cdir, row), y0[0].float().cpu().numpy())
            if (i + 1) % 100 == 0:
                print('  %d/%d' % (i + 1, len(rows)))
        write_metadata(cdir, dict(base_ckpt=a.base_ckpt, base_sha256=base_sha,
                                  split=tag, n=len(rows), dtype='float32',
                                  tile_size=getattr(cfg, 'tile_size', 256),
                                  tile_overlap=getattr(cfg, 'tile_overlap', 96),
                                  no_reference=True, no_ref_texture=True,
                                  no_ref_illum=True, no_global_illum=True))
    print('cache done')


if __name__ == '__main__':
    main()
