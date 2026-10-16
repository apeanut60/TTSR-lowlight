#!/usr/bin/env python
"""Build Nikon train/eval manifests for v27 domain FT.

Train: man500 Nano Nikon minus Eval stems 3001-3020 (3 Nano overlaps dropped).
Eval: Eval/Nikon 20 triplets with nanobanana_ref_Nikon.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from PIL import Image

DATA = '/root/data/datasets/data1'
MAN500 = os.path.join(DATA, '.nanobanana_sample_manifest_500/Nikon.txt')
OUT = '/root/data/experiments/TTSR-lowlight-v27-nikon/manifests'
EVAL_STEMS = {str(i) for i in range(3001, 3021)}
COLS = [
    'sample_id', 'camera', 'split', 'low_path', 'gt_path', 'high_path',
    'nano_path', 'low_hw', 'gt_hw', 'nano_hw', 'low_dtype', 'gt_dtype',
    'nano_dtype', 'decode_status',
]


def hw(path):
    with Image.open(path) as im:
        w, h = im.size
        dtype = str(im.mode)
    return '%dx%d' % (w, h), 'uint8'


def find_nano(stem, split):
    if split == 'eval':
        d = os.path.join(DATA, 'Eval/Nikon/nanobanana_ref_Nikon')
        for ext in ('.png', '.jpg', '.jpeg'):
            p = os.path.join(d, stem + ext)
            if os.path.isfile(p):
                return p
    else:
        d = os.path.join(DATA, 'Training data/Nikon/nanobanana_ref')
        for ext in ('.jpg', '.png', '.jpeg'):
            p = os.path.join(d, stem + ext)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError('nano for %s/%s' % (split, stem))


def row(sample_id, split, low, high, nano):
    low_hw, _ = hw(low)
    gt_hw, _ = hw(high)
    nano_hw, _ = hw(nano)
    return dict(
        sample_id=sample_id, camera='Nikon', split=split,
        low_path=low, gt_path=high, high_path=high, nano_path=nano,
        low_hw=low_hw, gt_hw=gt_hw, nano_hw=nano_hw,
        low_dtype='uint8', gt_dtype='uint8', nano_dtype='uint8',
        decode_status='ok',
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    # --- train ---
    train_rows = []
    dropped = []
    for line in open(MAN500):
        low = line.strip()
        if not low:
            continue
        stem = os.path.splitext(os.path.basename(low))[0]
        if stem in EVAL_STEMS:
            dropped.append(stem)
            continue
        high = low.replace('/low/', '/high/')
        nano = find_nano(stem, 'train')
        assert os.path.isfile(low) and os.path.isfile(high)
        train_rows.append(row('Nikon/%s.jpg' % stem, 'train', low, high, nano))
    train_rows.sort(key=lambda r: r['sample_id'])

    # --- eval ---
    eval_rows = []
    low_dir = os.path.join(DATA, 'Eval/Nikon/low')
    for name in sorted(os.listdir(low_dir)):
        if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = os.path.splitext(name)[0]
        low = os.path.join(low_dir, name)
        high = os.path.join(DATA, 'Eval/Nikon/high', name)
        if not os.path.isfile(high):
            high = os.path.join(DATA, 'Eval/Nikon/high', stem + '.jpg')
        nano = find_nano(stem, 'eval')
        eval_rows.append(row('Nikon/%s.jpg' % stem, 'eval', low, high, nano))

    # also write filtered Nikon.txt for main.py-style loaders
    txt_path = os.path.join(OUT, 'Nikon.txt')
    with open(txt_path, 'w') as f:
        for r in train_rows:
            f.write(r['low_path'] + '\n')

    train_csv = os.path.join(OUT, 'manifests_train.csv')
    eval_csv = os.path.join(OUT, 'manifests_eval.csv')
    for path, rows in ((train_csv, train_rows), (eval_csv, eval_rows)):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    identity = dict(
        camera='Nikon',
        train_n=len(train_rows),
        eval_n=len(eval_rows),
        man500_source=MAN500,
        dropped_eval_stems=dropped,
        train_csv=train_csv,
        eval_csv=eval_csv,
        train_sha256=sha256_file(train_csv),
        eval_sha256=sha256_file(eval_csv),
        note='Train excludes Eval stems present in man500 (3004/3016/3017)',
    )
    with open(os.path.join(OUT, 'manifest_identity.json'), 'w') as f:
        json.dump(identity, f, indent=2)
    print(json.dumps(identity, indent=2))
    assert len(eval_rows) == 20, len(eval_rows)
    assert len(train_rows) == 497, len(train_rows)
    print('OK train=%d eval=%d dropped=%s' % (
        len(train_rows), len(eval_rows), dropped))


if __name__ == '__main__':
    main()
