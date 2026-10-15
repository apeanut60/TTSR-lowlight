#!/usr/bin/env python
"""v2.6: Low/GT/Nano share crop coords from schedule."""
from __future__ import annotations

import csv
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from local_refine_runtime import read_manifest_csv  # noqa: E402
from train_v26_random_crop import ScheduleCropSet, load_schedule  # noqa: E402

SCHED = '/root/data/experiments/TTSR-lowlight-v26/manifests/random_crop_s42.csv'
V2 = '/root/data/experiments/TTSR-lowlight-v2'


def test_aligned_crops():
    tr = read_manifest_csv(os.path.join(V2, 'manifests/manifests_train.csv'))
    by_id = {r['sample_id']: r for r in tr}
    ds = ScheduleCropSet(load_schedule(SCHED)[:3], by_id)
    for i in range(3):
        b = ds[i]
        assert b['low'].shape == b['high'].shape == b['nano'].shape
        assert b['low'].shape[-2:] == (192, 192)
        print('step', i + 1, b['sample_id'], 'crop', b['crop_y'], b['crop_x'],
              'shape', tuple(b['low'].shape))


if __name__ == '__main__':
    test_aligned_crops()
    print('test_v26_alignment OK')
