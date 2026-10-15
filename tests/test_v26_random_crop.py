#!/usr/bin/env python
"""v2.6: schedule gives different crops across visits; C1/C2 share schedule."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from train_v25_loss_balance import stable_crop_xy  # noqa: E402

SCHED = '/root/data/experiments/TTSR-lowlight-v26/manifests/random_crop_s42.csv'


def test_v25_fixed_vs_v26_dynamic():
    # v2.5 fixed
    a = stable_crop_xy('Huawei/00001.jpg', 42, 720, 960, 192)
    b = stable_crop_xy('Huawei/00001.jpg', 42, 720, 960, 192)
    assert a == b
    rows = list(csv.DictReader(open(SCHED)))
    crops = [(int(r['crop_y']), int(r['crop_x']))
             for r in rows if r['sample_id'] == 'Huawei/00001.jpg']
    assert len(crops) >= 2
    assert len(set(crops)) == len(crops), crops
    print('v2.5 fixed=%s; v2.6 00001 crops=%s' % (a, crops))


def test_coverage_all_images():
    rows = list(csv.DictReader(open(SCHED)))
    assert len(rows) == 2000
    per = defaultdict(set)
    for r in rows:
        per[r['sample_id']].add((r['crop_y'], r['crop_x']))
    assert len(per) == 500
    assert all(len(v) == 4 for v in per.values())
    print('coverage OK: 500 images × 4 unique crops')


if __name__ == '__main__':
    test_v25_fixed_vs_v26_dynamic()
    test_coverage_all_images()
    print('test_v26_random_crop OK')
