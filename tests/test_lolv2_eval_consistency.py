"""RGB metric entry points agree, and Official Test is not the early-stop set."""
from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from eval_nikon_export import channel_metrics, score_rgb_tensors, to_uint8  # noqa: E402
from lolv2_split import (  # noqa: E402
    assert_schedule_in_fit, load_official_test, prepare_fit_val, select_monitor_rows,
)


def _rows(ids):
    out = []
    for i in ids:
        out.append(dict(
            sample_id=i, camera='LOLv2Syn', split='train',
            low_path=i, gt_path=i, high_path=i, nano_path=i,
            low_hw='8x8', gt_hw='8x8', nano_hw='8x8',
            low_dtype='uint8', gt_dtype='uint8', nano_dtype='uint8',
            decode_status='ok',
        ))
    return out


def _write_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_rgb_entries_match_and_are_not_y_ssim():
    rng = np.random.RandomState(0)
    pred = torch.from_numpy(rng.rand(1, 3, 32, 28).astype(np.float32) * 2 - 1)
    gt = torch.from_numpy(rng.rand(1, 3, 32, 28).astype(np.float32) * 2 - 1)
    a = score_rgb_tensors(pred, gt)
    b = channel_metrics(to_uint8(pred), to_uint8(gt))
    assert a['psnr_rgb'] == b['psnr_rgb']
    assert a['ssim_rgb'] == b['ssim_rgb']
    mean_c = (a['ssim_R'] + a['ssim_G'] + a['ssim_B']) / 3.0
    assert abs(a['ssim_rgb'] - mean_c) < 1e-6
    assert a['ssim_rgb'] != a['ssim_y']


def test_val_split_blocks_official_test():
    root = tempfile.mkdtemp()
    try:
        ids = ['r%03d' % i for i in range(20)]
        rows = _rows(ids)
        _write_manifest(os.path.join(root, 'manifests_train.csv'), rows)
        # A file that must not be required for val selection.
        fit, val, meta = prepare_fit_val(rows, root, seed=7, val_ratio=0.1)
        fit_ids = {r['sample_id'] for r in fit}
        val_ids = {r['sample_id'] for r in val}
        assert not (fit_ids & val_ids)
        assert fit_ids | val_ids == set(ids)
        assert meta['scene_isolation'] == 'unconfirmed'
        monitor, name = select_monitor_rows(root, 'val', False, val)
        assert name == 'val'
        assert {r['sample_id'] for r in monitor} == val_ids
        try:
            select_monitor_rows(root, 'test', True, val)
            assert False, 'test split must not be a monitor set'
        except SystemExit:
            pass
        try:
            load_official_test(root, 'val', True)
            assert False
        except SystemExit as exc:
            assert 'blocked' in str(exc).lower() or 'Official' in str(exc)
        try:
            load_official_test(root, 'test', False)
            assert False
        except SystemExit:
            pass
        assert not os.path.isfile(os.path.join(root, 'manifests_eval.csv'))
        leaked = [{'sample_id': val[0]['sample_id']}]
        try:
            assert_schedule_in_fit(leaked, fit_ids, val_ids)
            assert False
        except SystemExit as exc:
            assert 'validation' in str(exc)
        fit2, val2, _ = prepare_fit_val(rows, root, seed=7, val_ratio=0.1)
        assert [r['sample_id'] for r in fit2] == [r['sample_id'] for r in fit]
        assert [r['sample_id'] for r in val2] == [r['sample_id'] for r in val]
        try:
            prepare_fit_val(rows, root, seed=8, val_ratio=0.1)
            assert False
        except SystemExit as exc:
            assert 'seed' in str(exc)
    finally:
        shutil.rmtree(root)


def test_checkpoint_keys_load():
    path = (
        '/root/data/experiments/TTSR-lowlight-lolv2syn/checkpoints/'
        'S1c_nano_lr5e6_fromS1b1100_s44/checkpoint_00400.pt')
    if not os.path.isfile(path):
        return
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'p2' in state:
        state = state['p2']
    assert any(k.startswith('MainNet.global_illum_head.') for k in state)
    assert any(k.startswith('MainNet.ref_illum.') for k in state)
    assert any(k.startswith('RefCorrection.') for k in state)
    assert any(k.startswith('MainNet.SFE.') for k in state)
