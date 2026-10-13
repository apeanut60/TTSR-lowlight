#!/usr/bin/env python
"""V7-RIVER Official Test ablation: Base/normal/self/zero/shuffled.

Selection criterion (user): Gain_ref / Dependency on Official Test — not dev64.
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

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest  # noqa: E402
from local_refine_runtime import metrics as _metrics  # noqa: E402
from model.V6LLFormerRefBridge import (base_forward_m11,  # noqa: E402
                                       load_frozen_llformer_base, tiled_v6_forward)
from model.V7Model import V7Model  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a6_runtime import dump_json, nanmean  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402
from v7_river_runtime import ROOT, SRC, VALIDATION_ROOT, setting_root

REF_MODES = ('normal', 'self', 'zero', 'shuffled')


def _ref_for_mode(y0, ref, donor, mode):
    if mode == 'normal':
        return ref
    if mode == 'self':
        return y0
    if mode == 'zero':
        return torch.zeros_like(ref)
    if mode == 'shuffled':
        return donor
    raise SystemExit('unknown mode %r' % mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--setting', default='A', choices=['A', 'B'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke_ckpt', action='store_true')
    a = ap.parse_args(_CLI)

    arm_root = setting_root(a.setting, a.root)
    lock = json.load(open(os.path.join(arm_root, 'artifact_lock.json')))
    ckpt_dir = 'smoke_checkpoints' if a.smoke_ckpt else 'checkpoints'
    path = os.path.join(arm_root, ckpt_dir, 'ckpt_%06d.pt' % a.step)
    blob = torch.load(path, map_location='cpu')
    model = V7Model().to(a.device).eval()
    model.load_state_dict(blob['model'], strict=True)
    model.apply_setting_freeze(a.setting)
    for p in model.parameters():
        p.requires_grad_(False)

    base = load_frozen_llformer_base(lock['base_ckpt'], a.device)
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    test_manifest = os.path.join(a.src_root, 'manifests', 'test.csv')
    pairs = pairs_from_manifest(test_manifest)
    y0_cache = os.path.join(a.root, lock['cache_name'], 'test')
    if not os.path.isdir(y0_cache):
        raise SystemExit('missing Test Y0 cache: %s' % y0_cache)
    ds = TrainSet(ns, crop_size=0, pairs=pairs, y0_cache=y0_cache,
                  mismatch_map={}, split='Test')
    n = len(pairs) if not a.limit else min(a.limit, len(pairs))

    rows = []
    print('=== V7-RIVER Official Test setting=%s @%d n=%d ==='
          % (a.setting, a.step, n), flush=True)
    with torch.no_grad():
        for i in range(n):
            name, lr, hr, ref, y0, _ = ds._load(i)
            j = (i + 1) % n
            _, _, _, donor, _, _ = ds._load(j)
            x = lr[None].to(a.device)
            ht = hr[None].to(a.device)
            y0t = y0[None].to(a.device)
            rt = ref[None].to(a.device)
            don = donor[None].to(a.device)
            y_base = base_forward_m11(base, x)
            base_psnr, base_ssim, _ = _metrics(y_base, ht)
            rec = dict(name=name, base_psnr=base_psnr, base_ssim=base_ssim)
            for mode in REF_MODES:
                r = _ref_for_mode(y0t, rt, don, mode)
                y = tiled_v6_forward(base, model, x, y0t, r)
                psnr, ssim, _ = _metrics(y, ht)
                rec['%s_psnr' % mode] = psnr
                rec['%s_ssim' % mode] = ssim
                rec['%s_dpsnr' % mode] = psnr - base_psnr
            rows.append(rec)
            if (i + 1) % 10 == 0 or i == 0:
                print('  [%d] %s base=%.3f normal_d=%.3f'
                      % (i, name, base_psnr, rec['normal_dpsnr']), flush=True)

    summary = dict(
        split='official_Test', setting=a.setting.upper(), step=a.step, n=n,
        ckpt=path,
    )
    summary['base_psnr_mean'] = float(nanmean([r['base_psnr'] for r in rows]))
    for mode in REF_MODES:
        d = [r['%s_dpsnr' % mode] for r in rows]
        summary['%s_psnr_mean' % mode] = float(nanmean(
            [r['%s_psnr' % mode] for r in rows]))
        summary['%s_dpsnr_mean' % mode] = float(nanmean(d))
        summary['%s_dpsnr_p10' % mode] = float(np.percentile(d, 10))
    summary['Gain_ref'] = (
        summary['normal_psnr_mean'] - summary['zero_psnr_mean'])
    summary['Dependency'] = (
        summary['normal_psnr_mean'] - summary['shuffled_psnr_mean'])
    summary['normal_minus_self'] = (
        summary['normal_psnr_mean'] - summary['self_psnr_mean'])

    if summary['Gain_ref'] >= 0.05 and summary['Dependency'] >= 0.05:
        summary['verdict'] = 'PASS_case3'
    elif summary['Gain_ref'] > 0.01:
        summary['verdict'] = 'PARTIAL_case2'
    else:
        summary['verdict'] = 'FAIL_case1_ref_unused'

    out_dir = os.path.join(arm_root, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    dump_json(os.path.join(out_dir, 'test_ablation_%06d.json' % a.step),
              json_ready(dict(summary=summary, rows=rows)))

    # append CSV at validation results/
    csv_path = os.path.join(VALIDATION_ROOT, 'results', 'reference_ablation.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'setting', 'step', 'n', 'base_psnr', 'normal_psnr', 'zero_psnr',
            'shuffled_psnr', 'self_psnr', 'Gain_ref', 'Dependency', 'verdict',
        ])
        if write_header:
            w.writeheader()
        w.writerow(dict(
            setting=a.setting.upper(), step=a.step, n=n,
            base_psnr=summary['base_psnr_mean'],
            normal_psnr=summary['normal_psnr_mean'],
            zero_psnr=summary['zero_psnr_mean'],
            shuffled_psnr=summary['shuffled_psnr_mean'],
            self_psnr=summary['self_psnr_mean'],
            Gain_ref=summary['Gain_ref'],
            Dependency=summary['Dependency'],
            verdict=summary['verdict'],
        ))

    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    main()
