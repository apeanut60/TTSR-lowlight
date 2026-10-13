#!/usr/bin/env python
"""V7 eval: Base Y0 vs TD+C2 Ref; modes normal/self/zero/shuffled on dev."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics as _metrics  # noqa: E402
from model.td_v6_adapter import V7TaskDecoupledModel  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a6_runtime import dump_json, nanmean  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402
from v7_runtime import ARM_A1, ROOT, SRC

REF_MODES = ('normal', 'self', 'zero', 'shuffled')


def _to_m11(x01: torch.Tensor) -> torch.Tensor:
    return x01.clamp(0, 1) * 2.0 - 1.0


def _ref_for_mode(t, mode, donor_r):
    if mode == 'normal':
        return t['R']
    if mode == 'self':
        return t['Y0']
    if mode == 'zero':
        return torch.zeros_like(t['R'])
    if mode == 'shuffled':
        return donor_r
    raise SystemExit('unknown ref mode %r' % mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--full_dep', action='store_true')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    path = os.path.join(a.root, ARM_A1, 'checkpoints', 'ckpt_%06d.pt' % a.step)
    blob = torch.load(path, map_location='cpu')
    model = V7TaskDecoupledModel(c2_ckpt=lock['c2_ckpt']).to(a.device).eval()
    model.load_state_dict(blob['model'], strict=True)

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       lock['split_json'])
    mmap = json.load(open(lock['mismatch_dev']))
    ds = make_dataset(
        ns, splits['dev'],
        os.path.join(a.root, lock['cache_name'], 'refiner_train'), mmap)
    n = len(splits['dev']) if not a.limit else min(a.limit, len(splits['dev']))
    modes = REF_MODES if a.full_dep else ('normal',)

    rows = []
    print('=== V7 eval @%d n=%d modes=%s ===' % (a.step, n, modes), flush=True)
    with torch.no_grad():
        for i in range(n):
            t = sample_tensors(ds, i, 'correct', a.device)
            # shuffled donor: next image's correct Ref
            j = (i + 1) % n
            donor = sample_tensors(ds, j, 'correct', a.device)['R']
            base_psnr, base_ssim, _ = _metrics(t['Y0'], t['H'])
            rec = dict(name=t['name'], base_psnr=base_psnr, base_ssim=base_ssim)
            for mode in modes:
                r = _ref_for_mode(t, mode, donor)
                out = model.forward_m11(t['Y0'], r)
                y_m11 = _to_m11(out['y'])
                psnr, ssim, _ = _metrics(y_m11, t['H'])
                rec['%s_psnr' % mode] = psnr
                rec['%s_ssim' % mode] = ssim
                rec['%s_dpsnr' % mode] = psnr - base_psnr
                rec['%s_c_mean' % mode] = float(
                    out['raw_confidence_h4'].mean().cpu())
                rec['%s_itex_abs' % mode] = float(out['i_tex'].abs().mean().cpu())
            rows.append(rec)
            if (i + 1) % 8 == 0 or i == 0:
                print('  [%d] %s base=%.3f normal_d=%.3f'
                      % (i, t['name'], rec['base_psnr'],
                         rec.get('normal_dpsnr', float('nan'))), flush=True)

    summary = dict(step=a.step, n=n, modes=list(modes))
    for mode in modes:
        d = [r['%s_dpsnr' % mode] for r in rows]
        summary['%s_dpsnr_mean' % mode] = float(nanmean(d))
        summary['%s_dpsnr_p10' % mode] = float(np.percentile(d, 10))
        summary['%s_psnr_mean' % mode] = float(nanmean(
            [r['%s_psnr' % mode] for r in rows]))
        summary['%s_c_mean' % mode] = float(nanmean(
            [r['%s_c_mean' % mode] for r in rows]))
    summary['base_psnr_mean'] = float(nanmean([r['base_psnr'] for r in rows]))
    if 'normal' in modes and 'shuffled' in modes:
        summary['normal_minus_shuf'] = (
            summary['normal_dpsnr_mean'] - summary['shuffled_dpsnr_mean'])
        summary['normal_minus_zero'] = (
            summary['normal_dpsnr_mean'] - summary['zero_dpsnr_mean'])
        summary['normal_minus_self'] = (
            summary['normal_dpsnr_mean'] - summary['self_dpsnr_mean'])
        large_harm = sum(1 for r in rows if r['shuffled_dpsnr'] < -0.5) / n
        summary['shuf_large_harm_rate'] = float(large_harm)

    out_dir = os.path.join(a.root, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    dump_json(os.path.join(out_dir, 'eval_step_%06d.json' % a.step),
              json_ready(dict(summary=summary, rows=rows)))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
