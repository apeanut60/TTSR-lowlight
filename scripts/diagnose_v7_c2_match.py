#!/usr/bin/env python
"""Gate1: C2 match correctness — correct vs shuffled Ref (P/C stats + warp)."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.td_v6_adapter import V7TaskDecoupledModel, _rgb01  # noqa: E402
from model.V5FlowAlign import warp_by_flow  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a6_runtime import dump_json  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402
from v7_runtime import (CACHE_NAME, MISMATCH_TRAIN, REFERENCE_VARIANT, ROOT,  # noqa: E402
                        SRC, SPLIT_JSON, C2_CKPT)


def _stats(name, conf, margin, flow):
    c = conf.detach().float().cpu().numpy()
    m = margin.detach().float().cpu().numpy()
    f = flow.detach().float().cpu().numpy()
    return dict(
        tag=name,
        conf_mean=float(c.mean()), conf_p50=float(np.median(c)),
        conf_p90=float(np.percentile(c, 90)),
        margin_mean=float(m.mean()),
        flow_abs_mean=float(np.abs(f).mean()),
        flow_abs_p90=float(np.percentile(np.abs(f), 90)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args(_CLI)

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    lock = json.load(open(lock_path)) if os.path.isfile(lock_path) else {}
    c2 = lock.get('c2_ckpt', C2_CKPT)

    model = V7TaskDecoupledModel(c2_ckpt=c2).to(a.device).eval()
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = REFERENCE_VARIANT
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       SPLIT_JSON)
    mmap_path = lock.get('mismatch_train', MISMATCH_TRAIN)
    mmap = json.load(open(mmap_path))
    # use train cache; diagnose on first n train rows (same Y0)
    y0_dir = os.path.join(a.root, CACHE_NAME, 'refiner_train')
    rows = splits['train'][:a.n]
    ds = make_dataset(ns, rows, y0_dir, mmap)

    rows_out = []
    with torch.no_grad():
        for i in range(len(rows)):
            t = sample_tensors(ds, i, 'correct', a.device)
            t_mis = sample_tensors(ds, i, 'mismatch', a.device)
            y0 = _rgb01(t['Y0'])
            ref = _rgb01(t['R'])
            ref_mis = _rgb01(t_mis['R'])
            ref_roll = torch.roll(ref, shifts=(ref.shape[-2] // 3,
                                               ref.shape[-1] // 3), dims=(-2, -1))
            out_ok = model(y0, ref)
            out_mis = model(y0, ref_mis)
            out_roll = model(y0, ref_roll)
            out_self = model(y0, y0)

            ref_h4 = F.interpolate(ref, size=out_ok['flow_h4'].shape[-2:],
                                   mode='bilinear', align_corners=False)
            y0_h4 = F.interpolate(y0, size=out_ok['flow_h4'].shape[-2:],
                                  mode='bilinear', align_corners=False)
            warp_ok = warp_by_flow(ref_h4, out_ok['flow_h4'])
            s_ok = _stats('correct', out_ok['raw_confidence_h4'],
                          out_ok['margin_h4'], out_ok['flow_h4'])
            s_mis = _stats('mismatch', out_mis['raw_confidence_h4'],
                           out_mis['margin_h4'], out_mis['flow_h4'])
            s_roll = _stats('roll', out_roll['raw_confidence_h4'],
                            out_roll['margin_h4'], out_roll['flow_h4'])
            s_sf = _stats('self', out_self['raw_confidence_h4'],
                          out_self['margin_h4'], out_self['flow_h4'])
            rec = dict(
                name=t['name'],
                correct=s_ok, mismatch=s_mis, roll=s_roll, self_ref=s_sf,
                delta_conf_correct_minus_mismatch=(
                    s_ok['conf_mean'] - s_mis['conf_mean']),
                delta_conf_correct_minus_roll=(
                    s_ok['conf_mean'] - s_roll['conf_mean']),
                self_flow_abs_mean=s_sf['flow_abs_mean'],
                warp_l1_to_y0_h4=float(F.l1_loss(warp_ok, y0_h4)),
            )
            rows_out.append(rec)
            print('%s  C_ok=%.3f C_mis=%.3f C_roll=%.3f C_self=%.3f  '
                  'dC_mis=%.3f  self|f|=%.3f'
                  % (t['name'], s_ok['conf_mean'], s_mis['conf_mean'],
                     s_roll['conf_mean'], s_sf['conf_mean'],
                     rec['delta_conf_correct_minus_mismatch'],
                     s_sf['flow_abs_mean']), flush=True)

    mean_dc = float(np.mean(
        [r['delta_conf_correct_minus_mismatch'] for r in rows_out]))
    mean_self_f = float(np.mean([r['self_flow_abs_mean'] for r in rows_out]))
    report = dict(
        n=len(rows_out),
        mean_delta_conf_correct_minus_mismatch=mean_dc,
        mean_delta_conf_correct_minus_roll=float(np.mean(
            [r['delta_conf_correct_minus_roll'] for r in rows_out])),
        mean_self_flow_abs=mean_self_f,
        gate1_hint=(
            'OK: correct>mismatch conf; self flow near 0'
            if mean_dc > 0.02 and mean_self_f < 2.0 else
            'WEAK: C2 conf barely prefers correct vs mismatch — '
            'smoke allowed; formal A1 needs review'),
        rows=rows_out,
    )
    out_path = os.path.join(a.root, 'diagnostics', 'gate1_c2_match.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump_json(out_path, json_ready(report))
    print('=== Gate1 report %s ===' % out_path, flush=True)
    print('  mean ΔC(correct-mismatch)=%.4f  mean self|flow|=%.4f'
          % (mean_dc, mean_self_f), flush=True)
    print('  %s' % report['gate1_hint'], flush=True)


if __name__ == '__main__':
    main()
