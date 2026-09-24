#!/usr/bin/env python
"""Five-condition evaluation for the decoder+texture paired plan (section F2).

    N0                frozen base, texture off
    D0                decoder-tail fine-tuned, no reference (the matched control)
    D1_correct        decoder-tail + A3, correct generated references (candidate)
    D1_a3_off         D1 weights with the texture path bypassed (dependency probe;
                      NOT expected to equal N0 -- the decoder has moved)
    D1_mismatch       D1 weights with a fixed within-camera derangement of the
                      references (correspondence probe)

Both reference settings are reported: `nano` (generated, main result) and `hr`
diagnostic. `--ref_dir` must be set per camera -- an empty ref_dir makes
data1.TestSet silently fall back to HR, which is how the previous round's
table got mislabelled.

Usage
    python scripts/eval_decoder_texture_pair.py \
      --n0  <N0 ep40> --d0 <D0 step3000> --d1 <D1 step3000> \
      --run_dir <D1 run dir> --out <per_image.csv>
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0]]      # option.py parses sys.argv at import time

import model.TTSREnhance as TTE                               # noqa: E402
from dataset import data1                                     # noqa: E402
from option import parser as option_parser                    # noqa: E402
from trainer import Trainer                                   # noqa: E402
from utils import calc_psnr_and_ssim                          # noqa: E402


def load_args_from_run(run_dir):
    ns = option_parser.parse_args([])
    for line in open(os.path.join(run_dir, 'args.txt'), encoding='utf-8'):
        parts = line.split()
        if len(parts) < 2:
            continue
        key, val = parts[0], parts[1]
        if val in ('True', 'False'):
            val = (val == 'True')
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        setattr(ns, key, val)
    return ns


def build(cfg, ckpt, device):
    model = TTE.TTSREnhance(cfg).to(device)
    model = TTE.load_pretrained_weights(model, ckpt, device)
    model.eval()
    quiet = logging.getLogger('eval_pair_%d' % id(model))
    quiet.addHandler(logging.NullHandler())
    quiet.setLevel(logging.CRITICAL)
    return model, Trainer(cfg, quiet, None, model, {})


def derangement(n):
    assert n >= 2
    return [(i + 1) % n for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n0', required=True)
    ap.add_argument('--d0', required=True)
    ap.add_argument('--d1', required=True)
    ap.add_argument('--run_dir',
                    default='/root/data/experiments/retinex_v11/D1_decoder_texture_s42')
    ap.add_argument('--dataset_dir', default='/root/data/datasets/data1')
    ap.add_argument('--out', default='')
    args = ap.parse_args(_CLI_ARGV)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = load_args_from_run(args.run_dir)
    cfg.cpu = False
    cfg.num_gpu = 1
    cfg.dataset_dir = args.dataset_dir
    cfg.no_ref_illum = True
    cfg.ref_correction = False
    cfg.no_global_illum = True

    NANO_REF = {
        'Huawei': getattr(cfg, 'data1_nanobanana_eval_huawei_ref_dir', ''),
        'Nikon': getattr(cfg, 'data1_nanobanana_eval_nikon_ref_dir', ''),
    }
    for cam, d in NANO_REF.items():
        if not d or not os.path.isdir(d):
            raise SystemExit('missing Nano reference dir for %s: %r' % (cam, d))

    def make_set(camera, ref_dir):
        a = option_parser.parse_args([])
        a.dataset_dir = args.dataset_dir
        a.ref_degrade = False
        a.data1_camera = camera
        a.ref_dir = ref_dir
        return data1.TestSet(args=a, ref_level='1')

    SETTINGS = {'nano': {c: make_set(c, NANO_REF[c]) for c in NANO_REF},
                'hr': {c: make_set(c, '') for c in NANO_REF}}
    for s, per in SETTINGS.items():
        print('%s: %s' % (s, {c: len(v) for c, v in sorted(per.items())}))

    models = {}
    for tag, path in (('N0', args.n0), ('D0', args.d0), ('D1', args.d1)):
        models[tag] = build(cfg, path, device)

    # (label, model tag, texture on, use permuted reference)
    CONDS = [
        ('N0', 'N0', False, False),
        ('D0', 'D0', False, False),
        ('D1_correct', 'D1', True, False),
        ('D1_a3_off', 'D1', False, False),
        ('D1_mismatch', 'D1', True, True),
    ]
    keys = [c[0] for c in CONDS]

    rows = []
    with torch.no_grad():
        for sname, per_cam in SETTINGS.items():
            for cam, ds in sorted(per_cam.items()):
                perm = derangement(len(ds))
                for i in range(len(ds)):
                    s = ds[i]
                    lr = torch.as_tensor(s['LR']).unsqueeze(0).float().to(device)
                    hr = torch.as_tensor(s['HR']).unsqueeze(0).float().to(device)
                    sp = ds[perm[i]]
                    row = dict(ref_setting=sname,
                               sample_id=os.path.basename(ds.pairs[i][0]),
                               camera=cam)
                    for label, mtag, texture, use_perm in CONDS:
                        model, tr = models[mtag]
                        if use_perm:
                            r = torch.as_tensor(sp['Ref']).unsqueeze(0).float().to(device)
                            rs = torch.as_tensor(sp['Ref_sr']).unsqueeze(0).float().to(device)
                        else:
                            r = torch.as_tensor(s['Ref']).unsqueeze(0).float().to(device)
                            rs = torch.as_tensor(s['Ref_sr']).unsqueeze(0).float().to(device)
                        model.args.no_ref_texture = (not texture)
                        sr = tr._tiled_forward(lr, lr, r, rs)[0]
                        row[label] = float(calc_psnr_and_ssim(sr, hr)[3])
                    rows.append(row)

    for sname in SETTINGS:
        sel_all = [r for r in rows if r['ref_setting'] == sname]
        print()
        print('══ reference setting: %s ══' % sname)
        print('  %-10s %s' % ('camera', ' '.join('%11s' % k for k in keys)))
        for cam in sorted(set(r['camera'] for r in sel_all)) + ['ALL', '0.6HW+0.4NK']:
            if cam == '0.6HW+0.4NK':
                val = {k: 0.6 * np.mean([r[k] for r in sel_all if r['camera'] == 'Huawei'])
                          + 0.4 * np.mean([r[k] for r in sel_all if r['camera'] == 'Nikon'])
                       for k in keys}
            else:
                s2 = sel_all if cam == 'ALL' else [r for r in sel_all if r['camera'] == cam]
                val = {k: np.mean([r[k] for r in s2]) for k in keys}
            print('  %-10s %s' % (cam, ' '.join('%11.3f' % val[k] for k in keys)))
        proj = {k: 0.6 * np.mean([r[k] for r in sel_all if r['camera'] == 'Huawei'])
                   + 0.4 * np.mean([r[k] for r in sel_all if r['camera'] == 'Nikon'])
                for k in keys}
        print('  Δ vs N0  : %s' % ' '.join('%+11.4f' % (proj[k] - proj['N0']) for k in keys))
        print('  Δ vs D0  : %s' % ' '.join('%+11.4f' % (proj[k] - proj['D0']) for k in keys))
        for k in ['D0', 'D1_correct']:
            d = np.array([r[k] - r['N0'] for r in sel_all])
            print('  per-image %s−N0: mean %+.4f median %+.4f better %d/%d worst %+.3f'
                  % (k, d.mean(), np.median(d), int((d > 0).sum()), len(d), d.min()))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['ref_setting', 'sample_id', 'camera'] + keys)
            w.writeheader()
            w.writerows(rows)
        print('\nper-image metrics -> %s' % args.out)


if __name__ == '__main__':
    main()
