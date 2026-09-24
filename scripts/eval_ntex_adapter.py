#!/usr/bin/env python
"""Four-condition evaluation for the Ntex adapter experiment (plan section 8).

    E0  base N0,        texture off   -> the frozen baseline
    E1  step3000,       texture off   -> must equal E0 (guards against drift)
    E2  step3000,       texture on, correct Nano references    -> the effect
    E3  step3000,       texture on, mismatched Nano references -> sensitivity

E3 permutes the *reference* within each camera block using a fixed cyclic
shift with no fixed points. low/high are untouched and Ref/Ref_sr move
together, so the only thing that changes is whether the reference corresponds
to the input. This is an inference-sensitivity diagnostic; it does not replace
a capacity-matched no-reference training control.

All four conditions see the identical 50 eval images and the identical
evaluator, so the only differences are the weights and the reference tensor.

Usage
    python scripts/eval_ntex_adapter.py \
        --n0 /root/data/experiments/retinex_v11/N0_fixed_s42/model/model_00040.pt \
        --trained /root/data/experiments/retinex_v11/Ntex_adapter_s42/model/model_03000.pt \
        --out /root/data/experiments/retinex_v11/Ntex_adapter_s42/per_image.csv
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# option.py runs `args = parser.parse_args()` at import time, so it would choke
# on this script's own flags. Stash them, hide them from that import, then parse
# them ourselves.
_CLI_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset import data1                                    # noqa: E402
import model.TTSREnhance as TTE                               # noqa: E402
from option import parser as option_parser                    # noqa: E402
from trainer import Trainer                                   # noqa: E402
from utils import calc_psnr_and_ssim                          # noqa: E402


def load_args_from_run(run_dir):
    """Rebuild the run's Namespace from its args.txt snapshot."""
    ns = option_parser.parse_args([])
    args_txt = os.path.join(run_dir, 'args.txt')
    if not os.path.isfile(args_txt):
        raise SystemExit('no args.txt in %s' % run_dir)
    for line in open(args_txt, encoding='utf-8'):
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


def build(args, ckpt, device):
    """Model + a Trainer, so tiled inference reuses exactly the eval path the
    official evaluator uses (960x720 is far too large for a single forward with
    the SearchTransfer similarity matrix)."""
    model = TTE.TTSREnhance(args).to(device)
    model = TTE.load_pretrained_weights(model, ckpt, device)
    model.eval()
    quiet = logging.getLogger('eval_ntex_%d' % id(model))
    quiet.addHandler(logging.NullHandler())
    quiet.setLevel(logging.CRITICAL)
    return model, Trainer(args, quiet, None, model, {})


def as_batch(t):
    return torch.as_tensor(t).unsqueeze(0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n0', required=True, help='frozen base checkpoint')
    ap.add_argument('--trained', required=True, help='checkpoint after adapter training')
    ap.add_argument('--run_dir', default='/root/data/experiments/retinex_v11/Ntex_adapter_s42')
    ap.add_argument('--dataset_dir', default='/root/data/datasets/data1')
    ap.add_argument('--out', default='')
    args = ap.parse_args(_CLI_ARGV)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = load_args_from_run(args.run_dir)
    cfg.dataset_dir = args.dataset_dir
    cfg.no_reference = False
    cfg.no_ref_illum = True
    cfg.ref_correction = False
    cfg.no_global_illum = True
    cfg.cpu = False
    cfg.num_gpu = 1
    cfg.tile_size = getattr(cfg, 'tile_size', 256)

    # Two reference settings. `hr` is the diagnostic (Ref = HR, what an empty
    # --ref_dir gives); `nano` is the deployment-like main result and needs the
    # external generated-reference directories -- an empty ref_dir makes
    # data1.TestSet silently fall back to HR, so this must be explicit.
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

    SETTINGS = {
        'nano': {cam: make_set(cam, NANO_REF[cam]) for cam in NANO_REF},
        'hr': {cam: make_set(cam, '') for cam in NANO_REF},
    }
    for sname, per_cam in SETTINGS.items():
        print('%s: %s' % (sname, {c: len(s) for c, s in sorted(per_cam.items())}))

    def derangement(n):
        """Cyclic shift with no fixed point (n >= 2)."""
        assert n >= 2
        return [(i + 1) % n for i in range(n)]

    m_base, t_base = build(cfg, args.n0, device)
    m_tr, t_tr = build(cfg, args.trained, device)

    rows = []
    with torch.no_grad():
        for sname, per_cam in SETTINGS.items():
            for cam, ds in sorted(per_cam.items()):
                perm = derangement(len(ds))
                for i in range(len(ds)):
                    s = ds[i]
                    lr = as_batch(s['LR']).to(device)
                    hr = as_batch(s['HR']).to(device)
                    ref = as_batch(s['Ref']).to(device)
                    refsr = as_batch(s['Ref_sr']).to(device)
                    sp = ds[perm[i]]
                    ref_p = as_batch(sp['Ref']).to(device)
                    refsr_p = as_batch(sp['Ref_sr']).to(device)

                    def psnr(model, trainer, texture, r, rs):
                        model.args.no_ref_texture = (not texture)
                        sr = trainer._tiled_forward(lr, lr, r, rs)[0]
                        return float(calc_psnr_and_ssim(sr, hr)[3])

                    rows.append(dict(
                        ref_setting=sname,
                        sample_id=os.path.basename(ds.pairs[i][0]), camera=cam,
                        e0_base_noref=psnr(m_base, t_base, False, ref, refsr),
                        e1_step3000_noref=psnr(m_tr, t_tr, False, ref, refsr),
                        e2_correct=psnr(m_tr, t_tr, True, ref, refsr),
                        e3_mismatch=psnr(m_tr, t_tr, True, ref_p, refsr_p),
                    ))

    keys = ['e0_base_noref', 'e1_step3000_noref', 'e2_correct', 'e3_mismatch']
    for sname in SETTINGS:
        sel_all = [r for r in rows if r['ref_setting'] == sname]
        print()
        print('══ reference setting: %s ══' % sname)
        print('  %-10s %10s %10s %10s %10s' % ('camera', *keys))
        for cam in sorted(set(r['camera'] for r in sel_all)) + ['ALL', '0.6HW+0.4NK']:
            if cam == 'ALL':
                sel = sel_all
            elif cam == '0.6HW+0.4NK':
                sel = None
            else:
                sel = [r for r in sel_all if r['camera'] == cam]
            if sel is None:
                val = {k: 0.6 * np.mean([r[k] for r in sel_all if r['camera'] == 'Huawei'])
                          + 0.4 * np.mean([r[k] for r in sel_all if r['camera'] == 'Nikon'])
                       for k in keys}
            else:
                val = {k: np.mean([r[k] for r in sel]) for k in keys}
            print('  %-10s %s' % (cam, ' '.join('%10.3f' % val[k] for k in keys)))
        proj = {k: 0.6 * np.mean([r[k] for r in sel_all if r['camera'] == 'Huawei'])
                   + 0.4 * np.mean([r[k] for r in sel_all if r['camera'] == 'Nikon'])
                for k in keys}
        print('  E1 − E0 (should be ~0)   : %+.4f' % (proj[keys[1]] - proj[keys[0]]))
        print('  E2 − E0 (main result)    : %+.4f' % (proj[keys[2]] - proj[keys[0]]))
        print('  E2 − E3 (correspondence) : %+.4f' % (proj[keys[2]] - proj[keys[3]]))
        d = np.array([r[keys[2]] - r[keys[0]] for r in sel_all])
        print('  per-image E2−E0: mean %+.4f median %+.4f better %d/%d worst %+.3f'
              % (d.mean(), np.median(d), int((d > 0).sum()), len(d), d.min()))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(
                f, fieldnames=['ref_setting', 'sample_id', 'camera'] + keys)
            w.writeheader()
            w.writerows(rows)
        print('\nper-image metrics -> %s' % args.out)


if __name__ == '__main__':
    main()
