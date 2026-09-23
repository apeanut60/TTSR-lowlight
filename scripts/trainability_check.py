#!/usr/bin/env python
"""Fixed-crop trainability check (Retinex V1.1 plan, sections 5.1 / 5.2).

Overfits a handful of fixed 128x128 crops — no augmentation, no shuffling, no
validation — for a fixed number of updates and reports the reconstruction
metric every `--every` steps. With only 8 crops, a model that cannot drive its
L1 down by 50% is telling you something is wrong with the data path, the
gradients or the optimiser, not with generalisation.

Two arms share the same initial weights and the same crops:

    rec   : L1 reconstruction only                     (plan's L0_rec)
    old   : the historical recipe                      (plan's L2_old_recipe)
            rec 1.0 + per 0.1 + illum_smooth 1.0 + color 0.5 + exposure 1.0

The reported metric is always `L1(sr, hr)` so the two arms are directly
comparable, regardless of what each arm actually optimises.

Usage
    python scripts/trainability_check.py --steps 1000 --lr 1e-4
    python scripts/trainability_check.py --arm rec --steps 300 --every 25
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from imageio import imread

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.TTSREnhance import TTSREnhance                     # noqa: E402
from model.Vgg19 import Vgg19                                 # noqa: E402
from loss.loss_enhance import (ColorConstancyLoss, ExposureControlLoss,   # noqa: E402
                               IlluminationSmoothnessLoss, PerceptualLoss,
                               ReconstructionLoss)


def build_model(kind, seed):
    torch.manual_seed(seed)
    args = argparse.Namespace(
        enhance_backbone='retinexformer', retinex_n_feat=40,
        retinex_num_blocks='1,2,2',
        num_res_blocks='8+8+4+2', n_feats=64, res_scale=1.0, ref_illum_pool=8,
        freeze_lte=True, ref_correction=False, ref_correction_feats=16,
        no_global_illum=True, no_reference=True, no_ref_texture=True,
        no_ref_illum=False, ref_illum_const_ref=False, oracle_matching='off',
    )
    return TTSREnhance(args).cuda()


def load_crops(root, n, size, seed=0):
    """`n` fixed centre crops, loaded in a deterministic order."""
    low_dir = os.path.join(root, 'Training data', 'Huawei', 'low')
    high_dir = os.path.join(root, 'Training data', 'Huawei', 'high')
    names = sorted(os.listdir(low_dir))[:n]
    lrs, hrs = [], []
    for name in names:
        for d, out in ((low_dir, lrs), (high_dir, hrs)):
            img = imread(os.path.join(d, name))
            if img.ndim == 2:
                img = np.stack([img] * 3, -1)
            t = torch.from_numpy(
                img[..., :3].astype(np.float32) / 127.5 - 1.).permute(2, 0, 1)
            _, h, w = t.shape
            t = t[:, (h - size) // 2:(h - size) // 2 + size,
                  (w - size) // 2:(w - size) // 2 + size]
            out.append(t)
    return torch.stack(lrs).cuda(), torch.stack(hrs).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/data/datasets/data1')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--every', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--crops', type=int, default=8)
    ap.add_argument('--crop_size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--arm', default='both',
                    choices=['rec', 'per', 'smooth', 'color', 'exposure',
                             'old', 'fixed', 'singles', 'recipe', 'both'],
                    help='"singles" runs rec plus each auxiliary term on its own, '
                         'so the reconstruction cost can be attributed; '
                         '"recipe" compares rec / old / old-without-exposure')
    args = ap.parse_args()

    lr_in, hr = load_crops(args.data, args.crops, args.crop_size)
    print('crops: %s from %s' % (tuple(lr_in.shape), args.data))
    print('steps: %d   lr: %g   seed: %d\n' % (args.steps, args.lr, args.seed))

    # Per-arm weights for the auxiliary terms (plan's L2_old_recipe).
    WEIGHTS = {'per': 0.1, 'smooth': 1.0, 'color': 0.5, 'exposure': 1.0}
    # Which auxiliary terms each arm switches on.
    ARM_TERMS = {
        'rec': [], 'per': ['per'], 'smooth': ['smooth'], 'color': ['color'],
        'exposure': ['exposure'],
        'old': ['per', 'smooth', 'color', 'exposure'],
        'fixed': ['per', 'smooth', 'color'],          # old minus exposure
    }
    ARM_SETS = {
        'both': ['rec', 'old'],
        'singles': ['rec', 'per', 'smooth', 'color', 'exposure', 'old'],
        'recipe': ['rec', 'old', 'fixed'],
    }
    arms = ARM_SETS.get(args.arm, [args.arm])

    results = {}
    for arm in arms:
        # Same initial weights for both arms.
        model = build_model(arm, args.seed)
        model.train()
        uses_vgg = 'per' in ARM_TERMS[arm]
        vgg = Vgg19(requires_grad=False).cuda().eval() if uses_vgg else None
        rec_loss = ReconstructionLoss('l1')
        active = ARM_TERMS[arm]
        loss_fns = {}
        for k in active:
            loss_fns[k] = {'per': PerceptualLoss,
                           'smooth': IlluminationSmoothnessLoss,
                           'color': ColorConstancyLoss,
                           'exposure': lambda: ExposureControlLoss(mean_val=0.4)}[k]()

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=args.lr)

        rows = []
        for step in range(1, args.steps + 1):
            sr = model(lr=lr_in, lrsr=lr_in, ref=lr_in, refsr=lr_in)[0]
            l1 = rec_loss(sr, hr)
            total = l1
            for k in active:
                w = WEIGHTS[k]
                if k == 'per':
                    total = total + w * loss_fns[k](vgg((sr + 1) / 2), vgg((hr + 1) / 2))
                else:
                    total = total + w * loss_fns[k](sr)
            opt.zero_grad(set_to_none=True)
            total.backward()
            gnorm = float(torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in params
                if p.grad is not None)))
            nn = sum(1 for p in params if p.grad is not None
                     and not torch.isfinite(p.grad).all())
            opt.step()

            if step == 1 or step % args.every == 0:
                with torch.no_grad():
                    rows.append(dict(
                        step=step, l1=float(l1), total=float(total),
                        gnorm=gnorm, nonfinite=nn,
                        mx=float(sr.max()), mn=float(sr.min()),
                        oob=float(sr.abs().gt(1).float().mean())))

        results[arm] = rows
        desc = {'rec': 'L1 only', 'old': 'the historical weighted sum',
                'fixed': 'historical weighted sum minus exposure'}.get(
            arm, 'L1 + %s' % ' + '.join('%s(%g)' % (k, WEIGHTS[k]) for k in active))
        print('── %s  (optimises: %s)' % (arm, desc))
        print('  %6s %10s %10s %10s %9s %8s' % (
            'step', 'L1(sr,hr)', 'total', '|grad|', 'out_max', 'oob%'))
        for r in rows:
            print('  %6d %10.5f %10.5f %10.3f %9.4f %7.4f%%' % (
                r['step'], r['l1'], r['total'], r['gnorm'], r['mx'],
                r['oob'] * 100))
        # With a coarse --every there may be only a couple of rows; averaging
        # rows[:3] and rows[-3:] would then compare a set with itself. Fall
        # back to the single first/last row when there are too few samples.
        if len(rows) >= 6:
            first = np.mean([r['l1'] for r in rows[:3]])
            last = np.mean([r['l1'] for r in rows[-3:]])
        else:
            first, last = rows[0]['l1'], rows[-1]['l1']
        drop = 1.0 - last / first
        print('  首窗 L1 %.5f -> 末窗 L1 %.5f   降幅 %.1f%%   %s\n'
              % (first, last, drop * 100,
                 'PASS (>=50%)' if drop >= 0.5 else 'FAIL (<50%)'))
        del model, opt
        torch.cuda.empty_cache()

    if len(results) > 1:
        base = results.get('rec')
        base_last = None
        if base:
            base_last = (np.mean([r['l1'] for r in base[-3:]])
                         if len(base) >= 6 else base[-1]['l1'])
        print('══ 汇总（末窗 L1，越低越好）══')
        print('  %-10s %10s %10s %12s' % ('arm', '首窗', '末窗', '相对 rec-only'))
        for a in arms:
            rows = results[a]
            if len(rows) >= 6:
                f = np.mean([r['l1'] for r in rows[:3]])
                l = np.mean([r['l1'] for r in rows[-3:]])
            else:
                f, l = rows[0]['l1'], rows[-1]['l1']
            rel = '' if base_last is None else '%+.1f%%' % ((l / base_last - 1) * 100)
            print('  %-10s %10.5f %10.5f %12s' % (a, f, l, rel))


if __name__ == '__main__':
    main()
