#!/usr/bin/env python
"""V3-A.4.1 Diagnostic A (§3-§9): where does q_opt change?

  FF  full proposal -> full target
  FC  full proposal -> crop target        (isolates the crop/local-target effect)
  CC  crop proposal -> crop target        (what verifier training actually sees)

Read-only: no training, no checkpoints, all parameters frozen.
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

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest, read_model_image  # noqa: E402
from local_refine_runtime import metrics                          # noqa: E402
from model.V3A4Verifier import V3A4Refiner                        # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a41_runtime import (aggregate_qopt_stats, compare_actions,  # noqa: E402
                           apply_geometry, crop_tensor, mode_cc, mode_ccg,
                           mode_fc, mode_fcg, qopt_components, read_crop_manifest,
                           verify_v3a41_artifact_lock)
from v3a4_runtime import load_r1_proposal_strict                  # noqa: E402
from v3a_runtime import exposure_gain                             # noqa: E402

V1 = '/root/data/experiments/v1'  # placeholder, replaced below
SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
STATES = ('correct', 'true_dark_g0.5', 'mismatch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--root', default='/root/data/experiments/v3a41_target_audit')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--splits', default='dev,train')
    ap.add_argument('--limit', type=int, default=0,
                    help='debug: only the first N images per split')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    dev = a.device
    for sub in ('qopt_audit', 'action_consistency'):
        os.makedirs(os.path.join(a.root, sub), exist_ok=True)

    verify_v3a41_artifact_lock(a.root, a.src_root, v4_root=a.v4_root)
    eps_e = json.load(open(os.path.join(a.v4_root, 'action_stats',
                                        'energy.json')))['eps_energy']
    crops = read_crop_manifest(os.path.join(a.root, 'crops', 'crop_manifest.csv'))
    by_split = {'train': {}, 'dev': {}}
    for r in crops:
        by_split[r['split']].setdefault(r['sample_id'], []).append(r)
    for s in by_split:
        for k in by_split[s]:
            by_split[s][k].sort(key=lambda x: int(x['crop_id']))

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    model = V3A4Refiner('none').to(dev).eval()
    load_r1_proposal_strict(model, os.path.join(a.src_root, R1_CK), dev)
    for p in model.parameters():
        p.requires_grad_(False)

    split = json.load(open(os.path.join(a.v4_root, 'splits', 'split.json'),
                           encoding='utf-8'))
    all_pairs = {p[0]: p for p in pairs_from_manifest(
        os.path.join(a.src_root, 'manifests', 'refiner_train.csv'))}
    acc = {}          # (split, mode, state) -> list of per-pixel tensors
    per_image = []    # §9
    act_rows = []     # §8
    for tag in [s.strip() for s in a.splits.split(',') if s.strip()]:
        ids = split['train' if tag == 'train' else 'dev']
        rows = [all_pairs[i] for i in ids]
        if a.limit:
            rows = rows[:a.limit]
        ds = TrainSet(ns, crop_size=0, pairs=rows,
                      y0_cache=os.path.join(a.src_root, a.cache_name,
                                            'refiner_train'), split='Train')
        mmap = json.load(open(os.path.join(
            a.v4_root, 'mappings',
            'mismatch_%s.json' % ('train_575' if tag == 'train' else 'dev_64')),
            encoding='utf-8'))
        for i, (name, low, high) in enumerate(rows):
            _n, lr, hr, ref, y0, _m = ds._load(i)
            donor = ds.ref_map[mmap[name]]
            mis = read_model_image(donor, size=hr.shape[-2:])
            refs = {'correct': ref, 'true_dark_g0.5': exposure_gain(ref, 0.5),
                    'mismatch': mis}
            y0d, hrd = y0[None].to(dev), hr[None].to(dev)
            cl = by_split[tag][name]
            img_rec = dict(sample_id=name, split=tag)
            with torch.no_grad():
                for s in STATES:
                    rd = refs[s][None].to(dev)
                    _sr, aux = model.proposal(y0d, rd)
                    D_full = aux['gate'] * aux['delta']
                    # ---- FF
                    c = qopt_components(y0d, hrd, D_full)
                    for k in ('q_opt', 'q_raw', 'N', 'Z', 'energy'):
                        acc.setdefault((tag, 'FF', s), {}).setdefault(k, []).append(
                            c[k].flatten().cpu())
                    img_rec['FF_%s' % s] = float(c['q_opt'].mean())
                    # ---- FC / FCG / CC / CCG per crop
                    cc_q, cc_d = [], []
                    for cr in cl:
                        top, left = int(cr['top']), int(cr['left'])
                        sz = int(cr['height'])
                        geom = (int(cr['rot_k']), int(cr['flip_h']), int(cr['flip_w']))
                        box = (top, left, sz, geom)
                        y0c = crop_tensor(y0d, top, left, sz, sz)
                        d_fc = crop_tensor(D_full, top, left, sz, sz)
                        c_fc = mode_fc(y0d, hrd, D_full, box)
                        c_fcg = mode_fcg(y0d, hrd, D_full, box)
                        c_cc, aux_cc, d_cc = mode_cc(y0d, hrd, rd, box,
                                                     model.proposal)
                        c_ccg, aux_ccg, d_ccg = mode_ccg(y0d, hrd, rd, box,
                                                         model.proposal)
                        for m_name, mc in (('FC', c_fc), ('FCG', c_fcg),
                                           ('CC', c_cc), ('CCG', c_ccg)):
                            for k in ('q_opt', 'q_raw', 'N', 'Z', 'energy'):
                                acc.setdefault((tag, m_name, s), {}).setdefault(
                                    k, []).append(mc[k].flatten().cpu())
                        cc_q.append(float(c_cc['q_opt'].mean()))
                        cc_d.append(float(c_fc['q_opt'].mean()))
                        y0g = apply_geometry(y0c, *geom)
                        # §8: compare the crop-context proposal against the
                        # crop+geometry proposal. CCG sees what training sees.
                        m8 = compare_actions(d_fc, d_cc)
                        m8.update(geom='%d%d%d' % geom,
                                  FCG_minus_FC=float(c_fcg['q_opt'].mean()
                                                     - c_fc['q_opt'].mean()),
                                  CCG_minus_CC=float(c_ccg['q_opt'].mean()
                                                     - c_cc['q_opt'].mean()))
                        m8.update(sample_id=name, split=tag, crop_id=int(cr['crop_id']),
                                  state=s,
                                  gate_MAE=float((aux['gate'][..., top:top + sz,
                                                          left:left + sz]
                                                  - aux_cc['gate']).abs().mean()),
                                  delta_MAE=float((crop_tensor(aux['delta'], top, left,
                                                               sz, sz)
                                                   - aux_cc['delta']).abs().mean()),
                                  # CC vs CCG: pure augmentation effect
                                  gate_MAE_geom=float(
                                      (apply_geometry(aux_cc['gate'], *geom)
                                       - aux_ccg['gate']).abs().mean()),
                                  delta_MAE_geom=float(
                                      (apply_geometry(aux_cc['delta'], *geom)
                                       - aux_ccg['delta']).abs().mean()))
                        act_rows.append(m8)
                    img_rec['FC_%s' % s] = float(np.mean(cc_d))
                    img_rec['CC_%s' % s] = float(np.mean(cc_q))
            per_image.append(img_rec)
            if (i + 1) % 100 == 0:
                print('  %s %d/%d' % (tag, i + 1, len(rows)))

    # ---- §6 statistics + §7 gap table
    summary, gap_table = {}, []
    for (tag, mode, s), blobs in acc.items():
        comp = {k: torch.cat(v) for k, v in blobs.items()}
        summary['%s|%s|%s' % (tag, mode, s)] = aggregate_qopt_stats(comp, eps_e)
        del comp
    MODES = ('FF', 'FC', 'FCG', 'CC', 'CCG')
    MODE_DESC = {'FF': ('full', 'full', 'none'),
                 'FC': ('full', 'crop', 'none'),
                 'FCG': ('full', 'crop', 'train-geometry(control)'),
                 'CC': ('crop', 'crop', 'none'),
                 'CCG': ('crop', 'crop', 'train-geometry')}
    for tag in ('train', 'dev'):
        for mode in MODES:
            c = summary['%s|%s|correct' % (tag, mode)]['q_opt']['mean']
            d = summary['%s|%s|true_dark_g0.5' % (tag, mode)]['q_opt']['mean']
            m = summary['%s|%s|mismatch' % (tag, mode)]['q_opt']['mean']
            prop, tgt, aug = MODE_DESC[mode]
            gap_table.append(dict(split=tag, mode=mode,
                                  proposal=prop, target=tgt, augmentation=aug,
                                  gap_gt=c - 0.5 * (d + m),
                                  q_correct=c, q_dark=d, q_mismatch=m))
    # all three states per (split, mode) -- keeping only `correct` made it
    # impossible to tell later whether N, Z or clipping drove a change
    for tag in ('train', 'dev'):
        for mode in MODES:
            blob = {s: summary['%s|%s|%s' % (tag, mode, s)] for s in STATES}
            json.dump(blob, open(os.path.join(
                a.root, 'qopt_audit', '%s_%s.json' % (tag, mode.lower())), 'w',
                encoding='utf-8'), indent=2, sort_keys=True)

    # ---- §9 per-image consistency
    per = {}
    for tag in ('train', 'dev'):
        recs = [r for r in per_image if r['split'] == tag]
        modes = {}
        for mode in MODES:
            g = np.array([r['%s_correct' % mode] - 0.5 * (r['%s_true_dark_g0.5' % mode]
                                                           + r['%s_mismatch' % mode])
                          for r in recs])
            modes[mode] = g
        def cor(a_, b_):
            return float(np.corrcoef(a_, b_)[0, 1]) if len(a_) > 2 else float('nan')
        per[tag] = dict(
            n=len(recs),
            gap=dict({m: dict(mean=float(v.mean()), median=float(np.median(v)),
                              p10=float(np.percentile(v, 10)),
                              p90=float(np.percentile(v, 90)),
                              frac_pos=float((v > 0).mean()))
                      for m, v in modes.items()}),
            corr={
                'FF_FC': cor(modes['FF'], modes['FC']),
                'FC_CC': cor(modes['FC'], modes['CC']),
                'CC_CCG': cor(modes['CC'], modes['CCG']),
                'FF_CCG': cor(modes['FF'], modes['CCG']),
                'FC_FCG': cor(modes['FC'], modes['FCG'])})

    # ---- §8 action consistency summary
    # A single pooled average would hide e.g. a mismatch-only crop-context
    # shift, so report per split x state as well.
    keys = ('MAE', 'RMSE', 'cosine', 'norm_ratio', 'gate_MAE', 'delta_MAE',
            'gate_MAE_geom', 'delta_MAE_geom', 'FCG_minus_FC', 'CCG_minus_CC')

    def _agg(rs):
        return {k: float(np.mean([r[k] for r in rs])) for k in keys}

    act_summary = dict(n=len(act_rows), pooled=_agg(act_rows),
                       by_split_state={})
    for tag in ('train', 'dev'):
        for s in STATES:
            sub = [r for r in act_rows if r['split'] == tag and r['state'] == s]
            if sub:
                act_summary['by_split_state']['%s|%s' % (tag, s)] = _agg(sub)

    json.dump(dict(gap_table=gap_table, per_image=per, action=act_summary),
              open(os.path.join(a.root, 'qopt_audit', 'summary.json'), 'w',
                   encoding='utf-8'), indent=2, sort_keys=True)
    with open(os.path.join(a.root, 'action_consistency', 'per_crop.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(act_rows[0].keys()))
        w.writeheader()
        w.writerows(act_rows)
    json.dump(act_summary, open(os.path.join(a.root, 'action_consistency',
                                             'summary.json'), 'w',
                                encoding='utf-8'), indent=2, sort_keys=True)
    with open(os.path.join(a.root, 'qopt_audit', 'per_image.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)

    # ---- report
    print()
    print('══ D1: gap_gt 表（§7 / §19 表 1） ══')
    print('  %-6s %-4s %-8s %-6s %-18s %10s %10s %10s %10s'
          % ('split', 'mode', 'proposal', 'target', 'augmentation', 'gap_gt',
             'q_corr', 'q_dark', 'q_mis'))
    for r in gap_table:
        print('  %-6s %-4s %-8s %-6s %-18s %+10.4f %10.4f %10.4f %10.4f'
              % (r['split'], r['mode'], r['proposal'], r['target'],
                 r['augmentation'], r['gap_gt'], r['q_correct'], r['q_dark'],
                 r['q_mismatch']))
    print()
    print('  per-image gap:')
    for tag in ('train', 'dev'):
        p = per[tag]
        print('    %-6s ' % tag + '  '.join(
            '%s %+.3f(pos %.2f)' % (m, p['gap'][m]['mean'], p['gap'][m]['frac_pos'])
            for m in MODES))
        print('           corr ' + '  '.join('%s %+.3f' % (k, v)
                                              for k, v in p['corr'].items()))
    print()
    print('══ D1: proposal action consistency (§8) ══')
    print('  pooled (n=%d): D_MAE %.4f  cosine %.4f  norm_ratio %.4f  '
          'gate_MAE %.4f  geom gate_MAE %.4f'
          % (act_summary['n'], act_summary['pooled']['MAE'],
             act_summary['pooled']['cosine'], act_summary['pooled']['norm_ratio'],
             act_summary['pooled']['gate_MAE'],
             act_summary['pooled']['gate_MAE_geom']))
    print('  %-14s %9s %9s %9s %11s' % ('split|state', 'D_MAE', 'cosine',
                                         'gate_MAE', 'geom_gate'))
    for k, v in act_summary['by_split_state'].items():
        print('  %-14s %9.4f %9.4f %9.4f %11.4f'
              % (k, v['MAE'], v['cosine'], v['gate_MAE'], v['gate_MAE_geom']))
    print('  geometry control: FCG-FC %+.4f (should be ~0)   '
          'augmentation effect: CCG-CC %+.4f'
          % (act_summary['pooled']['FCG_minus_FC'],
             act_summary['pooled']['CCG_minus_CC']))


if __name__ == '__main__':
    main()
