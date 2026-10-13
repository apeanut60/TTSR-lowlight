#!/usr/bin/env python
"""V8-B0 Gate1 eval — formal_dev64 + microset-heldout16.

Per-image PSNR: Base, Self, Nano Normal/Shuffle/Self/ZeroT.
Also train-batch mean MSE (from train_meta) and Ref-dependence / severe drop.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import torch

REPO = '/root/projects/TTSR-lowlight'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK, metrics as _metrics  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a6_runtime import dump_json, git_head, require_ckpt  # noqa: E402
from v3b_runtime import (  # noqa: E402
    b0_forward, b0_loss, match_features_mode, proposal_core)
from v8_b0_runtime import (  # noqa: E402
    ARM_NANO, ARM_SELF, ROOT, SRC, Y0_CACHE, arm_root, setting_paths)

MODES = ('nano_normal', 'nano_self', 'nano_shuffle', 'nano_zeroT', 'self_ref')
SEVERE_DROP_DB = 1.0


def _psnr(y, h):
    return float(_metrics(y, h)[0])


def eval_split(head, wrapper, ds, pairs, device, split_tag):
    rows = []
    with torch.no_grad():
        for idx, p in enumerate(pairs):
            t = sample_tensors(ds, p['i'], 'correct', device)
            # shuffle donor within split
            donor = pairs[(idx + 1) % len(pairs)]
            td = sample_tensors(ds, donor['i'], 'correct', device)
            y0, h, r_nano, r_shuf = t['Y0'], t['H'], t['R'], td['R']
            rec = dict(
                split=split_tag, name=p['name'],
                psnr_base=_psnr(y0, h),
            )
            # Self-ref protocol: R=Y0 rematch
            f0, tt = match_features_mode(wrapper, y0, y0, mode='self')
            y, _ = b0_forward(head, f0, tt, y0)
            rec['psnr_self_ref'] = _psnr(y, h)

            # Nano normal
            f0, tt = match_features_mode(wrapper, y0, r_nano, mode='normal')
            y_n, _ = b0_forward(head, f0, tt, y0)
            rec['psnr_nano_normal'] = _psnr(y_n, h)
            rec['mse_nano_normal'] = float(b0_loss(y_n, h))

            # Nano shuffle
            f0, tt = match_features_mode(wrapper, y0, r_shuf, mode='shuffled')
            y_s, _ = b0_forward(head, f0, tt, y0)
            rec['psnr_nano_shuffle'] = _psnr(y_s, h)

            # Nano self (= same as self_ref when rematching Y0; keep explicit)
            rec['psnr_nano_self'] = rec['psnr_self_ref']

            # ZeroT
            f0, tt = match_features_mode(wrapper, y0, r_nano, mode='zero')
            y_z, _ = b0_forward(head, f0, tt, y0)
            rec['psnr_nano_zeroT'] = _psnr(y_z, h)

            rec['dpsnr_nano_normal_minus_base'] = (
                rec['psnr_nano_normal'] - rec['psnr_base'])
            rec['dpsnr_nano_normal_minus_self'] = (
                rec['psnr_nano_normal'] - rec['psnr_self_ref'])
            rec['dpsnr_nano_normal_minus_shuffle'] = (
                rec['psnr_nano_normal'] - rec['psnr_nano_shuffle'])
            rec['dpsnr_nano_normal_minus_zeroT'] = (
                rec['psnr_nano_normal'] - rec['psnr_nano_zeroT'])
            rec['severe_vs_base'] = bool(
                rec['dpsnr_nano_normal_minus_base'] < -SEVERE_DROP_DB)
            rows.append(rec)
    return rows


def summarize(rows):
    def avg(k):
        return float(np.mean([r[k] for r in rows]))
    return dict(
        n=len(rows),
        psnr_base=avg('psnr_base'),
        psnr_self_ref=avg('psnr_self_ref'),
        psnr_nano_normal=avg('psnr_nano_normal'),
        psnr_nano_shuffle=avg('psnr_nano_shuffle'),
        psnr_nano_zeroT=avg('psnr_nano_zeroT'),
        dpsnr_n_base=avg('dpsnr_nano_normal_minus_base'),
        dpsnr_n_self=avg('dpsnr_nano_normal_minus_self'),
        dpsnr_n_shuffle=avg('dpsnr_nano_normal_minus_shuffle'),
        dpsnr_n_zeroT=avg('dpsnr_nano_normal_minus_zeroT'),
        mse_nano_normal=avg('mse_nano_normal'),
        n_severe_vs_base=int(sum(r['severe_vs_base'] for r in rows)),
        ref_dependence_ok=bool(
            avg('dpsnr_nano_normal_minus_self') > 0.03
            and avg('dpsnr_nano_normal_minus_zeroT') > 0.03),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, default=400)
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    paths = setting_paths(a.root)
    os.makedirs(paths['gate1'], exist_ok=True)

    wrapper = load_proposal(lock['proposal_ckpt'], a.device)
    core = proposal_core(wrapper)
    core.match.chunk = EVAL_QUERY_CHUNK

    splits = load_rows(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                       lock['split_json'])
    name_to_train = {r[0]: r for r in splits['train']}
    name_to_dev = {r[0]: r for r in splits['dev']}
    ns = option_parser.parse_args([])
    ns.dataset_dir = '/root/data/datasets/lol-v2-real'
    ns.v3a_ref_variant = lock['reference_variant']
    y0_dir = os.path.join(Y0_CACHE, 'refiner_train')
    mmap_tr = json.load(open(lock['mismatch_train']))
    mmap_dv = json.load(open(lock['mismatch_dev']))

    held_ids = list(lock['microset_heldout_ids'])
    held_rows = [name_to_train[n] for n in held_ids]
    ds_h = make_dataset(ns, held_rows, y0_dir, mmap_tr)
    pairs_h = [dict(i=i, name=held_ids[i]) for i in range(len(held_ids))]

    dev_ids = list(lock['formal_dev_ids'])
    dev_rows = [name_to_dev[n] for n in dev_ids]
    ds_d = make_dataset(ns, dev_rows, y0_dir, mmap_dv)
    pairs_d = [dict(i=i, name=dev_ids[i]) for i in range(len(dev_ids))]

    report = dict(
        gate='V8_B0_Gate1',
        step=a.step,
        repo_commit=git_head(),
        lock_init_sha=lock['init_sha'],
        arms={},
    )
    all_csv_rows = []

    for arm in (ARM_SELF, ARM_NANO):
        ckpt = os.path.join(arm_root(arm, a.root), 'checkpoints',
                            'ckpt_%06d.pt' % a.step)
        require_ckpt(ckpt, formal=True)
        blob = torch.load(ckpt, map_location='cpu')
        if int(blob['step']) != a.step:
            raise SystemExit('step mismatch %s' % ckpt)
        head = V3B0ResidualFusion().to(a.device).eval()
        head.load_state_dict(blob['model'], strict=True)
        meta_path = os.path.join(arm_root(arm, a.root), 'train_meta.json')
        train_meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}

        print('=== eval %s @%d ===' % (arm, a.step), flush=True)
        rows_d = eval_split(head, wrapper, ds_d, pairs_d, a.device, 'formal_dev')
        rows_h = eval_split(
            head, wrapper, ds_h, pairs_h, a.device, 'microset_heldout')
        for r in rows_d + rows_h:
            r['arm'] = arm
            all_csv_rows.append(r)
        sum_d, sum_h = summarize(rows_d), summarize(rows_h)
        report['arms'][arm] = dict(
            ckpt=ckpt,
            train_batch_mse=train_meta.get('batch_mse_train32'),
            formal_dev=sum_d,
            microset_heldout=sum_h,
            severe_dev=sum_d['n_severe_vs_base'],
            severe_heldout=sum_h['n_severe_vs_base'],
        )
        print('  dev  N−Base=%+.3f  N−Self=%+.3f  N−Shuf=%+.3f  severe=%d'
              % (sum_d['dpsnr_n_base'], sum_d['dpsnr_n_self'],
                 sum_d['dpsnr_n_shuffle'], sum_d['n_severe_vs_base']),
              flush=True)
        print('  held N−Base=%+.3f  N−Self=%+.3f  N−Shuf=%+.3f  severe=%d'
              % (sum_h['dpsnr_n_base'], sum_h['dpsnr_n_self'],
                 sum_h['dpsnr_n_shuffle'], sum_h['n_severe_vs_base']),
              flush=True)

    # Paired comparison Nano vs Self arm on same eval
    nano = report['arms'][ARM_NANO]
    selfa = report['arms'][ARM_SELF]
    report['paired_comparison'] = dict(
        note='same init/seed/schedule; Ref source differs',
        train_batch_mse=dict(
            Self=selfa['train_batch_mse'], Nano=nano['train_batch_mse']),
        formal_dev_dpsnr_n_base=dict(
            Self=selfa['formal_dev']['dpsnr_n_base'],
            Nano=nano['formal_dev']['dpsnr_n_base']),
        heldout_dpsnr_n_base=dict(
            Self=selfa['microset_heldout']['dpsnr_n_base'],
            Nano=nano['microset_heldout']['dpsnr_n_base']),
        heldout_ref_dep_nano=nano['microset_heldout']['ref_dependence_ok'],
        heldout_ref_dep_self_arm=selfa['microset_heldout']['ref_dependence_ok'],
    )
    # Gate1 pass: both arms trained; no severe majority; Nano shows better
    # ref dependence than chance on at least one split — diagnostic, not 3k go
    report['gate1_complete'] = True
    report['authorize_long_train'] = False
    report['forbid'] = list(lock.get('forbidden', []))

    csv_path = os.path.join(paths['gate1'], 'per_image_psnr.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(all_csv_rows)

    out_json = os.path.join(paths['gate1'], 'G1_report.json')
    dump_json(out_json, report)

    md = os.path.join(paths['gate1'], 'Gate1_report.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# V8-B0 Gate 1\n\n')
        f.write('- **step:** %d\n' % a.step)
        f.write('- **commit:** `%s`\n' % report['repo_commit'])
        f.write('- **init_sha:** `%s`\n' % lock['init_sha'][:16])
        f.write('- **authorize_long_train:** **False**\n\n')
        for arm in (ARM_SELF, ARM_NANO):
            ar = report['arms'][arm]
            f.write('## %s\n\n' % arm)
            f.write('train-batch mean MSE (32): **%.6f**\n\n'
                    % (ar['train_batch_mse'] or float('nan')))
            for tag, key in (('formal_dev64', 'formal_dev'),
                             ('microset-heldout16', 'microset_heldout')):
                s = ar[key]
                f.write('### %s\n\n' % tag)
                f.write('| Base | Self | NanoN | Shuffle | ZeroT | '
                        'N−Base | N−Self | N−Shuf | severe |\n')
                f.write('|---|---|---|---|---|---|---|---|---|\n')
                f.write('| %.3f | %.3f | %.3f | %.3f | %.3f | %+.3f | '
                        '%+.3f | %+.3f | %d |\n\n'
                        % (s['psnr_base'], s['psnr_self_ref'],
                           s['psnr_nano_normal'], s['psnr_nano_shuffle'],
                           s['psnr_nano_zeroT'], s['dpsnr_n_base'],
                           s['dpsnr_n_self'], s['dpsnr_n_shuffle'],
                           s['n_severe_vs_base']))
        f.write('## Verdict\n\n')
        f.write('Gate 1 diagnostic complete. No Official Test tuning. '
                'No 3k/20k auto-start.\n')
        f.write('\nCSV: `%s`\n' % csv_path)

    print(json.dumps(report['paired_comparison'], indent=2), flush=True)
    print('wrote %s' % md, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
