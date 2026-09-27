#!/usr/bin/env python
"""V3-A.4 §6-§14: train one verifier arm (c0 / c1-raw / c2-norm).

Everything except the action conditioning is shared across the three arms:
frozen proposal, split, split-isolated mismatch map, states, crop, batch order,
seed, q_opt, energy mask, loss, optimizer, LR, steps, verifier capacity.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.dataloader import _worker_init                        # noqa: E402
from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest    # noqa: E402
from local_refine_runtime import sha256, verify_cache              # noqa: E402
from model.V3A4Verifier import ACTION_CH, V3A4Refiner, build_shared_init  # noqa: E402
from option import parser as option_parser                         # noqa: E402
from v3a2_runtime import action_optimal_gate, masked_smooth_l1     # noqa: E402
from v3a4_runtime import (action_features_raw, freeze_proposal,    # noqa: E402
                          gap, load_r1_proposal_strict, masked_mae,
                          masked_rmse, pixel_corr)
from v3a_runtime import exposure_gain                              # noqa: E402

R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
ARMS = {'c0': 'none', 'c1': 'raw', 'c2': 'norm'}
TAGS = {'c0': 'C0_clean_s42', 'c1': 'C1_raw_clean_s42', 'c2': 'C2_norm_clean_s42'}


def build_args(data_dir, variant):
    ns = option_parser.parse_args([])
    ns.dataset = 'lolv2real_v3a'
    ns.dataset_dir = data_dir
    ns.v3a_ref_variant = variant
    ns.no_reference = False
    ns.train_crop_size = 128
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=sorted(ARMS))
    ap.add_argument('--src_root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--root', default='/root/data/experiments/v3a4_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--drop_step', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_after_drop', type=float, default=5e-5)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--limit_steps', type=int, default=0)
    a = ap.parse_args(_CLI)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = ARMS[a.arm]
    run_dir = os.path.join(a.root, TAGS[a.arm])
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise SystemExit('refusing to overwrite %s' % run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    t0 = time.time()
    print('=== V3-A.4 %s (mode=%s) -> %s' % (a.arm, mode, run_dir))
    torch.manual_seed(a.seed)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json'), encoding='utf-8'))
    split = json.load(open(os.path.join(a.root, 'splits', 'split.json'),
                           encoding='utf-8'))
    if sha256(os.path.join(a.src_root, R1_CK)) != lock['proposal_sha256']:
        raise SystemExit('proposal changed since the artifact lock')
    tr_ids = set(split['train'])
    all_pairs = pairs_from_manifest(os.path.join(a.src_root, 'manifests',
                                                 'refiner_train.csv'))
    pairs = sorted([p for p in all_pairs if p[0] in tr_ids], key=lambda p: p[0])
    if len(pairs) != 575:
        raise SystemExit('expected 575, got %d' % len(pairs))
    cache = os.path.join(a.src_root, a.cache_name, 'refiner_train')
    verify_cache(cache, strict=True)
    if sha256(os.path.join(cache, 'metadata.json')) != lock['cache_metadata_sha256']:
        raise SystemExit('cache metadata changed since the lock')

    mmap = json.load(open(os.path.join(a.root, 'mappings',
                                       'mismatch_train_575.json'), encoding='utf-8'))
    eps_e = json.load(open(os.path.join(a.root, 'action_stats',
                                        'energy.json')))['eps_energy']
    rms = json.load(open(os.path.join(a.root, 'action_stats',
                                      'action_norm.json')))['rms_D']
    print('states: correct + true_dark_g0.5 + mismatch | eps %.3e | rms %s'
          % (eps_e, ['%.5f' % x for x in rms]))

    os.makedirs(os.path.join(a.root, 'init'), exist_ok=True)
    init = build_shared_init(seed=a.seed, action_rms=rms)
    for k, v in init.items():
        torch.save(v, os.path.join(a.root, 'init',
                                   'v3a4_shared_init_s%d_%s.pt' % (a.seed, k)))
    m = V3A4Refiner(mode, action_rms=rms).to(dev)
    m.load_state_dict(init[mode])
    load_r1_proposal_strict(m, os.path.join(a.src_root, R1_CK), dev)
    frozen = freeze_proposal(m)
    nv = sum(p.numel() for p in m.verifier.parameters())
    print('verifier params %d (head0 in=%d)' % (nv, m.verifier.head0.in_channels))

    args = build_args(a.data_dir, a.variant)
    ds = TrainSet(args, crop_size=a.crop, pairs=pairs, y0_cache=cache,
                  mismatch_map=mmap, gen_corrupt=False)
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, worker_init_fn=_worker_init, generator=g)
    opt = torch.optim.Adam(m.verifier.parameters(), lr=a.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0)
    steps = a.limit_steps or a.steps
    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', encoding='utf-8')
    step, data_pass = 0, 0
    m.train()

    def head0_split():
        """§14: split head0's pre-activation into common / action parts."""
        if mode == 'none':
            return None
        w = m.verifier.head0.weight
        return w[:, :160].detach(), w[:, 160:].detach()

    while step < steps:
        ds.set_data_pass(data_pass)
        for batch in dl:
            if step >= steps:
                break
            step += 1
            lr = a.lr if step <= a.drop_step else a.lr_after_drop
            for gp in opt.param_groups:
                gp['lr'] = lr
            if m.proposal.training:
                raise SystemExit('proposal left eval mode at step %d' % step)
            y0 = batch['Y0'].to(dev)
            hr = batch['HR'].to(dev)
            low = batch['LR'].to(dev)
            ref = batch['Ref'].to(dev)
            refs = {'correct': ref,
                    'true_dark_g0.5': exposure_gain(ref, 0.5),
                    'mismatch': batch['Ref_mis'].to(dev)}
            harmful = ['true_dark_g0.5', 'mismatch']

            targets = {}
            with torch.no_grad():
                for name, r in refs.items():
                    _sr, aux = m.proposal(y0, r)
                    D = aux['gate'] * aux['delta']
                    q_opt, e = action_optimal_gate(y0, hr, D)
                    targets[name] = (D, q_opt, (e > eps_e).float())

            opt.zero_grad(set_to_none=True)
            total = 0.0
            row = dict(step=step, data_pass=data_pass, lr=lr, arm=a.arm)
            qv_map, qo_map = {}, {}
            for name, r in refs.items():
                D, q_opt, mask = targets[name]
                _out, aux = m(y0, r, low=low)
                q_v4 = aux['q_v4']
                l_gate = masked_smooth_l1(q_v4, q_opt, mask)
                qf = F.interpolate(q_v4, size=y0.shape[-2:], mode='bilinear',
                                   align_corners=False)
                l_rec = (y0 + qf * D - hr).abs().mean()
                w = 1.0 if name == 'correct' else 0.5
                (w * (l_gate + 0.1 * l_rec)).backward()
                total += w * float(l_gate + 0.1 * l_rec)
                qv_map[name], qo_map[name] = q_v4, q_opt
                row['%s_gate' % name] = float(l_gate)
                row['%s_mae' % name] = masked_mae(q_v4, q_opt, mask)
                row['%s_rmse' % name] = masked_rmse(q_v4, q_opt, mask)
                row['%s_corr' % name] = pixel_corr(q_v4, q_opt, mask)
                for lbl, t in (('q_v', q_v4), ('q_opt', q_opt)):
                    f = t.flatten().float()
                    for q, nm in ((0.10, 'p10'), (0.50, 'p50'), (0.90, 'p90')):
                        row['%s_%s_%s' % (name, lbl, nm)] = float(torch.quantile(f, q))
                    row['%s_%s_mean' % (name, lbl)] = float(f.mean())
                    row['%s_%s_std' % (name, lbl)] = float(f.std())
                row['%s_correction_mean' % name] = float((qf * D).abs().mean())
                d4, e4 = action_features_raw(D)
                row['%s_D4_mean' % name] = float(d4.mean())
                row['%s_D4_std' % name] = float(d4.std())
                row['%s_D4_p90' % name] = float(torch.quantile(
                    d4.abs().flatten().float(), 0.90))
                if mode == 'raw':
                    row['%s_E4_mean' % name] = float(e4.mean())
                    row['%s_E4_std' % name] = float(e4.std())
                    row['%s_E4_p90' % name] = float(torch.quantile(
                        e4.flatten().float(), 0.90))
                if mode == 'norm' and aux['d4'] is not None:
                    row['%s_D4n_mean' % name] = float(aux['d4'].mean())
                    row['%s_D4n_std' % name] = float(aux['d4'].std())
                    row['%s_D4n_p90' % name] = float(torch.quantile(
                        aux['d4'].abs().flatten().float(), 0.90))
                    row['%s_E4n_mean' % name] = float(aux['e4'].mean())
                    row['%s_E4n_p90' % name] = float(torch.quantile(
                        aux['e4'].flatten().float(), 0.90))
            gp_, gt_, ge_ = gap(qv_map, qo_map, harmful)
            row['loss'] = total
            row['gap_pred'], row['gap_gt'], row['gap_error'] = gp_, gt_, ge_
            row['grad_norm'] = float(torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in m.verifier.parameters()
                if p.grad is not None)))
            if mode != 'none':
                wc, wa = head0_split()
                last = m.verifier.head0
                with torch.no_grad():
                    store = {}
                    h = last.register_forward_hook(
                        lambda mod, inp, out, s=store: s.__setitem__('x', inp[0].detach()))
                    _o, _a = m(y0, ref, low=low)
                    h.remove()
                    x = store['x']
                zc = F.conv2d(x[:, :160], wc)
                za = F.conv2d(x[:, 160:], wa)
                row['head0_common_abs'] = float(zc.abs().mean())
                row['head0_action_abs'] = float(za.abs().mean())
                row['head0_action_ratio'] = float(
                    za.abs().mean() / (zc.abs().mean() + 1e-12))
                gwc, gwa = last.weight.grad, None
                row['common_weight_l2'] = float(wc.norm())
                row['action_weight_l2'] = float(wa.norm())
            opt.step()
            if step % a.log_every == 0 or step == 1:
                log_f.write(json.dumps(row) + '\n')
                log_f.flush()
                if step % (a.log_every * 5) == 0 or step == 1:
                    extra = ('' if mode == 'none' else
                             ' ratio=%.3f aw=%.3f' % (row['head0_action_ratio'],
                                                      row['action_weight_l2']))
                    print('  step %d/%d lr=%.2e loss=%.4f corr=%.3f mae=%.3f '
                          'gap %+.3f/%+.3f err %+.3f%s'
                          % (step, steps, lr, total, row['correct_corr'],
                             row['correct_mae'], gp_, gt_, ge_, extra))
            if step in (500, 1000, 2000) or step == steps:
                torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(),
                                global_step=step, arm=a.arm, mode=mode,
                                eps_energy=eps_e, rms_D=rms,
                                config=dict(vars(a))),
                           os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
        if step < steps:
            data_pass += 1

    drift = max(float((m.proposal.state_dict()[k] - frozen[k]).abs().max())
                for k in frozen)
    print('proposal drift = %.3e (must be 0)' % drift)
    if drift != 0.0:
        raise SystemExit('proposal changed during training')
    json.dump(dict(vars(a), arm=a.arm, mode=mode, verifier_params=nv,
                   eps_energy=eps_e, rms_D=rms, proposal_drift=drift,
                   artifact_lock_sha256=sha256(os.path.join(
                       a.root, 'artifact_lock.json'))),
              open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    log_f.close()
    print('=== done in %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
