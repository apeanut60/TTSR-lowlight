#!/usr/bin/env python
"""V3-A refiner training on LOL-v2-real (plan §4-§7).

  R1 = naive low-frequency refiner   (correct reference only, no q_star)
  R2 = hallucination-aware refiner   (correct + mismatch + corrupt, q_star)

Both arms share one initialisation and one frozen Base-R. The three reference
states of R2 are three forwards accumulated into a single optimizer update.

    python scripts/train_v3a_lolv2real.py --arm r1
    python scripts/train_v3a_lolv2real.py --arm r2
"""

import argparse
import csv
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
from local_refine_runtime import metrics, sha256                   # noqa: E402
from model.V3ARefiner import V3ARefiner, count_params              # noqa: E402
from option import parser as option_parser                         # noqa: E402
from v3a_runtime import (d8, mismatch_permutation, scalar_d,       # noqa: E402
                         usefulness)


def build_args(root, data_dir, variant):
    ns = option_parser.parse_args([])
    ns.dataset = 'lolv2real_v3a'
    ns.dataset_dir = data_dir
    ns.v3a_ref_variant = variant
    ns.no_reference = False
    ns.train_crop_size = 128
    return ns


def tau_for(root, args, pairs, cache, device):
    """tau = median(|d|) over the training samples (plan §4). Cached to disk."""
    p = os.path.join(root, 'tau.json')
    if os.path.isfile(p):
        return json.load(open(p))['tau']
    ds = TrainSet(args, crop_size=0, pairs=pairs, y0_cache=cache, split='Train')
    vals = []
    for i in range(len(pairs)):
        name, lr, hr, ref, y0, _ = ds._load(i)   # full images, no crop
        vals.append(float(scalar_d(y0[None], ref[None], hr[None])[0]))
    tau = float(np.median(np.abs(np.array(vals)))) + 1e-6
    json.dump(dict(tau=tau, n=len(vals),
                   note='median(|d|) over refiner-train samples, +1e-6'),
              open(p, 'w', encoding='utf-8'), indent=2)
    return tau


def state_losses(model, y0, hr, ref, tau):
    out, aux = model(y0, ref)
    q_star, _d = usefulness(y0, ref, hr, tau)
    l_rec = (out - hr).abs().mean()
    l_low = (d8(out) - d8(hr)).abs().mean()
    q8 = aux['gate8'].clamp(1e-6, 1 - 1e-6)
    l_trust = F.binary_cross_entropy(q8, q_star.clamp(1e-6, 1 - 1e-6))
    l_reject = ((1.0 - q_star) * (d8(out) - d8(y0)).abs()).mean()
    total = 1.0 * l_rec + 0.1 * l_low + 0.1 * l_trust + 0.1 * l_reject
    return total, dict(rec=float(l_rec), low=float(l_low),
                       trust=float(l_trust), reject=float(l_reject),
                       q=float(aux['gate'].mean()), q8=float(aux['gate8'].mean()),
                       q_star=float(q_star.mean()))


@torch.no_grad()
def evaluate(model, args, pairs, cache, tau, device, tag):
    ds = TrainSet(args, crop_size=0, pairs=pairs, y0_cache=cache, split='Test')
    ds.set_data_pass(0)
    perm = mismatch_permutation([dict(camera='LOLv2real')] * len(pairs))
    rows = []
    for i in range(len(pairs)):
        _n, _lr, hr, ref, y0, _mis = ds._load(i)
        donor = ds.ref_map[os.path.basename(pairs[perm[i]][1])]
        from dataset.lolv2real_v3a import read_model_image
        mis = read_model_image(donor, size=hr.shape[-2:])
        from v3a_runtime import splice_corrupt
        rng = np.random.default_rng(1234 + i)
        cor = splice_corrupt(ref, mis, rng)
        rec = dict(sample_id=os.path.basename(pairs[i][1]))
        rec['Base'] = metrics(y0[None].to(device), hr[None].to(device))
        for name, r in (('correct', ref), ('mismatch', mis), ('corrupt', cor)):
            out, aux = model(y0[None].to(device), r[None].to(device))
            rec[name] = metrics(out, hr[None].to(device))
            if name == 'correct':
                rec['q'] = float(aux['gate'].mean())
        out_byp, _ = model(y0[None].to(device), None)
        rec['bypass'] = metrics(out_byp, hr[None].to(device))
        rows.append(rec)
        if (i + 1) % 25 == 0:
            print('    eval %s %d/%d' % (tag, i + 1, len(pairs)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['r1', 'r2'])
    ap.add_argument('--root', default='/root/data/experiments/v3a_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--drop_step', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_after_drop', type=float, default=5e-5)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--limit_steps', type=int, default=0)
    a = ap.parse_args(_CLI)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = os.path.join(a.root, 'R%s_%s_s%d'
                           % (a.arm[1], 'naive' if a.arm == 'r1' else 'hallucination_aware',
                              a.seed))
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise SystemExit('refusing to overwrite %s' % run_dir)
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print('=== V3-A %s -> %s' % (a.arm.upper(), run_dir))

    torch.manual_seed(a.seed)
    man = os.path.join(a.root, 'manifests', 'refiner_train.csv')
    pairs = pairs_from_manifest(man)
    cache = os.path.join(a.root, 'cache_y0', 'refiner_train')
    args = build_args(a.root, a.data_dir, a.variant)
    tau = tau_for(a.root, args, pairs, cache, device)
    print('tau = %.6f  (n=%d train pairs)' % (tau, len(pairs)))

    ids = [os.path.basename(l) for _, l, _ in pairs]
    perm = mismatch_permutation([dict(camera='LOLv2real')] * len(ids))
    mmap = {ids[i]: ids[perm[i]] for i in range(len(ids))}
    os.makedirs(os.path.join(a.root, 'mappings'), exist_ok=True)
    mpath = os.path.join(a.root, 'mappings', 'refiner_mismatch.json')
    if not os.path.isfile(mpath):
        json.dump(mmap, open(mpath, 'w', encoding='utf-8'), indent=2)
    else:
        mmap = json.load(open(mpath, encoding='utf-8'))

    model = V3ARefiner().to(device)
    n_par = count_params(model)
    init_path = os.path.join(a.root, 'v3a_init_s%d.pt' % a.seed)
    if os.path.isfile(init_path):
        model.load_state_dict(torch.load(init_path, map_location=device))
        print('loaded shared init %s' % os.path.basename(init_path))
    else:
        torch.save(model.state_dict(), init_path)
        print('saved shared init %s' % os.path.basename(init_path))
    assert float(model.p_out.weight.abs().max()) == 0.0, 'proposal not zero-init'
    print('params = %d' % n_par)

    ds = TrainSet(args, crop_size=a.crop, pairs=pairs, y0_cache=cache,
                  mismatch_map=mmap, gen_corrupt=(a.arm == 'r2'))
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, worker_init_fn=_worker_init, generator=g)
    if len(dl) == 0:
        raise SystemExit('empty dataloader')
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0)

    test_pairs = pairs_from_manifest(os.path.join(a.root, 'manifests', 'test.csv'))
    test_cache = os.path.join(a.root, 'cache_y0', 'test')
    steps = a.limit_steps or a.steps
    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', encoding='utf-8')
    hist = []

    def run_eval(step):
        print('--- eval @ %d' % step)
        rows = evaluate(model, args, test_pairs, test_cache, tau, device, 'step%d' % step)
        keys = ['Base', 'correct', 'bypass', 'mismatch', 'corrupt']
        summ = {k: float(np.mean([r[k][0] for r in rows])) for k in keys}
        summ_ssim = {k: float(np.mean([r[k][1] for r in rows])) for k in keys}
        d_scheme = float(np.mean([r['correct'][0] - r['Base'][0] for r in rows]))
        d_safe = float(np.mean([r['mismatch'][0] - r['Base'][0] for r in rows]))
        d_cor = float(np.mean([r['corrupt'][0] - r['Base'][0] for r in rows]))
        print('    ' + '  '.join('%s=%.4f' % (k, summ[k]) for k in keys))
        print('    d_scheme=%+.4f  d_mismatch=%+.4f  d_corrupt=%+.4f'
              % (d_scheme, d_safe, d_cor))
        hist.append(dict(step=step, psnr=summ, ssim_y=summ_ssim,
                         d_scheme=d_scheme, d_mismatch=d_safe, d_corrupt=d_cor,
                         rows=rows))
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        global_step=step, arm=a.arm, tau=tau,
                        config=dict(vars(a), root=a.root)),
                   os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
        model.train()

    model.train()
    run_eval(0)
    step, data_pass = 0, 0
    while step < steps:
        ds.set_data_pass(data_pass)
        for batch in dl:
            if step >= steps:
                break
            step += 1
            lr = a.lr if step <= a.drop_step else a.lr_after_drop
            for gp in opt.param_groups:
                gp['lr'] = lr
            y0 = batch['Y0'].to(device)
            hr = batch['HR'].to(device)
            states = [('correct', batch['Ref'].to(device))]
            if a.arm == 'r2':
                states += [('mismatch', batch['Ref_mis'].to(device)),
                           ('corrupt', batch['Ref_cor'].to(device))]
            weights = {'correct': 1.0, 'mismatch': 0.5, 'corrupt': 0.5}
            opt.zero_grad(set_to_none=True)
            tot, parts = 0.0, {}
            for name, ref in states:
                loss, info = state_losses(model, y0, hr, ref, tau)
                (weights[name] * loss).backward()
                tot += weights[name] * float(loss)
                for k, v in info.items():
                    parts['%s_%s' % (name, k)] = v
            gnorm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                         for p in model.parameters()
                                         if p.grad is not None)))
            opt.step()
            if step % a.log_every == 0 or step == 1:
                row = dict(step=step, data_pass=data_pass, lr=lr, loss=tot,
                           grad_norm=gnorm, **parts)
                log_f.write(json.dumps(row) + '\n')
                log_f.flush()
                if step % (a.log_every * 5) == 0 or step == 1:
                    print('  step %d/%d lr=%.2e loss=%.5f |g|=%.2e '
                          'q=%.3f q*=%s'
                          % (step, steps, lr, tot, gnorm, parts['correct_q'],
                             '%.3f' % parts['correct_q_star'] if a.arm == 'r2' else 'n/a'))
            if step % a.eval_every == 0 or step == steps:
                run_eval(step)
        if step < steps:
            data_pass += 1

    json.dump(hist, open(os.path.join(run_dir, 'eval_history.json'), 'w',
                         encoding='utf-8'), indent=1)
    json.dump(dict(vars(a), tau=tau, params=n_par, root=a.root,
                   init_sha256=sha256(init_path),
                   manifest_sha256=sha256(man),
                   cache_metadata_sha256=sha256(os.path.join(cache, 'metadata.json'))),
              open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    log_f.close()
    print('=== done V3-A %s in %.1f min' % (a.arm.upper(), (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
