#!/usr/bin/env python
"""V3-A.1 training: V2-stable proposal + low-anchor verifier (plan §5, §8, §12).

  R1  = V2 proposal, correct reference only, reconstruction-side losses only
  R2  = same proposal init + verifier, correct/mismatch/corrupt, trust + safe

Hard requirements enforced here (not by convention):
  * the true low input is passed to every forward that involves the verifier;
  * R1's optimizer holds only proposal parameters, and its loss function has no
    trust/safety term at all;
  * the Y0 cache is verified against base + manifest before anything runs;
  * a blow-up guard aborts if the correction leaves a sane range early on.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.dataloader import _worker_init                        # noqa: E402
from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest    # noqa: E402
from local_refine_runtime import sha256, verify_cache              # noqa: E402
from loss.loss_enhance import (ColorConstancyLoss, IlluminationSmoothnessLoss,  # noqa: E402
                               PerceptualLoss, ReconstructionLoss)
from model.V3ARefiner import V3ARefiner                            # noqa: E402
from model.V3A1Refiner import V3A1Refiner, count_params            # noqa: E402
from model.Vgg19 import Vgg19                                      # noqa: E402
from option import parser as option_parser                         # noqa: E402
from v3a1_runtime import (compute_tau, loss_r1_naive_from_out,     # noqa: E402
                          loss_r2_state, usefulness_at)
from v3a_runtime import mismatch_permutation, rescale_reference    # noqa: E402

BASE = '/root/data/experiments/retinex_v11/N0_fixed_s42/model/model_00040.pt'
MISMATCH_WEIGHT = 0.5
CORRUPT_WEIGHT = 0.5
BLOWUP_LIMIT = 0.10          # mean |Y_hat - Y0|, plan §12


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
    ap.add_argument('--arm', required=True, choices=['r1', 'r2'])
    ap.add_argument('--root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--base_ckpt', default=BASE,
                    help='frozen base checkpoint the Y0 cache must have been built from')
    ap.add_argument('--cache_name', default='cache_y0')
    ap.add_argument('--bad_states', default='mismatch,corrupt',
                    help='R2 bad-reference states. The plan default is '
                         'mismatch,corrupt; on LOL neither is actually harmful '
                         '(+0.004 / +0.124 vs Base) while a darker reference is '
                         '(-0.393), so "dark" can be added to give the verifier '
                         'something real to reject.')
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = 'R1_v2stable_naive_s%d' % a.seed if a.arm == 'r1' \
        else 'R2_v2stable_verifier_s%d' % a.seed
    run_dir = os.path.join(a.root, tag)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise SystemExit('refusing to overwrite %s' % run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    t0 = time.time()
    print('=== V3-A.1 %s -> %s' % (a.arm.upper(), run_dir))

    torch.manual_seed(a.seed)
    man = os.path.join(a.root, 'manifests', 'refiner_train.csv')
    pairs = pairs_from_manifest(man)
    cache = os.path.join(a.root, a.cache_name, 'refiner_train')

    # --- strict provenance (plan §10) -------------------------------------
    meta = verify_cache(cache, strict=True)
    if sha256(a.base_ckpt) != meta.get('base_checkpoint_sha256'):
        raise SystemExit('cache was built from a different base checkpoint '
                         '(cache says %s, --base_ckpt is %s)'
                         % (meta.get('base_checkpoint_sha256', '?')[:16],
                            sha256(a.base_ckpt)[:16]))
    if sha256(man) != meta.get('manifest_sha256'):
        raise SystemExit('cache was built from a different manifest')
    if meta.get('sample_count') != len(pairs):
        raise SystemExit('cache sample_count %s != manifest %d'
                         % (meta.get('sample_count'), len(pairs)))
    print('cache provenance OK (base %s, n=%d)' % (meta['_base_sha'][:16], len(pairs)))

    args = build_args(a.data_dir, a.variant)

    # --- tau, non-cancelling definition (plan §6.1) -----------------------
    # keyed by cache: tau is defined against the base's Y0, so a different base
    # must not inherit the previous one's value
    tau_path = os.path.join(a.root, 'tau_%s.json' % a.cache_name)
    tau_ds = TrainSet(args, crop_size=0, pairs=pairs, y0_cache=cache, split='Train')
    if os.path.isfile(tau_path):
        tinfo = json.load(open(tau_path, encoding='utf-8'))
        tau = tinfo['tau']
    else:
        tau, vals = compute_tau(pairs, tau_ds)
        tinfo = dict(tau=tau, tau_method='median_i(mean_xy|d_i|)', n=len(vals),
                     train_manifest_sha=sha256(man), q_factor=4)
        json.dump(tinfo, open(tau_path, 'w', encoding='utf-8'), indent=2)
    print('tau = %.6f  method=%s' % (tau, tinfo.get('tau_method')))

    # --- mismatch map (fixed for the run) ---------------------------------
    os.makedirs(os.path.join(a.root, 'mappings'), exist_ok=True)
    mpath = os.path.join(a.root, 'mappings', 'refiner_mismatch.json')
    ids = [os.path.basename(l) for _, l, _ in pairs]
    if os.path.isfile(mpath):
        mmap = json.load(open(mpath, encoding='utf-8'))
    else:
        perm = mismatch_permutation([dict(camera='LOLv2real')] * len(ids))
        mmap = {ids[i]: ids[perm[i]] for i in range(len(ids))}
        json.dump(mmap, open(mpath, 'w', encoding='utf-8'), indent=2)

    # --- model + shared init ---------------------------------------------
    model = V3A1Refiner().to(device)
    idir = os.path.join(a.root, 'init')
    os.makedirs(idir, exist_ok=True)
    p_init = os.path.join(idir, 'v3a1_proposal_init_s%d.pt' % a.seed)
    v_init = os.path.join(idir, 'v3a1_verifier_init_s%d.pt' % a.seed)
    if os.path.isfile(p_init) and os.path.isfile(v_init):
        model.proposal.load_state_dict(torch.load(p_init, map_location=device))
        model.verifier.load_state_dict(torch.load(v_init, map_location=device))
        print('loaded shared inits')
    else:
        torch.save(model.proposal.state_dict(), p_init)
        torch.save(model.verifier.state_dict(), v_init)
        print('saved shared inits')
    assert float(model.proposal.c_out.weight.abs().max()) == 0.0, 'C_out not zero-init'
    print('proposal %d params, verifier %d params'
          % (sum(p.numel() for p in model.proposal.parameters()),
             sum(p.numel() for p in model.verifier.parameters())))

    vgg = Vgg19(requires_grad=False).to(device).eval()
    rec = ReconstructionLoss('l1')
    per = PerceptualLoss()
    smooth = IlluminationSmoothnessLoss()
    color = ColorConstancyLoss()
    rec_fn = lambda o, h: rec(o, h)
    per_fn = lambda o, h: per(vgg((o + 1.) / 2.), vgg((h + 1.) / 2.))
    sm_fn = lambda o: smooth(o)
    co_fn = lambda o: color(o)

    params = list(model.proposal.parameters())
    if a.arm == 'r2':
        params += list(model.verifier.parameters())
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0)

    ds = TrainSet(args, crop_size=a.crop, pairs=pairs, y0_cache=cache,
                  mismatch_map=mmap, gen_corrupt=(a.arm == 'r2'))
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, worker_init_fn=_worker_init, generator=g)
    if len(dl) == 0:
        raise SystemExit('empty dataloader')

    steps = a.limit_steps or a.steps
    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', encoding='utf-8')
    model.train()
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
            low = batch['LR'].to(device)          # the true X -- never Y0
            opt.zero_grad(set_to_none=True)
            row = dict(step=step, data_pass=data_pass, lr=lr, arm=a.arm)

            if a.arm == 'r1':
                out, aux = model.proposal(y0, batch['Ref'].to(device))
                total, info = loss_r1_naive_from_out(out, hr, rec_fn, per_fn,
                                                     sm_fn, co_fn)
                total.backward()
                row.update(info)
                row['gate_v2'] = float(aux['gate'].mean())
                row['q_v'] = None
                corr = y0 + aux['gate'] * aux['delta']
            else:
                ref_correct = batch['Ref'].to(device)
                pool = {'correct': (ref_correct, False),
                        'mismatch': (batch['Ref_mis'].to(device), True),
                        'corrupt': (batch['Ref_cor'].to(device), True),
                        'dark': (rescale_reference(ref_correct, 0.5), True)}
                want = ['correct'] + [s.strip() for s in a.bad_states.split(',') if s.strip()]
                for s in want:
                    if s not in pool:
                        raise SystemExit('unknown bad state %r' % s)
                states = [(s, pool[s][0], pool[s][1]) for s in want]
                w = {s: (1.0 if s == 'correct' else MISMATCH_WEIGHT) for s in want}
                total = 0.0
                for name, ref, bad in states:
                    out, aux = model(y0, ref, low=low)
                    q_star, _ = usefulness_at(y0, ref, hr, tau)
                    loss, info = loss_r2_state(out, hr, y0, aux['q_v4'], q_star,
                                               rec_fn, per_fn, sm_fn, co_fn, bad=bad)
                    (w[name] * loss).backward()
                    total += w[name] * float(loss)
                    for k, v in info.items():
                        row['%s_%s' % (name, k)] = v
                    row['%s_gate_v2' % name] = float(aux['gate_v2'].mean())
                    row['%s_q_v' % name] = float(aux['q_v'].mean())
                    if name == 'correct':
                        corr = out
                row['loss'] = total
            gnorm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                         for p in params if p.grad is not None)))
            opt.step()
            with torch.no_grad():
                d = (corr - y0).abs()
                row['corr_mean'] = float(d.mean())
                row['corr_p95'] = float(torch.quantile(d.flatten().float(), 0.95))
                row['corr_max_img'] = float(d.mean(dim=(1, 2, 3)).max())
                row['grad_norm'] = gnorm
            if row['corr_mean'] > BLOWUP_LIMIT:
                raise SystemExit('BLOW-UP guard tripped at step %d: mean|Yhat-Y0|=%.4f'
                                 % (step, row['corr_mean']))
            if step % a.log_every == 0 or step == 1:
                log_f.write(json.dumps(row) + '\n')
                log_f.flush()
                if step % (a.log_every * 5) == 0 or step == 1:
                    extra = '' if a.arm == 'r1' else \
                        ' q_v=%.3f' % row['correct_q_v']
                    print('  step %d/%d lr=%.2e loss=%.4f |g|=%.2e '
                          'corr=%.4f(%.4f)%s'
                          % (step, steps, lr, row.get('loss', float(total)),
                             gnorm, row['corr_mean'], row['corr_max_img'], extra))
        if step < steps:
            data_pass += 1

    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                    global_step=step, arm=a.arm, tau=tau,
                    config=dict(vars(a), root=a.root)),
               os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
    json.dump(dict(vars(a), root=a.root, tau=tau, tau_method=tinfo.get('tau_method'),
                   proposal_params=sum(p.numel() for p in model.proposal.parameters()),
                   verifier_params=sum(p.numel() for p in model.verifier.parameters()),
                   proposal_init_sha256=sha256(p_init),
                   verifier_init_sha256=sha256(v_init),
                   manifest_sha256=sha256(man),
                   cache_metadata_sha256=sha256(os.path.join(cache, 'metadata.json')),
                   data_passes=data_pass + 1),
              open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    log_f.close()
    print('=== done V3-A.1 %s in %.1f min' % (a.arm.upper(), (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
