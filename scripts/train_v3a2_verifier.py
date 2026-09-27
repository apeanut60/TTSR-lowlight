#!/usr/bin/env python
"""V3-A.2: freeze the R1 proposal, train only the verifier (plan §2, §7-§10).

States are correct / dark / noise. The target is the action-optimal gate, so the
verifier learns "how much of THIS correction should I apply", not "is this
reference good". No BCE, no safety term, no perceptual/smooth/colour.
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
from model.V3A1Refiner import V3A1Refiner                          # noqa: E402
from option import parser as option_parser                         # noqa: E402
from v3a2_runtime import (action_optimal_gate, masked_smooth_l1)   # noqa: E402
from v3a_runtime import rescale_reference                          # noqa: E402

R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
BLOWUP_LIMIT = 0.10


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
    ap.add_argument('--root', default='/root/data/experiments/v3a1_lolv2real',
                    help='source of manifests / Y0 cache / frozen proposal')
    ap.add_argument('--out_root', default='/root/data/experiments/v3a2_lolv2real')
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
    ap.add_argument('--noise_sigma', type=float, default=0.20)
    ap.add_argument('--limit_steps', type=int, default=0)
    a = ap.parse_args(_CLI)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = os.path.join(a.out_root, 'verifier_s%d' % a.seed)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise SystemExit('refusing to overwrite %s' % run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_root, 'logs'), exist_ok=True)
    t0 = time.time()
    print('=== V3-A.2 verifier -> %s' % run_dir)

    torch.manual_seed(a.seed)
    man = os.path.join(a.root, 'manifests', 'refiner_train.csv')
    pairs = pairs_from_manifest(man)
    cache = os.path.join(a.root, a.cache_name, 'refiner_train')
    meta = verify_cache(cache, strict=True)
    if meta.get('sample_count') != len(pairs):
        raise SystemExit('cache/manifest sample_count mismatch')
    print('cache OK (base %s, n=%d)' % (meta['_base_sha'][:16], len(pairs)))

    eps_energy = json.load(open(os.path.join(
        a.out_root, 'action_upper', 'energy.json'), encoding='utf-8'))['eps_energy']
    print('eps_energy = %.6e' % eps_energy)

    args = build_args(a.data_dir, a.variant)
    m = V3A1Refiner().to(device)
    m.load_state_dict(torch.load(os.path.join(a.root, R1_CK),
                                 map_location=device)['model'])
    frozen = {k: v.clone() for k, v in m.proposal.state_dict().items()}
    for p in m.proposal.parameters():
        p.requires_grad_(False)
    m.proposal.eval()
    print('frozen proposal <- %s' % R1_CK)
    print('verifier params = %d' % sum(p.numel() for p in m.verifier.parameters()))

    ds = TrainSet(args, crop_size=a.crop, pairs=pairs, y0_cache=cache)
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, worker_init_fn=_worker_init, generator=g)
    if len(dl) == 0:
        raise SystemExit('empty dataloader')
    opt = torch.optim.Adam(m.verifier.parameters(), lr=a.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0)
    steps = a.limit_steps or a.steps
    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', encoding='utf-8')
    step, data_pass = 0, 0
    m.train()
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
            low = batch['LR'].to(device)
            ref = batch['Ref'].to(device)
            ngen = torch.Generator(device=device).manual_seed(a.seed * 100003 + step)
            states = [('correct', ref, 1.0),
                      ('dark', rescale_reference(ref, 0.5), 0.5),
                      ('noise', (ref + a.noise_sigma * torch.randn(
                          ref.shape, generator=ngen, device=device)).clamp(-1, 1), 0.5)]

            with torch.no_grad():
                prop = {}
                for name, r, _w in states:
                    _sr, aux = m.proposal(y0, r)
                    D = aux['gate'] * aux['delta']
                    q_opt, e = action_optimal_gate(y0, hr, D)
                    prop[name] = (D, q_opt, (e > eps_energy).float(),
                                  (e > 0).float())

            opt.zero_grad(set_to_none=True)
            total = 0.0
            row = dict(step=step, data_pass=data_pass, lr=lr)
            for name, r, w in states:
                D, q_opt, mask, valid = prop[name]
                _out, aux = m(y0, r, low=low)
                q_v4 = aux['q_v4']
                l_gate = masked_smooth_l1(q_v4, q_opt, mask)
                qf = F.interpolate(q_v4, size=y0.shape[-2:], mode='bilinear',
                                   align_corners=False)
                y_hat = y0 + qf * D
                l_rec = (y_hat - hr).abs().mean()
                loss = 1.0 * l_gate + 0.1 * l_rec
                (w * loss).backward()
                total += w * float(loss)
                row['%s_gate' % name] = float(l_gate)
                row['%s_rec' % name] = float(l_rec)
                row['%s_qv' % name] = float(q_v4.mean())
                row['%s_qopt' % name] = float(q_opt.mean())
                row['%s_valid' % name] = float(valid.mean())
                if name == 'correct':
                    row['corr'] = float((qf * D).abs().mean())
                    qvc, qoc = q_v4, q_opt
                else:
                    row['%s_corr' % name] = float((qf * D).abs().mean())
            row['loss'] = total
            # pixelwise correlation between prediction and target, on the
            # positions that actually carry a correction
            sel = prop['correct'][2] > 0
            row['corr_qv_qopt'] = float(torch.corrcoef(torch.stack(
                [qvc[sel].flatten(), qoc[sel].flatten()]))[0, 1]) if sel.any() else 0.0
            gnorm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                         for p in m.verifier.parameters()
                                         if p.grad is not None)))
            row['grad_norm'] = gnorm
            opt.step()
            if row['corr'] > BLOWUP_LIMIT:
                raise SystemExit('blow-up guard: mean|Yhat-Y0| = %.4f' % row['corr'])
            if step % a.log_every == 0 or step == 1:
                log_f.write(json.dumps(row) + '\n')
                log_f.flush()
                if step % (a.log_every * 5) == 0 or step == 1:
                    print('  step %d/%d lr=%.2e loss=%.4f gate=%.4f |g|=%.2e '
                          'q_v c/d/n=%.3f/%.3f/%.3f  q* c/d/n=%.3f/%.3f/%.3f  rho=%.3f'
                          % (step, steps, lr, total, row['correct_gate'], gnorm,
                             row['correct_qv'], row['dark_qv'], row['noise_qv'],
                             row['correct_qopt'], row['dark_qopt'], row['noise_qopt'],
                             row['corr_qv_qopt']))
            if step in (500, 1000, 2000) or step == steps:
                torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(),
                                global_step=step, eps_energy=eps_energy,
                                config=dict(vars(a))),
                           os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
        if step < steps:
            data_pass += 1

    now = m.proposal.state_dict()
    drift = max(float((now[k] - frozen[k]).abs().max()) for k in frozen)
    print('proposal max drift after training = %.3e (must be 0)' % drift)
    if drift != 0.0:
        raise SystemExit('proposal changed during V3-A.2 training')
    json.dump(dict(vars(a), eps_energy=eps_energy, proposal_drift=drift,
                   proposal_sha256=sha256(os.path.join(a.root, R1_CK)),
                   manifest_sha256=sha256(man),
                   cache_metadata_sha256=sha256(os.path.join(cache, 'metadata.json'))),
              open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    log_f.close()
    print('=== done in %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
