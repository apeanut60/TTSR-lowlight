#!/usr/bin/env python
"""V3-A.3 §8-§11: train one verifier control (C0 or C1).

C0 and C1 differ in exactly one thing: whether the verifier is shown the
proposal's own pending correction (D4/E4). Frozen proposal, split, reference
states, target, loss, optimizer, batch order and step budget are all shared.
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
from model.V3A3Verifier import V3A3Refiner, build_shared_init      # noqa: E402
from option import parser as option_parser                         # noqa: E402
from v3a2_runtime import action_optimal_gate, energy_threshold, masked_smooth_l1  # noqa: E402
from v3a_runtime import contrast_compress, exposure_gain           # noqa: E402

R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def build_args(data_dir, variant):
    ns = option_parser.parse_args([])
    ns.dataset = 'lolv2real_v3a'
    ns.dataset_dir = data_dir
    ns.v3a_ref_variant = variant
    ns.no_reference = False
    ns.train_crop_size = 128
    return ns


def make_states(batch, harmful, dev, step):
    """The reference states for one batch: correct + the audited harmful ones."""
    ref = batch['Ref'].to(dev)
    out = {'correct': ref}
    for s in harmful:
        if s == 'mismatch':
            out[s] = batch['Ref_mis'].to(dev)
        elif s.startswith('true_dark_g'):
            out[s] = exposure_gain(ref, float(s.split('g')[-1]))
        elif s.startswith('true_bright_g'):
            out[s] = exposure_gain(ref, float(s.split('g')[-1]))
        elif s.startswith('contrast_compress'):
            out[s] = contrast_compress(ref, float(s.rsplit('_', 1)[-1]))
        elif s.startswith('gaussian_noise'):
            g = torch.Generator(device=dev).manual_seed(step)
            out[s] = (ref + 0.20 * torch.randn(ref.shape, generator=g,
                                               device=dev)).clamp(-1, 1)
        else:
            raise SystemExit('unknown state %r' % s)
    return out


def get_eps_energy(root, pairs, ds, m, dev, log_every=200):
    """p10 of the proposal's per-pixel correction energy on the train split."""
    p = os.path.join(root, 'action_upper', 'energy.json')
    if os.path.isfile(p):
        return json.load(open(p, encoding='utf-8'))['eps_energy']
    energies = []
    with torch.no_grad():
        for i in range(len(pairs)):
            _n, _lr, hr, ref, y0, _mm = ds._load(i)
            _sr, aux = m.proposal(y0[None].to(dev), ref[None].to(dev))
            _q, e = action_optimal_gate(y0[None].to(dev), hr[None].to(dev),
                                        aux['gate'] * aux['delta'])
            energies.append(e.cpu())
            if (i + 1) % log_every == 0:
                print('  energy %d/%d' % (i + 1, len(pairs)))
    eps = energy_threshold(energies)
    os.makedirs(os.path.join(root, 'action_upper'), exist_ok=True)
    json.dump(dict(eps_energy=eps, n=len(pairs), split='verifier_train',
                   method='p10 of per-pixel proposal energy'),
              open(p, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True, choices=['c0', 'c1'])
    ap.add_argument('--src_root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--root', default='/root/data/experiments/v3a3_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--states', default='')
    ap.add_argument('--noise_sigma', type=float, default=0.20)
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
    tag = ('C0_v3a2_control_s%d' % a.seed if a.control == 'c0'
           else 'C1_v3a3_action_s%d' % a.seed)
    run_dir = os.path.join(a.root, tag)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise SystemExit('refusing to overwrite %s' % run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    t0 = time.time()
    print('=== V3-A.3 %s -> %s' % (a.control.upper(), run_dir))
    torch.manual_seed(a.seed)

    split = json.load(open(os.path.join(a.root, 'splits', 'split.json'),
                           encoding='utf-8'))
    tr_ids = set(split['train'])
    all_pairs = pairs_from_manifest(os.path.join(a.src_root, 'manifests',
                                                 'refiner_train.csv'))
    pairs = sorted([p for p in all_pairs if p[0] in tr_ids], key=lambda p: p[0])
    if len(pairs) != 575:
        raise SystemExit('expected 575 train samples, got %d' % len(pairs))
    cache = os.path.join(a.src_root, a.cache_name, 'refiner_train')
    verify_cache(cache, strict=True)

    audit = json.load(open(os.path.join(a.root, 'corruption_audit', 'audit.json'),
                           encoding='utf-8'))
    harmful = ([s.strip() for s in a.states.split(',') if s.strip()]
               if a.states else audit['harmful_states'])
    if not harmful:
        raise SystemExit('no harmful state to train rejection on')
    print('states = correct + %s' % harmful)

    os.makedirs(os.path.join(a.root, 'init'), exist_ok=True)
    s0, s1 = build_shared_init(seed=a.seed)
    torch.save(s0, os.path.join(a.root, 'init',
                                'v3a23_shared_init_s%d_c0.pt' % a.seed))
    torch.save(s1, os.path.join(a.root, 'init',
                                'v3a23_shared_init_s%d_c1.pt' % a.seed))
    m = V3A3Refiner(use_action=(a.control == 'c1')).to(dev)
    m.load_state_dict(s1 if a.control == 'c1' else s0)
    # The R1 checkpoint stores a *V3A1Refiner* dict with 'proposal.'/'verifier.'
    # prefixes. Loading it into m.proposal matches nothing (every key misses)
    # and silently leaves the proposal at its zero-init, which makes D == 0.
    # Strip the prefix and load strictly.
    r1_sd = torch.load(os.path.join(a.src_root, R1_CK), map_location=dev)['model']
    prop_sd = {k[len('proposal.'):]: v for k, v in r1_sd.items()
               if k.startswith('proposal.')}
    if len(prop_sd) != len(m.proposal.state_dict()):
        raise SystemExit('proposal key count mismatch: %d vs %d'
                         % (len(prop_sd), len(m.proposal.state_dict())))
    m.proposal.load_state_dict(prop_sd, strict=True)
    if float(m.proposal.c_out.weight.abs().max()) == 0.0:
        raise SystemExit('proposal c_out is still zero -- R1 weights not loaded')
    frozen = {k: v.clone() for k, v in m.proposal.state_dict().items()}
    for p in m.proposal.parameters():
        p.requires_grad_(False)
    m.proposal.eval()
    nv = sum(p.numel() for p in m.verifier.parameters())
    print('verifier params = %d  (head0 in=%d)' % (nv, m.verifier.head0.in_channels))

    args = build_args(a.data_dir, a.variant)
    mmap = json.load(open(os.path.join(a.src_root, 'mappings',
                                       'refiner_mismatch.json'), encoding='utf-8'))
    ds = TrainSet(args, crop_size=a.crop, pairs=pairs, y0_cache=cache,
                  mismatch_map=mmap, gen_corrupt=False)
    eps_e = get_eps_energy(a.root, pairs, ds, m, dev)
    print('eps_energy = %.6e' % eps_e)
    if eps_e <= 0.0:
        raise SystemExit('eps_energy is zero: the proposal produced no correction '
                         '(check that the R1 weights actually loaded)')

    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, worker_init_fn=_worker_init, generator=g)
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
            y0 = batch['Y0'].to(dev)
            hr = batch['HR'].to(dev)
            low = batch['LR'].to(dev)
            refs = make_states(batch, harmful, dev, step)

            targets = {}
            with torch.no_grad():
                for name, r in refs.items():
                    _sr, aux = m.proposal(y0, r)
                    D = aux['gate'] * aux['delta']
                    q_opt, e = action_optimal_gate(y0, hr, D)
                    targets[name] = (D, q_opt, (e > eps_e).float())

            opt.zero_grad(set_to_none=True)
            total = 0.0
            row = dict(step=step, data_pass=data_pass, lr=lr, control=a.control)
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
                sel = mask > 0
                row['%s_gate' % name] = float(l_gate)
                row['%s_gate_masked_mae' % name] = float(
                    masked_smooth_l1(q_v4, q_opt, mask))
                row['%s_corr_qv_qopt' % name] = float(torch.corrcoef(torch.stack(
                    [q_v4[sel].flatten(), q_opt[sel].flatten()]))[0, 1]) \
                    if sel.any() else 0.0
                for lbl, t in (('q_v', q_v4), ('q_opt', q_opt)):
                    f = t.flatten().float()
                    row['%s_%s_mean' % (name, lbl)] = float(f.mean())
                    row['%s_%s_std' % (name, lbl)] = float(f.std())
                    row['%s_%s_p10' % (name, lbl)] = float(torch.quantile(f, 0.10))
                    row['%s_%s_p50' % (name, lbl)] = float(torch.quantile(f, 0.50))
                    row['%s_%s_p90' % (name, lbl)] = float(torch.quantile(f, 0.90))
                row['%s_correction_mean' % name] = float((qf * D).abs().mean())
                d4 = F.interpolate(D, size=q_v4.shape[-2:], mode='area')
                row['%s_D4_mean' % name] = float(d4.mean())
                row['%s_D4_std' % name] = float(d4.std())
                e4 = (d4 ** 2).mean(dim=1)
                row['%s_E4_mean' % name] = float(e4.mean())
                row['%s_E4_p90' % name] = float(torch.quantile(e4.flatten().float(), 0.90))
            row['loss'] = total
            row['gap_pred'] = float(np.mean(
                [qv_map['correct'].mean().item()]
                + [-qv_map[s].mean().item() for s in harmful]))
            row['gap_gt'] = float(np.mean(
                [qo_map['correct'].mean().item()]
                + [-qo_map[s].mean().item() for s in harmful]))
            row['grad_norm'] = float(torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in m.verifier.parameters()
                if p.grad is not None)))
            opt.step()
            if step % a.log_every == 0 or step == 1:
                log_f.write(json.dumps(row) + '\n')
                log_f.flush()
                if step % (a.log_every * 5) == 0 or step == 1:
                    print('  step %d/%d lr=%.2e loss=%.4f corr=%.3f mae=%.3f '
                          'q_v[%s]=%s  q*[%s]=%s  gapp=%+.3f gapg=%+.3f'
                          % (step, steps, lr, total,
                             row['correct_corr_qv_qopt'],
                             row['correct_gate_masked_mae'],
                             ','.join(['c'] + harmful),
                             ','.join('%.3f' % row['%s_q_v_mean' % s]
                                      for s in ['correct'] + harmful),
                             ','.join(['c'] + harmful),
                             ','.join('%.3f' % row['%s_q_opt_mean' % s]
                                      for s in ['correct'] + harmful),
                             row['gap_pred'], row['gap_gt']))
            if step in (500, 1000, 2000) or step == steps:
                torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(),
                                global_step=step, control=a.control,
                                harmful_states=harmful, eps_energy=eps_e,
                                config=dict(vars(a))),
                           os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
        if step < steps:
            data_pass += 1

    drift = max(float((m.proposal.state_dict()[k] - frozen[k]).abs().max())
                for k in frozen)
    print('proposal drift = %.3e (must be 0)' % drift)
    if drift != 0.0:
        raise SystemExit('proposal changed during training')
    json.dump(dict(vars(a), control=a.control, harmful_states=harmful,
                   verifier_params=nv, eps_energy=eps_e, proposal_drift=drift,
                   proposal_sha256=sha256(os.path.join(a.src_root, R1_CK)),
                   split_sha256=sha256(os.path.join(a.root, 'splits', 'split.json'))),
              open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    log_f.close()
    print('=== done in %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
