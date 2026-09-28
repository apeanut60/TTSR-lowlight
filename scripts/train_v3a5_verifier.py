#!/usr/bin/env python
"""V3-A.5 §9-§15: train ONE arm (A0 dense-BlockH4 or A1 G64) on full images.

Fairness is structural, not conventional:

  * both arms take their initial weights from the SAME ``build_shared_init``
    sample, so the common trunk is bit-equal and the zero-initialised head makes
    step 0 identical (q = 0.5) for both;
  * identical optimizer / lr schedule / steps / grad accumulation / state
    sampling / loss weights;
  * the ONLY difference is the target geometry fed to the head (and therefore
    the target the gate is regressed to).

The proposal is frozen (``requires_grad=False``) and its parameter drift is
asserted to be exactly 0 at the end of training.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics                          # noqa: E402
from model.V3A5Verifier import V3A5Verifier, build_shared_init     # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,  # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (ARM_MODE, ARMS, STATES, STATE_PROBS,    # noqa: E402
                          action_optimal_target, bit_equal,
                          block_energy, check_worktree, energy_mask,
                          expand_gate, parameter_l1_drift, prepare_geometry,
                          sample_states, snapshot_, target_geometry,
                          state_dict_sha, verifier_loss,
                          verify_v3a5_artifact_lock)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V43 = '/root/data/experiments/v3a43_fine_resolution'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def lr_at(step, lr0=1e-4, lr1=5e-5, switch=2000):
    return lr0 if step < switch else lr1


class CorrectionCache(object):
    """Memoise D = frozen_proposal(Y0, R) per (image, state).

    The proposal is frozen and deterministic, so D is a pure function of
    (image, state): caching it is numerically identical to recomputing it, and
    it removes ~80% of the per-step cost (measured 0.67 s of 0.85 s). Entries
    live in CPU RAM (400x600x3 fp32 = 2.9 MB each) under a byte budget; once the
    budget is reached the cache simply stops growing and later steps recompute.
    """

    def __init__(self, max_gb=8.0):
        self.store = {}
        self.hits = 0
        self.misses = 0
        self.bytes = 0
        self.max_bytes = int(float(max_gb) * (1 << 30))

    def get(self, key, produce, device):
        v = self.store.get(key)
        if v is None:
            self.misses += 1
            v = produce().detach().to('cpu')
            if self.bytes + v.numel() * v.element_size() <= self.max_bytes:
                self.store[key] = v.clone()
                self.bytes += v.numel() * v.element_size()
        else:
            self.hits += 1
        return v.to(device)

    def stats(self):
        return dict(entries=len(self.store), hits=self.hits, misses=self.misses,
                    gigabytes=round(self.bytes / float(1 << 30), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/root/data/experiments/v3a5_g64_verifier')
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v43_root', default=V43)
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0,
                    help='smoke: only the first N train images (and fewer steps '
                         'unless --steps is given explicitly)')
    ap.add_argument('--cache_d_max_gb', type=float, default=8.0,
                    help='CPU RAM budget for the frozen-correction memo (0 = off)')
    a = ap.parse_args(_CLI)

    arm = a.arm
    arm_dir = os.path.join(a.root, arm)
    os.makedirs(os.path.join(arm_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)

    if a.limit and a.steps == 3000:
        a.steps = max(4, a.limit * 2)          # keep smokes quick by default

    git_head, git_dirty, wt_warning = check_worktree(a.limit)
    lock = verify_v3a5_artifact_lock(a.root, a.src_root, v4_root=a.v4_root,
                                     v43_root=a.v43_root)
    if a.steps != lock['steps'] and a.limit == 0:
        raise SystemExit('--steps %d != locked steps %d (a formal run must use the '
                         'locked schedule)' % (a.steps, lock['steps']))
    if a.grad_accum != lock['grad_accum_default'] and a.limit == 0:
        raise SystemExit('--grad_accum %d != locked %d'
                         % (a.grad_accum, lock['grad_accum_default']))
    mode = ARM_MODE[a.arm]
    thr = float(lock['energy_threshold'])

    logf = open(os.path.join(a.root, 'logs', 'train_%s.log' % arm), 'w',
                encoding='utf-8')

    def log(msg=''):
        print(msg)
        logf.write(msg + '\n')
        logf.flush()

    log('V3-A.5 training: %s (mode %s)' % (arm, mode))
    log('  commit      : %s  dirty=%d' % ((git_head or '?')[:8], len(git_dirty)))
    if wt_warning:
        log('  git WARNING : %s' % wt_warning)
    log('  steps       : %d (lr 1e-4 -> 5e-5 at 2000), grad_accum %d'
        % (a.steps, a.grad_accum))
    log('  seed        : %d   states %s (p=%s)' % (a.seed, STATES, STATE_PROBS))
    log('  energy thr  : %.6e (p%.0f, from the lock)' % (thr, lock['energy_pctl']))
    log('  limit       : %s' % (a.limit or 'none'))
    log()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    all_rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                         os.path.join(a.v4_root, 'splits', 'split.json'))
    train_rows, dev_rows = all_rows['train'], all_rows['dev']
    # the mismatch donor maps are split-isolated: using the train map on dev
    # rows would look up names that are not in it (and vice versa)
    mmap_train = json.load(open(os.path.join(a.v4_root, 'mappings',
                                             'mismatch_train_575.json'),
                                encoding='utf-8'))
    mmap_dev = json.load(open(os.path.join(a.v4_root, 'mappings',
                                           'mismatch_dev_64.json'), encoding='utf-8'))
    ds = make_dataset(ns, train_rows, os.path.join(a.src_root, a.cache_name,
                                                   'refiner_train'), mmap_train)
    n_train = len(train_rows) if not a.limit else min(a.limit, len(train_rows))
    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    prop_before = snapshot_(proposal.proposal)

    # one shared initialisation for both arms (§15)
    init = build_shared_init('dense_block_h4', 'g64', seed=int(lock['seed']))
    model = V3A5Verifier(mode).to(a.device)
    model.load_state_dict({k: v.clone() for k, v in init[mode].items()}, strict=True)
    if not bit_equal(init['dense_block_h4'], init['g64']):
        raise SystemExit('the two arms did not receive a bit-equal initialisation')
    log('  init: common weights bit-equal across arms (sha %s); head2 zero-init '
        '-> q=0.5' % state_dict_sha(init[mode])[:12])

    # every LOLv2-real sample is 400x600, so the geometry is built once and the
    # size is asserted per sample rather than assumed silently
    geoms = {m: prepare_geometry(target_geometry(400, 600, m), a.device)
             for m in ('g64', 'dense_block_h4')}
    target_shape = geoms[mode]['shape']

    def sample_step(i, state, cache_key=None, cache=None):
        t = sample_tensors(ds, i, state, a.device)
        if tuple(t['H'].shape[-2:]) != (400, 600):
            raise SystemExit('%s: expected 400x600, got %s'
                             % (t['name'], tuple(t['H'].shape[-2:])))
        with torch.no_grad():
            if cache is not None and cache_key is not None:
                # D depends only on (image, state) under the frozen proposal, so
                # the memo is exact, not an approximation
                D = cache.get(cache_key,
                              lambda: correction(proposal.proposal, t['Y0'],
                                                 t['R'])[0], a.device)
            else:
                D, _sr = correction(proposal.proposal, t['Y0'], t['R'])
            g = geoms[mode]
            tgt = action_optimal_target(t['Y0'], t['H'], D, g)
            mask = energy_mask(block_energy(D, g), thr)
        q_v = model(t['X'], t['Y0'], t['R'], target_shape=target_shape,
                    strict_native=(mode == 'dense_block_h4'))
        q_full = expand_gate(q_v, geoms[mode])
        loss, gate_t, out_t = verifier_loss(q_v, tgt['q_grid'], mask, q_full,
                                            t['Y0'], t['H'], D)
        return loss, gate_t, out_t, q_v, tgt, mask

    # §15 step-0 diagnostics on a fixed small dev subset
    model.eval()
    q_means, step0_psnr = [], []
    dev_ds = make_dataset(ns, dev_rows, os.path.join(a.src_root, a.cache_name,
                                                     'refiner_train'), mmap_dev)
    with torch.no_grad():
        for i in range(min(8, len(dev_rows))):
            for state in STATES:
                t = sample_tensors(dev_ds, i, state, a.device)
                D, _sr = correction(proposal.proposal, t['Y0'], t['R'])
                q_v = model(t['X'], t['Y0'], t['R'], target_shape=target_shape,
                            strict_native=(mode == 'dense_block_h4'))
                q_means.append(float(q_v.mean()))
                step0_psnr.append(metrics(t['Y0'] + expand_gate(q_v, geoms[mode]) * D,
                                          t['H'])[0])
    step0 = dict(q_mean=float(np.mean(q_means)), q_std=float(np.std(q_means)),
                 psnr_mean=float(np.mean(step0_psnr)), n=len(step0_psnr))
    log('  step0: q mean %.4f (std %.4f), PSNR %.4f dB on %d dev cells'
        % (step0['q_mean'], step0['q_std'], step0['psnr_mean'], step0['n']))
    if abs(step0['q_mean'] - 0.5) > 1e-6:
        raise SystemExit('step0 gate is not the shared 0.5 init: %.6f'
                         % step0['q_mean'])

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.train()
    cache = CorrectionCache(a.cache_d_max_gb)
    idx_seq = np.random.default_rng(int(a.seed)).integers(0, n_train,
                                                          size=a.steps * a.grad_accum)
    state_seq = sample_states(a.steps * a.grad_accum, int(a.seed) + 1)
    hist = []
    t0 = time.time()
    for step in range(a.steps):
        lr = lr_at(step)
        for grp in opt.param_groups:
            grp['lr'] = lr
        opt.zero_grad(set_to_none=True)
        agg = np.zeros(3, dtype=np.float64)
        for k in range(a.grad_accum):
            j = step * a.grad_accum + k
            i, state = int(idx_seq[j]), state_seq[j]
            loss, gate_t, out_t, q_v, tgt, mask = sample_step(
                i, state, cache_key=(i, state) if a.cache_d_max_gb > 0 else None,
                cache=cache)
            (loss / a.grad_accum).backward()
            agg += np.array([float(loss), float(gate_t), float(out_t)])
        opt.step()
        agg /= a.grad_accum
        if step % a.log_every == 0 or step == a.steps - 1:
            row = dict(step=step, loss=agg[0], gate=agg[1], out=agg[2], lr=lr,
                       q_v_mean=float(q_v.mean()), q_opt_mean=float(tgt['q_grid'].mean()),
                       mask_valid=float(mask.mean()))
            hist.append(row)
            log('  step %5d/%d  loss %.5f (gate %.5f + 0.1*out %.5f)  lr %.1e  '
                'q_v %.3f q_opt %.3f mask %.3f  %.0fs'
                % (step, a.steps, row['loss'], row['gate'], row['out'], lr,
                   row['q_v_mean'], row['q_opt_mean'], row['mask_valid'],
                   time.time() - t0))
        if step + 1 in (1000, 2000, a.steps):
            blob = dict(model=model.state_dict(), step=step + 1, mode=mode,
                        arm=arm, target_shape=list(target_shape),
                        energy_threshold=thr, seed=a.seed, limit=a.limit)
            p = os.path.join(arm_dir, 'checkpoints', 'ckpt_%06d.pt' % (step + 1))
            torch.save(blob, p)
            # 'last.pt' is the smoke-run handle; a formal eval insists on the
            # canonical locked step name instead (see eval_v3a5_dev.py)
            torch.save(blob, os.path.join(arm_dir, 'checkpoints', 'last.pt'))
            log('  checkpoint -> %s (and last.pt)' % p)

    drift = parameter_l1_drift(prop_before, snapshot_(proposal.proposal))
    if drift != 0.0:
        raise SystemExit('the frozen proposal drifted by %g during training' % drift)
    train_json = dict(arm=arm, mode=mode, steps=a.steps, grad_accum=a.grad_accum,
                      seed=a.seed, limit=a.limit, energy_threshold=thr,
                      states=list(STATES), state_probs=list(STATE_PROBS),
                      target_shape=list(target_shape), history=hist,
                      step0=step0, proposal_drift=drift,
                      wall_seconds=time.time() - t0,
                      correction_cache=cache.stats(),
                      git_head=git_head, git_dirty_count=len(git_dirty),
                      checkpoint='checkpoints/ckpt_%06d.pt' % a.steps)
    json.dump(train_json, open(os.path.join(arm_dir, 'train_metrics.json'), 'w',
                               encoding='utf-8'), indent=2, sort_keys=True)
    json.dump(dict(arm=arm, mode=mode, args=vars(a), lock_commit=lock['repo_commit'],
                   init_sha=state_dict_sha(init[mode]),
                   init_sha_both_arms_equal=bit_equal(init['dense_block_h4'],
                                                      init['g64']),
                   geometry=dict(shape=list(target_shape),
                                 edges_y=[int(v) for v in geoms[mode]['edges'][0]],
                                 edges_x=[int(v) for v in geoms[mode]['edges'][1]])),
              open(os.path.join(arm_dir, 'args.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    log('done: %s (proposal drift 0, %.0fs)' % (arm, time.time() - t0))
    logf.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
