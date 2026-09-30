#!/usr/bin/env python
"""V3-A.5E1 Visual-Feature Predictability Audit (zero verifier training).

Question: does MultiScale A1 head-front feature Z at G64 predict q*_G64
across images? (completes the gap left by E0's 10-evidence probe)

Freeze shared-init A1 (step0 residual≡0 ⇒ Z = G64-pool(F_common));
no H in features. Writes under --root diagnostics + e1_verdict.json.
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

from model.V3A5D2Verifier import V3A5D2Verifier                          # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, action_optimal_target, block_energy,  # noqa: E402
                          energy_mask, prepare_geometry, target_geometry)
from v3a5c_runtime import dump_json                                     # noqa: E402
from v3a5e_runtime import (fit_ridge, mae, pearson, predict_ridge,      # noqa: E402
                           r2_score, spearman)
from v3a5e1_runtime import (fit_mlp_torch, nn_same_vs_cross,            # noqa: E402
                            predict_mlp_torch, verdict_e1)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
D2_INIT = '/root/data/experiments/v3a5d2_rf_full/init/shared_init_s42.pt'
ROOT = '/root/data/experiments/v3a5e1_visual_z'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
REF_H, REF_W = 400, 600


def extract_z(model, X, Y0, R, geom):
    h = model.common_features(X, Y0, R)
    if model.use_context:
        h = model.context(h)
    return model.prepare_features(h, geom)  # [1,C,64,64]


def collect_split(model, proposal, ds, n, geom, thr, device, split,
                  limit=0, per_image=48, seed=0):
    """Subsample energy-masked Z blocks; keep image-id for same/cross NN."""
    n = n if not limit else min(limit, n)
    rng = np.random.default_rng(seed + (0 if split == 'train' else 1))
    Xs, ys, ids = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            for state in STATES:
                t = sample_tensors(ds, i, state, device)
                D, _ = correction(proposal.proposal, t['Y0'], t['R'])
                tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
                mask = energy_mask(block_energy(D, geom), thr).bool().reshape(-1)
                Z = extract_z(model, t['X'], t['Y0'], t['R'], geom)
                flat_z = Z.reshape(Z.shape[1], -1).float().cpu().numpy().T
                flat_q = tgt['q_grid'].reshape(-1).float().cpu().numpy()
                m = mask.cpu().numpy()
                flat_z, flat_q = flat_z[m], flat_q[m]
                if flat_q.size == 0:
                    continue
                if flat_q.size > per_image:
                    sel = rng.choice(flat_q.size, size=per_image, replace=False)
                    flat_z, flat_q = flat_z[sel], flat_q[sel]
                # image key = name|state so same-image NN stays within pair
                key = '%s|%s' % (t['name'], state)
                Xs.append(flat_z)
                ys.append(flat_q)
                ids.extend([key] * flat_q.size)
            if (i + 1) % 25 == 0 or i + 1 == n:
                print('  [%s] %d/%d (%.0fs) blocks=%d'
                      % (split, i + 1, n, time.time() - t0,
                         sum(x.shape[0] for x in Xs)), flush=True)
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    return X, y, np.asarray(ids)


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    return mu, sd


def apply_std(X, mu, sd):
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--init_pt', default=D2_INIT)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--per_image', type=int, default=48)
    ap.add_argument('--nn_max', type=int, default=8000)
    ap.add_argument('--mlp_steps', type=int, default=1500)
    a = ap.parse_args(_CLI)

    os.makedirs(os.path.join(a.root, 'diagnostics'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    v5a = json.load(open(os.path.join(a.v5a_root, 'artifact_lock.json')))
    thr = float(v5a['energy_threshold'])

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       os.path.join(a.v4_root, 'splits', 'split.json'))
    mmaps = {
        'train': json.load(open(os.path.join(a.v4_root, 'mappings',
                                             'mismatch_train_575.json'))),
        'dev': json.load(open(os.path.join(a.v4_root, 'mappings',
                                           'mismatch_dev_64.json'))),
    }

    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    init = torch.load(a.init_pt, map_location='cpu')
    model = V3A5D2Verifier('A1_multiscale').to(a.device).eval()
    model.load_state_dict(init['A1_multiscale'], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    print('A1 context residual max|w|=%.3e (expect ~0 at shared init)'
          % model.context_residual_max_abs(), flush=True)
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)

    data = {}
    for tag in ('train', 'dev'):
        print('=== collect %s Z ===' % tag, flush=True)
        ds = make_dataset(
            ns, splits[tag],
            os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmaps[tag])
        X, y, ids = collect_split(
            model, proposal, ds, len(splits[tag]), geom, thr, a.device, tag,
            limit=a.limit, per_image=a.per_image)
        data[tag] = dict(X=X, y=y, ids=ids)
        print('  blocks=%d  dim=%d  images_keys=%d'
              % (X.shape[0], X.shape[1], len(set(ids.tolist()))), flush=True)

    np.savez_compressed(
        os.path.join(a.root, 'diagnostics', 'z_blocks.npz'),
        X_train=data['train']['X'], y_train=data['train']['y'],
        ids_train=data['train']['ids'],
        X_dev=data['dev']['X'], y_dev=data['dev']['y'],
        ids_dev=data['dev']['ids'])

    mu, sd = standardize_fit(data['train']['X'])
    Xtr = apply_std(data['train']['X'], mu, sd)
    Xdv = apply_std(data['dev']['X'], mu, sd)
    ytr, ydv = data['train']['y'], data['dev']['y']

    print('=== ridge probe ===', flush=True)
    w = fit_ridge(Xtr, ytr, l2=1e-1)
    pred_tr = predict_ridge(w, Xtr)
    pred_dv = predict_ridge(w, Xdv)
    ridge = dict(
        train_r2=r2_score(ytr, pred_tr), dev_r2=r2_score(ydv, pred_dv),
        train_corr=pearson(ytr, pred_tr), dev_corr=pearson(ydv, pred_dv),
        train_spearman=spearman(ytr, pred_tr),
        dev_spearman=spearman(ydv, pred_dv),
        train_mae=mae(ytr, pred_tr), dev_mae=mae(ydv, pred_dv),
        z_dim=int(Xtr.shape[1]),
    )

    print('=== torch MLP probe (steps=%d) ===' % a.mlp_steps, flush=True)
    # fit on already-standardized? fit_mlp_torch re-std — pass raw train X
    mlp = fit_mlp_torch(
        data['train']['X'], ytr, hidden=128, steps=a.mlp_steps,
        device=a.device if a.device.startswith('cuda') else 'cpu')
    mlp_tr = predict_mlp_torch(mlp, data['train']['X'])
    mlp_dv = predict_mlp_torch(mlp, data['dev']['X'])
    mlp_m = dict(
        train_r2=r2_score(ytr, mlp_tr), dev_r2=r2_score(ydv, mlp_dv),
        train_corr=pearson(ytr, mlp_tr), dev_corr=pearson(ydv, mlp_dv),
        train_mae=mae(ytr, mlp_tr), dev_mae=mae(ydv, mlp_dv),
        steps=a.mlp_steps, hidden=128,
    )
    # drop net before json
    block_probe = dict(ridge=ridge, mlp=mlp_m, feature='A1_Z_G64_shared_init')
    dump_json(os.path.join(a.root, 'diagnostics', 'block_probe.json'), block_probe)

    # primary = better train corr of ridge vs mlp (mlp now properly trained)
    use = 'mlp' if mlp_m['train_corr'] >= ridge['train_corr'] else 'ridge'
    primary = block_probe[use]

    print('=== same vs cross-image NN ===', flush=True)
    nn = nn_same_vs_cross(
        Xtr, ytr, data['train']['ids'], k=5, max_q=a.nn_max, seed=0)
    # also: each dev block → k NN in train (cross-split)
    rng = np.random.default_rng(1)
    n_dv_q = min(a.nn_max, Xdv.shape[0])
    qix = rng.choice(Xdv.shape[0], size=n_dv_q, replace=False)
    # subsample train pool
    n_pool = min(30000, Xtr.shape[0])
    pool = rng.choice(Xtr.shape[0], size=n_pool, replace=False)
    Xp, yp = Xtr[pool], ytr[pool]
    cross_split_abs, cross_split_ys, cross_split_ynn = [], [], []
    for i in qix:
        d = np.sqrt(((Xp - Xdv[i]) ** 2).sum(axis=1))
        nn_ix = np.argpartition(d, 5)[:5]
        pred = float(yp[nn_ix].mean())
        cross_split_abs.append(abs(float(ydv[i]) - pred))
        cross_split_ys.append(float(ydv[i]))
        cross_split_ynn.append(pred)
    nn['dev_to_train'] = dict(
        mean_abs=float(np.mean(cross_split_abs)),
        median_abs=float(np.median(cross_split_abs)),
        spearman=spearman(cross_split_ys, cross_split_ynn),
        pearson=pearson(cross_split_ys, cross_split_ynn),
        n=int(len(cross_split_abs)),
        pool_n=int(n_pool),
    )
    dump_json(os.path.join(a.root, 'diagnostics', 'nn.json'), nn)

    verdict = verdict_e1(primary, nn)
    verdict['primary_probe'] = use
    verdict['energy_threshold'] = thr
    verdict['init_pt'] = a.init_pt
    verdict['per_image'] = a.per_image
    verdict['n_train_blocks'] = int(Xtr.shape[0])
    verdict['n_dev_blocks'] = int(Xdv.shape[0])
    verdict['z_dim'] = int(Xtr.shape[1])
    verdict['context_residual_max_abs'] = model.context_residual_max_abs()
    verdict['note'] = (
        'shared-init A1; residual≈0 so Z≡G64-pool(F_common); '
        'tests visual X/Y0/R features, not trained memorization')
    dump_json(os.path.join(a.root, 'diagnostics', 'e1_verdict.json'), verdict)

    dump_json(os.path.join(a.root, 'artifact_lock.json'), dict(
        stage='V3-A.5E1',
        root=a.root,
        prior='E0_EVIDENCE_UNPREDICTABLE',
        init_pt=a.init_pt,
        energy_threshold=thr,
        z='A1 MultiScale → G64 pool (pre-head)',
        note='visual Z predictability; no verifier training; no H in Z',
    ))

    print('VERDICT', verdict['label'], 'next=', verdict['next_step'], flush=True)
    print('  meaning:', verdict['meaning'], flush=True)
    print('  %s train R2=%.3f corr=%.3f | dev R2=%.3f corr=%.3f'
          % (use, primary['train_r2'], primary['train_corr'],
             primary['dev_r2'], primary['dev_corr']), flush=True)
    print('  NN same sp=%.3f | cross sp=%.3f | dev→train sp=%.3f'
          % (nn['same'].get('spearman', float('nan')),
             nn['cross'].get('spearman', float('nan')),
             nn['dev_to_train'].get('spearman', float('nan'))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
