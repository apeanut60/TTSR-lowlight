#!/usr/bin/env python
"""V3-A.5E1b Trained-Z Closure Audit.

Loads D2.1 A1 ckpt_020000 (trained MultiScale), extracts G64 Z on
train64 vs dev64, with strict image_id / pair_id / (gy,gx).

NN regimes: same-pair-any, same-pair-far{4,8}, same-image≠state,
strict cross-image, plus dev→train.
"""

import argparse
import hashlib
import json
import os
import subprocess
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
                          energy_mask, prepare_geometry, state_dict_sha,
                          target_geometry)
from v3a5c_runtime import dump_json                                     # noqa: E402
from v3a5e_runtime import (fit_ridge, mae, pearson, predict_ridge,      # noqa: E402
                           r2_score, spearman)
from v3a5e1_runtime import fit_mlp_torch, predict_mlp_torch             # noqa: E402
from v3a5e1b_runtime import (file_sha256, nn_dev_to_train,              # noqa: E402
                             nn_regimes_e1b, verdict_e1b)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
D21 = '/root/data/experiments/v3a5d21_scale64'
CKPT = os.path.join(D21, 'A1_multiscale/checkpoints/ckpt_020000.pt')
ROOT = '/root/data/experiments/v3a5e1b_trained_z'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
REF_H, REF_W = 400, 600
GH, GW = 64, 64


def extract_z(model, X, Y0, R, geom):
    h = model.common_features(X, Y0, R)
    if model.use_context:
        h = model.context(h)
    return model.prepare_features(h, geom)


def collect_split(model, proposal, ds, names, name_to_i, geom, thr, device,
                  split, per_image=96, seed=0, limit=0):
    names = names[:limit] if limit else names
    rng = np.random.default_rng(seed + (0 if split == 'train' else 7))
    Xs, ys, imgs, pairs, sts, gys, gxs = [], [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for j, name in enumerate(names):
            for state in STATES:
                t = sample_tensors(ds, name_to_i[name], state, device)
                D, _ = correction(proposal.proposal, t['Y0'], t['R'])
                tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
                mask = energy_mask(block_energy(D, geom), thr).bool().reshape(-1)
                Z = extract_z(model, t['X'], t['Y0'], t['R'], geom)
                C = Z.shape[1]
                flat_z = Z.reshape(C, -1).float().cpu().numpy().T
                flat_q = tgt['q_grid'].reshape(-1).float().cpu().numpy()
                m = mask.cpu().numpy()
                idx = np.flatnonzero(m)
                if idx.size == 0:
                    continue
                if idx.size > per_image:
                    sel = rng.choice(idx.size, size=per_image, replace=False)
                    idx = idx[sel]
                gy = (idx // GW).astype(np.int32)
                gx = (idx % GW).astype(np.int32)
                flat_z = flat_z[idx]
                flat_q = flat_q[idx]
                pair = '%s|%s' % (name, state)
                n = flat_q.size
                Xs.append(flat_z)
                ys.append(flat_q)
                imgs.extend([name] * n)
                pairs.extend([pair] * n)
                sts.extend([state] * n)
                gys.append(gy)
                gxs.append(gx)
            if (j + 1) % 8 == 0 or j + 1 == len(names):
                print('  [%s] %d/%d (%.0fs) blocks=%d'
                      % (split, j + 1, len(names), time.time() - t0,
                         sum(x.shape[0] for x in Xs)), flush=True)
    return dict(
        X=np.concatenate(Xs, axis=0).astype(np.float32),
        y=np.concatenate(ys, axis=0).astype(np.float32),
        image_id=np.asarray(imgs),
        pair_id=np.asarray(pairs),
        state=np.asarray(sts),
        gy=np.concatenate(gys),
        gx=np.concatenate(gxs),
    )


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    return mu, sd


def apply_std(X, mu, sd):
    return (X - mu) / sd


def sha_json(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest()


def git_head(repo):
    try:
        return subprocess.check_output(
            ['git', '-C', repo, 'rev-parse', 'HEAD'],
            text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--d21_root', default=D21)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--per_image', type=int, default=96)
    ap.add_argument('--nn_max', type=int, default=6000)
    ap.add_argument('--mlp_steps', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(_CLI)

    os.makedirs(os.path.join(a.root, 'diagnostics'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)

    thr = float(json.load(open(os.path.join(a.v5a_root, 'artifact_lock.json')))
                ['energy_threshold'])
    train64 = json.load(open(os.path.join(a.d21_root, 'subset',
                                          'train64_ids.json')))
    train_ids = list(train64['ids'])
    mmap_train = json.load(open(os.path.join(
        a.d21_root, 'subset', 'train64_mismatch_map.json')))
    mmap_dev = json.load(open(os.path.join(
        a.v4_root, 'mappings', 'mismatch_dev_64.json')))
    split_path = os.path.join(a.v4_root, 'splits', 'split.json')
    split = json.load(open(split_path))
    dev_ids = list(split['dev'])

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    all_rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                         split_path)
    rows_train = {r[0]: r for r in all_rows['train']}
    rows_dev = {r[0]: r for r in all_rows['dev']}
    train_rows = [rows_train[i] for i in train_ids]
    # dev names may be in split['dev'] as ids matching row[0]
    missing = [i for i in dev_ids if i not in rows_dev]
    if missing:
        # sometimes split stores bare names already in rows
        raise SystemExit('dev ids missing from rows: %s' % missing[:5])
    dev_rows = [rows_dev[i] for i in dev_ids]

    proposal_path = os.path.join(a.src_root, R1_CK)
    proposal = load_proposal(proposal_path, a.device)
    blob = torch.load(a.ckpt, map_location='cpu')
    model = V3A5D2Verifier('A1_multiscale').to(a.device).eval()
    model.load_state_dict(blob['model'], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    ctx_res = model.context_residual_max_abs()
    print('loaded %s step=%s ctx_residual_max|w|=%.3e'
          % (a.ckpt, blob.get('step'), ctx_res), flush=True)
    if ctx_res == 0.0:
        print('WARN: trained context residual still zero — MultiScale inactive',
              flush=True)
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)

    ds_tr = make_dataset(
        ns, train_rows,
        os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmap_train)
    ds_dv = make_dataset(
        ns, dev_rows,
        os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmap_dev)
    name_to_i_tr = {n: i for i, n in enumerate(train_ids)}
    name_to_i_dv = {n: i for i, n in enumerate(dev_ids)}

    print('=== collect train64 Z_trained ===', flush=True)
    tr = collect_split(
        model, proposal, ds_tr, train_ids, name_to_i_tr, geom, thr, a.device,
        'train', per_image=a.per_image, seed=a.seed, limit=a.limit)
    print('  blocks=%d dim=%d images=%d pairs=%d'
          % (tr['X'].shape[0], tr['X'].shape[1],
             len(set(tr['image_id'].tolist())),
             len(set(tr['pair_id'].tolist()))), flush=True)

    print('=== collect dev64 Z_trained ===', flush=True)
    dv = collect_split(
        model, proposal, ds_dv, dev_ids, name_to_i_dv, geom, thr, a.device,
        'dev', per_image=a.per_image, seed=a.seed, limit=a.limit)
    print('  blocks=%d dim=%d images=%d pairs=%d'
          % (dv['X'].shape[0], dv['X'].shape[1],
             len(set(dv['image_id'].tolist())),
             len(set(dv['pair_id'].tolist()))), flush=True)

    np.savez_compressed(
        os.path.join(a.root, 'diagnostics', 'z_blocks.npz'),
        X_train=tr['X'], y_train=tr['y'],
        image_id_train=tr['image_id'], pair_id_train=tr['pair_id'],
        state_train=tr['state'], gy_train=tr['gy'], gx_train=tr['gx'],
        X_dev=dv['X'], y_dev=dv['y'],
        image_id_dev=dv['image_id'], pair_id_dev=dv['pair_id'],
        state_dev=dv['state'], gy_dev=dv['gy'], gx_dev=dv['gx'])

    mu, sd = standardize_fit(tr['X'])
    Xtr = apply_std(tr['X'], mu, sd)
    Xdv = apply_std(dv['X'], mu, sd)
    ytr, ydv = tr['y'], dv['y']

    print('=== ridge / MLP probes ===', flush=True)
    w = fit_ridge(Xtr, ytr, l2=1e-1)
    ridge = dict(
        train_r2=r2_score(ytr, predict_ridge(w, Xtr)),
        dev_r2=r2_score(ydv, predict_ridge(w, Xdv)),
        train_corr=pearson(ytr, predict_ridge(w, Xtr)),
        dev_corr=pearson(ydv, predict_ridge(w, Xdv)),
        train_spearman=spearman(ytr, predict_ridge(w, Xtr)),
        dev_spearman=spearman(ydv, predict_ridge(w, Xdv)),
        train_mae=mae(ytr, predict_ridge(w, Xtr)),
        dev_mae=mae(ydv, predict_ridge(w, Xdv)),
        z_dim=int(Xtr.shape[1]),
    )
    mlp = fit_mlp_torch(
        tr['X'], ytr, hidden=128, steps=a.mlp_steps,
        device=a.device if a.device.startswith('cuda') else 'cpu', seed=a.seed)
    mlp_m = dict(
        train_r2=r2_score(ytr, predict_mlp_torch(mlp, tr['X'])),
        dev_r2=r2_score(ydv, predict_mlp_torch(mlp, dv['X'])),
        train_corr=pearson(ytr, predict_mlp_torch(mlp, tr['X'])),
        dev_corr=pearson(ydv, predict_mlp_torch(mlp, dv['X'])),
        train_mae=mae(ytr, predict_mlp_torch(mlp, tr['X'])),
        dev_mae=mae(ydv, predict_mlp_torch(mlp, dv['X'])),
        steps=a.mlp_steps, hidden=128,
    )
    dump_json(os.path.join(a.root, 'diagnostics', 'block_probe.json'),
              dict(ridge=ridge, mlp=mlp_m, feature='A1_Z_G64_trained_d21_20k'))
    use = 'mlp' if mlp_m['train_corr'] >= ridge['train_corr'] else 'ridge'
    primary = mlp_m if use == 'mlp' else ridge

    print('=== NN regimes (train64) ===', flush=True)
    nn = nn_regimes_e1b(
        Xtr, ytr, tr['image_id'], tr['pair_id'], tr['state'], tr['gy'], tr['gx'],
        k=5, max_q=a.nn_max, seed=a.seed)
    print('=== NN dev64 → train64 ===', flush=True)
    d2t = nn_dev_to_train(Xtr, ytr, Xdv, ydv, k=5, max_q=a.nn_max, seed=a.seed + 1)
    dump_json(os.path.join(a.root, 'diagnostics', 'nn.json'),
              dict(train64=nn, dev_to_train=d2t))

    verdict = verdict_e1b(primary, nn, d2t)
    verdict.update(dict(
        primary_probe=use,
        ckpt=a.ckpt,
        ckpt_step=int(blob.get('step', -1)),
        context_residual_max_abs=ctx_res,
        energy_threshold=thr,
        per_image=a.per_image,
        n_train_blocks=int(Xtr.shape[0]),
        n_dev_blocks=int(Xdv.shape[0]),
        z_dim=int(Xtr.shape[1]),
        n_train_images=len(train_ids),
        n_dev_images=len(dev_ids),
    ))
    dump_json(os.path.join(a.root, 'diagnostics', 'e1b_verdict.json'), verdict)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lock = dict(
        stage='V3-A.5E1b',
        root=a.root,
        prior='E1_INIT_Z_CROSS_PAIR_WEAK',
        repo_head=git_head(repo),
        ckpt=a.ckpt,
        ckpt_sha256=file_sha256(a.ckpt),
        ckpt_step=int(blob.get('step', -1)),
        model_state_sha=state_dict_sha(blob['model']),
        proposal_ckpt=proposal_path,
        proposal_sha256=file_sha256(proposal_path),
        split_json=split_path,
        split_sha256=file_sha256(split_path),
        train64_ids_sha256=sha_json(train64),
        mismatch_train64_sha256=file_sha256(os.path.join(
            a.d21_root, 'subset', 'train64_mismatch_map.json')),
        mismatch_dev64_sha256=file_sha256(os.path.join(
            a.v4_root, 'mappings', 'mismatch_dev_64.json')),
        energy_threshold=thr,
        variant=a.variant,
        cache_name=a.cache_name,
        per_image=a.per_image,
        nn_max=a.nn_max,
        mlp_steps=a.mlp_steps,
        seed=a.seed,
        z='trained A1 MultiScale → G64 pool (pre-head)',
        context_residual_max_abs=ctx_res,
        note='E1b closure: strict image_id + spatial-far + trained Z',
    )
    dump_json(os.path.join(a.root, 'artifact_lock.json'), lock)

    print('VERDICT', verdict['label'], 'next=', verdict['next_step'], flush=True)
    print('  meaning:', verdict['meaning'], flush=True)
    print('  %s train corr=%.3f R2=%.3f | dev corr=%.3f R2=%.3f'
          % (use, primary['train_corr'], primary['train_r2'],
             primary['dev_corr'], primary['dev_r2']), flush=True)
    for key in ('same_pair_any', 'same_pair_far_4', 'same_pair_far_8',
                'same_image_diff_state', 'strict_cross_image'):
        sp = nn[key].get('spearman', float('nan'))
        print('  NN %-28s sp=%.3f n=%d' % (key, sp, nn[key].get('n', 0)),
              flush=True)
    print('  NN dev→train sp=%.3f mean|Δq|=%.3f'
          % (d2t.get('spearman', float('nan')), d2t.get('mean_abs', float('nan'))),
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
