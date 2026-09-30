#!/usr/bin/env python
"""V3-A.5E Target Predictability Audit (zero verifier training).

Question: is q*_G64 recoverable from observables (X/Y0/R + frozen proposal
evidence), or does it require GT-only / image-identity information?

Writes under --root:
  diagnostics/{block_probe,image_probe,nn,conditional,verdict}.json
  findings_v3a5e.md  (via separate step or --write_findings)
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

from model.V3A5DEvidenceProbe import V3A5DEvidenceProbe                 # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, action_optimal_target, block_energy,  # noqa: E402
                          energy_mask, prepare_geometry, target_geometry)
from v3a5c_runtime import dump_json                                     # noqa: E402
from v3a5d_runtime import (CHANNEL_SOURCE_SCALE, analysis_maps_from_evidence,  # noqa: E402
                           pool_spatial_map_to_geom)
from v3a5d1_runtime import stack_narrow_evidence_g64                    # noqa: E402
from v3a5e_runtime import (FEATURE_NAMES, conditional_variance,         # noqa: E402
                           fit_mlp, fit_ridge, mae, nn_ambiguity,
                           pearson, predict_mlp, predict_ridge, r2_score,
                           spearman, verdict_e)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
ROOT = '/root/data/experiments/v3a5e_predictability'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
REF_H, REF_W = 400, 600


def collect_split(probe, proposal, ds, n, geom, thr, device, split, limit=0):
    """Accumulate block-level feature matrix + q* + image-level rows."""
    n = n if not limit else min(limit, n)
    # store lists then vstack
    Xs, ys, img_rows = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            for state in STATES:
                t = sample_tensors(ds, i, state, device)
                _, aux, evidence = probe(t['Y0'], t['R'], check_equiv=False)
                D = aux['D']
                tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
                mask = energy_mask(block_energy(D, geom), thr)
                # narrow 3 + extra match/proposal maps + energy
                maps = analysis_maps_from_evidence(evidence)
                feats = []
                for name in FEATURE_NAMES:
                    if name == 'energy':
                        feats.append(block_energy(D, geom))
                        continue
                    raw = maps[name]
                    scale = CHANNEL_SOURCE_SCALE.get(name)
                    if scale is None:
                        # confidence_entropy uses same spatial as entropy (H/2)
                        scale = 2 if name == 'confidence_entropy' else 1
                    feats.append(pool_spatial_map_to_geom(raw, scale, geom))
                F = torch.cat(feats, dim=1)  # [1,C,64,64]
                q = tgt['q_grid']            # [1,1,64,64]
                m = mask.bool().reshape(-1).cpu().numpy()
                flat_f = F.reshape(F.shape[1], -1).float().cpu().numpy().T  # [Nblk,C]
                flat_q = q.reshape(-1).float().cpu().numpy()
                flat_f = flat_f[m]
                flat_q = flat_q[m]
                Xs.append(flat_f)
                ys.append(flat_q)
                img_rows.append(dict(
                    split=split, name=t['name'], state=state,
                    mean_q=float(flat_q.mean()) if flat_q.size else float('nan'),
                    **{FEATURE_NAMES[j]: float(flat_f[:, j].mean())
                       for j in range(len(FEATURE_NAMES))},
                ))
            if (i + 1) % 25 == 0 or i + 1 == n:
                print('  [%s] %d/%d (%.0fs)' % (split, i + 1, n, time.time() - t0),
                      flush=True)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, img_rows


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    return mu, sd


def apply_std(X, mu, sd):
    return (X - mu) / sd


def image_matrix(rows, split):
    sub = [r for r in rows if r['split'] == split]
    X = np.array([[r[n] for n in FEATURE_NAMES] for r in sub], dtype=np.float64)
    y = np.array([r['mean_q'] for r in sub], dtype=np.float64)
    return X, y, sub


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--nn_max', type=int, default=15000)
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
    model = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    probe = V3A5DEvidenceProbe(model.proposal).to(a.device).eval()
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)

    data = {}
    all_img = []
    for tag in ('train', 'dev'):
        print('=== collect %s ===' % tag, flush=True)
        ds = make_dataset(
            ns, splits[tag],
            os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmaps[tag])
        X, y, imgs = collect_split(
            probe, model, ds, len(splits[tag]), geom, thr, a.device, tag,
            limit=a.limit)
        data[tag] = dict(X=X, y=y)
        all_img.extend(imgs)
        print('  blocks=%d  features=%d' % (X.shape[0], X.shape[1]), flush=True)

    # persist lightweight arrays
    np.savez_compressed(
        os.path.join(a.root, 'diagnostics', 'blocks.npz'),
        X_train=data['train']['X'], y_train=data['train']['y'],
        X_dev=data['dev']['X'], y_dev=data['dev']['y'],
        feature_names=np.array(FEATURE_NAMES))
    dump_json(os.path.join(a.root, 'diagnostics', 'image_rows.json'), all_img)

    # ── block ridge + MLP probe ─────────────────────────────────────────
    mu, sd = standardize_fit(data['train']['X'])
    Xtr = apply_std(data['train']['X'], mu, sd)
    Xdv = apply_std(data['dev']['X'], mu, sd)
    ytr, ydv = data['train']['y'], data['dev']['y']

    w = fit_ridge(Xtr, ytr, l2=1e-1)
    pred_tr = predict_ridge(w, Xtr)
    pred_dv = predict_ridge(w, Xdv)
    # subsample for MLP speed
    rng = np.random.default_rng(0)
    n_mlp = min(80000, Xtr.shape[0])
    sel = rng.choice(Xtr.shape[0], size=n_mlp, replace=False)
    mlp = fit_mlp(Xtr[sel], ytr[sel], hidden=64, steps=600, seed=0)
    mlp_tr = predict_mlp(mlp, Xtr)
    mlp_dv = predict_mlp(mlp, Xdv)

    block_probe = dict(
        feature_names=list(FEATURE_NAMES),
        ridge=dict(
            train_r2=r2_score(ytr, pred_tr), dev_r2=r2_score(ydv, pred_dv),
            train_corr=pearson(ytr, pred_tr), dev_corr=pearson(ydv, pred_dv),
            train_spearman=spearman(ytr, pred_tr),
            dev_spearman=spearman(ydv, pred_dv),
            train_mae=mae(ytr, pred_tr), dev_mae=mae(ydv, pred_dv),
            weights={FEATURE_NAMES[i]: float(w[i]) for i in range(len(FEATURE_NAMES))},
            bias=float(w[-1]),
        ),
        mlp=dict(
            train_r2=r2_score(ytr, mlp_tr), dev_r2=r2_score(ydv, mlp_dv),
            train_corr=pearson(ytr, mlp_tr), dev_corr=pearson(ydv, mlp_dv),
            train_mae=mae(ytr, mlp_tr), dev_mae=mae(ydv, mlp_dv),
            n_train_sub=int(n_mlp),
        ),
    )
    # use better of ridge/mlp for verdict primary
    use = 'mlp' if block_probe['mlp']['train_r2'] >= block_probe['ridge']['train_r2'] else 'ridge'
    primary = block_probe[use]
    dump_json(os.path.join(a.root, 'diagnostics', 'block_probe.json'), block_probe)

    # ── image-level probe ───────────────────────────────────────────────
    Xitr, yitr, _ = image_matrix(all_img, 'train')
    Xidv, yidv, _ = image_matrix(all_img, 'dev')
    imu, isd = standardize_fit(Xitr)
    wi = fit_ridge(apply_std(Xitr, imu, isd), yitr, l2=1e-1)
    ip_tr = predict_ridge(wi, apply_std(Xitr, imu, isd))
    ip_dv = predict_ridge(wi, apply_std(Xidv, imu, isd))
    image_probe = dict(
        train_corr=pearson(yitr, ip_tr), dev_corr=pearson(yidv, ip_dv),
        train_spearman=spearman(yitr, ip_tr),
        dev_spearman=spearman(yidv, ip_dv),
        train_r2=r2_score(yitr, ip_tr), dev_r2=r2_score(yidv, ip_dv),
        train_mae=mae(yitr, ip_tr), dev_mae=mae(yidv, ip_dv),
        n_train=int(len(yitr)), n_dev=int(len(yidv)),
    )
    dump_json(os.path.join(a.root, 'diagnostics', 'image_probe.json'), image_probe)

    # ── NN ambiguity (block, train only; + cross-split image) ───────────
    print('=== NN ambiguity ===', flush=True)
    nn_block = nn_ambiguity(Xtr, ytr, k=5, max_n=a.nn_max, seed=0)
    nn_image = nn_ambiguity(apply_std(Xitr, imu, isd), yitr, k=5,
                            max_n=Xitr.shape[0], seed=0)
    # cross: each dev image → k NN in train
    Xs_tr = apply_std(Xitr, imu, isd)
    Xs_dv = apply_std(Xidv, imu, isd)
    cross_abs = []
    for i in range(Xs_dv.shape[0]):
        d = np.sqrt(((Xs_tr - Xs_dv[i]) ** 2).sum(axis=1))
        nn = np.argpartition(d, 5)[:5]
        cross_abs.append(abs(float(yidv[i]) - float(yitr[nn].mean())))
    nn = dict(
        block=nn_block,
        image_train=nn_image,
        image_dev_to_train=dict(
            mean_abs=float(np.mean(cross_abs)),
            median_abs=float(np.median(cross_abs)),
            n=len(cross_abs)),
        block_spearman_nn=nn_block.get('spearman_nn'),
    )
    dump_json(os.path.join(a.root, 'diagnostics', 'nn.json'), nn)

    # ── conditional variance on strongest single features ───────────────
    cond = {}
    for j, name in enumerate(FEATURE_NAMES):
        cond[name] = conditional_variance(Xtr[:, j], ytr, n_bins=10)
    dump_json(os.path.join(a.root, 'diagnostics', 'conditional.json'), cond)

    verdict = verdict_e(
        dict(train_r2=primary['train_r2'], dev_r2=primary['dev_r2'],
             train_corr=primary['train_corr'], dev_corr=primary['dev_corr']),
        nn,
        image_probe,
    )
    verdict['primary_probe'] = use
    verdict['energy_threshold'] = thr
    verdict['limit'] = a.limit
    verdict['n_train_blocks'] = int(Xtr.shape[0])
    verdict['n_dev_blocks'] = int(Xdv.shape[0])
    dump_json(os.path.join(a.root, 'diagnostics', 'e_verdict.json'), verdict)

    lock = dict(
        stage='V3-A.5E',
        root=a.root,
        energy_threshold=thr,
        feature_names=list(FEATURE_NAMES),
        prior='D2.1 Case B',
        note='target predictability from proposal/match observables only (no H)',
    )
    dump_json(os.path.join(a.root, 'artifact_lock.json'), lock)

    print('VERDICT', verdict['label'], 'next=', verdict['next_step'], flush=True)
    print('  meaning:', verdict['meaning'], flush=True)
    print('  block %s train R2=%.3f corr=%.3f | dev R2=%.3f corr=%.3f'
          % (use, primary['train_r2'], primary['train_corr'],
             primary['dev_r2'], primary['dev_corr']), flush=True)
    print('  image ridge train corr=%.3f dev corr=%.3f'
          % (image_probe['train_corr'], image_probe['dev_corr']), flush=True)
    print('  NN block spearman=%.3f mean|Δq|=%.3f'
          % (nn_block.get('spearman_nn', float('nan')),
             nn_block.get('mean_abs', float('nan')),), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
