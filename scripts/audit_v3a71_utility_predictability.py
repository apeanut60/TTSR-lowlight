#!/usr/bin/env python
"""V3-A.7.1 Utility Predictability Audit (zero verifier training).

Predict real-valued U_block / U_image from F0 evidence, F1 init-Z, F2 trained-Z.
No new verifier training. Official Test forbidden.
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

from local_refine_runtime import metrics as _metrics                    # noqa: E402
from model.V3A5D2Verifier import V3A5D2Verifier                          # noqa: E402
from model.V3A5DEvidenceProbe import V3A5DEvidenceProbe                 # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, block_energy, energy_mask,            # noqa: E402
                          expand_gate, prepare_geometry, target_geometry)
from v3a5d_runtime import (CHANNEL_SOURCE_SCALE,                        # noqa: E402
                           analysis_maps_from_evidence, pool_spatial_map_to_geom)
from v3a5e_runtime import (FEATURE_NAMES, fit_ridge, mae, pearson,      # noqa: E402
                           predict_ridge, spearman)
from v3a5e1_runtime import fit_mlp_torch, predict_mlp_torch             # noqa: E402
from v3a5e1b_runtime import file_sha256, nn_dev_to_train, nn_regimes_e1b  # noqa: E402
from v3a6_runtime import dump_json, git_head, hard_verify_lock          # noqa: E402
from v3a7_runtime import block_utility
from v3a71_runtime import (FORMAL_LOCK_KEYS, decision_regret,           # noqa: E402
                           image_utility, json_ready, mse_image, pack_reg,
                           ranked_bins, sign_metrics, utility_target_stats,
                           verdict_v3a71)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
V6 = '/root/data/experiments/v3a6_decision_gate'
D21 = '/root/data/experiments/v3a5d21_scale64'
ROOT = '/root/data/experiments/v3a71_utility_predictability'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
F2_CKPT = os.path.join(D21, 'A1_multiscale/checkpoints/ckpt_020000.pt')
REF_H, REF_W = 400, 600
GH, GW = 64, 64


def extract_z(model, X, Y0, R, geom):
    h = model.common_features(X, Y0, R)
    if model.use_context:
        h = model.context(h)
    return model.prepare_features(h, geom)


def stack_f0(maps, D, geom):
    feats = []
    for name in FEATURE_NAMES:
        if name == 'energy':
            feats.append(block_energy(D, geom))
            continue
        raw = maps[name]
        scale = CHANNEL_SOURCE_SCALE.get(name)
        if scale is None:
            scale = 2 if name == 'confidence_entropy' else 1
        feats.append(pool_spatial_map_to_geom(raw, scale, geom))
    return torch.cat(feats, dim=1)


def std_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0).clip(min=1e-6)
    return mu, sd


def apply_std(X, mu, sd):
    return (X - mu) / sd


def collect_split(probe, z_init, z_tr, proposal, ds, n, geom, thr, device,
                  split, per_image=96, seed=0, limit=0):
    n = n if not limit else min(limit, n)
    rng = np.random.default_rng(seed + (0 if split == 'train' else 11))
    blocks = dict(F0=[], F1=[], F2=[], U=[], image_id=[], pair_id=[],
                  state=[], gy=[], gx=[])
    images = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            for state in STATES:
                t = sample_tensors(ds, i, state, device)
                _, aux, evidence = probe(t['Y0'], t['R'], check_equiv=False)
                D = aux['D']
                maps = analysis_maps_from_evidence(evidence)
                F0 = stack_f0(maps, D, geom)
                Z1 = extract_z(z_init, t['X'], t['Y0'], t['R'], geom)
                Z2 = extract_z(z_tr, t['X'], t['Y0'], t['R'], geom)
                Ub = block_utility(t['Y0'], t['H'], D, geom)
                Ui = float(image_utility(t['Y0'], t['H'], D))
                mask = energy_mask(block_energy(D, geom), thr).bool().reshape(-1)
                m = mask.cpu().numpy()
                idx = np.flatnonzero(m)
                def flat(T):
                    return T.reshape(T.shape[1], -1).float().cpu().numpy().T
                f0, f1, f2 = flat(F0), flat(Z1), flat(Z2)
                u = Ub.reshape(-1).float().cpu().numpy()
                if idx.size:
                    if idx.size > per_image:
                        sel = rng.choice(idx.size, size=per_image, replace=False)
                        take = idx[sel]
                    else:
                        take = idx
                    key = '%s|%s' % (t['name'], state)
                    nb = take.size
                    blocks['F0'].append(f0[take])
                    blocks['F1'].append(f1[take])
                    blocks['F2'].append(f2[take])
                    blocks['U'].append(u[take])
                    blocks['image_id'].extend([t['name']] * nb)
                    blocks['pair_id'].extend([key] * nb)
                    blocks['state'].extend([state] * nb)
                    blocks['gy'].append((take // GW).astype(np.int32))
                    blocks['gx'].append((take % GW).astype(np.int32))
                    w = m.astype(np.float64)
                    wsum = w.sum() if w.sum() > 0 else 1.0
                    gF0 = (f0 * w[:, None]).sum(0) / wsum
                    gF1 = (f1 * w[:, None]).sum(0) / wsum
                    gF2 = (f2 * w[:, None]).sum(0) / wsum
                else:
                    gF0 = f0.mean(0); gF1 = f1.mean(0); gF2 = f2.mean(0)
                mse_b = float(mse_image(t['Y0'], t['H']))
                mse_r = float(mse_image(t['Y0'] + D, t['H']))
                images.append(dict(
                    split=split, image_id=t['name'], state=state,
                    U=Ui, sign=int(Ui > 0), abs_U=abs(Ui),
                    mse_base=mse_b, mse_r1=mse_r,
                    psnr_base=float(_metrics(t['Y0'], t['H'])[0]),
                    psnr_r1=float(_metrics(t['Y0'] + D, t['H'])[0]),
                    F0=gF0.astype(np.float32),
                    F1=gF1.astype(np.float32),
                    F2=gF2.astype(np.float32),
                ))
            if (i + 1) % 25 == 0 or i + 1 == n:
                nb = sum(x.shape[0] for x in blocks['F0']) if blocks['F0'] else 0
                print('  [%s] %d/%d (%.0fs) blocks=%d imgs=%d'
                      % (split, i + 1, n, time.time() - t0, nb, len(images)),
                      flush=True)
    out_b = {}
    for k in ('F0', 'F1', 'F2', 'U'):
        out_b[k] = np.concatenate(blocks[k], axis=0) if blocks[k] else np.zeros((0, 1))
    out_b['image_id'] = np.asarray(blocks['image_id'])
    out_b['pair_id'] = np.asarray(blocks['pair_id'])
    out_b['state'] = np.asarray(blocks['state'])
    out_b['gy'] = np.concatenate(blocks['gy']) if blocks['gy'] else np.zeros((0,), np.int32)
    out_b['gx'] = np.concatenate(blocks['gx']) if blocks['gx'] else np.zeros((0,), np.int32)
    return out_b, images


def probe_pair(Xtr, ytr, Xdv, ydv, tag, device, mlp_steps):
    mu, sd = std_fit(Xtr)
    w = fit_ridge(apply_std(Xtr, mu, sd), ytr, l2=1e-1)
    r_tr = predict_ridge(w, apply_std(Xtr, mu, sd))
    r_dv = predict_ridge(w, apply_std(Xdv, mu, sd))
    ridge = dict(train=pack_reg(ytr, r_tr), dev=pack_reg(ydv, r_dv))
    mlp = fit_mlp_torch(Xtr, ytr, hidden=128, steps=mlp_steps, device=device)
    m_tr = predict_mlp_torch(mlp, Xtr)
    m_dv = predict_mlp_torch(mlp, Xdv)
    mlpd = dict(train=pack_reg(ytr, m_tr), dev=pack_reg(ydv, m_dv),
                steps=mlp_steps)
    use = 'mlp' if mlpd['train']['pearson'] >= ridge['train']['pearson'] else 'ridge'
    pred_tr = m_tr if use == 'mlp' else r_tr
    pred_dv = m_dv if use == 'mlp' else r_dv
    signs = dict(train=sign_metrics(ytr, pred_tr), dev=sign_metrics(ydv, pred_dv))
    bins = ranked_bins(pred_dv, ydv)
    row = dict(
        feature=tag, primary=use,
        train_corr=ridge['train']['pearson'] if use == 'ridge' else mlpd['train']['pearson'],
        dev_corr=(ridge['dev']['pearson'] if use == 'ridge' else mlpd['dev']['pearson']),
        train_r2=ridge['train']['r2'] if use == 'ridge' else mlpd['train']['r2'],
        dev_r2=ridge['dev']['r2'] if use == 'ridge' else mlpd['dev']['r2'],
        auroc=signs['dev']['auroc'],
        auprc=signs['dev']['auprc'],
        balanced_acc=signs['dev']['balanced_acc'],
        brier=signs['dev']['brier'],
        ridge=ridge, mlp=mlpd, sign=signs, bins=bins,
    )
    return row, pred_tr, pred_dv, dict(kind=use, ridge_w=w, mu=mu, sd=sd, mlp=mlp)


def image_decision_from_pred(images, pred):
    psnr, d_base, d_r1, regrets = [], [], [], []
    for im, uhat in zip(images, pred):
        q = 1.0 if uhat > 0 else 0.0
        p = im['psnr_r1'] if q else im['psnr_base']
        mse = im['mse_r1'] if q else im['mse_base']
        psnr.append(p)
        d_base.append(p - im['psnr_base'])
        d_r1.append(p - im['psnr_r1'])
        regrets.append(decision_regret(mse, im['mse_base'], im['mse_r1']))
    rg = np.asarray(regrets, dtype=np.float64)
    return dict(
        mean_psnr=float(np.mean(psnr)),
        delta_base=float(np.mean(d_base)),
        delta_r1=float(np.mean(d_r1)),
        mean_regret=float(rg.mean()),
        median_regret=float(np.median(rg)),
        p90_regret=float(np.percentile(rg, 90)),
        n=len(psnr),
    )


@torch.no_grad()
def block_decision_psnr(z_model, feat, probe_pack, ds, n, geom, thr, proposal,
                        device, limit=0):
    """Full-map binary gate from U_pred on F1/F2 Z (dev only)."""
    n = n if not limit else min(limit, n)
    psnrs, regrets, d_base, d_r1 = [], [], [], []
    for i in range(n):
        for state in STATES:
            t = sample_tensors(ds, i, state, device)
            D, _ = correction(proposal.proposal, t['Y0'], t['R'])
            Z = extract_z(z_model, t['X'], t['Y0'], t['R'], geom)
            flat = Z.reshape(Z.shape[1], -1).float().cpu().numpy().T
            if probe_pack['kind'] == 'mlp':
                uhat = predict_mlp_torch(probe_pack['mlp'], flat)
            else:
                uhat = predict_ridge(
                    probe_pack['ridge_w'],
                    apply_std(flat, probe_pack['mu'], probe_pack['sd']))
            q = torch.from_numpy((uhat > 0).astype(np.float32)).to(device)
            q = q.view(1, 1, GH, GW)
            yhat = t['Y0'] + expand_gate(q, geom) * D
            p = float(_metrics(yhat, t['H'])[0])
            pb = float(_metrics(t['Y0'], t['H'])[0])
            pr = float(_metrics(t['Y0'] + D, t['H'])[0])
            mse_p = float(mse_image(yhat, t['H']))
            mse_b = float(mse_image(t['Y0'], t['H']))
            mse_r = float(mse_image(t['Y0'] + D, t['H']))
            psnrs.append(p); d_base.append(p - pb); d_r1.append(p - pr)
            regrets.append(decision_regret(mse_p, mse_b, mse_r))
    rg = np.asarray(regrets, dtype=np.float64)
    return dict(mean_psnr=float(np.mean(psnrs)),
                delta_base=float(np.mean(d_base)),
                delta_r1=float(np.mean(d_r1)),
                mean_regret=float(rg.mean()),
                median_regret=float(np.median(rg)),
                p90_regret=float(np.percentile(rg, 90)),
                n=len(psnrs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--v6_root', default=V6)
    ap.add_argument('--f2_ckpt', default=F2_CKPT)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--per_image', type=int, default=96)
    ap.add_argument('--mlp_steps', type=int, default=1500)
    ap.add_argument('--nn_max', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--formal', action='store_true', default=True)
    a = ap.parse_args(_CLI)

    os.makedirs(os.path.join(a.root, 'diagnostics'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)

    v5a = json.load(open(os.path.join(a.v5a_root, 'artifact_lock.json')))
    v6 = json.load(open(os.path.join(a.v6_root, 'artifact_lock.json')))
    proposal_path = os.path.join(a.src_root, R1_CK)
    split_path = os.path.join(a.v4_root, 'splits', 'split.json')
    mmap_tr = os.path.join(a.v4_root, 'mappings', 'mismatch_train_575.json')
    mmap_dv = os.path.join(a.v4_root, 'mappings', 'mismatch_dev_64.json')
    init_path = os.path.join(a.v6_root, 'init', 'shared_init_s42.pt')

    lock = dict(
        stage='V3-A.7.1',
        root=a.root,
        prior='V3A7_CASE_D_NO_GAIN',
        repo_commit=git_head(),
        proposal_ckpt=proposal_path,
        proposal_sha256=file_sha256(proposal_path),
        cache_name=a.cache_name,
        cache_metadata_sha256=v5a.get('cache_metadata_sha256'),
        split_json=split_path,
        split_sha256=file_sha256(split_path),
        mismatch_train=mmap_tr,
        mismatch_train_sha256=file_sha256(mmap_tr),
        mismatch_dev=mmap_dv,
        mismatch_dev_sha256=file_sha256(mmap_dv),
        energy_threshold=float(v5a['energy_threshold']),
        energy_stats_sha256=v5a.get('energy_stats_sha256'),
        reference_variant=a.variant,
        geometry='g64',
        architecture='V3A5D2Verifier.A1_multiscale',
        bottleneck=64,
        init_path=init_path,
        init_sha=v6.get('init_sha'),
        f2_ckpt=a.f2_ckpt,
        f2_ckpt_sha256=file_sha256(a.f2_ckpt),
        updates=0,
        grad_accum=4,
        lr=1e-4,
        seed=a.seed,
        pair_schedule_seed=42,
        states=list(STATES),
        mask_mode='g64_proposal_energy_expand',
        official_test_allowed=False,
        per_image=a.per_image,
        note='zero-train utility predictability; F0/F1/F2 × block/image',
    )
    dump_json(os.path.join(a.root, 'artifact_lock.json'), json_ready(lock))
    live = dict(
        proposal_sha256=file_sha256(proposal_path),
        split_sha256=file_sha256(split_path),
        mismatch_train_sha256=file_sha256(mmap_tr),
        mismatch_dev_sha256=file_sha256(mmap_dv),
        reference_variant=a.variant,
        geometry='g64',
        architecture='V3A5D2Verifier.A1_multiscale',
        bottleneck=64,
        official_test_allowed=False,
        energy_stats_sha256=v5a.get('energy_stats_sha256'),
        cache_metadata_sha256=v5a.get('cache_metadata_sha256'),
        init_sha=v6.get('init_sha'),
        updates=0,
        grad_accum=4,
        lr=1e-4,
        seed=a.seed,
        pair_schedule_seed=42,
        states=list(STATES),
        mask_mode='g64_proposal_energy_expand',
        repo_commit=git_head(),
    )
    expected = {k: lock[k] for k in FORMAL_LOCK_KEYS if k in lock}
    hard_verify_lock(lock, expected, formal=a.formal)
    hard_verify_lock(lock, {k: live[k] for k in expected}, formal=a.formal)

    thr = float(lock['energy_threshold'])
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       split_path)
    mmaps = {
        'train': json.load(open(mmap_tr)),
        'dev': json.load(open(mmap_dv)),
    }
    proposal = load_proposal(proposal_path, a.device)
    probe = V3A5DEvidenceProbe(proposal.proposal).to(a.device).eval()
    init = torch.load(init_path, map_location='cpu')
    z_init = V3A5D2Verifier('A1_multiscale').to(a.device).eval()
    z_init.load_state_dict(init['A1_decision_mse'], strict=True)
    for p in z_init.parameters():
        p.requires_grad_(False)
    print('F1 residual max|w|=%.3e (expect 0)' % z_init.context_residual_max_abs(),
          flush=True)
    blob = torch.load(a.f2_ckpt, map_location='cpu')
    z_tr = V3A5D2Verifier('A1_multiscale').to(a.device).eval()
    z_tr.load_state_dict(blob['model'], strict=True)
    for p in z_tr.parameters():
        p.requires_grad_(False)
    print('F2 residual max|w|=%.3e step=%s' % (
        z_tr.context_residual_max_abs(), blob.get('step')), flush=True)
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)

    data_b, data_i = {}, {}
    for tag in ('train', 'dev'):
        print('=== collect %s ===' % tag, flush=True)
        ds = make_dataset(
            ns, splits[tag],
            os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmaps[tag])
        b, imgs = collect_split(
            probe, z_init, z_tr, proposal, ds, len(splits[tag]), geom, thr,
            a.device, tag, per_image=a.per_image, seed=a.seed, limit=a.limit)
        data_b[tag] = b
        data_i[tag] = imgs
        print('  blocks=%d  images=%d' % (b['U'].shape[0], len(imgs)), flush=True)

    np.savez_compressed(
        os.path.join(a.root, 'diagnostics', 'block_rows.npz'),
        train_F0=data_b['train']['F0'], train_F1=data_b['train']['F1'],
        train_F2=data_b['train']['F2'], train_U=data_b['train']['U'],
        train_image_id=data_b['train']['image_id'],
        train_pair_id=data_b['train']['pair_id'],
        train_state=data_b['train']['state'],
        train_gy=data_b['train']['gy'], train_gx=data_b['train']['gx'],
        dev_F0=data_b['dev']['F0'], dev_F1=data_b['dev']['F1'],
        dev_F2=data_b['dev']['F2'], dev_U=data_b['dev']['U'],
        dev_image_id=data_b['dev']['image_id'],
        dev_pair_id=data_b['dev']['pair_id'],
        dev_state=data_b['dev']['state'],
        dev_gy=data_b['dev']['gy'], dev_gx=data_b['dev']['gx'])

    def img_mat(rows, key):
        X = np.stack([r[key] for r in rows]).astype(np.float64)
        y = np.array([r['U'] for r in rows], dtype=np.float64)
        return X, y

    dump_json(os.path.join(a.root, 'diagnostics', 'image_meta.json'), json_ready([
        {k: (v.tolist() if isinstance(v, np.ndarray) else v)
         for k, v in r.items() if k not in ('F0', 'F1', 'F2')}
        for r in data_i['train'] + data_i['dev']]))
    np.savez_compressed(
        os.path.join(a.root, 'diagnostics', 'image_rows.npz'),
        train_F0=np.stack([r['F0'] for r in data_i['train']]),
        train_F1=np.stack([r['F1'] for r in data_i['train']]),
        train_F2=np.stack([r['F2'] for r in data_i['train']]),
        train_U=np.array([r['U'] for r in data_i['train']]),
        train_image_id=np.array([r['image_id'] for r in data_i['train']]),
        train_state=np.array([r['state'] for r in data_i['train']]),
        dev_F0=np.stack([r['F0'] for r in data_i['dev']]),
        dev_F1=np.stack([r['F1'] for r in data_i['dev']]),
        dev_F2=np.stack([r['F2'] for r in data_i['dev']]),
        dev_U=np.array([r['U'] for r in data_i['dev']]),
        dev_image_id=np.array([r['image_id'] for r in data_i['dev']]),
        dev_state=np.array([r['state'] for r in data_i['dev']]),
    )

    # distributions
    dist = {'block': {}, 'image': {}}
    for spl in ('train', 'dev'):
        dist['block'][spl] = {'overall': utility_target_stats(data_b[spl]['U'])}
        for st in STATES:
            m = data_b[spl]['state'] == st
            dist['block'][spl][st] = utility_target_stats(data_b[spl]['U'][m])
        uy = np.array([r['U'] for r in data_i[spl]])
        dist['image'][spl] = {'overall': utility_target_stats(uy)}
        for st in STATES:
            uy_s = np.array([r['U'] for r in data_i[spl] if r['state'] == st])
            dist['image'][spl][st] = utility_target_stats(uy_s)
    dump_json(os.path.join(a.root, 'diagnostics', 'utility_distributions.json'),
              json_ready(dist))

    device = a.device if str(a.device).startswith('cuda') else 'cpu'
    block_probe, image_probe = {}, {}
    block_packs, image_pred_dev = {}, {}

    print('=== block probes ===', flush=True)
    for feat in ('F0', 'F1', 'F2'):
        print('  %s' % feat, flush=True)
        row, _, pred_dv, pack = probe_pair(
            data_b['train'][feat], data_b['train']['U'],
            data_b['dev'][feat], data_b['dev']['U'],
            'block_%s' % feat, device, a.mlp_steps)
        block_probe[feat] = row
        block_packs[feat] = pack

    print('=== image probes ===', flush=True)
    for feat in ('F0', 'F1', 'F2'):
        print('  %s' % feat, flush=True)
        Xtr, ytr = img_mat(data_i['train'], feat)
        Xdv, ydv = img_mat(data_i['dev'], feat)
        row, _, pred_dv, pack = probe_pair(
            Xtr, ytr, Xdv, ydv, 'image_%s' % feat, device, min(a.mlp_steps, 800))
        image_probe[feat] = row
        image_pred_dev[feat] = pred_dv

    dump_json(os.path.join(a.root, 'diagnostics', 'block_probe.json'),
              json_ready(block_probe))
    dump_json(os.path.join(a.root, 'diagnostics', 'image_probe.json'),
              json_ready(image_probe))

    print('=== NN F2 ===', flush=True)
    mu, sd = std_fit(data_b['train']['F2'])
    nn_block = nn_regimes_e1b(
        data_b['train']['F2'], data_b['train']['U'],
        data_b['train']['image_id'], data_b['train']['pair_id'],
        data_b['train']['state'], data_b['train']['gy'], data_b['train']['gx'],
        k=5, max_q=a.nn_max, seed=a.seed)
    nn_d2t = nn_dev_to_train(
        apply_std(data_b['train']['F2'], mu, sd), data_b['train']['U'],
        apply_std(data_b['dev']['F2'], mu, sd), data_b['dev']['U'],
        k=5, max_q=a.nn_max, seed=a.seed + 1)
    imu, isd = std_fit(img_mat(data_i['train'], 'F2')[0])
    Xitr, yitr = img_mat(data_i['train'], 'F2')
    Xidv, yidv = img_mat(data_i['dev'], 'F2')
    nn_img_d2t = nn_dev_to_train(
        apply_std(Xitr, imu, isd), yitr, apply_std(Xidv, imu, isd), yidv,
        k=5, max_q=min(a.nn_max, Xidv.shape[0]), pool_n=Xitr.shape[0],
        seed=a.seed + 2)
    # strict cross-image on train images
    nn_img_cross = []
    Xs = apply_std(Xitr, imu, isd)
    ids = np.array([r['image_id'] for r in data_i['train']])
    k_img = 5
    for i in range(len(yitr)):
        others = np.flatnonzero(ids != ids[i])
        if others.size < 1:
            nn_img_cross.append(float('nan'))
            continue
        kk = min(k_img, others.size)
        d = np.sqrt(((Xs[others] - Xs[i]) ** 2).sum(axis=1))
        nn = others[np.argpartition(d, kk - 1)[:kk]]
        nn_img_cross.append(float(yitr[nn].mean()))
    nn_image = dict(
        strict_cross_image=dict(
            spearman=spearman(yitr, nn_img_cross),
            pearson=pearson(yitr, nn_img_cross),
            mae=mae(yitr, nn_img_cross),
            sign_agree=float(np.mean((np.array(nn_img_cross) > 0) == (yitr > 0))),
            n=len(yitr)),
        dev_to_train=nn_img_d2t,
    )
    dump_json(os.path.join(a.root, 'diagnostics', 'nn.json'), json_ready(
              dict(block_F2=nn_block, block_F2_dev_to_train=nn_d2t,
                   image_F2=nn_image)))

    print('=== regret / decision ===', flush=True)
    regret = {}
    for feat in ('F0', 'F1', 'F2'):
        regret['image_%s' % feat] = image_decision_from_pred(
            data_i['dev'], image_pred_dev[feat])
    # always-base / always-r1 / oracle-sign
    def const_dec(images, q):
        psnr, regrets = [], []
        for im in images:
            p = im['psnr_r1'] if q else im['psnr_base']
            mse = im['mse_r1'] if q else im['mse_base']
            psnr.append(p)
            regrets.append(decision_regret(mse, im['mse_base'], im['mse_r1']))
        rg = np.asarray(regrets)
        return dict(mean_psnr=float(np.mean(psnr)), mean_regret=float(rg.mean()),
                    median_regret=float(np.median(rg)),
                    p90_regret=float(np.percentile(rg, 90)), q=q)
    regret['always_base'] = const_dec(data_i['dev'], 0)
    regret['always_r1'] = const_dec(data_i['dev'], 1)
    # train-chosen global accept/reject
    tr_psnr0 = float(np.mean([r['psnr_base'] for r in data_i['train']]))
    tr_psnr1 = float(np.mean([r['psnr_r1'] for r in data_i['train']]))
    gq = 1 if tr_psnr1 >= tr_psnr0 else 0
    regret['global_constant'] = const_dec(data_i['dev'], gq)
    regret['global_constant']['chosen_from_train_q'] = gq
    oracle_q = np.array([1.0 if r['U'] > 0 else 0.0 for r in data_i['dev']])
    regret['oracle_sign'] = image_decision_from_pred(data_i['dev'], oracle_q)

    ds_dev = make_dataset(
        ns, splits['dev'],
        os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmaps['dev'])
    print('  block F2 full-map decision (dev)', flush=True)
    regret['block_F2'] = block_decision_psnr(
        z_tr, 'F2', block_packs['F2'], ds_dev, len(splits['dev']), geom, thr,
        proposal, a.device, limit=a.limit)
    dump_json(os.path.join(a.root, 'diagnostics', 'regret.json'), json_ready(regret))

    table = {}
    for level, store in (('block', block_probe), ('image', image_probe)):
        for feat in ('F0', 'F1', 'F2'):
            r = store[feat]
            key = '%s_%s' % (level, feat)
            table[key] = dict(
                train_corr=r['train_corr'], dev_corr=r['dev_corr'],
                dev_r2=r['dev_r2'], auroc=r['auroc'],
                decision_psnr=regret.get(key, {}).get('mean_psnr'),
                mean_regret=regret.get(key, {}).get('mean_regret'),
                bins=r.get('bins'),
            )
    # image decision psnr already in regret['image_F*']
    for feat in ('F0', 'F1', 'F2'):
        table['image_%s' % feat]['decision_psnr'] = regret['image_%s' % feat]['mean_psnr']
        table['image_%s' % feat]['mean_regret'] = regret['image_%s' % feat]['mean_regret']
    table['block_F2']['decision_psnr'] = regret['block_F2']['mean_psnr']
    table['block_F2']['mean_regret'] = regret['block_F2']['mean_regret']

    dump_json(os.path.join(a.root, 'diagnostics', 'summary_table.json'),
              json_ready(table))
    verdict = verdict_v3a71(table, regret)
    verdict['distributions'] = {
        'block_train_pos': dist['block']['train']['overall']['pos_frac'],
        'image_train_pos': dist['image']['train']['overall']['pos_frac'],
        'image_dev_mean_U': dist['image']['dev']['overall']['mean'],
    }
    dump_json(os.path.join(a.root, 'diagnostics', 'verdict.json'),
              json_ready(verdict))

    print('VERDICT', verdict['label'], 'next=', verdict['next_step'], flush=True)
    print('  meaning:', verdict['meaning'], flush=True)
    print('  block F2  train/dev corr=%.3f/%.3f auroc=%.3f'
          % (block_probe['F2']['train_corr'], block_probe['F2']['dev_corr'],
             block_probe['F2']['auroc']), flush=True)
    print('  image F2  train/dev corr=%.3f/%.3f auroc=%.3f'
          % (image_probe['F2']['train_corr'], image_probe['F2']['dev_corr'],
             image_probe['F2']['auroc']), flush=True)
    print('  regret image_F2=%.5f  block_F2=%.5f  global_const=%.5f  oracle=%.5f'
          % (regret['image_F2']['mean_regret'], regret['block_F2']['mean_regret'],
             regret['global_constant']['mean_regret'],
             regret['oracle_sign']['mean_regret']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
