"""V3-A.4.1 runtime: pure diagnostics, no training anywhere.

Two questions:
  A. where does q_opt change -- full vs crop proposal, full vs crop target?
  B. how much of the spatial H/4 oracle headroom does a *global scalar* gate
     already capture?

Everything here is read-only: ``torch.no_grad()`` at the call sites, no
parameters changed, no checkpoint written.
"""

import csv
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

Q_FACTOR = 4


# ── §5 q_opt decomposition ──────────────────────────────────────────────────

def qopt_components(y0, hr, d_correction, factor=Q_FACTOR, eps=1e-8):
    """-> dict(q_opt, q_raw, N, Z, energy), all at H/factor.

    Numerically identical to ``v3a2_runtime.action_optimal_gate`` for q_opt.
    ``d_correction`` must already contain the proposal gate.
    """
    h, w = y0.shape[-2:]
    th, tw = max(1, h // factor), max(1, w // factor)
    y4 = F.interpolate(y0, size=(th, tw), mode='area')
    h4 = F.interpolate(hr, size=(th, tw), mode='area')
    d4 = F.interpolate(d_correction, size=(th, tw), mode='area')
    N = ((h4 - y4) * d4).sum(dim=1, keepdim=True)
    Z = (d4 ** 2).sum(dim=1, keepdim=True)
    q_raw = N / (Z + eps)
    return dict(q_opt=q_raw.clamp(0.0, 1.0), q_raw=q_raw, N=N, Z=Z,
                energy=(d4 ** 2).mean(dim=1, keepdim=True))


def global_action_optimal_gate(y0, hr, d_correction, eps=1e-8):
    """§10: one scalar per image, minimising global RGB-MSE along D."""
    N = ((hr - y0) * d_correction).sum(dim=(1, 2, 3), keepdim=True)
    Z = (d_correction ** 2).sum(dim=(1, 2, 3), keepdim=True)
    q = (N / (Z + eps)).clamp(0.0, 1.0)
    return q, N, Z


# ── §6 statistics ───────────────────────────────────────────────────────────

def _q(t, ps=(10, 25, 50, 75, 90)):
    f = t.flatten().float()
    out = {('p%d' % p): float(torch.quantile(f, p / 100.0)) for p in ps}
    return out


def aggregate_qopt_stats(comp, eps_energy):
    """-> the full §6 statistic block for one (split, mode, state)."""
    q, qr, N, Z, e = (comp[k] for k in ('q_opt', 'q_raw', 'N', 'Z', 'energy'))
    s = {}
    s['q_opt'] = dict(mean=float(q.mean()), std=float(q.std()), **_q(q))
    s['q_opt'].update(frac_eq0=float((q == 0).float().mean()),
                      frac_eq1=float((q == 1).float().mean()))
    s['q_raw'] = dict(mean=float(qr.mean()), median=float(qr.median()),
                      **_q(qr, ps=(10, 90)))
    s['q_raw'].update(frac_lt0=float((qr < 0).float().mean()),
                      frac_gt1=float((qr > 1).float().mean()))
    s['N'] = dict(mean=float(N.mean()), median=float(N.median()),
                  **_q(N, ps=(10, 90)), frac_gt0=float((N > 0).float().mean()))
    s['Z'] = dict(mean=float(Z.mean()), median=float(Z.median()),
                  **_q(Z, ps=(10, 90)))
    s['energy'] = dict(mean=float(e.mean()), **_q(e, ps=(10, 50, 90)),
                       frac_gt_eps=float((e > eps_energy).float().mean()))
    return s


# ── §4 deterministic crop manifest ──────────────────────────────────────────

def build_fixed_crop_manifest(rows, split, crop=128, k=16, seed=20260927):
    """Deterministic (sample, crop_id) -> (top, left). Written once, then read.

    The seed must not come from Python's builtin ``hash()``: string hashing is
    salted per process (PYTHONHASHSEED), so the "fixed" manifest would differ
    between runs and machines.
    """
    out = []
    for name, low, _high in rows:
        from PIL import Image
        w, h = Image.open(low).size          # only the size is needed here
        rng = np.random.default_rng(stable_seed(seed, name))
        for cid in range(k):
            top = int(rng.integers(0, h - crop + 1)) if h > crop else 0
            left = int(rng.integers(0, w - crop + 1)) if w > crop else 0
            out.append(dict(sample_id=name, split=split, crop_id=cid,
                            top=top, left=left, height=crop, width=crop,
                            seed=seed))
    return out


def stable_seed(*parts):
    """Process/machine-independent 32-bit seed from any tuple of values."""
    s = '|'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(s).digest()[:8], 'little') % (2 ** 32)


def verify_v3a41_artifact_lock(root, src_root,
                               ckpt_name='R1_v2stable_naive_s42/checkpoint_03000.pt',
                               cache_name='cache_y0_lolbase', v4_root=None):
    """Re-check every SHA in the V3-A.4.1 lock before a diagnostic runs."""
    lock_path = os.path.join(root, 'artifact_lock.json')
    if not os.path.isfile(lock_path):
        raise SystemExit('audit artifact lock missing: %s' % lock_path)
    lock = json.load(open(lock_path, encoding='utf-8'))
    v4 = v4_root or '/root/data/experiments/v3a4_lolv2real'
    paths = {
        'proposal_sha256': os.path.join(src_root, ckpt_name),
        'cache_metadata_sha256': os.path.join(src_root, cache_name,
                                              'refiner_train', 'metadata.json'),
        'manifest_sha256': os.path.join(src_root, 'manifests', 'refiner_train.csv'),
        'split_sha256': os.path.join(v4, 'splits', 'split.json'),
        'mismatch_train_sha256': os.path.join(v4, 'mappings',
                                              'mismatch_train_575.json'),
        'mismatch_dev_sha256': os.path.join(v4, 'mappings',
                                            'mismatch_dev_64.json'),
        'energy_stats_sha256': os.path.join(v4, 'action_stats', 'energy.json'),
        'crop_manifest_sha256': os.path.join(root, 'crops', 'crop_manifest.csv'),
    }
    bad = []
    for key, p in paths.items():
        if key not in lock:
            raise SystemExit('audit lock missing %s' % key)
        if not os.path.isfile(p):
            raise SystemExit('locked artifact not found: %s' % p)
        if hashlib.sha256(open(p, 'rb').read()).hexdigest() != lock[key]:
            bad.append(key)
    if bad:
        raise SystemExit('audit lock mismatch on %s -- rerun '
                         'scripts/setup_v3a41_audit.py' % ', '.join(bad))
    return lock


def read_crop_manifest(path):
    return list(csv.DictReader(open(path, encoding='utf-8')))


def crop_tensor(t, top, left, h, w):
    """Crop [B,C,H,W] -> [B,C,h,w] at (top,left). Same coords for every input."""
    return t[..., int(top):int(top) + int(h), int(left):int(left) + int(w)]


# ── §8 proposal action consistency ──────────────────────────────────────────

def compare_actions(d_full_crop, d_crop):
    """-> dict of MAE / RMSE / cosine / norm ratio between two correction fields."""
    a, b = d_full_crop.flatten().float(), d_crop.flatten().float()
    diff = a - b
    na, nb = float(a.norm()), float(b.norm())
    cos = float((a * b).sum() / (na * nb + 1e-12))
    return dict(MAE=float(diff.abs().mean()), RMSE=float(diff.pow(2).mean().sqrt()),
                cosine=cos, norm_full=na, norm_crop=nb,
                norm_ratio=nb / (na + 1e-12))


# ── helpers ─────────────────────────────────────────────────────────────────

def load_v3a4_lock(root):
    return json.load(open(os.path.join(root, 'artifact_lock.json'), encoding='utf-8'))
