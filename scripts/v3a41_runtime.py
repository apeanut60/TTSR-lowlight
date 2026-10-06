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
import subprocess

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
            # drawn AFTER top/left, in the same order the dataset's _geometry
            # draws them, so the audit's geometry is representative of training
            rot_k = int(rng.integers(0, 4))
            flip_h = int(rng.integers(0, 2))
            flip_w = int(rng.integers(0, 2))
            out.append(dict(sample_id=name, split=split, crop_id=cid,
                            top=top, left=left, height=crop, width=crop,
                            rot_k=rot_k, flip_h=flip_h, flip_w=flip_w, seed=seed))
    return out


def apply_geometry(t, rot_k, flip_h, flip_w):
    """Mirror ``dataset.lolv2real_v3a._geometry``'s post-crop order exactly:
    rot90 -> flip(vertical) -> flip(horizontal), on the two SPATIAL axes.

    ``_geometry`` operates on ``[C,H,W]`` tensors, so its ``dims=(1,2)`` are
    (H,W). The audit works on ``[B,C,H,W]``, where the spatial axes are (2,3) --
    using (1,2) there rotates channel-against-height, which is not a geometry
    augmentation at all. That mistake was caught by the FC/FCG control.
    """
    if t.dim() == 4:
        hd, wd = 2, 3
    elif t.dim() == 3:
        hd, wd = 1, 2
    else:
        raise ValueError('apply_geometry expects [B,C,H,W] or [C,H,W], got %s'
                         % (tuple(t.shape),))
    if int(rot_k):
        t = torch.rot90(t, int(rot_k), dims=(hd, wd))
    if int(flip_h):
        t = torch.flip(t, dims=(hd,))
    if int(flip_w):
        t = torch.flip(t, dims=(wd,))
    return t.contiguous()


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
    # the lock also pins the code commit: the diagnostics are only valid for
    # the revision they were generated with. Re-run setup after any code change.
    try:
        head = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                            # noqa: BLE001
        head = None
    if head and lock.get('repo_commit') and lock['repo_commit'] != head:
        raise SystemExit('audit lock was generated at commit %s but HEAD is %s '
                         '-- rerun scripts/setup_v3a41_audit.py on the final '
                         'revision' % (lock['repo_commit'][:8], head[:8]))
    return lock


def read_crop_manifest(path):
    return list(csv.DictReader(open(path, encoding='utf-8')))


def crop_tensor(t, top, left, h, w):
    """Crop [B,C,H,W] -> [B,C,h,w] at (top,left). Same coords for every input."""
    return t[..., int(top):int(top) + int(h), int(left):int(left) + int(w)]


# ── §3/§4 the five q_opt modes ──────────────────────────────────────────────
#
#   FF   full proposal -> full target            (what the dev evaluator does)
#   FC   full proposal -> crop target            (isolates the crop/target effect)
#   CC   crop proposal -> crop target            (no geometry augmentation)
#   CCG  crop+geometry proposal -> crop+geom target   <- what training sees
#   FCG  full proposal, cropped+geometry target  (control: geometry must not
#        change the target algebra, only the proposal's response to it)

def mode_ff(y0, hr, d_full):
    return qopt_components(y0, hr, d_full)


def mode_fc(y0, hr, d_full, box):
    top, left, sz, _ = box
    return qopt_components(crop_tensor(y0, top, left, sz, sz),
                           crop_tensor(hr, top, left, sz, sz),
                           crop_tensor(d_full, top, left, sz, sz))


def mode_cc(y0, hr, ref, box, proposal_fn):
    top, left, sz, _ = box
    y0c = crop_tensor(y0, top, left, sz, sz)
    rc = crop_tensor(ref, top, left, sz, sz)
    _sr, aux = proposal_fn(y0c, rc)
    D = aux['gate'] * aux['delta']
    return qopt_components(y0c, crop_tensor(hr, top, left, sz, sz), D), aux, D


def mode_ccg(y0, hr, ref, box, proposal_fn):
    """The mode that matches verifier training: crop, THEN the same rot/flip
    the dataset applies, then the proposal."""
    top, left, sz, (k, fh, fw) = box
    y0g = apply_geometry(crop_tensor(y0, top, left, sz, sz), k, fh, fw)
    hrg = apply_geometry(crop_tensor(hr, top, left, sz, sz), k, fh, fw)
    rg = apply_geometry(crop_tensor(ref, top, left, sz, sz), k, fh, fw)
    _sr, aux = proposal_fn(y0g, rg)
    D = aux['gate'] * aux['delta']
    return qopt_components(y0g, hrg, D), aux, D


def mode_fcg(y0, hr, d_full, box):
    top, left, sz, (k, fh, fw) = box
    return qopt_components(
        apply_geometry(crop_tensor(y0, top, left, sz, sz), k, fh, fw),
        apply_geometry(crop_tensor(hr, top, left, sz, sz), k, fh, fw),
        apply_geometry(crop_tensor(d_full, top, left, sz, sz), k, fh, fw))


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
