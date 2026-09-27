"""V3-A.4 runtime: protocol fixes + normalized action conditioning.

Fixes four defects found in the V3-A.3 code:

1. ``gap_pred/gap_gt`` used ``mean([q_correct, -q_h1, -q_h2])``, which divides
   by 3 instead of averaging the harmful states, so the gap changed sign
   (with the V3-A.3 numbers: -0.146 instead of +0.147). Because the same wrong
   formula was used for the target, the two agreed with each other and the
   "ordering recovered" claim was comparing two identically-wrong numbers.
2. the R1 checkpoint was loaded with ``m.proposal.load_state_dict(ck['model'],
   strict=False)`` -- prefixed keys match nothing, the proposal keeps its
   zero-init, and the gate-endpoint test then passes *trivially* because
   q=1 and q=0 both return Y0. ``load_r1_proposal_strict`` makes that
   impossible.
3. train and dev shared one mismatch map built over all 639 samples, so a
   train donor could be a dev image.
4. ``masked_smooth_l1`` was reported under the name ``masked_mae``.
"""

import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from v3a2_runtime import masked_smooth_l1                       # re-export

__all__ = ['load_r1_proposal_strict', 'gap', 'masked_mae', 'masked_rmse',
           'pixel_corr', 'masked_smooth_l1', 'action_features_raw',
           'action_features_norm', 'compute_action_rms', 'build_mismatch_map',
           'assert_action_connectivity']


# ── 2.4 strict proposal loading ─────────────────────────────────────────────

def load_r1_proposal_strict(model, ckpt_path, device='cpu'):
    """Load ONLY the proposal weights, strictly, and refuse a degenerate source.

    ``model.load_state_dict(ck['model'], strict=False)`` on the *submodule* is
    the trap this replaces: every key misses, nothing raises, and the proposal
    stays at its zero initialisation.
    """
    ck = torch.load(ckpt_path, map_location=device)
    full = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    pref = {k[len('proposal.'):]: v for k, v in full.items()
            if k.startswith('proposal.')}
    target = model.proposal.state_dict()
    if len(pref) != len(target):
        raise SystemExit('proposal key count mismatch: %d in checkpoint vs %d '
                         'in model (%s)' % (len(pref), len(target), ckpt_path))
    model.proposal.load_state_dict(pref, strict=True)
    for k in ('c_out.weight', 'c_out.bias'):
        if k in pref and float(pref[k].abs().max()) == 0.0:
            raise SystemExit('%s is all-zero in %s -- proposal is degenerate'
                             % (k, ckpt_path))
    return model


def verify_v3a4_artifact_lock(root, src_root, ckpt_name='R1_v2stable_naive_s42/checkpoint_03000.pt',
                              cache_name='cache_y0_lolbase'):
    """Re-check EVERY SHA in artifact_lock.json at run time.

    The acceptance script already does this once, but a training or evaluation
    entry point that only checks the proposal and the cache will happily run on
    a split / mismatch map / stats file that changed after acceptance.
    """
    lock_path = os.path.join(root, 'artifact_lock.json')
    if not os.path.isfile(lock_path):
        raise SystemExit('artifact lock missing: %s' % lock_path)
    lock = json.load(open(lock_path, encoding='utf-8'))
    paths = {
        'proposal_sha256': os.path.join(src_root, ckpt_name),
        'cache_metadata_sha256': os.path.join(src_root, cache_name,
                                              'refiner_train', 'metadata.json'),
        'manifest_sha256': os.path.join(src_root, 'manifests', 'refiner_train.csv'),
        'split_sha256': os.path.join(root, 'splits', 'split.json'),
        'mismatch_train_sha256': os.path.join(root, 'mappings',
                                              'mismatch_train_575.json'),
        'mismatch_dev_sha256': os.path.join(root, 'mappings',
                                            'mismatch_dev_64.json'),
        'energy_stats_sha256': os.path.join(root, 'action_stats', 'energy.json'),
        'action_norm_sha256': os.path.join(root, 'action_stats', 'action_norm.json'),
    }
    drift = []
    for key, p in paths.items():
        if key not in lock:
            raise SystemExit('artifact lock is missing %s' % key)
        if not os.path.isfile(p):
            raise SystemExit('locked artifact not found: %s' % p)
        if _sha(p) != lock[key]:
            drift.append(key)
    if drift:
        raise SystemExit('artifact lock mismatch on: %s -- rerun '
                         'scripts/setup_v3a4_protocol.py' % ', '.join(drift))
    return lock


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def freeze_proposal(model):
    """Freeze AND keep eval mode. ``model.train()`` afterwards re-enables it."""
    for p in model.proposal.parameters():
        p.requires_grad_(False)
    model.proposal.eval()
    if model.proposal.training:
        raise AssertionError('proposal is in training mode')
    return {k: v.clone() for k, v in model.proposal.state_dict().items()}


# ── 2.1 correct correct-vs-harmful gap ──────────────────────────────────────

def gap(qv_map, qo_map, harmful):
    """gap = q(correct) - mean_over_harmful q(harmful). One formula for both."""
    def _g(m):
        c = float(m['correct'].mean())
        h = float(np.mean([float(m[s].mean()) for s in harmful]))
        return c - h
    gp, gt = _g(qv_map), _g(qo_map)
    return gp, gt, gp - gt


# ── 2.2 real masked metrics ─────────────────────────────────────────────────

def masked_mae(q_v, q_opt, mask):
    d = (q_v - q_opt).abs()
    m = mask.to(d.dtype)
    return float((m * d).sum() / m.sum().clamp(min=1.0))


def masked_rmse(q_v, q_opt, mask):
    d = (q_v - q_opt) ** 2
    m = mask.to(d.dtype)
    return float(torch.sqrt((m * d).sum() / m.sum().clamp(min=1.0)))


def pixel_corr(q_v, q_opt, mask):
    sel = mask > 0
    if int(sel.sum()) < 2:
        return float('nan')
    a = q_v[sel].flatten().float()
    b = q_opt[sel].flatten().float()
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float('nan')
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


# ── 5. train-only action normalization ──────────────────────────────────────

def action_features_raw(d_correction, factor=4):
    h, w = d_correction.shape[-2:]
    th, tw = max(1, h // factor), max(1, w // factor)
    d4 = F.interpolate(d_correction, size=(th, tw), mode='area')
    e4 = (d4 ** 2).mean(dim=1, keepdim=True)
    return d4, e4


def action_features_norm(d_correction, rms, eps=1e-6, factor=4):
    """D4/rms per channel, then E4 from the normalized D. No mean subtraction:
    D = 0 has the explicit meaning "no correction" and must stay 0."""
    d4, _ = action_features_raw(d_correction, factor)
    r = torch.as_tensor(rms, dtype=d4.dtype, device=d4.device).view(1, -1, 1, 1)
    d4n = d4 / (r + eps)
    e4n = (d4n ** 2).mean(dim=1, keepdim=True)
    return d4n, e4n


def compute_action_rms(model, pairs, ds, device):
    """Per-channel RMS of D4 over verifier-train correct references only."""
    acc = None
    n = 0
    with torch.no_grad():
        for i in range(len(pairs)):
            _name, _lr, _hr, ref, y0, _m = ds._load(i)
            _sr, aux = model.proposal(y0[None].to(device), ref[None].to(device))
            d4, _e4 = action_features_raw(aux['gate'] * aux['delta'])
            s = (d4 ** 2).mean(dim=(0, 2, 3)).cpu()
            acc = s if acc is None else acc + s
            n += 1
    return (acc / n).sqrt().tolist()


# ── 2.3 split-isolated mismatch maps ────────────────────────────────────────

def build_mismatch_map(ids, seed=20260927):
    """Cyclic derangement WITHIN one split. Never crosses a split boundary."""
    ids = list(ids)
    if len(ids) < 2:
        raise SystemExit('need >=2 ids for a derangement')
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(1, len(ids)))
    perm = {ids[i]: ids[(i + shift) % len(ids)] for i in range(len(ids))}
    if any(k == v for k, v in perm.items()):
        raise AssertionError('fixed point in derangement')
    return perm


# ── 8. action connectivity ──────────────────────────────────────────────────

def assert_action_connectivity(model, action_ch, device='cpu'):
    """Forward + backward: a nonzero action channel must move the output.

    Forward: zero the common weights, set one action weight to 1, and check that
    changing D changes the verifier output. Backward: the action weights must
    receive gradient.
    """
    # A zero-initialised proposal gives D == 0 for every reference, so the
    # action channels would be identically zero and this check would fail for
    # the wrong reason. Make the proposal non-degenerate first.
    with torch.no_grad():
        model.proposal.c_out.weight.normal_(0, 0.3)
        model.proposal.c_out.bias.normal_(0, 0.05)
    g = torch.Generator().manual_seed(0)
    y0 = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
    low = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
    ref = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
    ref2 = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1

    saved = {k: v.clone() for k, v in model.verifier.head0.state_dict().items()}
    with torch.no_grad():
        model.verifier.head0.weight.zero_()
        model.verifier.head0.bias.zero_()
        model.verifier.head0.weight[0, -action_ch] = 1.0
        _o1, a1 = model(y0, ref, low=low)
        _o2, a2 = model(y0, ref2, low=low)
    fwd = float((a1['q_v4'] - a2['q_v4']).abs().max())

    for k, v in saved.items():
        model.verifier.head0.state_dict()[k].copy_(v)
    model.train()
    _o, aux = model(y0, ref, low=low)
    aux['q_v4'].mean().backward()
    gw = model.verifier.head0.weight.grad
    bwd = 0.0 if gw is None else float(gw[:, -action_ch:].abs().max())
    model.zero_grad(set_to_none=True)
    return fwd, bwd
