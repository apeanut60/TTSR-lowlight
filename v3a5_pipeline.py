"""V3-A.5 shared plumbing: data, frozen proposal, per-sample tensors.

Imported only by the V3-A.5 scripts (after they stash sys.argv), because it
pulls in the dataset module, which imports ``option`` and parses args at import
time. Pure target/mask/metric logic stays in ``v3a5_runtime``.
"""

import json

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest
from model.V3A4Verifier import V3A4Refiner
from v3a4_runtime import load_r1_proposal_strict
from v3a_runtime import exposure_gain


def load_proposal(ckpt_path, device):
    """Frozen R1 proposal inside its V3-A.4 wrapper (the verifier is unused)."""
    model = V3A4Refiner('none').to(device).eval()
    load_r1_proposal_strict(model, ckpt_path, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.proposal.eval()
    return model


def load_rows(manifest_path, split_json):
    """-> dict of pair-lists per split tag, from the refiner manifest."""
    pairs = {p[0]: p for p in pairs_from_manifest(manifest_path)}
    split = json.load(open(split_json, encoding='utf-8'))
    return {tag: [pairs[i] for i in split[tag]] for tag in ('train', 'dev')}


def make_dataset(args_ns, rows, y0_cache, mismatch_map):
    """Full-image (crop_size=0) dataset with the split-safe mismatch donor."""
    return TrainSet(args_ns, crop_size=0, pairs=rows, y0_cache=y0_cache,
                    mismatch_map=mismatch_map, split='Train')


def sample_tensors(ds, i, state, device):
    """-> dict(name, X, Y0, H, R) as [1,3,H,W] tensors on ``device``.

    ``X`` is the true low-light input; ``R`` is the reference for this state:
    correct / true_dark_g0.5 (exposure_gain 0.5) / mismatch (donor reference).
    """
    name, lr, hr, ref, y0, mis = ds._load(i)
    if y0 is None:
        raise SystemExit('%s: no Y0 in the cache' % name)
    if state == 'correct':
        r = ref
    elif state == 'true_dark_g0.5':
        r = exposure_gain(ref, 0.5)
    elif state == 'mismatch':
        if mis is None:
            raise SystemExit('%s: no mismatch donor was loaded' % name)
        r = mis
    else:
        raise SystemExit('unknown state %r' % state)
    return dict(name=name, X=lr[None].to(device), Y0=y0[None].to(device),
                H=hr[None].to(device), R=r[None].to(device))


def correction(proposal, y0, reference):
    """-> (D, sr) where D = g_v2 * delta and sr = Y0 + D (the R1 output)."""
    sr, aux = proposal(y0, reference)
    return aux['gate'] * aux['delta'], sr
