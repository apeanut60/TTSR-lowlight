"""Shared runtime for the V2 local-reference-refinement experiment.

Holds everything that must be *identical* across the two arms and across
training/evaluation, so that "the two arms differ only in the second input"
is enforced by construction rather than by convention:

  * loading the frozen N0 and running it (tiled) to produce Y0;
  * the strict reference manifest (no sorted-index guessing, no HR fallback);
  * the Y0 float32 cache (both arms read the same base output);
  * the metric implementation used by every condition.
"""

import hashlib
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model.TTSREnhance as TTE                                # noqa: E402
from option import parser as option_parser                     # noqa: E402
from trainer import Trainer                                    # noqa: E402
from utils import calc_psnr_and_ssim                           # noqa: E402


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_args_from_run(run_dir):
    ns = option_parser.parse_args([])
    for line in open(os.path.join(run_dir, 'args.txt'), encoding='utf-8'):
        parts = line.split()
        if len(parts) < 2:
            continue
        key, val = parts[0], parts[1]
        if val in ('True', 'False'):
            val = (val == 'True')
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        setattr(ns, key, val)
    return ns


def load_frozen_n0(base_ckpt, base_run_dir, device='cuda'):
    """Frozen N0 with every reference path off, plus a Trainer for tiling."""
    if not os.path.isfile(base_ckpt):
        raise SystemExit('base checkpoint not found: %s' % base_ckpt)
    cfg = load_args_from_run(base_run_dir)
    cfg.cpu = (device == 'cpu')
    cfg.num_gpu = 1
    cfg.dataset_dir = '/root/data/datasets/data1'
    # The V2 refiner is the only place a reference is allowed to matter.
    cfg.no_reference = True
    cfg.no_ref_texture = True
    cfg.no_ref_illum = True
    cfg.no_global_illum = True
    cfg.ref_correction = False
    model = TTE.TTSREnhance(cfg).to(device)
    _assert_base_keys_loadable(model, base_ckpt)
    model = TTE.load_pretrained_weights(model, base_ckpt, device)
    model.eval()
    quiet = logging.getLogger('n0_%d' % id(model))
    quiet.addHandler(logging.NullHandler())
    quiet.setLevel(logging.CRITICAL)
    # Build the Trainer while the parameters are still trainable -- its
    # constructor refuses to run with an empty optimizer, and we only need it
    # for `_tiled_forward`. Freeze afterwards; no optimizer step is ever taken.
    tr = Trainer(cfg, quiet, None, model, {})
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tr, cfg


def _assert_base_keys_loadable(model, base_ckpt):
    """Refuse to run on a silently partial pretrained load.

    ``load_pretrained_weights`` copies only keys whose name *and* shape match and
    reports a count; anything else keeps its random init without complaining.
    That is fine for the original transfer-learning use, but this runtime claims
    to be a *frozen* N0, so every key in the checkpoint must be loadable.
    """
    sd = torch.load(base_ckpt, map_location='cpu')
    msd = model.state_dict()
    bad = [k for k, v in sd.items()
           if k not in msd or tuple(msd[k].shape) != tuple(v.shape)]
    if bad:
        raise SystemExit(
            'base checkpoint %s has %d key(s) that cannot load into the model '
            '(first: %s); refusing a partial load'
            % (base_ckpt, len(bad), ', '.join(bad[:5])))
    return len(sd)


@torch.no_grad()
def infer_n0(model, trainer, low):
    """low: [B,3,H,W] in [-1,1] -> Y0, same shape, no clamp."""
    return trainer._tiled_forward(low, low, low, low)[0]


# ─── strict reference manifest ────────────────────────────────────────────

def _ref_for(low_path, ref_subdir='nanobanana_ref'):
    """Reference file for a low image: same camera dir, same basename."""
    cam_dir = os.path.dirname(os.path.dirname(low_path))
    d = os.path.join(cam_dir, ref_subdir)
    stem = os.path.splitext(os.path.basename(low_path))[0]
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG'):
        p = os.path.join(d, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def build_manifest(sample_ids, dataset_dir, split, ref_subdir='nanobanana_ref'):
    """Strict (camera, basename) pairing. Missing refs are an error.

    `sample_ids` are the already-verified lists from the Nano manifests, so the
    sample *set* matches the previous rounds; the mapping to a reference file is
    rebuilt here explicitly instead of relying on sorted-index order.
    """
    rows = []
    seen = set()
    for sid in sample_ids:
        low = sid if os.path.isabs(sid) else os.path.join(dataset_dir, 'Training data', sid)
        if not os.path.isfile(low):
            raise SystemExit('%s: low image missing: %s' % (split, low))
        cam = os.path.basename(os.path.dirname(os.path.dirname(low)))
        key = (cam, os.path.basename(low))
        if key in seen:
            raise SystemExit('%s: duplicate sample %s' % (split, key))
        seen.add(key)
        high = os.path.join(os.path.dirname(os.path.dirname(low)), 'high',
                            os.path.basename(low))
        if not os.path.isfile(high):
            raise SystemExit('%s: high image missing: %s' % (split, high))
        ref = _ref_for(low, ref_subdir)
        if ref is None:
            raise SystemExit('%s: no generated reference for %s — refusing to '
                             'fall back to HR' % (split, low))
        rows.append(dict(sample_id='%s/%s' % (cam, os.path.basename(low)),
                         camera=cam, low_path=low, high_path=high,
                         nano_path=ref, split=split))
    return rows


def read_manifest_ids(manifest_dir, camera):
    p = os.path.join(manifest_dir, camera + '.txt')
    with open(p, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def read_manifest_csv(path):
    """Read one of the CSV manifests written by the cache preparation step."""
    import csv as _csv
    if not os.path.isfile(path):
        raise SystemExit('manifest CSV not found: %s' % path)
    with open(path, encoding='utf-8') as f:
        return list(_csv.DictReader(f))


# ─── Y0 cache ─────────────────────────────────────────────────────────────

def cache_path(cache_dir, row):
    return os.path.join(cache_dir, row['sample_id'].replace('/', '__') + '.npy')


def prepare_cache(rows, model, trainer, cache_dir, from_uint8=None):
    """Write one float32 .npy per sample holding the full-image Y0."""
    os.makedirs(cache_dir, exist_ok=True)
    for i, row in enumerate(rows):
        low = from_uint8(row['low_path'])
        y0 = infer_n0(model, trainer, low)
        np.save(cache_path(cache_dir, row), y0[0].float().cpu().numpy())
        if (i + 1) % 50 == 0:
            print('  cached %d/%d' % (i + 1, len(rows)))
    return cache_dir


def load_cache(cache_dir, row):
    p = cache_path(cache_dir, row)
    if not os.path.isfile(p):
        raise SystemExit('Y0 cache miss: %s' % p)
    return np.load(p)


def write_metadata(cache_dir, extra):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(extra, f, indent=2, sort_keys=True)


def verify_cache(cache_dir, expect_manifest_sha=None, strict=False):
    """Check a Y0 cache is complete and self-consistent before training/eval.

    Returns the metadata dict. ``complete`` is only set by the V2.1 cache
    writer, so a missing flag means the cache predates the tiling fix -- that is
    reported loudly rather than silently accepted, because such a cache has the
    0-weight outer ring baked in.
    """
    path = os.path.join(cache_dir, 'metadata.json')
    if not os.path.isfile(path):
        raise SystemExit('cache metadata missing: %s' % path)
    meta = json.load(open(path, encoding='utf-8'))
    base = meta.get('base_checkpoint_sha256') or meta.get('base_sha256')
    if not base:
        raise SystemExit('%s records no base checkpoint hash' % path)
    meta['_base_sha'] = base
    if not meta.get('complete'):
        msg = ('cache %s has no completion flag: generated before the V2.1 '
               'tiling fix, so its outer 1px ring is 0' % cache_dir)
        if strict:
            raise SystemExit(msg)
        print('[cache] WARNING: %s' % msg)
    if expect_manifest_sha and meta.get('manifest_sha256') \
            and meta['manifest_sha256'] != expect_manifest_sha:
        raise SystemExit('cache %s was built from a different manifest' % cache_dir)
    return meta


# ─── metrics ──────────────────────────────────────────────────────────────

EVAL_QUERY_CHUNK = 1024


def metrics(sr, hr):
    """sr, hr: [1,3,H,W] in [-1,1] -> (psnr_rgb, ssim_y, mse).

    ``utils.calc_ssim`` returns the SSIM of the Y channel (its per-channel loop
    is dead code for 3-channel input), so the value must not be labelled RGB.
    """
    if sr.shape != hr.shape:
        raise ValueError('shape mismatch %s vs %s — refuse to silently crop'
                         % (tuple(sr.shape), tuple(hr.shape)))
    t = calc_psnr_and_ssim(sr.detach(), hr.detach())
    return float(t[3]), float(t[1]), float(t[2])


# ─── the single evaluation implementation shared by training and eval ─────

def mismatch_permutation(rows):
    """Fixed cyclic shift within each camera block (no fixed points)."""
    perm = list(range(len(rows)))
    for cam in sorted({r['camera'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['camera'] == cam]
        k = len(idx)
        if k > 1:
            for pos, i in enumerate(idx):
                perm[i] = idx[(pos + 1) % k]
    assert all(perm[i] != i for i in range(len(rows))), 'fixed point in permutation'
    return perm


@torch.no_grad()
def evaluate_conditions(refiner, rows, eval_cache_dir, conditions, device,
                        mismatch=None, verbose=True):
    """conditions: list of (label, mode).

    mode 'bypass' returns the cached Y0 untouched; 'self' feeds Y0 as the second
    input; 'nano' feeds the generated reference; 'mismatch' feeds a different
    sample's reference from the same camera. Every condition shares the same
    cached Y0, so they differ only in weights / second input.
    """
    refiner.eval()
    # Full images (960x720 -> 480x360 = 172,800 queries) would need ~1.8 GB for
    # a dense [B,32,81,n] candidate tensor, so evaluation always uses the
    # query-chunked path. Chunking changes the computation order only, never the
    # parameters -- dense/chunk equivalence is asserted in the acceptance test.
    if hasattr(refiner, 'match'):
        refiner.match.chunk = EVAL_QUERY_CHUNK
    out = {label: {} for label, _ in conditions}
    if mismatch is None:
        mismatch = mismatch_permutation(rows)
    for i, row in enumerate(rows):
        y0 = torch.from_numpy(np.ascontiguousarray(load_cache(eval_cache_dir, row)))
        y0 = y0[None].to(device)
        hr = _read_eval_tensor(row['high_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        mis = _read_eval_tensor(rows[mismatch[i]]['nano_path'], device,
                                size=y0.shape[-2:])
        for label, mode in conditions:
            if mode == 'bypass':
                sr = y0
            else:
                r = {'self': y0, 'nano': nano, 'mismatch': mis}[mode]
                sr = refiner(y0.detach(), r.detach())[0]
            out[label][row['sample_id']] = dict(
                camera=row['camera'], **_metric_triple(sr, hr))
    if verbose:
        _print_table(out, rows)
    return out


def _metric_triple(sr, hr):
    p, s, m = metrics(sr, hr)
    return dict(psnr_rgb=p, ssim_y=s, mse=m)


def _read_eval_tensor(path, device, size=None):
    from imageio import imread
    im = imread(path)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    t = torch.from_numpy(
        (im.astype(np.float32) / 127.5 - 1.).transpose(2, 0, 1))[None]
    if size is not None and tuple(t.shape[-2:]) != tuple(size):
        t = torch.nn.functional.interpolate(
            t, size=tuple(size), mode='bilinear', align_corners=False)
    return t.to(device)


def _print_table(out, rows):
    labels = list(out.keys())
    cams = sorted({r['camera'] for r in rows})
    print('  %-10s %s' % ('camera', ' '.join('%12s' % l for l in labels)))
    summary = {}
    for cam in cams + ['ALL', '0.6HW+0.4NK']:
        vals = {}
        for l in labels:
            ids = [sid for sid, d in out[l].items()
                   if cam == 'ALL' or d['camera'] == cam]
            if cam == '0.6HW+0.4NK':
                v = 0.6 * _mean(out[l], 'Huawei', 'psnr_rgb') + \
                    0.4 * _mean(out[l], 'Nikon', 'psnr_rgb')
            else:
                v = float(np.mean([out[l][i]['psnr_rgb'] for i in ids]))
            vals[l] = v
        summary[cam] = vals
        print('  %-10s %s' % (cam, ' '.join('%12.3f' % vals[l] for l in labels)))
    return summary


def _mean(per_image, camera, key):
    v = [d[key] for d in per_image.values() if d['camera'] == camera]
    return float(np.mean(v))
