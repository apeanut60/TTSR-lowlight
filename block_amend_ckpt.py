"""Checkpoint schema, atomic save, and resume helpers for block-amend."""
from __future__ import annotations

import io
import os
import tempfile
from typing import Any, Dict, Optional, Set

import torch

from model.restoration_config import (
    MODEL_VARIANT, RESTORATION_CONFIG_KEYS, require_model_variant,
)

# v3: packs carry a top-level ``model_variant`` (csfi_lfca_v1) and the
# restoration-block identity in ``config``. v2 packs are the IGAB variant.
CHECKPOINT_SCHEMA_VERSION = 3

# Keys that must match between CLI resume and checkpoint (protocol lock).
LOCKED_CONFIG_KEYS = (
    'backbone', 'num_res_blocks', 'n_feats', 'res_scale',
) + RESTORATION_CONFIG_KEYS + (
    'enable_lightup',
    'use_global_illum', 'use_ref_illum', 'ref_correction',
    'freeze_lte', 'freeze_stages',
    'query_illum_align', 'texture_fusion',
    'stage_mode_lv3', 'stage_mode_lv2', 'stage_mode_lv1',
    'eval_mean_align', 'eval_chroma_gain', 'eval_tta',
    'ref_degrade', 'train_crop_size', 'batch_size',
    'lr_rate', 'lr_rate_lte', 'decay', 'gamma',
    'rec_loss_type', 'rec_w', 'ms_ssim_w',
    'per_w', 'tpl_w', 'adv_w', 'illum_smooth_w', 'color_w',
    'exposure_w', 'illum_match_w', 'ref_correct_w',
    'seed', 'split_seed', 'val_ratio',
    'train_ref_subdir', 'dataset',
    'tile_size', 'tile_overlap',
)

# Explicit ops overrides allowed on resume (must appear on CLI).
RESUME_ALLOWED_OVERRIDES = frozenset({
    'cpu', 'num_gpu', 'num_workers', 'num_epochs',
    'save_dir', 'run_name', 'log_file_name', 'print_every', 'val_every',
    'mode', 'resume_checkpoint', 'split_dir', 'dataset_dir',
})


def atomic_torch_save(obj: Any, path: str) -> None:
    """Serialize ``obj`` to ``path`` via temp file + os.replace.

    Temp name uses prefix ``tmp_ckpt_`` and suffix ``.pt`` (no leading-dot /
    extensionless paths). On failure the existing target is left untouched.
    """
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix='tmp_ckpt_', suffix='.pt', dir=directory)
        with os.fdopen(fd, 'wb') as f:
            fd = None  # ownership transferred
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def capture_rng_state(train_generator=None, val_generator=None) -> Dict:
    import random
    import numpy as np
    state = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
        'train_generator': (
            train_generator.get_state() if train_generator is not None else None),
        'val_generator': (
            val_generator.get_state() if val_generator is not None else None),
    }
    return state


def restore_rng_state(state: Dict, train_generator=None, val_generator=None) -> None:
    import random
    import numpy as np
    if not state:
        raise RuntimeError('Missing RNG state in checkpoint')
    required = ('python_random', 'numpy', 'torch_cpu')
    missing = [k for k in required if k not in state]
    if missing:
        raise RuntimeError('Incomplete RNG state, missing: %s' % missing)
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state.get('torch_cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    if train_generator is not None and state.get('train_generator') is not None:
        train_generator.set_state(state['train_generator'])
    if val_generator is not None and state.get('val_generator') is not None:
        val_generator.set_state(state['val_generator'])


def resolved_config_from_args(args) -> Dict:
    """Full effective config for checkpoint (not only BASELINE_FIXED)."""
    keys = set(LOCKED_CONFIG_KEYS) | {
        'dataset_dir', 'train_ref_subdir', 'test_ref_subdir',
        'num_epochs', 'val_every', 'print_every', 'num_workers',
        'init_scope', 'init_checkpoint', 'enhance_mode',
        'ref_illum_pool', 'ref_correction_feats',
        'query_illum_alpha', 'query_hist_beta',
        'query_gamma_min', 'query_gamma_max', 'query_hist_bins',
        'texture_fusion_lambda', 'texture_fusion_s_source',
        'gate_mode', 'gate_shift', 'gate_smooth_kernel',
        'tile_window', 'beta1', 'beta2', 'eps',
        'metric_protocol',
    }
    out = {}
    for k in sorted(keys):
        if hasattr(args, k):
            v = getattr(args, k)
            if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                out[k] = v
            else:
                out[k] = str(v)
    return out


def build_checkpoint_pack(
    *,
    model,
    opt,
    sched,
    epoch: int,
    global_step: int,
    best_val_psnr: float,
    best_epoch: int,
    validation_completed: bool,
    args,
    split_identity: Dict,
    init_scope: str,
    init_checkpoint: str,
    run_id: str,
    rng_state: Dict,
    metric_protocol: str,
) -> Dict:
    model_sd = {
        k.replace('module.', ''): v.detach().cpu()
        for k, v in model.state_dict().items() if '_copy' not in k
    }
    config = resolved_config_from_args(args)
    missing_id = [k for k in RESTORATION_CONFIG_KEYS if k not in config]
    if missing_id:
        raise RuntimeError(
            'Refusing to write a checkpoint without model identity; args lack %s'
            % missing_id)
    if config['model_variant'] != MODEL_VARIANT:
        raise RuntimeError(
            'args.model_variant=%r but this code builds %r'
            % (config['model_variant'], MODEL_VARIANT))
    return {
        'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
        'model_variant': config['model_variant'],
        'model_state_dict': model_sd,
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'epoch': int(epoch),
        'global_step': int(global_step),
        'best_val_psnr': float(best_val_psnr),
        'best_epoch': int(best_epoch),
        'validation_completed': bool(validation_completed),
        'config': config,
        'metric_protocol': metric_protocol,
        'init_scope': init_scope,
        'init_checkpoint': init_checkpoint or '',
        'run_id': run_id,
        'split_identity': dict(split_identity),
        'rng': rng_state,
    }


def require_resume_fields(raw: Dict, path: str = '') -> None:
    # Model identity first: an IGAB-variant pack must be told *why* it is
    # refused, not just that its schema is old.
    require_model_variant(raw, path)
    required = [
        'checkpoint_schema_version', 'model_variant', 'model_state_dict',
        'optimizer_state_dict', 'scheduler_state_dict',
        'epoch', 'global_step', 'best_val_psnr', 'best_epoch',
        'config', 'split_identity', 'rng', 'metric_protocol',
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise RuntimeError(
            'Incomplete resume checkpoint (schema=%s). Missing: %s. '
            'Cannot guarantee equivalent continuation; refusing warm-start '
            'fallback.'
            % (raw.get('checkpoint_schema_version'), missing))
    ver = int(raw['checkpoint_schema_version'])
    if ver != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            'Unsupported checkpoint_schema_version=%s (expected %s)'
            % (ver, CHECKPOINT_SCHEMA_VERSION))


def explicit_cli_dests(argv) -> Set[str]:
    """Dest names explicitly present on the command line."""
    out = set()
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('--'):
            name = a[2:].split('=', 1)[0].replace('-', '_')
            out.add(name)
        i += 1
    return out


def merge_resume_config(
    ckpt_config: Dict,
    cli_ns,
    explicit: Set[str],
    baseline_fixed: Dict,
) -> Any:
    """Build args from checkpoint config; apply only allowed explicit overrides."""
    from types import SimpleNamespace
    merged = dict(baseline_fixed)
    merged.update(ckpt_config)
    # Always take path/device ops from CLI when present
    applied = {}
    for key in explicit:
        if key in ('mode', 'resume_checkpoint', 'init_scope', 'init_checkpoint',
                   'model_path', 'eval_split', 'save_images',
                   'allow_partial_coverage'):
            if hasattr(cli_ns, key):
                merged[key] = getattr(cli_ns, key)
                applied[key] = getattr(cli_ns, key)
            continue
        if key not in RESUME_ALLOWED_OVERRIDES:
            # Locked keys: reject if differs from ckpt
            if key in LOCKED_CONFIG_KEYS and hasattr(cli_ns, key):
                cli_v = getattr(cli_ns, key)
                ckpt_v = ckpt_config.get(key, merged.get(key))
                if cli_v != ckpt_v:
                    raise RuntimeError(
                        'Resume refuses override of locked config %s: '
                        'cli=%r ckpt=%r' % (key, cli_v, ckpt_v))
            continue
        if hasattr(cli_ns, key):
            merged[key] = getattr(cli_ns, key)
            applied[key] = getattr(cli_ns, key)
    # Runtime-only options that are neither in the checkpoint config nor in the
    # fixed baseline (log_file_name, cpu, num_gpu, split_dir, ...) take their
    # CLI value/default. Fill-only: never replaces a checkpoint/baseline value.
    for key, value in vars(cli_ns).items():
        if key not in merged:
            merged[key] = value
    # Carry mode from CLI
    if hasattr(cli_ns, 'mode'):
        merged['mode'] = cli_ns.mode
    return SimpleNamespace(**merged), applied


def check_grad_finite(model, epoch, step, sample_id, loss, sr) -> None:
    """Raise before optimizer.step if any participating grad is non-finite."""
    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(name)
    if bad:
        raise RuntimeError(
            'Non-finite gradients before step: epoch=%s step=%s id=%s '
            'loss=%s sr_min=%s sr_max=%s params=%s'
            % (epoch, step, sample_id,
               float(loss.detach()) if loss.numel() == 1 else 'tensor',
               float(sr.detach().min()) if sr is not None else None,
               float(sr.detach().max()) if sr is not None else None,
               bad[:12]))
