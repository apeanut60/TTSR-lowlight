"""Weight init / load modes for block-amend and TTSREnhance.

Modes
-----
- ``fresh``: keep code init (soft inject / zero Estimator readout / ImageNet LTE).
- ``lte_only``: load only ``LTE.*`` from a checkpoint; refuse MainNet/RefCorrection.
- ``resume`` / ``inference``: strict full-architecture load (no shape-filtered partial).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from block_amend_ckpt import load_local_checkpoint
from model.restoration_config import reject_legacy_igab_state


SUPPORTED_CONTAINERS = ('state_dict', 'model', 'model_state_dict', 'net')


def unwrap_checkpoint(obj: Any, path: str) -> Dict[str, torch.Tensor]:
    """Extract a flat state_dict from known checkpoint containers."""
    if not isinstance(obj, dict):
        raise ValueError(
            'Unrecognized checkpoint format at %s: expected dict, got %s'
            % (path, type(obj).__name__))

    # Already a flat state_dict (tensor values).
    tensor_keys = [k for k, v in obj.items() if torch.is_tensor(v)]
    if tensor_keys and len(tensor_keys) >= max(1, len(obj) // 2):
        state = obj
    else:
        found = None
        for key in SUPPORTED_CONTAINERS:
            if key in obj and isinstance(obj[key], dict):
                found = obj[key]
                break
        if found is None:
            raise ValueError(
                'Unrecognized checkpoint container at %s. Supported keys: %s. '
                'Got top-level keys: %s'
                % (path, ', '.join(SUPPORTED_CONTAINERS),
                   ', '.join(sorted(obj.keys())[:40])))
        state = found

    # Strip DataParallel prefix when present.
    if any(str(k).startswith('module.') for k in state):
        state = {str(k).replace('module.', '', 1): v for k, v in state.items()}
    return state


def count_trainable(module: nn.Module) -> Tuple[int, int]:
    n_train, n_frozen = 0, 0
    for p in module.parameters():
        n = p.numel()
        if p.requires_grad:
            n_train += n
        else:
            n_frozen += n
    return n_train, n_frozen


def log_param_groups(model: nn.Module, logger=None) -> Dict[str, Tuple[int, int]]:
    wrap = model.module if hasattr(model, 'module') else model
    parts = {
        'MainNet': wrap.MainNet,
        'LTE': wrap.LTE,
        'SearchTransfer': wrap.SearchTransfer,
    }
    rc = getattr(wrap, 'RefCorrection', None)
    if rc is not None:
        parts['RefCorrection'] = rc
    out = {}
    for name, mod in parts.items():
        tr, fr = count_trainable(mod)
        out[name] = (tr, fr)
        msg = '[Params] %s trainable=%d frozen=%d' % (name, tr, fr)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
    return out


def _clone_params(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def assert_state_equal(before: Dict[str, torch.Tensor],
                       module: nn.Module, label: str) -> None:
    after = module.state_dict()
    if set(before) != set(after):
        raise AssertionError('%s state_dict keys changed' % label)
    for k in before:
        if not torch.equal(before[k], after[k].detach().cpu()):
            raise AssertionError(
                '%s parameter %s changed after load' % (label, k))


def load_lte_only(model: nn.Module, checkpoint_path: str, device='cpu',
                  logger=None) -> int:
    """Load only LTE.* tensors. MainNet / RefCorrection must stay untouched."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'LTE-only init checkpoint not found: %r' % checkpoint_path)

    wrap = model.module if hasattr(model, 'module') else model
    before_main = _clone_params(wrap.MainNet)
    before_ref = None
    if getattr(wrap, 'RefCorrection', None) is not None:
        before_ref = _clone_params(wrap.RefCorrection)

    # CPU load. LTE tensors are copied into the module, which may live on CUDA.
    # ``device`` is kept for callers; the pack itself is not moved there.
    raw = load_local_checkpoint(checkpoint_path, logger=logger)
    state = unwrap_checkpoint(raw, checkpoint_path)

    lte_prefix = 'LTE.'
    lte_ckpt = {k[len(lte_prefix):]: v for k, v in state.items()
                if k.startswith(lte_prefix) and not k.startswith('LTE_copy.')}
    if not lte_ckpt:
        raise RuntimeError(
            'No LTE.* tensors found in checkpoint %s' % checkpoint_path)

    model_lte = wrap.LTE.state_dict()
    missing = [k for k in model_lte if k not in lte_ckpt]
    unexpected = [k for k in lte_ckpt if k not in model_lte]
    shape_bad = []
    for k, v in lte_ckpt.items():
        if k in model_lte and tuple(model_lte[k].shape) != tuple(v.shape):
            shape_bad.append((k, tuple(v.shape), tuple(model_lte[k].shape)))

    if missing or unexpected or shape_bad:
        raise RuntimeError(
            'LTE-only load failed for %s: missing=%d unexpected=%d '
            'shape_mismatch=%d. missing[:8]=%s shape[:4]=%s'
            % (checkpoint_path, len(missing), len(unexpected), len(shape_bad),
               missing[:8], shape_bad[:4]))

    wrap.LTE.load_state_dict(lte_ckpt, strict=True)
    # Keep LTE_copy in sync for perceptual path.
    if hasattr(wrap, 'LTE_copy'):
        wrap.LTE_copy.load_state_dict(wrap.LTE.state_dict(), strict=False)

    assert_state_equal(before_main, wrap.MainNet, 'MainNet')
    if before_ref is not None:
        assert_state_equal(before_ref, wrap.RefCorrection, 'RefCorrection')

    n = len(lte_ckpt)
    msg = ('[Init] mode=lte_only path=%s loaded_LTE_tensors=%d '
           'MainNet_from_ckpt=False RefCorrection_from_ckpt=False'
           % (checkpoint_path, n))
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    return n


def load_strict_state(model: nn.Module, checkpoint_path: str, device='cpu',
                      logger=None, mode='inference', preloaded=None) -> int:
    """Strict load of model weights (architecture must match).

    The pack is read on CPU (or ``preloaded`` is reused) and copied into the
    module's existing parameters, which already sit on ``device``. The pack
    is not moved onto CUDA as a whole. Pass ``preloaded`` to avoid a second
    read of a pack that was already validated.
    """
    if preloaded is not None:
        raw = preloaded
    else:
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError('Checkpoint not found: %r' % checkpoint_path)
        raw = load_local_checkpoint(checkpoint_path, logger=logger)
    # Prefer nested model state when present (resume packs).
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state = unwrap_checkpoint(raw, checkpoint_path)
        # unwrap already handled model_state_dict
    elif isinstance(raw, dict) and any(
            torch.is_tensor(v) for v in raw.values()):
        # May be a resume pack with mixed keys — prefer 'model'/'state_dict'
        if any(k in raw for k in SUPPORTED_CONTAINERS):
            state = unwrap_checkpoint(raw, checkpoint_path)
        elif all(torch.is_tensor(v) or not str(k).endswith('_state_dict')
                 for k, v in list(raw.items())[:5]):
            # Heuristic: if looks like pure state_dict
            tensor_ratio = sum(1 for v in raw.values() if torch.is_tensor(v))
            if tensor_ratio == len(raw) or (
                    'LTE.slice1.0.weight' in raw or 'MainNet.SFE.conv_head.weight' in raw
                    or any(str(k).startswith('LTE.') for k in raw)):
                state = unwrap_checkpoint(raw, checkpoint_path)
            else:
                state = unwrap_checkpoint(raw, checkpoint_path)
        else:
            state = unwrap_checkpoint(raw, checkpoint_path)
    else:
        state = unwrap_checkpoint(raw, checkpoint_path)

    # An IGAB-variant state can never be (even partially) loaded into LFCA.
    _wrap = model.module if hasattr(model, 'module') else model
    if type(getattr(_wrap, 'MainNet', None)).__name__ == 'CSFIUNet':
        reject_legacy_igab_state(state.keys(), checkpoint_path)

    # Drop LTE_copy from required keys if absent in ckpt.
    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if '_copy' in k and k not in model_state:
            continue
        filtered[k] = v

    missing, unexpected = [], []
    try:
        incompatible = model.load_state_dict(filtered, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
    except Exception as e:
        raise RuntimeError('Strict load failed for %s: %s' % (checkpoint_path, e))

    # Ignore missing LTE_copy.*
    missing = [k for k in missing if '_copy' not in k]
    unexpected = [k for k in unexpected if '_copy' not in k]
    if missing or unexpected:
        raise RuntimeError(
            '%s load rejected for %s: missing=%d unexpected=%d; '
            'missing[:8]=%s unexpected[:8]=%s'
            % (mode, checkpoint_path, len(missing), len(unexpected),
               missing[:8], unexpected[:8]))

    n = len([k for k in filtered if k in model_state])
    msg = ('[Init] mode=%s path=%s loaded_tensors=%d strict=True'
           % (mode, checkpoint_path, n))
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    return n


def apply_init_mode(model: nn.Module, args, device='cpu', logger=None) -> str:
    """Apply init according to ``args.init_scope``.

    ``fresh`` (default): no checkpoint load.
    ``lte_only``: requires ``args.init_checkpoint``.
    ``resume`` / ``inference``: handled by callers with full packs; here only
    supports loading a bare weight file when ``init_checkpoint`` is set.
    """
    scope = str(getattr(args, 'init_scope', 'fresh') or 'fresh').strip().lower()
    path = getattr(args, 'init_checkpoint', None) or ''

    if scope in ('fresh', 'none', ''):
        msg = ('[Init] mode=fresh source=code_init+ImageNet_LTE '
               'MainNet_from_ckpt=False')
        if logger:
            logger.info(msg)
        else:
            print(msg)
        log_param_groups(model, logger)
        return 'fresh'

    if scope == 'lte_only':
        if not path:
            raise ValueError('--init_scope lte_only requires --init_checkpoint')
        load_lte_only(model, path, device=device, logger=logger)
        log_param_groups(model, logger)
        return 'lte_only'

    if scope in ('inference', 'resume_weights'):
        if not path:
            raise ValueError('--init_scope %s requires --init_checkpoint' % scope)
        load_strict_state(model, path, device=device, logger=logger, mode=scope)
        log_param_groups(model, logger)
        return scope

    raise ValueError(
        'Unknown init_scope %r (expected fresh|lte_only|inference)' % scope)
