"""Checkpoint schema, atomic save, and resume helpers for block-amend.

Schema 4 (this file): same model (``csfi_lfca_v1``) as schema 3, but the pack
is tensors plus plain Python values. RNG state is an explicit encoding, not a
pickled NumPy object, so a ``weights_only=True`` load can read it.

The whole pack is loaded on CPU. Model weights are copied into the already
built module; optimizer tensors are placed by ``Optimizer.load_state_dict``
(momentum follows the parameter, ``step`` is left where Adam stored it).
Nothing here moves the pack onto CUDA.
"""
from __future__ import annotations

import inspect
import os
import tempfile
from typing import Any, Dict, Optional, Set

import torch

from model.restoration_config import (
    MODEL_VARIANT, RESTORATION_CONFIG_KEYS, require_model_variant,
)

# v4: safe RNG encoding (no NumPy objects in the pack). Model is still
# csfi_lfca_v1; this is a checkpoint-format change, not an architecture change.
# v3: same model, NumPy RNG objects. v2: IGAB variant.
CHECKPOINT_SCHEMA_VERSION = 4
RNG_FORMAT = 'block_amend_rng_v1'
_SCHEMA3_HINT = (
    ' Schema 3 stores a pickled NumPy RNG state, which a weights_only load '
    'cannot read. It is not migrated automatically and the original file is '
    'never overwritten. For a trusted local schema-3 file run '
    'migrate_schema3_checkpoint(src, dst); that writes a new file. '
    'Do not edit the saved split hash to force a match.'
)

_LOAD_SUPPORTS_WEIGHTS_ONLY = (
    'weights_only' in inspect.signature(torch.load).parameters)

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


def _log(logger, msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def load_local_checkpoint(path: str, trust_legacy: bool = False, logger=None):
    """Load one user-specified local checkpoint onto CPU.

    When this PyTorch accepts ``weights_only``, the default is
    ``weights_only=True``. A failed safe load is never retried with
    ``weights_only=False``. ``trust_legacy=True`` is the explicit opt-in for
    a trusted local file (schema-3 migration) and is logged as such.

    This environment's PyTorch 1.12 ``torch.load`` has no ``weights_only``
    parameter. That branch only loads the path the caller named, and the log
    says the safe unpickler is not available. It is not an error fallback.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError('Checkpoint not found: %r' % path)
    if _LOAD_SUPPORTS_WEIGHTS_ONLY:
        if trust_legacy:
            _log(logger,
                 '[Checkpoint] explicit trust_legacy load (weights_only=False) '
                 'of local file %s. This is not an automatic retry.' % path)
            return torch.load(path, map_location='cpu', weights_only=False)
        return torch.load(path, map_location='cpu', weights_only=True)
    _log(logger,
         '[Checkpoint] PyTorch %s torch.load has no weights_only argument; '
         'loading the caller-specified local file %s without the safe '
         'unpickler. This is the installed runtime, not a fallback after a '
         'failed safe load.' % (torch.__version__, path))
    if trust_legacy:
        _log(logger,
             '[Checkpoint] trust_legacy requested, but this PyTorch has no '
             'weights_only switch; load of %s is the ordinary local load.'
             % path)
    return torch.load(path, map_location='cpu')


def _cpu_byte(value, name: str):
    """RNG byte state: torch.uint8, 1-D, contiguous, CPU.

    A CUDA-mapped tensor is moved with ``.cpu()``. A wrong dtype is an error;
    it is not cast into looking like a valid state.
    """
    if not torch.is_tensor(value):
        raise RuntimeError(
            '%s must be a torch.uint8 tensor, got %s' % (name, type(value).__name__))
    if value.dtype != torch.uint8:
        raise RuntimeError(
            '%s dtype is %s; expected torch.uint8. Refusing to cast a bad '
            'RNG state into a valid one.' % (name, value.dtype))
    if value.dim() != 1 or value.numel() == 0:
        raise RuntimeError(
            '%s must be a non-empty 1-D tensor, got shape %s'
            % (name, tuple(value.shape)))
    out = value.detach()
    if out.device.type != 'cpu':
        out = out.cpu()
    if not out.is_contiguous():
        out = out.contiguous()
    return out


def _encode_python_random(state) -> Dict:
    try:
        version, mt, gauss = state
        mt_list = [int(x) for x in mt]
    except (TypeError, ValueError) as e:
        raise RuntimeError('python RNG state is not a (version, tuple, gauss) triple: %s' % e)
    if int(version) != 3 or len(mt_list) != 625:
        raise RuntimeError(
            'unsupported python RNG state: version=%r len=%d (expected 3, 625)'
            % (version, len(mt_list)))
    if any(x < 0 for x in mt_list) or not 0 <= mt_list[-1] <= 624:
        raise RuntimeError('python RNG state has an out-of-range MT index or word')
    if gauss is not None and not isinstance(gauss, (int, float)):
        raise RuntimeError('python RNG gauss cache has type %s' % type(gauss).__name__)
    return {
        'format': 'python_random_v1',
        'version': 3,
        'state': mt_list,
        'gauss': None if gauss is None else float(gauss),
    }


def _decode_python_random(encoded) -> tuple:
    if not isinstance(encoded, dict) or encoded.get('format') != 'python_random_v1':
        raise RuntimeError('python RNG encoding is not python_random_v1')
    if encoded.get('version') != 3:
        raise RuntimeError('python RNG version=%r (expected 3)' % encoded.get('version'))
    state = encoded.get('state')
    if (not isinstance(state, list) or len(state) != 625
            or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                       for x in state)
            or not 0 <= state[-1] <= 624):
        raise RuntimeError('python RNG state must be 625 non-negative ints, index <= 624')
    gauss = encoded.get('gauss')
    if gauss is not None and not isinstance(gauss, (int, float)):
        raise RuntimeError('python RNG gauss cache is not a float or null')
    return (3, tuple(state), None if gauss is None else float(gauss))


def _encode_numpy(state) -> Dict:
    import numpy as np
    try:
        name, keys, pos, has_gauss, cached = state
    except (TypeError, ValueError) as e:
        raise RuntimeError('numpy RNG state is not the MT19937 5-tuple: %s' % e)
    if name != 'MT19937':
        raise RuntimeError('unsupported numpy bit generator %r (expected MT19937)' % name)
    keys = np.asarray(keys)
    if keys.shape != (624,) or keys.dtype != np.uint32:
        raise RuntimeError(
            'numpy RNG keys must be uint32[624], got %s %s' % (keys.dtype, keys.shape))
    pos = int(pos)
    has_gauss = int(has_gauss)
    if not 0 <= pos <= 624 or has_gauss not in (0, 1):
        raise RuntimeError('numpy RNG position/has_gauss out of range')
    return {
        'format': 'numpy_mt19937_v1',
        'bit_generator_name': 'MT19937',
        'keys': [int(x) for x in keys.tolist()],
        'position': pos,
        'has_gauss': has_gauss,
        'cached_gaussian': float(cached),
    }


def _decode_numpy(encoded):
    import numpy as np
    if not isinstance(encoded, dict) or encoded.get('format') != 'numpy_mt19937_v1':
        raise RuntimeError('numpy RNG encoding is not numpy_mt19937_v1')
    if encoded.get('bit_generator_name') != 'MT19937':
        raise RuntimeError(
            'numpy bit_generator_name=%r' % encoded.get('bit_generator_name'))
    keys = encoded.get('keys')
    if (not isinstance(keys, list) or len(keys) != 624
            or not all(isinstance(x, int) and not isinstance(x, bool)
                       and 0 <= x <= 0xFFFFFFFF for x in keys)):
        raise RuntimeError('numpy RNG keys must be 624 uint32 ints')
    pos = encoded.get('position')
    has_gauss = encoded.get('has_gauss')
    cached = encoded.get('cached_gaussian')
    if not isinstance(pos, int) or not 0 <= pos <= 624:
        raise RuntimeError('numpy RNG position out of range: %r' % pos)
    if has_gauss not in (0, 1):
        raise RuntimeError('numpy RNG has_gauss must be 0 or 1, got %r' % has_gauss)
    if isinstance(cached, bool) or not isinstance(cached, (int, float)):
        raise RuntimeError('numpy RNG cached_gaussian is not a float')
    return ('MT19937', np.array(keys, dtype=np.uint32), pos, has_gauss, float(cached))


def _cuda_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def _encode_cuda(states) -> Optional[list]:
    n = _cuda_device_count()
    if n == 0:
        if states not in (None, [], ()):
            raise RuntimeError(
                'CUDA RNG state was captured with no visible CUDA device')
        return None
    if states is None or len(list(states)) != n:
        raise RuntimeError(
            'CUDA RNG capture expected %d device state(s), got %s'
            % (n, 0 if states is None else len(list(states))))
    return [_cpu_byte(t, 'torch_cuda[%d]' % i) for i, t in enumerate(states)]


def _restore_cuda(saved) -> None:
    n = _cuda_device_count()
    if saved is None:
        if n != 0:
            raise RuntimeError(
                'Checkpoint has no CUDA RNG state but this process sees %d '
                'CUDA device(s). Refusing to reseed or drop CUDA RNG and '
                'still call the resume exact.' % n)
        return
    if not isinstance(saved, (list, tuple)):
        raise RuntimeError(
            'torch_cuda RNG state must be a list of CPU uint8 tensors or null, '
            'got %s' % type(saved).__name__)
    if len(saved) != n:
        raise RuntimeError(
            'CUDA RNG device-count mismatch: checkpoint has %d state(s), this '
            'process has %d CUDA device(s). Refusing to drop part of the '
            'saved CUDA RNG and claim an exact resume.' % (len(saved), n))
    states = [_cpu_byte(t, 'torch_cuda[%d]' % i) for i, t in enumerate(saved)]
    if n:
        torch.cuda.set_rng_state_all(states)


def capture_rng_state(train_generator=None, val_generator=None) -> Dict:
    """Schema-4 RNG pack: plain values and CPU uint8 tensors only."""
    import random
    import numpy as np
    if train_generator is None or val_generator is None:
        raise RuntimeError(
            'Refusing to capture a resumable RNG state without both the train '
            'and val DataLoader generators (would otherwise reseed them).')
    cuda_live = torch.cuda.get_rng_state_all() if _cuda_device_count() else None
    return {
        'format': RNG_FORMAT,
        'python_random': _encode_python_random(random.getstate()),
        'numpy': _encode_numpy(np.random.get_state()),
        'torch_cpu': _cpu_byte(torch.get_rng_state(), 'torch_cpu'),
        'torch_cuda': _encode_cuda(cuda_live),
        'train_generator': _cpu_byte(train_generator.get_state(), 'train_generator'),
        'val_generator': _cpu_byte(val_generator.get_state(), 'val_generator'),
    }


def restore_rng_state(state: Dict, train_generator=None, val_generator=None) -> None:
    """Restore RNG for an epoch-boundary resume.

    CPU and DataLoader generator states are checked (uint8, 1-D) and moved to
    CPU if a load mapped them onto CUDA. A wrong dtype is rejected. Missing
    generator state is an error: resume does not reseed.

    Exact continuation is only claimed for the same machine class and an
    epoch boundary. This does not claim CPU/GPU or cross-version bitwise
    equality, and it does not restore a mid-epoch iterator.
    """
    import random
    import numpy as np
    if not isinstance(state, dict) or not state:
        raise RuntimeError('Missing RNG state in checkpoint')
    if state.get('format') != RNG_FORMAT:
        raise RuntimeError(
            'RNG state format is %r, expected %r. Old schema-3 NumPy RNG '
            'objects are not restored and are not reseeded in their place.%s'
            % (state.get('format'), RNG_FORMAT, _SCHEMA3_HINT))
    required = (
        'python_random', 'numpy', 'torch_cpu', 'torch_cuda',
        'train_generator', 'val_generator',
    )
    # torch_cuda may be null (captured with no CUDA). Every other field is required.
    missing = [k for k in required if k not in state]
    if missing:
        raise RuntimeError(
            'Incomplete RNG state, missing %s. Refusing to reseed.' % missing)
    if state['train_generator'] is None or state['val_generator'] is None:
        raise RuntimeError(
            'Checkpoint has no DataLoader generator state. Refusing to reseed '
            'and continue.')
    if train_generator is None or val_generator is None:
        raise RuntimeError(
            'restore_rng_state requires the live train and val generators. '
            'Refusing to restore only part of the RNG.')
    random.setstate(_decode_python_random(state['python_random']))
    np.random.set_state(_decode_numpy(state['numpy']))
    torch.set_rng_state(_cpu_byte(state['torch_cpu'], 'torch_cpu'))
    _restore_cuda(state['torch_cuda'])
    train_generator.set_state(_cpu_byte(state['train_generator'], 'train_generator'))
    val_generator.set_state(_cpu_byte(state['val_generator'], 'val_generator'))


def convert_legacy_rng(state: Dict) -> Dict:
    """Encode a schema-3 RNG dict (live tuples + tensors) as schema 4.

    Does not invent missing generator state.
    """
    if not isinstance(state, dict):
        raise RuntimeError('legacy RNG state is not a dict')
    if state.get('format') == RNG_FORMAT:
        raise RuntimeError('RNG state is already %s' % RNG_FORMAT)
    required = (
        'python_random', 'numpy', 'torch_cpu', 'train_generator', 'val_generator',
    )
    missing = [k for k in required if state.get(k) is None]
    if missing:
        raise RuntimeError(
            'schema-3 RNG is missing %s; refusing to invent it' % missing)
    cuda = state.get('torch_cuda', None) if 'torch_cuda' in state else None
    if 'torch_cuda' not in state:
        raise RuntimeError('schema-3 RNG has no torch_cuda field; refusing to invent it')
    return {
        'format': RNG_FORMAT,
        'python_random': _encode_python_random(state['python_random']),
        'numpy': _encode_numpy(state['numpy']),
        'torch_cpu': _cpu_byte(state['torch_cpu'], 'torch_cpu'),
        'torch_cuda': _encode_cuda(cuda),
        'train_generator': _cpu_byte(state['train_generator'], 'train_generator'),
        'val_generator': _cpu_byte(state['val_generator'], 'val_generator'),
    }


def migrate_schema3_checkpoint(src: str, dst: str, logger=None) -> str:
    """Write a schema-4 copy of a trusted local schema-3 pack.

    The source file is only read. ``dst`` must not already exist. Split hashes
    are copied, not recomputed: a hash-version change still fails the identity
    check, and this function will not edit them to make resume pass.
    """
    if os.path.abspath(src) == os.path.abspath(dst):
        raise RuntimeError('refusing to overwrite the source checkpoint %s' % src)
    if os.path.exists(dst):
        raise FileExistsError('refusing to overwrite %s' % dst)
    raw = load_local_checkpoint(src, trust_legacy=True, logger=logger)
    if not isinstance(raw, dict):
        raise RuntimeError('schema-3 migration expected a checkpoint dict')
    ver = raw.get('checkpoint_schema_version')
    if ver != 3:
        raise RuntimeError(
            'migrate_schema3_checkpoint only accepts schema 3, got %r' % ver)
    require_model_variant(raw, src)
    required = (
        'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
        'epoch', 'global_step', 'best_val_psnr', 'best_epoch',
        'config', 'split_identity', 'rng', 'metric_protocol', 'model_variant',
    )
    missing = [k for k in required if k not in raw]
    if missing:
        raise RuntimeError(
            'schema-3 pack is incomplete (%s); refusing to invent fields' % missing)
    out = dict(raw)
    out['rng'] = convert_legacy_rng(raw['rng'])
    out['checkpoint_schema_version'] = CHECKPOINT_SCHEMA_VERSION
    out['migrated_from_schema'] = 3
    atomic_torch_save(out, dst)
    _log(logger,
         '[Checkpoint] migrated schema 3 -> %d into %s (source left untouched). '
         'Split identity was copied, not recomputed.'
         % (CHECKPOINT_SCHEMA_VERSION, dst))
    return dst


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
    if 'checkpoint_schema_version' not in raw:
        raise RuntimeError(
            'Incomplete resume checkpoint. Missing: checkpoint_schema_version. '
            'Cannot guarantee equivalent continuation; refusing warm-start '
            'fallback.')
    ver = int(raw['checkpoint_schema_version'])
    if ver != CHECKPOINT_SCHEMA_VERSION:
        hint = _SCHEMA3_HINT if ver == 3 else ''
        raise RuntimeError(
            'Unsupported checkpoint_schema_version=%s (expected %s).%s'
            % (ver, CHECKPOINT_SCHEMA_VERSION, hint))
    rng = raw.get('rng')
    if not isinstance(rng, dict) or rng.get('format') != RNG_FORMAT:
        raise RuntimeError(
            'Checkpoint RNG is not %s. Refusing to reseed or to treat a '
            'schema-3 NumPy RNG object as a current state.%s'
            % (RNG_FORMAT, _SCHEMA3_HINT))


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
