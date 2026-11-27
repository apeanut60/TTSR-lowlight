"""Model identity + restoration-block config for the block-amend CSFI-U-Net.

Buildable variants, selected by ``model_variant`` (no plugin registry):

- ``csfi_lfca_v1``: local affine gain head + hard top-1 transfer.
- ``csfi_lfca_mgain_topk_v2``: multi-scale spatial gain + top-k soft transfer.

The IGAB variant (``csfi_igab_v0``, commit 5777e63 and earlier) is refused.
Top-k and temperature add no parameters, so a pack's behavior is the saved
config, not the state_dict shapes.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional, Set, Tuple

MODEL_VARIANT = 'csfi_lfca_v1'
MODEL_VARIANT_V2 = 'csfi_lfca_mgain_topk_v2'
BUILDABLE_VARIANTS = (MODEL_VARIANT, MODEL_VARIANT_V2)
LEGACY_IGAB_VARIANT = 'csfi_igab_v0'
LEGACY_IGAB_COMMIT = '5777e63'

# Shared restoration trunk. Both variants lock these.
_SHARED_RESTORATION = dict(
    restoration_block='lfca',
    restoration_blocks='1,1',      # depth at coarse(H/4),middle(H/2)
    restoration_heads=1,
    restoration_ffn_expansion=4,
    lowpass_kernel='binomial',
    lowpass_kernel_size=5,
)

# Gain-head architecture. v1 records it and does not build it.
_GAIN_ARCH = dict(
    gain_channels='32,64',
    gain_res_blocks='2,2,1',
    gain_attn_heads=4,
    gain_attn_depth=1,
    gain_attn_ffn_expansion=2,
    gain_max_tokens=400,
    gain_downsample='binomial5_stride2',
    gain_readout_scale=4,
)

# v1 behavior. match_* below hard_top1 are unused by that path.
_V1_BEHAVIOR = dict(
    gain_estimator='local_affine',
    matching_mode='hard_top1',
    match_topk=1,
    match_temperature=0.1,
    match_query_chunk_size=128,
    match_transfer_recompute=True,
    # Unused by the local affine head. Recorded so the key exists.
    gain_grid_policy='legacy_aspect_first',
    **_GAIN_ARCH,
)

# v2 defaults. Matching k / temperature / chunk may differ; the gain
# architecture stays the fixed starting point. Fresh runs use the scaled
# token grid. A pack that omits gain_grid_policy is legacy; see
# apply_absent_grid_policy.
_V2_BEHAVIOR = dict(
    gain_estimator='multiscale_spatial_v1',
    matching_mode='topk_soft',
    match_topk=4,
    match_temperature=0.1,
    match_query_chunk_size=128,
    match_transfer_recompute=True,
    gain_grid_policy='scaled_budget_v1',
    **_GAIN_ARCH,
)
GRID_POLICIES = ('legacy_aspect_first', 'scaled_budget_v1')

RESTORATION_DEFAULTS = dict(
    model_variant=MODEL_VARIANT,
    **_SHARED_RESTORATION,
    **_V1_BEHAVIOR,
)
V2_DEFAULTS = dict(
    model_variant=MODEL_VARIANT_V2,
    **_SHARED_RESTORATION,
    **_V2_BEHAVIOR,
)
RESTORATION_CONFIG_KEYS = tuple(RESTORATION_DEFAULTS)
_LOCKED_GAIN_ARCH_KEYS = tuple(_GAIN_ARCH)
_MATCH_RANGE_KEYS = (
    'match_topk', 'match_temperature', 'match_query_chunk_size',
    'match_transfer_recompute',
)

# Deprecated CLI/config aliases -> neutral names.
DEPRECATED_ALIASES = {
    'igab_blocks': 'restoration_blocks',
    'igab_heads': 'restoration_heads',
}

LEGACY_IGAB_STATE_MARKERS = ('coarse_igab.', 'middle_igab.')


def parse_restoration_blocks(spec) -> Tuple[int, int]:
    """'1,1' | '2' | (1, 1) -> (coarse_depth, middle_depth)."""
    if isinstance(spec, (list, tuple)):
        parts = [int(x) for x in spec]
    elif isinstance(spec, int):
        parts = [spec]
    else:
        parts = [int(x) for x in str(spec).replace(' ', '').split(',') if x != '']
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    if len(parts) != 2 or min(parts) < 1:
        raise ValueError(
            'restoration_blocks expects "coarse,middle" depths >= 1, got %r'
            % (spec,))
    return parts[0], parts[1]


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 0:
            return False
        if value == 1:
            return True
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes'):
        return True
    if text in ('0', 'false', 'no'):
        return False
    raise ValueError('expected a boolean, got %r' % (value,))


def _canon_int_list(value):
    if isinstance(value, (list, tuple)):
        parts = [int(x) for x in value]
    else:
        parts = [int(x) for x in str(value).replace(' ', '').split(',') if x != '']
    if not parts:
        raise ValueError('empty integer list %r' % (value,))
    return ','.join(str(x) for x in parts)


def _canon(key, value):
    if key == 'restoration_blocks':
        return '%d,%d' % parse_restoration_blocks(value)
    if key in ('restoration_heads', 'restoration_ffn_expansion',
               'lowpass_kernel_size', 'gain_attn_heads', 'gain_attn_depth',
               'gain_attn_ffn_expansion', 'gain_max_tokens',
               'gain_readout_scale', 'match_topk', 'match_query_chunk_size'):
        return int(value)
    if key == 'match_temperature':
        return float(value)
    if key == 'match_transfer_recompute':
        return _as_bool(value)
    if key in ('gain_channels', 'gain_res_blocks'):
        return _canon_int_list(value)
    if key == 'gain_downsample':
        text = str(value).strip().lower()
        if text == 'fixed_binomial5_stride2':
            return 'binomial5_stride2'
        return text
    return str(value).strip().lower()


def _getter(args):
    if isinstance(args, dict):
        return lambda k: args.get(k)
    return lambda k: getattr(args, k, None)


def _not_buildable(variant):
    hint = ''
    if variant == LEGACY_IGAB_VARIANT:
        hint = (' The IGAB variant is only buildable from commit %s.'
                % LEGACY_IGAB_COMMIT)
    raise ValueError(
        'model_variant=%r is not buildable by this code (builds %s).%s'
        % (variant, ' or '.join(BUILDABLE_VARIANTS), hint))


def validate_restoration_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    missing = [k for k in RESTORATION_CONFIG_KEYS if k not in cfg or cfg[k] is None]
    if missing:
        raise ValueError('restoration config missing %s' % missing)
    out = {k: _canon(k, cfg[k]) for k in RESTORATION_CONFIG_KEYS}
    variant = out['model_variant']
    if variant not in BUILDABLE_VARIANTS:
        _not_buildable(variant)
    # Depth and head count stay configurable. The block type does not.
    for key in ('restoration_block', 'restoration_ffn_expansion',
                'lowpass_kernel', 'lowpass_kernel_size'):
        if out[key] != RESTORATION_DEFAULTS[key]:
            raise ValueError(
                '%s fixes %s=%r, got %r'
                % (variant, key, RESTORATION_DEFAULTS[key], out[key]))
    if out['restoration_heads'] < 1:
        raise ValueError('restoration_heads must be >= 1')
    for key in _LOCKED_GAIN_ARCH_KEYS:
        if out[key] != _GAIN_ARCH[key]:
            raise ValueError(
                '%s fixes %s=%r, got %r'
                % (variant, key, _GAIN_ARCH[key], out[key]))
    if variant == MODEL_VARIANT:
        for key, expect in _V1_BEHAVIOR.items():
            if out[key] != expect:
                raise ValueError(
                    '%s fixes %s=%r, got %r' % (variant, key, expect, out[key]))
    else:
        if out['gain_estimator'] != _V2_BEHAVIOR['gain_estimator']:
            raise ValueError(
                '%s fixes gain_estimator=%r, got %r' % (
                    variant, _V2_BEHAVIOR['gain_estimator'],
                    out['gain_estimator']))
        if out['matching_mode'] != 'topk_soft':
            raise ValueError(
                '%s fixes matching_mode=%r, got %r' % (
                    variant, 'topk_soft', out['matching_mode']))
        if out['match_topk'] < 1:
            raise ValueError('match_topk must be a positive integer')
        temp = out['match_temperature']
        if temp != temp or temp in (float('inf'), float('-inf')) or temp <= 0:
            raise ValueError(
                'match_temperature must be a finite positive number, got %r' % (
                    temp,))
        if out['match_query_chunk_size'] < 1:
            raise ValueError('match_query_chunk_size must be >= 1')
        if not isinstance(out['match_transfer_recompute'], bool):
            raise ValueError('match_transfer_recompute must be a boolean')
        if out['gain_grid_policy'] not in GRID_POLICIES:
            raise ValueError(
                'gain_grid_policy must be one of %s, got %r' % (
                    GRID_POLICIES, out['gain_grid_policy']))
    return out


def apply_absent_grid_policy(merged: Dict, ckpt_config: Dict) -> None:
    """Old v2 packs have no gain_grid_policy. That means the aspect-first grid.

    Called after the checkpoint is copied onto the baseline, so a newer
    baseline default cannot replace the missing field.
    """
    variant = str(merged.get('model_variant', '')).strip().lower()
    if variant != MODEL_VARIANT_V2:
        return
    recorded = ckpt_config.get('gain_grid_policy', None)
    if recorded is None or recorded == '':
        merged['gain_grid_policy'] = 'legacy_aspect_first'


def restoration_config_from_args(args) -> Dict[str, Any]:
    """Resolved neutral-name config. Pure: does not mutate ``args``.

    ``None`` means "not set" and takes the default. Deprecated ``igab_*``
    aliases warn; an alias that disagrees with a set neutral name is an error.
    """
    get = _getter(args)
    variant = get('model_variant')
    if variant is None:
        variant = MODEL_VARIANT
    variant = str(variant).strip().lower()
    defaults = V2_DEFAULTS if variant == MODEL_VARIANT_V2 else RESTORATION_DEFAULTS
    cfg = {}
    for key, default in defaults.items():
        v = get(key)
        cfg[key] = default if v is None else v
    cfg['model_variant'] = variant
    for old, new in DEPRECATED_ALIASES.items():
        old_v = get(old)
        if old_v is None:
            continue
        warnings.warn(
            '%s is a deprecated alias of %s (the block is LFCA, not IGAB); '
            'use %s.' % (old, new, new), FutureWarning, stacklevel=2)
        new_v = get(new)
        if new_v is not None and _canon(new, new_v) != _canon(new, old_v):
            raise ValueError(
                'Conflicting %s=%r (deprecated alias) and %s=%r'
                % (old, old_v, new, new_v))
        cfg[new] = old_v
    return validate_restoration_config(cfg)


def resolve_restoration_aliases(ns, explicit: Optional[Set[str]] = None):
    """In-place: write neutral names on ``ns`` and drop deprecated aliases.

    Call on a CLI namespace whose restoration/alias options default to None.
    ``explicit`` (set of CLI dests) is updated so an explicit alias counts as
    an explicit neutral option for resume locking.
    """
    cfg = restoration_config_from_args(ns)
    for old, new in DEPRECATED_ALIASES.items():
        if getattr(ns, old, None) is not None and explicit is not None:
            if old in explicit:
                explicit.add(new)
        if hasattr(ns, old):
            delattr(ns, old)
        if explicit is not None:
            explicit.discard(old)
    for k, v in cfg.items():
        setattr(ns, k, v)
    return ns


def _strip_module(key) -> str:
    key = str(key)
    return key[len('module.'):] if key.startswith('module.') else key


def has_legacy_igab_keys(state_keys: Iterable) -> bool:
    for k in state_keys:
        k = _strip_module(k)
        if k.startswith('MainNet.') and any(
                m in k for m in LEGACY_IGAB_STATE_MARKERS):
            return True
    return False


def _incompatible_msg(path, found, how) -> str:
    buildable = ' or '.join(BUILDABLE_VARIANTS)
    return (
        'Checkpoint %s is model_variant=%s (%s); this code builds %s. '
        'IGAB (MainNet.coarse_igab.*/middle_igab.*) is not either LFCA '
        'variant, so it cannot be resumed, evaluated, or partially loaded. '
        'Use commit %s for %s checkpoints.'
        % (path or '<checkpoint>', found, how, buildable,
           LEGACY_IGAB_COMMIT, LEGACY_IGAB_VARIANT))


def reject_legacy_igab_state(state_keys: Iterable, path: str = '') -> None:
    """Raise if a flat state_dict belongs to the IGAB variant."""
    if has_legacy_igab_keys(state_keys):
        raise RuntimeError(_incompatible_msg(
            path, LEGACY_IGAB_VARIANT, 'inferred from coarse_igab/middle_igab keys'))


def checkpoint_model_variant(raw) -> Tuple[Optional[str], str]:
    """(variant, how) for a checkpoint pack or bare state_dict."""
    if not isinstance(raw, dict):
        return None, 'not a dict'
    cfg = raw.get('config') if isinstance(raw.get('config'), dict) else {}
    state = raw.get('model_state_dict')
    if not isinstance(state, dict):
        state = raw
    legacy_keys = has_legacy_igab_keys(state.keys())
    declared = raw.get('model_variant') or cfg.get('model_variant')
    if legacy_keys:
        return LEGACY_IGAB_VARIANT, (
            'inferred from coarse_igab/middle_igab keys'
            + ('; declared %r' % declared if declared else ''))
    if declared:
        return str(declared), 'declared'
    if 'igab_blocks' in cfg or 'igab_heads' in cfg:
        return LEGACY_IGAB_VARIANT, 'inferred from igab_* config, no model_variant'
    return None, 'no model_variant recorded'


def require_model_variant(raw, path: str = '') -> str:
    """Refuse packs that are not a buildable variant (checked first)."""
    variant, how = checkpoint_model_variant(raw)
    if variant not in BUILDABLE_VARIANTS:
        raise RuntimeError(_incompatible_msg(path, variant, how))
    cfg = raw.get('config') if isinstance(raw.get('config'), dict) else {}
    top = raw.get('model_variant')
    if top and cfg.get('model_variant') and top != cfg['model_variant']:
        raise RuntimeError(
            'Checkpoint %s declares model_variant=%r but config says %r'
            % (path or '<checkpoint>', top, cfg['model_variant']))
    return variant
