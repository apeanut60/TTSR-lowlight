"""Model identity + restoration-block config for the block-amend CSFI-U-Net.

Current variant: ``csfi_lfca_v1`` (LFCA at coarse H/4 and middle H/2).
The previous variant (IGAB at the same two positions, commit 5777e63 and
earlier) is referred to as ``csfi_igab_v0``; its checkpoints never declared a
``model_variant`` and are recognised by their state_dict keys / config.

No registry: this code builds exactly one variant and refuses the other.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional, Set, Tuple

MODEL_VARIANT = 'csfi_lfca_v1'
LEGACY_IGAB_VARIANT = 'csfi_igab_v0'
LEGACY_IGAB_COMMIT = '5777e63'

RESTORATION_DEFAULTS = dict(
    model_variant=MODEL_VARIANT,
    restoration_block='lfca',
    restoration_blocks='1,1',      # depth at coarse(H/4),middle(H/2)
    restoration_heads=1,
    restoration_ffn_expansion=4,
    lowpass_kernel='binomial',
    lowpass_kernel_size=5,
)
RESTORATION_CONFIG_KEYS = tuple(RESTORATION_DEFAULTS)

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


def _canon(key, value):
    if key == 'restoration_blocks':
        return '%d,%d' % parse_restoration_blocks(value)
    if key in ('restoration_heads', 'restoration_ffn_expansion',
               'lowpass_kernel_size'):
        return int(value)
    return str(value).strip().lower()


def _getter(args):
    if isinstance(args, dict):
        return lambda k: args.get(k)
    return lambda k: getattr(args, k, None)


def validate_restoration_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: _canon(k, cfg[k]) for k in RESTORATION_CONFIG_KEYS}
    if out['model_variant'] != MODEL_VARIANT:
        hint = ''
        if out['model_variant'] == LEGACY_IGAB_VARIANT:
            hint = (' The IGAB variant is only buildable from commit %s.'
                    % LEGACY_IGAB_COMMIT)
        raise ValueError(
            'model_variant=%r is not buildable by this code (builds %r).%s'
            % (out['model_variant'], MODEL_VARIANT, hint))
    for key in ('restoration_block', 'restoration_ffn_expansion',
                'lowpass_kernel', 'lowpass_kernel_size'):
        if out[key] != RESTORATION_DEFAULTS[key]:
            raise ValueError(
                '%s fixes %s=%r, got %r'
                % (MODEL_VARIANT, key, RESTORATION_DEFAULTS[key], out[key]))
    if out['restoration_heads'] < 1:
        raise ValueError('restoration_heads must be >= 1')
    return out


def restoration_config_from_args(args) -> Dict[str, Any]:
    """Resolved neutral-name config. Pure: does not mutate ``args``.

    ``None`` means "not set" and takes the default. Deprecated ``igab_*``
    aliases warn; an alias that disagrees with a set neutral name is an error.
    """
    get = _getter(args)
    cfg = {}
    for key, default in RESTORATION_DEFAULTS.items():
        v = get(key)
        cfg[key] = default if v is None else v
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
    return (
        'Checkpoint %s is model_variant=%s (%s); this code builds %s. '
        'The restoration block differs (IGAB: MainNet.coarse_igab.*/'
        'middle_igab.* with illumination-modulated Value; LFCA: '
        'MainNet.coarse_freq_block.*/middle_freq_block.*), so it cannot be '
        'resumed, evaluated, or partially/shape-filtered loaded here. '
        'Train %s fresh (optionally --init_scope lte_only), or use commit %s '
        'for %s checkpoints.'
        % (path or '<checkpoint>', found, how, MODEL_VARIANT, MODEL_VARIANT,
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
    """Refuse packs that are not ``MODEL_VARIANT`` (checked before anything else)."""
    variant, how = checkpoint_model_variant(raw)
    if variant != MODEL_VARIANT:
        raise RuntimeError(_incompatible_msg(path, variant, how))
    cfg = raw.get('config') if isinstance(raw.get('config'), dict) else {}
    top = raw.get('model_variant')
    if top and cfg.get('model_variant') and top != cfg['model_variant']:
        raise RuntimeError(
            'Checkpoint %s declares model_variant=%r but config says %r'
            % (path or '<checkpoint>', top, cfg['model_variant']))
    return variant
