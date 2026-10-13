"""V7-RIVER runtime: paths, lock fields, settings A/B."""

from __future__ import annotations

import os

from model.V6LLFormerRefBridge import CANONICAL_BASE, INJECTION_POINT
from model.V7Model import ARCHITECTURE
from v3b_runtime import GRAD_ACCUM, LR, SEED

ARM_A = 'A_fusion_only'
ARM_B = 'B_fusion_align'
SETTING_ARMS = {'A': ARM_A, 'B': ARM_B}

DEFAULT_UPDATES = 3000
CKPT_STEPS = (0, 100, 1000, 3000)
OPTIMIZER = 'Adam'
LOSS = 'MSE'
ZERO_INIT = True
REFERENCE_VARIANT = 'nanobanana_ref_v2'

ROOT = '/root/data/experiments/V7_RIVER_validation/experiments'
SRC = '/root/data/experiments/v3a1_lolv2real'
SPLIT_JSON = '/root/data/experiments/v3a4_lolv2real/splits/split.json'
MISMATCH_TRAIN = (
    '/root/data/experiments/v3a4_lolv2real/mappings/mismatch_train_575.json')
Y0_CACHE_SRC = (
    '/root/data/experiments/v6_llformer_aligned_ref/cache_y0_llformer_e110')
CACHE_NAME = 'cache_y0_llformer_e110'
BASE_CKPT = CANONICAL_BASE
VALIDATION_ROOT = '/root/data/experiments/V7_RIVER_validation'
# Selection / ablation on Official Test (user override; not dev64)
EVAL_SPLIT = 'official_Test'

FORMAL_LOCK_KEYS = (
    'stage', 'repo_commit', 'setting', 'arm',
    'base_ckpt', 'base_ckpt_sha256', 'base_cache_metadata_sha256',
    'split_sha256', 'mismatch_train_sha256',
    'reference_variant', 'eval_split',
    'architecture', 'match_encoder', 'texture_encoder', 'fusion',
    'search_range', 'correlation_type', 'warp_type', 'dcn_kernel',
    'zero_init', 'injection_point',
    'loss', 'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
    'frozen_base', 'frozen_align', 'trainable_modules', 'v7_init_sha',
)


def lock_architecture_fields(setting: str):
    setting = setting.upper()
    frozen_align = setting == 'A'
    trainable = (['V7RiverFusion_H4', 'V7RiverFusion_H2'] if frozen_align else [
        'MatchEncoder', 'TextureEncoder', 'ChunkedHardMatcher',
        'OffsetNet', 'DCN', 'FlowDeformFuse',
        'V7RiverFusion_H4', 'V7RiverFusion_H2',
    ])
    return dict(
        stage='V7.RIVER',
        setting=setting,
        arm=SETTING_ARMS[setting],
        architecture=ARCHITECTURE,
        match_encoder='V5MatchEncoder_3_32_64s2_64_96s2_96',
        texture_encoder='TexturePyramid_(16, 32, 64)',
        fusion='V7RiverFusion_AttnQKV_zero_out',
        search_range='global',
        correlation_type='chunked_global_cosine',
        warp_type='backward_grid_sample',
        dcn_kernel=3,
        zero_init=ZERO_INIT,
        injection_point=INJECTION_POINT,
        loss=LOSS,
        official_test_allowed=True,
        eval_split=EVAL_SPLIT,
        frozen_base=True,
        frozen_align=frozen_align,
        trainable_modules=trainable,
        base_ckpt=BASE_CKPT,
        reference_variant=REFERENCE_VARIANT,
    )


def ensure_y0_cache_link(root: str = ROOT) -> str:
    dst = os.path.join(root, CACHE_NAME)
    if os.path.isdir(dst):
        return dst
    os.makedirs(root, exist_ok=True)
    if not os.path.isdir(Y0_CACHE_SRC):
        raise SystemExit('missing Y0 cache at %s' % Y0_CACHE_SRC)
    os.symlink(Y0_CACHE_SRC, dst)
    print('[V7-RIVER] linked Y0 cache → %s' % dst, flush=True)
    return dst


def setting_root(setting: str, root: str = ROOT) -> str:
    return os.path.join(root, SETTING_ARMS[setting.upper()])
