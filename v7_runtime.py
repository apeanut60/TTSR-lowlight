"""V7 Task-Decoupled Ref on frozen LLFormer — lock, paths, constants."""

from __future__ import annotations

import os

from model.c2_extractor import C2_FEATURE_CKPT
from model.V6LLFormerRefBridge import CANONICAL_BASE
from v3b_runtime import GRAD_ACCUM, LR, SEED

ARM_A1 = 'A1_td_c2'
ARMS = (ARM_A1,)

# Stage A1 budget (pre-registered; not V6's 30k)
DEFAULT_UPDATES = 10000
CKPT_STEPS = (0, 100, 1000, 3000, 5000, 10000)
EVAL_STEPS = (1000, 3000, 10000)
OPTIMIZER = 'Adam'
LOSS = 'L1_rec_tex'  # L_rec + L_tex, paper-guided L1-only
ZERO_INIT = True
REFERENCE_VARIANT = 'nanobanana_ref_v2'
PAIR_STATES_A1 = ('correct',)  # correct-only first round

ROOT = '/root/data/experiments/v7_td_llformer_c2'
SRC = '/root/data/experiments/v3a1_lolv2real'
SPLIT_JSON = '/root/data/experiments/v3a4_lolv2real/splits/split.json'
MISMATCH_TRAIN = (
    '/root/data/experiments/v3a4_lolv2real/mappings/mismatch_train_575.json')
MISMATCH_DEV = (
    '/root/data/experiments/v3a4_lolv2real/mappings/mismatch_dev_64.json')
# Reuse V6 E110 Y0 cache (same Base SHA)
Y0_CACHE_SRC = (
    '/root/data/experiments/v6_llformer_aligned_ref/cache_y0_llformer_e110')
CACHE_NAME = 'cache_y0_llformer_e110'
BASE_CKPT = CANONICAL_BASE
C2_CKPT = C2_FEATURE_CKPT

FORMAL_LOCK_KEYS_V7 = (
    'stage', 'repo_commit',
    'base_ckpt', 'base_ckpt_sha256', 'base_cache_metadata_sha256',
    'c2_ckpt', 'c2_ckpt_sha256',
    'split_sha256', 'mismatch_train_sha256', 'mismatch_dev_sha256',
    'reference_variant', 'pair_states',
    'architecture', 'match_encoder', 'texture_encoder',
    'search_range', 'correlation_type', 'warp_type', 'dcn_kernel',
    'zero_init', 'injection_point',
    'loss', 'optimizer', 'lr', 'weight_decay',
    'updates', 'grad_accum', 'seed', 'pair_schedule_seed',
    'checkpoint_steps', 'official_test_allowed',
    'frozen_base', 'frozen_c2', 'trainable_modules', 'v7_init_sha',
)


def lock_architecture_fields():
    return dict(
        stage='V7.0',
        architecture='V7_task_decoupled_c2_llformer',
        match_encoder='C2_ContrasExtractorSep_feature_extraction',
        texture_encoder='V6TextureEncoder_16_32_64',
        search_range='global',
        correlation_type='chunked_global_cosine',
        warp_type='bilinear_flow',
        dcn_kernel=3,
        zero_init=ZERO_INIT,
        injection_point='post_y0_att',  # NOT D2/D1
        loss=LOSS,
        official_test_allowed=False,
        frozen_base=True,
        frozen_c2=True,
        pair_states=list(PAIR_STATES_A1),
        trainable_modules=[
            'TextureEncoder', 'FlowDeformTextureAlign',
            'ContentFeatureH4', 'TextureResidualDecoder',
            'AdaptiveTextureTransfer',
        ],
        c2_ckpt=C2_CKPT,
        base_ckpt=BASE_CKPT,
    )


def ensure_y0_cache_link(root: str = ROOT) -> str:
    """Symlink V6 E110 Y0 cache into V7 root if missing."""
    dst = os.path.join(root, CACHE_NAME)
    if os.path.isdir(dst):
        return dst
    os.makedirs(root, exist_ok=True)
    if not os.path.isdir(Y0_CACHE_SRC):
        raise SystemExit('missing Y0 cache at %s' % Y0_CACHE_SRC)
    os.symlink(Y0_CACHE_SRC, dst)
    print('[V7] linked Y0 cache → %s' % dst, flush=True)
    return dst
