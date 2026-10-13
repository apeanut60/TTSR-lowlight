"""V8-B0 / LLFormer-E110 residual pilot — paths and constants.

Reuses historical V3-B.0 head + frozen R1 SharedEncoder/LocalSoftMatch.
Y0 comes from frozen LLFormer E110 cache (not Retinexformer).
"""

from __future__ import annotations

import os

SEED = 42
LR = 1e-4
GRAD_ACCUM = 1
UPDATES_GATE1 = 400
N_TRAIN = 32
OPTIMIZER = 'Adam'
LOSS = 'MSE'
ARCHITECTURE = 'V3B0ResidualFusion'
ARM_SELF = 'B0_Self'
ARM_NANO = 'B0_Nano'

REPO = '/root/projects/TTSR-lowlight'
ROOT = '/root/data/experiments/V8_B0_llformer_e110'
SRC = '/root/data/experiments/v3a1_lolv2real'
SPLIT_JSON = '/root/data/experiments/v3a4_lolv2real/splits/split.json'
MISMATCH_TRAIN = (
    '/root/data/experiments/v3a4_lolv2real/mappings/mismatch_train_575.json')
MISMATCH_DEV = (
    '/root/data/experiments/v3a4_lolv2real/mappings/mismatch_dev_64.json')
R1_CKPT = os.path.join(SRC, 'R1_v2stable_naive_s42', 'checkpoint_03000.pt')
Y0_CACHE = (
    '/root/data/experiments/v6_llformer_aligned_ref/cache_y0_llformer_e110')
BASE_CKPT = (
    '/root/data/experiments/llformer_lolv2real/canonical/'
    'llformer_lolv2real_base_E110.pth')
EXCLUSIVE_IDS = (
    '/root/projects/TTSR-lowlight/docs/v7_code_review_fix/gate2_3a/'
    'exclusive_id_lists.json')
REFERENCE_VARIANT = 'nanobanana_ref_v2'

# Plan file was not present in workspace; Gate0/1 follow user query + B0 protocol.
PLAN_NOTE = (
    'V8_B0_LLFormer_E110_PSNR_PLAN.md not found in repo at run time; '
    'implemented from Gate0/Gate1 query spec + historical V3-B.0')


def arm_root(arm: str, root: str = ROOT) -> str:
    return os.path.join(root, 'arms', arm)


def setting_paths(root: str = ROOT):
    return dict(
        root=root,
        init=os.path.join(root, 'init'),
        gate0=os.path.join(root, 'gate0'),
        gate1=os.path.join(root, 'gate1'),
        logs=os.path.join(root, 'logs'),
    )
