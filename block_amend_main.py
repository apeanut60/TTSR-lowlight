#!/usr/bin/env python3
"""Block-amend CSFI-U-Net baseline entry (fresh / resume / eval / preflight).

Does not register historical extra_test loaders or Official Test during training.
GT never enters the reference path. See docs/BLOCK_AMEND_TRAINING_BASELINE.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
import uuid
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from block_amend_ckpt import (
    CHECKPOINT_SCHEMA_VERSION, LOCKED_CONFIG_KEYS, RESUME_ALLOWED_OVERRIDES,
    atomic_torch_save, build_checkpoint_pack, capture_rng_state,
    check_grad_finite, explicit_cli_dests, load_local_checkpoint,
    merge_resume_config, require_resume_fields, resolved_config_from_args,
    restore_rng_state,
)
from dataset.block_amend_lolv2 import (
    REFERENCE_SOURCE, BlockAmendEvalSet, BlockAmendTrainSet, assert_split_identity,
    collect_triplets, load_manifest_items, manifest_identity_hash,
    materialize_items, save_split_manifests, split_train_val,
    validate_split_manifests, write_coverage_manifests, MANIFEST_HASH_VERSION,
)
from model.TTSREnhance import TTSREnhance
from model.restoration_config import (
    MODEL_VARIANT, RESTORATION_DEFAULTS, reject_legacy_igab_state,
    require_model_variant, resolve_restoration_aliases,
)
from model.weight_init import apply_init_mode, load_strict_state, log_param_groups
from model.tiled_infer import tiled_forward
from utils_metrics_block_amend import (
    PROTOCOL_NAME, metrics_rgb_float_clipped_v1, to_uint8_png,
)
from loss.loss_enhance import get_loss_dict_enhance
from utils import Logger


BASELINE_FIXED = dict(
    backbone='csfi_unet',
    num_res_blocks='8+8+4+2',
    n_feats=64,
    res_scale=1.0,
    # model identity: csfi_lfca_v1 / lfca / depth 1,1 / heads 1 / FFN x4 /
    # binomial 5x5 low-pass (see model/restoration_config.py)
    **RESTORATION_DEFAULTS,
    enable_lightup=True,
    use_global_illum=False,
    use_ref_illum=False,
    ref_correction=True,
    freeze_lte=True,
    freeze_stages='',
    query_illum_align=False,
    texture_fusion=False,
    stage_mode_lv3='normal',
    stage_mode_lv2='normal',
    stage_mode_lv1='normal',
    eval_mean_align=False,
    eval_chroma_gain=1.0,
    eval_tta=False,
    enhance_mode=True,
    ref_degrade=False,
    ref_illum_pool=8,
    ref_correction_feats=16,
    query_illum_alpha=0.0,
    query_hist_beta=0.0,
    query_gamma_min=0.3,
    query_gamma_max=1.0,
    query_hist_bins=256,
    texture_fusion_lambda=0.0,
    texture_fusion_s_source='a0',
    gate_mode=None,
    gate_shift='',
    gate_smooth_kernel=31,
    collect_gate_stats=False,
    collect_residual_decomp=False,
    disable_T_lv3=False,
    disable_T_lv2=False,
    disable_T_lv1=False,
    tile_size=256,
    tile_overlap=96,
    tile_window='cosine',
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    rec_loss_type='l1',
    rec_w=1.0,
    ms_ssim_w=1.0,
    per_w=0.0,
    tpl_w=0.0,
    adv_w=0.0,
    illum_smooth_w=0.0,
    color_w=0.0,
    exposure_w=0.0,
    illum_match_w=0.0,
    ref_correct_w=0.0,
)


def _git_info(root: str) -> dict:
    info = {'commit': None, 'dirty': None}
    try:
        info['commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, text=True).strip()
        info['dirty'] = bool(dirty)
    except Exception as e:
        info['error'] = str(e)
    return info


def worker_init_fn(worker_id: int):
    """Top-level picklable worker init (Python/NumPy only)."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_parser():
    p = argparse.ArgumentParser(description='block-amend CSFI baseline')
    p.add_argument('--mode', type=str, required=True,
                   choices=['preflight', 'train', 'resume', 'eval'])
    p.add_argument('--dataset_dir', type=str,
                   default='/root/data/datasets/lol-v2-real')
    p.add_argument('--train_ref_subdir', type=str, default='nanobanana_ref_v2')
    p.add_argument('--test_ref_subdir', type=str, default='nanobanana_ref_v3')
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--run_name', type=str, default='')
    p.add_argument('--init_scope', type=str, default='fresh',
                   choices=['fresh', 'lte_only', 'inference'])
    p.add_argument('--init_checkpoint', type=str, default='')
    p.add_argument('--resume_checkpoint', type=str, default='')
    p.add_argument('--model_path', type=str, default='',
                   help='Eval: a complete checkpoint pack (config + model_variant). '
                        'A bare state_dict is refused.')
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--split_dir', type=str, default='',
                   help='Existing split manifests dir (split_train.json / split_val.json)')
    p.add_argument('--allow_partial_coverage', action='store_true',
                   help='Write include/exclude manifests when some Nano refs missing')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--train_crop_size', type=int, default=160)
    p.add_argument('--lr_rate', type=float, default=1e-4)
    p.add_argument('--lr_rate_lte', type=float, default=1e-6)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--decay', type=int, default=50)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--print_every', type=int, default=10)
    p.add_argument('--val_every', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--num_gpu', type=int, default=1)
    p.add_argument('--eval_split', type=str, default='val',
                   choices=['val', 'test'],
                   help='eval mode: internal val or Official Test')
    p.add_argument('--save_images', action='store_true')
    p.add_argument('--log_file_name', type=str, default='block_amend.log')
    p.add_argument('--persistent_workers', action='store_true',
                   help='Not supported for epoch-aware aug; will error if set')
    # Model identity. None = baseline default (see RESTORATION_DEFAULTS).
    p.add_argument('--model_variant', type=str, default=None,
                   help='Asserted model identity; this code only builds %s'
                        % MODEL_VARIANT)
    p.add_argument('--restoration_blocks', type=str, default=None,
                   help='Restoration block depth coarse(H/4),middle(H/2); '
                        'baseline 1,1')
    p.add_argument('--restoration_heads', type=int, default=None,
                   help='Restoration block heads; baseline 1')
    p.add_argument('--igab_blocks', type=str, default=None,
                   help='DEPRECATED alias of --restoration_blocks')
    p.add_argument('--igab_heads', type=int, default=None,
                   help='DEPRECATED alias of --restoration_heads')
    return p


def make_args(cli) -> SimpleNamespace:
    d = dict(BASELINE_FIXED)
    d.update(vars(cli))
    d['dataset'] = 'block_amend_lolv2'
    d['logger_name'] = 'block_amend'
    d['metric_protocol'] = PROTOCOL_NAME
    return SimpleNamespace(**d)


def ensure_save_dir_fresh(save_dir: str):
    if os.path.isdir(save_dir) and os.listdir(save_dir):
        raise SystemExit(
            'save_dir is non-empty (%s). Fresh training refuses to reuse it. '
            'Use --mode resume with --resume_checkpoint, or choose a new RUN_NAME.'
            % save_dir)
    os.makedirs(save_dir, exist_ok=True)


def build_model(args, device):
    model = TTSREnhance(args).to(device)
    if args.freeze_stages:
        from trainer import Trainer
        model, names = Trainer.freeze_mainnet_stages(
            model, args.freeze_stages, args.num_gpu)
        print('[Freeze] stages=%s n_params=%d' % (args.freeze_stages, len(names)))
    return model


def build_optimizer(model, args):
    wrap = model.module if hasattr(model, 'module') else model
    if not args.use_global_illum:
        for n, p in wrap.MainNet.named_parameters():
            if n.startswith('global_illum_head.'):
                p.requires_grad = False
    if not args.use_ref_illum:
        for n, p in wrap.MainNet.named_parameters():
            if n.startswith('ref_illum.'):
                p.requires_grad = False
    groups = []
    main_params = [p for p in wrap.MainNet.parameters() if p.requires_grad]
    if main_params:
        groups.append({'params': main_params, 'lr': args.lr_rate})
    if wrap.RefCorrection is not None:
        rc = [p for p in wrap.RefCorrection.parameters() if p.requires_grad]
        if rc:
            groups.append({'params': rc, 'lr': args.lr_rate})
    lte = [p for p in wrap.LTE.parameters() if p.requires_grad]
    if lte:
        groups.append({'params': lte, 'lr': args.lr_rate_lte})
    if not groups:
        raise RuntimeError('No trainable parameters')
    seen = set()
    for g in groups:
        for p in g['params']:
            if id(p) in seen:
                raise RuntimeError('Duplicate parameter in optimizer groups')
            seen.add(id(p))
    opt = optim.Adam(groups, betas=(args.beta1, args.beta2), eps=args.eps)
    sched = optim.lr_scheduler.StepLR(opt, step_size=args.decay, gamma=args.gamma)
    return opt, sched


def _load_locked_split(split_dir, args, require_files=True):
    train_path = os.path.join(split_dir, 'split_train.json')
    val_path = os.path.join(split_dir, 'split_val.json')
    if require_files and (not os.path.isfile(train_path) or not os.path.isfile(val_path)):
        raise FileNotFoundError(
            'Locked split manifests missing under %s (train=%s val=%s). '
            'Resume/eval must not re-split.'
            % (split_dir, os.path.isfile(train_path), os.path.isfile(val_path)))
    train_items = load_manifest_items(train_path)
    val_items = load_manifest_items(val_path)
    identity = validate_split_manifests(
        train_items, val_items, args.dataset_dir, args.train_ref_subdir,
        batch_size=args.batch_size, drop_last=True)
    return train_items, val_items, identity


def prepare_data(args, logger, mode, explicit_split_dir: bool):
    """Create or load locked train/val. Never re-split on resume."""
    split_dir = args.split_dir or os.path.join(args.save_dir, 'splits')
    train_path = os.path.join(split_dir, 'split_train.json')
    val_path = os.path.join(split_dir, 'split_val.json')
    have = os.path.isfile(train_path) and os.path.isfile(val_path)

    if mode == 'resume':
        if not have:
            raise RuntimeError(
                'Resume requires existing split manifests at %s' % split_dir)
        train_items, val_items, identity = _load_locked_split(split_dir, args)
        logger.info('Resume locked split: %s' % json.dumps(identity))
    elif mode == 'train' and explicit_split_dir:
        if not have:
            raise RuntimeError(
                '--split_dir was set but manifests missing at %s' % split_dir)
        train_items, val_items, identity = _load_locked_split(split_dir, args)
        logger.info('Using explicit split_dir: %s' % json.dumps(identity))
    elif mode == 'train' and have and not explicit_split_dir:
        # save_dir/splits already exists under a fresh empty check — rare
        train_items, val_items, identity = _load_locked_split(split_dir, args)
        logger.info('Using existing split: %s' % json.dumps(identity))
    else:
        # fresh create
        if getattr(args, 'allow_partial_coverage', False):
            cov = write_coverage_manifests(
                args.dataset_dir, os.path.join(args.save_dir, 'coverage'),
                subset='real', ref_subdir=args.train_ref_subdir, split='Train',
                allow_partial=True)
            logger.info('Coverage: %s' % json.dumps(cov))
            include = load_manifest_items(cov['include_manifest'])
        else:
            include, _ = collect_triplets(
                args.dataset_dir, 'Train', subset='real',
                ref_subdir=args.train_ref_subdir, require_all=True)
        train_items, val_items = split_train_val(
            include, val_ratio=args.val_ratio, split_seed=args.split_seed)
        identity = save_split_manifests(
            train_items, val_items, split_dir,
            dataset_dir=args.dataset_dir, ref_subdir=args.train_ref_subdir,
            batch_size=args.batch_size)
        if getattr(args, 'allow_partial_coverage', False):
            identity['coverage'] = 'explicit_partial'
            identity['include_count'] = int(cov['include_count'])
            identity['exclude_missing_ref'] = int(cov['exclude_missing_ref'])
        else:
            identity['coverage'] = 'strict'
            identity['include_count'] = len(include)
            identity['exclude_missing_ref'] = 0
        logger.info('Created split: %s' % json.dumps(identity))

    if getattr(args, 'persistent_workers', False):
        raise RuntimeError(
            'persistent_workers=True is not supported with epoch-aware '
            'augmentation (set_epoch would not reach workers).')

    train_set = BlockAmendTrainSet(
        train_items, crop_size=args.train_crop_size,
        ref_degrade=args.ref_degrade, seed=args.seed)
    val_set = BlockAmendEvalSet(val_items)

    train_gen = torch.Generator()
    train_gen.manual_seed(int(args.seed))
    val_gen = torch.Generator()
    val_gen.manual_seed(int(args.seed) + 1009)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=train_gen, persistent_workers=False)
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=val_gen, persistent_workers=False)

    return (train_loader, val_loader, train_set, val_set,
            train_items, val_items, split_dir, identity, train_gen, val_gen)


def evaluate_loader(model, loader, args, device, save_dir=None, tag='val',
                    split_identity=None):
    model.eval()
    rows = []
    psnr_sum = ssim_sum = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            lr = batch['LR'].to(device)
            lrsr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            refsr = batch['Ref_sr'].to(device)
            sid = batch['id'][0] if isinstance(batch['id'], (list, tuple)) else batch['id']
            if isinstance(sid, torch.Tensor):
                sid = str(sid.item()) if sid.numel() == 1 else str(sid)
            sr = tiled_forward(
                model, lr, lrsr, ref, refsr,
                tile_size=args.tile_size, overlap=args.tile_overlap,
                apply_illum=True, apply_ref_illum=True)
            m = metrics_rgb_float_clipped_v1(sr, hr)
            m['id'] = sid
            rows.append(m)
            psnr_sum += m['psnr_rgb']
            ssim_sum += m['ssim_rgb']
            if save_dir and getattr(args, 'save_images', False):
                from imageio import imsave
                out = os.path.join(save_dir, 'images_%s' % tag)
                os.makedirs(out, exist_ok=True)
                imsave(os.path.join(out, '%s.png' % sid), to_uint8_png(sr))
    n = max(len(rows), 1)
    summary = {
        'protocol': PROTOCOL_NAME,
        'n': len(rows),
        'psnr_rgb_mean': psnr_sum / n,
        'ssim_rgb_mean': ssim_sum / n,
        'per_image': rows,
        'split_identity': split_identity,
        'manifest_hash_version': MANIFEST_HASH_VERSION,
    }
    return summary


def save_run_meta_fresh(args, root, identity, model, run_id, extra=None):
    wrap = model.module if hasattr(model, 'module') else model
    n_params = sum(p.numel() for p in wrap.parameters())
    identity_fn = getattr(wrap.MainNet, 'restoration_identity', None)
    meta = {
        'model_variant': getattr(args, 'model_variant', None),
        'restoration': identity_fn() if identity_fn else None,
        'n_params_mainnet_trainable': int(sum(
            p.numel() for p in wrap.MainNet.parameters() if p.requires_grad)),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'run_id': run_id,
        'git': _git_info(root),
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'n_params': int(n_params),
        'metric_protocol': PROTOCOL_NAME,
        'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
        'manifest_hash_version': MANIFEST_HASH_VERSION,
        'reference_source': 'generated_nano',
        'train_ref_subdir': args.train_ref_subdir,
        'test_ref_subdir': args.test_ref_subdir,
        'augmentation': 'paired_crop_flip_rot_epoch_aware',
        'ref_degrade': bool(args.ref_degrade),
        'init_scope': args.init_scope,
        'init_checkpoint': args.init_checkpoint,
        'config': resolved_config_from_args(args),
        'split_identity': identity,
        'resume_history': [],
    }
    if extra:
        meta.update(extra)
    path = os.path.join(args.save_dir, 'run_meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def append_resume_record(save_dir, record: dict):
    path = os.path.join(save_dir, 'run_meta.json')
    if os.path.isfile(path):
        with open(path) as f:
            meta = json.load(f)
    else:
        meta = {'resume_history': []}
    meta.setdefault('resume_history', []).append(record)
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)


def _assert_best_identity_safe(save_dir, pack, logger):
    """Refuse resume that would risk overwriting a better best with worse state."""
    best_path = os.path.join(save_dir, 'checkpoint_best_val_psnr.pt')
    if not os.path.isfile(best_path):
        return
    best = load_local_checkpoint(best_path)
    if not isinstance(best, dict):
        raise RuntimeError('Corrupt best checkpoint at %s' % best_path)
    if best.get('run_id') and pack.get('run_id') and best['run_id'] != pack['run_id']:
        raise RuntimeError(
            'best checkpoint run_id=%s != resume pack run_id=%s'
            % (best.get('run_id'), pack.get('run_id')))
    # Historical best in pack must not be worse than on-disk best bookkeeping
    on_disk = float(best.get('best_val_psnr', -1))
    in_pack = float(pack.get('best_val_psnr', -1))
    if on_disk > in_pack + 1e-8:
        raise RuntimeError(
            'Resume last.best_val_psnr=%.6f < on-disk best=%.6f (epoch %s). '
            'Refusing to continue (would risk overwriting better best).'
            % (in_pack, on_disk, best.get('best_epoch')))
    logger.info('Best identity OK: best_val_psnr=%.4f best_epoch=%s'
                % (on_disk, best.get('best_epoch')))


def train_loop(args, logger, model, opt, sched, train_loader, val_loader,
               train_set, train_gen, val_gen, split_identity, run_id,
               start_epoch=1, global_step=0, best_psnr=-1.0, best_epoch=0):
    device = next(model.parameters()).device
    loss_all = get_loss_dict_enhance(args, logger)
    last_path = os.path.join(args.save_dir, 'checkpoint_last.pt')
    best_path = os.path.join(args.save_dir, 'checkpoint_best_val_psnr.pt')

    for epoch in range(start_epoch, args.num_epochs + 1):
        # Epoch-aware aug BEFORE creating the iterator (persistent_workers=False).
        train_set.set_epoch(epoch)
        model.train()
        logger.info('Epoch %d lr=%e' % (epoch, opt.param_groups[0]['lr']))
        n_batches = 0
        for i_batch, batch in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            lr = batch['LR'].to(device)
            lrsr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            refsr = batch['Ref_sr'].to(device)
            sid = batch.get('id', '?')
            sr, *_ = model(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
            loss = args.rec_w * loss_all['rec_loss'](sr, hr)
            loss_terms = {'rec': float(loss.detach())}
            if 'ms_ssim_loss' in loss_all and args.ms_ssim_w > 0:
                ms = args.ms_ssim_w * loss_all['ms_ssim_loss'](sr, hr)
                loss = loss + ms
                loss_terms['ms_ssim'] = float(ms.detach())
            if (not torch.isfinite(loss).all()) or (not torch.isfinite(sr).all()):
                raise RuntimeError(
                    'Non-finite loss/output epoch=%d step=%d id=%s loss=%s terms=%s'
                    % (epoch, i_batch + 1, sid, loss.detach(), loss_terms))
            loss.backward()
            check_grad_finite(model, epoch, i_batch + 1, sid, loss, sr)
            opt.step()
            global_step += 1
            n_batches += 1
            if (i_batch + 1) % args.print_every == 0:
                logger.info('epoch %d batch %d loss %.6f'
                            % (epoch, i_batch + 1, float(loss)))

        if n_batches == 0:
            raise RuntimeError(
                'Epoch %d produced zero batches (check batch_size/drop_last)'
                % epoch)

        # Scheduler once per completed epoch after optimizer updates.
        sched.step()

        did_val = (epoch % args.val_every == 0)
        if did_val:
            summary = evaluate_loader(
                model, val_loader, args, device,
                save_dir=args.save_dir, tag='val_ep%05d' % epoch,
                split_identity=split_identity)
            with open(os.path.join(
                    args.save_dir, 'val_metrics_ep%05d.json' % epoch), 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(
                'val epoch %d PSNR_rgb %.4f SSIM %.4f (%s)'
                % (epoch, summary['psnr_rgb_mean'], summary['ssim_rgb_mean'],
                   PROTOCOL_NAME))
            # Strict improvement only (ties do not overwrite best).
            if summary['psnr_rgb_mean'] > best_psnr:
                best_psnr = float(summary['psnr_rgb_mean'])
                best_epoch = int(epoch)

        # Build pack AFTER val/best update so last carries true historical best.
        rng_state = capture_rng_state(train_gen, val_gen)
        pack = build_checkpoint_pack(
            model=model, opt=opt, sched=sched,
            epoch=epoch, global_step=global_step,
            best_val_psnr=best_psnr, best_epoch=best_epoch,
            validation_completed=did_val,
            args=args, split_identity=split_identity,
            init_scope=args.init_scope,
            init_checkpoint=getattr(args, 'init_checkpoint', '') or '',
            run_id=run_id, rng_state=rng_state,
            metric_protocol=PROTOCOL_NAME,
        )

        # Save best first (if improved this epoch), then last.
        if did_val and best_epoch == epoch:
            atomic_torch_save(pack, best_path)
            logger.info('New best_val_psnr=%.4f best_epoch=%d' % (best_psnr, best_epoch))
        atomic_torch_save(pack, last_path)
        if getattr(args, 'save_every_epoch', False):
            ep_path = os.path.join(
                args.save_dir, 'checkpoint_epoch_%04d.pt' % epoch)
            atomic_torch_save(pack, ep_path)
            logger.info('Saved epoch snapshot %s' % ep_path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit = explicit_cli_dests(argv)
    parser = build_parser()
    cli = parser.parse_args(argv)
    root = os.path.dirname(os.path.abspath(__file__))
    # Neutral restoration names on the CLI namespace; deprecated igab_* aliases
    # warn, conflict with an explicit neutral option is an error, and an
    # explicit alias counts as an explicit neutral option for resume locking.
    try:
        resolve_restoration_aliases(cli, explicit)
    except ValueError as e:
        raise SystemExit(str(e))

    if cli.run_name:
        cli.save_dir = os.path.join(cli.save_dir, cli.run_name)

    device = torch.device('cpu' if cli.cpu else 'cuda')

    # ---------- resume: config from checkpoint first ----------
    if cli.mode == 'resume':
        if not cli.resume_checkpoint:
            raise SystemExit('resume requires --resume_checkpoint')
        # One CPU load. Reused for the weight/optimizer/RNG restore below;
        # the pack is not read again and is not moved onto CUDA.
        raw = load_local_checkpoint(cli.resume_checkpoint)
        require_resume_fields(raw, cli.resume_checkpoint)
        args, applied = merge_resume_config(
            raw['config'], cli, explicit, BASELINE_FIXED)
        args.resume_checkpoint = cli.resume_checkpoint
        args.save_dir = cli.save_dir
        if 'dataset_dir' in explicit:
            args.dataset_dir = cli.dataset_dir
        if 'split_dir' in explicit:
            args.split_dir = cli.split_dir
        args.mode = 'resume'
        # Extending num_epochs must not rebuild StepLR from scratch later
        os.makedirs(args.save_dir, exist_ok=True)
    elif cli.mode == 'eval':
        if not cli.model_path:
            raise SystemExit('eval requires --model_path')
        raw = load_local_checkpoint(cli.model_path)
        # Eval only rebuilds a model from a pack that carries its own config
        # and model_variant. A bare state_dict cannot recover the behavior.
        pack_ok = (isinstance(raw, dict) and isinstance(raw.get('config'), dict)
                   and isinstance(raw.get('model_state_dict'), dict)
                   and (raw.get('model_variant') or raw['config'].get('model_variant')))
        if not pack_ok:
            if isinstance(raw, dict):
                state = raw.get('model_state_dict', raw)
                if isinstance(state, dict):
                    reject_legacy_igab_state(state.keys(), cli.model_path)
            raise SystemExit(
                'eval requires a complete checkpoint pack with config and '
                'model_variant. A bare state_dict cannot safely rebuild the '
                'behavior config (%s).' % cli.model_path)
        # Other variants (IGAB) are refused before a model is built.
        require_model_variant(raw, cli.model_path)
        args, applied = merge_resume_config(
            raw['config'], cli, explicit, BASELINE_FIXED)
        args.model_path = cli.model_path
        args.mode = 'eval'
        args.eval_split = cli.eval_split
        args.save_images = cli.save_images
        args.save_dir = cli.save_dir
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args = make_args(cli)
        applied = {}
        if args.mode == 'train':
            ensure_save_dir_fresh(args.save_dir)
        else:
            os.makedirs(args.save_dir, exist_ok=True)

    logger = Logger(
        os.path.join(args.save_dir, args.log_file_name),
        getattr(args, 'logger_name', 'block_amend')).get_log()
    logger.info(
        'Model identity: model_variant=%s restoration_block=%s blocks=%s '
        'heads=%s ffn_expansion=%s lowpass=%s%s'
        % (args.model_variant, args.restoration_block, args.restoration_blocks,
           args.restoration_heads, args.restoration_ffn_expansion,
           args.lowpass_kernel, args.lowpass_kernel_size))

    # Seed only for fresh/preflight before any RNG capture; resume restores later.
    if args.mode in ('train', 'preflight', 'eval') and int(args.seed) >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available() and not args.cpu:
            torch.cuda.manual_seed_all(args.seed)

    if args.mode == 'preflight':
        logger.info('Preflight only — no training')
        logger.info('Coverage mode: %s' % (
            'explicit partial coverage' if args.allow_partial_coverage
            else 'strict coverage'))
        logger.info('Python %s  PyTorch %s' % (sys.version.split()[0], torch.__version__))
        if args.allow_partial_coverage:
            cov = write_coverage_manifests(
                args.dataset_dir, os.path.join(args.save_dir, 'coverage'),
                subset='real', ref_subdir=args.train_ref_subdir,
                allow_partial=True)
            logger.info(json.dumps(cov, indent=2))
        else:
            ok, missing = collect_triplets(
                args.dataset_dir, 'Train', subset='real',
                ref_subdir=args.train_ref_subdir, require_all=True)
            logger.info('Train triplets ok=%d missing_ref=%d' % (len(ok), len(missing)))
        test_ok, test_miss = collect_triplets(
            args.dataset_dir, 'Test', subset='real',
            ref_subdir=args.test_ref_subdir, require_all=False)
        logger.info('Test triplets available=%d missing_ref=%d'
                    % (len(test_ok), len(test_miss)))
        model = build_model(args, device)
        apply_init_mode(model, args, device=device, logger=logger)
        log_param_groups(model, logger)
        logger.info('Preflight OK')
        return

    if args.mode == 'eval':
        # Identity is checked before the model is built and before any
        # forward, metric file or image export.
        if args.eval_split == 'test':
            items, _ = collect_triplets(
                args.dataset_dir, 'Test', subset='real',
                ref_subdir=args.test_ref_subdir, require_all=True)
            items = materialize_items(items, args.dataset_dir)
            # Official Test has its own identity. It is not required to match
            # the checkpoint's internal val, and training never opens it.
            identity = {
                'role': 'official_test',
                'train_hash': None,
                'val_hash': None,
                'test_hash': manifest_identity_hash(
                    items, args.dataset_dir, 'test', args.test_ref_subdir),
                'ref_subdir': args.test_ref_subdir,
                'reference_source': REFERENCE_SOURCE,
                'manifest_hash_version': MANIFEST_HASH_VERSION,
            }
            eval_items = items
            tag = 'official_test'
        else:
            split_dir = args.split_dir or os.path.join(
                os.path.dirname(args.model_path), 'splits')
            if args.split_dir == '' and not os.path.isdir(split_dir):
                split_dir = os.path.join(args.save_dir, 'splits')
            _train_items, val_items, identity = _load_locked_split(split_dir, args)
            assert_split_identity(identity, raw['split_identity'], 'eval val')
            eval_items = val_items
            tag = 'val'
        model = build_model(args, device)
        load_strict_state(model, args.model_path, device=device, logger=logger,
                          mode='inference', preloaded=raw)
        loader = DataLoader(BlockAmendEvalSet(eval_items), batch_size=1,
                            shuffle=False, num_workers=args.num_workers,
                            persistent_workers=False)
        summary = evaluate_loader(
            model, loader, args, device, save_dir=args.save_dir, tag=tag,
            split_identity=identity)
        summary['checkpoint_path'] = args.model_path
        summary['checkpoint_epoch'] = raw.get('epoch')
        summary['model_variant'] = raw.get('model_variant')
        summary['metric_protocol'] = PROTOCOL_NAME
        out = os.path.join(args.save_dir, 'eval_%s.json' % tag)
        with open(out, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info('Eval %s PSNR=%.4f SSIM=%.4f -> %s'
                    % (tag, summary['psnr_rgb_mean'], summary['ssim_rgb_mean'], out))
        return

    # ---------- train / resume training ----------
    explicit_split = 'split_dir' in explicit and bool(cli.split_dir)
    (train_loader, val_loader, train_set, val_set,
     train_items, val_items, split_dir, identity,
     train_gen, val_gen) = prepare_data(
        args, logger, mode=args.mode, explicit_split_dir=explicit_split)

    if args.mode == 'resume':
        # ``raw`` is the pack already read on CPU and validated above.
        assert_split_identity(identity, raw['split_identity'], 'resume')
        run_id = raw.get('run_id') or str(uuid.uuid4())
        model = build_model(args, device)
        load_strict_state(model, args.resume_checkpoint, device=device,
                          logger=logger, mode='resume', preloaded=raw)
        opt, sched = build_optimizer(model, args)
        # Adam.load_state_dict places momentum on the parameter device and
        # leaves ``step`` where this PyTorch stored it. Do not .cuda() the
        # whole optimizer state.
        opt.load_state_dict(raw['optimizer_state_dict'])
        sched.load_state_dict(raw['scheduler_state_dict'])
        # RNG last, after every construction that consumes randomness, and
        # before train_loop creates the next iterator.
        restore_rng_state(raw['rng'], train_gen, val_gen)
        start_epoch = int(raw['epoch']) + 1
        global_step = int(raw['global_step'])
        best_psnr = float(raw['best_val_psnr'])
        best_epoch = int(raw.get('best_epoch', 0))
        _assert_best_identity_safe(args.save_dir, raw, logger)
        append_resume_record(args.save_dir, {
            'resumed_at': datetime.utcnow().isoformat() + 'Z',
            'from_checkpoint': args.resume_checkpoint,
            'next_epoch': start_epoch,
            'best_val_psnr': best_psnr,
            'best_epoch': best_epoch,
            'explicit_overrides': applied,
            'split_identity': identity,
        })
        logger.info(
            'Resumed schema=%s epoch=%d -> next=%d step=%d best=%.4f@%d overrides=%s'
            % (raw['checkpoint_schema_version'], raw['epoch'], start_epoch,
               global_step, best_psnr, best_epoch, applied))
    else:
        if args.init_scope == 'lte_only' and not args.init_checkpoint:
            raise SystemExit('lte_only requires --init_checkpoint')
        if args.init_scope == 'fresh' and args.init_checkpoint:
            raise SystemExit(
                'fresh init must not set --init_checkpoint (got %s)'
                % args.init_checkpoint)
        run_id = str(uuid.uuid4())
        model = build_model(args, device)
        apply_init_mode(model, args, device=device, logger=logger)
        opt, sched = build_optimizer(model, args)
        save_run_meta_fresh(args, root, identity, model, run_id,
                            extra={'split_dir': split_dir, 'mode': 'train'})
        start_epoch, global_step, best_psnr, best_epoch = 1, 0, -1.0, 0

    train_loop(
        args, logger, model, opt, sched, train_loader, val_loader,
        train_set, train_gen, val_gen, identity, run_id,
        start_epoch=start_epoch, global_step=global_step,
        best_psnr=best_psnr, best_epoch=best_epoch)


if __name__ == '__main__':
    main()
