#!/usr/bin/env python3
"""Block-amend CSFI-U-Net baseline entry (fresh / resume / eval / preflight).

Does not register historical extra_test loaders or Official Test during training.
GT never enters the reference path. See docs/BLOCK_AMEND_TRAINING_BASELINE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.block_amend_lolv2 import (
    BlockAmendEvalSet, BlockAmendTrainSet, collect_triplets,
    load_manifest_items, manifest_hash, save_split_manifests, split_train_val,
    write_coverage_manifests,
)
from model.TTSREnhance import TTSREnhance
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
    igab_blocks='1,1',
    igab_heads=1,
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
                   help='For eval: checkpoint pack or bare weights')
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
    return p


def make_args(cli) -> SimpleNamespace:
    d = dict(BASELINE_FIXED)
    d.update(vars(cli))
    d['dataset'] = 'block_amend_lolv2'
    d['logger_name'] = 'block_amend'
    return SimpleNamespace(**d)


def ensure_save_dir_fresh(save_dir: str):
    if os.path.isdir(save_dir) and os.listdir(save_dir):
        raise SystemExit(
            'save_dir is non-empty (%s). Fresh training refuses to reuse it. '
            'Use --mode resume with --resume_checkpoint, or choose a new RUN_NAME.'
            % save_dir)
    os.makedirs(save_dir, exist_ok=True)


def atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.tmp_ckpt_', dir=os.path.dirname(path) or '.')
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def build_model(args, device):
    model = TTSREnhance(args).to(device)
    # freeze_lte already applied in TTSREnhance via LTE(requires_grad=False)
    if args.freeze_stages:
        from trainer import Trainer
        model, names = Trainer.freeze_mainnet_stages(
            model, args.freeze_stages, args.num_gpu)
        print('[Freeze] stages=%s n_params=%d' % (args.freeze_stages, len(names)))
    return model


def build_optimizer(model, args):
    wrap = model.module if hasattr(model, 'module') else model
    groups = []
    main_params = [p for p in wrap.MainNet.parameters() if p.requires_grad]
    # Skip disabled photometric heads from "active" messaging; they may still
    # exist with requires_grad True but use_*=False means unused in forward.
    if not args.use_global_illum:
        for n, p in wrap.MainNet.named_parameters():
            if n.startswith('global_illum_head.'):
                p.requires_grad = False
    if not args.use_ref_illum:
        for n, p in wrap.MainNet.named_parameters():
            if n.startswith('ref_illum.'):
                p.requires_grad = False
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
    # Dedup check
    ids = []
    for g in groups:
        for p in g['params']:
            if id(p) in ids:
                raise RuntimeError('Duplicate parameter in optimizer groups')
            ids.append(id(p))
    opt = optim.Adam(groups, betas=(args.beta1, args.beta2), eps=args.eps)
    sched = optim.lr_scheduler.StepLR(opt, step_size=args.decay, gamma=args.gamma)
    return opt, sched


def prepare_data(args, logger):
    split_dir = args.split_dir or os.path.join(args.save_dir, 'splits')
    if args.mode == 'preflight' or not (
            os.path.isfile(os.path.join(split_dir, 'split_train.json'))
            and os.path.isfile(os.path.join(split_dir, 'split_val.json'))):
        if args.allow_partial_coverage:
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
        meta = save_split_manifests(train_items, val_items, split_dir)
        logger.info('Created split: %s' % json.dumps(meta))
    else:
        train_items = load_manifest_items(
            os.path.join(split_dir, 'split_train.json'))
        val_items = load_manifest_items(
            os.path.join(split_dir, 'split_val.json'))
        # Validate manifested files still exist
        for it in train_items + val_items:
            for k in ('low', 'normal', 'ref'):
                if not os.path.isfile(it[k]):
                    raise FileNotFoundError(
                        'Manifested file missing: id=%s %s=%s'
                        % (it['id'], k, it[k]))
        logger.info(
            'Using existing split train=%d val=%d hashes=%s/%s'
            % (len(train_items), len(val_items),
               manifest_hash(train_items), manifest_hash(val_items)))

    train_set = BlockAmendTrainSet(
        train_items, crop_size=args.train_crop_size,
        ref_degrade=args.ref_degrade, seed=args.seed)
    val_set = BlockAmendEvalSet(val_items)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, train_items, val_items, split_dir


def evaluate_loader(model, loader, args, device, save_dir=None, tag='val'):
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
    }
    return summary


def save_run_meta(args, root, train_items, val_items, model, extra=None):
    wrap = model.module if hasattr(model, 'module') else model
    n_params = sum(p.numel() for p in wrap.parameters())
    meta = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'git': _git_info(root),
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'n_params': int(n_params),
        'metric_protocol': PROTOCOL_NAME,
        'reference_source': 'generated_nano',
        'train_ref_subdir': args.train_ref_subdir,
        'test_ref_subdir': args.test_ref_subdir,
        'augmentation': 'paired_crop_flip_rot',
        'ref_degrade': bool(args.ref_degrade),
        'init_scope': args.init_scope,
        'init_checkpoint': args.init_checkpoint,
        'config': {k: getattr(args, k) for k in BASELINE_FIXED},
        'train_hash': manifest_hash(train_items),
        'val_hash': manifest_hash(val_items),
        'train_count': len(train_items),
        'val_count': len(val_items),
    }
    if extra:
        meta.update(extra)
    path = os.path.join(args.save_dir, 'run_meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def train_loop(args, logger, model, opt, sched, train_loader, val_loader,
               start_epoch=1, global_step=0, best_psnr=-1.0):
    device = next(model.parameters()).device
    # Minimal loss dict (reuse enhance losses; only rec + ms_ssim active)
    loss_all = get_loss_dict_enhance(args, logger)
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        logger.info('Epoch %d lr=%e' % (epoch, opt.param_groups[0]['lr']))
        for i_batch, batch in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            lr = batch['LR'].to(device)
            lrsr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            refsr = batch['Ref_sr'].to(device)
            sr, *_ = model(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
            loss = args.rec_w * loss_all['rec_loss'](sr, hr)
            if 'ms_ssim_loss' in loss_all and args.ms_ssim_w > 0:
                loss = loss + args.ms_ssim_w * loss_all['ms_ssim_loss'](sr, hr)
            if (not torch.isfinite(loss).all()) or (not torch.isfinite(sr).all()):
                raise RuntimeError(
                    'Non-finite at epoch=%d step=%d id=%s loss=%s'
                    % (epoch, i_batch + 1, batch.get('id', '?'), loss.detach()))
            loss.backward()
            opt.step()
            global_step += 1
            if (i_batch + 1) % args.print_every == 0:
                logger.info('epoch %d batch %d loss %.6f'
                            % (epoch, i_batch + 1, float(loss)))

        # Scheduler after optimizer steps for this epoch.
        sched.step()

        # last checkpoint (atomic)
        pack = {
            'model_state_dict': {
                k.replace('module.', ''): v
                for k, v in model.state_dict().items() if '_copy' not in k
            },
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_val_psnr': best_psnr,
            'config': {k: getattr(args, k) for k in BASELINE_FIXED},
            'init_scope': args.init_scope,
            'metric_protocol': PROTOCOL_NAME,
            'rng': {
                'torch': torch.get_rng_state(),
                'cuda': (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
        }
        atomic_torch_save(pack, os.path.join(args.save_dir, 'checkpoint_last.pt'))

        if epoch % args.val_every == 0:
            summary = evaluate_loader(
                model, val_loader, args, device,
                save_dir=args.save_dir, tag='val_ep%05d' % epoch)
            with open(os.path.join(
                    args.save_dir, 'val_metrics_ep%05d.json' % epoch), 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(
                'val epoch %d PSNR_rgb %.4f SSIM %.4f (%s)'
                % (epoch, summary['psnr_rgb_mean'], summary['ssim_rgb_mean'],
                   PROTOCOL_NAME))
            if summary['psnr_rgb_mean'] > best_psnr:
                best_psnr = summary['psnr_rgb_mean']
                pack['best_val_psnr'] = best_psnr
                atomic_torch_save(
                    pack, os.path.join(args.save_dir, 'checkpoint_best_val_psnr.pt'))
                logger.info('New best_val_psnr=%.4f' % best_psnr)


def main():
    cli = build_parser().parse_args()
    args = make_args(cli)
    root = os.path.dirname(os.path.abspath(__file__))
    if args.run_name:
        args.save_dir = os.path.join(args.save_dir, args.run_name)

    if args.mode == 'train':
        ensure_save_dir_fresh(args.save_dir)
    else:
        os.makedirs(args.save_dir, exist_ok=True)

    logger = Logger(args.log_file_name, args.logger_name).get_log()
    # Redirect log file into save_dir
    log_path = os.path.join(args.save_dir, args.log_file_name)
    fh = __import__('logging').FileHandler(log_path)
    logger.addHandler(fh)

    device = torch.device('cpu' if args.cpu else 'cuda')
    if args.seed >= 0:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.mode == 'preflight':
        logger.info('Preflight only — no training')
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
        # Also check Official Test refs exist (do not evaluate)
        test_ok, test_miss = collect_triplets(
            args.dataset_dir, 'Test', subset='real',
            ref_subdir=args.test_ref_subdir, require_all=False)
        logger.info('Test triplets available=%d missing_ref=%d'
                    % (len(test_ok), len(test_miss)))
        # Build model once to verify init
        model = build_model(args, device)
        apply_init_mode(model, args, device=device, logger=logger)
        log_param_groups(model, logger)
        logger.info('Preflight OK')
        return

    if args.mode == 'eval':
        if not args.model_path:
            raise SystemExit('eval requires --model_path')
        model = build_model(args, device)
        load_strict_state(model, args.model_path, device=device, logger=logger,
                          mode='inference')
        if args.eval_split == 'test':
            items, _ = collect_triplets(
                args.dataset_dir, 'Test', subset='real',
                ref_subdir=args.test_ref_subdir, require_all=True)
            loader = DataLoader(BlockAmendEvalSet(items), batch_size=1,
                                shuffle=False, num_workers=args.num_workers)
            tag = 'official_test'
        else:
            split_dir = args.split_dir or os.path.join(args.save_dir, 'splits')
            items = load_manifest_items(os.path.join(split_dir, 'split_val.json'))
            loader = DataLoader(BlockAmendEvalSet(items), batch_size=1,
                                shuffle=False, num_workers=args.num_workers)
            tag = 'val'
        summary = evaluate_loader(
            model, loader, args, device, save_dir=args.save_dir, tag=tag)
        out = os.path.join(args.save_dir, 'eval_%s.json' % tag)
        with open(out, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info('Eval %s PSNR=%.4f SSIM=%.4f -> %s'
                    % (tag, summary['psnr_rgb_mean'], summary['ssim_rgb_mean'], out))
        return

    # train / resume
    train_loader, val_loader, train_items, val_items, split_dir = prepare_data(
        args, logger)
    model = build_model(args, device)

    start_epoch, global_step, best_psnr = 1, 0, -1.0
    if args.mode == 'resume':
        if not args.resume_checkpoint:
            raise SystemExit('resume requires --resume_checkpoint')
        raw = torch.load(args.resume_checkpoint, map_location=device)
        if not isinstance(raw, dict) or 'model_state_dict' not in raw:
            raise SystemExit(
                'Resume checkpoint must be a training pack with model_state_dict')
        # Config check
        cfg = raw.get('config') or {}
        for k, v in BASELINE_FIXED.items():
            if k in cfg and cfg[k] != v and k in (
                    'backbone', 'num_res_blocks', 'n_feats', 'enable_lightup'):
                raise SystemExit(
                    'Resume config mismatch for %s: ckpt=%r expected=%r' % (k, cfg[k], v))
        load_strict_state(model, args.resume_checkpoint, device=device,
                          logger=logger, mode='resume')
        opt, sched = build_optimizer(model, args)
        if 'optimizer_state_dict' in raw:
            opt.load_state_dict(raw['optimizer_state_dict'])
        if 'scheduler_state_dict' in raw:
            sched.load_state_dict(raw['scheduler_state_dict'])
        start_epoch = int(raw.get('epoch', 0)) + 1
        global_step = int(raw.get('global_step', 0))
        best_psnr = float(raw.get('best_val_psnr', -1.0))
        logger.info('Resumed from epoch %d global_step %d best=%.4f'
                    % (start_epoch - 1, global_step, best_psnr))
    else:
        # fresh
        if args.init_scope == 'lte_only' and not args.init_checkpoint:
            raise SystemExit('lte_only requires --init_checkpoint')
        if args.init_scope == 'fresh' and args.init_checkpoint:
            raise SystemExit(
                'fresh init must not set --init_checkpoint (got %s)'
                % args.init_checkpoint)
        apply_init_mode(model, args, device=device, logger=logger)
        opt, sched = build_optimizer(model, args)

    save_run_meta(args, root, train_items, val_items, model,
                  extra={'split_dir': split_dir, 'mode': args.mode})
    train_loop(args, logger, model, opt, sched, train_loader, val_loader,
               start_epoch=start_epoch, global_step=global_step,
               best_psnr=best_psnr)


if __name__ == '__main__':
    main()
