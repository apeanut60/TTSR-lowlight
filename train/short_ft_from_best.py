#!/usr/bin/env python3
"""Short finetune from an internal-val best pack.

Loads model weights only. A new Adam is built at a fixed learning rate.
StepLR is kept so the pack still has a scheduler, but gamma is 1 so the
rate does not change. The locked split of the source run is reused.
Every epoch is written to checkpoint_epoch_XXXX.pt in a new directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from block_amend_ckpt import load_local_checkpoint, require_resume_fields
from block_amend_main import (
    BASELINE_FIXED, build_model, build_optimizer, ensure_save_dir_fresh,
    prepare_data, train_loop,
)
from dataset.block_amend_lolv2 import assert_split_identity
from model.weight_init import load_strict_state
from utils import Logger
from utils_metrics_block_amend import PROTOCOL_NAME


def _find_split_dir(checkpoint):
    cur = os.path.dirname(os.path.abspath(checkpoint))
    for _ in range(5):
        split_dir = os.path.join(cur, 'splits')
        if os.path.isfile(os.path.join(split_dir, 'split_train.json')):
            return split_dir
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise SystemExit('locked split missing above %s' % checkpoint)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--rec_w', type=float, default=None)
    p.add_argument('--ms_ssim_w', type=float, default=None)
    args_cli = p.parse_args()

    raw = load_local_checkpoint(args_cli.checkpoint)
    require_resume_fields(raw, args_cli.checkpoint)
    split_dir = _find_split_dir(args_cli.checkpoint)

    merged = dict(BASELINE_FIXED)
    merged.update(raw['config'])
    merged['lr_rate'] = float(args_cli.lr)
    merged['gamma'] = 1.0
    merged['num_epochs'] = int(args_cli.epochs)
    merged['save_dir'] = args_cli.save_dir
    merged['split_dir'] = split_dir
    merged['mode'] = 'short_ft'
    merged['save_every_epoch'] = True
    merged['init_scope'] = 'short_ft'
    merged['init_checkpoint'] = os.path.abspath(args_cli.checkpoint)
    merged['log_file_name'] = 'block_amend.log'
    merged['logger_name'] = 'block_amend'
    merged['cpu'] = False
    merged['save_images'] = False
    merged['metric_protocol'] = PROTOCOL_NAME
    if args_cli.rec_w is not None:
        merged['rec_w'] = float(args_cli.rec_w)
    if args_cli.ms_ssim_w is not None:
        merged['ms_ssim_w'] = float(args_cli.ms_ssim_w)
    args = SimpleNamespace(**merged)

    ensure_save_dir_fresh(args.save_dir)
    logger = Logger(
        os.path.join(args.save_dir, args.log_file_name),
        args.logger_name).get_log()
    logger.info(
        'short finetune source=%s parent_epoch=%s lr=%s epochs=%s '
        'gamma=1 rec_w=%s ms_ssim_w=%s '
        '(fixed lr, new Adam, L1 + MS-SSIM, locked split %s)'
        % (args.init_checkpoint, raw.get('epoch'), args.lr_rate,
           args.num_epochs, args.rec_w, args.ms_ssim_w, split_dir))

    device = torch.device('cuda')
    model = build_model(args, device)
    load_strict_state(
        model, args.init_checkpoint, device=device, logger=logger,
        mode='short_ft', preloaded=raw)
    opt, sched = build_optimizer(model, args)
    if abs(float(opt.param_groups[0]['lr']) - float(args.lr_rate)) > 1e-12:
        raise RuntimeError('optimizer lr is not the requested fixed lr')

    (train_loader, val_loader, train_set, _val_set,
     _train_items, _val_items, _split_dir, identity,
     train_gen, val_gen) = prepare_data(
        args, logger, mode='resume', explicit_split_dir=True)
    assert_split_identity(identity, raw['split_identity'], 'short_ft')

    meta = {
        'source_checkpoint': args.init_checkpoint,
        'source_epoch': int(raw['epoch']),
        'source_best_val_psnr': float(raw['best_val_psnr']),
        'lr': float(args.lr_rate),
        'epochs': int(args.num_epochs),
        'gamma': 1.0,
        'rec_w': float(args.rec_w),
        'ms_ssim_w': float(args.ms_ssim_w),
        'loss': 'rec_w * L1 + ms_ssim_w * (1 - MS-SSIM)',
        'optimizer': 'new Adam, source moments not restored',
        'split_dir': split_dir,
        'dataset_dir': args.dataset_dir,
        'train_ref_subdir': args.train_ref_subdir,
        'test_ref_subdir': args.test_ref_subdir,
        'note': 'Test images are not in this training set.',
    }
    with open(os.path.join(args.save_dir, 'short_ft_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    train_loop(
        args, logger, model, opt, sched, train_loader, val_loader,
        train_set, train_gen, val_gen, identity, str(uuid.uuid4()),
        start_epoch=1, global_step=0, best_psnr=-1.0, best_epoch=0)
    logger.info('short finetune finished')


if __name__ == '__main__':
    main()
