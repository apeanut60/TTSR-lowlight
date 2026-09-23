from option import args
from utils import mkExpDir
from dataset import dataloader
from model import TTSR, TTSREnhance
from loss.loss import get_loss_dict
from loss.loss_enhance import get_loss_dict_enhance
from trainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### make save_dir
    _logger = mkExpDir(args)

    ### reproducibility (needed for fair A/B ablations)
    if getattr(args, 'seed', -1) >= 0:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        _logger.info('Random seed: %d (init + data order reproducible; '
                     'cudnn.benchmark=False)' % args.seed)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None
    # Extra eval loaders (generated-reference test sets) are injected for
    # training AND for --eval runs, so a standalone evaluation of a checkpoint
    # reports the same full set of reference settings that training validation
    # does. `_dataloader is None` already covers --test, so no test/eval flag is
    # needed here. Each add_* helper is internally guarded by its own option.
    if (_dataloader is not None):
        if (args.dataset.lower() == 'mixed_lolv2_data1'):
            dataloader.add_mixed_eval(args, _dataloader)
            if getattr(args, 'eval_lolv2real_gt', False):
                dataloader.add_lolv2real_gt_eval(args, _dataloader)
        elif (args.dataset.lower() == 'mixed_data1_nanobanana_lolv2'):
            dataloader.add_mixed_data1_nanobanana_eval(args, _dataloader)
        elif (args.dataset.lower() == 'data1_nanobanana'):
            dataloader.add_data1_eval(args, _dataloader)
        elif (args.dataset.lower() == 'lolv2_nanobanana'):
            dataloader.add_data1_eval(args, _dataloader)
            dataloader.add_lolv2_nanobanana_eval(args, _dataloader)
        elif (args.dataset.lower() == 'lolv2_nanobanana_mixed'):
            dataloader.add_data1_eval(args, _dataloader)
            dataloader.add_lolv2_nanobanana_eval(args, _dataloader)
        elif (args.dataset.lower() == 'mixed_data1_lolv2_nanobanana'):
            dataloader.add_mixed_data1_lolv2_nanobanana_eval(args, _dataloader)
        elif (args.dataset.lower() in ('lolv2real', 'lolv2syn')):
            if getattr(args, 'eval_lolv2_nanobanana', False):
                subset = 'real' if args.dataset.lower() == 'lolv2real' else 'syn'
                dataloader.add_lolv2_nanobanana_eval_subset(
                    args, _dataloader, subset)
        else:
            dataloader.add_data1_eval(args, _dataloader)
            if getattr(args, 'eval_lol_nanobanana', False):
                dataloader.add_lol_nanobanana_eval(args, _dataloader)

    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    is_enhance = getattr(args, 'enhance_mode', False)

    if is_enhance:
        _model = TTSREnhance.TTSREnhance(args).to(device)
        _logger.info('Using TTSREnhance model (1:1 low-light enhancement mode)')
        # Optionally load pre-trained TTSR weights for transfer learning
        if args.load_pretrain and args.pretrain_path:
            _logger.info('Loading pre-trained weights from: ' + args.pretrain_path)
            _model = TTSREnhance.load_pretrained_weights(
                _model, args.pretrain_path, device)
        if getattr(args, 'freeze_stages', ''):
            _model, frozen_names = Trainer.freeze_mainnet_stages(
                _model, args.freeze_stages, args.num_gpu)
            _logger.info('Frozen MainNet stages: %s; frozen parameter count: %d'
                         % (args.freeze_stages, len(frozen_names)))
        if getattr(args, 'adapter_only', False):
            # Freeze everything, then re-enable ONLY the H/4 texture adapter.
            # The Trainer builds its param groups afterwards, so the optimizer
            # ends up holding nothing but the adapter.
            base = getattr(args, 'adapter_base_ckpt', '')
            if (not base) or (not os.path.isfile(base)):
                raise SystemExit(
                    '--adapter_only requires an existing --adapter_base_ckpt '
                    '(got %r)' % base)
            _logger.info('adapter-only base checkpoint: ' + base)
            _model = TTSREnhance.load_pretrained_weights(_model, base, device)
            _net = _model.module if hasattr(_model, 'module') else _model
            _net.requires_grad_(False)
            _adapter = _net.MainNet.denoiser.ref_adapter
            _adapter.requires_grad_(True)
            _trainable = [n for n, p in _net.named_parameters() if p.requires_grad]
            if (not _trainable) or any('ref_adapter' not in n for n in _trainable):
                raise SystemExit('adapter-only: unexpected trainable set %s'
                                 % _trainable[:8])
            _logger.info('adapter-only: %d trainable tensors, all under '
                         'MainNet.denoiser.ref_adapter (%d params)'
                         % (len(_trainable),
                            sum(p.numel() for p in _adapter.parameters())))
        # Save the freshly constructed weights so a later ablation arm can
        # share a *verified* initialisation instead of relying on the seed
        # reproducing the same construction order. Load it with
        # `--load_pretrain True --pretrain_path <dir>/model/init.pt`.
        if (not args.test) and (not args.eval):
            _init_path = os.path.join(args.save_dir, 'model', 'init.pt')
            if not os.path.exists(_init_path):
                os.makedirs(os.path.dirname(_init_path), exist_ok=True)
                torch.save(_model.state_dict(), _init_path)
                _logger.info('Saved initial weights to ' + _init_path)
    else:
        _model = TTSR.TTSR(args).to(device)
        _logger.info('Using TTSR model (4x super-resolution mode)')

    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    if is_enhance:
        _loss_all = get_loss_dict_enhance(args, _logger)
    else:
        _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate()
    elif getattr(args, 'adapter_only', False):
        t.train_adapter_only()
    else:
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        last_completed_epoch = 0
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            last_completed_epoch = epoch
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
                if getattr(t, 'data1_psnr_floor_violated', False):
                    _logger.info(
                        'EARLY STOP: data1 PSNR floor violation at epoch %d'
                        % epoch)
                    break

        # Always keep the last completed epoch, even when
        # num_epochs % save_every != 0 (otherwise the tail of every run is lost).
        if last_completed_epoch:
            ckpt = os.path.join(args.save_dir, 'model',
                                'model_%05d.pt' % last_completed_epoch)
            if not os.path.exists(ckpt):
                _logger.info('Epoch %d was not covered by save_every; saving it.'
                             % last_completed_epoch)
                t.save(last_completed_epoch)
