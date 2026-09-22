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
    if (_dataloader is not None and (not args.test) and (not args.eval)):
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
    else:
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
                if getattr(t, 'data1_psnr_floor_violated', False):
                    _logger.info(
                        'EARLY STOP: data1 PSNR floor violation at epoch %d'
                        % epoch)
                    break
