from utils import calc_psnr_and_ssim, mean_align, chroma_gain
from model import Vgg19

import os
import math
import numpy as np
from imageio import imread, imsave
from PIL import Image

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
import lpips


class Trainer():
    @staticmethod
    def freeze_mainnet_stages(model, freeze_stages, num_gpu=1):
        """Freeze named MainNet stages by setting requires_grad=False.

        Returns (model, frozen_param_names).
        """
        if (not freeze_stages):
            return model, []

        mainnet = model.module.MainNet if hasattr(model, 'module') else model.MainNet
        stages = [s.strip().lower() for s in freeze_stages.split(',') if s.strip()]

        stage_prefixes = {
            'sfe': ('SFE',),
            'stage1': ('conv11_', 'RB11'),
            'stage2': ('conv22_', 'ex12.', 'RB21', 'RB22', 'conv21_'),
            'stage3': ('conv33_', 'ex123.', 'RB31', 'RB32', 'RB33',
                       'conv31_', 'conv32_'),
            'merge': ('merge_tail',),
        }

        prefixes = []
        for stage in stages:
            if stage not in stage_prefixes:
                raise ValueError('Unknown freeze_stages value: %s' % stage)
            prefixes.extend(stage_prefixes[stage])

        frozen_names = []
        for name, param in mainnet.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                param.requires_grad = False
                frozen_names.append(name)

        return model, frozen_names

    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)

        wrapped = self.model.module if hasattr(self.model, 'module') else self.model
        mainnet = wrapped.MainNet
        lte = wrapped.LTE
        ref_head = getattr(wrapped, 'RefCorrection', None)
        illum_prefix = 'global_illum_head.'
        refillum_prefix = 'ref_illum.'
        stage2_prefixes = ('conv22_', 'ex12.', 'RB21', 'RB22', 'conv21_')
        stage2_params = []
        illum_params = []
        refillum_params = []
        other_mainnet_params = []
        for name, param in mainnet.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(illum_prefix):
                illum_params.append(param)
            elif name.startswith(refillum_prefix):
                refillum_params.append(param)
            elif any(name.startswith(prefix) for prefix in stage2_prefixes):
                stage2_params.append(param)
            else:
                other_mainnet_params.append(param)

        lr_stage2 = getattr(args, 'lr_rate_stage2', -1)
        if lr_stage2 is None or lr_stage2 <= 0:
            lr_stage2 = args.lr_rate

        lte_params = [p for p in lte.parameters() if p.requires_grad]
        ref_head_params = []
        if ref_head is not None:
            ref_head_params = [p for p in ref_head.parameters() if p.requires_grad]
        lr_refhead = getattr(args, 'lr_rate_refhead', -1)
        if lr_refhead is None or lr_refhead <= 0:
            lr_refhead = args.lr_rate
        lr_illum = getattr(args, 'lr_rate_illum', -1)
        if lr_illum is None or lr_illum <= 0:
            lr_illum = args.lr_rate
        lr_refillum = getattr(args, 'lr_rate_refillum', -1)
        if lr_refillum is None or lr_refillum <= 0:
            lr_refillum = args.lr_rate

        self.params = []
        if other_mainnet_params:
            self.params.append({"params": other_mainnet_params, "lr": args.lr_rate})
        if stage2_params:
            self.params.append({"params": stage2_params, "lr": lr_stage2})
        if ref_head_params:
            self.params.append({"params": ref_head_params, "lr": lr_refhead})
        if illum_params:
            self.params.append({"params": illum_params, "lr": lr_illum})
        if refillum_params:
            self.params.append({"params": refillum_params, "lr": lr_refillum})
        if lte_params:
            self.params.append({"params": lte_params, "lr": args.lr_rate_lte})
        if not self.params:
            raise RuntimeError('No trainable parameters remain after freezing')

        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        sched = getattr(args, 'lr_scheduler', 'step').lower()
        if sched == 'cosine':
            t_max = max(1, int(getattr(args, 'num_epochs', 1)))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max,
                eta_min=float(getattr(args, 'eta_min', 1e-7)))
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_psnr_y = 0.
        self.max_psnr_y_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0
        self.data1_psnr_floor_violated = False
        self.data1_last_psnr = None

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            pretrained = torch.load(model_path, map_location=self.device)
            # This loader is shape-filtered; never let it paint an IGAB-variant
            # checkpoint into the LFCA backbone (model_variant differs).
            _wrap = self.model.module if hasattr(self.model, 'module') else self.model
            if type(getattr(_wrap, 'MainNet', None)).__name__ == 'CSFIUNet':
                from model.restoration_config import reject_legacy_igab_state
                reject_legacy_igab_state(pretrained.keys(), model_path)
            model_state_dict = self.model.state_dict()

            # Load only matching layers (support partial weight loading)
            matched, skipped = 0, 0
            for k, v in pretrained.items():
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    model_state_dict[k] = v
                    matched += 1
                else:
                    skipped += 1

            self.model.load_state_dict(model_state_dict)
            self.logger.info('Loaded %d matching layers (skipped %d incompatible)'
                             % (matched, skipped))

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def _tiled_forward(self, lr, lr_sr, ref, ref_sr, tile_size=None, overlap=None,
                       return_diag=False):
        """Tiled inference via shared ``model.tiled_infer`` (no args rewrite).

        Returns the same tuple shape as a normal forward for caller compatibility.
        ``return_diag`` is accepted but diagnostics are None under tiling.
        """
        from model.tiled_infer import tiled_forward

        tile_size = tile_size or getattr(self.args, 'tile_size', 256)
        overlap = overlap if (overlap is not None) else getattr(
            self.args, 'tile_overlap', 96)

        fwd = {}
        use_qia = getattr(self.args, 'query_illum_align', False)
        alpha = float(getattr(self.args, 'query_illum_alpha', 0.0))
        beta = float(getattr(self.args, 'query_hist_beta', 0.0))
        if use_qia and (alpha > 0.0 or beta > 0.0) and ref is not None:
            from model.query_illum_align import estimate_query_align_maps
            g, lut, _ = estimate_query_align_maps(
                lr_sr, ref,
                alpha=alpha,
                gamma_min=float(getattr(self.args, 'query_gamma_min', 0.3)),
                gamma_max=float(getattr(self.args, 'query_gamma_max', 1.0)),
                bins=int(getattr(self.args, 'query_hist_bins', 256)),
            )
            fwd['query_gamma_override'] = g
            if beta > 0.0:
                fwd['query_hist_lut_override'] = lut
        for k in ('stage_mode_lv3', 'stage_mode_lv2', 'stage_mode_lv1'):
            if getattr(self.args, k, None) is not None:
                fwd[k] = getattr(self.args, k)

        sr = tiled_forward(
            self.model, lr, lr_sr, ref, ref_sr,
            tile_size=int(tile_size), overlap=int(overlap),
            enable_grad=False,
            apply_illum=True, apply_ref_illum=True,
            fwd_extra=fwd,
        )
        if return_diag:
            return sr, None, None, None, None, None, None
        return sr, None, None, None, None

    def _tta_forward(self, one_fn, lr, lr_sr, ref, ref_sr):
        """8-view test-time augmentation: 4 rotations x 2 flips, averaged.

        ``one_fn(lr, lr_sr, ref, ref_sr)`` must return a [1,3,H,W] tensor.
        The per-image illumination head only uses mean/std of the low input,
        which are invariant to flips/rotations, so the views stay consistent.
        """
        acc, n = None, 0
        for k in range(4):
            for flip in (False, True):
                def _t(x):
                    y = torch.rot90(x, k, dims=(2, 3))
                    return torch.flip(y, dims=(3,)).contiguous() if flip else y.contiguous()

                def _inv(y):
                    if flip:
                        y = torch.flip(y, dims=(3,))
                    return torch.rot90(y, -k, dims=(2, 3)).contiguous()

                out = _inv(one_fn(_t(lr), _t(lr_sr), _t(ref), _t(ref_sr)))
                acc = out if acc is None else acc + out
                n += 1
        return acc / n

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        # Epoch-level StepLR: step once *after* the epoch's optimizer updates
        # (see end of this method). Never advance before the first step.
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))

        epoch_rec, epoch_ms, epoch_n = 0.0, 0.0, 0
        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref = sample_batched['Ref']
            ref_sr = sample_batched['Ref_sr']
            has_ref_head = hasattr(self.model, 'RefCorrection')
            if not has_ref_head and hasattr(self.model, 'module'):
                has_ref_head = hasattr(self.model.module, 'RefCorrection')

            # Full-image gamma / hist LUT from the dataset (v1r6/v1r7).
            # Used only when Query illum align is enabled; MainNet sees raw `lr`.
            fwd_extra = {}
            if getattr(self.args, 'query_illum_align', False):
                alpha = float(getattr(self.args, 'query_illum_alpha', 0.0))
                beta = float(getattr(self.args, 'query_hist_beta', 0.0))
                if (alpha > 0.0 or beta > 0.0) and 'query_gamma' in sample_batched:
                    fwd_extra['query_gamma_override'] = sample_batched[
                        'query_gamma'].view(-1)
                if beta > 0.0 and 'query_hist_lut' in sample_batched:
                    lut = sample_batched['query_hist_lut']
                    if lut.ndim == 1:
                        lut = lut.view(1, -1)
                    fwd_extra['query_hist_lut_override'] = lut

            if has_ref_head:
                model_out = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr,
                                       return_ref=True, **fwd_extra)
                sr, S, T_lv3, T_lv2, T_lv1, ref_corrected = model_out
            else:
                sr, S, T_lv3, T_lv2, T_lv1 = self.model(
                    lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr, **fwd_extra)
                ref_corrected = None

            ### calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss = rec_loss
            epoch_rec += float(rec_loss.item())
            epoch_n += 1
            if (is_print):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()) )

            if (not is_init):
                if ('ms_ssim_loss' in self.loss_all):
                    ms_ssim_loss = (getattr(self.args, 'ms_ssim_w', 0.0)
                                    * self.loss_all['ms_ssim_loss'](sr, hr))
                    loss += ms_ssim_loss
                    epoch_ms += float(ms_ssim_loss.item())
                    if (is_print):
                        self.logger.info('ms_ssim_loss: %.10f'
                                         % (ms_ssim_loss.item()))
                if ('per_loss' in self.loss_all):
                    sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                    with torch.no_grad():
                        hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                    per_loss = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                    loss += per_loss
                    if (is_print):
                        self.logger.info( 'per_loss: %.10f' %(per_loss.item()) )
                if ('tpl_loss' in self.loss_all):
                    sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1, 
                        S, T_lv3, T_lv2, T_lv1)
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info( 'tpl_loss: %.10f' %(tpl_loss.item()) )
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr)
                    loss += adv_loss
                    if (is_print):
                        self.logger.info( 'adv_loss: %.10f' %(adv_loss.item()) )

                ### Low-light specific losses
                if ('illum_smooth_loss' in self.loss_all):
                    illum_loss = self.args.illum_smooth_w * self.loss_all['illum_smooth_loss'](sr)
                    loss += illum_loss
                    if (is_print):
                        self.logger.info( 'illum_smooth_loss: %.10f' %(illum_loss.item()) )
                if ('color_loss' in self.loss_all):
                    color_loss = self.args.color_w * self.loss_all['color_loss'](sr)
                    loss += color_loss
                    if (is_print):
                        self.logger.info( 'color_loss: %.10f' %(color_loss.item()) )
                if ('exposure_loss' in self.loss_all):
                    exposure_loss = self.args.exposure_w * self.loss_all['exposure_loss'](sr)
                    loss += exposure_loss
                    if (is_print):
                        self.logger.info( 'exposure_loss: %.10f' %(exposure_loss.item()) )

                if ('illum_match_loss' in self.loss_all):
                    illum_match = self.args.illum_match_w * self.loss_all['illum_match_loss'](sr, hr)
                    loss += illum_match
                    if (is_print):
                        self.logger.info( 'illum_match_loss: %.10f' %(illum_match.item()) )

                ref_correct_w = getattr(self.args, 'ref_correct_w', 0.0)
                if ref_correct_w > 0 and ref_corrected is not None:
                    ref_correct_loss = ref_correct_w * F.l1_loss(ref_corrected, hr)
                    loss += ref_correct_loss
                    if (is_print):
                        self.logger.info('ref_correct_loss: %.10f'
                                         % (ref_correct_loss.item()))

            if not torch.isfinite(loss).all() or not torch.isfinite(sr).all():
                sid = sample_batched.get('id', None)
                raise RuntimeError(
                    'Non-finite loss/output at epoch=%s step=%s sample_id=%s '
                    'loss=%s sr_finite=%s'
                    % (current_epoch, i_batch + 1, sid,
                       float(loss.detach().cpu()) if loss.numel() == 1 else 'tensor',
                       bool(torch.isfinite(sr).all())))

            loss.backward()
            self.optimizer.step()

        if epoch_n > 0:
            self.logger.info(
                'epoch %d train_avg  rec_loss: %.6f  ms_ssim_loss: %.6f  (n=%d)'
                % (current_epoch, epoch_rec / epoch_n, epoch_ms / epoch_n, epoch_n))

        # Epoch-level LR schedule: exactly one step after optimizer updates.
        if not is_init:
            self.scheduler.step()

        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_dir = os.path.join(self.args.save_dir, 'model')
            os.makedirs(model_dir, exist_ok=True)
            model_name = os.path.join(model_dir,
                'model_' + str(current_epoch).zfill(5) + '.pt')
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        if (self.args.dataset == 'CUFED'):
            self.model.eval()
            with torch.no_grad():
                psnr, psnr_rgb, ssim, mse, lpips, cnt = 0., 0., 0., 0., 0., 0
                ap_psnr, ap_psnr_rgb, ap_ssim, ap_mse, ap_lpips = 0., 0., 0., 0., 0.
                do_align = getattr(self.args, 'eval_mean_align', True)
                cg = getattr(self.args, 'eval_chroma_gain', 1.0)
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']

                    if getattr(self.args, 'eval_tta', False):
                        sr = self._tta_forward(
                            lambda a, b, c, d: self.model(lr=a, lrsr=b, ref=c, refsr=d)[0],
                            lr, lr_sr, ref, ref_sr)
                    else:
                        sr, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                    sr = chroma_gain(sr, cg)
                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'.png'), sr_save)
                    
                    _psnr, _ssim, _mse, _psnr_rgb = calc_psnr_and_ssim(sr.detach(), hr.detach())
                    _lpips = self.lpips_fn(sr, hr).item()
                    psnr += _psnr; ssim += _ssim; mse += _mse; lpips += _lpips
                    psnr_rgb += _psnr_rgb
                    if do_align:
                        sr_al = mean_align(sr, hr)
                        a_p, a_s, a_m, a_prgb = calc_psnr_and_ssim(sr_al.detach(), hr.detach())
                        ap_psnr += a_p; ap_ssim += a_s; ap_mse += a_m; ap_psnr_rgb += a_prgb
                        ap_lpips += self.lpips_fn(sr_al, hr).item()

                psnr_ave, ssim_ave = psnr / cnt, ssim / cnt
                mse_ave, lpips_ave = mse / cnt, lpips / cnt
                self.logger.info(
                    '%s  PSNR: %.3f  SSIM: %.4f  LPIPS: %.4f  MSE: %.2f  (PSNR/SSIM=RGB)'
                    % (self.args.dataset, psnr_rgb / cnt, ssim_ave, lpips_ave, mse_ave))
                if do_align:
                    self.logger.info(
                        '%s  PSNR(mean-aligned): %.3f  SSIM(mean-aligned): %.4f  '
                        'LPIPS(mean-aligned): %.4f  MSE(mean-aligned): %.2f  (RGB)'
                        % (self.args.dataset, ap_psnr_rgb / cnt, ap_ssim / cnt,
                           ap_lpips / cnt, ap_mse / cnt))
                if (psnr_rgb / cnt > self.max_psnr):
                    self.max_psnr = psnr_rgb / cnt
                    self.max_psnr_epoch = current_epoch
                if (psnr_ave > self.max_psnr_y):
                    self.max_psnr_y = psnr_ave
                    self.max_psnr_y_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('%s  PSNR(max): %.3f (%d)  SSIM(max): %.4f (%d)'
                    %(self.args.dataset, self.max_psnr, self.max_psnr_epoch,
                      self.max_ssim, self.max_ssim_epoch))

        else:
            # Every non-CUFED dataset evaluates its own test set here, so all
            # datasets report the same metric set (PSNR, PSNRy, SSIM, LPIPS).
            self.model.eval()
            with torch.no_grad():
                psnr, psnr_rgb, ssim, mse, lpips, cnt = 0., 0., 0., 0., 0., 0
                ap_psnr, ap_psnr_rgb, ap_ssim, ap_mse, ap_lpips = 0., 0., 0., 0., 0.
                do_align = getattr(self.args, 'eval_mean_align', True)
                cg = getattr(self.args, 'eval_chroma_gain', 1.0)
                per_image = []
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']

                    if getattr(self.args, 'eval_tta', False):
                        sr = self._tta_forward(
                            lambda a, b, c, d: self._tiled_forward(
                                lr=a, lr_sr=b, ref=c, ref_sr=d)[0],
                            lr, lr_sr, ref, ref_sr)
                    else:
                        sr, _, _, _, _ = self._tiled_forward(
                            lr=lr, lr_sr=lr_sr, ref=ref, ref_sr=ref_sr)
                    sr = chroma_gain(sr, cg)
                    if (self.args.eval_save_results):
                        sr_save = (sr + 1.) * 127.5
                        sr_save = np.transpose(
                            sr_save.squeeze().round().cpu().numpy(),
                            (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results',
                                            str(i_batch).zfill(5) + '.png'),
                               sr_save)
                        lr_save = (lr + 1.) * 127.5
                        lr_save = np.transpose(
                            lr_save.squeeze().round().cpu().numpy(),
                            (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results',
                                            str(i_batch).zfill(5) + '_input.png'),
                               lr_save)

                    _psnr, _ssim, _mse, _psnr_rgb = calc_psnr_and_ssim(sr.detach(), hr.detach())
                    _lpips = self.lpips_fn(sr, hr).item()
                    psnr += _psnr
                    ssim += _ssim
                    mse += _mse
                    lpips += _lpips
                    psnr_rgb += _psnr_rgb
                    per_image.append({
                        'idx': i_batch,
                        'psnr_rgb': float(_psnr_rgb),
                        'ssim_rgb': float(_ssim),
                    })
                    if do_align:
                        sr_al = mean_align(sr, hr)
                        a_p, a_s, a_m, a_prgb = calc_psnr_and_ssim(sr_al.detach(), hr.detach())
                        ap_psnr += a_p
                        ap_ssim += a_s
                        ap_mse += a_m
                        ap_psnr_rgb += a_prgb
                        ap_lpips += self.lpips_fn(sr_al, hr).item()

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                mse_ave = mse / cnt
                lpips_ave = lpips / cnt
                self.logger.info(
                    '%s  PSNR: %.3f  SSIM: %.4f  LPIPS: %.4f  MSE: %.2f  (PSNR/SSIM=RGB)'
                    % (self.args.dataset, psnr_rgb / cnt, ssim_ave, lpips_ave, mse_ave))
                if do_align:
                    self.logger.info(
                        '%s  PSNR(mean-aligned): %.3f  SSIM(mean-aligned): %.4f  '
                        'LPIPS(mean-aligned): %.4f  MSE(mean-aligned): %.2f  (RGB)'
                        % (self.args.dataset, ap_psnr_rgb / cnt, ap_ssim / cnt,
                           ap_lpips / cnt, ap_mse / cnt))
                if getattr(self.args, 'eval_per_image', False):
                    self._save_per_image_eval(
                        self.args.dataset, current_epoch, per_image,
                        float(psnr_rgb / cnt), float(ssim_ave))
                if psnr_rgb / cnt > self.max_psnr:
                    self.max_psnr = psnr_rgb / cnt
                    self.max_psnr_epoch = current_epoch
                if psnr_ave > self.max_psnr_y:
                    self.max_psnr_y = psnr_ave
                    self.max_psnr_y_epoch = current_epoch
                if ssim_ave > self.max_ssim:
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('%s  PSNR(max): %.3f (%d)  SSIM(max): %.4f (%d)'
                                 % (self.args.dataset, self.max_psnr, self.max_psnr_epoch,
                                    self.max_ssim, self.max_ssim_epoch))

        self._evaluate_extra_tests(current_epoch)
        self.logger.info('Evaluation over.')

    def _save_per_image_eval(self, name, epoch, per_image, psnr_mean, ssim_mean):
        import json
        out_dir = os.path.join(self.args.save_dir, 'per_image_eval')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, '%s_e%05d.json' % (name, int(epoch)))
        with open(path, 'w') as f:
            json.dump({
                'name': name,
                'epoch': int(epoch),
                'psnr_rgb_mean': psnr_mean,
                'ssim_rgb_mean': ssim_mean,
                'per_image': per_image,
            }, f, indent=2)
        self.logger.info('Wrote per-image eval: %s' % path)

    def _evaluate_extra_tests(self, current_epoch=0):
        """Evaluate extra validation sets, such as data1 during finetune."""
        if (self.dataloader is None):
            return

        extra_test = self.dataloader.get('extra_test', {})
        if (not extra_test):
            return

        for name, loader in extra_test.items():
            self.model.eval()
            psnr, psnr_rgb, ssim, mse, lpips, cnt = 0., 0., 0., 0., 0., 0
            ap_psnr, ap_psnr_rgb, ap_ssim, ap_mse, ap_lpips = 0., 0., 0., 0., 0.
            do_align = getattr(self.args, 'eval_mean_align', True)
            cg = getattr(self.args, 'eval_chroma_gain', 1.0)
            with torch.no_grad():
                for sample_batched in loader:
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']

                    if getattr(self.args, 'eval_tta', False):
                        sr = self._tta_forward(
                            lambda a, b, c, d: self._tiled_forward(
                                lr=a, lr_sr=b, ref=c, ref_sr=d)[0],
                            lr, lr_sr, ref, ref_sr)
                    else:
                        sr, _, _, _, _ = self._tiled_forward(
                            lr=lr, lr_sr=lr_sr, ref=ref, ref_sr=ref_sr)
                    sr = chroma_gain(sr, cg)
                    _psnr, _ssim, _mse, _psnr_rgb = calc_psnr_and_ssim(sr.detach(), hr.detach())
                    _lpips = self.lpips_fn(sr, hr).item()
                    psnr += _psnr
                    ssim += _ssim
                    mse += _mse
                    lpips += _lpips
                    psnr_rgb += _psnr_rgb
                    if do_align:
                        sr_al = mean_align(sr, hr)
                        a_p, a_s, a_m, a_prgb = calc_psnr_and_ssim(sr_al.detach(), hr.detach())
                        ap_psnr += a_p
                        ap_ssim += a_s
                        ap_mse += a_m
                        ap_psnr_rgb += a_prgb
                        ap_lpips += self.lpips_fn(sr_al, hr).item()

            if (cnt == 0):
                self.logger.info('%s  evaluation skipped: empty loader' % name)
                continue

            psnr_ave = psnr / cnt
            ssim_ave = ssim / cnt
            mse_ave = mse / cnt
            lpips_ave = lpips / cnt
            self.logger.info(
                '%s  PSNR: %.3f  SSIM: %.4f  LPIPS: %.4f  MSE: %.2f  (PSNR/SSIM=RGB)'
                % (name, psnr_rgb / cnt, ssim_ave, lpips_ave, mse_ave))
            if do_align:
                self.logger.info(
                    '%s  PSNR(mean-aligned): %.3f  SSIM(mean-aligned): %.4f  '
                    'LPIPS(mean-aligned): %.4f  MSE(mean-aligned): %.2f  (RGB)'
                    % (name, ap_psnr_rgb / cnt, ap_ssim / cnt,
                       ap_lpips / cnt, ap_mse / cnt))

            if name == 'data1':
                self.data1_last_psnr = psnr_rgb / cnt

            if name == 'data1_nanobanana_huawei':
                floor = getattr(self.args, 'data1_nanobanana_huawei_floor', 20.4)
                if (getattr(self.args, 'early_stop_on_data1', False)
                        and (psnr_rgb / cnt) < floor):
                    self.data1_psnr_floor_violated = True
                    self.logger.info(
                        'EARLY STOP: data1_nanobanana_huawei PSNR(RGB) %.3f < floor %.3f'
                        % (psnr_rgb / cnt, floor))

            if name == 'data1_nanobanana_nikon':
                floor = getattr(self.args, 'data1_nanobanana_nikon_floor', 17.0)
                if (getattr(self.args, 'early_stop_on_data1', False)
                        and (psnr_rgb / cnt) < floor):
                    self.data1_psnr_floor_violated = True
                    self.logger.info(
                        'EARLY STOP: data1_nanobanana_nikon PSNR(RGB) %.3f < floor %.3f'
                        % (psnr_rgb / cnt, floor))

    def test(self):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' %(self.args.lr_path))
        self.logger.info('ref path:    %s' %(self.args.ref_path))

        is_enhance = getattr(self.args, 'enhance_mode', False)

        ### LR and LR_sr
        LR = imread(self.args.lr_path)
        h1, w1 = LR.shape[:2]
        if is_enhance:
            # 1:1 enhancement — no upscaling
            h1, w1 = h1//4*4, w1//4*4
            LR = LR[:h1, :w1, :]
            LR_sr = LR.copy()
        else:
            LR_sr = np.array(Image.fromarray(LR).resize((w1*4, h1*4), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = imread(self.args.ref_path)
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2, :]
        if is_enhance:
            # 1:1 enhancement — same resolution
            Ref_sr = Ref.copy()
        else:
            Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
            Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        ### to tensor
        LR_t = torch.from_numpy(LR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        LR_sr_t = torch.from_numpy(LR_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_t = torch.from_numpy(Ref.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_sr_t = torch.from_numpy(Ref_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            sr, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t)
            sr_save = (sr+1.) * 127.5
            sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
            imsave(save_path, sr_save)
            self.logger.info('output path: %s' %(save_path))

        self.logger.info('Test over.')
