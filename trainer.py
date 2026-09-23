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

        if not frozen_names:
            # The prefix table above is MainNetEnhance-specific. A different
            # backbone would silently freeze nothing, so fail loudly instead.
            raise RuntimeError(
                'freeze_stages=%r matched no parameters — the stage prefix '
                'table in freeze_mainnet_stages does not describe this '
                'backbone (%s)' % (freeze_stages, type(mainnet).__name__))

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

        # Guard against silently-empty groups. A renamed submodule would
        # otherwise move its parameters into the generic mainnet group and make
        # --lr_rate_refillum / --lr_rate_illum / --lr_rate_stage2 no-ops
        # instead of raising.
        ref_illum_mod = getattr(mainnet, 'ref_illum', None)
        if ref_illum_mod is not None:
            n_trainable = len([p for p in ref_illum_mod.parameters()
                               if p.requires_grad])
            if n_trainable and not refillum_params:
                raise RuntimeError(
                    'ref_illum has %d trainable tensors but the refillum '
                    'optimizer group is empty — the prefix "ref_illum." must '
                    'match this backbone\'s submodule name.' % n_trainable)
        seen = {}
        for gi, g in enumerate(self.params):
            for p in g['params']:
                if id(p) in seen:
                    raise RuntimeError(
                        'parameter appears in two optimizer groups (%d and %d)'
                        % (seen[id(p)], gi))
                seen[id(p)] = gi
        for gi, g in enumerate(self.params):
            self.logger.info('optimizer group %d/%d: %d tensors, lr=%g'
                             % (gi + 1, len(self.params), len(g['params']),
                                g['lr']))

        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
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

    def _tiled_forward(self, lr, lr_sr, ref, ref_sr, tile_size=None, overlap=None):
        """Tiled inference to avoid OOM on large images.
        Splits input into overlapping tiles, runs the model on each,
        and blends overlapping regions with a configurable window.

        Args:
            lr, lr_sr, ref, ref_sr: [1, C, H, W] tensors
            tile_size: tile edge length
            overlap: overlap between adjacent tiles

        Returns:
            sr: [1, C, H, W] enhanced image
            S, T_lv3, T_lv2, T_lv1: aggregated (tensors from last tile)
        """
        tile_size = tile_size or getattr(self.args, 'tile_size', 256)
        overlap = overlap if (overlap is not None) else getattr(
            self.args, 'tile_overlap', 96)
        overlap = max(0, min(overlap, tile_size // 2))
        window = getattr(self.args, 'tile_window', 'cosine').lower()

        _, _, H, W = lr.shape
        if H <= tile_size and W <= tile_size:
            return self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)

        stride = max(1, tile_size - overlap)
        # Number of tiles in each dimension
        n_h = max(1, (H - overlap + stride - 1) // stride)
        n_w = max(1, (W - overlap + stride - 1) // stride)

        # Output accumulator and weight map for feathering
        sr_accum = torch.zeros(1, 3, H, W, device=lr.device)
        weight_accum = torch.zeros(1, 1, H, W, device=lr.device)

        # Feathering mask: 1.0 in center, decays to 0 at edges
        mask_y = torch.ones(tile_size, device=lr.device)
        mask_x = torch.ones(tile_size, device=lr.device)
        if overlap > 0:
            if window == 'cosine':
                ramp = 0.5 - 0.5 * torch.cos(
                    torch.linspace(0, math.pi, overlap, device=lr.device))
            else:
                ramp = torch.linspace(0, 1, overlap, device=lr.device)
            mask_y[:overlap] = ramp
            mask_y[-overlap:] = ramp.flip(0)
            mask_x[:overlap] = ramp
            mask_x[-overlap:] = ramp.flip(0)
        feather = mask_y[:, None] * mask_x[None, :]  # [tile, tile]
        feather = feather.view(1, 1, tile_size, tile_size)

        last_S, last_T3, last_T2, last_T1 = None, None, None, None
        # The global illumination head is a per-image transform. Run the
        # backbone per tile without it, then apply it once on the stitched
        # result so tiles cannot get mismatched gamma/gain (visible seams).
        net = self.model.module if hasattr(self.model, 'module') else self.model
        has_illum = hasattr(net, 'apply_illum_head')
        has_refillum = hasattr(net, 'apply_ref_illum')

        for i_h in range(n_h):
            y0 = min(i_h * stride, H - tile_size)
            y1 = y0 + tile_size
            for i_w in range(n_w):
                x0 = min(i_w * stride, W - tile_size)
                x1 = x0 + tile_size

                tile_lr = lr[:, :, y0:y1, x0:x1]
                tile_lr_sr = lr_sr[:, :, y0:y1, x0:x1]
                tile_ref = ref[:, :, y0:y1, x0:x1]
                tile_ref_sr = ref_sr[:, :, y0:y1, x0:x1]

                tile_kwargs = dict(lr=tile_lr, lrsr=tile_lr_sr,
                                   ref=tile_ref, refsr=tile_ref_sr)
                if has_illum:
                    tile_kwargs['apply_illum'] = False
                if has_refillum:
                    tile_kwargs['apply_ref_illum'] = False
                tile_sr, S, T3, T2, T1 = self.model(**tile_kwargs)

                sr_accum[:, :, y0:y1, x0:x1] += tile_sr * feather
                weight_accum[:, :, y0:y1, x0:x1] += feather
                last_S, last_T3, last_T2, last_T1 = S, T3, T2, T1

        sr = sr_accum / weight_accum.clamp(min=1e-8)
        if has_refillum:
            sr = net.apply_ref_illum(sr, lr, ref)
        if has_illum:
            sr = net.apply_illum_head(sr, lr)
        return sr, last_S, last_T3, last_T2, last_T1

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
        if (not is_init):
            self.scheduler.step()
        # Log every group, not just the first: a silently-empty group (wrong
        # submodule name) would otherwise be invisible here.
        self.logger.info('Current epoch learning rates: ' + '  '.join(
            'g%d(%d tensors)=%.3e' % (i + 1, len(g['params']), g['lr'])
            for i, g in enumerate(self.optimizer.param_groups)))

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

            if has_ref_head:
                model_out = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr,
                                       return_ref=True)
                sr, S, T_lv3, T_lv2, T_lv1, ref_corrected = model_out
            else:
                sr, S, T_lv3, T_lv2, T_lv1 = self.model(
                    lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                ref_corrected = None

            ### calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss = rec_loss
            if (is_print):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()) )

            if (not is_init):
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

            loss.backward()
            self.optimizer.step()

        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.save(current_epoch)

    def save(self, current_epoch):
        """Write a checkpoint for this epoch.

        Split out of ``train`` so the driver can force a final checkpoint on
        the last epoch even when ``num_epochs % save_every != 0`` (otherwise
        the tail of every run is silently lost).
        """
        self.logger.info('saving the model...')
        tmp = self.model.state_dict()
        model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if
            (('SearchNet' not in key) and ('_copy' not in key))}
        model_dir = os.path.join(self.args.save_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        model_name = os.path.join(model_dir,
            'model_' + str(current_epoch).zfill(5) + '.pt')
        torch.save(model_state_dict, model_name)
        return model_name

    def _lowlight_loss(self, sr, hr):
        """The shared low-light objective.

        Mirrors the non-init branch of ``train`` exactly (same terms, same
        weights, same order of accumulation) so an adapter-only run optimises
        precisely what a full run would. Returns (total, {name: weighted value}).
        """
        parts = {}
        total = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
        parts['rec_loss'] = float(total.detach())
        if ('per_loss' in self.loss_all):
            sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
            with torch.no_grad():
                hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
            l = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
            total = total + l
            parts['per_loss'] = float(l.detach())
        if ('illum_smooth_loss' in self.loss_all):
            l = self.args.illum_smooth_w * self.loss_all['illum_smooth_loss'](sr)
            total = total + l
            parts['illum_smooth_loss'] = float(l.detach())
        if ('color_loss' in self.loss_all):
            l = self.args.color_w * self.loss_all['color_loss'](sr)
            total = total + l
            parts['color_loss'] = float(l.detach())
        if ('exposure_loss' in self.loss_all):
            l = self.args.exposure_w * self.loss_all['exposure_loss'](sr)
            total = total + l
            parts['exposure_loss'] = float(l.detach())
        if ('illum_match_loss' in self.loss_all):
            l = self.args.illum_match_w * self.loss_all['illum_match_loss'](sr, hr)
            total = total + l
            parts['illum_match_loss'] = float(l.detach())
        return total, parts

    def save_adapter(self, step, tag=None):
        """Save the adapter alone plus a full model, both keyed by optimizer step."""
        wrapped = self.model.module if hasattr(self.model, 'module') else self.model
        adapter = wrapped.MainNet.denoiser.ref_adapter
        model_dir = os.path.join(self.args.save_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        suffix = str(step).zfill(5) + ('' if tag is None else '_' + tag)
        a_path = os.path.join(model_dir, 'adapter_' + suffix + '.pt')
        torch.save(adapter.state_dict(), a_path)
        m_path = self.save(step)
        self.logger.info('saved adapter %s and model %s' % (a_path, m_path))
        return a_path, m_path

    def train_adapter_only(self):
        """Step-based training of the H/4 texture adapter only.

        Everything else has requires_grad=False and is absent from the
        optimizer, but gradients still flow *through* the frozen network so the
        adapter is optimised against the real reconstruction objective (plan
        section B2: freezing parameters is not the same as disabling autograd).
        """
        args = self.args
        total_steps = int(getattr(args, 'adapter_steps', 3000))
        drop_step = int(getattr(args, 'adapter_lr_drop_step', 2000))
        lr_hi = float(getattr(args, 'adapter_lr', args.lr_rate))
        lr_lo = float(getattr(args, 'adapter_lr_after_drop', 5e-5))
        eval_every = int(getattr(args, 'adapter_eval_every', 1000))
        print_every = max(1, int(getattr(args, 'print_every', 100)))

        wrapped = self.model.module if hasattr(self.model, 'module') else self.model
        adapter = wrapped.MainNet.denoiser.ref_adapter
        ada_params = [p for p in adapter.parameters() if p.requires_grad]
        if not ada_params:
            raise RuntimeError('adapter-only mode but the adapter has no '
                               'trainable parameters')
        n_train = sum(p.numel() for p in ada_params)
        n_opt = sum(p.numel() for g in self.optimizer.param_groups
                    for p in g['params'])
        self.logger.info('adapter-only: %d tensors / %d params trainable; '
                         'optimizer holds %d params'
                         % (len(ada_params), n_train, n_opt))
        if n_opt != n_train:
            raise RuntimeError('optimizer holds %d params but the adapter has %d '
                               '— only the adapter may be trained here'
                               % (n_opt, n_train))

        # Snapshot the frozen tensors so we can prove they never moved.
        frozen = {n: p.detach().clone() for n, p in wrapped.named_parameters()
                  if not p.requires_grad}

        self.model.eval()          # no Dropout/BN here, but keep the intent
        adapter.train()

        loader = self.dataloader['train']

        def batches():
            while True:
                for b in loader:
                    yield b

        self.logger.info('adapter-only: evaluating the base model at step 0')
        self.evaluate(current_epoch=0)
        self.save_adapter(0)

        step = 0
        for sample_batched in batches():
            if step >= total_steps:
                break
            step += 1
            lr = lr_hi if step <= drop_step else lr_lo
            for g in self.optimizer.param_groups:
                g['lr'] = lr

            sample_batched = self.prepare(sample_batched)
            lr_in = sample_batched['LR']
            hr = sample_batched['HR']
            sr = self.model(lr=lr_in, lrsr=sample_batched['LR_sr'],
                            ref=sample_batched['Ref'],
                            refsr=sample_batched['Ref_sr'])[0]
            loss, parts = self._lowlight_loss(sr, hr)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in ada_params
                if p.grad is not None)))
            self.optimizer.step()

            if step % print_every == 0 or step == 1:
                self.logger.info(
                    'adapter step %d/%d lr=%.2e loss=%.6f |grad|=%.3e %s'
                    % (step, total_steps, lr, float(loss.detach()), gnorm,
                       ' '.join('%s=%.5f' % (k, v) for k, v in parts.items())))

            if step % eval_every == 0 or step == total_steps:
                self.model.eval()
                self.evaluate(current_epoch=step)
                self.save_adapter(step)
                adapter.train()

        moved = [n for n, p in wrapped.named_parameters()
                 if not p.requires_grad and not torch.equal(frozen[n], p)]
        if moved:
            raise RuntimeError('frozen parameters changed during adapter-only '
                               'training: %s' % moved[:5])
        self.logger.info('adapter-only: all %d frozen tensors are bit-identical '
                         'to the starting point' % len(frozen))
        return step

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        if (self.args.dataset == 'CUFED'):
            self.model.eval()
            with torch.no_grad():
                psnr, psnr_rgb, ssim, mse, cnt = 0., 0., 0., 0., 0
                ap_psnr, ap_psnr_rgb, ap_ssim, ap_mse = 0., 0., 0., 0.
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
                    psnr += _psnr; ssim += _ssim; mse += _mse
                    psnr_rgb += _psnr_rgb
                    if do_align:
                        sr_al = mean_align(sr, hr)
                        a_p, a_s, a_m, a_prgb = calc_psnr_and_ssim(sr_al.detach(), hr.detach())
                        ap_psnr += a_p; ap_ssim += a_s; ap_mse += a_m; ap_psnr_rgb += a_prgb

                psnr_ave, ssim_ave = psnr / cnt, ssim / cnt
                mse_ave = mse / cnt
                self.logger.info('%s  PSNR: %.3f  PSNRy: %.3f  SSIM: %.4f  MSE: %.2f' %(self.args.dataset, psnr_rgb / cnt, psnr_ave, ssim_ave, mse_ave))
                if do_align:
                    self.logger.info('%s  PSNR(mean-aligned): %.3f  PSNRy(mean-aligned): %.3f  SSIM(mean-aligned): %.4f  MSE(mean-aligned): %.2f'
                                     % (self.args.dataset, ap_psnr_rgb / cnt, ap_psnr / cnt, ap_ssim / cnt,
                                        ap_mse / cnt))
                if (psnr_rgb / cnt > self.max_psnr):
                    self.max_psnr = psnr_rgb / cnt
                    self.max_psnr_epoch = current_epoch
                if (psnr_ave > self.max_psnr_y):
                    self.max_psnr_y = psnr_ave
                    self.max_psnr_y_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('%s  PSNR(max): %.3f (%d)  PSNRy(max): %.3f (%d)  SSIM(max): %.4f (%d)'
                    %(self.args.dataset, self.max_psnr, self.max_psnr_epoch,
                      self.max_psnr_y, self.max_psnr_y_epoch,
                      self.max_ssim, self.max_ssim_epoch))

        else:
            # Every non-CUFED dataset evaluates its own test set here, so all
            # datasets report the same metric set (PSNR, PSNRy, SSIM, MSE).
            self.model.eval()
            with torch.no_grad():
                psnr, psnr_rgb, ssim, mse, cnt = 0., 0., 0., 0., 0
                ap_psnr, ap_psnr_rgb, ap_ssim, ap_mse = 0., 0., 0., 0.
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
                        # Also save low-light input for comparison
                        lr_save = (lr + 1.) * 127.5
                        lr_save = np.transpose(
                            lr_save.squeeze().round().cpu().numpy(),
                            (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results',
                                            str(i_batch).zfill(5) + '_input.png'),
                               lr_save)

                    # Calculate PSNR, SSIM, MSE
                    _psnr, _ssim, _mse, _psnr_rgb = calc_psnr_and_ssim(sr.detach(), hr.detach())
                    psnr += _psnr
                    ssim += _ssim
                    mse += _mse
                    psnr_rgb += _psnr_rgb
                    if do_align:
                        sr_al = mean_align(sr, hr)
                        a_p, a_s, a_m, a_prgb = calc_psnr_and_ssim(sr_al.detach(), hr.detach())
                        ap_psnr += a_p
                        ap_ssim += a_s
                        ap_mse += a_m
                        ap_psnr_rgb += a_prgb

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                mse_ave = mse / cnt
                self.logger.info('%s  PSNR: %.3f  PSNRy: %.3f  SSIM: %.4f  MSE: %.2f'
                                 % (self.args.dataset, psnr_rgb / cnt, psnr_ave, ssim_ave, mse_ave))
                if do_align:
                    self.logger.info('%s  PSNR(mean-aligned): %.3f  PSNRy(mean-aligned): %.3f  SSIM(mean-aligned): %.4f  MSE(mean-aligned): %.2f'
                                     % (self.args.dataset, ap_psnr_rgb / cnt, ap_psnr / cnt, ap_ssim / cnt,
                                        ap_mse / cnt))
                if psnr_rgb / cnt > self.max_psnr:
                    self.max_psnr = psnr_rgb / cnt
                    self.max_psnr_epoch = current_epoch
                if psnr_ave > self.max_psnr_y:
                    self.max_psnr_y = psnr_ave
                    self.max_psnr_y_epoch = current_epoch
                if ssim_ave > self.max_ssim:
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('%s  PSNR(max): %.3f (%d)  PSNRy(max): %.3f (%d)  SSIM(max): %.4f (%d)'
                                 % (self.args.dataset, self.max_psnr, self.max_psnr_epoch,
                                    self.max_psnr_y, self.max_psnr_y_epoch,
                                    self.max_ssim, self.max_ssim_epoch))

        self._evaluate_extra_tests(current_epoch)
        self.logger.info('Evaluation over.')

    def _evaluate_extra_tests(self, current_epoch=0):
        """Evaluate extra validation sets, such as data1 during finetune."""
        if (self.dataloader is None):
            return

        extra_test = self.dataloader.get('extra_test', {})
        if (not extra_test):
            return

        for name, loader in extra_test.items():
            self.model.eval()
            psnr, psnr_rgb, ssim, mse, cnt = 0., 0., 0., 0., 0
            ap_psnr, ap_psnr_rgb, ap_ssim, ap_mse = 0., 0., 0., 0.
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
                    psnr += _psnr
                    ssim += _ssim
                    mse += _mse
                    psnr_rgb += _psnr_rgb
                    if do_align:
                        sr_al = mean_align(sr, hr)
                        a_p, a_s, a_m, a_prgb = calc_psnr_and_ssim(sr_al.detach(), hr.detach())
                        ap_psnr += a_p
                        ap_ssim += a_s
                        ap_mse += a_m
                        ap_psnr_rgb += a_prgb

            if (cnt == 0):
                self.logger.info('%s  evaluation skipped: empty loader' % name)
                continue

            psnr_ave = psnr / cnt
            ssim_ave = ssim / cnt
            mse_ave = mse / cnt
            self.logger.info('%s  PSNR: %.3f  PSNRy: %.3f  SSIM: %.4f  MSE: %.2f'
                             % (name, psnr_rgb / cnt, psnr_ave, ssim_ave, mse_ave))
            if do_align:
                self.logger.info('%s  PSNR(mean-aligned): %.3f  PSNRy(mean-aligned): %.3f  SSIM(mean-aligned): %.4f  MSE(mean-aligned): %.2f'
                                 % (name, ap_psnr_rgb / cnt, ap_psnr / cnt, ap_ssim / cnt,
                                    ap_mse / cnt))

            if name == 'data1':
                self.data1_last_psnr = psnr_ave

            if name == 'data1_nanobanana_huawei':
                floor = getattr(self.args, 'data1_nanobanana_huawei_floor', 20.4)
                if (getattr(self.args, 'early_stop_on_data1', False)
                        and psnr_ave < floor):
                    self.data1_psnr_floor_violated = True
                    self.logger.info(
                        'EARLY STOP: data1_nanobanana_huawei PSNRy %.3f < floor %.3f'
                        % (psnr_ave, floor))

            if name == 'data1_nanobanana_nikon':
                floor = getattr(self.args, 'data1_nanobanana_nikon_floor', 17.0)
                if (getattr(self.args, 'early_stop_on_data1', False)
                        and psnr_ave < floor):
                    self.data1_psnr_floor_violated = True
                    self.logger.info(
                        'EARLY STOP: data1_nanobanana_nikon PSNRy %.3f < floor %.3f'
                        % (psnr_ave, floor))

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
