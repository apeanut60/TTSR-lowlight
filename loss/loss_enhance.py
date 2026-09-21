"""
Loss functions for TTSR-lowlight enhancement.
Combines original TTSR losses with low-light specific losses:
- Illumination Smoothness: TV regularization to prevent over-enhancement
- Color Constancy: Grey-World assumption to prevent color shifts
- Exposure Control: Keeps enhanced image brightness in a target range
"""

from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─── Original TTSR Losses (unchanged) ────────────────────────────────────

class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        self.type = type
        self.eps = 1e-6
        if type == 'l1':
            self.loss = nn.L1Loss()
        elif type == 'l2':
            self.loss = nn.MSELoss()
        elif type == 'charbonnier':
            self.loss = None
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr):
        if self.type == 'charbonnier':
            diff = sr - hr
            return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return self.loss(sr, hr)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, sr_relu5_1, hr_relu5_1):
        return F.mse_loss(sr_relu5_1, hr_relu5_1)


class TransferalPerceptualLoss(nn.Module):
    """Transferal perceptual loss for same-resolution enhancement.
    T_lv1/2/3 are at their native VGG resolutions (lv1@H×W, lv2@H/2×W/2, lv3@H/4×W/4).
    Maps (sr_lv*) are matched via bicubic interpolation where needed.
    """
    def __init__(self, use_S=True, type='l2'):
        super(TransferalPerceptualLoss, self).__init__()
        self.use_S = use_S
        self.type = type

    def forward(self, map_lv3, map_lv2, map_lv1, S, T_lv3, T_lv2, T_lv1):
        """
        Args:
            map_lv1/2/3: LTE features of enhanced image at 3 VGG scales
            S:             soft-attention map (lv3 resolution: H/4×W/4)
            T_lv3:         transferred texture (lv3: H/4×W/4)
            T_lv2:         transferred texture (lv2: H/2×W/2)
            T_lv1:         transferred texture (lv1: H×W)
        """
        # Upsample T and S to match map spatial sizes for lv2/lv1
        T_lv2_up = F.interpolate(
            T_lv2, size=map_lv2.shape[-2:], mode='bicubic')
        T_lv1_up = F.interpolate(
            T_lv1, size=map_lv1.shape[-2:], mode='bicubic')

        if self.use_S:
            S_lv3 = torch.sigmoid(S)
            S_lv2 = torch.sigmoid(F.interpolate(
                S, size=map_lv2.shape[-2:], mode='bicubic'))
            S_lv1 = torch.sigmoid(F.interpolate(
                S, size=map_lv1.shape[-2:], mode='bicubic'))
        else:
            S_lv3, S_lv2, S_lv1 = 1., 1., 1.

        if self.type == 'l1':
            loss_texture = F.l1_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.l1_loss(map_lv2 * S_lv2, T_lv2_up * S_lv2)
            loss_texture += F.l1_loss(map_lv1 * S_lv1, T_lv1_up * S_lv1)
            loss_texture /= 3.
        elif self.type == 'l2':
            loss_texture = F.mse_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.mse_loss(map_lv2 * S_lv2, T_lv2_up * S_lv2)
            loss_texture += F.mse_loss(map_lv1 * S_lv1, T_lv1_up * S_lv1)
            loss_texture /= 3.

        return loss_texture


# ─── Low-Light Specific Losses ───────────────────────────────────────────

class IlluminationSmoothnessLoss(nn.Module):
    """Total Variation loss on the enhanced image.
    Penalizes large pixel-to-pixel variations, preventing noise amplification
    and over-enhancement artifacts common in low-light enhancement.
    """

    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()

    def forward(self, x):
        # x: [N, C, H, W], range [-1, 1]
        batch_size, channels, h, w = x.size()
        # Horizontal and vertical differences
        count_h = channels * (w - 1) * h
        count_w = channels * (h - 1) * w
        h_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        w_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        return 2. * (h_tv / count_h + w_tv / count_w) / batch_size


class ColorConstancyLoss(nn.Module):
    """Grey-World color constancy loss.
    Encourages the R, G, B channel means to be similar,
    preventing unnatural color shifts during enhancement.
    """

    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, x):
        # x: [N, 3, H, W], range [-1, 1]
        # Convert from [-1, 1] to [0, 1] for mean calculation
        x_norm = (x + 1.) / 2.
        mean_rgb = x_norm.mean(dim=[2, 3])  # [N, 3]
        # Penalize pairwise differences
        d_rg = torch.pow(mean_rgb[:, 0] - mean_rgb[:, 1], 2)
        d_rb = torch.pow(mean_rgb[:, 0] - mean_rgb[:, 2], 2)
        d_gb = torch.pow(mean_rgb[:, 1] - mean_rgb[:, 2], 2)
        return (d_rg + d_rb + d_gb).mean()


class ExposureControlLoss(nn.Module):
    """Encourage the enhanced image to have a target average brightness.
    For low-light enhancement, the target is typically around 0.4-0.6 in [0, 1]
    (which is 0.0 ± 0.2 in [-1, 1] range).

    Uses a soft constraint: loss increases when brightness deviates from target.
    """

    def __init__(self, patch_size=16, mean_val=0.4):
        super(ExposureControlLoss, self).__init__()
        self.patch_size = patch_size
        self.mean_val = mean_val

    def forward(self, x):
        # x: [N, 3, H, W], range [-1, 1]
        x_norm = (x + 1.) / 2.  # to [0, 1]
        # Average pooling to get local mean intensity
        x_pool = F.avg_pool2d(x_norm, kernel_size=self.patch_size)
        # L1 distance from target
        return torch.abs(x_pool - self.mean_val).mean()


class IlluminationMatchLoss(nn.Module):
    """Match the low-frequency (illumination/tone) component of the prediction
    to the GT.

    Both images are average-pooled to a coarse grid before the L1, so the loss
    focuses on global brightness and low-frequency tone instead of detail.
    """

    def __init__(self, factor=8):
        super(IlluminationMatchLoss, self).__init__()
        self.factor = max(1, int(factor))

    def forward(self, sr, hr):
        if self.factor > 1:
            sr_d = F.avg_pool2d(sr, self.factor)
            hr_d = F.avg_pool2d(hr, self.factor)
        else:
            sr_d, hr_d = sr, hr
        return F.l1_loss(sr_d, hr_d)


# ─── Adversarial Loss (kept from original) ───────────────────────────────

class AdversarialLoss(nn.Module):
    def __init__(self, logger, use_cpu=False, num_gpu=1, gan_type='WGAN_GP',
                 gan_k=1, lr_dis=1e-4, train_crop_size=128):

        super(AdversarialLoss, self).__init__()
        self.logger = logger
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = discriminator.Discriminator(
            train_crop_size).to(self.device)
        if num_gpu > 1:
            self.discriminator = nn.DataParallel(
                self.discriminator, list(range(num_gpu)))
        if gan_type in ['WGAN_GP', 'GAN']:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)

    def forward(self, fake, real):
        fake_detach = fake.detach()

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True,
                        only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif self.gan_type == 'GAN':
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.

            loss_d.backward()
            self.optimizer.step()

        d_fake_for_g = self.discriminator(fake)
        if self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()
        elif self.gan_type == 'GAN':
            valid_score = torch.ones(real.size(0), 1).to(self.device)
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        return loss_g

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict


# ─── Loss Dictionary Builder ─────────────────────────────────────────────

def get_loss_dict_enhance(args, logger):
    """Build the loss dictionary for low-light enhancement training.
    Returns dict of loss modules. Weights are applied during training in trainer.
    """
    loss = {}

    # Reconstruction loss — always used
    if abs(args.rec_w) <= 1e-8:
        raise SystemExit(
            'NotImplementedError: ReconstructionLoss must exist!')
    loss['rec_loss'] = ReconstructionLoss(
        type=getattr(args, 'rec_loss_type', 'l1'))

    # Perceptual loss
    if abs(args.per_w) > 1e-8:
        loss['per_loss'] = PerceptualLoss()

    # Transferal perceptual loss
    if abs(args.tpl_w) > 1e-8:
        loss['tpl_loss'] = TransferalPerceptualLoss(
            use_S=args.tpl_use_S, type=args.tpl_type)

    # Adversarial loss
    if abs(args.adv_w) > 1e-8:
        loss['adv_loss'] = AdversarialLoss(
            logger=logger, use_cpu=args.cpu, num_gpu=args.num_gpu,
            gan_type=args.GAN_type, gan_k=args.GAN_k,
            lr_dis=args.lr_rate_dis,
            train_crop_size=args.train_crop_size)

    # Low-light specific losses
    if abs(args.illum_smooth_w) > 1e-8:
        loss['illum_smooth_loss'] = IlluminationSmoothnessLoss()

    if abs(args.color_w) > 1e-8:
        loss['color_loss'] = ColorConstancyLoss()

    if abs(args.exposure_w) > 1e-8:
        loss['exposure_loss'] = ExposureControlLoss(
            mean_val=getattr(args, 'exposure_target', 0.4))

    if abs(getattr(args, 'illum_match_w', 0.0)) > 1e-8:
        loss['illum_match_loss'] = IlluminationMatchLoss(
            factor=getattr(args, 'illum_match_factor', 8))

    return loss
