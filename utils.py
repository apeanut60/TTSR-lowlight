import math
import numpy as np
import logging
import cv2
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def mkExpDir(args):
    if (os.path.exists(args.save_dir)):
        if (not args.reset):
            # Allow eval/test to reuse existing directory
            if (args.eval or args.test):
                pass
            else:
                raise SystemExit('Error: save_dir "' + args.save_dir + '" already exists! Please set --reset True to delete the folder.')
        else:
            shutil.rmtree(args.save_dir)
            os.makedirs(args.save_dir, exist_ok=True)
    else:
        os.makedirs(args.save_dir, exist_ok=True)
    # os.makedirs(os.path.join(args.save_dir, 'img'))

    if ((not args.eval) and (not args.test)):
        os.makedirs(os.path.join(args.save_dir, 'model'), exist_ok=True)
    
    if ((args.eval and args.eval_save_results) or args.test):
        os.makedirs(os.path.join(args.save_dir, 'save_results'), exist_ok=True)

    # Save args only for training runs (eval/test would overwrite training args)
    if ((not args.eval) and (not args.test)):
        args_file = open(os.path.join(args.save_dir, 'args.txt'), 'w')
        for k, v in vars(args).items():
            args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')
        args_file.close()

    _logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name), 
        logger_name=args.logger_name).get_log()

    return _logger


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        #H_out = (H_in+2P-K)/S + 1 所以这个conv的输出尺寸和输入尺寸相同
        super(MeanShift, self).__init__(3, 3, kernel_size=1)  
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False


def calc_psnr(img1, img2):
    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
    diff = (img1 - img2) / 255.0
    diff[:,:,0] = diff[:,:,0] * 65.738 / 256.0
    diff[:,:,1] = diff[:,:,1] * 129.057 / 256.0
    diff[:,:,2] = diff[:,:,2] * 25.064 / 256.0

    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))
    return -10 * math.log10(mse)


def calc_psnr_rgb(img1, img2):
    """Standard three-channel PSNR over RGB (peak value 255).

    Args:
        img1, img2: [h, w, c], range [0, 255]
    """
    diff = (img1.astype(np.float64) - img2.astype(np.float64)) / 255.0
    mse = float(np.mean(np.power(diff, 2)))
    if mse <= 0:
        return float('inf')
    return -10.0 * math.log10(mse)
    
  
def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        # the same outputs as MATLAB's
    border = 0
    img1_y = np.dot(img1, [65.738,129.057,25.064])/256.0+16.0
    img2_y = np.dot(img2, [65.738,129.057,25.064])/256.0+16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def mean_align(sr, hr):
    """Scale ``sr`` so that its Y-channel mean matches ``hr``'s.

    Args:
        sr, hr: pytorch tensors in [-1, 1], shape [N, 3, H, W].

    Returns:
        Aligned ``sr`` in [-1, 1], clamped to the valid [0, 255] pixel range.
    """
    sr_c = (sr + 1.) * 127.5
    hr_c = (hr + 1.) * 127.5
    w = torch.tensor([65.738, 129.057, 25.064],
                     dtype=sr.dtype, device=sr.device).view(1, 3, 1, 1) / 256.0
    sr_y = (sr_c * w).sum(dim=1, keepdim=True)
    hr_y = (hr_c * w).sum(dim=1, keepdim=True)
    scale = hr_y.mean(dim=(2, 3), keepdim=True) / (
        sr_y.mean(dim=(2, 3), keepdim=True) + 1e-6)
    return (sr_c * scale).clamp(0., 255.) / 127.5 - 1.


def chroma_gain(sr, gain):
    """Scale chroma (Cb/Cr, BT.601) about neutral by ``gain``.

    Args:
        sr: pytorch tensor in [-1, 1], shape [N, 3, H, W].
        gain: >1 boosts saturation, 1.0 is a no-op.

    Returns:
        Tensor in [-1, 1]. This is a fixed colour-calibration post-process; it
        never uses the GT.
    """
    if gain is None or abs(gain - 1.0) < 1e-8:
        return sr
    x = (sr + 1.) * 127.5
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.
    cb = (b - y) * 0.564 + 128.
    cr = 128. + gain * (cr - 128.)
    cb = 128. + gain * (cb - 128.)
    r2 = y + 1.402 * (cr - 128.)
    g2 = y - 0.344136 * (cb - 128.) - 0.714136 * (cr - 128.)
    b2 = y + 1.772 * (cb - 128.)
    out = torch.clamp(torch.cat((r2, g2, b2), dim=1), 0., 255.)
    return out / 127.5 - 1.


def calc_psnr_and_ssim(sr, hr):
    ### args:
        # sr: pytorch tensor, range [-1, 1]
        # hr: pytorch tensor, range [-1, 1]

    ### prepare data
    sr = (sr+1.) * 127.5
    hr = (hr+1.) * 127.5
    if (sr.size() != hr.size()):
        h_min = min(sr.size(2), hr.size(2))
        w_min = min(sr.size(3), hr.size(3))
        sr = sr[:, :, :h_min, :w_min]
        hr = hr[:, :, :h_min, :w_min]

    img1 = np.transpose(sr.squeeze().round().cpu().numpy(), (1,2,0))
    img2 = np.transpose(hr.squeeze().round().cpu().numpy(), (1,2,0))

    psnr = calc_psnr(img1, img2)
    psnr_rgb = calc_psnr_rgb(img1, img2)
    ssim = calc_ssim(img1, img2)
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

    return psnr, ssim, mse, psnr_rgb
