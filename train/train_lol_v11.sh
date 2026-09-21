#!/bin/bash
# TTSR-lowlight v11 — v10/e55 续训, MainNet S sigmoid 修复
#   修复: MainNetEnhance 中 S_up 加 torch.sigmoid, 对齐 tpl_loss 的 [0,1] 软门控
#   纯色暗区 S≈0.5 中性 → 抑制随机纹理注入 → 减少棋盘格伪影
#   从 v10/e55 (PSNR 20.359, LPIPS 0.3157) 续训, CSFI 只需适配 S 值域变化

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ttsr 2>/dev/null || true
python main.py \
    --dataset data1 \
    --dataset_dir /root/data/datasets/data1 \
    --save_dir /root/data/experiments/TTSR-lowlight-v11 \
    --reset True \
    --enhance_mode True \
    --num_res_blocks 8+8+4+2 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 12 \
    --train_crop_size 128 \
    --num_init_epochs 0 \
    --num_epochs 40 \
    --print_every 10 \
    --save_every 5 \
    --val_every 5 \
    --decay 20 \
    --gamma 0.5 \
    --lr_rate 5e-5 \
    --lr_rate_lte 5e-6 \
    --rec_w 1.3 \
    --per_w 0.15 \
    --tpl_w 0.1 \
    --tpl_use_S True \
    --illum_smooth_w 1.0 \
    --color_w 0.2 \
    --exposure_w 1.0 \
    --ref_degrade True \
    --ref_color_jitter 0.2 \
    --ref_shift_range 4 \
    --ref_blur_sigma 2.0 \
    --load_pretrain True \
    --pretrain_path /root/data/experiments/TTSR-lowlight-v10/model/model_00055.pt \
    --freeze_lte False \
    --num_gpu 1 \
    --num_workers 4
