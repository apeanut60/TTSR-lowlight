#!/bin/bash
# TTSR-lowlight v5 — 提升饱和度实验
# 基于 v4 best (epoch 80, PSNR 20.97) 续训
# 调整: color_w 0.5→0.2, per_w 0.1→0.15, exp_w 1.0→0.8
# 目标: 减少 Grey-World 削饱和度，增加 VGG 色彩感知，释放曝光动态范围

source /root/miniconda3/envs/ttsr/bin/activate 2>/dev/null || true
/root/miniconda3/envs/ttsr/bin/python main.py \
    --dataset LOL \
    --dataset_dir /root/data/datasets/LOLdataset \
    --save_dir /root/data/experiments/TTSR-lowlight-v5 \
    --reset True \
    --enhance_mode True \
    --num_res_blocks 8+8+4+2 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 4 \
    --train_crop_size 128 \
    --num_init_epochs 0 \
    --num_epochs 50 \
    --print_every 10 \
    --save_every 5 \
    --val_every 5 \
    --decay 30 \
    --gamma 0.5 \
    --lr_rate 5e-5 \
    --lr_rate_lte 5e-6 \
    --rec_w 1.0 \
    --per_w 0.15 \
    --tpl_w 0.1 \
    --tpl_use_S True \
    --illum_smooth_w 1.0 \
    --color_w 0.2 \
    --exposure_w 0.8 \
    --ref_degrade True \
    --ref_color_jitter 0.2 \
    --ref_shift_range 4 \
    --ref_blur_sigma 2.0 \
    --load_pretrain True \
    --pretrain_path /root/data/experiments/TTSR-lowlight-v4/model/model_00080.pt \
    --freeze_lte False \
    --num_gpu 1 \
    --num_workers 4
