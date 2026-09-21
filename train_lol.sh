#!/bin/bash
# TTSR-lowlight v2 — Reference Degradation Training
# 参考图像退化: 颜色抖动 + 位置偏移 + 高斯模糊

python main.py \
    --dataset LOL \
    --dataset_dir /root/data/datasets/LOLdataset \
    --save_dir /root/data/experiments/TTSR-lowlight-v2 \
    --reset True \
    --enhance_mode True \
    --num_res_blocks 8+8+4+2 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 4 \
    --train_crop_size 128 \
    --num_init_epochs 5 \
    --num_epochs 50 \
    --print_every 10 \
    --save_every 5 \
    --val_every 5 \
    --lr_rate 1e-4 \
    --lr_rate_lte 1e-5 \
    --rec_w 1.0 \
    --per_w 0.1 \
    --tpl_w 0.1 \
    --tpl_use_S True \
    --illum_smooth_w 1.0 \
    --color_w 0.5 \
    --exposure_w 1.0 \
    --ref_degrade True \
    --ref_color_jitter 0.2 \
    --ref_shift_range 4 \
    --ref_blur_sigma 2.0 \
    --load_pretrain False \
    --pretrain_path /root/data/pretrain_models/TTSR-rec.pt \
    --freeze_lte False \
    --num_gpu 1 \
    --num_workers 4
