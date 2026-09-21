#!/bin/bash
# TTSR-lowlight v10 — data1, 修复 SearchTransfer stride + fold 归一化 + CSFI d3 恢复
#   从 v6/e40 续训（v6 CSFI d3 原生 weights 匹配 v10，无需重学 dilation）
#   修复内容:
#     1. SearchTransfer: ref_lv2/1 unfold stride 1→2/4, fold 回原生分辨率
#     2. fold 归一化: /k² → 逐像素掩码 (消除 stride>1 的 2.2× 网格偏差)
#     3. CSFI2/3: dilation=3 恢复 (棋盘格根因是 stride, 不是 dilation)

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ttsr 2>/dev/null || true
python main.py \
    --dataset data1 \
    --dataset_dir /root/data/datasets/data1 \
    --save_dir /root/data/experiments/TTSR-lowlight-v10 \
    --reset True \
    --enhance_mode True \
    --num_res_blocks 8+8+4+2 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 12 \
    --train_crop_size 128 \
    --num_init_epochs 0 \
    --num_epochs 60 \
    --print_every 10 \
    --save_every 5 \
    --val_every 5 \
    --decay 30 \
    --gamma 0.5 \
    --lr_rate 5e-5 \
    --lr_rate_lte 5e-6 \
    --rec_w 1.3 \
    --per_w 0.15 \
    --tpl_w 0.12 \
    --tpl_use_S True \
    --illum_smooth_w 1.0 \
    --color_w 0.2 \
    --exposure_w 1.0 \
    --ref_degrade True \
    --ref_color_jitter 0.2 \
    --ref_shift_range 4 \
    --ref_blur_sigma 2.0 \
    --load_pretrain True \
    --pretrain_path /root/data/experiments/TTSR-lowlight-v6/model/model_00040.pt \
    --freeze_lte False \
    --num_gpu 1 \
    --num_workers 4
