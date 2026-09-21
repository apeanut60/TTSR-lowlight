#!/bin/bash
# TTSR-lowlight v14 — data1, 深化模型 (10+10+6+4), v12/e20 续训
#   方案 A: 保持 n_feats=64, 增加 ResBlock 深度, 兼容 v12 权重
#   SFE 8→10, Stage1 8→10, Stage2 4→6, Stage3 2→4
#   参数量 ~7M (+50%), 新层随机初始化, 匹配层加载 v12/e20
#   crop=160 + rec_w=1.5 + 低饱和度 loss (color_w=0.05)

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ttsr 2>/dev/null || true
python main.py \
    --dataset data1 \
    --dataset_dir /root/data/datasets/data1 \
    --save_dir /root/data/experiments/TTSR-lowlight-v14 \
    --reset True \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 12 \
    --train_crop_size 160 \
    --num_init_epochs 0 \
    --num_epochs 60 \
    --print_every 10 \
    --save_every 5 \
    --val_every 5 \
    --decay 30 \
    --gamma 0.5 \
    --lr_rate 5e-5 \
    --lr_rate_lte 5e-6 \
    --rec_w 1.5 \
    --per_w 0.15 \
    --tpl_w 0.1 \
    --tpl_use_S True \
    --illum_smooth_w 0.5 \
    --color_w 0.05 \
    --exposure_w 0.5 \
    --ref_degrade True \
    --ref_color_jitter 0.2 \
    --ref_shift_range 4 \
    --ref_blur_sigma 2.0 \
    --load_pretrain True \
    --pretrain_path /root/data/experiments/TTSR-lowlight-v12/model/model_00020.pt \
    --freeze_lte False \
    --num_gpu 1 \
    --num_workers 4
