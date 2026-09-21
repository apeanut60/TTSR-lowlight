#!/bin/bash
# TTSR-lowlight v12 — v11/e10 续训, 低 lr 微调 + 降 Grey-World 解饱和度
#   v11 问题: lr=5e-5 对微调过高 e10 后 L1 反弹; 画面发白饱和度低
#   修复: lr↓2e-5, decay@10, color_w 0.2→0.05, exposure_w 1.0→0.5, illum_w 1.0→0.5
#   从 v11/e10 (PSNR 20.292, LPIPS 0.3070) 续训

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ttsr 2>/dev/null || true
python main.py \
    --dataset data1 \
    --dataset_dir /root/data/datasets/data1 \
    --save_dir /root/data/experiments/TTSR-lowlight-v12 \
    --reset True \
    --enhance_mode True \
    --num_res_blocks 8+8+4+2 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 12 \
    --train_crop_size 128 \
    --num_init_epochs 0 \
    --num_epochs 20 \
    --print_every 10 \
    --save_every 5 \
    --val_every 5 \
    --decay 10 \
    --gamma 0.5 \
    --lr_rate 2e-5 \
    --lr_rate_lte 2e-6 \
    --rec_w 1.3 \
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
    --pretrain_path /root/data/experiments/TTSR-lowlight-v11/model/model_00010.pt \
    --freeze_lte False \
    --num_gpu 1 \
    --num_workers 4
