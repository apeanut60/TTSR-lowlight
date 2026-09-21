#!/bin/bash
# TTSR-lowlight v13 — data2 + v12/e20 增强 ref + 轻度退化 + 强力训练
#   策略: v12 增强图质量已高, 只需轻度退化保持鲁棒性
#        + rec_w↑增强像素约束 + crop↑更大感受野 + 80ep 充分收敛
#   退化: jitter 0.2→0.08, shift 4→2, blur 2.0→1.0 (匹配增强图质量)

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ttsr 2>/dev/null || true
python main.py \
    --dataset data2 \
    --dataset_dir /root/data/datasets/data2 \
    --save_dir /root/data/experiments/TTSR-lowlight-v13 \
    --reset True \
    --enhance_mode True \
    --num_res_blocks 8+8+4+2 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 12 \
    --train_crop_size 160 \
    --num_init_epochs 0 \
    --num_epochs 80 \
    --print_every 10 \
    --save_every 5 \
    --val_every 5 \
    --decay 40 \
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
    --ref_color_jitter 0.08 \
    --ref_shift_range 2 \
    --ref_blur_sigma 0.1 \
    --load_pretrain True \
    --pretrain_path /root/data/experiments/TTSR-lowlight-v12/model/model_00020.pt \
    --freeze_lte False \
    --num_gpu 1 \
    --num_workers 4
