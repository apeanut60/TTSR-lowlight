#!/bin/bash
# TTSR-lowlight v8.1 — data2 clean ref + degrade, v8e10续训
#   数据集 data2: ref 来自 v6e40, 退化训练增强鲁棒性
#   freeze_lte=True: 冻结VGG, 精调MainNet
#   decay=30, lr=2.5e-5: 从峰值缓降探索

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ttsr 2>/dev/null || true
python main.py \
    --dataset data2 \
    --dataset_dir /root/data/datasets/data2 \
    --save_dir /root/data/experiments/TTSR-lowlight-v8.1 \
    --reset True \
    --enhance_mode True \
    --num_res_blocks 8+8+4+2 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 12 \
    --train_crop_size 128 \
    --num_init_epochs 0 \
    --num_epochs 50 \
    --print_every 10 \
    --save_every 5 \
    --val_every 5 \
    --decay 30 \
    --gamma 0.5 \
    --lr_rate 2.5e-5 \
    --lr_rate_lte 5e-6 \
    --rec_w 1.3 \
    --per_w 0.15 \
    --tpl_w 0.1 \
    --tpl_use_S True \
    --illum_smooth_w 1.0 \
    --color_w 0.2 \
    --exposure_w 1.0 \
    --ref_degrade True \
    --load_pretrain True \
    --pretrain_path /root/data/experiments/TTSR-lowlight-v8/model/model_00010.pt \
    --freeze_lte True \
    --num_gpu 1 \
    --num_workers 4
