#!/bin/bash
# TTSR-lowlight v6 — data1 新起点
#   继承 v5 最佳 PSNR 21.997 的 loss 体系 (Grey-World + 低曝光 + VGG 感知)
#   从 v5/e10 续训，迁移至 data1 多设备数据集
#   rec_w=1.5 加强重建约束, exposure_w=1.0 适度提亮, 无 sat_loss

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ttsr 2>/dev/null || true
python main.py \
    --dataset data1 \
    --dataset_dir /root/data/datasets/data1 \
    --save_dir /root/data/experiments/TTSR-lowlight-v6 \
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
    --exposure_w 0.9 \
    --ref_degrade True \
    --ref_color_jitter 0.2 \
    --ref_shift_range 4 \
    --ref_blur_sigma 2.0 \
    --load_pretrain True \
    --pretrain_path /root/data/experiments/TTSR-lowlight-v5/model/model_00010.pt \
    --freeze_lte False \
    --num_gpu 1 \
    --num_workers 4
