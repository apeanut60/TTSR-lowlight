#!/bin/bash
# TTSR-lowlight v7.1 — data2 clean ref
#   数据集 data2: ref 来自 v6e40 增强图像 (gen_ref.py 产出)
#   ref_degrade=False: 使用干净对齐参考, 模型专注纹理迁移
#   从 v7.1/e20 (PSNR 20.471) 续训, loss 权重继承 v7.1

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ttsr 2>/dev/null || true
python main.py \
    --dataset data2 \
    --dataset_dir /root/data/datasets/data2 \
    --save_dir /root/data/experiments/TTSR-lowlight-v7.2 \
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
    --rec_w 1.5 \
    --per_w 0.15 \
    --tpl_w 0.1 \
    --tpl_use_S True \
    --illum_smooth_w 1.0 \
    --color_w 0.1 \
    --exposure_w 1 \
    --ref_degrade False \
    --load_pretrain True \
    --pretrain_path /root/data/experiments/TTSR-lowlight-v7/model/model_00005.pt \
    --freeze_lte False \
    --num_gpu 1 \
    --num_workers 4
