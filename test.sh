#!/bin/bash
# TTSR-lowlight 测试脚本 — data1 Eval 集
# 用法: bash test.sh [model_epoch] [version]
# 示例: bash test.sh 15 v6   → v6 epoch 15

EPOCH="${1:-45}"
VER="${2:-v9}"

MODEL_PATH="/root/data/experiments/TTSR-lowlight-${VER}/model/model_$(printf '%05d' $EPOCH).pt"
SAVE_DIR="/root/data/experiments/TTSR-lowlight-${VER}"

echo "============================================"
echo " TTSR-lowlight Test — data1 Eval"
echo " Version: $VER  Epoch: $EPOCH"
echo " Model : $MODEL_PATH"
echo " Save  : $SAVE_DIR"
echo "============================================"

conda run -n ttsr python main.py \
    --dataset data1 \
    --dataset_dir /root/data/datasets/data1 \
    --save_dir "$SAVE_DIR" \
    --reset False \
    --log_file_name test_data1_eval.log \
    --eval True \
    --eval_save_results True \
    --enhance_mode True \
    --ref_degrade True \
    --ref_color_jitter 0.2 \
    --ref_shift_range 4 \
    --ref_blur_sigma 2.0 \
    --model_path "$MODEL_PATH" \
    --num_workers 4 \
    --num_gpu 1

echo ""
echo "Output: $SAVE_DIR/save_results/"
echo "Log   : $SAVE_DIR/test_data1_eval.log"
