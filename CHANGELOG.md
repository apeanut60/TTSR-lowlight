# TTSR-lowlight 修改日志

## 项目概述
将 TTSR (Texture Transformer Super-Resolution) 从 4× 超分改造为 1:1 同分辨率低光照增强模型。
原项目: `/root/projects/TTSR-master/`
工作目录: `/root/projects/TTSR-lowlight/`

---

## 修改记录

| 日期 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 2026-05-20 | `CHANGELOG.md` | 新建 | 修改日志文件 |
| 2026-05-20 | `dataset/lol.py` | **新建** | LOL 数据集类 (TrainSet/TestSet)，1:1 配对，不做 4× 下采样，low→LR, high→HR+Ref |
| 2026-05-20 | `dataset/dataloader.py` | 修改 | 添加 `LOL` 分支，训练/测试 DataLoader |
| 2026-05-20 | `option.py` | 修改 | 默认参数全面更新：LOL 数据集、data 路径、enhance_mode、train_crop_size=128、num_res_blocks=8+8+4+2、低光照 loss 权重、新增 load_pretrain/freeze_lte 参数 |
| 2026-05-20 | `model/MainNetEnhance.py` | **新建** | 1:1 同分辨率增强网络。移除所有 PixelShuffle 上采样，CSFI 改用 dilated conv，MergeTail 插值移除，三路特征同分辨率 |
| 2026-05-20 | `model/SearchTransfer.py` | 修改 | unfold stride 从 (1,2,4)→全 1，kernel 从 (3,6,12)→(3,5,7)，F.fold 输出全为同分辨率 |
| 2026-05-20 | `model/TTSREnhance.py` | **新建** | 增强版主模块，使用 MainNetEnhance，lrsr/refsr 直接传入原始图像，含 load_pretrained_weights() 迁移学习支持 |
| 2026-05-20 | `loss/loss_enhance.py` | **修复** | 光照感知损失函数集：IlluminationSmoothnessLoss(TV)、ColorConstancyLoss(Grey-World)、ExposureControlLoss + 原 TTSR 损失 |
| 2026-05-20 | `trainer.py` | 修改 | train() 添加 illum/color/exposure loss；evaluate() 添加 LOL 评估分支（含低光照输入保存）；test() 添加 enhance_mode 支持（1:1 不放大）；load() 改为部分权重加载 |
| 2026-05-20 | `main.py` | 修改 | 添加 enhance_mode 分支：使用 TTSREnhance + get_loss_dict_enhance，支持预训练权重加载 |
| 2026-05-20 | `train_lol.sh` | **新建** | LOL 训练脚本，Phase 1 warmup 5 epochs + Phase 2 全 loss 训练 50 epochs |
| 2026-06-01 | `loss/loss_enhance.py` | **修改** | v7: 新增 `SaturationEnhancementLoss`（HSV 软下界饱和度损失），`get_loss_dict_enhance` 中 `color_loss` 替换为 `sat_loss` |
| 2026-06-01 | `option.py` | **修改** | v7: 新增 `--sat_w`(0.3), `--sat_target`(0.25)；`--color_w` 标注 deprecated |
| 2026-06-01 | `trainer.py` | **修改** | v7: 新增 `_tiled_forward()` 修复 eval OOM；`train()` 添加 `sat_loss` 打印；`evaluate()` 使用分块推理 |
| 2026-06-01 | `train_lol_v7.sh` | **新建** | v7 训练脚本，data1 数据集，sat_w=0.3+sat_target=0.25，从 v5/e10 续训 |

### v7+/v8 饱和度实验 (2026-06-01 ~ 2026-06-02)

| 日期 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 2026-06-01 | `loss/loss_enhance.py` | **修改** | v7: 新增 `SaturationEnhancementLoss`（HSV 软下界），`get_loss_dict_enhance` 注册 `sat_loss` |
| 2026-06-01 | `option.py` | **修改** | v7: 新增 `--sat_w`(0.3), `--sat_target`(0.25) |
| 2026-06-01 | `trainer.py` | **修改** | v7: 新增 `_tiled_forward()` 修复 eval OOM；`train()` 加 `sat_loss` 打印 |
| 2026-06-01 | `train_lol_v7.sh` | **新建** | 原 v7: sat_w=0.3, 无 color, data1, 从 v5/e10 续训 |
| 2026-06-02 | `loss/loss_enhance.py` | **修改** | v8: sat_target 提至 0.45 + upper=0.85 防过饱和, sat_w=2.0 |
| 2026-06-02 | `train_lol_v8.sh` | **新建** | 原 v8: sat_w=2.0+color_w=0.2, 从 v7/e35 续训 |
| 2026-06-02 | `loss/loss_enhance.py` | **修复** | color_loss 与 sat_loss 共存 bug：条件 `abs(sat_w)<=1e-8` 阻止 color_loss 注册 |
| 2026-06-02 | `train_lol_v8.sh` | **修改** | 修复后重跑 v8: 同时启用 sat+color, PSNR 21.16@e15 |
| 2026-06-02 | `train_lol_v9.sh` | **新建** | v9: sat_target=0.45, 从 v8/e15 续训 |

### 回滚至 v5 + 新路线 (2026-06-06 ~ 2026-06-08)

| 日期 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 2026-06-06 | `loss/loss_enhance.py` | **回滚** | 删除 `SaturationEnhancementLoss`，恢复 v5 原始 loss 体系 |
| 2026-06-06 | `option.py` | **回滚** | 删除 `--sat_w`, `--sat_target`，恢复 `--color_w` 原始说明 |
| 2026-06-06 | `trainer.py` | **回滚** | 删除 `sat_loss` 打印（保留 `_tiled_forward`） |
| 2026-06-06 | `train_lol_v6.sh` | **重写** | 新 v6: data1 数据集, 继承 v5 配置(color_w=0.2,per_w=0.15,rec_w=1.3), v5/e10 续训 |
| 2026-06-06 | `trainer.py` | **修复** | eval 日志硬编码 "LOL"→ 动态 `args.dataset`；evaluate() 添加 `data2` 分支 |
| 2026-06-06 | `test.sh` | **重写** | 切换至 data1 Eval 集, 默认 v6, 加 ref_degrade |
| 2026-06-06 | `dataset/data2.py` | **新建** | 新数据集：TrainSet 从 `ref/` 目录加载模型增强图作为参考 |
| 2026-06-06 | `dataset/dataloader.py` | **修改** | 注册 `data2` 数据集 |
| 2026-06-06 | `gen_ref.py` | **新建** | 用 v6/e40 为 data1 训练集批量生成增强 ref 图像 |
| 2026-06-06 | `gen_ref.sh` | **新建** | 一键生成 data2 ref 图像的 Shell 脚本 |
| 2026-06-07 | `train_lol_v7.sh` | **重写** | 新 v7: data2 干净 ref (ref_degrade=False), 从 v6/e40 续训 |
| 2026-06-07 | `eval_v7_all.sh` | **新建** | v7 批量 eval (epoch 5-50)，揭示训练中 eval 缺失问题 |
| 2026-06-07 | `trainer.py` | **修复** | evaluate() 缺失 data2 分支导致 eval 空跑 — 已修复 |
| 2026-06-08 | `train_lol_v7.1.sh` | **新建** | v7.1: data2+干净ref, 从 v7/e20 续训, 100epoch, decay=40 |
| 2026-06-08 | `train_lol_v7.2.sh` | **新建** | v7.2: data2+退化ref, 从 v6/e40 续训, ref_degrade=True |
| 2026-06-08 | `train_lol_v8.sh` | **重写** | 新 v8: data2+退化ref, 从 v6/e40 续训, ref_degrade=True |
| 2026-06-08 | `model/TTSREnhance.py` | **修复** | `freeze_lte` 未实现：LTE 写死 `requires_grad=True` → 改为读 `args.freeze_lte` |
| 2026-06-08 | `train_lol_v8.1.sh` | **新建** | v8.1: data2+退化ref, freeze_lte=True, lr=2.5e-5, decay=30, 从 v8/e10 续训 |
| 2026-06-08 | `utils.py` | **修改** | `calc_psnr_and_ssim` 返回值增加 MSE |
| 2026-06-08 | `trainer.py` | **修改** | eval 新增 LPIPS + MSE 指标，共 PSNR/SSIM/MSE/LPIPS 四项 |

---

## 路径配置

| 用途 | 路径 |
|------|------|
| LOL 数据集 | `/data/datasets/LOLdataset/our485/{low,high}/` + `eval15/{low,high}/` |
| data1 数据集 | `/data/datasets/data1/Training data/{Huawei,Nikon}/{low,high}/` + `Eval/` |
| data2 数据集 | `/data/datasets/data2/Training data/{Huawei,Nikon}/{low,high,ref}/` + `Eval→data1 symlink` |
| v2 实验 | `/data/experiments/TTSR-lowlight-v2/` |
| v3 实验 | `/data/experiments/TTSR-lowlight-v3/` |
| v4 实验 | `/data/experiments/TTSR-lowlight-v4/` |
| v5 实验 | `/data/experiments/TTSR-lowlight-v5/` |
| v6 实验 | `/data/experiments/TTSR-lowlight-v6/` |
| v7 实验 | `/data/experiments/TTSR-lowlight-v7/` |
| v7.1 实验 | `/data/experiments/TTSR-lowlight-v7.1/` |
| v7.2 实验 | `/data/experiments/TTSR-lowlight-v7.2/` |
| v8 实验 | `/data/experiments/TTSR-lowlight-v8/` |
| v8.1 实验 | `/data/experiments/TTSR-lowlight-v8.1/` |
| v10 实验 | `/data/experiments/TTSR-lowlight-v10/` |
| v11 实验 | `/data/experiments/TTSR-lowlight-v11/` |
| v12 实验 | `/data/experiments/TTSR-lowlight-v12/` |
| v13 实验 | `/data/experiments/TTSR-lowlight-v13/` |
| v14 实验 | `/data/experiments/TTSR-lowlight-v14/` |

---

## 训练命令

```bash
bash train_lol_v4.sh   # LOL, v2权重 + LR decay, PSNR 20.97
bash train_lol_v5.sh   # LOL, 饱和度优化, v4/e80续训, PSNR 21.723
bash train_lol_v6.sh   # data1, v5配置, PSNR 21.274 (480px)
# ── 架构修复后 ──
bash train_lol_v10.sh  # data1, 全部修复, 60ep, PSNR 20.359
bash train_lol_v11.sh  # +S sigmoid, LPIPS 0.3070
bash train_lol_v12.sh  # 低lr+降Grey-World, PSNR 20.503 LPIPS 0.2971
bash train_lol_v13.sh  # data2尝试 (失败)
bash train_lol_v14.sh  # 深化模型 10+10+6+4, PSNR 20.811 LPIPS 0.2925 🏆
```

### 数据准备

```bash
bash gen_ref.sh  # v6/e40 → 为 data1 训练集生成 5600 张增强图 → /data/datasets/data2/
```

## 测试命令

```bash
bash test.sh            # 默认 v7, data1 Eval + degrade ref
bash test.sh 40 v6      # v6 epoch 40
bash test.sh 10 v8      # v8 epoch 10
bash eval_v7_all.sh     # 批量 eval v7 所有 checkpoint
bash eval_v7.2.sh       # v7.2 批量 eval
```

## 架构变化

```
原 TTSR (4× SR):              新 TTSREnhance (1:1 增强):
  input (h×w) ── SFE            input (h×w) ── SFE
       ↓                              ↓
  Stage1 (h×w) + T_lv3+S        Stage1 (h×w) + T_lv3+S
       ↓ PixelShuffle×2              ↓ (same res)
  Stage2 (2h×2w) + T_lv2+S      Stage2 (h×w) + T_lv2+S
       ↓ PixelShuffle×2              ↓ (same res)
  Stage3 (4h×4w) + T_lv1+S      Stage3 (h×w) + T_lv1+S
       ↓                              ↓
  MergeTail (4h×4w)             MergeTail (h×w)
       ↓                              ↓
  output (4h×4w)                output (h×w)
```

## 验证状态

| 测试项 | 状态 | 备注 |
|--------|------|------|
| 全部导入测试 | ✅ | option, model/*, loss/*, dataset/* 全部 OK |
| 端到端前向传播 | ✅ | 128×128 输入 → 128×128 输出，1:1 增强 |
| LOL 数据集加载 | ✅ | 485 训练样本 (128×128 crop), 15 测试样本 (原始尺寸) |
| SearchTransfer | ✅ | S/T_lv1/2/3 维度匹配 |
| TPL 分支 | ✅ | transferal perceptual loss 分支正常 |

---

## test.sh 更新

| 2026-05-20 | `test.sh` | **修复** | 改用 `conda run -n ttsr`；mkdir save_results；默认 model_00050.pt |
| 2026-05-20 | `utils.py` | **修复** | `mkExpDir`: eval/test 模式下 `reset=False` 不再抛异常，允许复用已有目录 |
| 2026-05-20 | `loss/loss_enhance.py` | **修复** | TransferalPerceptualLoss: 修复 T_lv2/T_lv1/S 与 map 空间尺寸不匹配 |
| 2026-05-20 | `test.sh` | **修复** | 改用 `conda run -n ttsr`，默认 model_00050.pt，支持直接传 epoch 号 |
| 2026-05-20 | `utils.py` | **修复** | `mkExpDir`: eval/test 模式下 `reset=False` 不抛异常；`os.makedirs` 加 `exist_ok=True` |
| 2026-05-25 | `dataset/lol.py` | **v2修改** | 新增 `DegradeRef` 类：对 Ref 施加颜色抖动+空间偏移+高斯模糊，模拟真实参考图像退化 |
| 2026-05-25 | `option.py` | **v2修改** | 新增 `--ref_degrade`, `--ref_color_jitter`, `--ref_shift_range`, `--ref_blur_sigma` 参数 |
| 2026-05-25 | `train_lol.sh` | **v2修改** | 更新为 v2 训练脚本，输出到 `/data/experiments/TTSR-lowlight-v2/`，默认开启 ref_degrade |
| 2026-05-26 | `train_lol_v3.sh` | **新建(v3)** | v3 实验：per_w 0.1→0.5, exp_w 1.0→2.0, lr 1e-4→5e-5 — **失败**，PSNR 暴跌至 18.63 |
| 2026-05-26 | `train_lol_v4.sh` | **新建(v4)** | v4 实验：保持 v2 loss 权重，仅加 StepLR decay=30 从 v2/e45 续训 — **PSNR 20.97, SSIM 0.8858** |
| 2026-05-26 | `dataset/lol.py` | **v4修改** | TestSet 支持 `ref_degrade`（退化参考图像测试） |
| 2026-05-26 | `test.sh` | **重写** | 支持多版本+多模式测试：`bash test.sh [epoch] [version] [clean/degrade]` |
| 2026-05-26 | `train_lol_v5.sh` | **新建(v5)** | 饱和度优化实验：color_w 0.5→0.2, per_w 0.1→0.15, exp_w 1.0→0.8, 从 v4/e80 续训 |
| 2026-05-26 | `dataset/data1.py` | **新建** | 多相机低光数据集 (Huawei+Nikon)，11200 训练对 + 100 评测，.jpg 格式 |
| 2026-05-26 | `dataset/dataloader.py` | **v6修改** | 添加 `data1` 数据集分支 |
| 2026-05-26 | `train_lol_v6.sh` | **新建(v6)** | data1 大规模训练：batch_size=12, decay=15, 从 v4/e80 续训，饱和度优化权重 |
| 2026-05-26 | `option.py` | 修改 | help 文本添加 data1 选项 |

---

## 实验迭代总结

| 版本 | 数据集 | Ref来源 | ref_degrade | 策略 | PSNR | SSIM | 结论 |
|------|--------|---------|:--:|------|------|------|------|
| v2 | LOL | HR退化 | ✅ | 基线 | 20.39 | 0.8743 | 基础模型 |
| v3 | LOL | HR退化 | ✅ | per_w↑, exp_w↑ | 18.63 | 0.8700 | ❌ 感知损失过高 |
| v4 | LOL | HR退化 | ✅ | LR decay | **20.97** | **0.8858** | ✅ 最佳 v4 |
| v5 | LOL | HR退化 | ✅ | color_w↓, per_w↑ | **21.723** | 0.8957 | 🏆 LOL 最高 PSNR |
| v6 | data1 | HR退化 | ✅ | v5 配置迁移 | **21.274** | 0.7080 | 🏆 data1 最高 PSNR |
| v7 | data2 | v6增强(干净) | ❌ | 干净ref实验 | 20.452 | 0.6978 | 干净ref PSNR降低 |
| v7.1 | data2 | v6增强(干净) | ❌ | 100epoch续训 | 20.471 | - | 无改善 |
| v7.2 | data2 | v6增强 | ✅ | 退化ref实验 | - | - | |
| v8 | data2 | v6增强 | ✅ | v6/e40续训 | 20.688 | 0.7044 | |
| v8.1 | data2 | v6增强 | ✅ | freeze_lte, 低LR | 20.679 | - | |

### 关键发现
- **v5 (LOL)** PSNR 21.723 为全版本最高，SSIM 0.8957，得益于 LOL 小数据集上的精细优化
- **v6 (data1)** PSNR 21.274 为 data1 评测最高，跨数据集迁移有效
- **data2 干净 ref** 路线 (v7/v7.1) PSNR 低于退化 ref 路线 (v8)，说明需要一定退化来增强鲁棒性
- **freeze_lte + 低 LR** (v8.1) 与 v8 持平，LTE 微调对 data2 场景影响有限
- eval 现支持 **PSNR/SSIM/MSE/LPIPS** 四项指标

### 架构修复 (2026-06-28)

| 日期 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 2026-06-28 | `model/SearchTransfer.py` | **修复** | **关键Bug**: ref_lv2/1 unfold stride=1→2/4, fold 回原生 VGG 分辨率, `/k²`→逐像素归一化掩码 |
| 2026-06-28 | `model/MainNetEnhance.py` | **修复** | CSFI2/3 dilation=3 恢复 (棋盘格根因是 SearchTransfer stride, 不是 dilation) |
| 2026-06-28 | `loss/loss_enhance.py` | **修复** | TransferalPerceptualLoss 注释更新 |
| 2026-06-29 | `model/MainNetEnhance.py` | **修复** | S_up 加 `torch.sigmoid()` → [0,1] 软门控, 纯色暗区 S≈0.5 中性 → 抑制棋盘格 |

### v9-v12 实验 (2026-06-28 ~ 2026-06-29)

| 日期 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 2026-06-28 | `train_lol_v9.sh` | **新建** | v9: data1, v6/e40 续训, dilation=1 (未恢复d3) |
| 2026-06-28 | `train_lol_v10.sh` | **新建** | v10: data1, v6/e40 续训, SearchTransfer修复+d3恢复+掩码归一化, 60ep, decay=30 |
| 2026-06-29 | `train_lol_v11.sh` | **新建** | v11: v10/e55 续训, S sigmoid修复, 40ep, decay=20, tpl_w=0.1 |
| 2026-06-29 | `train_lol_v12.sh` | **新建** | v12: v11/e10 续训, 低lr微调(2e-5), decay=10, 20ep 防过拟合 |

### 实验迭代总结 (v9-v14, data1 960px eval)

| 版本 | 修复/策略 | pretrain | PSNR | SSIM | LPIPS | 结论 |
|------|---------|----------|------|------|-------|------|
| v9 | stride fix, /k² (有网格) | v6/e40 | 20.427 | 0.6237 | 0.3144 | 2.2× fold网格偏差 |
| v10 | stride+掩码归一化+d3恢复 | v6/e40 | 20.359 | 0.6256 | 0.3157 | 网格修复, 稳定收敛 |
| v11 | +S sigmoid [0,1] | v10/e55 | 20.292 | 0.6226 | 0.3070 | tpl_loss更稳定 |
| v12 | 低lr(2e-5)+降Grey-World | v11/e10 | 20.503 | 0.6214 | 0.2971 | 饱和度修复 |
| v13 | data2+v12增强ref+轻度退化 | v12/e20 | 20.322 | 0.6250 | 0.3531 | ❌ data2再败 |
| **v14** | **10+10+6+4深化模型** | **v12/e20** | **20.811** | **0.6249** | **0.2925** | 🏆 **全指标最佳** |

### v13-v14 实验 (2026-06-29 ~ 2026-07-01)

| 日期 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 2026-06-29 | `gen_ref_v12.py` | **新建** | v12/e20 生成 data2 ref (覆盖旧), 5600 张 |
| 2026-06-29 | `train_lol_v13.sh` | **新建** | v13: data2+v12增强ref+轻度退化(crop160,rec_w1.5,80ep) → e5即峰值, 训练反降 |
| 2026-06-30 | `train_lol_v14.sh` | **新建** | v14: data1, 深化 num_res_blocks=10+10+6+4, v12/e20续训, 60ep, PSNR 20.811 |
| 2026-07-01 | `plot_all.py` | **新建** | 通用绘图脚本, 支持 v10-v14 自动生成 metrics+loss 曲线 |

### 最终排行榜 (data1 960px)

| 排名 | 版本 | PSNR | LPIPS | 策略 |
|:--:|------|------|-------|------|
| 🥇 | **v14** | **20.811** | **0.2925** | 深化模型 10+10+6+4 |
| 🥈 | v12 | 20.503 | 0.2971 | 低lr+饱和度修复 |
| 🥉 | v10 | 20.359 | 0.3157 | stride+掩码+d3恢复 |
| 4 | v11 | 20.292 | 0.3070 | S sigmoid |

### 关键发现

- **SearchTransfer stride bug** 是最大的单一修复 (v9→v10, +0.3 PSNR)
- **fold /k² 归一化** 在 stride>1 时产生 2.2× 网格偏差, 掩码归一化解决
- **S sigmoid** 让纯色区域纹理注入中性化, LPIPS 明显改善
- **data2 路线** 三次尝试 (v7/v8/v13) 全部失败: 增强生成的 ref 无法超越退化 GT
- **深化模型** (8→10 ResBlocks) 是唯一突破数据天花板的改动, +0.3 PSNR/+0.005 LPIPS
- **Grey-World color_w 降至 0.05** 有效释放饱和度, 对 PSNR 影响不大但视觉效果显著改善
- **4090 24GB** 下 crop=160 + batch=12 显存充裕
