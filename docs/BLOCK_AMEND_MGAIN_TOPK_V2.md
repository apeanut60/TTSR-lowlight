# `csfi_lfca_mgain_topk_v2`

这是在当前 `block-amend` 代码上新加的模型身份，不是对 ep4 波动的修复，也不是一次性能实验。本文只记录结构、配置、工程验收和尚未验证的部分。不把这次改动写成 PSNR 提升。

仓库：`/root/projects/TTSR-lowlight-v1-data1SOTA`  
身份：`model_variant = csfi_lfca_mgain_topk_v2`  
旧身份保持可构建：`csfi_lfca_v1`（局部仿射增益头 + hard top-1）  
checkpoint schema 仍是 4。两种身份靠 `model_variant` 和完整配置区分，没有为换模型升 schema。

## 1. 两处改动和并行关系

输出仍是两条路径相加：

```text
X = Low，坐标 [-1, 1]
I = (X + 1) / 2

log_gain = GainEstimator(I)
base = 2 * (I * exp(clamp(log_gain, -1.5, 1.5))) - 1

rgb_residual = CSFI-U-Net / LFCA(原始 Low, 迁移参考特征)
output = base + rgb_residual
```

改动只有两处：

1. 增益头从局部仿射 `IlluminationEstimator` 换成 `MultiScaleSpatialGainEstimator`（`model/multiscale_gain.py`）。
2. `SearchTransfer` 在 `matching_mode=topk_soft` 时，把 top-1 硬迁移换成 top-k 加权迁移。默认 k=4，temperature=0.1。

调用路径：

- 训练和直接前向：`TTSREnhance.forward` → LTE Query 仍来自原始 Low → `SearchTransfer` → `CSFIUNet.forward` → `encode_low_stem`。
- 分块推理：`tiled_forward` 对 pad 后的整图调用一次 `encode_low_stem`，再用 `crop_stem` 把 `log_gain` 和 `base` 裁进每个 tile。tile 内不再跑增益头。SearchTransfer 的搜索范围仍是当前 tile，没有改。
- `enable_lightup=False` 时 `base = Low`，和旧实现一样，不会另做一套曝光修正。
- 没有参考时，恢复主干仍走无参考分支，增益头照常运行。

SFE 读的是原始 Low，不是 `base`。LTE Query 也读原始 Low。增益头不读 Reference、GT、T 或 S。LFCA 仍只吃特征，不吃照度 embedding，也不把新照度特征乘到 attention Value 上。最终加法和裁回原尺寸的规则没改。模型输出端没有新的训练 clamp。`log_gain` 的截断仍是 [-1.5, 1.5]。

训练看到的是 crop，推理时增益头看到的是 pad 后的整图。这两处曝光上下文不一样。本轮没有把全图缩略图或 crop 坐标送进网络，因此不能说训练和整图推理的曝光上下文相同。

## 2. 增益头

输入 `low01 [B,3,H,W]`，输出 `log_gain [B,3,H,W]`。前向还返回融合后的 1/4 尺度特征 `Zf [B,32,h4,w4]`，调用方只用 `log_gain`，不另造一张全分辨率照度 embedding。

固定起点（不是调出来的最优值）：

| 项 | 值 |
| --- | --- |
| `gain_channels` | 32, 64 |
| `gain_res_blocks` | 2, 2, 1 |
| `gain_attn_heads` | 4 |
| `gain_attn_depth` | 1 |
| `gain_attn_ffn_expansion` | 2 |
| `gain_max_tokens` | 400 |
| `gain_downsample` | `binomial5_stride2` |
| `gain_readout_scale` | 4 |

结构：

```text
M = mean(low01)
I4 = concat(low01, M)                         # 4 通道
I_q = BlurDown2(BlurDown2(I4))                # 实际 1/4 网格
Z4 = GELU(Conv3x3(4→32)) + ResBlock32 × 2
Z8 = GELU(Conv3x3(32→64)(BlurDown2(Z4))) + ResBlock64 × 2
Z8_context = 有 token 上限的空间注意力(Z8)
Zf = GELU(Conv1x1(concat(Z4, bilinear(Z8_context)))) + ResBlock32 × 1
g4 = Conv3x3(32→3)(Zf)                        # 权重和 bias 都是 0
log_gain = bilinear(g4 → low01 的 H,W)
```

残差块是 `Z + Conv3x3(GELU(Conv3x3(Z)))`，没有 BatchNorm、InstanceNorm 或额外门控。

`BlurDown2` 复用 LFCA 的 `FixedBinomialLowPass`：核 `[1,4,6,4,1]/16` 的外积，depthwise，stride-1，再取 `[..., ::2, ::2]`。核是 buffer，不进 optimizer。padding 规则仍是正常尺寸 reflect、过短轴 replicate。这里没有改 LFCA 的数值定义。奇数边用实际下采样结果，不手写 `H//8`。主干仍只 pad 到 4 的倍数，增益分支自己处理奇数粗网格。

所有 bilinear 都是 `align_corners=False`，目标边长用张量的实际形状。

最粗尺度是真正的空间 attention，不是 LFCA 的通道 attention。`Z8` 为 `[B,64,h8,w8]`。`h8*w8 <= 400` 时用原网格，不放大。否则在 `ph<=h8`、`pw<=w8`、`ph*pw<=400` 里按长宽比选一个确定网格，极端长条和单像素轴也守住 400。然后：

```text
P = adaptive_avg_pool2d(Z8, (ph, pw))
A(P) = 一个 pre-LN 空间 Transformer block
DeltaP = A(P) - P
Z8_context = Z8 + bilinear(DeltaP → Z8)
```

不池化时 `Z8_context` 等于 `A(Z8)`，原 `Z8` 保留，不会被池化后再上采样的结果整张换掉。Block 内部：`Z = P + DWConv3x3(P)`；Q/K/V 来自 `LayerNorm(Z)`；4 头，每头 16 维；`scores = QK^T / sqrt(16)`；softmax 在最后一维；FFN 是 64→128→64，中间 GELU。attention 形状是 `[B,4,N,N]`。dropout 为 0。只有这一个 block，而且只在粗尺度、最多 400 个 token 上跑。

最后的 `Conv3x3(32→3)` 权重和 bias 都初始化为 0。因此新建模型 `log_gain = 0`，`exp(log_gain) = 1`，`base` 等于原始 Low。前面的卷积和 attention 用普通初始化，没有整网置零。读出层为零时，第一步前层梯度是 0。这是预期现象。读出层更新之后，前层卷积和空间 attention 才有梯度。验收里用一步 SGD 更新读出层，再反传，确认 `in_conv`、`to_q` 和位置卷积的梯度非零。

这一支提供跨区域上下文。它不是有物理监督的照明分解，低分辨率 gain 也不是严格频带约束。残差网络仍然可以改曝光和颜色。

## 3. Top-k 迁移

`hard_top1` 原样保留，给旧模型和 k=1 回归用。新模式：

```text
matching_mode = topk_soft
match_topk = 4
match_temperature = 0.1
match_query_chunk_size = 128
match_transfer_recompute = True
```

冻结 LTE、lv3 的 3×3 patch、沿 patch 维的 L2 normalize、余弦相似度、参考预处理和搜索范围都没改。没有新的可训练 Q/K、多头匹配、可学习温度、局部窗口、位置惩罚、同坐标强制候选、高低频分权重、新 confidence gate 或候选权重 dropout。`SearchTransfer` 的可训练参数仍是 0。

```text
Query_unfold: [B, D, Nq]
Key_unfold:   [B, Nr, D]
scores = Key_norm @ Query_norm          # [B, Nr, Nq]，不再除 sqrt(D)
k_eff = min(k, Nr)
top_scores, top_indices = topk(scores, k_eff, dim=1)
weights = softmax((top_scores - max_k) / temperature, dim=1)
```

`weights` 在候选维求和为 1。k、temperature、Query/Key/Value 网格不合法时直接报错，不截断、不重排对不上的参考特征。

`S` 仍是 `max_j(scores)`，再 reshape 成 `[B,1,Hq,Wq]`。它不是最大 softmax 权重，不是熵，不是平均候选相似度，也没有除过 temperature。后面的 `sigmoid(S)` 和三处参考注入没改。候选内部的 weights 不是整张参考的可靠性，`S` 也仍然不是校准过的置信度。

三个 Value 尺度共用同一套 lv3 的 `top_indices` 和同一套 `weights`，不在每个尺度重新检索。混合发生在 fold 之前：

```text
lv3: kernel 3, padding 1, stride 1
lv2: kernel 6, padding 2, stride 2
lv1: kernel 12, padding 4, stride 4

mixed_s = sum_r weights[r] * value_patch_s[top_indices[r]]
T_s = fold(mixed_s) / overlap_count_s
```

返回顺序仍是 `S, T_lv3, T_lv2, T_lv1`。覆盖计数只计 Query patch 的空间重叠。实现上用单通道 ones 做 fold，再广播到各通道。和旧的 `fold(ones_like(unfold))` 逐元素一致，恒等 unfold/fold 能还原输入，边缘不会因为计数少一截而变暗。权重不再额外除以 k。

显存上不构造完整的 `[B, D_value, k, Nq]`。检索按 `match_query_chunk_size` 对 Query 分块，每块只对当前可见 Key 打分，留下 top-k 分数、索引和 `S`，不保存完整 `[B, Nr, Nq]`。聚合也按块 gather，一个尺度做完再做下一个尺度。需要梯度时，对这一块的加权 gather 使用 PyTorch 1.12 的 `torch.utils.checkpoint` 重算；输入都不需要梯度，或处于 `eval` / `no_grad` 时不重算。不 detach `top_scores`、weights 或 Value。不把整个 SearchTransfer 放进 `no_grad`，也不靠冻结 RefCorrection 省显存。checkpoint 的输入是这一块自己的张量，反向不会串到最后一块或最后一个尺度。

分块只减小工作集，搜索范围仍是当前可见的全部 Key。这不是把二次搜索变成线性搜索。

`k_eff=1` 用 `torch.max` 取索引，权重为 1，前向和反向与 `hard_top1` 对齐，包括并列时的索引。`k>1` 的并列沿用 `torch.topk` 的规则，不声称跨设备排序相同。top-k 的索引本身不可微；损失能通过 softmax 权重回到被选中的相似度。

`return_diag` 仍在。默认只加小的标量：top-1 原始余弦、最大候选权重、候选熵、`k_eff`、各尺度 `T` 的尺寸。top-k 的 scores/indices/weights 只在显式 `return_match_candidates=True` 时放进诊断，并且是 detach 的。生产前向里的 scores 和 weights 不 detach。增益侧的分位数和 clamp 饱和比例由 `summarize_log_gain` 提供，`collect_gain_diag=False` 时不在每次前向里计算。这些统计不回写温度，也不控制新的 gate。

## 4. 保持不变的部分

LFCA 的数学、RB 数量、SFE、下采样、skip、CSFI2/CSFI3、MergeTail、三处参考注入、`sigmoid(S)` 和 `stage_mode` 都没改。LTE 仍冻结。RefCorrection 仍训练。

训练默认仍是原来的完整配方，不继承 short-ft 的 9:1：

- L1 + MS-SSIM，权重 1:1
- Adam，lr `1e-4`
- batch 4，crop 160
- 100 epoch，`StepLR(50, 0.5)`，seed 42
- 内部 val 的 PSNR 选 best

没有新增 base 单独监督、gain TV、频带损失、匹配损失或可靠性损失。数据划分、参考文件和已有数据视图没改。历史 split 的独立性限制还在，不靠这个模型去改。

训练工程协议也没改：epoch-aware 配对增强、manifest 身份和参考覆盖检查、GT 不进参考或模型输入、atomic checkpoint、RNG / optimizer / scheduler resume、内部 val 选 best、Official Test 单独执行、指标 `rgb_float_clipped_v1`。

## 5. 身份和恢复

配置分支只有两个，没有新的注册表。

- `csfi_lfca_v1`：`gain_estimator=local_affine`，`matching_mode=hard_top1`。旧的 `IlluminationEstimator` 类还在。历史 v1 pack 按自己的配置构建并严格加载，不会在加载后悄悄改用新增益头或 top-k。
- `csfi_lfca_mgain_topk_v2`：`gain_estimator=multiscale_spatial_v1`，`matching_mode=topk_soft`。默认 fresh training。把旧主干做部分匹配加载不算 resume。

下面这些字段进入 resolved config、run metadata、checkpoint 和 eval 重建。top-k 和 temperature 不增加权重，但会改变前向，所以不能只看 state_dict 的形状：

```text
gain_estimator, gain_channels, gain_res_blocks,
gain_attn_heads, gain_attn_depth, gain_attn_ffn_expansion,
gain_max_tokens, gain_downsample, gain_readout_scale,
matching_mode, match_topk, match_temperature,
match_query_chunk_size, match_transfer_recompute
```

LTE-only 初始化只加载 `LTE.*`，不加载旧增益头或其他主干参数。固定低通 buffer 不进 optimizer。新增的可训练增益参数进 MainNet 的 optimizer 组，且只进一次。

v2 启动脚本是 `train/train_block_amend_mgain_topk_v2.sh`。它不覆盖 `train/train_block_amend_csfi_lolv2.sh` 的 v1 实验身份。

## 6. 参数量、环境和验收

环境：Python 3.10.20，PyTorch 1.12.1+cu113。GPU 验收在 1× RTX 4090 24GB 上做。`TORCH_HOME=/root/data/cache/torch`，使用已有的 `vgg19-dcbb9e9d.pth`，这次没有下载权重。

| 模块 | 参数量 |
| --- | --- |
| 旧 `IlluminationEstimator(64)` | 2,179 |
| `MultiScaleSpatialGainEstimator` | 260,963 |
| `CSFIUNet` v1，`n_feats=64`，`8+8+4+2` | 3,857,588 |
| `CSFIUNet` v2，同一主干 | 4,116,372 |
| 增益头替换带来的增量 | 258,784 |
| `SearchTransfer`（两种模式） | 0 |
| `RefCorrection`（16 通道，默认） | 895 |
| LTE | 冻结，训练参数 0 |

v2 默认可训练参数是 MainNet 4,116,372 + RefCorrection 895。token 上限 400。Query chunk 默认 128。

工程验收命令（已执行）：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python tests/test_mgain_topk_v2.py
python tests/test_block_amend_lfca.py
python tests/test_block_amend_round2.py
python tests/test_block_amend_round3.py
python tests/test_csfi_unet_stage_a.py
```

`test_mgain_topk_v2.py`、`test_block_amend_lfca.py`、`test_block_amend_round2.py`、`test_block_amend_round3.py`、`test_csfi_unet_stage_a.py` 均已通过。stage-a 使用缓存中的 VGG19，走的是 v1 冒烟，用来确认旧主干还能构建。v2 单测里的 LTE 是卷积 stub。

v2 单测覆盖：160、奇数、非正方形和小尺寸的输入输出；各级实际尺度；token 不超过 400 且不超过原网格；极端长宽比；无池化时 `Z8_context == A(Z8)`；attention 是 `N×N`；fresh `log_gain` 严格为 0，`base` 与 Low 的差在 1e-5 内。最终 output 不要求等于 Low，因为残差还在。读出层更新后前层梯度接通。dense reference 与分块实现的前向、Query/Key/Value 梯度、重算开关和不同 chunk 一致。覆盖 batch>1、Query/Reference 网格不同、非正方形、k=1、k>Nr、全零输入、非法 temperature、权重和为 1、`S` 等于最大余弦、三尺度共用候选、fold 覆盖计数、k=1 与旧实现的前向和反向、k>1 时 T 的损失能回到 Query。

整体 smoke（LTE 用卷积 stub，不是 VGG）：有参考、无参考、奇数尺寸、单 tile 与直接前向一致、多 tile 只调用一次整图增益头、gain/base 裁剪位置一致、输出尺寸和有限性、损失仍是 L1 + MS-SSIM、RefCorrection 有梯度、LTE 参数 `requires_grad=False`。另有一次合成数据上的 v2 fresh / save / resume / 内部 val eval；v1 配置不会建成 v2；LTE-only 不改新主干的零读出。schema 仍是 4。

GPU 显存检查（已执行，一次，不是训练）：真实缓存 VGG、RefCorrection 开启、batch=4、crop=160、k=4、temperature=0.1、chunk=128、recompute=True。合成输入上做一次 L1 + MS-SSIM 前向和反向。

```text
max_memory_allocated = 2,820,077,056 字节（2,689 MiB）
max_memory_reserved  = 2,977,955,840 字节（2,840 MiB）
```

这次没有 OOM，没有改 chunk、batch、crop、k 或结构。这次 loss 的数值只说明反传跑通，不是质量结论。

单测里的 LTE 是 stub。上面的显存检查才使用缓存里的 VGG19。stub 结果不能当成真实 VGG 的匹配或画质验收。

## 解释边界

这次增加的是增益头的空间表达，以及候选从硬 top-1 变成加权混合。没有证明旧照度头造成了 ep4 的波动。gain 不是有物理监督的真实照明。低分辨率 gain 不是严格频带约束。残差仍能改曝光和颜色。top-k 仍有离散的候选选择。softmax 权重不是整体置信度。混合候选可能把纹理抹平。曝光头不看 Query，所以这里不声称匹配特征变好了。Query 分块不是全量搜索复杂度下降。不宣称首创，也不宣称 PSNR 一定会涨。

沿用的已有部件：CSFI-U-Net、LFCA、`FixedBinomialLowPass`、LTE / VGG19 特征、原来的 unfold-fold 迁移和 `sigmoid(S)` 注入。新文件是 `model/multiscale_gain.py`；`SearchTransfer` 只是在旧 hard top-1 旁边加了 top-k 路径。

## 训练和评测命令

下面的命令只列出，本轮没有执行。不要用 Test 分数选模块、超参数或训练起点。不要把 v2 写进已有 v1 run 目录。

```bash
cd /root/projects/TTSR-lowlight-v1-data1SOTA
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ttsr
export TORCH_HOME=/root/data/cache/torch
EXP=/root/data/experiments/TTSR-lowlight-block-amend
DATA=/root/data/datasets/lol-v2-real
RUN=$EXP/csfi_lfca_mgain_topk_v2_lolv2_run1

# preflight
python block_amend_main.py --mode preflight \
  --model_variant csfi_lfca_mgain_topk_v2 \
  --gain_estimator multiscale_spatial_v1 \
  --matching_mode topk_soft --match_topk 4 --match_temperature 0.1 \
  --match_query_chunk_size 128 --match_transfer_recompute true \
  --dataset_dir "$DATA" --save_dir "$EXP/preflight_mgain_topk_v2" \
  --init_scope fresh

# fresh。脚本自己生成带时间戳的 RUN_NAME；要固定目录就设 RUN_NAME。
RUN_NAME=csfi_lfca_mgain_topk_v2_lolv2_run1 \
  bash train/train_block_amend_mgain_topk_v2.sh

# resume
python block_amend_main.py --mode resume \
  --dataset_dir "$DATA" --save_dir "$RUN" \
  --resume_checkpoint "$RUN/checkpoint_last.pt" \
  --split_dir "$RUN/splits"

# internal val
python block_amend_main.py --mode eval --eval_split val \
  --dataset_dir "$DATA" \
  --model_path "$RUN/checkpoint_best_val_psnr.pt" \
  --split_dir "$RUN/splits" \
  --save_dir "$RUN/eval_val"

# Official Test
python block_amend_main.py --mode eval --eval_split test --save_images \
  --dataset_dir "$DATA" --test_ref_subdir nanobanana_ref_v3 \
  --model_path "$RUN/checkpoint_best_val_psnr.pt" \
  --save_dir "$RUN/official_test"
```

Huawei / Nikon 沿用原来的数据视图和 `ALLOW_PARTIAL_COVERAGE`，换 `DATA`、`TRAIN_REF`、`TEST_REF` 和另一个 `RUN_NAME`。不要写进 v1 的 run 目录，也不要从 short-ft 的 9:1 配方接着训。
