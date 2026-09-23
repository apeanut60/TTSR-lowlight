# Retinexformer V1 —— 改动、验证与结论

> 日期：2026-09-23
> 范围：新增 Retinexformer 主干 + H/4 单点纹理适配器，以及 R0/R1/R2/R3 四组配对实验
> 前置文档：`FINDINGS_reference_branch.md`（参考分支的逐组件诊断）
> 本文件只记录**这一轮**的内容，所有数字均为本仓库实测

---

## 0. 摘要

| 结论 | 状态 |
|---|---|
| 代码改动（新主干 + 适配器 + 5 个开关 + 训练防护） | ✅ 完成，旧路径逐位回归通过 |
| 与官方 Retinexformer 实现的数值一致性 | ✅ wrapper 数据流逐位一致（0.000e+00） |
| 四组配对实验 | ⚠️ 跑完但**结果不可用**，见第 4 节 |
| **根因 1：脚本漏了 `--decay`，LR 全程不衰减** | ✅ 已定位并修复 |
| **根因 2：训练损失从 init 结束就停在平坦区** | ✅ 已定位，**尚未修复** |
| 能站住的结论 | R2 > R3 的**符号**；参考增量很小 |
| 必须撤回的结论 | R2−R3 的**幅度**；所有晚期绝对数值 |

**一句话**：这一轮的工程改动是可信的，但**这四组训练没有回答它想问的问题**——因为训练配方本身有问题（缺 `--decay`，且模型从早期就进入平坦区）。在修好之前重跑是浪费。

---

## 1. 本轮做了什么

### 1.1 新增/修改的代码

| 文件 | 内容 |
|---|---|
| `model/retinexformer_arch.py` **(新)** | 官方实现字节级复制（371 行），加来源/许可证头，去掉 `pdb` import。依赖 torch + einops |
| `model/RetinexRefMainNet.py` **(新)** | `RefTextureAdapter` + `RetinexDenoiser`（在 bottleneck 后插一处）+ `RetinexRefMainNet`（MainNet 契约封装） |
| `option.py` | 新增 `--enhance_backbone {ttsr,retinexformer}`、`--retinex_n_feat`、`--retinex_num_blocks`、`--no_ref_texture`、`--no_global_illum`、`--texture_lv3_only` |
| `model/TTSREnhance.py` | 主干选择；pad 到 4 的倍数并裁回；`use_texture` / `apply_ref_illum` / `apply_illum` 三档独立开关；`inject_lv2/lv1` 透传 |
| `model/MainNetEnhance.py` | `forward` 新增 `inject_lv2` / `inject_lv1`（供 R3 对照） |
| `trainer.py` | `freeze_stages` 空匹配断言；`ref_illum` 组非空断言；参数重复检测；逐组 LR 日志；`save()` 抽出 |
| `main.py` | 训练结束强制保存最后完成的 epoch；`--eval` 也注入生成参考评测 loader |
| `run_retinex_v1.sh` **(新)** | 四组配对运行脚本 |

**环境**：需要 `einops`（原环境没有，已装 0.8.2）。

### 1.2 融合设计

```
low[-1,1] → [0,1] → Illumination_Estimator → Xc → encoder → bottleneck(dim·2^level, H/4)
                                                                    │
low/ref → 冻结 LTE → SearchTransfer → S, T_lv3(256ch, H/4) → A3 ─────┤
                                                                    ▼
                                            decoder → Y01 → [-1,1] → RefIllumTransfer(整图一次) → sr
```

```
A3 = Conv1x1(dim+256 → dim) → GELU → Conv3x3(dim → dim, pad=1)   # 末层零初始化
F3_new = F3 + sigmoid(S) · A3(cat(F3, T_lv3))
```

默认 `n_feat=40`、`num_blocks=[1,2,2]`（与官方 LOL-v1 配置一致），瓶颈 160 通道，故 A3 输入 416。

### 1.3 训练配置

| 项目 | 值 |
|---|---|
| 数据 | **data1 (LSRW)**，5600 训练对（Huawei 2450 + Nikon 3150），50 张评测（HW 30 + NK 20） |
| batch / crop / seed | 8 / 128 / 42 |
| 阶段 | 2 init + 40 常规 epoch |
| LR | `lr_rate 1e-4`、`lr_rate_refillum 1e-4` 显式、LTE 冻结排除 |
| **LR 调度** | **无**（脚本漏传 `--decay`，默认 999999）← 见 4.1 |
| 损失 | rec 1.0 / per 0.1 / illum_smooth 1.0 / color 0.5 / exposure 1.0；tpl=adv=ref_correct=illum_match=0 |
| 关闭 | RefCorrection、GlobalIllumHead |
| 评测 | `data1`（HR 参考，诊断）+ `data1_nanobanana_huawei` / `_nikon`（生成参考，主结果） |
| 耗时 | 约 107 分钟/组，四组约 7 小时 |

---

## 2. 验收清单（不训练的验证）

| 验收项 | 结果 |
|---|---|
| 旧默认 TTSR 路径回归 | ✅ 五个输出**逐位不变**（0.000e+00） |
| 与官方 backbone 数值一致 | ✅ 同 `[0,1]` 输入下 internals **逐位一致**（0.000e+00） |
| — 经 `[-1,1]↔[0,1]` 往返则差 5.376e-05 | 网络对该量级扰动放大 **~420×**（往返误差 1.49e-08）。计划的 `<1e-6` 阈值在此口径下不可达，应改为同标度比较（→0）或放宽到 ~1e-4 |
| 适配器零初始化时开纹理无影响 | ✅ 0.000e+00 |
| 末层零初始化不永久阻断前层 | ✅ 第 0 步 `conv2.grad=5.34e-02`、`conv1.grad=0.000e+00`；**一步后 `conv1.grad=2.21e-02`** |
| `no_reference` 换参考输出不变 | ✅ 0.000e+00（Retinex 估计器仍执行，因为它只用 low） |
| `no_ref_texture` 时 RefIllumTransfer 仍收参考 | ✅ 换参考输出差 7.47e-03 > 0 |
| 非 4 倍数尺寸 | ✅ 255×257 padding 到 256×260，输出裁回 255×257，S/T 网格 63×64 同步裁齐 |
| 网格不匹配时报错 | ✅ 抛 ValueError 并提示需 pad 到 4 的倍数 |
| `freeze_stages` 用错主干报错 | ✅ 新主干 + 旧 stage 名 → RuntimeError；空串 → 正常 no-op |
| 参数组与 LR 记录 | ✅ 初始化时逐组打印 + 每 epoch 打印全部组 |
| 最终 checkpoint 强制落盘 | ✅ 实测 `num_epochs=3, save_every=10` 会保存 `model_00003.pt` |

**实测 optimizer 分组**（R0/R1/R2）：`g1 = 126 tensors`（主干+适配器，1e-4）、`g2 = 6 tensors`（ref_illum，1e-4）。R3 为 4 组。

**未实现**：训练恢复（optimizer / scheduler / global_step / RNG）。`trainer.load()` 只做部分权重加载且仅用于 `--test/--eval`。计划里"能恢复继续训练"是**新需求**，本轮未做。

---

## 3. 四组配对实验的结果

### 3.1 设计

| run | 主干 | 纹理 | 参考光照 | 用途 |
|---|---|---|---|---|
| R0 | retinexformer | 关 | 关 | 无参考基线 |
| R1 | 同 R0 | 关 | 开 | 参考光照增量 |
| R2 | 同 R0 | T_lv3 单点 | 开 | 完整 V1 |
| R3 | **旧 ttsr 主干** | 只注入 T_lv3 | 开 | 主干对照（计划外，我加的） |

R3 的作用：计划把"换主干"和"纹理从三点改单点"绑在一起，R2−R3 才能隔离主干。

### 3.2 结果（PSNR）

**epoch 30**

| 参考设定 | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| HR 参考（50 张，诊断） | 18.144 | 18.783 | 18.857 | 18.478 |
| Nano-Huawei（30 张） | 19.057 | 19.335 | 19.375 | 18.805 |
| Nano-Nikon（20 张） | 16.768 | 16.689 | 16.631 | 16.562 |
| **Nano 加权（主结果）** | **18.141** | **18.277** | **18.277** | **17.908** |

**epoch 40**

| 参考设定 | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| HR 参考 | 18.258 | 18.818 | 18.881 | 17.903 |
| Nano-Huawei | 19.304 | 19.482 | 19.393 | 18.430 |
| Nano-Nikon | 16.682 | 16.598 | 16.811 | 16.484 |
| **Nano 加权（主结果）** | **18.255** | **18.328** | **18.360** | **17.652** |

**增量**

| 对比 | ep30 (HR / Nano加权) | ep40 (HR / Nano加权) |
|---|---|---|
| R1 − R0（参考光照） | +0.639 / +0.135 | +0.560 / +0.073 |
| R2 − R1（纹理） | +0.074 / **+0.001** | +0.063 / +0.032 |
| R2 − R0（完整参考） | +0.713 / +0.136 | +0.623 / +0.105 |
| R2 − R3（主干） | +0.379 / +0.370 | +0.978 / +0.709 |

### 3.3 收敛性（HR 参考，ep10/20/30/40）

| run | 序列 | 最好 |
|---|---|---|
| R0 | 17.734 18.230 18.144 18.258 | ep40 |
| R1 | 18.723 **18.969** 18.783 18.818 | **ep20** |
| R2 | 18.420 18.827 18.857 18.881 | ep40 |
| R3 | 17.902 18.313 **18.478** 17.903 | **ep30** |

**四组里两组非单调**，R3 在最后 10 个 epoch 掉 0.575 dB。

---

## 4. 诊断：为什么这些结果不可用

### 4.1 根因一：脚本漏了 `--decay`（我的 bug）

从日志逐 epoch 打印的 LR 可以确认：

```
R0/R1/R2/R3: 1.000e-04 × 42 个 epoch      ← 一次都没降
```

`option.py` 的 `--decay` 默认值是 **999999**，而 `run_retinex_v1.sh` 没有传。仓库历史约定：

```
--decay 10 : 54 个脚本      --decay 20 : 39 个脚本（最近的 p10-rgmsa 用它）
--decay 30 :  7 个脚本      --decay 999999 / 不传 : 28+14 个
```

**已修**：脚本现在传 `--decay 20 --gamma 0.5`。

### 4.2 根因二：模型从 init 结束就停在平坦区（**未修**）

每 epoch 平均 `rec_loss`：

```
R2:  init ep1/2 = 0.204 / 0.187      ← 全程最好的重建损失在 init 结束时
     常规 epoch = 0.223 0.224 0.219 ... 0.200 0.210    (40 个 epoch 只降 0.018)
R3:  init ep1/2 = 0.266 / 0.220
     常规 epoch = 0.247 0.249 0.227 ... 0.246 0.255
```

**最好的重建损失出现在 init 阶段结束时，之后 40 个 epoch 反而停在更差的位置。** 常规 epoch 一启用辅助损失（`exposure_w 1.0` / `illum_smooth_w 1.0` / `color_w 0.5`，与 `rec_w 1.0` **同权重**），重建立刻变差并再也不恢复。

在这个平坦区里，各 epoch 的评测值相差 ±0.6 dB，**而这个差异不由任何被记录的目标函数解释**。

这条与前置文档的一条未决问题吻合：**真值 GT 自己就违反 exposure 先验**——GT 上的 exposure loss = 0.217，比模型输出的 0.15 还高，却与 rec 同权重。

### 4.3 从 R3 的 ep30 续训 10 epoch（决定性对照）

| | 起点 ep30 | lr 1e-4 @+5 | lr 1e-4 @+10 | lr 1e-5 @+5 | lr 1e-5 @+10 |
|---|---|---|---|---|---|
| HR 参考 | 18.478 | 18.271 | **17.647** | 18.092 | **18.038** |
| Nano-HW | 18.805 | 18.824 | **18.106** | 18.695 | **18.568** |
| Nano-NK | 16.562 | 16.356 | **16.027** | 16.415 | **16.288** |

- **1e-4 复现塌陷**（17.647，与原始 run 的 ep40 = 17.903 同向同量级）
- **1e-5 只把损伤减半**，**没有止住**
- 两个 LR 下三个评测集全部单调下降

→ **降 LR 不是根治手段。**

### 4.4 已排除的假设

| 假设 | 检验 | 结论 |
|---|---|---|
| 权重发散 / 梯度爆炸 | checkpoint 范数：R2 71.4→74.5、R3 55.8→61.1；max\|w\| 恒为 4.464 | ❌ 排除，平滑增长 |
| 训练用退化参考 / 评测用干净参考的不匹配 | 用 `eval_ref_degrade=True`（与训练一致）重评 R3 四个 ckpt：17.951 / 18.252 / 18.452 / 17.971 | ❌ 排除，曲线与干净参考几乎重合（都峰值 ep30、都掉到 ~17.9） |
| 辅助损失把模型带偏 | 逐 epoch 均值（前10 vs 后10）：rec −0.018、per −0.003、exposure **+0.011**、color −0.0006、smooth −0.0003 | ❌ 不成立，没有显著的方向性 |
| 过拟合 | 训练损失全程不降（见 4.2） | ❌ 不像经典过拟合 |

---

## 5. 能站住 / 必须撤回的结论

### 5.1 能站住

1. **R2 > R3 的符号**：ep30 与 ep40 × 三个参考设定，**6/6 为正**。但幅度不可信（见下）。
2. **在 data1 + 生成参考下，整个参考分支的增量很小**：Nano 加权 +0.136（ep30）/ +0.105（ep40）。两个 epoch 一致。
3. **参考质量的影响远大于参考机制**：同样的四组，HR 参考下的增量是 +0.62~+0.71，生成参考下只有 +0.10~+0.14，差 5~6 倍。
4. **工程正确性**（第 2 节全部验收项）。

### 5.2 必须撤回 / 降级

1. **R2 − R3 的幅度**（我上一轮报的 +0.452 / +0.709）。它被 R3 的 ep40 回退放大。保守值是 ep30 的 **+0.37**，但仍有参数量混淆：R3 是旧主干（`8+8+4+2` ResBlock、n_feats=64），参数量远大于 Retinexformer（n_feat=40）。**R2−R3 同时包含"换主干"和"模型变小/更不易训崩"。**
2. **所有以单个 epoch 绝对值做的比较**。四组里两组非单调。
3. **"参考分支 +0.1 dB"不能作为结论**——它低于前置文档实测的种子噪声（0.2~0.8 dB），单种子测不出来。

### 5.3 需要补 caveat 的旧结论

前置文档（`FINDINGS_reference_branch.md`）里的 LOL 消融**同样没有 `--decay`**，也是在这套平坦区里测的：

- **仍然成立**：`tpl` 破坏匹配器（45.1% → 6.9%）——机制性效应，不依赖 PSNR 噪声；`RefIllumTransfer` 的训练价值 +1.06~1.64 dB——远大于种子噪声。
- **需要加 caveat**：`tpl` 的 −0.1 dB、修复组合的 +0.31 dB——**方向可信，效应量可疑**。

---

## 6. 下一步（按性价比）

### A（推荐，最便宜）先做辅助损失消融

把 `exposure_w / illum_smooth_w / color_w` 从 1.0 / 1.0 / 0.5 降到 0.1（或 0），跑 20 个 epoch，看两件事：

- `rec_loss` 是否能在 init 之后**继续下降**（而不是停在 0.20+）
- 评测曲线是否变**单调**

成本：约 46 分钟。**这是唯一能直接验证 4.2 节假设的实验。**

### B 确认 ep30→ep40 的塌陷是否系统性

用 `SEED=43` 重跑 R3 到 40 epoch（约 107 分钟），看是否同样峰值在 ep30。若不同 → 种子噪声主导，加种子无用，必须先修训练。

### C 直接加 `--decay 20` 重跑四组

```bash
bash run_retinex_v1.sh              # 四组约 6.7 h
SKIP_R3=1 bash run_retinex_v1.sh    # 三组约 5.0 h
```

**不建议先做**：4.3 节已证明降 LR 10 倍只把损伤减半，decay 治不了根。

### D 参数量对齐（若要做干净的主干比较）

R3 应把 `n_feats` 降到与 Retinexformer 可比，或反过来加宽 Retinexformer。否则 R2−R3 永远说不清。

---

## 7. 复现方式

### 7.1 训练（data1）

```bash
tmux new -s retinex
cd /root/projects/TTSR-lowlight
EPOCHS=40 bash run_retinex_v1.sh 2>&1 | tee /root/data/experiments/retinex_v1_driver.log
```

旋钮：`EPOCHS`(40) `SEED`(42) `SKIP_R3`(0) `VAL_EVERY`(10) `PRINT_EVERY`(100) `OUT_ROOT` `DATA`

### 7.2 单独评测一个 checkpoint（`--eval` 现在也会注入生成参考 loader）

```bash
python main.py --dataset data1 --dataset_dir /root/data/datasets/data1 \
  --enhance_mode True --enhance_backbone retinexformer \
  --retinex_n_feat 40 --retinex_num_blocks 1,2,2 --freeze_stages= \
  --num_workers 8 --num_gpu 1 --ref_correction False --no_global_illum True --tpl_w 0.0 \
  --eval True --model_path <ckpt> --eval_ref_degrade False --eval_data1 True \
  --save_dir <原实验目录>
```

会一次打出 `data1`（HR 参考）+ `data1_nanobanana_huawei` + `data1_nanobanana_nikon`。

### 7.3 从 checkpoint 续训（本轮用的技巧，不需要改代码）

```bash
python main.py ... --load_pretrain True --pretrain_path <ckpt> \
  --num_init_epochs 0 --num_epochs 10 --lr_rate <LR> --decay 999999 ...
```

`TTSREnhance.load_pretrained_weights` 会加载所有形状匹配的键（实测 `Loaded 204/204`）。

### 7.4 本轮实验产物

```
四组主实验 : /root/data/experiments/retinex_v1_data1_s42/{R0_noref,R1_refillum,R2_refillum_tex3,R3_ttsr_tex3only}/
续训对照   : /root/data/experiments/r3_continue_{lr1e-4,lr1e-5}/
退化参考评测: /root/data/experiments/r3_deg_ep{10,20,30,40}/
驱动日志   : /root/data/experiments/retinex_v1_data1_s42_driver.log
```

---

## 8. 关键数字速查

```
data1 规模          : 5600 训练对 / 50 评测图
单 epoch 耗时       : 137 s (700 batch @ batch8)，瓶颈是磁盘 I/O
并行无用            : 两组并行各自慢 2 倍，聚合吞吐不变
单组总耗时          : 约 107 min (2 init + 40 epoch + 4 次评测)
每次评测            : 约 70 s (100 张，tiled tile=256)

数据流等价性        : 同 [0,1] 输入下 wrapper vs 官方 = 0.000e+00
适配器零初始化      : 开/关纹理输出差 = 0.000e+00
conv1 梯度          : 第 0 步 0.000e+00 → 一步后 2.21e-02

LR 轨迹             : 1.000e-04 × 42 (未衰减)
R3 塌陷             : 18.478 (ep30) → 17.903 (ep40) = -0.575 dB
续训 1e-4 (10ep)    : 18.478 → 17.647
续训 1e-5 (10ep)    : 18.478 → 18.038
退化参考评测        : 17.951/18.252/18.452/17.971  (与干净参考几乎重合)
权重范数            : R2 71.4→74.5, R3 55.8→61.1, max|w|=4.464 恒定
init 结束的 rec_loss: R2 0.187, R3 0.220  ← 全程最低点
```
