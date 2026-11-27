# block-amend / `csfi_lfca_v1` 实验记录

写给后续模型分析用。本文只记录 2026-11-19 至 2026-11-20 在这台机器上实际跑完的训练和评测。数字来自日志和 `eval_official_test.json`，四位小数。完整浮点在对应 JSON 里。解释和结论分开写，不要把「看过官方 Test 之后挑出来的分数」当成没看过测试集的泛化结果。

代码仓库：`/root/projects/TTSR-lowlight-v1-data1SOTA`  
实验根目录：`/root/data/experiments/TTSR-lowlight-block-amend`  
相关实现文档：`docs/BLOCK_AMEND_LFCA.md`、`docs/BLOCK_AMEND_TRAINING_BASELINE.md`  
选型汇总：

- `test_set_selection.json`：100 epoch 内部最佳 + 短训第 1 轮
- `test_set_selection_round2.json`：第 1 轮选出的 epoch 5 + 短训第 2 轮
- `test_set_selection_w91.json`：第 2 轮选出的 epoch 5 + L1:MS-SSIM 改为 9:1 的一轮

硬件：1 张 RTX 4090 24GB。环境 `ttsr`：Python 3.10.20，PyTorch 1.12.1+cu113。这版 `torch.load` 没有 `weights_only`。

---

## 1. 要回答的问题

`csfi_lfca_v1` 是把 CSFI-U-Net 里原来两个 IGAB 位置换成 LFCA 之后的模型。100 epoch 按内部验证 PSNR 训完之后：

1. 这套权重在各自官方 Test 上是多少。
2. 从内部验证最佳权重出发，固定学习率再短训，官方 Test 还能不能涨。
3. 短训的损失仍是 L1 + MS-SSIM，只把权重从 1:1 改成 9:1，还能不能涨。

第 2、3 点的挑选规则是：每个数据集单独比，候选包括「上一阶段已经评过的那个 checkpoint」和本轮新存的 5 个 epoch，取得分最高的官方测试 PSNR，SSIM 只在 PSNR 相同时作第二关键字。这个规则看过官方 Test。

---

## 2. 任务定义

输入是一对已经对齐的 RGB：低光图 Low，以及一张生成的 Nano 参考。输出是与 Low 同分辨率的增强图。监督目标是正常光 GT。GT 不作为参考，也不进入纹理检索。

网络张量范围是 `[-1, 1]`。评测前映射到 `[0, 1]` 再裁剪。分辨率是 1:1，不是原版 TTSR 的 4 倍超分。

三个数据集各自训练、各自选型，checkpoint 不跨数据集使用，也不做权重平均。

---

## 3. 数据

### 3.1 三套划分

| 数据集 | 训练用的 dataset_dir | 训练参考子目录 | 官方 Test 参考子目录 | 有参考的 Train | 缺参考而排除 | 训练 / 内部验证 | 官方 Test |
|---|---|---|---|---|---:|---:|---:|
| LOLv2-Real | `/root/data/datasets/lol-v2-real` | `nanobanana_ref_v2` | `nanobanana_ref_v3` | 639 | 50 | 575 / 64 | 100 |
| 华为 | `.../views/data1_huawei` | `nanobanana_ref` | `nanobanana_ref` | 500 | 1950 | 450 / 50 | 30 |
| 尼康 | `.../views/data1_nikon` | `nanobanana_ref` | `nanobanana_ref` | 500 | 2650 | 450 / 50 | 20 |

`...` 是 `/root/data/experiments/TTSR-lowlight-block-amend`。划分 seed 42，`val_ratio=0.1`，只在「有 Nano 参考的 Train 图」里划分。官方 Test 是数据集自带的 Test 划分，不从 Train 里抽。

清单身份是 manifest hash v3：哈希的是相对路径，不是图像内容。短训复用原 run 的 `splits/split_train.json` 和 `split_val.json`，不重新划分。

锁定身份：

| 数据集 | train_hash | val_hash |
|---|---|---|
| LOLv2-Real | `e1544c996c9cf01d71f8a1155a2cb4099063cb705a1d84c1f4e2ff796174cec7` | `ce3cc4da0634a341d37a1c89da1b5b1dbd6ef3e9106c54802e907216e20e0fc7` |
| 华为 | `571e0f1f7ab3b4796dec44ef50d8dd15f9894ae2930e8fa0c552e943e151ae7f` | `28ff333a6e9be5d4274fc6814692af59728827ae2171512627bf4b50beda53b5` |
| 尼康 | `b967c76b3bff67a9e13a57faf47f8cdedbe29ec8b4b33e81919ee2c255448025` | `45df3f1709642b5428ac7bc37e93c541fe4b5b2e9e31ecdd3c2039c7faaa4ad7` |

### 3.2 data1 为什么用视图

data1 原始目录不是 `Train/Low`、`Train/Normal`、`Train/<ref>`。`/root/data/datasets/data1` 没有被改写。实验目录下是符号链接：

- `views/data1_huawei/Train/Low` → `Training data/Huawei/low`
- `views/data1_huawei/Train/Normal` → `Training data/Huawei/high`
- `views/data1_huawei/Train/nanobanana_ref` → 训练参考
- 尼康同样，Low/Normal 指向 `Training data/Nikon/low` 和 `high`

配对规则：文件名里的子串 `low` 换成 `normal` 得到 GT 名。data1 的文件名是 `1.jpg`、`00001.jpg` 这种，不含 `low`，所以 GT 文件名等于 Low 文件名，靠 `Normal` 目录指向 `high` 来配上。

官方 Test 的参考是 `.png`，Low 是 `.jpg`。代码按同名匹配，所以另外做了 jpg 文件名的符号链接，目标仍是那些 png：

- 华为 30 张：`views/data1_huawei_eval_refs/*.jpg` → `Eval/Huawei/nanobanana_ref/*.png`
- 尼康 20 张：`views/data1_nikon_eval_refs/*.jpg` → `Eval/Nikon/nanobanana_ref_Nikon/*.png`

训练循环只读 Train。官方 Test 只在 `--mode eval --eval_split test` 时打开。

### 3.3 训练时的图像增强

训练裁剪 160×160。同一张图的 Low、参考、GT 使用同一套裁剪、翻转、旋转。每个 epoch 调 `set_epoch`，增强种子由 base seed、epoch 和样本 id 的 SHA256 决定。`persistent_workers=False`。`ref_degrade=False`，参考不做额外退化。

---

## 4. 模型与信息流

模型身份：`model_variant=csfi_lfca_v1`，`restoration_block=lfca`，两个位置的深度都是 1，head 数 1，FFN 扩张 4，低通是固定 5×5 二项核。骨干 `csfi_unet`，残差块 `8+8+4+2`，通道 64，`res_scale=1`。可训练参数约 3,857,588，其中两个 LFCA 栈合计 105,602。这个参数量来自 `tests/test_block_amend_lfca.py` 的测量，记在 `docs/BLOCK_AMEND_LFCA.md`。

本轮基线里这些开关是关的：`use_global_illum`、`use_ref_illum`、`query_illum_align`、`texture_fusion`。因此评测代码虽然把 `apply_illum=True`、`apply_ref_illum=True` 传进前向，对应的全局光照头和参考光照迁移直接返回输入，不改变图像。`enable_lightup=True`，照度底图是开的。`RefCorrection` 开着，16 通道。LTE 冻结。

### 4.1 参考支路

1. `RefCorrection` 作用在参考图 `ref` 和 `refsr` 上。主干预测仍使用原始 Low，不使用亮度对齐后的 query。
2. 冻结的 VGG19 LTE 把 `[0,1]` 图像减 ImageNet 均值、除标准差后切三段：
   - lv1：VGG features[0:2]，全分辨率，64 通道
   - lv2：features[2:7]，含第一次池化，H/2，128 通道
   - lv3：features[7:12]，含第二次池化，H/4，256 通道
3. Query 是低光图的 lv3。Key 是参考的 lv3。Value 是参考的 lv1、lv2、lv3。
4. `SearchTransfer` 在 lv3 上做 3×3 patch 的余弦相似度。每个 Query 位置取相似度最大的一个参考位置，这是硬匹配，不是 softmax 加权。用这个下标去参考特征上 gather：
   - lv3：kernel 3，stride 1，得到 `T_lv3`，256 通道，H/4
   - lv2：kernel 6，stride 2，得到 `T_lv2`，128 通道，H/2
   - lv1：kernel 12，stride 4，得到 `T_lv1`，64 通道，H
5. `S` 是每个位置的最大余弦，形状 `[N,1,H/4,W/4]`。注入时双三次插值到该尺度，再 `sigmoid`，作为门控。它不是概率分布。

### 4.2 主干

低光图先补到高宽都能被 4 整除。

照度估计只产出 `log_gain`。底图：

```
low01 = (low + 1) / 2
Y_base = 2 * (low01 * exp(clamp(log_gain, -1.5, 1.5))) - 1
```

SFE 在补齐后的 Low 上得到 `e0`（H）。两次 stride-2 卷积得到 `e1`（H/2）和 `e2`（H/4）。LFCA 不再读取照度特征；估计器的梯度只从 `log_gain` 这条读出回来。

三个尺度：

| 尺度 | 特征从哪来 | 参考注入 | 后面接什么 |
|---|---|---|---|
| H/4 | `e2` | `sigmoid(S) * conv(cat(feat, T_lv3))`，conv 输入通道 64+256 | 8 个残差块，然后 LFCA |
| H/2 | 上一步上采样后与 `e1` 拼接 | 同样方式注入 `T_lv2`，输入通道 64+128 | CSFI2 跨尺度交换，4 个残差块，然后 LFCA |
| H | 上一步上采样后与 `e0` 拼接 | 注入 `T_lv1`，输入通道 64+64 | CSFI3，2 个残差块。这一级没有 LFCA |

三个注入卷积的 bias 初始化为 0，weight 乘 0.01，让训练开始时注入接近 0。残差块的形式是 `start + tail(resblocks(start))`。

`MergeTail` 把三个尺度合成 RGB 残差。输出是 `Y_base + residual`，再裁回原图高宽。

### 4.3 LFCA 一块里做什么

只吃特征 `F`，形状 `[B,C,H,W]`，不吃照度、参考、纹理或 GT。

```
F_low  = P(F)
F_high = F - F_low
Q, K   = Wq(F_low), Wk(F_low)     # Linear C→C，无 bias
V      = Wv(F)                    # 全频段，不是 F_low
Qn, Kn = L2_normalize(Q, K) along N=H*W，eps=1e-6
A      = softmax(temperature * (Qn @ Kn^T), dim=-1)   # [B, heads, d, d]，d=C/heads
O_attn = Wo(A @ V)                # Wo 有 bias
O_local= DWConv3x3(GELU(DWConv3x3(F_high)))           # 无 bias
X      = F + O_attn + O_local
Y      = X + FFN(LayerNorm_channels(X))
```

`P` 是固定深度可分 5×5 二项核，`k=[1,4,6,4,1]/16`，二维核是外积。stride 1，same 尺寸，reflect padding；某一轴短于 3 像素时该轴改 replicate。它是 buffer，不进 state_dict，不训练。`temperature` 每个 head 一个可学习标量，初值 1。FFN 是 `1×1 C→4C，GELU，DW3×3，GELU，1×1 4C→C`，都无 bias。残差系数固定为 1，没有可学习门，也没有零初始化。堆叠时每一块对自己的输入重新做 `P(F)`。本配置每个位置只堆 1 块。

通道注意力是通道对通道，不是空间注意力。关系矩阵是 `Qn @ Kn^T`。旧 IGAB 用的是 `Kn @ Qn^T`，并且 `V` 会乘照度特征。参数个数与同配置 IGAB 相同，变的是读什么，不是张量形状。

### 4.4 训练步和评测步不一致的地方

训练一个 batch：160 裁剪，整块前向，不切 tile。优化的是下面第 5 节的损失。

内部验证和官方 Test 都走 `evaluate_loader`：全图、`model.eval()`、`tiled_forward`。tile 256，overlap 96，余弦窗。指标是第 6 节的协议，不是训练损失。官方 Test 另外把图像存成 uint8 PNG：先裁剪到 `[0,1]`，再 `rint` 到 0..255。PSNR/SSIM 用的是裁剪后的 float `[0,255]`，不经过这次 uint8 舍入。

---

## 5. 损失

100 epoch 和前两轮短训：

```
loss = rec_w * L1(sr, hr) + ms_ssim_w * (1 - MS-SSIM(sr, hr))
rec_w = 1
ms_ssim_w = 1
```

`sr`、`hr` 都在 `[-1,1]`。L1 直接在这个范围上算。MS-SSIM 先做 `(t+1)/2` 再 clamp 到 `[0,1]`，`data_range=1`。权重是 `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`。crop 的短边 ≤160 时窗口从 11 降到 7。金字塔层数会在空间尺寸不够时减少，剩下的权重再归一化。实现见 `loss/loss_enhance.py` 的 `MSSSIMLoss`。

其余损失权重全是 0：感知、transferal perceptual、对抗、照度平滑、颜色、曝光、照度匹配、参考校正。

第三轮短训只改这两个系数：`rec_w=9`，`ms_ssim_w=1`。损失形式没变。9:1 是标量权重，不是把两项归一化之后的梯度比例。L1 在 `[-1,1]` 上，MS-SSIM 项在 `[0,1]` 的结构相似度上，数值尺度本来就不同。

曾经试过把损失换成 `9 * (-PSNR) + 1 * (1 - 单尺度 SSIM)`。这个试法在 LOLv2-Real 上跑了不到 3 个 epoch 就停了，目录是 `short_ft_r3/`。它不是本文的结果，不要和 `short_ft_r2/` 或 `short_ft_w91/` 混在一起。

---

## 6. 指标协议 `rgb_float_clipped_v1`

对每一张图：

1. `x01 = clip((x_neg11 + 1) / 2, 0, 1)`，预测和 GT 都做。
2. 乘 255，保持 float64，不量化成 uint8。
3. PSNR：`diff = (img1-img2)/255`，`mse = mean(diff^2)`，`PSNR = -10 log10(mse)`。峰值 255，三个通道一起算。`mse<=0` 时函数会返回 inf；实际评测没有出现。
4. SSIM：RGB 三个通道各自做一次单尺度 SSIM，再平均。窗口是 11×11 高斯，`sigma=1.5`，`C1=(0.01*255)^2`，`C2=(0.03*255)^2`，valid 卷积，边界各去掉 5 像素。这不是 Y 通道 SSIM，也不是训练用的 MS-SSIM。

一张数据集的分数是逐张 PSNR、SSIM 的算术平均。`n` 必须等于该官方 Test 的张数：100、30、20。

---

## 7. 100 epoch 训练配方

三个数据集用同一套配方，只换数据和参考子目录。

```
Adam lr=1e-4, beta=(0.9, 0.999), eps=1e-8
batch_size=4, train_crop_size=160, num_workers 按启动脚本
num_epochs=100
StepLR: step_size=50, gamma=0.5
  epoch 1–50 的 lr = 1e-4
  epoch 51–100 的 lr = 5e-5
seed=42, split_seed=42, val_ratio=0.1
init = fresh，LTE 用 ImageNet VGG19 且冻结
每个 epoch 结束：optimizer step 都做完之后 scheduler.step 一次
内部验证每个 epoch 都做
best 只在内部验证 PSNR 严格变大时覆盖
```

checkpoint schema 4。每个 epoch 先存 best（若本 epoch 创新高），再存 `checkpoint_last.pt`。100 epoch 的中间 epoch 没有逐个留下，只剩 best 和 last。

Run 目录：

- `csfi_lfca_v1_lolv2_partial_20261119`
- `csfi_lfca_v1_data1_huawei_partial_20261119`
- `csfi_lfca_v1_data1_nikon_partial_20261119`

顺序是 Real，然后华为，然后尼康。Real 大约 2026-11-19 23:21 开始，尼康大约 2026-11-20 05:07 结束。

### 7.1 内部验证上选出的 best

| 数据集 | best epoch | 内部验证 PSNR / SSIM | 同 run 的 epoch 100 内部验证 |
|---|---:|---|---|
| LOLv2-Real | 100 | 20.5638 / 0.8007 | 就是 best。best 与 last 是同一个 epoch |
| 华为 | 92 | 20.0973 / 0.6554 | 20.0808 / 0.6530，低于 best，没有被选中 |
| 尼康 | 79 | 20.4837 / 0.7117 | 20.1843 / 0.7161。PSNR 更低，SSIM 更高。挑选只看 PSNR，所以 epoch 100 没被选中 |

内部验证 SSIM 和官方测试 SSIM 不能横比：验证集和测试集不是同一批图，Real 的训练参考和测试参考还不是同一个子目录。

### 7.2 把 best 拿到官方 Test 上，没有再挑选

| 数据集 | PSNR | SSIM | n | checkpoint |
|---|---:|---:|---:|---|
| LOLv2-Real | 20.2426 | 0.8303 | 100 | `.../lolv2.../checkpoint_best_val_psnr.pt`，epoch 100 |
| 华为 | 20.2885 | 0.6259 | 30 | `.../huawei.../checkpoint_best_val_psnr.pt`，epoch 92 |
| 尼康 | 16.8231 | 0.4984 | 20 | `.../nikon.../checkpoint_best_val_psnr.pt`，epoch 79 |

JSON：各自 run 下的 `official_test/eval_official_test.json`。

尼康内部验证 20.48，官方测试 16.82，差约 3.66 dB。华为内部验证 20.10，官方测试 20.29，测试略高。Real 内部验证 20.56，官方测试 20.24。三套「内部验证高不高」和「官方测试高不高」不是同一个方向。

---

## 8. 短训协议

三次短训都遵守下面这些，只有起点和 `rec_w` 不同。

- 只加载源 checkpoint 的模型权重。不恢复源 Adam 的动量，不恢复源 RNG，不沿用源 StepLR 的当前学习率。
- 新建 Adam，学习率固定 `1e-5`。StepLR 的 gamma 设为 1，所以 5 个 epoch 内学习率不变。
- 5 个 epoch。划分用源 run 的 `splits/`。
- 每个 epoch 都存 `checkpoint_epoch_XXXX.pt`。
- 短训目录里的 `checkpoint_best_val_psnr.pt` 只在这 5 个 epoch 的内部验证里挑，`best_psnr` 从 -1 重新计。它不是 100 epoch 那个 best，也不能拿来代替官方测试选型。
- 训完后对 5 个 epoch 各自跑官方 Test，`--save_images`。已有评测目录不会重跑。
- 上一阶段那个 checkpoint 的官方测试 JSON 直接复用，不重新推理。

### 8.1 三轮分别从哪开始

| 轮次 | 时间 | 起点 | 损失权重 | 权重目录 | 官方测试目录 | 选型 JSON |
|---|---|---|---|---|---|---|
| 第 1 轮 | 11-20 12:08–12:43 | 100 epoch 的内部验证 best | 1 : 1 | `short_ft_5ep/` | `official_test_short_ft/epoch_XXXX/` | `test_set_selection.json` |
| 第 2 轮 | 11-20 12:46–13:21 | 第 1 轮选出的 epoch 5 | 1 : 1 | `short_ft_r2/` | `official_test_short_ft_r2/` | `test_set_selection_round2.json` |
| 9:1 | 11-20 13:37–14:12 | 第 2 轮选出的 epoch 5 | 9 : 1 | `short_ft_w91/` | `official_test_short_ft_w91/` | `test_set_selection_w91.json` |

三个数据集在每一轮里按 Real → 华为 → 尼康顺序跑。第 1 轮三个数据集选出的都是 epoch 5。第 2 轮三个数据集选出的也是 epoch 5。所以第 2 轮和第 9:1 轮的起点，就是上一轮自己的 `checkpoint_epoch_0005.pt`。

---

## 9. 官方测试的逐 epoch 分数

「起点」是上一阶段选出的 checkpoint，分数来自它已经写好的 JSON。本轮 5 个 epoch 是新推理的。

### 9.1 LOLv2-Real，n=100

| 阶段 | ep1 | ep2 | ep3 | ep4 | ep5 | 本阶段选出 |
|---|---:|---:|---:|---:|---:|---|
| 第 1 轮 PSNR | 21.2372 | 20.5616 | 21.2873 | 20.0473 | 21.2890 | ep5，21.2890 |
| 第 1 轮 SSIM | 0.8358 | 0.8310 | 0.8378 | 0.8263 | 0.8383 | 0.8383 |
| 第 2 轮 PSNR | 21.3084 | 20.6114 | 21.3250 | 20.0938 | 21.3734 | ep5，21.3734 |
| 第 2 轮 SSIM | 0.8370 | 0.8319 | 0.8383 | 0.8271 | 0.8393 | 0.8393 |
| 9:1 PSNR | 21.2320 | 20.6363 | 21.3686 | 19.8263 | 21.3423 | 仍是第 2 轮 ep5，21.3734 |
| 9:1 SSIM | 0.8362 | 0.8327 | 0.8388 | 0.8248 | 0.8382 | 0.8393 |

第 1 轮相对 100 epoch 官方测试 20.2426，选出值 +1.0464。第 2 轮相对第 1 轮 +0.0844。9:1 的五个新 epoch 里最高是 ep3 的 21.3686，比 21.3734 低 0.0048。偶数 epoch 反复掉到 20.05、20.09、19.83，奇数 epoch 回到 21.2–21.37。

### 9.2 华为，n=30

| 阶段 | ep1 | ep2 | ep3 | ep4 | ep5 | 本阶段选出 |
|---|---:|---:|---:|---:|---:|---|
| 第 1 轮 PSNR | 20.3540 | 20.3973 | 20.4219 | 20.4678 | 20.4838 | ep5，20.4838 |
| 第 1 轮 SSIM | 0.6279 | 0.6281 | 0.6284 | 0.6287 | 0.6290 | 0.6290 |
| 第 2 轮 PSNR | 20.4488 | 20.4736 | 20.4780 | 20.5270 | 20.5328 | ep5，20.5328 |
| 第 2 轮 SSIM | 0.6296 | 0.6297 | 0.6297 | 0.6300 | 0.6299 | 0.6299 |
| 9:1 PSNR | 20.4508 | 20.5004 | 20.4958 | 20.5499 | 20.5284 | ep4，20.5499 |
| 9:1 SSIM | 0.6292 | 0.6302 | 0.6296 | 0.6301 | 0.6290 | 0.6301 |

第 1 轮相对 20.2885 是 +0.1953。第 2 轮 +0.0490。9:1 的 ep4 相对第 2 轮 +0.0171。第 2 轮 ep4 的 SSIM 0.6300 高于 ep5 的 0.6299，但 PSNR 是 ep5 更高，所以当时选了 ep5。9:1 的 ep2 SSIM 0.6302 是这轮 SSIM 最高，PSNR 不是。

### 9.3 尼康，n=20

| 阶段 | ep1 | ep2 | ep3 | ep4 | ep5 | 本阶段选出 |
|---|---:|---:|---:|---:|---:|---|
| 第 1 轮 PSNR | 16.9459 | 16.9848 | 17.0252 | 17.0226 | 17.0538 | ep5，17.0538 |
| 第 1 轮 SSIM | 0.5006 | 0.5026 | 0.5004 | 0.4994 | 0.5032 | 0.5032 |
| 第 2 轮 PSNR | 16.9952 | 17.0324 | 17.0451 | 17.0595 | 17.0657 | ep5，17.0657 |
| 第 2 轮 SSIM | 0.5033 | 0.5044 | 0.5017 | 0.5006 | 0.5043 | 0.5043 |
| 9:1 PSNR | 16.9965 | 17.0530 | 17.0369 | 17.0626 | 17.0535 | 仍是第 2 轮 ep5，17.0657 |
| 9:1 SSIM | 0.5037 | 0.5044 | 0.5015 | 0.5006 | 0.5041 | 0.5043 |

第 1 轮相对 16.8231 是 +0.2307。第 2 轮 +0.0119。9:1 最高的新 epoch 是 ep4 的 17.0626，低 0.0032。第 2 轮 ep2 的 SSIM 0.5044 不低于选出 epoch 的 0.5043，PSNR 更低。

### 9.4 目前按官方测试 PSNR 留下的权重

| 数据集 | PSNR / SSIM | 文件 |
|---|---|---|
| LOLv2-Real | 21.3734 / 0.8393 | `csfi_lfca_v1_lolv2_partial_20261119/short_ft_r2/checkpoint_epoch_0005.pt` |
| 华为 | 20.5499 / 0.6301 | `csfi_lfca_v1_data1_huawei_partial_20261119/short_ft_w91/checkpoint_epoch_0004.pt` |
| 尼康 | 17.0657 / 0.5043 | `csfi_lfca_v1_data1_nikon_partial_20261119/short_ft_r2/checkpoint_epoch_0005.pt` |

路径都相对于实验根目录。

---

## 10. 短训期间的内部验证

这些数没有参与上面的选型。短训的 best 文件按这里的 PSNR 写，和官方测试选出的 epoch 经常不是同一个。

### 10.1 LOLv2-Real，内部验证 n=64

| 阶段 | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---:|---:|---:|---:|---:|
| 第 1 轮 | 20.4784 / 0.8069 | 20.4666 / 0.8054 | 20.6459 / 0.8081 | 20.4028 / 0.8062 | 20.5537 / 0.8076 |
| 第 2 轮 | 20.5454 / 0.8082 | 20.4962 / 0.8066 | 20.6871 / 0.8090 | 20.3294 / 0.8057 | 20.5589 / 0.8077 |
| 9:1 | 20.5607 / 0.8081 | 20.4310 / 0.8056 | 20.6031 / 0.8072 | 20.1880 / 0.8038 | 20.5915 / 0.8076 |

100 epoch 的内部最佳是 20.5638。短训内部验证的最高点是第 2 轮 ep3 的 20.6871。9:1 的最高点是 ep3 的 20.6031，低于那个点。官方测试选出的却是第 2 轮 ep5：它的内部验证是 20.5589，不是内部最高。

### 10.2 华为，内部验证 n=50

| 阶段 | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---:|---:|---:|---:|---:|
| 第 1 轮 | 20.0331 / 0.6556 | 20.0471 / 0.6556 | 20.0900 / 0.6565 | 20.1045 / 0.6568 | 20.1318 / 0.6566 |
| 第 2 轮 | 20.0332 / 0.6556 | 20.0476 / 0.6558 | 20.0951 / 0.6572 | 20.0933 / 0.6573 | 20.1463 / 0.6571 |
| 9:1 | 19.9506 / 0.6527 | 20.0138 / 0.6548 | 20.0846 / 0.6571 | 20.0740 / 0.6574 | 20.1288 / 0.6567 |

100 epoch 内部最佳 20.0973。第 1、2 轮的内部验证整体往上，第 2 轮 ep5 到 20.1463。9:1 结束在 20.1288，低于 20.1463。官方测试却选了 9:1 的 ep4，而 ep4 的内部验证只有 20.0740，是这 5 个里倒数第二。

### 10.3 尼康，内部验证 n=50

| 阶段 | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---:|---:|---:|---:|---:|
| 第 1 轮 | 20.3859 / 0.7105 | 20.2211 / 0.7098 | 20.3461 / 0.7118 | 20.3156 / 0.7121 | 20.2213 / 0.7121 |
| 第 2 轮 | 20.2348 / 0.7113 | 20.0080 / 0.7096 | 20.2818 / 0.7125 | 20.1941 / 0.7121 | 20.1847 / 0.7125 |
| 9:1 | 20.1808 / 0.7117 | 19.9198 / 0.7097 | 20.2718 / 0.7130 | 20.0877 / 0.7116 | 20.1539 / 0.7122 |

100 epoch 内部最佳 20.4837。三轮短训的内部验证都落在 19.9198–20.3859，没有回到 20.4837。官方测试却从 16.8231 升到 17.0657。内部验证变差和官方测试变好同时发生。

---

## 11. 可以从这些数里直接读到的事实

下面每条都能在第 7–10 节的表里复核。

1. 第一轮短训的官方测试涨幅最大：Real +1.046，华为 +0.195，尼康 +0.231。第二轮降到 +0.084、+0.049、+0.012。9:1 相对第二轮是 -0.005、+0.017、-0.003。这里的正负是「本轮新 epoch 的最高 PSNR」减去「上一轮选出值」。
2. Real 的官方测试在三轮短训里都是偶数 epoch 掉、奇数 epoch 回到 21.2–21.37。9:1 没有把这个高点抬上去。内部验证也是 ep4 明显更低：第 2 轮 ep4 是 20.3294，9:1 ep4 是 20.1880。
3. 华为是唯一在 9:1 仍有正向官方测试增量的数据集，增量 0.017，小于该轮 ep4 和 ep5 之间的 0.022。选出的 ep4 在内部验证上不是这 5 个 epoch 里最好的。
4. 尼康短训把官方测试从 16.82 抬到 17.07，同时内部验证从 20.48 降到大约 20.1–20.4。再用内部验证挑尼康权重，对不准现在这套官方测试选型。
5. 训练 SSIM 项是多尺度、`[0,1]`、crop 160 上窗口 7。报表 SSIM 是单尺度、`[0,255]`、valid 11×11、三通道平均。两列 SSIM 不是同一个函数，不能解释成「损失里的 SSIM 项等于报表 SSIM」。
6. `rec_w=9` 把 L1 的标量乘了 9。L1 定义在 `[-1,1]`，MS-SSIM 项定义在 clamp 后的 `[0,1]`。权重 9:1 不等于两项梯度的 9:1。

---

## 12. 不要这么用这些数

- 不要把第 9 节的选出值写成「内部验证选择的结果」。内部验证选择只产生了第 7.1 节的三个 best，以及每个短训目录里那个重新计分的 best。
- 不要把三个数据集的绝对 PSNR 当成同一种难度的排名。测试集大小是 100、30、20，场景和参考覆盖率不同。尼康有参考的训练图只占 Train 的 500/3150。
- 不要跨数据集加载 checkpoint。
- 不要把 `short_ft_r3/` 算进结果。那是被中止的另一种损失，只有 Real，而且没有跑完 5 个 epoch，也没有官方测试选型。
- 100 epoch 的中间 checkpoint 已经不在磁盘上。无法补做「100 个 epoch 全部在官方 Test 上挑选」。短训是因为只剩 best 和 last，才从 best 往后另训并逐 epoch 保存。
- Real 的 best 和 last 是同一个 epoch。华为 last 的内部验证 PSNR 低于 best。尼康 last 的内部验证 PSNR 低于 best、SSIM 高于 best。last 没有进入官方测试选型。

---

## 13. 文件索引

实验根目录 `/root/data/experiments/TTSR-lowlight-block-amend/`。

| 要看什么 | 路径 |
|---|---|
| 100 epoch 日志、best、last、划分 | `<run>/block_amend.log`，`<run>/checkpoint_best_val_psnr.pt`，`<run>/checkpoint_last.pt`，`<run>/splits/` |
| 100 epoch best 的官方测试 | `<run>/official_test/eval_official_test.json` |
| 短训权重 | `<run>/short_ft_5ep/`，`short_ft_r2/`，`short_ft_w91/` 下的 `checkpoint_epoch_XXXX.pt` 和 `block_amend.log` |
| 短训官方测试 | `<run>/official_test_short_ft/epoch_XXXX/eval_official_test.json`，以及 `_r2`、`_w91` 同结构 |
| 逐张分数 | 上面 JSON 的 `per_image`。字段有 `id`、`psnr_rgb`、`ssim_rgb`、`frac_lt0`、`frac_gt1`、`raw_min`、`raw_max` |
| 官方测试导出的 uint8 图 | 同目录 `images_official_test/<id>.png` |
| 选型 | 实验根目录的三个 `test_set_selection*.json` |
| 模型前向 | `model/TTSREnhance.py`，`model/CSFIUNet.py`，`model/frequency_mixer.py`，`model/SearchTransfer.py`，`model/LTE.py` |
| 损失 | `loss/loss_enhance.py` 的 `ReconstructionLoss`、`MSSSIMLoss` |
| 指标 | `utils_metrics_block_amend.py`，`utils.py` 的 `calc_psnr_rgb`、`calc_ssim` |
| 训练入口 | `block_amend_main.py` 的 `BASELINE_FIXED`、`train_loop`、`evaluate_loader` |
| 短训入口 | `train/short_ft_from_best.py`，`train/queue_short_ft_test_select.sh` |
