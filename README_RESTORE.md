# TTSR-lowlight — restored `v1-data1SOTA` (`14d0a3d`)

独立工作树，与当前 `projects/TTSR-lowlight`（`v5`）互不干扰。

| 项 | 值 |
|---|---|
| Git commit | `14d0a3dc30d626cf713c11bfb039dad23ffd8b95` |
| Branch (local) | `restore/v1-data1SOTA` |
| 对应实验 | `/root/data/experiments/TTSR-lowlight-v14` |
| 数据集 | `/root/data/datasets/data1`（LSRW） |
| 结构 | `TTSREnhance`，`num_res_blocks=10+10+6+4`，`n_feats=64` |

## 评测口径（已修正）

1. **Ref**：test 集使用 nanobanana ref（Huawei / Nikon 分相机），`ref_degrade=False`
2. **PSNR**：RGB 三通道（`calc_psnr_rgb`），不以 Y 为主分数
3. **SSIM**：RGB 三通道（R/G/B 各自 SSIM 再平均），不以 Y 为主分数

```bash
cd /root/projects/TTSR-lowlight-v1-data1SOTA
bash test_v14_data1.sh 40
```

## 权重

`/root/data/experiments/TTSR-lowlight-v14/model/model_XXXXX.pt`

## retry/v1-nano-adapt（分相机 nano 微调）

从 v14e40 起训，Huawei / Nikon **分开**用各自 nanobanana ref（manifest 250）。

**v1r2 配方**：冻 LTE；loss=`L1 + MS-SSIM`；其余 aux=0；`lr=1e-5`；crop=192。

```bash
bash train/train_v1r1_nano.sh Huawei
bash test_v1r1_nano.sh Huawei 1
```

实验目录：`/root/data/experiments/retry/TTSR-lowlight-v1r2-nano-{Huawei,Nikon}`  
v14e40 基线 RGB：Huawei 19.471 / Nikon 16.547
