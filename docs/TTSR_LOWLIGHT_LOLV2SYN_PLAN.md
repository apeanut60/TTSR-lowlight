# LOLv2-Syn 训练思路（承 Real）

## Real 上已经成立的结论

1. **域内 nano-P2** 是主杠杆（Huawei `00004` → MSE + random 192 + 低 LR + 密 early-stop）。
2. **H1**（冻 P2、Y0 cache、LocalRefine）提供域内 Enc/Match。
3. **H3 延续** = 域内 H1 特征 + **华为 H3@2000 头**，**不要再 FT**；对本域训头只会掉分。
4. 不做：GT-degrade Stage A、exposure、LLFormer 融合、华为 H3 整棵续训、scratch 在线 H3、joint A2。

## Syn 协议（同构）

| 阶段 | 做法 |
|---|---|
| 数据 | Train Low∩Normal∩`nanobanana_ref_v2`（同名配对，n≈739）；Test 100 + `nanobanana_ref_v3` |
| P0 | Huawei `model_00004` 全协议 RGB zero-shot |
| S1 | 域内 P2，lr **2e-6**，eval every 100，patience 3 |
| S2 | 冻 S1-best，cache Y0 → 训 H1 nano（batch 8 / crop 128 / lr 1e-4→5e-5） |
| S3 | H1-best Enc/Match + 华为 H3 头，**零样本组装**；FT 只作负对照 |

Init 仍从华为 `00004` 域重建，不从 Real L1 续（与 Nikon/Real 一致）。
