# block-amend: LFCA restoration block (`csfi_lfca_v1`)

Replaces the two IGAB positions of the CSFI-U-Net (coarse H/4, middle H/2) with
**LFCA — Low-Frequency Relation Channel Attention**. It keeps input-adaptive
channel-relation modelling and removes "illumination feature element-wise
modulating Value". Engineering change only: no training, no metric, no claim of
improvement in this document.

## Information flow

Before (`csfi_igab_v0`, commit `5777e63`), per block, `illu` = estimator feature
resized to the block's scale:

```
Q, K, V = Wq(F), Wk(F), Wv(F)
A      = softmax(rescale * (Kn @ Qn^T))            # channel x channel
O      = Wo(A @ (V * illu)) + pos_emb(V)           # pos_emb: DW3x3-GELU-DW3x3
X      = F + O ;  Y = X + FFN(LN(X))
```

After (`csfi_lfca_v1`), block input is the feature only:

```
F_low  = P(F)            F_high = F - F_low
Q, K   = Wq(F_low), Wk(F_low)          V = Wv(F)            # V keeps the full band
A      = softmax(temperature * (Qn @ Kn^T), dim=-1)         # [B, heads, d, d], d = C/heads
O_attn = Wo(A @ V)
O_local= DWConv3x3(GELU(DWConv3x3(F_high)))                 # bias-free; replaces pos_emb(V)
X      = F + O_attn + O_local ;  Y = X + FFN(LN_channels(X))
```

- `P`: fixed depthwise 5x5 binomial filter, `k = [1,4,6,4,1]/16`, `K = outer(k,k)`,
  stride 1, same H/W. Non-trainable buffer (not a Parameter, not in the
  state_dict). Reflect padding; an axis shorter than 3 px falls back to
  replicate on that axis (1x1 inputs work). No FFT, no trainable filter.
- `Qn`, `Kn`: L2-normalised along `N = H*W` (`eps = 1e-6`). `temperature`: one
  trainable scalar per head, init 1. Head split: channel `= head*d + j`.
- Linear `Wq/Wk/Wv` (C->C, no bias), `Wo` (C->C, bias). FFN: 1x1 C->4C, GELU,
  DW3x3, GELU, 1x1 4C->C (no bias). Residual coefficients are fixed to 1; no
  learnable gates; nothing in the block is zero-initialised.
- Stacked blocks recompute `P(F)` from their own input.
- Note the relation matrix is `Qn @ Kn^T` (IGAB used `Kn @ Qn^T`).

Code: `model/frequency_mixer.py` (`FixedBinomialLowPass`,
`LowFreqRelationChannelAttention`, `HighFreqLocalBranch`, `ConvFeedForward`,
`LFCABlock`, `LFCA`).

## Where it is wired

`model/CSFIUNet.py`, the only two call sites that changed:

```python
x11 = self._run_rbs(x11, self.RB11, self.conv11_tail)
x11 = self.coarse_freq_block(x11)        # was coarse_igab(x11, illu_h4)
...
x22 = self._run_rbs(x22, self.RB22, self.conv22_tail)
x22 = self.middle_freq_block(x22)        # was middle_igab(x22, illu_h2)
```

The fine (H) stage still has no attention block.

## Not changed

LTE, SearchTransfer, the three reference injections and `sigmoid(S)` gate
(`_inject`), SFE, both downsamplings, skips, CSFI2/CSFI3, residual-block
counts, MergeTail, RefCorrection, loss (L1 + MS-SSIM, weights 1/1), lr / batch /
crop / epochs / split, metric protocol, reference source, tiled inference,
atomic checkpointing, epoch-aware paired crop, manifest lock.

Light-up base is numerically unchanged:
`Y_base = 2 * (low01 * exp(clamp(log_gain, -1.5, 1.5))) - 1`, `Y = Y_base + residual`.
`IlluminationEstimator` moved verbatim to `model/illumination.py` (state_dict
path `MainNet.illumination_estimator.*` unchanged; `model.igab` re-exports it).

Consequences of removing the illumination input from the blocks:

- `encode_low_stem` / `crop_stem` no longer carry `illu_h`, `illu_h2`, `illu_h4`
  (their only readers were the two IGAB calls; tiled inference goes through
  `crop_stem`). `log_gain` stays in the stem.
- The estimator's `conv1` / `depth_conv` are now trained only through the
  `log_gain` readout (zero-initialised `conv2`). With `enable_lightup=False`
  the estimator would receive no gradient at all.

`model/igab.py` is kept as historical code; the backbone neither imports nor
instantiates `IGAB` / `IG_MSA`.

## Parameters

Trainable parameters, same C / heads / depth / FFN expansion (measured by
`tests/test_block_amend_lfca.py`):

| C | heads | depth | IGAB | LFCA |
|---|---|---|---|---|
| 64 | 1 | 1 | 52,801 | 52,801 |
| 64 | 4 | 2 | 105,608 | 105,608 |
| 16 | 1 | 1 | 3,985 | 3,985 |

Per block at C=64, heads=1: Q/K/V 12,288; temperature 1; output projection
4,160; local DW branch 1,152; LayerNorm 128; FFN 35,072. The counts are equal
because every trainable tensor has an IGAB counterpart of the same shape
(`rescale` -> `temperature`, `pos_emb` -> `local`); what changed is what the
tensors read, not how many there are. The low-pass adds 25 buffer values per
block, non-trainable and not saved. Baseline `CSFIUNet` (8+8+4+2, 64 ch):
3,857,588 trainable parameters, of which the two LFCA stacks are 105,602 —
the same totals as the IGAB variant.

With the same seed an `LFCA` stack and an `IGAB` stack initialise to identical
values tensor-for-tensor (tested at block level only, not for the whole model).

## Model identity and checkpoints

Config (in `BASELINE_FIXED`, resolved config, `run_meta.json`, checkpoint):

```
model_variant=csfi_lfca_v1  restoration_block=lfca
restoration_blocks=1,1  restoration_heads=1  restoration_ffn_expansion=4
lowpass_kernel=binomial  lowpass_kernel_size=5
```

- `checkpoint_schema_version = 4`. Packs still carry a top-level
  `model_variant`. Schema 4 is the checkpoint format (safe RNG encoding, CPU
  pack load). It is not an architecture change. Schema 3 packs of this same
  model are refused until an explicit `migrate_schema3_checkpoint(src, dst)`;
  that writes a new file and does not edit the saved split hash. All identity
  keys stay locked on resume. Operational rules (RNG, manifest v3, eval
  pack-only, coverage) are in `docs/BLOCK_AMEND_TRAINING_BASELINE.md`.
- `--igab_blocks` / `--igab_heads` survive only as deprecated CLI aliases of
  `--restoration_blocks` / `--restoration_heads`: they warn, an explicit
  conflict is an error, and resolved configs only contain the neutral names.
- **IGAB-variant checkpoints are incompatible.** State keys differ
  (`MainNet.coarse_igab.*` vs `MainNet.coarse_freq_block.*`) and the tensors
  mean different things even though their shapes match. They are refused, with
  a message naming both variants, by: `--mode resume`, `--mode eval` (pack or
  bare weights), `load_strict_state`, `TTSREnhance.load_pretrained_weights`,
  and the legacy shape-filtered `Trainer.load` when the backbone is CSFI. There
  is no partial / shape-filtered path. Use commit `5777e63` for such files.
- `--init_scope lte_only` still loads `LTE.*` only and leaves the whole backbone
  (LFCA included) at code init. Default remains `fresh`.
- The optimizer needs no change: all LFCA parameters are in the MainNet group
  exactly once; no IGAB parameters remain.

Also fixed while wiring this (present at `5777e63`, found by the new
main()-level test): `--mode resume` / `--mode eval` from a checkpoint pack
crashed with `AttributeError: ... 'log_file_name'` unless that option was typed
explicitly, because runtime-only CLI options were absent from the merged
config. `merge_resume_config` now fills such missing keys from the CLI
(fill-only; checkpoint and baseline values are never replaced).

## Tests

Environment: Python 3.10.20, PyTorch 1.12.1+cu113, CPU, `OMP_NUM_THREADS=8`,
`TORCH_HOME=/root/data/cache/torch` (VGG19 weights already cached; nothing was
downloaded). Run on 2026-11-19 on the working tree on top of `5777e63`.

```bash
python tests/test_block_amend_lfca.py      # new, LTE conv stub  -> passed
python tests/test_csfi_unet_stage_a.py     # real VGG19 LTE      -> passed
python tests/test_block_amend_baseline.py  # LTE conv stub       -> passed (13/13)
python tests/test_block_amend_round2.py    # LTE conv stub       -> passed
python block_amend_main.py --mode preflight --model_variant csfi_lfca_v1 --cpu \
  --allow_partial_coverage --save_dir /tmp/lfca_preflight          -> Preflight OK
```

All three pre-existing test files also passed on unmodified `5777e63` before
the change. Edits to them: neutral config names in the tiny args; the IGAB
shape test became an LFCA shape test; `test_resume_rejects` now expects the
model-identity error for a pack with no `model_variant` and keeps the
"Incomplete" check for a pack that declares one.

`tests/test_block_amend_lfca.py` covers: (A) kernel values/sum, buffer not
parameter, size preservation down to 1x1, constant invariance, `F_low + F_high
= F`, impulse response, dtype; (B) heads 1 and 4 against an independent
per-head loop implementation that also fails when the matrix direction is
transposed, attention shape and row sums, Q/K read `F_low` and V reads `F`
(forward hooks), non-divisible channels, zero / 1e-12 inputs; (C) zero in ->
zero out, no bias, block formula, no `pos_emb`, per-block `P(F)`; (D) parameter
counts and optimizer membership; (E) with-ref / no-ref / odd sizes, single tile
= direct forward, multi-tile, both blocks executed with one input each,
historical IGAB classes booby-trapped, light-up formula and "estimator feature
does not reach the restoration path", three L1 + MS-SSIM steps with finite
gradients, frozen LTE, RefCorrection updated; (F) train -> eval -> resume
through `block_amend_main.main()` on a synthetic tree, identity recorded in
pack and `run_meta.json`, eval rebuilt from checkpoint config, locked overrides
and aliases rejected, IGAB-variant pack / bare weights / "relabelled as LFCA"
pack rejected by resume, eval, `load_strict_state`, `load_pretrained_weights`
and `Trainer.load`, LTE-only leaves the backbone untouched.

`tests/test_csfi_unet_stage_a.py` additionally runs one baseline-size
(64 ch, 8+8+4+2) L1 + MS-SSIM step with the real VGG19 LTE on random 160x160
tensors and checks finite gradients.

The schema-4 closeout re-ran these four files and added
`tests/test_block_amend_round3.py`. That file includes one real CUDA resume
of a stub-LTE synthetic run (RNG stayed a CPU `uint8` tensor, then one more
optimizer step). It is an engineering smoke, not a trained-model result.
`weights_only=True` was not executed on this PyTorch 1.12, because
`torch.load` has no such argument.

Not verified:

- Any training on real data, any validation or Official Test number, any
  comparison against the IGAB variant.
- `DataParallel`, multi-worker loaders, mid-epoch resume, and GPU training
  of the real VGG LTE.
- Rejection of a *real* IGAB block-amend checkpoint file: none exists on this
  machine, so the tests use an LFCA pack re-keyed to the IGAB layout.
- Whole-model fresh-init parity with the IGAB variant.
- Stub-LTE results are engineering checks, not real-model validation. Gaps
  noted for round 2 remain (continuous-vs-resumed parameter equality,
  multi-worker augmentation consistency).

A later engineering pass (same model, schema 4) fixed resume RNG device
placement, made the manifest relative path the only path source, checks split
identity on an independent internal val, accepts only a full pack for eval,
and wires `ALLOW_PARTIAL_COVERAGE=0|1` in
`train/train_block_amend_csfi_lolv2.sh` (default 0). None of that changes the
LFCA definition above.

Data, unrelated to LFCA: a read-only scan on 2026-11-19 of
`/root/data/datasets/lol-v2-real` still saw 639 Train triplets and 50 Train
images with no file under `Train/nanobanana_ref_v2`, and 100/0 on Test
`nanobanana_ref_v3`. Those counts are an observation, not constants. Strict
`train` / `preflight` stop while any Train reference is missing. The explicit
subset is `ALLOW_PARTIAL_COVERAGE=1`, which drops only missing references.

## Commands (not executed as formal runs)

The copy-paste set, including strict and partial preflight, lives in
`docs/BLOCK_AMEND_TRAINING_BASELINE.md`. Preflight `save_dir` and the formal
run `save_dir` are different. Short form:

```bash
cd /root/projects/TTSR-lowlight-v1-data1SOTA
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ttsr
export TORCH_HOME=/root/data/cache/torch

# strict fresh training (stops if any Train reference is missing)
RUN_NAME=csfi_lfca_v1_lolv2_run1 ALLOW_PARTIAL_COVERAGE=0 \
  bash train/train_block_amend_csfi_lolv2.sh

# explicit existing-reference subset
RUN_NAME=csfi_lfca_v1_lolv2_partial_run1 ALLOW_PARTIAL_COVERAGE=1 \
  bash train/train_block_amend_csfi_lolv2.sh
```

## How to read this change

- LFCA is an inductive bias to be validated by training, not a result. Nothing
  here says PSNR/SSIM will rise, or that it will not fall.
- It is not a physical Retinex decomposition; the block no longer sees any
  illumination estimate.
- The binomial filter is a soft band split. It does not constrain the output's
  high frequencies, and reference high frequencies still enter through the
  unchanged injections.
- No claim that generation artifacts or negative transfer are removed, nor that
  matching improved (LTE / SearchTransfer are untouched).
- No novelty claim. Sources: IGAB / IG-MSA and the estimator come from
  RetinexFormer via the ASRT Stage-A port; transposed channel attention with
  L2-normalised Q/K and a learnable temperature follows Restormer's MDTA; the
  binomial kernel is the standard `[1,4,6,4,1]` blur. LFCA is a recombination
  of these for this backbone.
