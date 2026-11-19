# Block-amend training baseline

Engineering baseline for the CSFI-U-Net Stage-A port on LOLv2-Real with
**generated Nano references**. Reproducible start point — not a performance claim.

**Checkpoint schema:** `checkpoint_schema_version = 4`. The model is still
`csfi_lfca_v1`; schema 4 only changes the pack format (RNG is plain values
and CPU `uint8` tensors, not a pickled NumPy object). Schema 3 was the same
model with pickled NumPy RNG. Schema 2 was the IGAB variant (`csfi_igab_v0`).
Older / incomplete / other-variant packs are **rejected** on resume (no
warm-start fallback, no silent migration). See `docs/BLOCK_AMEND_LFCA.md` for
the model identity. The LFCA math, the two insertion points, and the training
recipe below were not changed in the schema-4 engineering closeout.

## What is implemented

| Area | Behavior |
|---|---|
| Init | `fresh` / `lte_only` (LTE.* only) / strict pack load for resume & eval |
| Data | ID-mapped Low/GT/Nano; no GT-as-Ref; paired crop/flip/rot |
| Epoch aug | `set_epoch(epoch)`; seed = SHA256(base_seed\|epoch\|sample_id); `persistent_workers=False` |
| Manifest | Identity hash **v3**. Relative path is the only path source; hash and `imread` use it. Not a content hash |
| Tiled infer | Shared `model/tiled_infer.py`; no silent `apply_*_cfg` |
| Metrics | `rgb_float_clipped_v1` |
| Scheduler | One `StepLR.step()` after each epoch’s optimizer updates |
| Checkpoint | Atomic `tmp_ckpt_*.pt` → `os.replace`; validate → update best → save best (if improved) → save last |
| Resume | One CPU pack load. Model, optimizer, scheduler, then RNG/generators. Locks split identity and config |

## Fixed training recipe (unchanged by the LFCA replacement)

```
backbone=csfi_unet  num_res_blocks=8+8+4+2  n_feats=64
model_variant=csfi_lfca_v1  restoration_block=lfca
restoration_blocks=1,1  restoration_heads=1  enable_lightup=True
use_global_illum=False  use_ref_illum=False
ref_correction=True  freeze_lte=True  freeze_stages=""
query_illum_align=False  texture_fusion=False
stage_mode_lv*=normal
batch_size=4  train_crop_size=160  lr=1e-4
num_epochs=100  StepLR(50, 0.5)
rec_w=1  ms_ssim_w=1  other losses off
seed=42  val_ratio=0.1  split_seed=42
ref_degrade=False
```

## Checkpoint pack (schema 4)

Required fields: `checkpoint_schema_version`, `model_variant`, `model_state_dict`,
`optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `global_step`,
`best_val_psnr`, `best_epoch`, `validation_completed`, `config` (full resolved),
`metric_protocol`, `init_scope`, `run_id`, `split_identity`, `rng`.

`rng.format` is `block_amend_rng_v1`:

- `python_random`: `version` 3, a 625-int MT state, `gauss` null or float.
- `numpy`: `bit_generator_name=MT19937`, `keys` (624 ints), `position`,
  `has_gauss` in {0, 1}, `cached_gaussian`.
- `torch_cpu`, `train_generator`, `val_generator`: 1-D contiguous `torch.uint8`
  on CPU. A wrong dtype is an error; it is not cast into a legal state.
- `torch_cuda`: `null` only when the pack was saved with no CUDA device,
  otherwise one CPU `uint8` tensor per CUDA device.

**Load.** Resume, eval, strict weight load, best-identity check, and LTE-only
import all go through `load_local_checkpoint`. The pack stays on CPU. Model
weights are copied into the already-built module. Optimizer state is applied
with `Optimizer.load_state_dict`: on this PyTorch, Adam momentum follows the
parameter onto its device and `step` stays where Adam stored it (CPU unless
`capturable`). Do not `.cuda()` every optimizer tensor. When `torch.load`
accepts `weights_only`, the default is `weights_only=True` and a failed safe
load is not retried. This machine's PyTorch 1.12.1+cu113 has no `weights_only`
argument, so the named local file is loaded with the ordinary unpickler and
the log says so. That is the installed runtime, not a fallback.

**Save order each epoch:** train → `scheduler.step()` → optional val →
update best if **strictly** better → build pack → save best (if this epoch) →
save last. `last.best_val_psnr` is the true historical best after that epoch.

**Resume order:** read and validate the pack on CPU → build model, optimizer,
scheduler, and the dataloaders → load model and optimizer/scheduler state →
restore RNG and the two generators → set the next epoch → the training loop
creates the next iterator. Nothing after the RNG restore re-initialises the
model or walks the loader. A missing RNG field or generator state is an error;
resume does not reseed. If the saved CUDA RNG list length is not the current
device count, resume stops. It does not drop states and still claim an exact
continuation. The claim is limited to the tested environment and an epoch
boundary. It is not CPU/GPU, cross-version, or mid-epoch bitwise equality.
Fresh training init is unchanged.

**Resume config:** checkpoint config wins; only explicit CLI ops overrides
(`cpu`, `num_gpu`, `num_workers`, `num_epochs` extend, paths/logs). Locked
architecture/loss/split/seed/crop/batch keys cannot be silently defaulted over.
Missing split manifests or a split-identity mismatch → hard fail (no re-split,
no edit of the saved hash). Resume uses the locked manifest. A reference that
appears later does not grow the training set.

**Eval** accepts only a complete pack (`config` + `model_variant` +
`model_state_dict`). A bare `state_dict` is refused before the model is built:
this entry cannot safely rebuild the behavior config from weights alone.
`--mode eval --eval_split val` compares `train_hash`, `val_hash`, `ref_subdir`,
`manifest_hash_version`, and `reference_source` with the checkpoint before any
forward, metric file, or image export. Official Test (`--eval_split test`) has
its own identity and is not required to match the internal val hash. The eval
JSON records the checkpoint path and epoch, `model_variant`, the metric
protocol, and the split identity that was actually used.
`load_lte_only` is not under this pack-only rule; it still loads the LTE
containers it already accepted.

**Compatibility.** New runs write schema 4 / `csfi_lfca_v1`. Schema 3 is
refused with a pointer to `migrate_schema3_checkpoint(src, dst)`. That helper
is opt-in, loads the named local file, writes a new file, and never overwrites
the source. It copies the split identity; it does not recompute or edit the
saved hash. A migrated pack whose manifest hash is still v2 will then fail the
v3 identity check, which is intentional. Schema 2 / IGAB weights stay refused.
There is no automatic migration of files already on disk.

## Manifest identity (v3)

The relative path is the only path fact. A new manifest stores `sample_id`,
canonical relative Low/GT/Ref paths, `dataset_dir`, `ref_subdir`, and
`reference_source=generated_nano`. At read time every absolute path is
`dataset_root + relative path`. Hash and `imread` use that same path. If an
old record has both an absolute path and `*_rel`, they must agree under the
recorded root; a disagreement names the sample, the field, and both paths and
stops. Neither path is chosen because it exists.

An explicit new root is allowed only after that check, and then Low, GT, and
Ref are all rebuilt under that one root.

Hash payload (sorted): hash version `v3`, `sample_id`, role (`train` / `val` /
`test`), source partition, canonical relative paths, `ref_subdir`,
`reference_source`. Internal train and internal val are both roles over the
source Train partition. Official Test is a separate role over source Test.
Same basenames under a different `ref_subdir` change the hash. The identity
hash does not detect a file replaced at the same path. `file_sha256` is a
content fingerprint and is not recomputed every epoch. Manifests are not
rewritten in place, and a checkpoint split hash is not updated to force a match.

Also still checked: unique ids inside each split, no id overlap, files exist,
no unexpected path reuse across samples or splits, Low/GT/Ref unambiguous,
GT is not the reference, both splits non-empty, `drop_last` does not yield
zero train batches.

## Reference coverage

`ALLOW_PARTIAL_COVERAGE` in `train/train_block_amend_csfi_lolv2.sh` is `0` or
`1`. The default `0` keeps the strict check. `1` is the only value that passes
`--allow_partial_coverage`. Any other value exits before Python starts.
Partial mode drops only samples whose generated reference is missing, writes
include/exclude plus a coverage report, and builds train/val from that fixed
include list. Missing GT, ambiguous ids, duplicate ids, and bad paths still
fail. GT is never used as a reference. Preflight uses the same flag. The
counts are whatever the dataset has at the time of the scan; they are not
constants in the script.

A read-only scan on 2026-11-19 of `/root/data/datasets/lol-v2-real` saw Train
`nanobanana_ref_v2` 639 triplets and 50 missing refs, and Test
`nanobanana_ref_v3` 100 triplets and 0 missing. That is an observation, not a
program constant, and it was not a training run.

## Tests (2026-11-19, this working tree on `663dbc3`)

Environment: Python 3.10.20, PyTorch 1.12.1+cu113, 1 CUDA device,
`OMP_NUM_THREADS=8`, `TORCH_HOME=/root/data/cache/torch`. `_StubLTE` replaces
VGG in the block-amend tests (no download). `test_csfi_unet_stage_a.py` uses
the cached real VGG19. `weights_only` is not a parameter of this `torch.load`,
so the safe-unpickler branch was not executed here; a unit test forces the
flag on and checks that a failed `weights_only=True` load is not retried.

```text
python tests/test_block_amend_round3.py    # passed, including a real CUDA resume
python tests/test_block_amend_round2.py    # passed
python tests/test_block_amend_baseline.py  # passed (13/13)
python tests/test_block_amend_lfca.py      # passed
python tests/test_csfi_unet_stage_a.py     # passed (real VGG19)
```

Round 3 covers: NumPy / Python / Torch RNG encode-decode and the next random
draw; CPU generator restore; a CUDA-mapped uint8 state moved back with `.cpu()`;
atomic save of a pack with no NumPy arrays; schema 3 refused and an explicit
migration that leaves the source bytes unchanged; absolute-vs-relative
disagreement rejected; ref-subdir identity change; root migration; missing,
duplicate, and cross-split reuse rejected; partial coverage drops only a
missing ref and still raises on a missing GT; shell `ALLOW_PARTIAL_COVERAGE`
0/1/illegal; internal val identity before forward; Official Test not matched
to the val hash; bare weights and IGAB weights refused before `build_model`;
one real GPU epoch, a rebuilt optimizer step (momentum on CUDA, Adam `step`
on CPU), then resume of the next epoch.

Still covered by the older files: atomic save failure safety, epoch-aware
paired crop, last/best not overwritten by a worse score, LTE-only isolation,
no GT-as-reference, shared tiled inference, LFCA math and shapes, old/new
model identity.

**Not verified:** a formal LOLv2 training run, any real-data PSNR/SSIM, Official
Test of a trained model, `weights_only=True` on a PyTorch that actually has
that argument, cross-machine or cross-version resume, mid-epoch resume, and
multi-worker augmentation equality. The GPU check is one synthetic epoch on
the stub LTE, not a quality result. `test_continuous_vs_resume_mini` still
does not compare final parameters bitwise.

## Commands (not executed as formal runs)

Preflight directories and the formal run directory are different. Strict
preflight will stop while Train references are missing.

```bash
cd /root/projects/TTSR-lowlight-v1-data1SOTA
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ttsr
export TORCH_HOME=/root/data/cache/torch
EXP=/root/data/experiments/TTSR-lowlight-block-amend
DATA=/root/data/datasets/lol-v2-real

# strict preflight
python block_amend_main.py --mode preflight --model_variant csfi_lfca_v1 \
  --dataset_dir "$DATA" --save_dir "$EXP/preflight_strict" --init_scope fresh

# explicit partial preflight (missing refs only)
python block_amend_main.py --mode preflight --model_variant csfi_lfca_v1 \
  --dataset_dir "$DATA" --save_dir "$EXP/preflight_partial" \
  --init_scope fresh --allow_partial_coverage

# strict fresh training
RUN_NAME=csfi_lfca_v1_lolv2_run1 ALLOW_PARTIAL_COVERAGE=0 \
  bash train/train_block_amend_csfi_lolv2.sh

# partial fresh training (existing reference subset; own save_dir)
RUN_NAME=csfi_lfca_v1_lolv2_partial_run1 ALLOW_PARTIAL_COVERAGE=1 \
  bash train/train_block_amend_csfi_lolv2.sh

# resume (locked manifest; does not grow if refs appear later)
python block_amend_main.py --mode resume \
  --dataset_dir "$DATA" \
  --save_dir "$EXP/csfi_lfca_v1_lolv2_partial_run1" \
  --resume_checkpoint "$EXP/csfi_lfca_v1_lolv2_partial_run1/checkpoint_last.pt" \
  --split_dir "$EXP/csfi_lfca_v1_lolv2_partial_run1/splits"

# internal val (must match the checkpoint split identity)
python block_amend_main.py --mode eval --eval_split val \
  --dataset_dir "$DATA" \
  --model_path "$EXP/csfi_lfca_v1_lolv2_partial_run1/checkpoint_best_val_psnr.pt" \
  --split_dir "$EXP/csfi_lfca_v1_lolv2_partial_run1/splits" \
  --save_dir "$EXP/csfi_lfca_v1_lolv2_partial_run1/eval_val"

# Official Test / export (own identity; not compared to internal val)
python block_amend_main.py --mode eval --eval_split test --save_images \
  --dataset_dir "$DATA" --test_ref_subdir nanobanana_ref_v3 \
  --model_path "$EXP/csfi_lfca_v1_lolv2_partial_run1/checkpoint_best_val_psnr.pt" \
  --save_dir "$EXP/csfi_lfca_v1_lolv2_partial_run1/official_test"
```
