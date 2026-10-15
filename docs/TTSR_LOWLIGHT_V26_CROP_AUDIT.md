# TTSR-lowlight v2.6 — Crop Sampling Audit

## v2.5 finding: **fixed crop per image ID**

Location: `scripts/train_v25_loss_balance.py`

```46:51:scripts/train_v25_loss_balance.py
def stable_crop_xy(sample_id, seed, H, W, crop):
    dig = hashlib.md5(('%d|%s' % (seed, sample_id)).encode()).hexdigest()
    top = int(dig[:8], 16) % (H - crop + 1)
    left = int(dig[8:16], 16) % (W - crop + 1)
    return top, left
```

`CropSetMD5.__getitem__` always calls `stable_crop_xy(row['sample_id'], self.seed, ...)`.

Consequence: for a given `(seed, sample_id, H, W, crop)`, **every visit of that image uses the identical (top, left)**.  
Example `Huawei/00001.jpg`, seed=42, 720×960, crop=192:

| Visit | top | left |
|---|---:|---:|
| 1 | (fixed) | (fixed) |
| 2 | same | same |
| 3 | same | same |
| 4 | same | same |

This is **not** dynamic random cropping. Spatial coverage per image is a single window for the entire 2000-step run (modulo which images are drawn).

## v2.6 protocol

- Crop size remains **192×192**.
- Each **training step** draws a new `(top, left)` from a precomputed schedule.
- Same image across different steps can (and usually does) get different crops.
- C1 and C2 share one schedule: `manifests/random_crop_s42.csv`.
- No flips/rotations (columns present, always 0).
- Train80 probe for logging still uses the **fixed** md5 crops from v2.5 for apples-to-apples fitting curves.
