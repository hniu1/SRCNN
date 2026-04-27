# SRCNN (Frontier, Lazy Data Pipeline)

This SRCNN workflow is upgraded for large-area training (CONUS → North America and beyond) by using:

- **Year-wise streaming preprocessing** (no full multi-year concatenation in RAM)
- **Memmap-backed cached arrays** (`.npy`) for LR/HR train/val
- **Lazy dataset loading** during training (sample-by-sample scaling)

## What changed

- `SRCNN_frontier.py`
  - no longer loads full `X`/`Y` into memory
  - auto-prepares cache on rank 0 if missing (or with `--prepare-data`)
  - trains from `SRCNNLazyDataset` over mmap files

- `dataread_mem_srcnn.py`
  - streaming readers and scaler fitting
  - writes `x_train.npy`, `x_val.npy`, `y_train.npy`, `y_val.npy`
  - writes one elevation grid `elev_lr_scaled_2d.npy`
  - saves `scaler.pkl` and `meta.json`

- `prepare_daymet.py`
  - standalone preprocessing entrypoint (like SRGAN flow)

## Downscaling modes

Choose one mode with `--downscale-mode` (or `DOWNSCALE_MODE` in `srcnn_srun.sh`):

- `1to0p25`
  - low: `Daymet_ERA5_{var}_{year}_1degto0p25deg.nc`
  - high: `Daymet_ERA5_{var}_{year}_0p25deg.nc`

- `0p25to0p0416` (default)
  - low: `Daymet_ERA5_{var}_{year}_0p25degto0p0416deg.nc`
  - high: `Daymet_ERA5_{var}_{year}_trim.nc`

## Preprocess only (optional)

```bash
cd /lustre/orion/proj-shared/cli138/7hn/SRCNN
python3 -u prepare_daymet.py \
  --base-dir /lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data \
  --dir-elev /lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/DEM \
  --exp SRCNN_v1 \
  --var tmax_dy \
  --year-start 1980 \
  --year-end 2014 \
  --downscale-mode 0p25to0p0416 \
  --scaler standard
```

## Train (lazy)

`SRCNN_frontier.py` will reuse existing cache, or build it if missing.

```bash
cd /lustre/orion/proj-shared/cli138/7hn/SRCNN
python3 -u SRCNN_frontier.py \
  --base-dir /lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data \
  --dir-elev /lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/DEM \
  --exp SRCNN_v1 \
  --var tmax_dy \
  --year-start 1980 \
  --year-end 2014 \
  --downscale-mode 0p25to0p0416 \
  --batch-size 8
```

To force cache rebuild:

```bash
python3 -u SRCNN_frontier.py ... --prepare-data
```

## Inference

### Python entrypoint

Use `srcnn_infer.py` to run batched inference from cached lazy data and a trained checkpoint:

```bash
cd /lustre/orion/proj-shared/cli138/7hn/SRCNN
python3 -u srcnn_infer.py \
  --exp SRCNN_v1 \
  --split val \
  --path-output ./output/SRCNN_v1 \
  --checkpoint-dir ./checkpoints_SRCNN_v1 \
  --batch-size 4 \
  --num-workers 0 \
  --output-prefix infer
```

### SLURM helper

You can also submit inference as a batch job:

```bash
cd /lustre/orion/proj-shared/cli138/7hn/SRCNN
sbatch srcnn_infer_srun.sh
```

Override defaults at submit time:

```bash
EXP=SRCNN_v1 SPLIT=val BATCH_SIZE=4 NUM_WORKERS=0 sbatch srcnn_infer_srun.sh
```

Inference writes outputs under `./checkpoints_<exp>/inference/`:

| File | Description |
|---|---|
| `y_{split}_predict_daily_{exp}_{var}.npy` | Model predictions, inverse-transformed, shape `(N, H, W, 1)` float32 |
| `y_{split}_daily_{exp}_{var}.npy` | True HR labels, inverse-transformed, shape `(N, H, W, 1)` float32 |
| `X_{split}_daily_{exp}_{var}.npy` | LR input (first channel), inverse-transformed, shape `(N, H, W, 1)` float32 |
| `{prefix}_metrics_{split}.json` | MSE (scaled space), sample count, file paths |

If `--save-scaled` is passed, it also keeps:

- `{prefix}_pred_scaled_{split}.npy`
- `{prefix}_true_scaled_{split}.npy`

### Spatial resolution at inference

Inference inputs are **always at HR resolution**, regardless of downscale mode. The LR field (`x_{split}.npy`) was already resampled to the HR grid during preprocessing, so all three arrays — LR input, HR target, DEM — share the same spatial dimensions `(H, W)`. The model sees `(2, H, W)` per sample (LR + elevation) and outputs `(1, H, W)`.

### Inference for different downscale modes

`srcnn_infer.py` does not need a `--downscale-mode` flag. It only reads the cached `.npy` files and `scaler.pkl`, which were already built for a specific mode during training. To run inference for each mode, point to the corresponding training outputs:

```bash
# 0p25to0p0416 run
python3 -u srcnn_infer.py \
  --exp SRCNN_0p25to0p0416 \
  --path-output ./output/SRCNN_0p25to0p0416 \
  --checkpoint-dir ./checkpoints_SRCNN_0p25to0p0416

# 1to0p25 run
python3 -u srcnn_infer.py \
  --exp SRCNN_1to0p25 \
  --path-output ./output/SRCNN_1to0p25 \
  --checkpoint-dir ./checkpoints_SRCNN_1to0p25
```

## Cache artifacts

Under `./output/<exp>/`:

- `x_train.npy`, `x_val.npy` (unscaled LR)
- `y_train.npy`, `y_val.npy` (unscaled HR)
- `elev_lr_scaled_2d.npy` (single 2D elevation channel)
- `meta.json`

Under `./checkpoints_<exp>/` (or custom checkpoint dir):

| File | Description |
|---|---|
| `scaler.pkl` | Fitted scaler (used by both training and inference) |
| `srcnn_best.pth` | Best model weights (lowest val loss) |
| `loss_history.json` | Full train/val loss per epoch (JSON) |
| `train_loss_daily_{exp}_{var}.npy` | Train loss per epoch, shape `(epochs,)` float32 |
| `val_loss_daily_{exp}_{var}.npy` | Val loss per epoch, shape `(epochs,)` float32 |
| `time_daily_{exp}_{var}.npy` | Wall-clock seconds per epoch, shape `(epochs,)` float32 |

## Notes

- NaN/Inf values are converted to `0.0` during preprocessing.
- For `pr`/`prcp`, negative values are clamped to `0.0`.
- Scaler is fit on **HR only** (same behavior as previous SRCNN code), then applied to both LR and HR in the lazy dataset.
