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

## Cache artifacts

Under `./output/<exp>/`:

- `x_train.npy`, `x_val.npy` (unscaled LR)
- `y_train.npy`, `y_val.npy` (unscaled HR)
- `elev_lr_scaled_2d.npy` (single 2D elevation channel)
- `meta.json`

Under `./checkpoints_<exp>/` (or custom checkpoint dir):

- `scaler.pkl`
- `srcnn_best.pth`
- `loss_history.json`

## Notes

- NaN/Inf values are converted to `0.0` during preprocessing.
- For `pr`/`prcp`, negative values are clamped to `0.0`.
- Scaler is fit on **HR only** (same behavior as previous SRCNN code), then applied to both LR and HR in the lazy dataset.
