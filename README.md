# SRCNN on Frontier

This README documents the **current** implementation in `SRCNN_frontier.py` and `srcnn_srun.sh`.

## Overview

`SRCNN_frontier.py` trains a 3-layer SRCNN model with MSE loss for climate downscaling.

Current pipeline behavior:
- Loads all selected years of LR and HR data into memory on rank 0.
- Applies sanitization/scaling in preprocessing helpers from `data_util_srcnn.py`.
- Concatenates LR + elevation as input channels.
- Broadcasts full arrays to all ranks in DDP runs.
- Splits data by time index: 80% train / 20% validation.

## Files

- `SRCNN_frontier.py`: training entry point (DDP-compatible)
- `srcnn_model.py`: SRCNN model definition
- `data_util_srcnn.py`: NetCDF read + scaling utilities
- `srcnn_srun.sh`: SLURM submission script

## Supported Downscaling Modes

Set with `--downscale-mode` (or `DOWNSCALE_MODE` in `srcnn_srun.sh`):

- `0p25to0p0416` (default)
  - low: `Daymet_ERA5_{var}_{year}_0p25degto0p0416deg.nc`
  - high: `Daymet_ERA5_{var}_{year}_trim.nc`

- `1to0p25`
  - low: `Daymet_ERA5_{var}_{year}_1degto0p25deg.nc`
  - high: `Daymet_ERA5_{var}_{year}_0p25deg.nc`

## Data Handling Notes

From `data_util_srcnn.py`:
- Non-finite values (`NaN`/`Inf`) are replaced with `0.0`.
- For precipitation variables (`pr`, `prcp`), negative values are clamped to `0.0`.
- `hr` is standardized (fit scaler on HR), and `lr` is transformed with the same scaler.
- Elevation is min-max scaled to `[0, 1]` and appended as an extra input channel.

## Running with SLURM

Submit with defaults:

```bash
cd /lustre/orion/proj-shared/cli138/7hn/SRCNN
sbatch srcnn_srun.sh
```

Run with another mode:

```bash
cd /lustre/orion/proj-shared/cli138/7hn/SRCNN
DOWNSCALE_MODE=1to0p25 sbatch srcnn_srun.sh
```

## Key CLI Arguments (`SRCNN_frontier.py`)

- `--base-dir`: input NetCDF directory
- `--dir-elev`: DEM directory
- `--exp`: experiment name (checkpoint folder suffix)
- `--downscale-mode`: `1to0p25` or `0p25to0p0416`
- `--var`: variable name (e.g., `tmax_dy`)
- `--year-start`, `--year-end`: inclusive year range
- `--epochs`, `--batch-size`, `--num-workers`
- `--amp`: enable autocast (bfloat16)

## Outputs

Under `./checkpoints_<exp>/`:
- `srcnn_best.pth`: best checkpoint by validation loss
- `loss_history.json`: train/validation history

Training logs are written by SLURM to:
- `logs/srcnn-<jobid>.out`
- `logs/srcnn-<jobid>.err`

## Known Limitation (Current Version)

This version reads full multi-year arrays into memory before training. For very large domains (e.g., all North America), memory can become a bottleneck during preprocessing and rank-0 broadcast.


