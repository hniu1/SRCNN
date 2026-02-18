import json
import os
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


class StreamingStandardScaler:
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.mean_ = None
        self.scale_ = None

    def partial_fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += x.size
        self.s1 += float(x.sum())
        self.s2 += float((x * x).sum())
        return self

    def finalize(self):
        if self.n == 0:
            raise ValueError("Cannot finalize scaler with zero samples")
        mean = self.s1 / self.n
        var = max(self.s2 / self.n - mean * mean, 0.0)
        std = float(np.sqrt(var) + self.eps)
        self.mean_ = mean
        self.scale_ = std
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x * self.scale_ + self.mean_


class StreamingMinMaxScaler:
    def __init__(self):
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None

    def partial_fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if self.data_min_ is None:
            self.data_min_ = xmin
            self.data_max_ = xmax
        else:
            self.data_min_ = min(self.data_min_, xmin)
            self.data_max_ = max(self.data_max_, xmax)
        return self

    def finalize(self):
        rng = self.data_max_ - self.data_min_
        self.scale_ = rng if rng > 0 else 1.0
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.data_min_) / self.scale_

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x * self.scale_ + self.data_min_


def _sanitize_nonfinite_zero(arr, var=None):
    arr = np.asarray(arr, dtype=np.float32)
    bad_mask = ~np.isfinite(arr)
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        arr = arr.copy()
        arr[bad_mask] = 0.0
    if var in ("pr", "prcp"):
        arr = arr.copy()
        arr[arr < 0] = 0.0
    return arr


def _open_ds(path: Path):
    engines = ["netcdf4", "h5netcdf", "scipy"]
    for eng in engines:
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception:
            continue
    raise RuntimeError(f"Failed to open {path}")


def _resolve_data_filename(var, year, res, downscale_mode):
    if downscale_mode == "1to0p25":
        if res == "low":
            return f"Daymet_ERA5_{var}_{year}_1degto0p25deg.nc"
        if res == "high":
            return f"Daymet_ERA5_{var}_{year}_0p25deg.nc"
    elif downscale_mode == "0p25to0p0416":
        if res == "low":
            return f"Daymet_ERA5_{var}_{year}_0p25degto0p0416deg.nc"
        if res == "high":
            return f"Daymet_ERA5_{var}_{year}_trim.nc"

    raise ValueError(
        f"Unsupported combination: downscale_mode={downscale_mode}, res={res}. "
        "Expected downscale_mode in ['1to0p25','0p25to0p0416'] and res in ['low','high']."
    )


def _read_year_data(base_dir: Path, var: str, year: int, res: str, downscale_mode: str):
    fname = _resolve_data_filename(var, year, res, downscale_mode)
    fpath = base_dir / fname
    ds = _open_ds(fpath)
    data_var = var if var in ds.data_vars else next(iter(ds.data_vars))
    arr = ds[data_var].load().values.astype(np.float32)
    ds.close()
    return _sanitize_nonfinite_zero(arr, var=var), fname


def _iter_year_pairs(base_dir, var, year_start, year_end, downscale_mode):
    for year in range(year_start, year_end + 1):
        lr, lr_name = _read_year_data(base_dir, var, year, "low", downscale_mode)
        hr, hr_name = _read_year_data(base_dir, var, year, "high", downscale_mode)
        t = min(lr.shape[0], hr.shape[0])
        if t <= 0:
            continue
        yield year, lr[:t], hr[:t], lr_name, hr_name


def _read_elev_2d_scaled(dir_elev: Path, downscale_mode: str, target_shape):
    if downscale_mode == "1to0p25":
        candidates = [
            "VICa_DEM_0p25deg.nc",
            "VICa_DEM_0p25deg_fill0.nc",
            "VICa_DEM_trim.nc",
        ]
    else:
        candidates = [
            "VICa_DEM_trim.nc",
            "VICa_DEM_0p0416deg.nc",
            "VICa_DEM_0p0416deg_fill0.nc",
        ]

    tried = []
    for name in candidates:
        path = dir_elev / name
        if not path.exists():
            tried.append(str(path))
            continue
        ds = _open_ds(path)
        dem_name = "DEM" if "DEM" in ds.data_vars else next(iter(ds.data_vars))
        elev2d = ds[dem_name].load().values.astype(np.float32)
        ds.close()
        elev2d = np.squeeze(elev2d)
        elev2d = _sanitize_nonfinite_zero(elev2d)
        elev2d = np.maximum(elev2d, 0.0)
        if elev2d.shape == target_shape:
            e_min = float(np.min(elev2d))
            e_max = float(np.max(elev2d))
            denom = (e_max - e_min) if e_max > e_min else 1.0
            return ((elev2d - e_min) / denom).astype(np.float32), str(path)
        tried.append(f"{path} (shape={elev2d.shape})")

    raise RuntimeError(
        "Could not find a matching DEM grid for target shape "
        f"{target_shape}. Tried: {tried}"
    )


def daymetread_lazy(
    path_output,
    checkpoint_dir,
    base_dir,
    dir_elev,
    var="tmax_dy",
    year_start=1980,
    year_end=2014,
    downscale_mode="0p25to0p0416",
    scaler_type="standard",
    train_fraction=0.8,
):
    path_output = Path(path_output)
    checkpoint_dir = Path(checkpoint_dir)
    base_dir = Path(base_dir)
    dir_elev = Path(dir_elev)

    path_output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Pass 0: infer shapes and total timesteps
    total_t = 0
    lr_hw = None
    hr_hw = None
    sample_files = []
    for year, lr_y, hr_y, lr_name, hr_name in _iter_year_pairs(base_dir, var, year_start, year_end, downscale_mode):
        if lr_hw is None:
            lr_hw = (lr_y.shape[1], lr_y.shape[2])
            hr_hw = (hr_y.shape[1], hr_y.shape[2])
        if (lr_y.shape[1], lr_y.shape[2]) != lr_hw or (hr_y.shape[1], hr_y.shape[2]) != hr_hw:
            raise RuntimeError(
                f"Spatial shape mismatch in year {year}. "
                f"Expected lr={lr_hw}, hr={hr_hw}, got lr={lr_y.shape[1:]}, hr={hr_y.shape[1:]}"
            )
        total_t += lr_y.shape[0]
        if len(sample_files) < 4:
            sample_files.append((year, lr_name, hr_name))

    if total_t == 0:
        raise RuntimeError("No timesteps found for preprocessing")

    n_train = int(train_fraction * total_t)
    n_val = total_t - n_train

    # Pass 1: fit scaler (fit on HR only to preserve existing SRCNN behavior)
    if scaler_type == "standard":
        scaler = StreamingStandardScaler()
    elif scaler_type == "minmax":
        scaler = StreamingMinMaxScaler()
    else:
        raise ValueError("scaler_type must be one of ['standard', 'minmax']")

    for _, _, hr_y, _, _ in _iter_year_pairs(base_dir, var, year_start, year_end, downscale_mode):
        scaler.partial_fit(hr_y)
    scaler.finalize()

    # Save scaler
    with open(checkpoint_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save elevation 2D (scaled) once
    elev2d_scaled, elev_source = _read_elev_2d_scaled(dir_elev, downscale_mode, target_shape=lr_hw)
    np.save(path_output / "elev_lr_scaled_2d.npy", elev2d_scaled.astype(np.float32))

    # Allocate memmaps for unscaled LR/HR
    x_train = np.lib.format.open_memmap(
        path_output / "x_train.npy", mode="w+", dtype=np.float32, shape=(n_train, lr_hw[0], lr_hw[1])
    )
    x_val = np.lib.format.open_memmap(
        path_output / "x_val.npy", mode="w+", dtype=np.float32, shape=(n_val, lr_hw[0], lr_hw[1])
    )
    y_train = np.lib.format.open_memmap(
        path_output / "y_train.npy", mode="w+", dtype=np.float32, shape=(n_train, hr_hw[0], hr_hw[1])
    )
    y_val = np.lib.format.open_memmap(
        path_output / "y_val.npy", mode="w+", dtype=np.float32, shape=(n_val, hr_hw[0], hr_hw[1])
    )

    # Pass 2: stream-write sequential split
    global_t = 0
    train_pos = 0
    val_pos = 0

    for _, lr_y, hr_y, _, _ in _iter_year_pairs(base_dir, var, year_start, year_end, downscale_mode):
        t_local = lr_y.shape[0]
        for k in range(t_local):
            if global_t < n_train:
                x_train[train_pos] = lr_y[k]
                y_train[train_pos] = hr_y[k]
                train_pos += 1
            else:
                x_val[val_pos] = lr_y[k]
                y_val[val_pos] = hr_y[k]
                val_pos += 1
            global_t += 1

        x_train.flush()
        x_val.flush()
        y_train.flush()
        y_val.flush()

    meta = {
        "var": var,
        "year_start": year_start,
        "year_end": year_end,
        "downscale_mode": downscale_mode,
        "total_timesteps": total_t,
        "n_train": n_train,
        "n_val": n_val,
        "lr_hw": [int(lr_hw[0]), int(lr_hw[1])],
        "hr_hw": [int(hr_hw[0]), int(hr_hw[1])],
        "scaler_type": scaler_type,
        "elev_source": elev_source,
        "sample_files": sample_files,
    }
    with open(path_output / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[daymetread_lazy] DONE")
    print(f"[daymetread_lazy] total_t={total_t}, train={n_train}, val={n_val}")


class SRCNNLazyDataset(Dataset):
    def __init__(self, path_output, scaler, split="train", use_elevation=True):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        path_output = Path(path_output)
        self.x = np.load(path_output / f"x_{split}.npy", mmap_mode="r")
        self.y = np.load(path_output / f"y_{split}.npy", mmap_mode="r")
        self.scaler = scaler
        self.use_elevation = use_elevation
        self.elev2d = None
        if use_elevation:
            self.elev2d = np.load(path_output / "elev_lr_scaled_2d.npy", mmap_mode="r").astype(np.float32, copy=False)

    def __len__(self):
        return self.x.shape[0]

    def _scale_2d(self, arr2d):
        flat = arr2d.reshape(-1, 1)
        scaled = self.scaler.transform(flat)
        return scaled.reshape(arr2d.shape).astype(np.float32, copy=False)

    def __getitem__(self, idx):
        lr = self._scale_2d(np.asarray(self.x[idx], dtype=np.float32))
        hr = self._scale_2d(np.asarray(self.y[idx], dtype=np.float32))

        if self.use_elevation:
            x = np.stack([lr, self.elev2d], axis=0)
        else:
            x = lr[None, ...]
        y = hr[None, ...]

        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_scaler(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def cached_files_exist(path_output, checkpoint_dir):
    path_output = Path(path_output)
    checkpoint_dir = Path(checkpoint_dir)
    required = [
        path_output / "x_train.npy",
        path_output / "x_val.npy",
        path_output / "y_train.npy",
        path_output / "y_val.npy",
        path_output / "elev_lr_scaled_2d.npy",
        path_output / "meta.json",
        checkpoint_dir / "scaler.pkl",
    ]
    return all(p.exists() for p in required)
