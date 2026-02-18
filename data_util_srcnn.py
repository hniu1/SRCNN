# data_util_srcnn.py
import numpy as np
import xarray as xr
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def _sanitize_nonfinite_zero(arr, name="array"):
    bad_mask = ~np.isfinite(arr)
    bad_count = int(bad_mask.sum())
    if bad_count == 0:
        return arr
    arr = arr.copy()
    arr[bad_mask] = 0.0
    print(f"[sanitize] {name}: replaced {bad_count} non-finite values with 0.0")
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


def read_data(var, res, year_start, year_end, base_dir: Path, downscale_mode="0p25to0p0416"):
    arrays = []
    for year in range(year_start, year_end + 1):
        fname = _resolve_data_filename(var, year, res, downscale_mode)

        fpath = base_dir / fname
        ds = _open_ds(fpath)
        data_var = var if var in ds else next(iter(ds.data_vars))
        raw = ds[data_var].load().values.astype(np.float32)
        ds.close()
        arr = raw
        arr = _sanitize_nonfinite_zero(arr, name=f"{fpath.name}:{data_var}")
        if var in ("pr", "prcp"):
            arr[arr < 0] = 0.0
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)

def read_elev(tt: int, dir_elev: Path):
    ds = _open_ds(dir_elev / "VICa_DEM_trim.nc")
    dem_name = "DEM" if "DEM" in ds.data_vars else next(iter(ds.data_vars))
    elev2d = ds[dem_name].load().values.astype(np.float32)
    ds.close()
    elev2d = np.squeeze(elev2d)
    elev2d = _sanitize_nonfinite_zero(elev2d, name=f"VICa_DEM_trim.nc:{dem_name}")
    elev2d = np.maximum(elev2d, 0.0)
    return np.tile(elev2d[None, ...], (tt, 1, 1))

def minmax_01(arr):
    if arr.ndim == 3:
        T, Y, X = arr.shape
        flat = arr.reshape(-1, 1)
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        T, Y, X, _ = arr.shape
        flat = arr.reshape(-1, 1)
    else:
        raise ValueError(f"minmax_01 expects [T,Y,X] or [T,Y,X,1], got {arr.shape}")
    flat = _sanitize_nonfinite_zero(flat, name="minmax_01_input")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(flat).reshape(T, Y, X, 1)
    return scaled.astype(np.float32), scaler

def standardize_like(arr, scaler=None):
    if arr.ndim == 3:
        T, Y, X = arr.shape
        flat = arr.reshape(-1, 1)
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        T, Y, X, _ = arr.shape
        flat = arr.reshape(-1, 1)
    else:
        raise ValueError(f"standardize_like expects [T,Y,X] or [T,Y,X,1], got {arr.shape}")
    flat = _sanitize_nonfinite_zero(flat, name="standardize_like_input")
    if scaler is None:
        scaler = StandardScaler()
        flat_s = scaler.fit_transform(flat)
    else:
        flat_s = scaler.transform(flat)
    return flat_s.reshape(T, Y, X, 1).astype(np.float32), scaler
