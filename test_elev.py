import numpy as np
import xarray as xr

def _open_ds(path: Path):
    engines = ["netcdf4", "h5netcdf", "scipy"]
    for eng in engines:
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception:
            pass
    raise RuntimeError(f"Failed to open {path}")

def read_elev(tt: int, dir_elev: Path):
    ds = _open_ds(dir_elev / "VICa_DEM_trim.nc")
    dem_name = "DEM" if "DEM" in ds.data_vars else next(iter(ds.data_vars))
    elev2d = ds[dem_name].load().values.astype(np.float32)
    ds.close()
    elev2d = np.maximum(elev2d, 0.0)
    return np.tile(elev2d[None, ...], (tt, 1, 1))



elev = read_elev(T, dir_elev)
