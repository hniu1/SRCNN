#!/usr/bin/env python3
import os
import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple

import numpy as np
import xarray as xr
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, PReLU, Activation
from tensorflow.keras.optimizers import Adam

# ----------------------------
# Config
# ----------------------------
EXP_NAME = "SRCNNvMSE_NA_R1"
DIR = Path("../data/")
DIR_ELEV = Path("../DEM/")

VAR_NAME = "tmax_dy"
YEAR_START, YEAR_END = 1980, 1981  # inclusive

BATCH_SIZE = 8
EPOCHS = 2
VAL_PATIENCE = 30
LR = 5e-5  # learning rate for Adam

# Enable mixed precision on Frontier if desired:
USE_MIXED_PRECISION = os.environ.get("USE_MIXED_PRECISION", "0") == "1"

# Reproducibility (best-effort)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Quiet TF logs a bit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ----------------------------
# Utilities
# ----------------------------
class TimingCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.logs = []
    def on_epoch_begin(self, epoch, logs=None):
        self._start = timer()
    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(timer() - self._start)

def _open_ds(path: Path) -> xr.Dataset:
    """
    Open a NetCDF with robust engine fallbacks:
    1) netcdf4  2) h5netcdf  3) scipy
    """
    engines = ["netcdf4", "h5netcdf", "scipy"]
    last_err = None
    for eng in engines:
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to open {path} with engines {engines}. Last error: {last_err}")

def read_data(var: str, deg: float, res: str, base_dir: Path = None) -> np.ndarray:
    """
    Read monthly files for each year in [YEAR_START, YEAR_END] and stack along time.
    Returns 3D array [T, Y, X] (loaded into memory).
    """
    base_dir = base_dir or DIR
    arrays = []
    for year in range(YEAR_START, YEAR_END + 1):
        if res == "low" and abs(deg - 0.25) < 1e-6:
            fname = f"Daymet_ERA5_{var}_{year}_0p25degto0p0416deg.nc"
        else:
            fname = f"Daymet_ERA5_{var}_{year}_trim.nc"
        fpath = base_dir / fname
        if not fpath.exists():
            print(f"Warning: missing {fpath}")
            continue
        ds = _open_ds(fpath)
        if var not in ds:
            # fall back to first data var if name doesn't match
            data_var = next(iter(ds.data_vars))
        else:
            data_var = var
        # load to memory now to close the file cleanly
        arr = ds[data_var].load().values  # [time, y, x]
        ds.close()
        arrays.append(arr)
    if not arrays:
        raise FileNotFoundError("No files found for the requested years/months.")
    return np.concatenate(arrays, axis=0).astype(np.float32)

def read_elev(tt: int) -> np.ndarray:
    """
    Reads DEM and tiles over time -> [T, Y, X], non-negative.
    """
    fpath = DIR_ELEV / "VICa_DEM_trim.nc"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing file: {fpath}")
    ds = _open_ds(fpath)
    # guess DEM variable if not exact
    dem_name = "DEM" if "DEM" in ds.data_vars else next(iter(ds.data_vars))
    elev2d = ds[dem_name].load().values.astype(np.float32)
    ds.close()
    elev2d = np.maximum(elev2d, 0.0)
    return np.tile(elev2d[None, ...], (tt, 1, 1))  # [T, Y, X]

def minmax_01(arr: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Scale [T, Y, X] or [T, Y, X, 1] to [0,1] and return 4D [T,Y,X,1] + scaler.
    """
    if arr.ndim == 3:
        T, Y, X = arr.shape
        flat = arr.reshape(-1, 1)
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        T, Y, X, _ = arr.shape
        flat = arr.reshape(-1, 1)
    else:
        raise ValueError(f"minmax_01 expects [T,Y,X] or [T,Y,X,1], got {arr.shape}")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(flat).reshape(T, Y, X, 1)
    return scaled.astype(np.float32), scaler

def standardize_like(arr: np.ndarray, scaler: StandardScaler = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize to zero mean / unit std. If scaler is None, fit; else transform.
    Returns 4D [T, Y, X, 1] and the scaler used.
    """
    if arr.ndim == 3:
        T, Y, X = arr.shape
        flat = arr.reshape(-1, 1)
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        T, Y, X, _ = arr.shape
        flat = arr.reshape(-1, 1)
    else:
        raise ValueError(f"standardize_like expects [T,Y,X] or [T,Y,X,1], got {arr.shape}")
    if scaler is None:
        scaler = StandardScaler()
        flat_s = scaler.fit_transform(flat)
    else:
        flat_s = scaler.transform(flat)
    return flat_s.reshape(T, Y, X, 1).astype(np.float32), scaler

def build_srcnn(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Simple SRCNN-style 3-layer conv net with PReLU. Linear output (float32).
    """
    layers = [
        Conv2D(64, kernel_size=(9, 9), padding="same", input_shape=input_shape),
        PReLU(),
        Conv2D(32, kernel_size=(1, 1), padding="same"),
        PReLU(),
        Conv2D(1, kernel_size=(5, 5), padding="same"),
        # Keep final dtype float32 so losses/metrics are stable under mixed precision
        Activation("linear", dtype="float32"),
    ]
    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=LR), loss="mse", metrics=["mse"])
    return model

def make_datasets(X, y, batch_size=BATCH_SIZE):
    return (tf.data.Dataset
            .from_tensor_slices((X, y))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

def inverse_write(y_pred_4d: np.ndarray, scaler: StandardScaler, name: str, exp: str):
    """
    y_pred_4d: [T, Y, X, 1]
    """
    T, Y, X, _ = y_pred_4d.shape
    y_flat = y_pred_4d.reshape(-1, 1)
    y_inv = scaler.inverse_transform(y_flat).reshape(T, Y, X, 1)
    np.save(f"./{name}_daily_{exp}_{VAR_NAME}.npy", y_inv.astype(np.float32))

# ----------------------------
# Main
# ----------------------------
def main():
    # Optional mixed precision (works with ROCm builds too)
    if USE_MIXED_PRECISION:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled (global policy = mixed_float16)")

    # GPU: allow memory growth (works on ROCm & CUDA)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print("Detected GPUs:", gpus)

    # Single-node multi-GPU if available
    strategy = tf.distribute.MirroredStrategy()
    print("Replicas in sync:", strategy.num_replicas_in_sync)

    # -------- Data I/O --------
    hr_all = read_data(VAR_NAME, deg=0.0416, res="high")  # [T,Y,X]
    lr_all = read_data(VAR_NAME, deg=0.25,   res="low")   # [T,Y,X]

    T = min(hr_all.shape[0], lr_all.shape[0])
    hr_all = hr_all[:T]
    lr_all = lr_all[:T]

    print("HR raw shape:", hr_all.shape, "min/max:", float(np.min(hr_all)), float(np.max(hr_all)))
    print("LR raw shape:", lr_all.shape, "min/max:", float(np.min(lr_all)), float(np.max(lr_all)))

    # Elevation
    elev_all = read_elev(T).astype(np.float32)  # [T,Y,X]
    elev_scaled, _ = minmax_01(elev_all)
    print("Elevation scaled min/max:", float(elev_scaled.min()), float(elev_scaled.max()))

    # Standardize HR and LR using the SAME scaler (fit on HR, apply to LR)
    hr_scaled, hr_scaler = standardize_like(hr_all, scaler=None)
    lr_scaled, _ = standardize_like(lr_all, scaler=hr_scaler)

    # Inputs: concat(LR, elev) along channels
    X = np.concatenate([lr_scaled, elev_scaled], axis=-1)  # [T,Y,X,2]
    y = hr_scaled                                          # [T,Y,X,1]

    # Split train/val/test (60/20/20 overall)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)

    print("Splits:", X_train.shape, X_val.shape, X_test.shape)

    # tf.data
    train_ds = make_datasets(X_train, y_train)
    val_ds   = make_datasets(X_val,   y_val)
    test_ds  = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Logging / checkpoints
    log_dir = Path("./logs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(f"./best_model_daily_{EXP_NAME}_{VAR_NAME}.keras")

    callbacks = [
        TimingCallback(),
        EarlyStopping(monitor="val_loss", patience=VAL_PATIENCE, restore_best_weights=True),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]

    # Build & train under strategy
    with strategy.scope():
        model = build_srcnn(input_shape=X_train.shape[1:])  # (Y, X, 2)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                        callbacks=callbacks, shuffle=True)

    # Save learning curves and time per epoch
    np.save(f"./train_loss_daily_{EXP_NAME}_{VAR_NAME}.npy",
            np.array(history.history["loss"], dtype=np.float32))
    np.save(f"./val_loss_daily_{EXP_NAME}_{VAR_NAME}.npy",
            np.array(history.history["val_loss"], dtype=np.float32))

    # Grab TimingCallback logs
    epoch_times = next(cb.logs for cb in callbacks if isinstance(cb, TimingCallback))
    np.save(f"./time_daily_{EXP_NAME}_{VAR_NAME}.npy", np.array(epoch_times, dtype=np.float32))

    # Persist models
    model.save(f"./final_model_daily_{EXP_NAME}_{VAR_NAME}.keras")

    # Predict on test and inverse-transform to original units
    y_test_pred = model.predict(test_ds)
    inverse_write(y_test_pred, hr_scaler, "y_test_predict", EXP_NAME)
    inverse_write(y_test,      hr_scaler, "y_test",         EXP_NAME)
    inverse_write(X_test[..., 0:1], hr_scaler, "X_test",    EXP_NAME)  # inverse LR (first channel)

    print("Done.")

if __name__ == "__main__":
    main()

