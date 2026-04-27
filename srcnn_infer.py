#!/usr/bin/env python

"""
Frontier-ready inference for lazy SRCNN pipeline.

Loads checkpoint from training, runs batched inference on cached lazy data,
and writes predictions in both scaled and original units.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from srcnn_model import SRCNN
from dataread_mem_srcnn import SRCNNLazyDataset, load_scaler


def build_parser():
    p = argparse.ArgumentParser(description="SRCNN inference (lazy cached data)")
    p.add_argument("--exp", type=str, default="SRCNN_v1")
    p.add_argument("--path-output", type=str, default=None,
                   help="Directory with cached x_*.npy/y_*.npy")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Directory containing srcnn_best.pth and scaler.pkl")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--var", type=str, default="tmax_dy",
                   help="Variable name used in output filenames")
    p.add_argument("--save-scaled", action="store_true",
                   help="Also save scaled predictions and labels")
    p.add_argument("--output-prefix", type=str, default="infer")
    return p


def inverse_4d(arr_4d, scaler):
    # arr_4d: [N, H, W, 1]
    n, h, w, c = arr_4d.shape
    flat = arr_4d.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(n, h, w, c).astype(np.float32)


def main():
    args = build_parser().parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(f"./checkpoints_{args.exp}")
    path_output = Path(args.path_output) if args.path_output else Path(f"./output/{args.exp}")

    ckpt_path = checkpoint_dir / "srcnn_best.pth"
    scaler_path = checkpoint_dir / "scaler.pkl"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device={device}")

    scaler = load_scaler(checkpoint_dir)
    dataset = SRCNNLazyDataset(path_output, scaler, split=args.split, use_elevation=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    sample_x, sample_y = dataset[0]
    in_channels = int(sample_x.shape[0])
    h = int(sample_y.shape[1])
    w = int(sample_y.shape[2])

    model = SRCNN(in_channels=in_channels).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_dir = checkpoint_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use same naming convention as old TF code:
    # y_{split}_predict_daily_{exp}_{var}.npy  (predictions, inverse-transformed)
    # y_{split}_daily_{exp}_{var}.npy          (true HR, inverse-transformed)
    # X_{split}_daily_{exp}_{var}.npy          (LR input first channel, inverse-transformed)
    exp_tag = f"{args.exp}_{args.var}"
    n = len(dataset)
    pred_scaled_path = out_dir / f"{args.output_prefix}_pred_scaled_{args.split}.npy"
    true_scaled_path = out_dir / f"{args.output_prefix}_true_scaled_{args.split}.npy"
    pred_inv_path  = out_dir / f"y_{args.split}_predict_daily_{exp_tag}.npy"
    true_inv_path  = out_dir / f"y_{args.split}_daily_{exp_tag}.npy"
    x_inv_path     = out_dir / f"X_{args.split}_daily_{exp_tag}.npy"

    pred_scaled = np.lib.format.open_memmap(pred_scaled_path, mode="w+", dtype=np.float32, shape=(n, h, w, 1))
    true_scaled = np.lib.format.open_memmap(true_scaled_path, mode="w+", dtype=np.float32, shape=(n, h, w, 1))
    x_scaled    = np.lib.format.open_memmap(out_dir / "_x_scaled_tmp.npy", mode="w+", dtype=np.float32, shape=(n, h, w, 1))

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (args.amp and device.type == "cuda") else nullcontext()

    offset = 0
    mse_sum = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            bs = x.shape[0]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast_ctx:
                pred = model(x)

            pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            y_np    = y.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            # first channel of x is the LR field (matching old X_test output)
            x_np    = x[:, 0:1, :, :].detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)

            pred_scaled[offset:offset + bs] = pred_np
            true_scaled[offset:offset + bs] = y_np
            x_scaled[offset:offset + bs]    = x_np

            mse_sum += float(np.mean((pred_np - y_np) ** 2)) * bs
            count += bs
            offset += bs

    pred_scaled.flush()
    true_scaled.flush()
    x_scaled.flush()

    pred_inv = inverse_4d(np.asarray(np.load(pred_scaled_path, mmap_mode="r")), scaler)
    true_inv = inverse_4d(np.asarray(np.load(true_scaled_path, mmap_mode="r")), scaler)
    x_inv    = inverse_4d(np.asarray(np.load(out_dir / "_x_scaled_tmp.npy", mmap_mode="r")), scaler)

    np.save(pred_inv_path, pred_inv)
    np.save(true_inv_path, true_inv)
    np.save(x_inv_path,    x_inv)

    if not args.save_scaled:
        pred_scaled_path.unlink(missing_ok=True)
        true_scaled_path.unlink(missing_ok=True)
    (out_dir / "_x_scaled_tmp.npy").unlink(missing_ok=True)

    metrics = {
        "split": args.split,
        "samples": int(n),
        "mse_scaled": float(mse_sum / max(1, count)),
        "checkpoint": str(ckpt_path),
        "pred_output": str(pred_inv_path),
        "true_output": str(true_inv_path),
        "x_output": str(x_inv_path),
    }

    with open(out_dir / f"{args.output_prefix}_metrics_{args.split}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[infer] done, samples={n}, mse_scaled={metrics['mse_scaled']:.6f}")
    print(f"[infer] pred: {pred_inv_path}")
    print(f"[infer] true: {true_inv_path}")
    print(f"[infer] X:    {x_inv_path}")


if __name__ == "__main__":
    main()
