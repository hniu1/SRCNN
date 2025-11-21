#!/usr/bin/env python

"""
Frontier-ready DDP PyTorch SRCNN
Matches your SRGAN training structure but with a simple MSE SRCNN.
Data pipeline is the same as your TensorFlow version.
"""

from mpi4py import MPI
import os
import argparse
import json
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from pathlib import Path

# Project-local modules
from srcnn_model import SRCNN
from data_util_srcnn import read_data, read_elev, minmax_01, standardize_like


# ================================ ARGS ================================

def build_parser():
    p = argparse.ArgumentParser(description="Frontier ROCm SRCNN (DDP)")

    p.add_argument("--master_addr", type=str)
    p.add_argument("--master_port", type=str)

    p.add_argument("--base-dir", type=str, default="/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data")
    p.add_argument("--dir-elev", type=str, default="/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/DEM")
    p.add_argument("--exp", type=str, default="SRCNN_v1")
    p.add_argument("--var", type=str, default="tmax_dy")
    p.add_argument("--year-start", type=int, default=1980)
    p.add_argument("--year-end", type=int, default=1981)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")

    return p


# =========================== DIST HELPERS =============================

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world():
    return dist.get_world_size() if is_dist() else 1


# =============================== DATASET ==============================

class SRCNNDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(self.Y[idx], dtype=torch.float32).permute(2, 0, 1)
        return x, y

    def __len__(self):
        return len(self.X)


# ============================== TRAIN LOOP ============================

def train_loop(device, args, X_train, Y_train, X_val, Y_val, checkpoint_dir):

    dataset = SRCNNDataset(X_train, Y_train)

    if is_dist():
        sampler = DistributedSampler(dataset, num_replicas=get_world(), rank=get_rank(),
                                     shuffle=True, drop_last=True)
    else:
        sampler = None

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(sampler is None),
                        sampler=sampler,
                        num_workers=args.num_workers,
                        pin_memory=False)

    val_dataset = SRCNNDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # -------- Model --------
    in_channels = X_train.shape[-1]
    model = SRCNN(in_channels=in_channels).to(device)

    if is_dist():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=False
        )

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # AMP context
    if args.amp:
        if device.type == "cuda":
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    best_val = float("inf")
    history = {"train": [], "val": []}

    # ==================== Training ====================
    for epoch in range(args.epochs):
        if is_dist() and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                pred = model(x)
                loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(loader))
        history["train"].append(train_loss)

        # ================== Validation ====================
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                with autocast_ctx:
                    pred = model(x)
                    loss = criterion(pred, y)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        history["val"].append(val_loss)

        if get_rank() == 0:
            print(f"[Epoch {epoch+1}/{args.epochs}] Train={train_loss:.6f}, Val={val_loss:.6f}")

        # Save best checkpoint
        if get_rank() == 0 and val_loss < best_val:
            best_val = val_loss
            torch.save(model.module.state_dict() if is_dist() else model.state_dict(),
                       os.path.join(checkpoint_dir, "srcnn_best.pth"))

    # Save losses
    if (not is_dist()) or get_rank() == 0:
        with open(os.path.join(checkpoint_dir, "loss_history.json"), "w") as f:
            json.dump(history, f, indent=2)


# ================================ MAIN ================================

def main():
    args = build_parser().parse_args()

    comm = MPI.COMM_WORLD
    world = comm.Get_size()
    rank = comm.Get_rank()

    # revise the code to be compatible with cpu only runs
    use_cuda = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if use_cuda else 0

    # ------------ Determine local rank ------------
    if use_cuda and num_gpus > 0:
        # Normal multi-GPU mode
        local_rank = rank % num_gpus
    else:
        # CPU-only: no GPUs available
        local_rank = 0

    # ------------ Set device ------------
    if use_cuda and num_gpus > 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")


    # Export vars for torch DDP
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    if world > 1:
        try:
            os.environ["MASTER_ADDR"] = str(args.master_addr)
            os.environ["MASTER_PORT"] = str(args.master_port)
        except Exception as e:
            raise RuntimeError("In multi-node DDP, --master-addr and --master-port must be set") from e

    if world > 1:
        # Multi-process run
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world,
        )
    else:
        # Single process — no need to initialize DDP
        print(f"[Rank {rank}] Single process run (no DDP). Device={device}")

    dist.barrier() if world > 1 else None

    print(f"[Rank {rank}] Initialized. device={device}, "
        f"use_cuda={use_cuda}, local_rank={local_rank}, world={world}")
    # dist.init_process_group(backend="nccl",
    #                         init_method="env://",
    #                         rank=rank,
    #                         world_size=world)

    # ---------------- Data ----------------
    base = Path(args.base_dir)
    dir_elev = Path(args.dir_elev)

    # read HR and LR data only for rank 0, then broadcast to other ranks
    X = None
    Y = None
    if rank == 0:
        print("Reading data...")
        hr = read_data(args.var, deg=0.0416, res="high",
                    year_start=args.year_start, year_end=args.year_end,
                    base_dir=base)

        lr = read_data(args.var, deg=0.25, res="low",
                    year_start=args.year_start, year_end=args.year_end,
                    base_dir=base)

        T = min(hr.shape[0], lr.shape[0])
        hr = hr[:T]
        lr = lr[:T] 

        elev = read_elev(T, dir_elev)

        elev_scaled, _ = minmax_01(elev)
        hr_scaled, scaler = standardize_like(hr)
        lr_scaled, _ = standardize_like(lr, scaler)

        X = np.concatenate([lr_scaled, elev_scaled], axis=-1)
        Y = hr_scaled

    # Broadcast prepared arrays to every rank so they can proceed independently
    if world > 1:
        X = comm.bcast(X, root=0)
        Y = comm.bcast(Y, root=0)
    
    # Wait until rank 0 completes before others proceed
    if world > 1 and dist.is_initialized():
        print(f"[Rank {rank}] Waiting at barrier after data load...", flush=True)
        dist.barrier()
        print(f"[Rank {rank}] Barrier released.", flush=True)

    # Train/val split
    n = len(X)
    idx = int(0.8 * n)
    X_train, X_val = X[:idx], X[idx:]
    Y_train, Y_val = Y[:idx], Y[idx:]

    # ---------------- Training ----------------
    checkpoint_dir = f"./checkpoints_{args.exp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if rank == 0:
        print(f"Shapes:")
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

    train_loop(device, args, X_train, Y_train, X_val, Y_val, checkpoint_dir)

    if world > 1 and is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
