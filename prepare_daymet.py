#!/usr/bin/env python3
import argparse
from pathlib import Path

from dataread_mem_srcnn import daymetread_lazy


def build_parser():
    p = argparse.ArgumentParser("Prepare lazy SRCNN cache data")
    p.add_argument("--base-dir", type=str, default="/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data")
    p.add_argument("--dir-elev", type=str, default="/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/DEM")
    p.add_argument("--exp", type=str, default="SRCNN_v1")
    p.add_argument("--path-output", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--var", type=str, default="tmax_dy")
    p.add_argument("--year-start", type=int, default=1980)
    p.add_argument("--year-end", type=int, default=2014)
    p.add_argument("--downscale-mode", type=str, default="0p25to0p0416",
                   choices=["1to0p25", "0p25to0p0416"])
    p.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"])
    return p


def main():
    args = build_parser().parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(f"./checkpoints_{args.exp}")
    path_output = Path(args.path_output) if args.path_output else Path(f"./output/{args.exp}")

    print("=" * 80)
    print("[prepare_daymet SRCNN] START")
    print(f"base_dir       : {args.base_dir}")
    print(f"dir_elev       : {args.dir_elev}")
    print(f"exp            : {args.exp}")
    print(f"path_output    : {path_output}")
    print(f"checkpoint_dir : {checkpoint_dir}")
    print(f"var            : {args.var}")
    print(f"years          : {args.year_start}-{args.year_end}")
    print(f"downscale_mode : {args.downscale_mode}")
    print(f"scaler         : {args.scaler}")
    print("=" * 80, flush=True)

    daymetread_lazy(
        path_output=path_output,
        checkpoint_dir=checkpoint_dir,
        base_dir=Path(args.base_dir),
        dir_elev=Path(args.dir_elev),
        var=args.var,
        year_start=args.year_start,
        year_end=args.year_end,
        downscale_mode=args.downscale_mode,
        scaler_type=args.scaler,
        train_fraction=0.8,
    )

    print("[prepare_daymet SRCNN] DONE", flush=True)


if __name__ == "__main__":
    main()
