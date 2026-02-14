"""Smoke test: print one batch shapes for mp4 vs hdf5 modes.

Run from repo root: /home/mkoziri/dev/tivit/tivit

Examples:
  export PIANOVAM_ROOT=/home/mkoziri/datasets/PianoVAM_v1.0
  export PIANOVAM_HDF5_ROOT=/home/mkoziri/datasets/PianoVAM_v1.0/HDF5

  # mp4
  python scripts/smoke_print_batch_shapes.py --config configs/pianovam_local.yaml --mode mp4 --split train --batch-size 1 --num-workers 0

  # hdf5
  python scripts/smoke_print_batch_shapes.py --config configs/pianovam_local.yaml --mode hdf5 --split train --batch-size 1 --num-workers 0
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import yaml

from tivit.data.datasets.pianovam_impl import make_dataloader


def inspect_batch(batch):
    try:
        keys = list(batch.keys())
    except Exception:
        keys = []
    print("Batch keys:", keys)

    v = batch.get("video") if isinstance(batch, dict) else None
    if v is not None:
        print("video type:", type(v))
        if isinstance(v, torch.Tensor):
            print("video.shape:", tuple(v.shape))
            print("video.dtype:", v.dtype)
            try:
                print("video.min/max:", float(v.min()), float(v.max()))
            except Exception:
                pass

    for key in ("onset", "offset", "pitch", "hand", "clef", "frame", "frames", "onsets"):
        if isinstance(batch, dict) and key in batch:
            val = batch[key]
            print(f"{key}: type={type(val)}", end=" ")
            if isinstance(val, torch.Tensor):
                print(f"shape={tuple(val.shape)} dtype={val.dtype}")
            else:
                print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config file (e.g., configs/pianovam_local.yaml)")
    p.add_argument("--mode", choices=["mp4", "hdf5"], default="mp4")
    p.add_argument("--hdf5-root", type=str, default=None)
    p.add_argument("--split", choices=["train", "valid", "test"], default="train")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    cfg = dict(cfg)
    cfg["dataset"] = dict(cfg.get("dataset", {}))

    # override for smoke
    cfg["dataset"]["batch_size"] = args.batch_size
    cfg["dataset"]["num_workers"] = args.num_workers

    if args.mode == "mp4":
        cfg["dataset"]["preprocessed_format"] = None
    else:
        cfg["dataset"]["preprocessed_format"] = "hdf5"
        # prefer CLI, else env
        h5 = args.hdf5_root or os.environ.get("PIANOVAM_HDF5_ROOT")
        if not h5:
            raise SystemExit("HDF5 mode requires --hdf5-root or env PIANOVAM_HDF5_ROOT")
        cfg["dataset"]["hdf5_root"] = h5

    print("Config:", args.config)
    print("Mode:", args.mode)
    print("Split:", args.split)
    print("dataset.root_dir:", cfg["dataset"].get("root_dir"))
    print("dataset.preprocessed_format:", cfg["dataset"].get("preprocessed_format"))
    if cfg["dataset"].get("preprocessed_format") == "hdf5":
        print("dataset.hdf5_root:", cfg["dataset"].get("hdf5_root"))

    dl = make_dataloader(cfg, args.split, drop_last=False)
    ds = getattr(dl, "dataset", None)
    ds_len = len(ds) if ds is not None else None
    print("dataset length:", ds_len)
    if ds_len is not None:
        b = int(cfg["dataset"].get("batch_size", 1))
        print("estimated batches:", math.ceil(ds_len / b))

    batch = next(iter(dl))
    print("--- Batch inspection ---")
    inspect_batch(batch)


if __name__ == "__main__":
    main()
