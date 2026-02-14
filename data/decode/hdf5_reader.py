"""HDF5 reader that matches `video_reader.load_clip` output.

Provides `load_clip_from_hdf5(path, cfg)` which returns a torch.Tensor
in shape (T,C,H,W), float32 in [0,1], matching `load_clip` behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def load_clip_from_hdf5(path: str | Path, cfg) -> torch.Tensor:
    """Load clip stored in HDF5 as dataset `frames` with shape (T,H,W,C) uint8.

    Returns tensor (T,C,H,W) float32 in [0,1]. If `cfg.channels==1` will
    convert to grayscale.
    """
    import h5py

    p = Path(path)
    with h5py.File(p, "r") as hf:
        if "frames" not in hf:
            raise RuntimeError(f"HDF5 file {p} missing 'frames' dataset")
        data = hf["frames"][:]
    # data is uint8 (T,H,W,C) -> convert to float32 [0,1]
    x = torch.from_numpy(data.astype(np.float32) / 255.0)
    # permute to T,C,H,W
    if x.ndim == 4:
        x = x.permute(0, 3, 1, 2)
    else:
        raise RuntimeError(f"Unexpected frames shape in {p}: {data.shape}")

    if cfg.channels == 1 and x.shape[1] == 3:
        x = (0.299 * x[:, 0:1, ...] + 0.587 * x[:, 1:2, ...] + 0.114 * x[:, 2:3, ...])

    # Resize if needed, matching video_reader behavior
    if getattr(cfg, "resize_hw", None):
        h, w = x.shape[-2:]
        if (h, w) != tuple(cfg.resize_hw):
            x = torch.nn.functional.interpolate(x, size=cfg.resize_hw, mode="area")

    return x


__all__ = ["load_clip_from_hdf5"]
