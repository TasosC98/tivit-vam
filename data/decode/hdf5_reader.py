"""HDF5 reader that matches `video_reader.load_clip` output.

Provides `load_clip_from_hdf5(path, cfg)` which returns a torch.Tensor
in shape (T,C,H,W), float32 in [0,1], matching `load_clip` behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def _frame_indices(total: int, cfg) -> list[int]:
    """Resolve clip indices using the same start/stride/pad semantics as video_reader."""

    frames = max(1, int(getattr(cfg, "frames", 1)))
    stride = max(1, int(getattr(cfg, "stride", 1)))
    requested_start = max(0, int(getattr(cfg, "start_frame", 0)))

    # The historical converter in this repo wrote one fixed clip per video. In
    # that case a non-zero requested start cannot be served from the HDF5 file;
    # callers should fall back to the original video rather than silently
    # returning frame 0 while labels are shifted to a later clip.
    if requested_start > 0 and total <= frames:
        raise RuntimeError(
            "HDF5 file appears to contain a fixed clip, not a full video; "
            f"cannot serve start_frame={requested_start} from {total} stored frames"
        )

    start = min(requested_start, max(total - 1, 0))
    idxs = list(range(start, total, stride))[:frames]
    if not idxs:
        idxs = [0]
    return idxs


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
        dset = hf["frames"]
        if dset.ndim != 4:
            raise RuntimeError(f"Unexpected frames shape in {p}: {dset.shape}")
        frames_requested = max(1, int(getattr(cfg, "frames", 1)))
        idxs = _frame_indices(int(dset.shape[0]), cfg)
        data = dset[idxs]
        if data.shape[0] < frames_requested:
            pad_count = frames_requested - int(data.shape[0])
            tail = np.repeat(data[-1:], pad_count, axis=0)
            data = np.concatenate([data, tail], axis=0)
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
