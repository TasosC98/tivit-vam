"""Convert a single video (or many) to per-video HDF5 files for training.

Usage (single video smoke test):
    python scripts/convert_videos_to_hdf5.py \
        --video /path/to/video.mp4 \
        --out-dir /path/to/out_dir \
        --frames 96 --stride 1 --resize 180 1536

This script uses the existing `tivit.data.decode.video_reader.load_clip`
to decode frames (so decode logic is shared) and stores frames as uint8
in dataset `frames` with shape (T,H,W,C). It also stores attributes
`fps` and `original_video_path` and an optional `label_raw` dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

import importlib.util

# Load video_reader.py directly to avoid importing data/__init__.py (which has side effects)
_VIDEO_READER_PATH = Path(__file__).resolve().parents[1] / "data" / "decode" / "video_reader.py"
spec = importlib.util.spec_from_file_location("video_reader", _VIDEO_READER_PATH)
video_reader = importlib.util.module_from_spec(spec)
assert spec and spec.loader

sys.modules[spec.name] = video_reader 
spec.loader.exec_module(video_reader)

VideoReaderConfig = video_reader.VideoReaderConfig
load_clip = video_reader.load_clip


def _get_fps(path: Path) -> float | None:
    try:
        import decord

        vr = decord.VideoReader(str(path))
        # decord's API varies; try to extract metadata
        try:
            return float(vr.get_avg_fps())
        except Exception:
            return float(vr.get_meta().get("fps", 0.0))
    except Exception:
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return float(fps) if fps and fps > 0 else None
        except Exception:
            return None


def convert_one(video_path: Path, out_dir: Path, frames: int, stride: int, resize: tuple[int, int], channels: int, label_path: Path | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = VideoReaderConfig(frames=frames, stride=stride, resize_hw=resize, channels=channels)
    x = load_clip(video_path, cfg)  # T,C,H,W float32 [0,1]
    # convert to uint8 and permute to T,H,W,C
    try:
        import torch

        x_uint8 = (x * 255.0).clamp(0, 255).to(torch.uint8)
        x_np = x_uint8.permute(0, 2, 3, 1).cpu().numpy()
    except Exception:
        x_np = (x.numpy() * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)

    import h5py

    out_path = out_dir / (video_path.stem + ".h5")
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("frames", data=x_np, dtype="uint8", compression="gzip", compression_opts=1)
        fps = _get_fps(video_path)
        if fps is not None:
            hf.attrs["fps"] = float(fps)
        hf.attrs["original_video_path"] = str(video_path)
        if label_path is not None and label_path.exists():
            try:
                with open(label_path, "rb") as f:
                    raw = f.read()
                # store raw label as variable-length bytes
                dt = h5py.special_dtype(vlen=bytes)
                hf.create_dataset("label_raw", data=raw, dtype=dt)
            except Exception:
                pass
    print(f"Wrote {out_path}")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--resize", nargs=2, type=int, metavar=("H", "W"), default=[0, 0])
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--label", type=str, default=None)
    args = p.parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        print("Video not found:", video_path)
        raise SystemExit(1)
    out_dir = Path(args.out_dir)
    label_path = Path(args.label) if args.label else None
    convert_one(video_path, out_dir, args.frames, args.stride, (args.resize[0], args.resize[1]), args.channels, label_path)


if __name__ == "__main__":
    main()
