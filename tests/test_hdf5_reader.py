"""HDF5 clip reader alignment checks."""

from __future__ import annotations

from dataclasses import dataclass
import tempfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    from tivit.data.decode.hdf5_reader import load_clip_from_hdf5
except ModuleNotFoundError:
    from data.decode.hdf5_reader import load_clip_from_hdf5


@dataclass
class _Cfg:
    frames: int
    stride: int
    start_frame: int
    resize_hw: tuple[int, int] | None = None
    channels: int = 1


try:
    import h5py
except ImportError:  # pragma: no cover - optional local dependency
    h5py = None  # type: ignore[assignment]


def _require_h5py() -> None:
    if h5py is None:
        raise SystemExit("h5py not available; skipping HDF5 reader checks")


def test_hdf5_reader_honors_start_and_stride(tmp_path: Path | None = None) -> None:
    _require_h5py()
    path = (tmp_path or Path(tempfile.mkdtemp())) / "video.h5"
    frames = np.zeros((10, 2, 2, 1), dtype=np.uint8)
    for idx in range(frames.shape[0]):
        frames[idx, :, :, 0] = idx
    with h5py.File(path, "w") as hf:
        hf.create_dataset("frames", data=frames)

    clip = load_clip_from_hdf5(path, _Cfg(frames=3, stride=2, start_frame=2))

    assert tuple(clip.shape) == (3, 1, 2, 2)
    observed = (clip[:, 0, 0, 0] * 255.0).round().to(dtype=clip.dtype).tolist()
    assert observed == [2.0, 4.0, 6.0]


def test_hdf5_fixed_clip_rejects_nonzero_start(tmp_path: Path | None = None) -> None:
    _require_h5py()
    path = (tmp_path or Path(tempfile.mkdtemp())) / "fixed_clip.h5"
    with h5py.File(path, "w") as hf:
        hf.create_dataset("frames", data=np.zeros((3, 2, 2, 1), dtype=np.uint8))

    try:
        load_clip_from_hdf5(path, _Cfg(frames=3, stride=1, start_frame=5))
    except RuntimeError as exc:
        assert "fixed clip" in str(exc)
    else:
        raise AssertionError("Expected fixed-clip HDF5 reader error")


def run_checks() -> None:
    test_hdf5_reader_honors_start_and_stride()
    test_hdf5_fixed_clip_rejects_nonzero_start()
    print("hdf5 reader checks passed")


if __name__ == "__main__":
    run_checks()
