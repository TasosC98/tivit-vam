"""Training soft-target config smoke checks."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # noqa: F401
except ImportError:
    print("torch not available; skipping training soft-target checks", file=sys.stderr)
    sys.exit(0)

from tivit.data.datasets.base import BasePianoDataset


class _DummyDataset(BasePianoDataset):
    def _resolve_root(self, root_dir):
        return Path(root_dir or ".")

    def _resolve_manifest(self):
        return None

    def _list_entries(self, root, split, manifest):
        return []

    def _read_labels(self, entry):
        return {}


def run_training_soft_targets_override() -> None:
    cfg = {
        "dataset": {
            "frames": 8,
            "decode_fps": 30.0,
            "frame_targets": {
                "enable": True,
                "note_min": 21,
                "note_max": 108,
            },
        },
        "training": {
            "soft_targets": {
                "enabled": True,
                "apply_to": {
                    "onset": True,
                    "pitch": False,
                    "offset": False,
                },
                "onset_kernel": [0.5, 1.0, 0.5],
                "frame_kernel": [0.5, 1.0, 0.5],
            }
        },
    }

    dataset = _DummyDataset(cfg, split="train", full_cfg=cfg)

    assert dataset.soft_target_cfg is not None
    assert dataset.soft_target_cfg.enabled is True
    assert dataset.soft_target_cfg.apply_onset is True
    assert dataset.soft_target_cfg.apply_pitch is False
    assert dataset.soft_target_cfg.apply_offset is False


def run_all() -> None:
    run_training_soft_targets_override()
    print("training soft-target checks passed")


if __name__ == "__main__":
    run_all()
