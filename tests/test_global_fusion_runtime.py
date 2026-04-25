"""Global fusion runtime smoke checks."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError:
    print("torch not available; skipping global fusion runtime checks", file=sys.stderr)
    sys.exit(0)

from tivit.decoder.global_fusion import fuse_outputs_with_tile_mask


def run_fuse_outputs_with_tile_mask() -> None:
    outputs = {
        "pitch_tile": torch.tensor(
            [[[[10.0, 8.0, -10.0, -10.0], [-10.0, -10.0, 7.0, 9.0]]]],
            dtype=torch.float32,
        ),
    }
    outputs["pitch_logits"] = outputs["pitch_tile"].mean(dim=2)

    tile_mask = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    cfg = {
        "global_fusion": {
            "enabled": True,
            "mode": "masked_mean",
            "apply_to": ["pitch"],
        }
    }

    fused, applied = fuse_outputs_with_tile_mask(outputs, tile_mask, cfg)

    assert applied == ("pitch",)
    fused_logits = fused["pitch_logits"]
    assert fused_logits.shape == (1, 1, 4)
    assert float(fused_logits[0, 0, 0]) == 10.0
    assert float(fused_logits[0, 0, 1]) == 8.0
    assert float(fused_logits[0, 0, 2]) == 7.0
    assert float(fused_logits[0, 0, 3]) == 9.0


def run_all() -> None:
    run_fuse_outputs_with_tile_mask()
    print("global fusion runtime checks passed")


if __name__ == "__main__":
    run_all()
