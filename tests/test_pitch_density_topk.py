"""Pitch density top-k mask checks."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

try:
    from tivit.postproc.patk_decode import clamp_probs
except ModuleNotFoundError:
    from postproc.patk_decode import clamp_probs


def test_clamp_probs_applies_top_k_cap_per_frame() -> None:
    probs = torch.tensor([[[0.90, 0.80, 0.70, 0.20]]])

    mask = clamp_probs(probs, 0.10, top_k=2)

    assert mask.tolist() == [[[True, True, False, False]]]


def test_clamp_probs_keeps_threshold_when_top_k_disabled() -> None:
    probs = torch.tensor([[[0.90, 0.80, 0.05, 0.20]]])

    mask = clamp_probs(probs, 0.10)

    assert mask.tolist() == [[[True, True, False, True]]]


def run_checks() -> None:
    test_clamp_probs_applies_top_k_cap_per_frame()
    test_clamp_probs_keeps_threshold_when_top_k_disabled()
    print("pitch density top-k checks passed")


if __name__ == "__main__":
    run_checks()
