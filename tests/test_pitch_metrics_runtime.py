"""Runtime checks for pitch/key understanding metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("numpy not available; skipping pitch metric runtime checks", file=sys.stderr)
    sys.exit(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tivit.metrics.pitch_metrics import key_state_counts, summarize_key_state_counts


def _assert_close(val: float, target: float, *, tol: float = 1e-6) -> None:
    assert math.isclose(val, target, rel_tol=tol, abs_tol=tol), f"{val} != {target}"


def test_perfect_key_state_match() -> None:
    target = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    pred = target.copy()

    counts = key_state_counts(pred, target)
    summary = summarize_key_state_counts(counts)

    _assert_close(summary["pitch_active_precision"], 1.0)
    _assert_close(summary["pitch_active_recall"], 1.0)
    _assert_close(summary["pitch_active_f1"], 1.0)
    _assert_close(summary["pitch_inactive_specificity"], 1.0)
    _assert_close(summary["pitch_overall_accuracy"], 1.0)
    _assert_close(summary["pitch_exact_frame_match"], 1.0)
    _assert_close(summary["pitch_frame_jaccard"], 1.0)
    _assert_close(summary["pitch_polyphony_mae"], 0.0)


def test_mixed_pressed_and_unpressed_errors() -> None:
    target = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    pred = np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    counts = key_state_counts(pred, target)
    summary = summarize_key_state_counts(counts)

    _assert_close(counts["tp"], 1.0)
    _assert_close(counts["fp"], 1.0)
    _assert_close(counts["fn"], 1.0)
    _assert_close(counts["tn"], 3.0)
    _assert_close(summary["pitch_active_precision"], 0.5)
    _assert_close(summary["pitch_active_recall"], 0.5)
    _assert_close(summary["pitch_active_f1"], 0.5)
    _assert_close(summary["pitch_inactive_specificity"], 0.75)
    _assert_close(summary["pitch_overall_accuracy"], 4.0 / 6.0)
    _assert_close(summary["pitch_balanced_accuracy"], 0.625)
    _assert_close(summary["pitch_exact_frame_match"], 0.0)
    _assert_close(summary["pitch_frame_jaccard"], 0.25)
    _assert_close(summary["pitch_polyphony_mae"], 1.0)
    _assert_close(summary["pitch_pred_active_per_frame"], 1.0)
    _assert_close(summary["pitch_gt_active_per_frame"], 1.0)


def run_checks() -> None:
    test_perfect_key_state_match()
    test_mixed_pressed_and_unpressed_errors()
    print("pitch metric runtime checks passed")


if __name__ == "__main__":
    run_checks()
