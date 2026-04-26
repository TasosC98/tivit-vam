"""PianoVAM label parsing checks."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tivit.data.datasets.pianovam_impl import _parse_event_row
except ModuleNotFoundError:
    from data.datasets.pianovam_impl import _parse_event_row


def test_five_column_tsv_defaults_to_key_offset() -> None:
    row = ["1.0", "2.0", "3.0", "60", "90"]

    onset, offset, pitch = _parse_event_row(row)

    assert onset == 1.0
    assert offset == 2.0
    assert pitch == 60


def test_five_column_tsv_can_use_frame_offset() -> None:
    row = ["1.0", "2.0", "3.0", "60", "90"]

    onset, offset, pitch = _parse_event_row(row, offset_column="frame_offset")

    assert onset == 1.0
    assert offset == 3.0
    assert pitch == 60


def run_checks() -> None:
    test_five_column_tsv_defaults_to_key_offset()
    test_five_column_tsv_can_use_frame_offset()
    print("pianovam key_offset parsing checks passed")


if __name__ == "__main__":
    run_checks()
