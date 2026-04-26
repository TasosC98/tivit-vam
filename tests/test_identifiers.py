"""Identifier normalization checks."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tivit.data.targets.identifiers import canonical_video_id
except ModuleNotFoundError:
    import importlib.util

    module_path = REPO_ROOT / "data" / "targets" / "identifiers.py"
    spec = importlib.util.spec_from_file_location("identifiers", module_path)
    assert spec and spec.loader
    identifiers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identifiers)
    canonical_video_id = identifiers.canonical_video_id


def test_pianovam_timestamp_id_is_unique() -> None:
    assert (
        canonical_video_id("/data/PianoVAM/Video/2024-02-17_00-12-50.mp4")
        == "pianovam_2024_02_17_00_12_50"
    )


def test_legacy_video_numeric_id_is_preserved() -> None:
    assert canonical_video_id("video_132.0.mp4") == "video_132"


def run_checks() -> None:
    test_pianovam_timestamp_id_is_unique()
    test_legacy_video_numeric_id_is_preserved()
    print("identifier checks passed")


if __name__ == "__main__":
    run_checks()
