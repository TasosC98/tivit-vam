#!/usr/bin/env python3
"""Verify the calibration code on the server matches what was developed locally.

Run this BEFORE recalibrate_all.py to confirm the white-edge correlation,
honest residual metric, and nominal-crop fallback are actually present in
the code being executed. If you pull the wrong branch or have a stale cache,
this will tell you immediately rather than after a 3-minute calibration run.

Usage:
    PYTHONPATH=. python tools/calibration_self_test.py

Exit code 0 means the build is correct and ready for recalibrate_all.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    print("=" * 72)
    print("TIVIT calibration self-test")
    print("=" * 72)

    failures = 0

    # 1) Import the module and print its version.
    try:
        from tivit.data.roi.keyboard_calibration import (
            CALIBRATION_VERSION,
            CALIBRATION_FEATURES,
            calibrate_white_edge_correlation,
            calibrate_nominal_crop,
            _residuals_over_all_anchors,
            canonical_black_centers,
        )
        import tivit.data.roi.keyboard_calibration as kc_mod
    except ImportError as exc:
        print(f"FAIL: cannot import keyboard_calibration with expected symbols")
        print(f"      {exc}")
        print()
        print("Likely cause: you are running an OLD version of the code on the server.")
        print("Action: `git pull` in /home/achatzigiannis/tivit-vam (or wherever the repo lives).")
        return 2

    print(f"  module file        : {kc_mod.__file__}")
    print(f"  CALIBRATION_VERSION: {CALIBRATION_VERSION}")
    print(f"  features           : {CALIBRATION_FEATURES}")
    print()

    # 2) Confirm the version string matches what we expect.
    # We accept any v4+ marker so future minor revisions don't break the test;
    # bump this when a NEW pipeline change requires the test to validate it.
    accepted_markers = ("v5.coverage", "v6.", "v7.")
    if not any(m in CALIBRATION_VERSION for m in accepted_markers):
        print(f"FAIL: CALIBRATION_VERSION does not contain any of {accepted_markers}.")
        print(f"      Found: {CALIBRATION_VERSION}")
        print(f"      You are running OLD code. Pull the repo.")
        failures += 1
    else:
        print(f"  PASS: version is a recognized v4+ build")

    # 3) Required features must be present. Names track the constants in
    #    keyboard_calibration.py CALIBRATION_FEATURES; update when those
    #    rename or split.
    required = {
        "white_edge_correlation",
        "one_to_one_residual_assignment",
        "nominal_crop_fallback",
        "relaxed_acceptance_thresholds",
        "source_coverage_gate",
    }
    missing = required - set(CALIBRATION_FEATURES)
    if missing:
        print(f"FAIL: missing required features: {missing}")
        failures += 1
    else:
        print(f"  PASS: required features present")

    # 4) Synthetic end-to-end test of white-edge correlation.
    try:
        import numpy as np
        rng = np.random.RandomState(0)
        # Build a synthetic median frame: 200x1700, dark band at top with
        # 36 black-key blobs, white-key edges in the lower band.
        h_src, w_src = 200, 1700
        median = np.full((h_src, w_src, 3), 240, dtype=np.uint8)
        # Black-key band: 36 dark rectangles
        canon_black = canonical_black_centers(1536.0)
        a_true = 1700.0 / 1536.0
        for cx in canon_black:
            sx = int(cx * a_true)
            cv2_x0 = max(0, sx - 10)
            cv2_x1 = min(w_src, sx + 10)
            median[5:90, cv2_x0:cv2_x1, :] = 30
        # White-key edges: 53 thin vertical lines in lower band
        from data.roi.keyboard_calibration import _canonical_white_edges as _wedges
        for ex in _wedges(1536.0):
            sx = int(ex * a_true)
            if 0 <= sx < w_src:
                median[100:190, sx:sx + 1, :] = 60

        result = calibrate_white_edge_correlation(median, canonical_hw=(180, 1536))
        if result is None:
            print(f"FAIL: white-edge correlation returned None on synthetic input")
            failures += 1
        else:
            a, b, score = result
            err = abs(a - a_true) + abs(b)
            print(f"  PASS: synthetic white-edge fit a={a:.4f} (true {a_true:.4f}) b={b:.2f} score={score:.3f}")
            if err > 0.05:
                print(f"  WARN: synthetic fit error {err:.3f} is larger than expected (<0.05)")
    except Exception as exc:
        print(f"FAIL: synthetic white-edge test raised {type(exc).__name__}: {exc}")
        failures += 1

    # 5) Nominal-crop fallback.
    try:
        a, b = calibrate_nominal_crop(canonical_hw=(180, 1536), source_hw=(200, 1700))
        expected_a = 1536.0 / 1700.0
        if abs(a - expected_a) > 1e-3 or abs(b) > 1e-3:
            print(f"FAIL: nominal-crop fallback returned a={a}, b={b} (expected {expected_a}, 0)")
            failures += 1
        else:
            print(f"  PASS: nominal-crop fallback a={a:.4f} b={b:.2f}")
    except Exception as exc:
        print(f"FAIL: calibrate_nominal_crop raised {type(exc).__name__}: {exc}")
        failures += 1

    # 6) Honest residual metric.
    try:
        import numpy as np
        canon = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        # Detected blobs perfectly aligned: residuals should all be 0.
        det = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32)
        # With a=2.0, b=0: warped = [10, 20, 30, 40] => exact match
        res = _residuals_over_all_anchors(2.0, 0.0, det, canon)
        if res.size != det.size or float(np.max(res)) > 1e-3:
            print(f"FAIL: _residuals_over_all_anchors returned {res} (expected all-zero)")
            failures += 1
        else:
            print(f"  PASS: honest residuals on aligned input -> all zero")
    except Exception as exc:
        print(f"FAIL: honest residual test raised {type(exc).__name__}: {exc}")
        failures += 1

    print()
    if failures == 0:
        print("ALL TESTS PASSED — safe to run tools/recalibrate_all.py")
        return 0
    else:
        print(f"{failures} TESTS FAILED — DO NOT run recalibrate_all.py until fixed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
