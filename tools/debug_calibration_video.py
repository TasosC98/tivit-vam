#!/usr/bin/env python3
"""Per-video calibration debug tool.

Runs the full calibration pipeline on a single video and saves intermediate
artifacts so we can SEE what's happening:

    median_frame.png           - the hand-suppressed median image we detect on
    detection_overlay.png      - red dots at every detected blob center
    seeded_overlay.png         - blue dots: where each detection lands in
                                 canonical space using the seed (a_seed, b_seed)
    fitted_overlay.png         - green dots: post-RANSAC canonical positions
                                 + magenta dots: the canonical black-key
                                 targets (so you can see anchor-to-target lines)
    polygons_on_median.png     - all 88 key polygons projected back onto the
                                 median frame using H_inv
    polygons_on_canonical.png  - the canonical 88-key layout (same every time;
                                 included so you can sanity-check the expected
                                 layout)
    residuals.txt              - per-anchor residuals + summary stats
    diagnostics.json           - everything above in one JSON

Usage:
    PYTHONPATH=. python tools/debug_calibration_video.py \\
        --config configs/default.yaml \\
        --config configs/overrides/manual_training.yaml \\
        --config configs/overrides/manual_training_keyroi_train.yaml \\
        --config configs/overrides/26-04-2026/manual_training_keyroi_video.yaml \\
        --video-id 2024-03-04_03-46-36 \\
        --split train \\
        --out-dir /home/achatzigiannis/tivit-logs/manual_training_keyroi_video_26-04-2026/calib_debug

Send back the `out-dir/<video_id>/` folder and the stdout. With those I can
see exactly whether detection misses keys, the seed picks the wrong offset,
or the fit residual is real perspective distortion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Calibrator imports require the dataset; bypass the geometry gate.
os.environ["TIVIT_GEOMETRY_GATE_DISABLED"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _crop_tuple(crop: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(crop, (list, tuple)) or len(crop) < 4:
        return None
    try:
        return (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--skip-seconds", type=float, default=1.0)
    args = ap.parse_args()

    import numpy as np
    import cv2

    from tivit.core.config import load_experiment_config
    from tivit.data.datasets.pianovam_impl import PianoVAMDataset
    from tivit.data.roi.keyboard_calibration import (
        CalibrationConfig,
        MedianFrameOptions,
        build_median_frame,
        canonical_black_centers,
        detect_black_key_centers,
        fit_homography_ransac,
        _estimate_black_band,
        _seed_affine_x,
        _to_gray,
        _polygons_for_canonical,
        _polygons_through_inverse_h,
        _MIDI_LOW,
        _midi_is_white,
    )

    cfg = dict(load_experiment_config(args.config or [Path("configs/default.yaml")]))
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    dataset_cfg["preprocessed_format"] = "video"
    dataset_cfg["shuffle"] = False
    cfg["dataset"] = dataset_cfg

    canonical_hw = dataset_cfg.get("canonical_hw") or [180, 1536]
    canonical_hw = (int(canonical_hw[0]), int(canonical_hw[1]))

    ds = PianoVAMDataset(cfg, split=args.split, full_cfg=cfg)
    target_entry = None
    for entry in ds.entries:
        if str(entry.video_id) == args.video_id:
            target_entry = entry
            break
    if target_entry is None:
        raise SystemExit(f"video {args.video_id} not found in split={args.split}")

    out_dir = Path(args.out_dir).expanduser().resolve() / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    crop = _crop_tuple(target_entry.metadata.get("crop") if isinstance(target_entry.metadata, Mapping) else None)
    print(f"video_path = {target_entry.video_path}")
    print(f"crop       = {crop}")

    median_opts = MedianFrameOptions(
        num_samples=int(args.num_samples),
        skip_seconds=float(args.skip_seconds),
        max_seconds=None,
    )
    median = build_median_frame(Path(target_entry.video_path), crop=crop, options=median_opts)
    h_src, w_src = median.shape[:2]
    print(f"median frame shape = ({h_src}, {w_src})")
    cv2.imwrite(str(out_dir / "median_frame.png"), median)

    gray = _to_gray(median)
    y_top, y_bot = _estimate_black_band(gray)
    print(f"black band y_top={y_top} y_bot={y_bot}  (band height={y_bot - y_top})")

    detected_xs = detect_black_key_centers(median)
    print(f"detected black-key anchors: n={detected_xs.size}")
    print(f"detection x range: min={float(detected_xs.min()) if detected_xs.size else None} "
          f"max={float(detected_xs.max()) if detected_xs.size else None}")

    detection_img = median.copy()
    cv2.rectangle(detection_img, (0, y_top), (w_src - 1, y_bot), (0, 255, 255), 1)
    for x in detected_xs.tolist():
        cv2.circle(detection_img, (int(round(x)), int(0.5 * (y_top + y_bot))), 4, (0, 0, 255), -1)
    cv2.imwrite(str(out_dir / "detection_overlay.png"), detection_img)

    canon_x = canonical_black_centers(float(canonical_hw[1]))
    a_seed, b_seed = _seed_affine_x(detected_xs, canon_x)
    print(f"seed affine: a={a_seed:.4f} b={b_seed:.2f}")

    seeded_canonical = detected_xs.astype(np.float64) * a_seed + b_seed
    seed_img = np.zeros((canonical_hw[0], canonical_hw[1], 3), dtype=np.uint8)
    seed_img[:] = (40, 40, 40)
    for cx in canon_x.tolist():
        cv2.line(seed_img, (int(cx), 0), (int(cx), canonical_hw[0] - 1), (255, 0, 255), 1)
    for sx in seeded_canonical.tolist():
        if 0 <= sx < canonical_hw[1]:
            cv2.circle(seed_img, (int(sx), canonical_hw[0] // 2), 5, (255, 255, 0), -1)
    cv2.imwrite(str(out_dir / "seeded_overlay.png"), seed_img)

    fit = fit_homography_ransac(
        source_black_xs=detected_xs,
        source_y=float(0.5 * (y_top + y_bot)),
        canonical_hw=canonical_hw,
        source_hw=(int(h_src), int(w_src)),
        iterations=400,
        inlier_threshold_px=3.0,
    )

    diagnostics: Dict[str, Any] = {
        "video_id": args.video_id,
        "video_path": str(target_entry.video_path),
        "split": args.split,
        "crop": list(crop) if crop is not None else None,
        "source_hw": [int(h_src), int(w_src)],
        "canonical_hw": list(canonical_hw),
        "black_band_y": [int(y_top), int(y_bot)],
        "detected_anchor_count": int(detected_xs.size),
        "detected_xs_first10": detected_xs[:10].tolist(),
        "detected_xs_last10": detected_xs[-10:].tolist(),
        "canon_xs_first10": canon_x[:10].tolist(),
        "canon_xs_last10": canon_x[-10:].tolist(),
        "seed_a": float(a_seed),
        "seed_b": float(b_seed),
    }

    if fit.homography is None:
        print("RANSAC FAILED: no homography produced")
        diagnostics["fit_homography"] = None
    else:
        H = fit.homography
        H_inv = np.linalg.inv(H).astype(np.float32)
        diagnostics["fit_homography"] = H.flatten().astype(float).tolist()
        diagnostics["fit_a"] = float(H[0, 0])
        diagnostics["fit_b"] = float(H[0, 2])
        diagnostics["fit_c_perspective"] = float(H[2, 0])
        diagnostics["matched_canonical_indices"] = fit.matched_canonical_indices.tolist()
        diagnostics["inlier_mask"] = fit.inlier_mask.tolist()
        diagnostics["residuals"] = fit.residuals_canonical_px.tolist()
        diagnostics["residual_median_px"] = float(np.median(fit.residuals_canonical_px))
        diagnostics["residual_p95_px"] = float(np.quantile(fit.residuals_canonical_px, 0.95))
        diagnostics["residual_max_px"] = float(np.max(fit.residuals_canonical_px))

        print(f"FIT: H[0]={H[0].tolist()} H[2]={H[2].tolist()}")
        print(f"     a={H[0,0]:.4f} b={H[0,2]:.2f} c={H[2,0]:.6e}")
        print(f"     matched anchors: {fit.matched_canonical_indices.size}")
        print(f"     inliers (residual<3px): {int(fit.inlier_mask.sum())}/{fit.inlier_mask.size}")
        print(f"     residual median={diagnostics['residual_median_px']:.3f} "
              f"p95={diagnostics['residual_p95_px']:.3f} max={diagnostics['residual_max_px']:.3f}")

        # Per-anchor residual list (matched canonical index -> residual)
        with (out_dir / "residuals.txt").open("w", encoding="utf-8") as handle:
            handle.write("matched_idx\tcanonical_idx\tcanonical_x\tinlier\tresidual\n")
            for i, ci in enumerate(fit.matched_canonical_indices.tolist()):
                handle.write(
                    f"{i}\t{ci}\t{canon_x[ci]:.2f}\t"
                    f"{int(fit.inlier_mask[i])}\t{fit.residuals_canonical_px[i]:.3f}\n"
                )

        # Fitted overlay: in canonical space, draw canon black-key positions
        # (magenta vertical lines) + projected detections (green dots).
        fitted_img = np.zeros((canonical_hw[0], canonical_hw[1], 3), dtype=np.uint8)
        fitted_img[:] = (40, 40, 40)
        for cx in canon_x.tolist():
            cv2.line(fitted_img, (int(cx), 0), (int(cx), canonical_hw[0] - 1), (255, 0, 255), 1)
        # Apply the projective formula explicitly for clarity:
        x_proj = np.zeros(detected_xs.size, dtype=np.float64)
        for i, x in enumerate(detected_xs.tolist()):
            denom = H[2, 0] * x + 1.0
            x_proj[i] = (H[0, 0] * x + H[0, 2]) / denom if abs(denom) > 1e-9 else 0.0
        for px in x_proj.tolist():
            if 0 <= px < canonical_hw[1]:
                cv2.circle(fitted_img, (int(px), canonical_hw[0] // 2), 5, (0, 255, 0), -1)
        cv2.imwrite(str(out_dir / "fitted_overlay.png"), fitted_img)

        # Project all 88 key polygons back onto the median frame
        canonical_polys = _polygons_for_canonical(canonical_hw)
        polys_source = _polygons_through_inverse_h(canonical_polys, H_inv)
        polys_img = median.copy()
        for k, poly in enumerate(polys_source):
            midi = _MIDI_LOW + k
            color = (50, 210, 80) if _midi_is_white(midi) else (255, 70, 50)
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(polys_img, [pts], isClosed=True, color=color, thickness=1)
        cv2.imwrite(str(out_dir / "polygons_on_median.png"), polys_img)

        # Canonical layout image (sanity check)
        canon_img = np.zeros((canonical_hw[0], canonical_hw[1], 3), dtype=np.uint8)
        canon_img[:] = (255, 255, 255)
        for k, poly in enumerate(canonical_polys):
            midi = _MIDI_LOW + k
            color = (50, 210, 80) if _midi_is_white(midi) else (255, 70, 50)
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canon_img, [pts], isClosed=True, color=color, thickness=1)
            if midi % 12 == 0:
                cv2.putText(canon_img, str(midi), (int(poly[0][0]) + 2, canonical_hw[0] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "polygons_on_canonical.png"), canon_img)

    with (out_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    print(f"\nWrote artifacts to {out_dir}")
    print("Files: median_frame.png  detection_overlay.png  seeded_overlay.png")
    print("       fitted_overlay.png  polygons_on_median.png  polygons_on_canonical.png")
    print("       residuals.txt  diagnostics.json")


if __name__ == "__main__":
    main()
