#!/usr/bin/env python3
"""Render per-video diagnostic images for a completed calibration run.

Reads `key_geometry/{split}/{video_id}.json` files, builds the median frame
for each video, and writes:

    <out-root>/<split>/<status>/<video_id>.png

The PNG is a stacked diagnostic image:
    Row 1: median frame WITH 88 key polygons (calibrated). Borders coloured by
           status (green=ok, yellow=loose, red=failed).
    Row 2: status banner with median residual + p95 + anchors + inlier ratio.
    Row 3: residual bar chart (matched canonical index -> residual px).
           Bars green if <3 px, yellow if <5 px, red if >=5 px.

Status folders make it easy to scan all failures vs all OK at once.

Usage:
    PYTHONPATH=. python tools/visualize_calibration_run.py \\
        --config configs/default.yaml \\
        --config configs/overrides/manual_training.yaml \\
        --config configs/overrides/manual_training_keyroi_train.yaml \\
        --config configs/overrides/06-05-2026/manual_training_keyroi_video_06_05_2026.yaml \\
        --geometry-root /home/achatzigiannis/tivit-logs/manual_training_keyroi_video_06_05_2026/key_geometry \\
        --out-root     /home/achatzigiannis/tivit-logs/manual_training_keyroi_video_06_05_2026/visualizations \\
        --splits train valid test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

os.environ["TIVIT_GEOMETRY_GATE_DISABLED"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_STATUS_COLORS = {
    "ok": (40, 200, 40),
    "manual_fixed": (40, 200, 40),
    "accepted_loose": (50, 200, 220),
    "failed": (60, 60, 220),
    "errored": (60, 60, 220),
}


def _crop_tuple(crop: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(crop, (list, tuple)) or len(crop) < 4:
        return None
    try:
        return (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))
    except (TypeError, ValueError):
        return None


def _draw_status_banner(width: int, height: int, payload: Mapping[str, Any], color):
    import cv2
    import numpy as np

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (28, 28, 28)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), color, 2)
    status = str(payload.get("calibration_status") or "?")
    med = payload.get("residual_median_px")
    p95 = payload.get("residual_p95_px")
    anchors = payload.get("black_key_anchor_count")
    inlier_ratio = payload.get("ransac_inlier_ratio")
    norm = (float(med) / 29.5384615) if (med is not None) else None
    line1 = f"status={status}  median={med:.2f}px  p95={p95:.2f}px" if med is not None else f"status={status}"
    line2 = (
        f"anchors={anchors}  inliers={int(round(100*float(inlier_ratio)))}%  norm={norm:.3f} (key-width)"
        if (anchors is not None and inlier_ratio is not None and norm is not None)
        else ""
    )
    cv2.putText(img, line1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if line2:
        cv2.putText(img, line2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def _draw_residual_chart(width: int, height: int, payload: Mapping[str, Any]):
    import cv2
    import numpy as np

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (28, 28, 28)
    matched = payload.get("matched_canonical_indices") if isinstance(payload.get("matched_canonical_indices"), list) else []
    residuals = payload.get("residual_per_key_px") if isinstance(payload.get("residual_per_key_px"), list) else []
    if not residuals or len(matched) != len(residuals):
        cv2.putText(img, "no per-anchor residuals stored", (12, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return img

    # Map canonical index 0..35 to bar-x in [margin, width-margin].
    margin_x, margin_y = 30, 18
    bar_area_h = height - 2 * margin_y
    n_canon = 36
    if not matched:
        return img
    max_r = max(max(float(r) for r in residuals), 6.0)
    bar_w = max(2, int((width - 2 * margin_x) / max(n_canon, 1)) - 1)

    # Axis line
    base_y = margin_y + bar_area_h
    cv2.line(img, (margin_x, base_y), (width - margin_x, base_y), (90, 90, 90), 1)
    # Reference horizontal lines at 3 and 5 px
    for ref_px, ref_color in [(3.0, (50, 180, 50)), (5.0, (60, 180, 220))]:
        ry = int(base_y - bar_area_h * (ref_px / max_r))
        cv2.line(img, (margin_x, ry), (width - margin_x, ry), ref_color, 1)
        cv2.putText(img, f"{ref_px:.0f}px", (4, ry + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ref_color, 1, cv2.LINE_AA)

    for ci, r in zip(matched, residuals):
        if not (0 <= int(ci) < n_canon):
            continue
        x = margin_x + int((width - 2 * margin_x) * (int(ci) + 0.5) / n_canon) - bar_w // 2
        bar_h = int(bar_area_h * (float(r) / max_r))
        color = (50, 200, 50) if float(r) < 3.0 else ((50, 200, 220) if float(r) < 5.0 else (60, 60, 220))
        cv2.rectangle(img, (x, base_y - bar_h), (x + bar_w, base_y), color, -1)

    cv2.putText(img, "residual per matched canonical key (px)", (margin_x, margin_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def _draw_polygons_on_median(median, payload: Mapping[str, Any], color):
    import cv2
    import numpy as np

    img = median.copy()
    h, w = img.shape[:2]
    polys = payload.get("key_polygons_source") if isinstance(payload.get("key_polygons_source"), list) else []
    midi_low = 21
    white_pitches = {0, 2, 4, 5, 7, 9, 11}
    for k, poly in enumerate(polys):
        midi = midi_low + k
        col = (50, 210, 80) if (midi % 12) in white_pitches else (255, 70, 50)
        try:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        except Exception:
            continue
        cv2.polylines(img, [pts], isClosed=True, color=col, thickness=1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 3)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--geometry-root", required=True, help="Directory with key_geometry/{split}/*.json")
    ap.add_argument("--out-root", required=True, help="Where to write visualizations")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--num-samples", type=int, default=24)
    args = ap.parse_args()

    import cv2
    import numpy as np

    from tivit.core.config import load_experiment_config
    from tivit.data.datasets.pianovam_impl import PianoVAMDataset
    from tivit.data.roi.keyboard_calibration import build_median_frame, MedianFrameOptions

    cfg = dict(load_experiment_config(args.config or [Path("configs/default.yaml")]))
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    dataset_cfg["preprocessed_format"] = "video"
    dataset_cfg["shuffle"] = False
    cfg["dataset"] = dataset_cfg

    geom_root = Path(args.geometry_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    median_opts = MedianFrameOptions(num_samples=int(args.num_samples), skip_seconds=1.0, max_seconds=None)

    counts = {"ok": 0, "accepted_loose": 0, "failed": 0, "errored": 0, "manual_fixed": 0}

    for split in args.splits:
        ds = PianoVAMDataset(cfg, split=split, full_cfg=cfg)
        for entry in ds.entries:
            video_id = entry.video_id
            json_path = geom_root / split / f"{video_id}.json"
            if not json_path.exists():
                print(f"[skip] {split}/{video_id}: no JSON")
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            status = str(payload.get("calibration_status") or "errored")
            counts[status] = counts.get(status, 0) + 1

            crop = _crop_tuple(entry.metadata.get("crop") if isinstance(entry.metadata, Mapping) else None)
            try:
                median = build_median_frame(Path(entry.video_path), crop=crop, options=median_opts)
            except Exception as exc:
                print(f"[FAIL] {split}/{video_id}: median frame error {exc}")
                continue

            color = _STATUS_COLORS.get(status, (200, 200, 200))
            row1 = _draw_polygons_on_median(median, payload, color)
            banner = _draw_status_banner(row1.shape[1], 70, payload, color)
            chart = _draw_residual_chart(row1.shape[1], 200, payload)
            stacked = np.concatenate([row1, banner, chart], axis=0)

            out_dir = out_root / split / status
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{video_id}.png"
            cv2.imwrite(str(out_path), stacked)
            print(f"[viz ] {split:5s}/{status:14s} {video_id} -> {out_path.name}")

    print()
    print(f"Wrote visualizations to {out_root}")
    print(
        f"counts: ok={counts['ok']} loose={counts['accepted_loose']} "
        f"failed={counts['failed']} manual_fixed={counts['manual_fixed']} errored={counts['errored']}"
    )


if __name__ == "__main__":
    main()
