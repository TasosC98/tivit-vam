#!/usr/bin/env python3
"""Per-video TSV<->video time-offset sweep using calibrated key polygons.

Why this is needed:
    The existing data/targets/av_sync.py uses bounding-box motion energy and
    runs per-clip during dataset sampling. With calibrated key_geometry/ now
    available, we can compute a much cleaner motion signal: the median-frame-
    differential intensity SUMMED INSIDE each calibrated key polygon. Comparing
    that signal's onsets against the TSV onset times under a sweep of time
    offsets pinpoints the per-video lag in milliseconds, which gets persisted
    so the dataset reads it deterministically at sample-time.

What this tool does:
    For each video that has a usable calibration:
      1. Load the calibrated key polygons in source coordinates.
      2. Compute a per-key motion-energy time series: |frame[t] - frame[t-1]|
         summed inside each polygon, sampled at decode_fps.
      3. Build a target time series: a Kronecker-delta-style 1 at each TSV
         onset time (per-key), zero elsewhere, blurred with a small Gaussian.
      4. For each candidate offset_ms in [-500, +500] (step 5 ms): shift the
         target series by `offset_ms` and compute Pearson correlation against
         the motion series. Pick the offset with maximum correlation.
      5. Save `video_time_offset_ms` and a `sync_quality` score (peak corr)
         into the existing key_geometry JSON so the dataset reads it.

Outputs:
    Updates each `key_geometry/{split}/{video_id}.json` to include:
        "sync": {
            "video_time_offset_ms": <int>,    # milliseconds; positive => video lags TSV
            "peak_correlation": <float>,      # 0..1; >0.4 is solid sync
            "sweep_max_ms": 500,
            "sweep_step_ms": 5,
            "fps_used": <float>,
            "n_frames_analyzed": <int>,
            "method": "polygon_motion_vs_onset_template",
            "status": "ok" | "low_correlation" | "errored"
        }
    Plus a top-level `sync_summary.csv` with per-video sync stats.

Usage:
    PYTHONPATH=. python tools/sync_sweep_per_video.py \\
        --config configs/default.yaml \\
        --config configs/overrides/manual_training.yaml \\
        --config configs/overrides/manual_training_keyroi_train.yaml \\
        --config configs/overrides/06-05-2026/manual_training_keyroi_video_06_05_2026.yaml \\
        --geometry-root /home/achatzigiannis/tivit-logs/manual_training_keyroi_video_06_05_2026/key_geometry \\
        --splits train valid test \\
        --max-videos 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Bypass the geometry-gate so we process every video.
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


def _build_polygon_motion_series(
    video_path: Path,
    *,
    crop: Optional[Tuple[int, int, int, int]],
    polygons_source: List[List[List[float]]],
    target_fps: float,
    max_frames: int,
):
    """Decode the video at target_fps; for each frame compute per-key motion
    energy = sum of |frame - prev_frame| inside each polygon.

    Returns:
        motion_series: numpy array shape (T, n_keys), float32
        timestamps_s: numpy array shape (T,), float32
        actual_fps: float (the FPS we actually decoded at, for sanity)
    """

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"unknown frame count for {video_path}")

    # We sample frames at target_fps. Step in original-video frames:
    step = max(1, int(round(src_fps / float(target_fps))))
    actual_fps = src_fps / float(step)
    n_keys = len(polygons_source)

    # Pre-build polygon masks (on the cropped frame coordinate system).
    if crop is not None:
        y0, y1, x0, x1 = crop
        crop_h, crop_w = max(0, y1 - y0), max(0, x1 - x0)
    else:
        # We'll resize masks once we know full-frame size from first decode.
        crop_h, crop_w = 0, 0
    masks = None  # build after first decode

    motion_series: List[np.ndarray] = []
    timestamps: List[float] = []
    prev_gray = None
    frame_idx = 0
    decoded = 0
    while frame_idx < total_frames and decoded < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if crop is not None:
            y0, y1, x0, x1 = crop
            frame = frame[max(0, y0): y1, max(0, x0): x1]
        if masks is None:
            crop_h, crop_w = frame.shape[:2]
            masks = np.zeros((n_keys, crop_h, crop_w), dtype=np.uint8)
            for k, poly in enumerate(polygons_source):
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(masks[k], [pts], 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = np.abs(gray - prev_gray)
            row = np.zeros(n_keys, dtype=np.float32)
            for k in range(n_keys):
                m = masks[k]
                if m.sum() > 0:
                    row[k] = float((diff * m).sum() / float(m.sum()))
            motion_series.append(row)
            timestamps.append(float(frame_idx) / src_fps)
        prev_gray = gray
        frame_idx += step
        decoded += 1

    cap.release()
    if not motion_series:
        raise RuntimeError(f"no motion frames decoded from {video_path}")
    return (
        np.stack(motion_series, axis=0),
        np.asarray(timestamps, dtype=np.float32),
        float(actual_fps),
    )


def _build_onset_template(
    timestamps_s,
    events: List[Tuple[float, float, int]],
    *,
    n_keys: int,
    note_min: int,
    sigma_s: float = 0.05,
):
    """Per-key onset envelope: for each (onset_t, _, pitch) place a Gaussian
    bump at t=onset_t, key=pitch-note_min, with std sigma_s seconds.

    Returns: numpy array shape (T, n_keys), float32.
    """

    import numpy as np

    T = int(timestamps_s.size)
    template = np.zeros((T, n_keys), dtype=np.float32)
    if T == 0 or not events:
        return template
    sigma_frames = max(1.0, sigma_s * (timestamps_s[1] - timestamps_s[0]) ** -1) if T > 1 else 1.0
    radius = int(round(3.0 * sigma_frames))
    for onset_t, _, pitch in events:
        try:
            key_idx = int(round(float(pitch))) - int(note_min)
        except (TypeError, ValueError):
            continue
        if key_idx < 0 or key_idx >= n_keys:
            continue
        # Find nearest sample
        idx = int(np.argmin(np.abs(timestamps_s - float(onset_t))))
        lo = max(0, idx - radius)
        hi = min(T, idx + radius + 1)
        if hi <= lo:
            continue
        rel = np.arange(lo, hi, dtype=np.float32) - float(idx)
        bump = np.exp(-0.5 * (rel / sigma_frames) ** 2)
        template[lo:hi, key_idx] = np.maximum(template[lo:hi, key_idx], bump.astype(np.float32))
    return template


def _correlation_at_lag(motion, template, lag_frames: int) -> float:
    """Pearson correlation between flattened motion and template, with the
    template shifted by `lag_frames` along the time axis (positive = template
    moves forward, i.e. video lags label by lag_frames)."""

    import numpy as np

    if motion.shape != template.shape:
        return -1.0
    T, K = motion.shape
    if abs(lag_frames) >= T:
        return -1.0
    if lag_frames >= 0:
        a = motion[lag_frames:, :].reshape(-1)
        b = template[: T - lag_frames, :].reshape(-1)
    else:
        a = motion[: T + lag_frames, :].reshape(-1)
        b = template[-lag_frames:, :].reshape(-1)
    if a.size == 0 or b.size == 0:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(float((a * a).sum()) * float((b * b).sum()))) + 1e-9
    return float((a * b).sum() / denom)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--geometry-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=2400, help="cap frames per video to keep tool fast (~80 s @30fps)")
    ap.add_argument("--sweep-max-ms", type=int, default=500)
    ap.add_argument("--sweep-step-ms", type=int, default=5)
    ap.add_argument("--low-correlation-threshold", type=float, default=0.20)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import numpy as np

    from tivit.core.config import load_experiment_config
    from tivit.data.datasets.pianovam_impl import PianoVAMDataset

    cfg = dict(load_experiment_config(args.config or [Path("configs/default.yaml")]))
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    dataset_cfg["preprocessed_format"] = "video"
    dataset_cfg["shuffle"] = False
    cfg["dataset"] = dataset_cfg
    note_min = int((dataset_cfg.get("frame_targets", {}) or {}).get("note_min", 21))

    geom_root = Path(args.geometry_root).expanduser().resolve()
    if not geom_root.exists():
        raise SystemExit(f"geometry root not found: {geom_root}")

    summary_rows: List[Dict[str, Any]] = []
    started = time.time()

    for split in args.splits:
        ds = PianoVAMDataset(cfg, split=split, full_cfg=cfg)
        limit = len(ds) if args.max_videos <= 0 else min(len(ds), args.max_videos)
        for i in range(limit):
            entry = ds.entries[i]
            video_id = entry.video_id
            geom_path = geom_root / split / f"{video_id}.json"
            if not geom_path.exists():
                print(f"[skip] {split}/{video_id}: no geometry JSON")
                continue
            payload = json.loads(geom_path.read_text(encoding="utf-8"))
            existing_sync = payload.get("sync") if isinstance(payload.get("sync"), dict) else None
            if existing_sync and not args.overwrite:
                print(f"[skip] {split}/{video_id}: sync already computed (offset={existing_sync.get('video_time_offset_ms')} ms)")
                summary_rows.append(
                    {
                        "split": split,
                        "video_id": video_id,
                        "video_time_offset_ms": existing_sync.get("video_time_offset_ms"),
                        "peak_correlation": existing_sync.get("peak_correlation"),
                        "status": existing_sync.get("status"),
                        "from_cache": True,
                    }
                )
                continue
            polygons_source = payload.get("key_polygons_source")
            if not isinstance(polygons_source, list) or not polygons_source:
                print(f"[skip] {split}/{video_id}: no key_polygons_source in geometry JSON")
                continue
            n_keys = len(polygons_source)
            crop = _crop_tuple(entry.metadata.get("crop") if isinstance(entry.metadata, Mapping) else None)

            # Read events
            try:
                raw = ds._read_labels(entry)
            except Exception as exc:
                print(f"[ERR ] {split}/{video_id}: label read failed ({exc})")
                continue
            events = raw.get("events", []) if isinstance(raw, Mapping) else []
            if not events:
                print(f"[skip] {split}/{video_id}: no labelled onsets")
                continue

            # Run motion + onset template + correlation sweep
            t0 = time.time()
            try:
                motion, timestamps, fps_used = _build_polygon_motion_series(
                    Path(entry.video_path),
                    crop=crop,
                    polygons_source=polygons_source,
                    target_fps=float(args.target_fps),
                    max_frames=int(args.max_frames),
                )
            except Exception as exc:
                print(f"[ERR ] {split}/{video_id}: motion build failed ({exc})")
                continue

            template = _build_onset_template(timestamps, events, n_keys=n_keys, note_min=note_min)

            best_corr = -1.0
            best_lag_ms = 0
            sweep_step_ms = max(1, int(args.sweep_step_ms))
            sweep_max_ms = max(sweep_step_ms, int(args.sweep_max_ms))
            ms_per_frame = 1000.0 / float(fps_used)
            for lag_ms in range(-sweep_max_ms, sweep_max_ms + 1, sweep_step_ms):
                lag_frames = int(round(float(lag_ms) / ms_per_frame))
                corr = _correlation_at_lag(motion, template, lag_frames)
                if corr > best_corr:
                    best_corr = corr
                    best_lag_ms = lag_ms

            status = "ok" if best_corr >= float(args.low_correlation_threshold) else "low_correlation"
            sync_payload = {
                "video_time_offset_ms": int(best_lag_ms),
                "peak_correlation": float(best_corr),
                "sweep_max_ms": int(sweep_max_ms),
                "sweep_step_ms": int(sweep_step_ms),
                "fps_used": float(fps_used),
                "n_frames_analyzed": int(motion.shape[0]),
                "method": "polygon_motion_vs_onset_template",
                "status": status,
            }
            payload["sync"] = sync_payload
            geom_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            elapsed = time.time() - t0
            tag = "OK   " if status == "ok" else "LOW  "
            print(
                f"[sync] {tag} {split:5s} {i + 1:3d}/{limit:3d} {video_id:30s} "
                f"offset={best_lag_ms:+5d}ms corr={best_corr:.3f} frames={motion.shape[0]} {elapsed:4.1f}s"
            )
            summary_rows.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "video_time_offset_ms": int(best_lag_ms),
                    "peak_correlation": float(best_corr),
                    "status": status,
                    "from_cache": False,
                }
            )

    out_path = geom_root / "sync_summary.csv"
    if summary_rows:
        fields = list({k for row in summary_rows for k in row.keys()})
        fields = ["split", "video_id", "video_time_offset_ms", "peak_correlation", "status", "from_cache"]
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in summary_rows:
                w.writerow({k: row.get(k) for k in fields})

    elapsed = time.time() - started
    n_ok = sum(1 for r in summary_rows if r.get("status") == "ok")
    n_low = sum(1 for r in summary_rows if r.get("status") == "low_correlation")
    print()
    print(f"Wrote sync info into geometry JSONs and {out_path}")
    print(f"counts: ok={n_ok} low_correlation={n_low} total={len(summary_rows)} elapsed={elapsed:.1f}s")

    if summary_rows:
        offsets = [r["video_time_offset_ms"] for r in summary_rows if r.get("video_time_offset_ms") is not None]
        if offsets:
            offsets_sorted = sorted(offsets)
            n = len(offsets_sorted)
            print(
                f"offset distribution: min={offsets_sorted[0]} ms  "
                f"median={offsets_sorted[n // 2]} ms  max={offsets_sorted[-1]} ms"
            )


if __name__ == "__main__":
    main()
