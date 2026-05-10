#!/usr/bin/env python3
"""Per-video TSV<->video time-offset sweep using calibrated key polygons.

v2 (2026-05-10): the v1 motion signal (whole-polygon |frame[t]-frame[t-1]|)
was dominated by hand crossing rather than key motion. Hand traversal of a
key polygon produces ~30x more pixel-difference than the key tip descending
~5 px. Result was peak correlations of 0.02-0.19 across all 71 videos.

v2 changes:
    1. Use only the BOTTOM 25% of each key polygon (key-tip region, where
       hands rarely reach). This suppresses hand motion.
    2. Compute |frame[t] - median_frame| (deviation from background) rather
       than |frame[t] - frame[t-1]|. The median frame is the "no hands, no
       presses" baseline; deviation from it spikes when a key is pressed.
    3. Aggregate motion across ALL keys into one 1-D signal. Correlate
       against an aggregate onset-density template. Single-channel
       correlation is more robust than 88 noisy per-key channels.
    4. Process at most 30 s of video per pass. Old version processed 80 s
       per video × 71 videos = 5+ hours; new version is ~30 s per video
       (~35 min total).
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


def _build_aggregate_motion_series(
    video_path: Path,
    *,
    crop: Optional[Tuple[int, int, int, int]],
    polygons_source: List[List[List[float]]],
    target_fps: float,
    max_seconds: float,
    bottom_fraction: float,
):
    """Decode the video at target_fps; for each frame compute aggregate
    motion energy = sum of |frame - median_frame| inside the bottom
    `bottom_fraction` of each key polygon, summed across all keys.

    Returns:
        motion_series: numpy array shape (T,), float32
        timestamps_s: numpy array shape (T,), float32
        actual_fps: float
        median_frame: numpy array (cropped frame shape), float32
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

    step = max(1, int(round(src_fps / float(target_fps))))
    actual_fps = src_fps / float(step)
    max_frames = max(60, int(round(max_seconds * actual_fps)))
    n_keys = len(polygons_source)

    # Pass 1: collect a sample of frames to build the median image.
    # We sample every Nth frame across the chosen window, capped at 24 frames.
    median_sample_count = 24
    median_step = max(1, max_frames // median_sample_count)
    sampled = []
    frame_idx = 0
    decoded = 0
    while frame_idx < total_frames and decoded < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            if crop is not None:
                y0, y1, x0, x1 = crop
                frame = frame[max(0, y0): y1, max(0, x0): x1]
            if decoded % median_step == 0 and len(sampled) < median_sample_count:
                sampled.append(frame.copy())
        frame_idx += step
        decoded += 1
    if not sampled:
        cap.release()
        raise RuntimeError("no frames sampled for median")
    h_min = min(f.shape[0] for f in sampled)
    w_min = min(f.shape[1] for f in sampled)
    sampled = [f[:h_min, :w_min] for f in sampled]
    median_frame = np.median(np.stack(sampled, axis=0), axis=0).astype(np.float32)
    median_gray = cv2.cvtColor(median_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    crop_h, crop_w = median_gray.shape

    # Build a single mask covering the BOTTOM bottom_fraction of every polygon.
    # Hands rarely reach this band — it's where the key tips live.
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    for poly in polygons_source:
        pts = np.asarray(poly, dtype=np.float32)
        if pts.size == 0:
            continue
        y_min = float(pts[:, 1].min())
        y_max = float(pts[:, 1].max())
        y_cut = y_min + (1.0 - float(bottom_fraction)) * (y_max - y_min)
        # Clip the polygon to bottom band by replacing y-coords below y_cut.
        clipped = pts.copy()
        clipped[:, 1] = np.maximum(clipped[:, 1], y_cut)
        cv2.fillPoly(mask, [clipped.astype(np.int32).reshape(-1, 1, 2)], 1)
    mask_area = float(mask.sum())
    if mask_area <= 0:
        cap.release()
        raise RuntimeError("bottom-region mask has zero area")

    # Pass 2: compute aggregate motion at every sampled frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    motion_series: List[float] = []
    timestamps: List[float] = []
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
        frame = frame[:crop_h, :crop_w]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(gray - median_gray)
        # Aggregate: sum |frame - median| inside the bottom-band mask.
        motion = float((diff * mask).sum() / mask_area)
        motion_series.append(motion)
        timestamps.append(float(frame_idx) / src_fps)
        frame_idx += step
        decoded += 1
    cap.release()
    if not motion_series:
        raise RuntimeError("no motion frames")
    return (
        np.asarray(motion_series, dtype=np.float32),
        np.asarray(timestamps, dtype=np.float32),
        float(actual_fps),
        median_gray,
    )


def _build_aggregate_onset_template(
    timestamps_s,
    events: List[Tuple[float, float, int]],
    *,
    sigma_s: float = 0.05,
):
    """Aggregate onset density: sum of Gaussian bumps at every onset time
    (regardless of pitch). Returns 1-D array shape (T,)."""

    import numpy as np

    T = int(timestamps_s.size)
    template = np.zeros(T, dtype=np.float32)
    if T < 2 or not events:
        return template
    dt = float(timestamps_s[1] - timestamps_s[0])
    sigma_frames = max(1.0, sigma_s / max(dt, 1e-6))
    radius = int(round(3.0 * sigma_frames))
    for onset_t, _, _ in events:
        try:
            t = float(onset_t)
        except (TypeError, ValueError):
            continue
        # nearest sample
        if t < float(timestamps_s[0]) or t > float(timestamps_s[-1]):
            continue
        idx = int(np.argmin(np.abs(timestamps_s - t)))
        lo = max(0, idx - radius)
        hi = min(T, idx + radius + 1)
        rel = np.arange(lo, hi, dtype=np.float32) - float(idx)
        bump = np.exp(-0.5 * (rel / sigma_frames) ** 2)
        template[lo:hi] += bump
    return template


def _correlation_at_lag(motion, template, lag_frames: int) -> float:
    import numpy as np
    if motion.shape != template.shape:
        return -1.0
    T = motion.shape[0]
    if abs(lag_frames) >= T:
        return -1.0
    if lag_frames >= 0:
        a = motion[lag_frames:]
        b = template[: T - lag_frames]
    else:
        a = motion[: T + lag_frames]
        b = template[-lag_frames:]
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
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="seconds of video per pass (was 80 s in v1; 30 s is plenty for ~hundreds of onsets)")
    ap.add_argument("--bottom-fraction", type=float, default=0.25,
                    help="fraction of each polygon's height to use (key-tip region)")
    ap.add_argument("--sweep-max-ms", type=int, default=500)
    ap.add_argument("--sweep-step-ms", type=int, default=5)
    ap.add_argument("--low-correlation-threshold", type=float, default=0.25,
                    help="below this, the result is rejected and lag_ms=0 will be used at training time")
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
                print(f"[skip] {split}/{video_id}: sync already computed")
                summary_rows.append({
                    "split": split, "video_id": video_id,
                    "video_time_offset_ms": existing_sync.get("video_time_offset_ms"),
                    "peak_correlation": existing_sync.get("peak_correlation"),
                    "status": existing_sync.get("status"),
                    "from_cache": True,
                })
                continue
            polygons_source = payload.get("key_polygons_source")
            if not isinstance(polygons_source, list) or not polygons_source:
                print(f"[skip] {split}/{video_id}: no key_polygons_source")
                continue
            crop = _crop_tuple(entry.metadata.get("crop") if isinstance(entry.metadata, Mapping) else None)
            try:
                raw = ds._read_labels(entry)
            except Exception as exc:
                print(f"[ERR ] {split}/{video_id}: label read failed ({exc})")
                continue
            events = raw.get("events", []) if isinstance(raw, Mapping) else []
            if not events:
                print(f"[skip] {split}/{video_id}: no labelled onsets")
                continue

            t0 = time.time()
            try:
                motion, timestamps, fps_used, _ = _build_aggregate_motion_series(
                    Path(entry.video_path),
                    crop=crop,
                    polygons_source=polygons_source,
                    target_fps=float(args.target_fps),
                    max_seconds=float(args.max_seconds),
                    bottom_fraction=float(args.bottom_fraction),
                )
            except Exception as exc:
                print(f"[ERR ] {split}/{video_id}: motion build failed ({exc})")
                continue

            # Filter events to the time window we actually decoded.
            t_min = float(timestamps[0])
            t_max = float(timestamps[-1])
            events_in_window = [(o, off, p) for (o, off, p) in events if t_min <= float(o) <= t_max]
            template = _build_aggregate_onset_template(timestamps, events_in_window)

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
            payload["sync"] = {
                "video_time_offset_ms": int(best_lag_ms),
                "peak_correlation": float(best_corr),
                "sweep_max_ms": int(sweep_max_ms),
                "sweep_step_ms": int(sweep_step_ms),
                "fps_used": float(fps_used),
                "n_frames_analyzed": int(motion.shape[0]),
                "n_onsets_in_window": int(len(events_in_window)),
                "method": "v2_aggregate_motion_vs_median_bottom25",
                "status": status,
            }
            geom_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tag = "OK   " if status == "ok" else "LOW  "
            print(
                f"[sync] {tag} {split:5s} {i + 1:3d}/{limit:3d} {video_id:30s} "
                f"offset={best_lag_ms:+5d}ms corr={best_corr:.3f} "
                f"frames={motion.shape[0]} onsets={len(events_in_window)} {time.time() - t0:4.1f}s"
            )
            summary_rows.append({
                "split": split, "video_id": video_id,
                "video_time_offset_ms": int(best_lag_ms),
                "peak_correlation": float(best_corr),
                "status": status, "from_cache": False,
            })

    out_path = geom_root / "sync_summary.csv"
    if summary_rows:
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
        offsets = [r["video_time_offset_ms"] for r in summary_rows if r.get("status") == "ok"]
        if offsets:
            offsets_sorted = sorted(offsets)
            n = len(offsets_sorted)
            print(
                f"OK offset distribution: min={offsets_sorted[0]} ms  "
                f"median={offsets_sorted[n // 2]} ms  max={offsets_sorted[-1]} ms"
            )
        else:
            print("(no OK syncs — training will use lag_ms=0 for all videos)")


if __name__ == "__main__":
    main()
