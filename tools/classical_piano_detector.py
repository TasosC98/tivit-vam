#!/usr/bin/env python3
"""Deterministic per-key motion detector for piano transcription.

This bypasses the trained model ENTIRELY. Pipeline:

    1. Parse the keyboard quadrilateral corners from PianoVAM metadata
       (Point_LT, Point_RT, Point_RB, Point_LB).
    2. Compute a homography that rectifies the keyboard quadrilateral
       to a canonical rectangle of size (H_canon, W_canon) = (180, 1536).
    3. Warp every video frame to canonical coordinates.
    4. Generate 88 uniform key polygons in canonical space.
    5. For each frame transition (t -> t+1), compute per-key motion
       energy = mean |frame[t+1] - frame[t]| inside the BOTTOM HALF of
       each key polygon (where the key tip moves, hand rarely reaches).
    6. Apply temporal smoothing then peak-detection per key to extract
       onset events: each peak = (time, midi_pitch).
    7. Compare to the TSV ground truth with onset tolerance ±50 ms.
       Report precision / recall / F1, plus per-pitch and white/black
       breakdowns.

Why this is the right baseline:
    - Marginal-prediction collapse is impossible: each key's prediction
      is computed independently from its own polygon's motion. There's
      no shared classifier to collapse to a constant.
    - The signal is real and well-documented: a piano key tip descends
      ~5 px when pressed, producing a measurable motion spike in the
      bottom of the polygon for ~50-150 ms.
    - It directly leverages the geometry calibration we spent weeks on.
    - If this gives F1 > 0.4 it proves the data + geometry are FINE,
      and the trained ViViT is the bottleneck.

Usage:
    PYTHONPATH=. python tools/classical_piano_detector.py \\
        --video /home/achatzigiannis/tivit-logs/2024-09-05_21-31-00.mp4 \\
        --tsv   /home/achatzigiannis/tivit-logs/2024-09-05_21-31-00.tsv \\
        --metadata-corners "253,441;1648,444;1651,602;249,598" \\
        --out-dir /home/achatzigiannis/tivit-logs/classical_baseline_14_05_2026 \\
        --target-fps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------------
# Canonical layout (same as our trained-model pipeline so we can compare
# apples to apples on the same key indexing).
# ----------------------------------------------------------------------------

_MIDI_LOW = 21
_MIDI_HIGH = 108
_N_KEYS = _MIDI_HIGH - _MIDI_LOW + 1   # 88
_CANON_H = 180
_CANON_W = 1536
_WHITE_PITCHES_MOD12 = {0, 2, 4, 5, 7, 9, 11}


def _is_white(midi: int) -> bool:
    return (int(midi) % 12) in _WHITE_PITCHES_MOD12


def _canonical_key_bounds() -> List[Tuple[float, float]]:
    """Return per-key [left, right] x-bounds in canonical coordinates."""

    white_w = _CANON_W / 52.0
    edges = [i * white_w for i in range(53)]
    bounds: List[Tuple[float, float]] = []
    white_idx = 0
    for k in range(_N_KEYS):
        midi = _MIDI_LOW + k
        if _is_white(midi):
            bounds.append((edges[white_idx], edges[white_idx + 1]))
            white_idx += 1
        else:
            l_center = 0.5 * (edges[white_idx - 1] + edges[white_idx])
            r_center = 0.5 * (edges[white_idx] + edges[white_idx + 1])
            center = 0.5 * (l_center + r_center)
            half = 0.3 * white_w
            bounds.append((center - half, center + half))
    return bounds


# ----------------------------------------------------------------------------
# Metadata parsing
# ----------------------------------------------------------------------------


def _parse_corner_str(s: str) -> Tuple[float, float]:
    """Parse 'x, y' style point string."""
    parts = [p.strip() for p in s.replace(",", " ").split()]
    if len(parts) < 2:
        raise ValueError(f"cannot parse corner '{s}'")
    return float(parts[0]), float(parts[1])


def _parse_corners_arg(arg: str) -> List[Tuple[float, float]]:
    """Parse '253,441;1648,444;1651,602;249,598' (LT;RT;RB;LB order)."""
    parts = arg.split(";")
    if len(parts) != 4:
        raise ValueError("--metadata-corners must contain 4 'x,y' points separated by ';'")
    return [_parse_corner_str(p) for p in parts]


def _parse_tsv(path: Path) -> List[Tuple[float, float, int]]:
    """Return list of (onset_s, key_offset_s, midi_pitch)."""
    events: List[Tuple[float, float, int]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                parts = line.split()
            if len(parts) < 4:
                continue
            try:
                onset = float(parts[0])
                key_off = float(parts[1])
                pitch = int(round(float(parts[3])))
                events.append((onset, key_off, pitch))
            except (ValueError, IndexError):
                continue
    return events


# ----------------------------------------------------------------------------
# Video decoding + per-key motion extraction
# ----------------------------------------------------------------------------


def _compute_homography_and_warp_size(
    corners_src: List[Tuple[float, float]],
):
    """Build homography mapping source quadrilateral -> canonical rectangle.

    corners_src order: [LT, RT, RB, LB] (top-left, top-right, bot-right, bot-left).
    Canonical rectangle: (0,0) (W,0) (W,H) (0,H).
    """

    import cv2
    import numpy as np

    src = np.array(corners_src, dtype=np.float32)
    dst = np.array(
        [[0, 0], [_CANON_W - 1, 0], [_CANON_W - 1, _CANON_H - 1], [0, _CANON_H - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def _build_key_masks() -> "np.ndarray":
    """Return (88, H, W) uint8 masks for the bottom-half of each key polygon."""

    import cv2
    import numpy as np

    bounds = _canonical_key_bounds()
    masks = np.zeros((_N_KEYS, _CANON_H, _CANON_W), dtype=np.uint8)
    # Use the BOTTOM HALF of each polygon. For white keys, y range = [H/2, H].
    # For black keys, the polygon only spans the upper 60% of H, so we use
    # the bottom half of that = [0.3H, 0.6H]. Hand rarely reaches there;
    # key motion is detectable.
    for k in range(_N_KEYS):
        midi = _MIDI_LOW + k
        l, r = bounds[k]
        if _is_white(midi):
            y0, y1 = int(0.55 * _CANON_H), int(0.95 * _CANON_H)
        else:
            y0, y1 = int(0.30 * _CANON_H), int(0.55 * _CANON_H)
        masks[k, y0:y1, int(l):int(r)] = 1
    return masks


def _extract_motion_series(
    video_path: Path,
    H_warp: "np.ndarray",
    masks: "np.ndarray",
    *,
    target_fps: float,
    max_seconds: Optional[float],
):
    """Decode the video at target_fps, warp each frame to canonical, then
    compute per-key motion energy = mean |frame[t]-frame[t-1]| inside each
    key's bottom-half mask.

    Returns (motion_series, timestamps, fps_used).
    """

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / float(target_fps))))
    actual_fps = src_fps / float(step)
    if max_seconds is not None and max_seconds > 0:
        max_frames = int(round(max_seconds * actual_fps))
    else:
        max_frames = (total_frames + step - 1) // step

    # Pre-flatten masks for fast inner product: (88, H*W)
    masks_flat = masks.reshape(_N_KEYS, -1).astype(np.float32)
    mask_areas = masks_flat.sum(axis=1)
    mask_areas[mask_areas == 0] = 1.0

    motion = np.zeros((max_frames, _N_KEYS), dtype=np.float32)
    timestamps = np.zeros(max_frames, dtype=np.float32)
    prev_warp_gray = None
    frame_idx = 0
    decoded = 0
    while frame_idx < total_frames and decoded < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        warp = cv2.warpPerspective(frame, H_warp, (_CANON_W, _CANON_H), flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_warp_gray is not None:
            diff = np.abs(gray - prev_warp_gray).reshape(-1)
            # per-key mean motion energy = (mask . diff) / mask_area
            energy = masks_flat @ diff  # shape (88,)
            motion[decoded] = energy / mask_areas
            timestamps[decoded] = float(frame_idx) / src_fps
        prev_warp_gray = gray
        frame_idx += step
        decoded += 1

    cap.release()
    motion = motion[:decoded]
    timestamps = timestamps[:decoded]
    return motion, timestamps, float(actual_fps)


# ----------------------------------------------------------------------------
# Onset extraction
# ----------------------------------------------------------------------------


def _smooth_1d(signal, kernel_size: int = 5):
    import numpy as np
    if signal.size < kernel_size:
        return signal
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(signal, kernel, mode="same").astype(np.float32)


def _detect_onsets_per_key(
    motion,
    timestamps,
    *,
    smoothing_window: int,
    peak_threshold_factor: float,
    min_event_gap_s: float,
):
    """For each key column, smooth and run a simple peak detector. Return list
    of (time_s, key_index, peak_value).
    """

    import numpy as np

    T, K = motion.shape
    events: List[Tuple[float, int, float]] = []
    fps = (T - 1) / max(float(timestamps[-1] - timestamps[0]), 1e-6) if T > 1 else 30.0
    min_gap_frames = max(1, int(round(min_event_gap_s * fps)))
    for k in range(K):
        s = _smooth_1d(motion[:, k], smoothing_window)
        # Adaptive threshold per key: mean + factor * std of the smoothed series.
        s_mean = float(s.mean())
        s_std = float(s.std())
        thr = s_mean + float(peak_threshold_factor) * s_std
        last_peak = -10**9
        for t in range(1, T - 1):
            if s[t] < thr:
                continue
            if s[t] <= s[t - 1] or s[t] < s[t + 1]:
                # local maximum required (with tie-break tolerance for plateaus)
                continue
            if t - last_peak < min_gap_frames:
                # Stronger peak wins
                if s[t] > s[last_peak]:
                    # Replace previous
                    events = [e for e in events if not (e[1] == k and abs(e[0] - timestamps[last_peak]) < 1e-6)]
                    last_peak = t
                    events.append((float(timestamps[t]), int(k), float(s[t])))
                continue
            last_peak = t
            events.append((float(timestamps[t]), int(k), float(s[t])))
    events.sort(key=lambda e: e[0])
    return events


# ----------------------------------------------------------------------------
# Note-level F1
# ----------------------------------------------------------------------------


def _note_f1(
    pred_events: List[Tuple[float, int]],
    ref_events: List[Tuple[float, float, int]],
    *,
    onset_tol_s: float,
):
    """Standard piano transcription F1: pitch must match, onset within tolerance.

    pred_events: list of (onset_s, midi_pitch).
    ref_events:  list of (onset_s, offset_s, midi_pitch).
    """

    used = [False] * len(ref_events)
    tp = 0
    fp = 0
    pred_sorted = sorted(pred_events, key=lambda e: e[0])
    ref_sorted = sorted(enumerate(ref_events), key=lambda x: x[1][0])
    # Group refs by pitch for fast lookup
    ref_by_pitch: Dict[int, List[Tuple[int, float, float, int]]] = {}
    for i, (on, off, p) in enumerate(ref_events):
        ref_by_pitch.setdefault(int(p), []).append((i, float(on), float(off), int(p)))

    pitch_tp = {}
    pitch_fp = {}
    pitch_fn = {}

    for pred_t, pred_p in pred_sorted:
        cands = ref_by_pitch.get(int(pred_p), [])
        best_idx = -1
        best_dt = float("inf")
        for i, r_on, r_off, _ in cands:
            if used[i]:
                continue
            dt = abs(float(pred_t) - r_on)
            if dt <= float(onset_tol_s) and dt < best_dt:
                best_dt = dt
                best_idx = i
        if best_idx >= 0:
            used[best_idx] = True
            tp += 1
            pitch_tp[pred_p] = pitch_tp.get(pred_p, 0) + 1
        else:
            fp += 1
            pitch_fp[pred_p] = pitch_fp.get(pred_p, 0) + 1
    fn = sum(1 for u in used if not u)
    for i, ev in enumerate(ref_events):
        if not used[i]:
            pitch_fn[int(ev[2])] = pitch_fn.get(int(ev[2]), 0) + 1

    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    # White/black breakdown
    def _split(d):
        white_tp = sum(v for k, v in d.items() if _is_white(k))
        black_tp = sum(v for k, v in d.items() if not _is_white(k))
        return white_tp, black_tp

    w_tp, b_tp = _split(pitch_tp)
    w_fp, b_fp = _split(pitch_fp)
    w_fn, b_fn = _split(pitch_fn)
    w_f1 = 2 * w_tp / max(2 * w_tp + w_fp + w_fn, 1)
    b_f1 = 2 * b_tp / max(2 * b_tp + b_fp + b_fn, 1)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "n_pred": int(len(pred_events)),
        "n_ref": int(len(ref_events)),
        "f1_white": float(w_f1),
        "f1_black": float(b_f1),
        "white_tp": int(w_tp),
        "white_fp": int(w_fp),
        "white_fn": int(w_fn),
        "black_tp": int(b_tp),
        "black_fp": int(b_fp),
        "black_fn": int(b_fn),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument(
        "--metadata-corners",
        required=True,
        help="'LT_x,LT_y;RT_x,RT_y;RB_x,RB_y;LB_x,LB_y' e.g. '253,441;1648,444;1651,602;249,598'",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 = process whole video")
    ap.add_argument("--smoothing-window", type=int, default=3)
    ap.add_argument("--peak-threshold-factor", type=float, default=3.0,
                    help="onset peak threshold = mean + factor*std of per-key motion")
    ap.add_argument("--min-event-gap-s", type=float, default=0.10,
                    help="minimum gap between events at the same pitch")
    ap.add_argument("--onset-tolerance-ms", type=float, default=50.0)
    args = ap.parse_args()

    import cv2
    import numpy as np

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    corners = _parse_corners_arg(args.metadata_corners)
    print(f"keyboard corners (LT,RT,RB,LB) = {corners}")

    H_warp = _compute_homography_and_warp_size(corners)
    masks = _build_key_masks()

    motion, timestamps, fps_used = _extract_motion_series(
        Path(args.video),
        H_warp,
        masks,
        target_fps=float(args.target_fps),
        max_seconds=float(args.max_seconds) if args.max_seconds > 0 else None,
    )
    print(f"decoded {motion.shape[0]} frames @ {fps_used:.2f} fps; "
          f"motion shape={motion.shape}, range=[{motion.min():.3f}, {motion.max():.3f}]")

    raw_events = _detect_onsets_per_key(
        motion,
        timestamps,
        smoothing_window=int(args.smoothing_window),
        peak_threshold_factor=float(args.peak_threshold_factor),
        min_event_gap_s=float(args.min_event_gap_s),
    )
    pred_events = [(t, int(k) + _MIDI_LOW) for (t, k, _v) in raw_events]
    ref_events = _parse_tsv(Path(args.tsv))
    print(f"detected {len(pred_events)} onsets; ground truth has {len(ref_events)} notes")

    onset_tol_s = float(args.onset_tolerance_ms) / 1000.0
    f1_result = _note_f1(pred_events, ref_events, onset_tol_s=onset_tol_s)
    print()
    print(f"=== Note-level F1 (tolerance ±{args.onset_tolerance_ms:.0f} ms) ===")
    print(json.dumps(f1_result, indent=2))

    # Write artifacts
    (out_dir / "predictions.csv").write_text(
        "\n".join([f"{t:.4f},{p}" for (t, p) in pred_events]),
        encoding="utf-8",
    )
    (out_dir / "ground_truth.csv").write_text(
        "\n".join([f"{on:.4f},{off:.4f},{p}" for (on, off, p) in ref_events]),
        encoding="utf-8",
    )
    (out_dir / "result.json").write_text(json.dumps({
        "video": str(args.video),
        "tsv": str(args.tsv),
        "params": {
            "target_fps": float(args.target_fps),
            "fps_used": float(fps_used),
            "smoothing_window": int(args.smoothing_window),
            "peak_threshold_factor": float(args.peak_threshold_factor),
            "min_event_gap_s": float(args.min_event_gap_s),
            "onset_tolerance_ms": float(args.onset_tolerance_ms),
        },
        "frames_decoded": int(motion.shape[0]),
        "metrics": f1_result,
    }, indent=2), encoding="utf-8")

    # Diagnostic plot: per-key motion heatmap + detected onsets + GT onsets
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.imshow(motion.T, aspect="auto", origin="lower", cmap="viridis",
                  extent=[timestamps[0], timestamps[-1] if timestamps.size else 1.0, 0, _N_KEYS])
        # Overlay detected events (white) and GT events (red)
        pred_t = [e[0] for e in pred_events]
        pred_k = [e[1] - _MIDI_LOW for e in pred_events]
        ax.scatter(pred_t, pred_k, s=20, marker="o", facecolors="none",
                   edgecolors="white", label=f"detected ({len(pred_events)})")
        gt_t = [e[0] for e in ref_events]
        gt_k = [int(e[2]) - _MIDI_LOW for e in ref_events]
        ax.scatter(gt_t, gt_k, s=20, marker="x", c="red",
                   label=f"GT ({len(ref_events)})")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("key index (0=A0, 87=C8)")
        ax.set_title(
            f"Classical per-key motion detector\n"
            f"F1={f1_result['f1']:.3f}  P={f1_result['precision']:.3f}  "
            f"R={f1_result['recall']:.3f}  TP={f1_result['tp']} FP={f1_result['fp']} FN={f1_result['fn']}"
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "motion_heatmap.png", dpi=120)
        plt.close(fig)

        # Save a per-key time series for one key around middle C (key 39)
        fig, ax = plt.subplots(figsize=(14, 4))
        sample_key = 39  # MIDI 60 (middle C)
        ax.plot(timestamps, motion[:, sample_key], color="steelblue", label=f"motion key{sample_key} (MIDI 60)")
        # GT onsets at this pitch
        gt_at_pitch = [on for (on, off, p) in ref_events if int(p) == sample_key + _MIDI_LOW]
        for t in gt_at_pitch:
            ax.axvline(t, color="red", alpha=0.5, linewidth=1)
        pred_at_pitch = [t for (t, p) in pred_events if p == sample_key + _MIDI_LOW]
        for t in pred_at_pitch:
            ax.axvline(t, color="green", alpha=0.5, linewidth=1, linestyle="--")
        ax.set_title(f"Motion at MIDI 60 (middle C). red=GT onsets, green=detected onsets")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("motion energy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "midi60_timeseries.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"plotting skipped ({exc})")

    print()
    print(f"Wrote artifacts to {out_dir}")
    print(f"  result.json, predictions.csv, ground_truth.csv, motion_heatmap.png, midi60_timeseries.png")


if __name__ == "__main__":
    main()
