#!/usr/bin/env python3
"""Classical per-key piano detector v2 — background subtraction + narrow tip
strip + spatial non-max suppression.

v1 (tools/classical_piano_detector.py) got F1 = 0.024 on the test video.
Diagnosis from the heatmap + time-series PNGs:
    - Hand motion (sweeping across keys to reach the target) dominates the
      frame-to-frame difference signal.
    - Adjacent-key crosstalk: motion at key K bleeds into K-1 / K+1.
    - Bursts of detections at each press.

v2 changes:
    1. BACKGROUND SUBTRACTION. Compute a clip-wide median frame (= hand
       averaged out + keys at rest). Per-frame signal = |frame[t] - median|
       INSIDE each key's narrow tip strip. Sustained deviation from rest
       reveals an actual key press (the tip is descended). Hand motion,
       being non-stationary, averages OUT of the median.
    2. NARROW TIP STRIP. Bottom 5-10% of each key polygon only (the
       physical key tip, ~3-5 px tall). Fingers rarely cover the front
       3 px of the tip during normal play.
    3. SPATIAL NMS. After per-key peak detection, if keys K-1 / K / K+1
       all have peaks within ±30 ms, keep only the strongest.
    4. PEAK PROMINENCE. Use peak prominence (height above surrounding
       baseline) instead of raw threshold. Prominence > P times the
       local median absolute deviation.

Usage (same args as v1):
    PYTHONPATH=. python tools/classical_piano_detector_v2.py \\
        --video /home/achatzigiannis/tivit-logs/2024-09-05_21-31-00.mp4 \\
        --tsv   /home/achatzigiannis/tivit-logs/2024-09-05_21-31-00.tsv \\
        --metadata-corners "253,441;1648,444;1651,602;249,598" \\
        --out-dir /home/achatzigiannis/tivit-logs/classical_baseline_14_05_2026/single_v2 \\
        --target-fps 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_MIDI_LOW = 21
_MIDI_HIGH = 108
_N_KEYS = _MIDI_HIGH - _MIDI_LOW + 1   # 88
_CANON_H = 180
_CANON_W = 1536
_WHITE_PITCHES_MOD12 = {0, 2, 4, 5, 7, 9, 11}


def _is_white(midi: int) -> bool:
    return (int(midi) % 12) in _WHITE_PITCHES_MOD12


def _canonical_key_bounds() -> List[Tuple[float, float]]:
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


def _parse_corner_str(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in s.replace(",", " ").split()]
    return float(parts[0]), float(parts[1])


def _parse_corners_arg(arg: str) -> List[Tuple[float, float]]:
    parts = arg.split(";")
    if len(parts) != 4:
        raise ValueError("--metadata-corners must contain 4 'x,y' points separated by ';'")
    return [_parse_corner_str(p) for p in parts]


def _parse_tsv(path: Path) -> List[Tuple[float, float, int]]:
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


def _homography(corners_src: List[Tuple[float, float]]):
    import cv2
    import numpy as np
    src = np.array(corners_src, dtype=np.float32)
    dst = np.array(
        [[0, 0], [_CANON_W - 1, 0], [_CANON_W - 1, _CANON_H - 1], [0, _CANON_H - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def _build_tip_masks(*, white_band=(0.92, 0.98), black_band=(0.52, 0.58)):
    """Very narrow key-tip strip. Defaults: bottom 6% for white keys,
    bottom 6% of the black-key region for black keys.
    """

    import numpy as np
    bounds = _canonical_key_bounds()
    masks = np.zeros((_N_KEYS, _CANON_H, _CANON_W), dtype=np.uint8)
    for k in range(_N_KEYS):
        midi = _MIDI_LOW + k
        l, r = bounds[k]
        y_band = white_band if _is_white(midi) else black_band
        y0 = int(y_band[0] * _CANON_H)
        y1 = max(y0 + 1, int(y_band[1] * _CANON_H))
        masks[k, y0:y1, int(l):int(r)] = 1
    return masks


def _build_median_warped_frame(
    video_path: Path,
    H_warp,
    *,
    n_samples: int = 32,
    target_fps: float = 30.0,
    max_seconds: Optional[float] = None,
):
    """Sample frames evenly, warp each, take per-pixel median in grayscale."""

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / float(target_fps))))
    if max_seconds is not None and max_seconds > 0:
        total_frames = min(total_frames, int(round(max_seconds * src_fps)))
    sample_step = max(1, total_frames // max(n_samples, 1))
    samples = []
    for f_idx in range(0, total_frames, sample_step):
        if len(samples) >= n_samples:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        warp = cv2.warpPerspective(frame, H_warp, (_CANON_W, _CANON_H), flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        samples.append(gray)
    cap.release()
    if not samples:
        raise RuntimeError("no frames sampled for median")
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.float32)


def _extract_bgsub_motion(
    video_path: Path,
    H_warp,
    median_gray,
    masks,
    *,
    target_fps: float,
    max_seconds: Optional[float],
):
    """For each frame: signal[t, k] = mean(|frame[t] - median| * mask[k]).
    This measures deviation from the "rest" baseline inside each key tip.
    """

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / float(target_fps))))
    actual_fps = src_fps / float(step)
    if max_seconds is not None and max_seconds > 0:
        max_frames = int(round(max_seconds * actual_fps))
    else:
        max_frames = (total + step - 1) // step

    masks_flat = masks.reshape(_N_KEYS, -1).astype(np.float32)
    mask_areas = masks_flat.sum(axis=1)
    mask_areas[mask_areas == 0] = 1.0
    median_flat = median_gray.reshape(-1)

    motion = np.zeros((max_frames, _N_KEYS), dtype=np.float32)
    timestamps = np.zeros(max_frames, dtype=np.float32)
    decoded = 0
    f_idx = 0
    while f_idx < total and decoded < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        warp = cv2.warpPerspective(frame, H_warp, (_CANON_W, _CANON_H), flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY).astype(np.float32).reshape(-1)
        diff = np.abs(gray - median_flat)
        energy = masks_flat @ diff
        motion[decoded] = energy / mask_areas
        timestamps[decoded] = float(f_idx) / src_fps
        f_idx += step
        decoded += 1
    cap.release()
    return motion[:decoded], timestamps[:decoded], float(actual_fps)


def _smooth_1d(signal, k: int = 3):
    import numpy as np
    if signal.size < k or k <= 1:
        return signal
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(signal, kernel, mode="same").astype(np.float32)


def _detect_peaks_per_key(motion, timestamps, *, smoothing: int, prominence_factor: float,
                          min_event_gap_s: float):
    """For each key column, find peaks with sufficient prominence.

    Prominence is height above the local baseline (here: median over the
    whole series). A peak passes if (peak_value - median) > prominence_factor * MAD.
    """

    import numpy as np
    T, K = motion.shape
    if T < 3:
        return []
    fps = (T - 1) / max(float(timestamps[-1] - timestamps[0]), 1e-6) if T > 1 else 30.0
    min_gap_frames = max(1, int(round(min_event_gap_s * fps)))
    events: List[Tuple[float, int, float]] = []
    for k in range(K):
        s = _smooth_1d(motion[:, k], smoothing)
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med))) + 1e-6
        threshold = med + float(prominence_factor) * mad
        last_peak_t = -10**9
        last_peak_val = -1.0
        for t in range(1, T - 1):
            if s[t] < threshold:
                continue
            if not (s[t] > s[t - 1] and s[t] >= s[t + 1]):
                continue
            if t - last_peak_t < min_gap_frames:
                # Replace previous if this is stronger
                if s[t] > last_peak_val:
                    if events and events[-1][1] == k:
                        events.pop()
                    last_peak_t = t
                    last_peak_val = float(s[t])
                    events.append((float(timestamps[t]), int(k), float(s[t])))
                continue
            last_peak_t = t
            last_peak_val = float(s[t])
            events.append((float(timestamps[t]), int(k), float(s[t])))
    events.sort(key=lambda e: e[0])
    return events


def _spatial_nms(events, *, time_radius_s: float, key_radius: int):
    """If events at adjacent keys fire within time_radius_s, keep only the
    strongest. This eliminates the adjacent-key crosstalk where motion at
    key K bleeds into K-1 and K+1.
    """

    if not events:
        return events
    events_sorted = sorted(events, key=lambda e: e[0])
    kept: List[Tuple[float, int, float]] = []
    for t, k, v in events_sorted:
        # Find any kept event within (time_radius_s, key_radius) — replace if weaker.
        dominated = False
        new_kept = []
        for kt, kk, kv in kept:
            if abs(t - kt) <= time_radius_s and abs(k - kk) <= key_radius:
                if v <= kv:
                    dominated = True
                # else: this event dominates the existing one — drop the existing
            else:
                new_kept.append((kt, kk, kv))
        if not dominated:
            new_kept.append((t, k, v))
        kept = new_kept
    kept.sort(key=lambda e: e[0])
    return kept


def _note_f1(pred_events, ref_events, *, onset_tol_s: float):
    used = [False] * len(ref_events)
    tp = fp = 0
    ref_by_pitch: Dict[int, List[Tuple[int, float, float, int]]] = {}
    for i, (on, off, p) in enumerate(ref_events):
        ref_by_pitch.setdefault(int(p), []).append((i, float(on), float(off), int(p)))
    pitch_tp: Dict[int, int] = {}
    pitch_fp: Dict[int, int] = {}
    pitch_fn: Dict[int, int] = {}
    for pt, pp in sorted(pred_events, key=lambda e: e[0]):
        cands = ref_by_pitch.get(int(pp), [])
        best_idx = -1
        best_dt = float("inf")
        for i, r_on, r_off, _ in cands:
            if used[i]:
                continue
            dt = abs(pt - r_on)
            if dt <= onset_tol_s and dt < best_dt:
                best_dt = dt
                best_idx = i
        if best_idx >= 0:
            used[best_idx] = True
            tp += 1
            pitch_tp[pp] = pitch_tp.get(pp, 0) + 1
        else:
            fp += 1
            pitch_fp[pp] = pitch_fp.get(pp, 0) + 1
    fn = sum(1 for u in used if not u)
    for i, ev in enumerate(ref_events):
        if not used[i]:
            pitch_fn[int(ev[2])] = pitch_fn.get(int(ev[2]), 0) + 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    def _split(d):
        w = sum(v for k, v in d.items() if _is_white(k))
        b = sum(v for k, v in d.items() if not _is_white(k))
        return w, b

    w_tp, b_tp = _split(pitch_tp)
    w_fp, b_fp = _split(pitch_fp)
    w_fn, b_fn = _split(pitch_fn)
    w_f1 = 2 * w_tp / max(2 * w_tp + w_fp + w_fn, 1)
    b_f1 = 2 * b_tp / max(2 * b_tp + b_fp + b_fn, 1)

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "n_pred": int(len(pred_events)), "n_ref": int(len(ref_events)),
        "f1_white": float(w_f1), "f1_black": float(b_f1),
        "white_tp": int(w_tp), "white_fp": int(w_fp), "white_fn": int(w_fn),
        "black_tp": int(b_tp), "black_fp": int(b_fp), "black_fn": int(b_fn),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--metadata-corners", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--smoothing-window", type=int, default=3)
    ap.add_argument("--prominence-factor", type=float, default=4.0,
                    help="onset peak prominence threshold = median + factor*MAD per key")
    ap.add_argument("--min-event-gap-s", type=float, default=0.10)
    ap.add_argument("--nms-time-radius-s", type=float, default=0.05,
                    help="spatial NMS: events at adjacent keys within this time -> drop weaker")
    ap.add_argument("--nms-key-radius", type=int, default=2,
                    help="spatial NMS: number of keys on each side to consider for NMS")
    ap.add_argument("--onset-tolerance-ms", type=float, default=50.0)
    ap.add_argument("--n-median-samples", type=int, default=32)
    args = ap.parse_args()

    import cv2
    import numpy as np

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    corners = _parse_corners_arg(args.metadata_corners)
    print(f"corners (LT,RT,RB,LB) = {corners}")
    H_warp = _homography(corners)
    masks = _build_tip_masks()

    median_gray = _build_median_warped_frame(
        Path(args.video), H_warp,
        n_samples=int(args.n_median_samples),
        target_fps=float(args.target_fps),
        max_seconds=float(args.max_seconds) if args.max_seconds > 0 else None,
    )
    print(f"built median frame: shape={median_gray.shape} mean={median_gray.mean():.1f}")

    motion, timestamps, fps_used = _extract_bgsub_motion(
        Path(args.video), H_warp, median_gray, masks,
        target_fps=float(args.target_fps),
        max_seconds=float(args.max_seconds) if args.max_seconds > 0 else None,
    )
    print(f"motion: shape={motion.shape}  range=[{motion.min():.2f}, {motion.max():.2f}]  fps={fps_used:.2f}")

    raw_events = _detect_peaks_per_key(
        motion, timestamps,
        smoothing=int(args.smoothing_window),
        prominence_factor=float(args.prominence_factor),
        min_event_gap_s=float(args.min_event_gap_s),
    )
    print(f"raw peaks: {len(raw_events)}")

    nms_events = _spatial_nms(
        raw_events,
        time_radius_s=float(args.nms_time_radius_s),
        key_radius=int(args.nms_key_radius),
    )
    print(f"after spatial NMS: {len(nms_events)}")

    pred_events = [(t, int(k) + _MIDI_LOW) for (t, k, _v) in nms_events]
    ref_events = _parse_tsv(Path(args.tsv))
    print(f"detected {len(pred_events)} onsets; GT has {len(ref_events)} notes")

    onset_tol_s = float(args.onset_tolerance_ms) / 1000.0
    f1_result = _note_f1(pred_events, ref_events, onset_tol_s=onset_tol_s)
    print()
    print(f"=== v2 Note-level F1 (tolerance ±{args.onset_tolerance_ms:.0f} ms) ===")
    print(json.dumps(f1_result, indent=2))

    (out_dir / "predictions.csv").write_text(
        "\n".join([f"{t:.4f},{p}" for (t, p) in pred_events]), encoding="utf-8"
    )
    (out_dir / "ground_truth.csv").write_text(
        "\n".join([f"{on:.4f},{off:.4f},{p}" for (on, off, p) in ref_events]), encoding="utf-8"
    )
    (out_dir / "result.json").write_text(json.dumps({
        "video": str(args.video),
        "tsv": str(args.tsv),
        "method": "v2_bgsub_tip_strip_nms",
        "params": vars(args),
        "metrics": f1_result,
    }, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.imshow(motion.T, aspect="auto", origin="lower", cmap="viridis",
                  extent=[timestamps[0], timestamps[-1] if timestamps.size else 1.0, 0, _N_KEYS])
        pt = [e[0] for e in pred_events]
        pk = [e[1] - _MIDI_LOW for e in pred_events]
        ax.scatter(pt, pk, s=22, marker="o", facecolors="none", edgecolors="white",
                   label=f"detected ({len(pred_events)})")
        rt = [e[0] for e in ref_events]
        rk = [int(e[2]) - _MIDI_LOW for e in ref_events]
        ax.scatter(rt, rk, s=22, marker="x", c="red", label=f"GT ({len(ref_events)})")
        ax.set_xlabel("time (s)"); ax.set_ylabel("key index (0=A0, 87=C8)")
        ax.set_title(
            f"v2 (bgsub + tip-strip + NMS)  "
            f"F1={f1_result['f1']:.3f}  P={f1_result['precision']:.3f}  R={f1_result['recall']:.3f}  "
            f"TP={f1_result['tp']} FP={f1_result['fp']} FN={f1_result['fn']}"
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "motion_heatmap.png", dpi=120)
        plt.close(fig)
        # Save the median frame too — sanity check that warping worked
        cv2.imwrite(str(out_dir / "median_warped.png"), median_gray.astype("uint8"))
    except Exception as exc:
        print(f"plotting skipped: {exc}")

    print(f"\nWrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
