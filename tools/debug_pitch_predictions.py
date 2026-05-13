#!/usr/bin/env python3
"""End-to-end pitch-prediction debug dumper.

Loads ONE video from the dataset, loads a trained model checkpoint, runs the
model on the clip, and saves an exhaustive diagnostic dump comparing
predictions to ground truth — frame-by-frame, key-by-key.

The output answers the questions we cannot answer from aggregate F1:
    Q1: For a known-active GT key K at frame T, what does the model predict?
        - Is the prediction at the right TIME but wrong PITCH? (geometry/calibration)
        - Or at the right PITCH but wrong TIME?    (sync drift)
        - Or random?                                (model didn't learn)
    Q2: Across all GT-active frames in the clip, is there a SYSTEMATIC
        offset between predicted and true pitch (e.g. always K-1 or K+1)?
        - Histogram of (argmax_pred - true_pitch)
        - Centered at 0 with small spread -> model is correct, threshold issue
        - Shifted by k -> calibration offset of k keys
        - Random/flat -> model didn't learn pitch
    Q3: For a known-active frame, what is the model's LOGIT distribution
        across all 88 keys? (Peak position, top-k, are GT keys in top-N)

Outputs (under <out-dir>/<video_id>/):
    - clip_info.json
        Video metadata + clip-level summary (median/p95 logits, n events)
    - per_frame_predictions.csv
        One row per frame: t, gt_active_keys, pred_top1, pred_top3, max_logit
    - pitch_offset_histogram.png
        Histogram of (argmax_pred_pitch - true_pitch) at GT-active frames.
        Diagnostic: shifted vs centered vs flat reveals root cause.
    - logit_heatmap.png
        88-key x T-frame heatmap of model's pitch logits with GT-active cells
        marked in red. Visual inspection: are the bright cells at GT positions?
    - gt_vs_pred_overlay.png
        Two-row image: GT active mask vs predicted active mask
    - first_10_onsets.txt
        For the first 10 ground-truth onsets, dump:
            - time, true_pitch
            - model's argmax pitch at that frame
            - model's top-5 pitches at that frame
            - model's logit for the true pitch

Usage:
    PYTHONPATH=. python tools/debug_pitch_predictions.py \\
        --config configs/default.yaml \\
        --config configs/overrides/manual_training.yaml \\
        --config configs/overrides/manual_training_keyroi_train.yaml \\
        --config configs/overrides/06-05-2026/manual_training_keyroi_video_06_05_2026.yaml \\
        --config configs/overrides/06-05-2026/phase5_fix_pitch_head.yaml \\
        --config configs/overrides/06-05-2026/phase5_eval_only_decoder_fix.yaml \\
        --video-id 2024-02-15_20-07-54 \\
        --split valid \\
        --out-dir /home/achatzigiannis/tivit-logs/phase7_debug_13_05_2026/pitch_debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Bypass geometry gate so any video can be inspected (even gated-out failed ones).
os.environ.setdefault("TIVIT_GEOMETRY_GATE_DISABLED", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--video-id", required=True, help="e.g. 2024-02-15_20-07-54")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out-dir", required=True, help="output root; per-video subdir is created")
    ap.add_argument("--checkpoint", default=None, help="optional explicit checkpoint path")
    ap.add_argument("--n-frames", type=int, default=96, help="frames per clip to analyze")
    args = ap.parse_args()

    import csv
    import numpy as np
    import torch
    import torch.nn.functional as F

    from tivit.core.config import load_experiment_config
    from tivit.data.datasets.pianovam_impl import PianoVAMDataset
    from tivit.models import build_model
    from tivit.pipelines._common import find_checkpoint, load_model_weights, setup_runtime

    cfg = dict(load_experiment_config(args.config or [Path("configs/default.yaml")]))
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    dataset_cfg["preprocessed_format"] = "video"
    dataset_cfg["shuffle"] = False
    cfg["dataset"] = dataset_cfg

    seed_val, det_flag, device = setup_runtime(cfg, seed=42, deterministic=False)

    note_min = int((dataset_cfg.get("frame_targets", {}) or {}).get("note_min", 21))
    note_max = int((dataset_cfg.get("frame_targets", {}) or {}).get("note_max", 108))
    n_keys = note_max - note_min + 1

    out_root = Path(args.out_dir).expanduser().resolve() / args.video_id
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Locate video in dataset ----
    ds = PianoVAMDataset(cfg, split=args.split, full_cfg=cfg)
    target_idx = None
    for i, e in enumerate(ds.entries):
        if str(e.video_id) == args.video_id:
            target_idx = i
            break
    if target_idx is None:
        raise SystemExit(f"video {args.video_id} not found in split={args.split}")

    print(f"[debug] split={args.split} video_id={args.video_id} idx={target_idx}")

    # ---- Load checkpoint + model ----
    model = build_model(cfg).to(device)
    model.eval()
    ckpt_path = find_checkpoint(cfg, args.checkpoint)
    if ckpt_path is None:
        raise SystemExit("no checkpoint found")
    epoch_loaded = load_model_weights(model, ckpt_path, device)
    print(f"[debug] loaded {ckpt_path} (epoch {epoch_loaded})")

    # ---- Run one sample through the dataset (gets video + targets) ----
    sample = ds[target_idx]
    video = sample["video"]
    pitch_gt = sample.get("pitch")
    onset_gt = sample.get("onset")
    events_raw = sample.get("metadata", {}).get("_events_after_sync") if isinstance(sample.get("metadata"), Mapping) else None

    if not isinstance(video, torch.Tensor):
        raise SystemExit(f"sample['video'] is not a tensor (got {type(video)})")
    if pitch_gt is None or onset_gt is None:
        raise SystemExit("sample missing pitch/onset targets — dataset misconfigured")

    # Add batch dim
    video_in = video.unsqueeze(0).to(device)

    # ---- Inference ----
    with torch.no_grad():
        out = model(video_in)
        # Output is typically a dict of head -> logits. Find pitch and onset.
        if isinstance(out, Mapping):
            # Real key names in this codebase are *_logits, not just the head name.
            pitch_logits = out.get("pitch_logits", out.get("pitch"))
            onset_logits = out.get("onset_logits", out.get("onset"))
            offset_logits = out.get("offset_logits", out.get("offset"))
        else:
            raise SystemExit(f"model output is not a mapping (got {type(out)}); cannot extract heads")

    if pitch_logits is None:
        keys_str = ", ".join(sorted(out.keys()) if isinstance(out, Mapping) else [])
        raise SystemExit(f"model output has no 'pitch_logits' head. Available keys: [{keys_str}]")

    # Squeeze batch
    pitch_logits = pitch_logits.squeeze(0).detach().cpu().numpy()  # (T, K) or (T, tiles, K)
    if pitch_logits.ndim == 3:
        # Per-tile output (T, tiles, K) — collapse tiles by mean across tiles
        pitch_logits = pitch_logits.mean(axis=1)
    onset_logits_np = onset_logits.squeeze(0).detach().cpu().numpy() if onset_logits is not None else None
    if onset_logits_np is not None and onset_logits_np.ndim == 3:
        onset_logits_np = onset_logits_np.mean(axis=1)

    pitch_gt_np = pitch_gt.detach().cpu().numpy() if torch.is_tensor(pitch_gt) else np.asarray(pitch_gt)
    onset_gt_np = onset_gt.detach().cpu().numpy() if torch.is_tensor(onset_gt) else np.asarray(onset_gt)
    if pitch_gt_np.ndim == 3:
        pitch_gt_np = pitch_gt_np.max(axis=1) if pitch_gt_np.shape[1] != n_keys else pitch_gt_np.mean(axis=0)
    if pitch_gt_np.ndim != 2:
        raise SystemExit(f"pitch_gt unexpected shape {pitch_gt_np.shape}; expected (T, K)")

    T, K = pitch_logits.shape
    print(f"[debug] logits shape T={T} K={K}  GT shape={pitch_gt_np.shape}")

    # ---- Clip-level summary ----
    pitch_probs = 1.0 / (1.0 + np.exp(-pitch_logits))
    onset_probs = 1.0 / (1.0 + np.exp(-onset_logits_np)) if onset_logits_np is not None else None
    gt_active_per_frame = (pitch_gt_np > 0.5).sum(axis=1).astype(float)
    pred_top1 = pitch_logits.argmax(axis=1)
    pred_top1_in_gt = np.array(
        [int(bool(pitch_gt_np[t, pred_top1[t]] > 0.5)) for t in range(T)]
    )

    clip_info = {
        "video_id": args.video_id,
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "epoch_loaded": int(epoch_loaded) if epoch_loaded is not None else None,
        "n_frames": int(T),
        "n_keys": int(K),
        "pitch_logit_min": float(pitch_logits.min()),
        "pitch_logit_median": float(np.median(pitch_logits)),
        "pitch_logit_max": float(pitch_logits.max()),
        "pitch_prob_active_per_frame_mean": float(((pitch_probs > 0.5).sum(axis=1)).mean()),
        "gt_active_per_frame_mean": float(gt_active_per_frame.mean()),
        "gt_active_frames": int((gt_active_per_frame > 0).sum()),
        "top1_in_gt_rate": float(pred_top1_in_gt.mean()),  # what % of frames have argmax pitch in GT
    }

    # ---- Per-frame predictions CSV ----
    with (out_root / "per_frame_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "frame_idx",
            "n_gt_active",
            "gt_active_keys",
            "pred_top1",
            "pred_top3",
            "pred_top1_logit",
            "max_logit",
            "top1_in_gt",
        ])
        for t in range(T):
            gt_keys = np.where(pitch_gt_np[t] > 0.5)[0].tolist()
            top3 = np.argsort(pitch_logits[t])[::-1][:3].tolist()
            w.writerow([
                t,
                int(gt_active_per_frame[t]),
                ",".join(map(str, gt_keys)),
                int(pred_top1[t]),
                ",".join(map(str, top3)),
                float(pitch_logits[t, pred_top1[t]]),
                float(pitch_logits[t].max()),
                int(pred_top1_in_gt[t]),
            ])

    # ---- Pitch offset histogram ----
    # For every (t, k_gt) where k_gt is active, record (argmax_pitch[t] - k_gt).
    # Centered at 0 with narrow spread -> model is correct
    # Shifted by k -> consistent calibration offset
    # Flat / random -> model has no pitch info
    offsets: list[int] = []
    for t in range(T):
        for k_gt in np.where(pitch_gt_np[t] > 0.5)[0]:
            offsets.append(int(pred_top1[t]) - int(k_gt))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(offsets, bins=np.arange(-87, 88), edgecolor="black")
        ax.set_title(
            f"Pitch offset: (argmax_pred - true_pitch)  N={len(offsets)}\n"
            f"video={args.video_id}  split={args.split}  ckpt_epoch={epoch_loaded}\n"
            "centered=0 small spread => model correct | shifted=k => calibration off by k keys | flat => didn't learn"
        )
        ax.set_xlabel("pred_pitch - true_pitch (MIDI key index difference)")
        ax.set_ylabel("count")
        ax.axvline(0, color="green", linestyle="--", linewidth=1)
        fig.tight_layout()
        fig.savefig(out_root / "pitch_offset_histogram.png", dpi=100)
        plt.close(fig)

        # Logit heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(pitch_logits.T, aspect="auto", cmap="viridis", origin="lower")
        # Mark GT-active cells in red
        gt_rows, gt_cols = np.where(pitch_gt_np.T > 0.5)
        ax.scatter(gt_cols, gt_rows, c="red", s=8, marker="x", label="GT active")
        ax.set_title(f"Pitch logits heatmap (88 keys x {T} frames). Red x = GT active.")
        ax.set_xlabel("frame")
        ax.set_ylabel("key index (0=A0)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_root / "logit_heatmap.png", dpi=100)
        plt.close(fig)

        # GT vs pred active overlay
        pred_active = (pitch_probs > 0.5).astype(int)
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        axes[0].imshow(pitch_gt_np.T, aspect="auto", cmap="Greens", origin="lower")
        axes[0].set_title("Ground truth (pitch[T, 88])")
        axes[0].set_ylabel("key index")
        axes[1].imshow(pred_active.T, aspect="auto", cmap="Reds", origin="lower")
        axes[1].set_title(f"Predicted (pitch_prob > 0.5)  pred_per_frame={pred_active.sum(axis=1).mean():.2f}  gt={gt_active_per_frame.mean():.2f}")
        axes[1].set_ylabel("key index")
        axes[1].set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(out_root / "gt_vs_pred_overlay.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        print(f"[debug] plotting skipped ({exc})")

    # ---- Top-N detail on the first GT-active frames ----
    with (out_root / "first_10_onsets.txt").open("w", encoding="utf-8") as fh:
        gt_active_frames_idx = np.where(gt_active_per_frame > 0)[0]
        for t in gt_active_frames_idx[:10]:
            gt_keys = np.where(pitch_gt_np[t] > 0.5)[0].tolist()
            top5 = np.argsort(pitch_logits[t])[::-1][:5].tolist()
            fh.write(f"frame={t}\n")
            fh.write(f"  GT keys (0-indexed): {gt_keys}    (MIDI: {[k + note_min for k in gt_keys]})\n")
            fh.write(f"  Top-5 pred keys    : {top5}    (MIDI: {[k + note_min for k in top5]})\n")
            fh.write(f"  Logit at GT keys   : {[float(pitch_logits[t, k]) for k in gt_keys]}\n")
            fh.write(f"  Logit at top-5     : {[float(pitch_logits[t, k]) for k in top5]}\n")
            fh.write(f"  Sigmoid at GT keys : {[float(pitch_probs[t, k]) for k in gt_keys]}\n")
            fh.write("\n")

    # Histogram summary in clip_info
    if offsets:
        offsets_arr = np.asarray(offsets)
        clip_info["offset_count"] = int(len(offsets))
        clip_info["offset_median"] = int(np.median(offsets_arr))
        clip_info["offset_mean"] = float(np.mean(offsets_arr))
        clip_info["offset_std"] = float(np.std(offsets_arr))
        clip_info["offset_at_zero_pct"] = float((offsets_arr == 0).mean() * 100.0)
        clip_info["offset_within_1_pct"] = float((np.abs(offsets_arr) <= 1).mean() * 100.0)
        clip_info["offset_within_3_pct"] = float((np.abs(offsets_arr) <= 3).mean() * 100.0)
        # Diagnosis
        med = clip_info["offset_median"]
        within_1 = clip_info["offset_within_1_pct"]
        if abs(med) >= 2 and within_1 < 25.0:
            clip_info["diagnosis"] = f"CALIBRATION_OFFSET: model predictions are systematically off by ~{med} keys"
        elif within_1 < 15.0:
            clip_info["diagnosis"] = "MODEL_NOT_LEARNED: predictions essentially random with respect to GT pitch"
        elif within_1 >= 40.0:
            clip_info["diagnosis"] = "MODEL_LEARNING: most predictions land within 1 key of GT — threshold/decoder tuning will help"
        else:
            clip_info["diagnosis"] = "MIXED: model partially learning, partial random"

    (out_root / "clip_info.json").write_text(json.dumps(clip_info, indent=2), encoding="utf-8")

    print()
    print(f"=== {args.video_id} debug summary ===")
    print(json.dumps(clip_info, indent=2))
    print()
    print(f"Wrote artifacts to {out_root}")


if __name__ == "__main__":
    main()
