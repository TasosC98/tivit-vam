#!/usr/bin/env python3
"""One-shot per-video keyboard calibration.

Walks the PianoVAM dataset (using the same config stack as training/audit) and
produces `key_geometry/{split}/{video_id}.json` for every entry. Writes a
summary CSV and an `excluded_videos.csv` listing every video that does NOT meet
the acceptance bar so the gated dataset loader can skip them in train.

Usage:
    PYTHONPATH=. python tools/recalibrate_all.py \\
        --config configs/default.yaml \\
        --config configs/overrides/manual_training.yaml \\
        --config configs/overrides/manual_training_keyroi_train.yaml \\
        --config configs/overrides/26-04-2026/manual_training_keyroi_video.yaml \\
        --splits train valid test \\
        --out-root /home/achatzigiannis/tivit-logs/manual_training_keyroi_video_26-04-2026/key_geometry \\
        --max-videos 0

Notes:
    - Reads videos directly from disk (no HDF5 path), uses the metadata crop.
    - Writes per-video JSON in the schema documented in
      data/roi/keyboard_calibration.py.
    - Status values: ok, accepted_loose, failed.
    - Failed videos are listed in excluded_videos.csv but the JSON is still
      written (with calibration_status='failed') so a manual UI can edit it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", default=None, help="Config stack")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--out-root", required=True, help="Directory for key_geometry/{split}/*.json")
    ap.add_argument("--max-videos", type=int, default=0, help="Per split, 0 means all")
    ap.add_argument("--num-samples", type=int, default=32, help="Frames sampled to build the median")
    ap.add_argument("--skip-seconds", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 means no limit")
    ap.add_argument("--inlier-threshold-px", type=float, default=3.0)
    ap.add_argument("--ransac-iterations", type=int, default=200)
    ap.add_argument("--th-ok-median-px", type=float, default=2.0)
    ap.add_argument("--th-ok-p95-px", type=float, default=4.0)
    ap.add_argument("--th-loose-median-px", type=float, default=4.0)
    ap.add_argument("--overwrite", action="store_true", help="Re-process videos that already have JSON")
    return ap.parse_args()


def _crop_tuple(crop: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(crop, (list, tuple)) or len(crop) < 4:
        return None
    try:
        return (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = _parse_args()
    from tivit.core.config import load_experiment_config
    from tivit.data.datasets.pianovam_impl import PianoVAMDataset
    from tivit.data.roi.keyboard_calibration import (
        CalibrationConfig,
        MedianFrameOptions,
        calibrate_video,
        write_geometry_json,
    )

    cfg = dict(load_experiment_config(args.config or [Path("configs/default.yaml")]))
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    dataset_cfg["preprocessed_format"] = "video"  # bypass HDF5 entirely
    dataset_cfg["shuffle"] = False
    cfg["dataset"] = dataset_cfg

    canonical_hw = dataset_cfg.get("canonical_hw") or [180, 1536]
    canonical_hw = (int(canonical_hw[0]), int(canonical_hw[1]))

    median_opts = MedianFrameOptions(
        num_samples=int(args.num_samples),
        skip_seconds=float(args.skip_seconds),
        max_seconds=float(args.max_seconds) if args.max_seconds > 0 else None,
    )
    calib_cfg = CalibrationConfig(
        canonical_hw=canonical_hw,
        median_options=median_opts,
        inlier_threshold_px=float(args.inlier_threshold_px),
        ransac_iterations=int(args.ransac_iterations),
        th_ok_median_px=float(args.th_ok_median_px),
        th_ok_p95_px=float(args.th_ok_p95_px),
        th_loose_median_px=float(args.th_loose_median_px),
    )

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    counts = {"ok": 0, "accepted_loose": 0, "failed": 0, "errored": 0}

    started = time.time()
    for split in args.splits:
        ds = PianoVAMDataset(cfg, split=split, full_cfg=cfg)
        limit = len(ds) if args.max_videos <= 0 else min(len(ds), args.max_videos)
        for idx in range(limit):
            entry = ds.entries[idx]
            video_id = entry.video_id
            json_path = out_root / split / f"{video_id}.json"
            if json_path.exists() and not args.overwrite:
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                    status = str(payload.get("calibration_status") or "")
                    summary_rows.append(
                        {
                            "split": split,
                            "video_id": video_id,
                            "calibration_status": status,
                            "residual_median_px": payload.get("residual_median_px"),
                            "residual_p95_px": payload.get("residual_p95_px"),
                            "ransac_inlier_ratio": payload.get("ransac_inlier_ratio"),
                            "black_key_anchor_count": payload.get("black_key_anchor_count"),
                            "json_path": str(json_path),
                            "video_path": str(entry.video_path),
                            "skipped_existing": True,
                        }
                    )
                    counts[status] = counts.get(status, 0) + 1
                    if status in {"failed"}:
                        excluded_rows.append(
                            {
                                "split": split,
                                "video_id": video_id,
                                "calibration_status": status,
                                "reason": "existing_json_failed",
                                "json_path": str(json_path),
                                "video_path": str(entry.video_path),
                            }
                        )
                    print(f"[skip ] split={split} {idx + 1}/{limit} video={video_id} status={status}", flush=True)
                    continue
                except Exception:
                    pass  # fall through and recompute

            crop = _crop_tuple(entry.metadata.get("crop") if isinstance(entry.metadata, Mapping) else None)
            t0 = time.time()
            try:
                result = calibrate_video(
                    Path(entry.video_path),
                    video_id=video_id,
                    crop=crop,
                    cfg=calib_cfg,
                )
                json_out = write_geometry_json(result, out_root, split)
                status = result.calibration_status
                counts[status] = counts.get(status, 0) + 1
                summary_rows.append(
                    {
                        "split": split,
                        "video_id": video_id,
                        "calibration_status": status,
                        "residual_median_px": result.residual_median_px,
                        "residual_p95_px": result.residual_p95_px,
                        "ransac_inlier_ratio": result.ransac_inlier_ratio,
                        "black_key_anchor_count": result.black_key_anchor_count,
                        "json_path": str(json_out),
                        "video_path": str(entry.video_path),
                        "skipped_existing": False,
                        "elapsed_s": round(time.time() - t0, 2),
                    }
                )
                if status == "failed":
                    excluded_rows.append(
                        {
                            "split": split,
                            "video_id": video_id,
                            "calibration_status": status,
                            "reason": (
                                f"residual_median_px={result.residual_median_px} "
                                f"p95={result.residual_p95_px} "
                                f"anchors={result.black_key_anchor_count} "
                                f"notes={result.notes}"
                            ),
                            "json_path": str(json_out),
                            "video_path": str(entry.video_path),
                        }
                    )
                print(
                    f"[calib] split={split} {idx + 1}/{limit} video={video_id} "
                    f"status={status} median={result.residual_median_px} p95={result.residual_p95_px} "
                    f"anchors={result.black_key_anchor_count} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                counts["errored"] = counts.get("errored", 0) + 1
                excluded_rows.append(
                    {
                        "split": split,
                        "video_id": video_id,
                        "calibration_status": "errored",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "json_path": "",
                        "video_path": str(entry.video_path),
                    }
                )
                summary_rows.append(
                    {
                        "split": split,
                        "video_id": video_id,
                        "calibration_status": "errored",
                        "residual_median_px": None,
                        "residual_p95_px": None,
                        "ransac_inlier_ratio": None,
                        "black_key_anchor_count": None,
                        "json_path": "",
                        "video_path": str(entry.video_path),
                        "skipped_existing": False,
                    }
                )
                print(f"[ERR ] split={split} {idx + 1}/{limit} video={video_id} {exc}", flush=True)
                traceback.print_exc()

    summary_path = out_root / "calibration_summary.csv"
    excluded_path = out_root / "excluded_videos.csv"

    if summary_rows:
        fields: List[str] = []
        for row in summary_rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)

    if excluded_rows:
        ef: List[str] = []
        for row in excluded_rows:
            for k in row:
                if k not in ef:
                    ef.append(k)
        with excluded_path.open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=ef)
            w.writeheader()
            w.writerows(excluded_rows)

    elapsed = time.time() - started
    print()
    print(f"Wrote {summary_path}")
    print(f"Wrote {excluded_path}" if excluded_rows else "(no excluded videos)")
    print(
        f"counts: ok={counts.get('ok', 0)} accepted_loose={counts.get('accepted_loose', 0)} "
        f"failed={counts.get('failed', 0)} errored={counts.get('errored', 0)} "
        f"total={sum(counts.values())} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
