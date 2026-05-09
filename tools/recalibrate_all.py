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
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# CRITICAL: This tool populates the geometry root, so it must build dataset
# entries BEFORE any geometry JSONs exist. Disable the gate before importing
# anything from the dataset package.
os.environ["TIVIT_GEOMETRY_GATE_DISABLED"] = "1"

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
    # Threshold defaults bumped 2026-05-08 to match the v4 calibration's
    # 1-to-1 residual metric. The old 2.0 / 4.0 defaults were calibrated for
    # the all-detections-snap-to-nearest metric, which over-counted spurious
    # blobs and inflated p95. With one-to-one assignment, ~50 of 71 videos
    # land in the <3 px median range; these thresholds let those pass as
    # "ok" rather than "loose".
    ap.add_argument("--th-ok-median-px", type=float, default=3.5)
    ap.add_argument("--th-ok-p95-px", type=float, default=8.0)
    ap.add_argument("--th-loose-median-px", type=float, default=6.0)
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
        CALIBRATION_VERSION,
        CALIBRATION_FEATURES,
        CalibrationConfig,
        MedianFrameOptions,
        calibrate_video,
        write_geometry_json,
    )

    # Version banner. If you are running this tool and don't see this banner,
    # you are running an older version of the code that doesn't have it.
    print("=" * 72)
    print(f"TIVIT calibration {CALIBRATION_VERSION}")
    print("Active features:")
    for feat in CALIBRATION_FEATURES:
        print(f"  - {feat}")
    print("=" * 72)

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
                notes_for_row = str(getattr(result, "notes", "") or "")
                if "method=white_edge_correlation" in notes_for_row:
                    row_method = "white_edge"
                elif "method=black_key_ransac" in notes_for_row:
                    row_method = "black_key"
                elif "method=nominal_crop" in notes_for_row:
                    row_method = "nominal_crop"
                else:
                    row_method = "unknown"
                summary_rows.append(
                    {
                        "split": split,
                        "video_id": video_id,
                        "calibration_status": status,
                        "residual_median_px": result.residual_median_px,
                        "residual_p95_px": result.residual_p95_px,
                        "ransac_inlier_ratio": result.ransac_inlier_ratio,
                        "black_key_anchor_count": result.black_key_anchor_count,
                        "method": row_method,
                        "notes": notes_for_row,
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
                med = result.residual_median_px
                p95 = result.residual_p95_px
                # Normalized residual = residual / white-key-width. <0.10 ≈ pitch
                # snaps to the right key for note recognition; >0.20 = adjacent
                # confusion; >0.30 = unusable.
                ww = float(result.white_key_width_px) if result.white_key_width_px else 1.0
                norm_med = (float(med) / ww) if (med is not None and ww > 0) else None
                inlier_pct = (
                    f"{int(round(100.0 * float(result.ransac_inlier_ratio)))}%"
                    if result.ransac_inlier_ratio is not None
                    else "n/a"
                )
                # Sparkline of pass/fail relative to white-key width:
                #   .  = within  8% of key width
                #   :  = within 17%
                #   x  = within 33%
                #   X  = beyond 33% (adjacent-key confusion)
                bar = ""
                if med is not None and ww > 0:
                    r = float(med) / ww
                    bar = "." if r < 0.085 else (":" if r < 0.17 else ("x" if r < 0.33 else "X"))
                tag = {"ok": "OK   ", "accepted_loose": "LOOSE", "failed": "FAIL "}.get(status, status)
                med_s = f"{med:5.2f}" if med is not None else "  n/a"
                p95_s = f"{p95:5.2f}" if p95 is not None else "  n/a"
                norm_s = f"{norm_med:.3f}" if norm_med is not None else "  n/a"
                # Extract method and we_corr from notes for at-a-glance diagnosis.
                notes = str(getattr(result, "notes", "") or "")
                method_short = "?"
                if "method=white_edge_correlation" in notes:
                    method_short = "WE"
                elif "method=black_key_ransac" in notes:
                    method_short = "BK"
                elif "method=nominal_crop" in notes:
                    method_short = "NC"
                print(
                    f"[calib] {tag} {split:5s} {idx + 1:3d}/{limit:3d} {video_id:30s} "
                    f"med={med_s} p95={p95_s} norm={norm_s} {bar} "
                    f"anchors={result.black_key_anchor_count or 0:3d} "
                    f"inliers={inlier_pct:>4s} method={method_short} {time.time() - t0:4.1f}s",
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

    # Distribution summary so we can see whether residuals improved between
    # runs without re-running the comparator. White-key width is 1536/52 ≈
    # 29.5 px; "norm" = median residual / white-key width.
    def _stats(values: List[float]) -> str:
        if not values:
            return "n=0"
        s = sorted(values)
        def _q(p: float) -> float:
            if not s:
                return float("nan")
            if len(s) == 1:
                return s[0]
            k = (len(s) - 1) * p
            lo, hi = int(k), min(int(k) + 1, len(s) - 1)
            return s[lo] * (hi - k) + s[hi] * (k - lo)
        return (
            f"n={len(s)} "
            f"min={s[0]:5.2f} median={_q(0.5):5.2f} p90={_q(0.90):5.2f} "
            f"p95={_q(0.95):5.2f} max={s[-1]:5.2f}"
        )

    by_split: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary_rows:
        by_split.setdefault(str(row.get("split") or ""), []).append(row)

    print()
    print("=== Residual distribution (white-key width = 29.54 px in canonical) ===")
    print("    Each row: status counts, median residual stats (px), normalized stats")
    for split_name in ["train", "valid", "test"]:
        rows = by_split.get(split_name, [])
        if not rows:
            continue
        c = {"ok": 0, "accepted_loose": 0, "failed": 0, "errored": 0}
        meds: List[float] = []
        norms: List[float] = []
        anchors: List[float] = []
        adj_conf = 0  # videos with median > 0.17 of key width (adjacent-key confusion likely)
        for r in rows:
            st = str(r.get("calibration_status") or "")
            c[st] = c.get(st, 0) + 1
            m = r.get("residual_median_px")
            if m is not None:
                meds.append(float(m))
                norm = float(m) / 29.5384615
                norms.append(norm)
                if norm > 0.17:
                    adj_conf += 1
            a = r.get("black_key_anchor_count")
            if a is not None:
                anchors.append(float(a))
        total = len(rows)
        usable_for_train = c.get("ok", 0) + c.get("manual_fixed", 0)
        usable_for_eval = usable_for_train + c.get("accepted_loose", 0)
        print(f"  [{split_name}] total={total}  "
              f"ok={c.get('ok', 0)}  loose={c.get('accepted_loose', 0)}  "
              f"failed={c.get('failed', 0)}  errored={c.get('errored', 0)}")
        print(f"           train-usable={usable_for_train}/{total} ({100*usable_for_train/max(total,1):.0f}%)  "
              f"eval-usable={usable_for_eval}/{total} ({100*usable_for_eval/max(total,1):.0f}%)")
        print(f"           median_residual_px : {_stats(meds)}")
        print(f"           normalized_residual: {_stats(norms)}")
        print(f"           anchors_per_video  : {_stats(anchors)}")
        print(f"           adjacent-confusion-risk videos (norm>0.17): {adj_conf}/{total}")

    # Method × status crosstab
    method_status: Dict[str, Dict[str, int]] = {}
    for row in summary_rows:
        m = str(row.get("method") or "unknown")
        st = str(row.get("calibration_status") or "")
        method_status.setdefault(m, {"ok": 0, "accepted_loose": 0, "failed": 0, "errored": 0})
        method_status[m][st] = method_status[m].get(st, 0) + 1
    print()
    print("=== Calibration method × status ===")
    print(f"  {'method':22s}  {'ok':>4s}  {'loose':>5s}  {'fail':>5s}  {'err':>4s}")
    for m, c in sorted(method_status.items()):
        print(
            f"  {m:22s}  {c.get('ok', 0):4d}  {c.get('accepted_loose', 0):5d}  "
            f"{c.get('failed', 0):5d}  {c.get('errored', 0):4d}"
        )

    # Overall sparkline across all videos
    all_meds = [float(r["residual_median_px"]) for r in summary_rows if r.get("residual_median_px") is not None]
    if all_meds:
        bins = {"<2px (excellent)": 0, "2-3px (ok)": 0, "3-5px (loose)": 0,
                "5-8px (poor)": 0, ">=8px (broken)": 0}
        for m in all_meds:
            if m < 2: bins["<2px (excellent)"] += 1
            elif m < 3: bins["2-3px (ok)"] += 1
            elif m < 5: bins["3-5px (loose)"] += 1
            elif m < 8: bins["5-8px (poor)"] += 1
            else: bins[">=8px (broken)"] += 1
        print()
        print("=== Median-residual histogram (all splits) ===")
        for label, n in bins.items():
            bar = "#" * int(round(40.0 * n / max(len(all_meds), 1)))
            print(f"  {label:22s} {n:3d} {bar}")


if __name__ == "__main__":
    main()
