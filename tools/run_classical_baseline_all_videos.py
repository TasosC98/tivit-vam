#!/usr/bin/env python3
"""Run the classical per-key detector on every video that has both a TSV and
PianoVAM metadata corners. Aggregates F1 per video into one CSV.

Usage:
    PYTHONPATH=. python tools/run_classical_baseline_all_videos.py \\
        --video-root /raid_storage/data_achatzigiannis/PianoVAM_v1.0/Video \\
        --tsv-root   /raid_storage/data_achatzigiannis/PianoVAM_v1.0/TSV \\
        --metadata   /raid_storage/data_achatzigiannis/PianoVAM_v1.0/metadata_v2.json \\
        --out-dir    /home/achatzigiannis/tivit-logs/classical_baseline_14_05_2026 \\
        --splits     train valid test \\
        --max-videos 0   # 0 = all

Outputs:
    <out-dir>/all_videos_f1.csv     summary across all evaluated videos
    <out-dir>/per_video/<id>/*      per-video artifacts (predictions, heatmap)
    <out-dir>/SUMMARY.txt           aggregate stats: mean/median F1, white vs black
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--tsv-root", required=True)
    ap.add_argument("--metadata", required=True, help="path to metadata_v2.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--smoothing-window", type=int, default=3)
    ap.add_argument("--peak-threshold-factor", type=float, default=3.0)
    ap.add_argument("--min-event-gap-s", type=float, default=0.10)
    ap.add_argument("--onset-tolerance-ms", type=float, default=50.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metadata, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if not isinstance(meta, dict):
        raise SystemExit("metadata_v2.json must be a dict")

    # Build index: record_time -> metadata entry
    index = {}
    for _, entry in meta.items():
        if not isinstance(entry, Mapping):
            continue
        rt = entry.get("record_time")
        if rt:
            index[str(rt).strip()] = entry

    rows: List[Dict[str, Any]] = []
    splits_seen: Dict[str, int] = {s: 0 for s in args.splits}

    for rec_time, entry in index.items():
        split = str(entry.get("split", "")).strip().lower()
        # Normalize ext-train -> train, etc.
        canon_split = (
            "train" if split in {"train", "ext-train"}
            else "valid" if split in {"valid", "val", "ext-valid"}
            else "test" if split in {"test", "ext-test"}
            else split
        )
        if canon_split not in args.splits:
            continue
        if args.max_videos > 0 and splits_seen.get(canon_split, 0) >= args.max_videos:
            continue
        # Look for video + TSV files
        video_path = Path(args.video_root) / f"{rec_time}.mp4"
        tsv_path = Path(args.tsv_root) / f"{rec_time}.tsv"
        if not video_path.exists() or not tsv_path.exists():
            continue
        # Need all 4 corner points
        try:
            lt = entry["Point_LT"]
            rt = entry["Point_RT"]
            rb = entry["Point_RB"]
            lb = entry["Point_LB"]
        except KeyError:
            continue
        corners_arg = ";".join([lt, rt, rb, lb])

        per_dir = out_dir / "per_video" / rec_time
        cmd = [
            sys.executable, str(REPO_ROOT / "tools" / "classical_piano_detector.py"),
            "--video", str(video_path),
            "--tsv", str(tsv_path),
            "--metadata-corners", corners_arg,
            "--out-dir", str(per_dir),
            "--target-fps", str(args.target_fps),
            "--max-seconds", str(args.max_seconds),
            "--smoothing-window", str(args.smoothing_window),
            "--peak-threshold-factor", str(args.peak_threshold_factor),
            "--min-event-gap-s", str(args.min_event_gap_s),
            "--onset-tolerance-ms", str(args.onset_tolerance_ms),
        ]
        print(f"\n>>> {canon_split}/{rec_time} ...")
        try:
            subprocess.run(cmd, check=True)
            with (per_dir / "result.json").open("r", encoding="utf-8") as fh:
                res = json.load(fh)
            m = res["metrics"]
            rows.append({
                "split": canon_split,
                "video_id": rec_time,
                "n_pred": m["n_pred"],
                "n_ref": m["n_ref"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "f1_white": m["f1_white"],
                "f1_black": m["f1_black"],
            })
            splits_seen[canon_split] = splits_seen.get(canon_split, 0) + 1
        except subprocess.CalledProcessError as exc:
            print(f"  FAILED: {exc}")
            continue

    # Aggregate
    csv_path = out_dir / "all_videos_f1.csv"
    if rows:
        fields = ["split", "video_id", "n_pred", "n_ref", "tp", "fp", "fn",
                  "precision", "recall", "f1", "f1_white", "f1_black"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # Summary
    lines: List[str] = [f"Evaluated {len(rows)} videos."]
    for split in args.splits:
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        f1s = [r["f1"] for r in sub]
        wf1s = [r["f1_white"] for r in sub]
        bf1s = [r["f1_black"] for r in sub]
        lines.append(
            f"  [{split}] n={len(sub)}  "
            f"mean_F1={statistics.fmean(f1s):.3f} median_F1={statistics.median(f1s):.3f} "
            f"max_F1={max(f1s):.3f} min_F1={min(f1s):.3f}  "
            f"mean_white_F1={statistics.fmean(wf1s):.3f} mean_black_F1={statistics.fmean(bf1s):.3f}"
        )
    summary = "\n".join(lines)
    print()
    print(summary)
    (out_dir / "SUMMARY.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
