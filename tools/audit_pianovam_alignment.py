#!/usr/bin/env python3
"""Export PianoVAM keyboard-boundary overlays and metadata sync reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit PianoVAM video/metadata/keyboard alignment")
    ap.add_argument("--config", action="append", default=None, help="Config stack to audit")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Splits to export")
    ap.add_argument("--out-dir", required=True, help="Directory for PNG overlays and summary files")
    ap.add_argument("--max-videos", type=int, default=0, help="Limit per split; 0 means all")
    ap.add_argument("--frame-index", type=int, default=0, help="Frame inside each sampled clip to draw")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any audit issue is found")
    return ap.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _geometry_signature(geometry: Mapping[str, Any] | None) -> str | None:
    """Stable hash of key_bounds_px (rounded) so two audits can detect whether
    the per-video geometry actually changed between runs."""

    if not isinstance(geometry, Mapping):
        return None
    bounds = geometry.get("key_bounds_px")
    if not isinstance(bounds, Sequence):
        return None
    import hashlib
    parts: list[str] = []
    for pair in bounds:
        if not isinstance(pair, Sequence) or len(pair) < 2:
            continue
        l = _safe_float(pair[0])
        r = _safe_float(pair[1])
        if l is None or r is None:
            continue
        parts.append(f"{l:.3f},{r:.3f}")
    if not parts:
        return None
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _video_to_rgb(video: torch.Tensor, *, frame_index: int, mean: Sequence[float], std: Sequence[float]) -> Any:
    import numpy as np

    if video.dim() != 5:
        raise ValueError(f"Expected sample video shape (T,C,K,H,W), got {tuple(video.shape)}")
    idx = max(0, min(int(frame_index), int(video.shape[0]) - 1))
    frame = video[idx].detach().float().cpu()  # C,K,H,W
    tiles = []
    channels = int(frame.shape[0])
    for tile_idx in range(int(frame.shape[1])):
        tile = frame[:, tile_idx].clone()
        for c in range(channels):
            m = float(mean[c] if c < len(mean) else mean[-1])
            s = float(std[c] if c < len(std) else std[-1])
            tile[c] = tile[c] * s + m
        tile = tile.clamp(0.0, 1.0)
        arr = tile.permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        tiles.append(arr)
    return (np.concatenate(tiles, axis=1) * 255.0).round().astype("uint8")


def _is_black_key(midi: int) -> bool:
    return int(midi) % 12 in {1, 3, 6, 8, 10}


def _draw_key_overlay(image_rgb: Any, geometry: Mapping[str, Any] | None, *, note_min: int) -> tuple[Any, dict[str, Any]]:
    import cv2
    import numpy as np

    overlay = image_rgb.copy()
    info: dict[str, Any] = {
        "geometry_present": False,
        "key_bounds_count": 0,
        "white_width_min": None,
        "white_width_max": None,
        "white_width_mean": None,
        "canonical_white_width": None,
        "bounds_look_canonical": None,
    }
    if not isinstance(geometry, Mapping):
        return overlay, info
    key_bounds = geometry.get("key_bounds_px")
    target_hw = geometry.get("target_hw") or geometry.get("canonical_hw")
    if not isinstance(key_bounds, Sequence) or not isinstance(target_hw, Sequence) or len(target_hw) < 2:
        return overlay, info
    target_w = _safe_float(target_hw[1])
    if target_w is None or target_w <= 0:
        return overlay, info

    h, w = overlay.shape[:2]
    scale_x = float(w) / target_w
    info["geometry_present"] = True
    info["key_bounds_count"] = len(key_bounds)

    canonical_white = float(target_w) / 52.0
    info["canonical_white_width"] = canonical_white
    white_widths: list[float] = []
    for idx, pair in enumerate(key_bounds):
        if not isinstance(pair, Sequence) or len(pair) < 2:
            continue
        midi = int(note_min) + idx
        if not _is_black_key(midi):
            l = _safe_float(pair[0]) or 0.0
            r = _safe_float(pair[1]) or 0.0
            white_widths.append(float(r - l))
    if white_widths:
        wmin, wmax = float(min(white_widths)), float(max(white_widths))
        wmean = float(sum(white_widths) / len(white_widths))
        info["white_width_min"] = wmin
        info["white_width_max"] = wmax
        info["white_width_mean"] = wmean
        info["bounds_look_canonical"] = bool(
            wmin >= 0.6 * canonical_white and wmax <= 1.4 * canonical_white
        )

    blended = overlay.copy()
    for idx, bound in enumerate(key_bounds):
        if not isinstance(bound, Sequence) or len(bound) < 2:
            continue
        left = _safe_float(bound[0])
        right = _safe_float(bound[1])
        if left is None or right is None:
            continue
        midi = int(note_min) + idx
        x1 = int(np.clip(round(left * scale_x), 0, w - 1))
        x2 = int(np.clip(round(right * scale_x), 0, w - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if _is_black_key(midi):
            y1, y2 = 0, max(1, int(round(h * 0.58)))
            color = (255, 70, 50)
            alpha = 0.18
        else:
            y1, y2 = 0, h - 1
            color = (50, 210, 80)
            alpha = 0.08
        cv2.rectangle(blended, (x1, y1), (x2, y2), color, thickness=-1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=1)
        if midi % 12 == 0:
            cv2.putText(
                overlay,
                str(midi),
                (max(0, x1 + 2), min(h - 5, y2 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    overlay = cv2.addWeighted(blended, 0.35, overlay, 0.65, 0)
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
    return overlay, info


def _target_count(sample: Mapping[str, Any], key: str) -> int | None:
    value = sample.get(key)
    if torch.is_tensor(value):
        return int((value > 0.5).sum().item())
    return None


def _event_stats(events: Sequence[Sequence[Any]]) -> dict[str, Any]:
    onsets: list[float] = []
    offsets: list[float] = []
    pitches: list[int] = []
    invalid_offsets = 0
    for event in events:
        if len(event) < 3:
            continue
        onset = _safe_float(event[0])
        offset = _safe_float(event[1])
        pitch = _safe_int(round(float(event[2]))) if _safe_float(event[2]) is not None else None
        if onset is None or offset is None or pitch is None:
            continue
        onsets.append(onset)
        offsets.append(offset)
        pitches.append(pitch)
        if offset <= onset:
            invalid_offsets += 1
    return {
        "events_total": len(onsets),
        "invalid_offsets": invalid_offsets,
        "min_onset": min(onsets) if onsets else None,
        "max_onset": max(onsets) if onsets else None,
        "min_offset": min(offsets) if offsets else None,
        "max_offset": max(offsets) if offsets else None,
        "min_pitch": min(pitches) if pitches else None,
        "max_pitch": max(pitches) if pitches else None,
    }


def _write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _audit_split(cfg: Mapping[str, Any], split: str, out_dir: Path, *, max_videos: int, frame_index: int) -> list[dict[str, Any]]:
    import cv2

    from tivit.data.datasets.pianovam_impl import PianoVAMDataset

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ds = PianoVAMDataset(cfg, split=split, full_cfg=cfg)
    note_min = int((cfg.get("dataset", {}) or {}).get("frame_targets", {}).get("note_min", 21))
    limit = len(ds) if max_videos <= 0 else min(len(ds), max_videos)
    rows: list[dict[str, Any]] = []

    for idx in range(limit):
        entry = ds.entries[idx]
        raw = ds._read_labels(entry)
        events = raw.get("events", []) if isinstance(raw, Mapping) else []
        row: dict[str, Any] = {
            "split": split,
            "index": idx,
            "video_id": entry.video_id,
            "canonical_id": __import__("tivit.data.targets.identifiers", fromlist=["canonical_video_id"]).canonical_video_id(entry.video_id),
            "video_path": str(entry.video_path),
            "label_path": str(entry.label_path) if entry.label_path else None,
            "label_exists": bool(entry.label_path and entry.label_path.exists()),
            "metadata_record_time": entry.metadata.get("record_time") if isinstance(entry.metadata, Mapping) else None,
            "metadata_split": entry.metadata.get("split") if isinstance(entry.metadata, Mapping) else None,
            "metadata_record_time_matches_video": (
                entry.metadata.get("record_time") == entry.video_path.stem if isinstance(entry.metadata, Mapping) and entry.metadata.get("record_time") else None
            ),
            "metadata_split_matches_request": (
                entry.metadata.get("split") == split if isinstance(entry.metadata, Mapping) and entry.metadata.get("split") else None
            ),
        }
        row.update(_event_stats(events))
        try:
            sample = ds[idx]
            debug = sample.get("_debug_extras", {})
            registration = debug.get("registration", {}) if isinstance(debug, Mapping) else {}
            geometry = registration.get("cache_geometry") if isinstance(registration, Mapping) else None
            image = _video_to_rgb(sample["video"], frame_index=frame_index, mean=ds.norm_mean, std=ds.norm_std)
            overlay, geom_info = _draw_key_overlay(image, geometry, note_min=note_min)
            overlay_path = split_dir / f"{idx:05d}_{entry.video_id}_keys.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            row.update(
                {
                    "status": "ok",
                    "overlay_path": str(overlay_path),
                    "registration_status": registration.get("status") if isinstance(registration, Mapping) else None,
                    "registration_err_before": registration.get("err_before") if isinstance(registration, Mapping) else None,
                    "registration_err_after": registration.get("err_after") if isinstance(registration, Mapping) else None,
                    "registration_err_white": registration.get("err_white") if isinstance(registration, Mapping) else None,
                    "registration_err_black": registration.get("err_black") if isinstance(registration, Mapping) else None,
                    "onset_target_count": _target_count(sample, "onset"),
                    "offset_target_count": _target_count(sample, "offset"),
                    "pitch_target_count": _target_count(sample, "pitch"),
                }
            )
            row.update(geom_info)
            err_after = _safe_float(registration.get("err_after")) if isinstance(registration, Mapping) else None
            white_mean = _safe_float(geom_info.get("white_width_mean"))
            row["normalized_err_after"] = (
                float(err_after) / float(white_mean)
                if (err_after is not None and white_mean is not None and white_mean > 0.0)
                else None
            )
            row["geometry_signature"] = _geometry_signature(geometry)
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc)})
        rows.append(row)
        print(
            "[audit] split={} {}/{} video={} status={} reg={} keys={}".format(
                split,
                idx + 1,
                limit,
                entry.video_id,
                row.get("status"),
                row.get("registration_status"),
                row.get("key_bounds_count"),
            ),
            flush=True,
        )
    return rows


def main() -> None:
    args = _parse_args()
    from tivit.core.config import load_experiment_config

    cfg = dict(load_experiment_config(args.config or [Path("configs/default.yaml")]))
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    testing_cfg = dict(dataset_cfg.get("testing", {}) or {})
    testing_cfg["enable_debug_extras"] = True
    dataset_cfg["testing"] = testing_cfg
    dataset_cfg["preprocessed_format"] = "video"
    dataset_cfg["shuffle"] = False
    cfg["dataset"] = dataset_cfg

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split in args.splits:
        rows.extend(
            _audit_split(
                cfg,
                str(split),
                out_dir,
                max_videos=int(args.max_videos),
                frame_index=int(args.frame_index),
            )
        )
    _write_summary(rows, out_dir)

    failed = [row for row in rows if row.get("status") != "ok"]
    missing_geometry = [row for row in rows if int(row.get("key_bounds_count") or 0) < 88]
    metadata_bad = [
        row
        for row in rows
        if row.get("metadata_record_time_matches_video") is False or row.get("metadata_split_matches_request") is False
    ]
    bounds_evaluated = [row for row in rows if row.get("bounds_look_canonical") is not None]
    bounds_canonical = [row for row in bounds_evaluated if row.get("bounds_look_canonical")]
    bounds_bad = [row for row in bounds_evaluated if row.get("bounds_look_canonical") is False]

    print(f"\nWrote audit to {out_dir}")
    print(f"Rows: {len(rows)} failed={len(failed)} missing_geometry={len(missing_geometry)} metadata_bad={len(metadata_bad)}")
    if bounds_evaluated:
        pct = 100.0 * len(bounds_canonical) / max(len(bounds_evaluated), 1)
        print(
            f"Key-bounds canonical alignment: {len(bounds_canonical)}/{len(bounds_evaluated)} videos OK"
            f" ({pct:.1f}%) — bad={len(bounds_bad)}"
        )
        if bounds_bad:
            print("First 10 mis-aligned videos:")
            for row in bounds_bad[:10]:
                print(
                    "  split={split} idx={idx} video={vid} white_w[min/mean/max]="
                    "{wmin:.2f}/{wmean:.2f}/{wmax:.2f} (canonical={canon:.2f})".format(
                        split=row.get("split"),
                        idx=row.get("index"),
                        vid=row.get("video_id"),
                        wmin=float(row.get("white_width_min") or 0.0),
                        wmean=float(row.get("white_width_mean") or 0.0),
                        wmax=float(row.get("white_width_max") or 0.0),
                        canon=float(row.get("canonical_white_width") or 0.0),
                    )
                )
    if args.strict and (failed or missing_geometry or metadata_bad or bounds_bad):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
