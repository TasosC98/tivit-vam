#!/usr/bin/env python3
"""Rebuild key_bounds_px in registration cache files using canonical/uniform spacing.

Why:
    `_key_bounds_from_white_edges` produced wonky per-image bounds (white-key
    widths spanning 17..49 px instead of the canonical ~29.54 px) because the
    homography-projected white edges fed into it were missing or duplicated.
    The rectified canvas itself is canonical after the homography + x_warp_ctrl
    are applied to the frames, so the correct `key_bounds_px` is the canonical
    evenly-spaced layout. This tool overwrites the corrupted bounds in-place.

Usage:
    PYTHONPATH=. python tools/rebuild_canonical_geometry.py \
        --cache reg_refined_pianovam.json

    PYTHONPATH=. python tools/rebuild_canonical_geometry.py \
        --cache reg_refined_pianovam.json \
        --out reg_refined_pianovam_canonical.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_MIDI_LOW = 21
_MIDI_HIGH = 108
_KEY_COUNT = _MIDI_HIGH - _MIDI_LOW + 1
_WHITE_PITCHES = {0, 2, 4, 5, 7, 9, 11}


def _midi_is_white(midi: int) -> bool:
    return (midi % 12) in _WHITE_PITCHES


def _canonical_key_bounds(width: float) -> List[List[float]]:
    """Return canonical evenly-spaced [left, right] pairs for 88 keys."""

    span = float(max(width, 1.0))
    white_w = span / 52.0
    # White-key edges (53 of them, from 0 to span)
    white_edges = [i * white_w for i in range(53)]
    bounds: List[List[float]] = []
    white_idx = 0
    for key_idx in range(_KEY_COUNT):
        midi = _MIDI_LOW + key_idx
        if _midi_is_white(midi):
            left = white_edges[white_idx]
            right = white_edges[white_idx + 1]
            bounds.append([float(left), float(right)])
            white_idx += 1
        else:
            # Black key: centered between adjacent white-key centers,
            # 0.6x the canonical white-key width.
            left_center = 0.5 * (white_edges[white_idx - 1] + white_edges[white_idx])
            right_center = 0.5 * (white_edges[white_idx] + white_edges[white_idx + 1])
            center = 0.5 * (left_center + right_center)
            half = 0.3 * white_w  # 0.6 * white_w / 2
            bounds.append([float(center - half), float(center + half)])
    return bounds


def _uniform_tile_bounds(num_tiles: int, width: float) -> List[List[float]]:
    if num_tiles <= 0:
        return []
    step = float(width) / float(num_tiles)
    return [[float(i * step), float((i + 1) * step)] for i in range(num_tiles)]


def _white_widths(bounds: Sequence[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for idx, pair in enumerate(bounds):
        midi = _MIDI_LOW + idx
        if _midi_is_white(midi):
            out.append(float(pair[1]) - float(pair[0]))
    return out


def _bounds_canonical_ok(bounds: Sequence[Sequence[float]], width: float, tolerance: float = 0.4) -> bool:
    if not bounds or width <= 0.0:
        return False
    canonical = float(width) / 52.0
    lo, hi = (1 - tolerance) * canonical, (1 + tolerance) * canonical
    for w in _white_widths(bounds):
        if w < lo or w > hi:
            return False
    return True


def _rebuild_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    target_hw = payload.get("target_hw")
    if not isinstance(target_hw, (list, tuple)) or len(target_hw) < 2:
        return payload
    width = float(target_hw[1])
    geometry = payload.get("geometry_meta")
    if not isinstance(geometry, dict):
        geometry = {}
    geometry["rectified_width"] = float(geometry.get("rectified_width") or width)
    geometry["target_hw"] = [int(target_hw[0]), int(target_hw[1])]
    geometry["key_bounds_px"] = _canonical_key_bounds(width)
    if not isinstance(geometry.get("tile_bounds_px"), list) or len(geometry.get("tile_bounds_px") or []) == 0:
        geometry["tile_bounds_px"] = _uniform_tile_bounds(3, width)
    payload["geometry_meta"] = geometry
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="Path to reg_refined_*.json to repair")
    ap.add_argument("--out", default=None, help="Optional output path. Defaults to in-place rewrite (with .bak backup).")
    ap.add_argument("--dry-run", action="store_true", help="Print what would change without writing.")
    args = ap.parse_args()

    cache_path = Path(args.cache).expanduser().resolve()
    if not cache_path.exists():
        raise SystemExit(f"cache file not found: {cache_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else cache_path

    with cache_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"unexpected cache schema in {cache_path}")

    repaired = 0
    already_ok = 0
    examined = 0
    for video_id, payload in list(data.items()):
        if not isinstance(payload, dict):
            continue
        examined += 1
        target_hw = payload.get("target_hw")
        width = float(target_hw[1]) if isinstance(target_hw, (list, tuple)) and len(target_hw) >= 2 else 0.0
        geometry = payload.get("geometry_meta") if isinstance(payload.get("geometry_meta"), dict) else {}
        existing = geometry.get("key_bounds_px") if isinstance(geometry, dict) else None
        if isinstance(existing, list) and _bounds_canonical_ok(existing, width):
            already_ok += 1
            continue
        widths = _white_widths(existing) if isinstance(existing, list) else []
        if widths:
            print(
                f"[repair] {video_id} target_hw={target_hw} "
                f"white_w min={min(widths):.2f} max={max(widths):.2f} mean={sum(widths)/len(widths):.2f} "
                f"(canonical={width/52.0:.2f})"
            )
        else:
            print(f"[repair] {video_id} target_hw={target_hw} (no key_bounds_px present)")
        data[video_id] = _rebuild_entry(payload)
        repaired += 1

    print(
        f"\nExamined {examined} entries — repaired={repaired} already_ok={already_ok}"
    )

    if args.dry_run:
        print("(dry-run: no files written)")
        return

    if out_path == cache_path and repaired > 0:
        backup = cache_path.with_suffix(cache_path.suffix + ".bak")
        shutil.copyfile(cache_path, backup)
        print(f"Backed up original to {backup}")

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    print(f"Wrote repaired cache to {out_path}")


if __name__ == "__main__":
    main()
