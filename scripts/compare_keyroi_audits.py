#!/usr/bin/env python3
"""Compare two PianoVAM audit summary.json files video-by-video.

Reads `summary.json` produced by `tools/audit_pianovam_alignment.py` from a
"before" and an "after" run and reports per-video registration deltas plus
aggregate distribution stats. Writes `audit_comparison.csv`.

Usage:
    PYTHONPATH=. python scripts/compare_keyroi_audits.py \\
        --before /path/to/audit_before_train/summary.json \\
        --after  /path/to/audit_after_fix/summary.json \\
        --out    /path/to/audit_comparison.csv

This is the source of truth for "did the geometry actually change?". If
`geometry_changed` is False everywhere, the audit overlay can look prettier
without the underlying per-video calibration moving — exactly the failure
mode we hit on the canonical-fallback patch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, List, Mapping


_FIELDS = [
    "split",
    "video_id",
    "registration_status_before",
    "registration_status_after",
    "registration_err_before__before",
    "registration_err_after__before",
    "registration_err_before__after",
    "registration_err_after__after",
    "delta_err_after",
    "white_width_mean__after",
    "normalized_err_after__before",
    "normalized_err_after__after",
    "delta_normalized_err_after",
    "key_bounds_count_before",
    "key_bounds_count_after",
    "geometry_signature_before",
    "geometry_signature_after",
    "geometry_changed",
    "bounds_look_canonical_before",
    "bounds_look_canonical_after",
    "overlay_path_before",
    "overlay_path_after",
]


def _load_rows(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit(f"{path} is not a list of rows")
    return data


def _index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        split = str(row.get("split") or "")
        vid = str(row.get("video_id") or "")
        if not split or not vid:
            continue
        out[(split, vid)] = row
    return out


def _f(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _percentile(values: List[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _aggregate(label: str, values: List[float], width_norm: List[float]) -> str:
    if not values:
        return f"{label}: n=0"
    return (
        f"{label}: n={len(values)} "
        f"mean={statistics.fmean(values):.3f} "
        f"median={statistics.median(values):.3f} "
        f"p90={_percentile(values, 0.90):.3f} "
        f"p95={_percentile(values, 0.95):.3f} "
        f"max={max(values):.3f} "
        f"|| normalized: median={statistics.median(width_norm):.3f} "
        f"p95={_percentile(width_norm, 0.95):.3f}" if width_norm else f"{label}: n={len(values)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True, help="summary.json from the BEFORE audit run")
    ap.add_argument("--after", required=True, help="summary.json from the AFTER audit run")
    ap.add_argument("--out", required=True, help="path for audit_comparison.csv")
    ap.add_argument("--err-bad-px", type=float, default=4.0, help="absolute err_after threshold considered 'bad'")
    ap.add_argument("--err-very-bad-px", type=float, default=5.0, help="absolute err_after threshold considered 'very bad'")
    ap.add_argument("--norm-bad", type=float, default=0.10, help="normalized err threshold considered 'bad'")
    ap.add_argument("--worst-n", type=int, default=15, help="how many worst videos to print")
    args = ap.parse_args()

    before_rows = _load_rows(Path(args.before).expanduser().resolve())
    after_rows = _load_rows(Path(args.after).expanduser().resolve())
    before = _index(before_rows)
    after = _index(after_rows)

    keys = sorted(set(before.keys()) | set(after.keys()))

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    err_after_after: List[float] = []
    norm_after_after: List[float] = []
    err_after_before: List[float] = []
    norm_after_before: List[float] = []
    fallback_worsened_after = 0
    fallback_worsened_before = 0
    bad_abs_after = 0
    very_bad_abs_after = 0
    bad_norm_after = 0
    geometry_changed = 0
    geometry_changed_with_improvement = 0
    canonical_only_after = 0

    deltas: List[tuple[str, str, float, float | None]] = []

    rows_out: List[dict[str, Any]] = []
    for split, vid in keys:
        b = before.get((split, vid), {})
        a = after.get((split, vid), {})

        sig_b = b.get("geometry_signature")
        sig_a = a.get("geometry_signature")
        changed = bool(sig_b and sig_a and sig_b != sig_a)

        eb_a = _f(a.get("registration_err_after"))
        eb_b = _f(b.get("registration_err_after"))
        nm_a = _f(a.get("normalized_err_after"))
        nm_b = _f(b.get("normalized_err_after"))

        if eb_a is not None:
            err_after_after.append(eb_a)
        if nm_a is not None:
            norm_after_after.append(nm_a)
        if eb_b is not None:
            err_after_before.append(eb_b)
        if nm_b is not None:
            norm_after_before.append(nm_b)

        if str(a.get("registration_status") or "") == "fallback_worsened":
            fallback_worsened_after += 1
        if str(b.get("registration_status") or "") == "fallback_worsened":
            fallback_worsened_before += 1
        if eb_a is not None and eb_a > args.err_bad_px:
            bad_abs_after += 1
        if eb_a is not None and eb_a > args.err_very_bad_px:
            very_bad_abs_after += 1
        if nm_a is not None and nm_a > args.norm_bad:
            bad_norm_after += 1
        if a.get("bounds_look_canonical") is True and not changed:
            canonical_only_after += 1
        if changed:
            geometry_changed += 1
            if eb_a is not None and eb_b is not None and eb_a < eb_b - 0.05:
                geometry_changed_with_improvement += 1

        delta_err = (eb_a - eb_b) if (eb_a is not None and eb_b is not None) else None
        delta_norm = (nm_a - nm_b) if (nm_a is not None and nm_b is not None) else None
        if eb_a is not None:
            deltas.append((split, vid, eb_a, delta_err))

        rows_out.append(
            {
                "split": split,
                "video_id": vid,
                "registration_status_before": b.get("registration_status"),
                "registration_status_after": a.get("registration_status"),
                "registration_err_before__before": b.get("registration_err_before"),
                "registration_err_after__before": b.get("registration_err_after"),
                "registration_err_before__after": a.get("registration_err_before"),
                "registration_err_after__after": a.get("registration_err_after"),
                "delta_err_after": delta_err,
                "white_width_mean__after": a.get("white_width_mean"),
                "normalized_err_after__before": nm_b,
                "normalized_err_after__after": nm_a,
                "delta_normalized_err_after": delta_norm,
                "key_bounds_count_before": b.get("key_bounds_count"),
                "key_bounds_count_after": a.get("key_bounds_count"),
                "geometry_signature_before": sig_b,
                "geometry_signature_after": sig_a,
                "geometry_changed": changed,
                "bounds_look_canonical_before": b.get("bounds_look_canonical"),
                "bounds_look_canonical_after": a.get("bounds_look_canonical"),
                "overlay_path_before": b.get("overlay_path"),
                "overlay_path_after": a.get("overlay_path"),
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    def _line(label: str, values: List[float]) -> None:
        if not values:
            print(f"  {label}: n=0")
            return
        print(
            f"  {label}: n={len(values)} mean={statistics.fmean(values):.3f} "
            f"median={statistics.median(values):.3f} p90={_percentile(values, 0.90):.3f} "
            f"p95={_percentile(values, 0.95):.3f} max={max(values):.3f}"
        )

    print(f"Wrote {out_path}")
    print(f"Videos compared: {len(keys)} (before={len(before)} after={len(after)})")
    print()
    print("err_after distribution (px):")
    _line("BEFORE", err_after_before)
    _line("AFTER ", err_after_after)
    print()
    print("normalized_err_after distribution (err / white_width_mean):")
    _line("BEFORE", norm_after_before)
    _line("AFTER ", norm_after_after)
    print()
    print(
        f"Fallback_worsened: before={fallback_worsened_before} after={fallback_worsened_after}"
    )
    print(
        f"After-run videos with err_after > {args.err_bad_px:.1f} px: {bad_abs_after}"
    )
    print(
        f"After-run videos with err_after > {args.err_very_bad_px:.1f} px: {very_bad_abs_after}"
    )
    print(
        f"After-run videos with normalized err > {args.norm_bad:.2f}: {bad_norm_after}"
    )
    print(
        f"Videos whose key_bounds_px ACTUALLY changed between runs: {geometry_changed}"
    )
    print(
        f"  ... of which err_after improved by >0.05 px: {geometry_changed_with_improvement}"
    )
    print(
        f"After-run videos still on canonical-uniform fallback (no change): {canonical_only_after}"
    )
    print()
    print(f"Worst-{args.worst_n} videos by err_after (after run):")
    deltas.sort(key=lambda r: r[2], reverse=True)
    for split, vid, err, dlt in deltas[: args.worst_n]:
        dlt_str = f"{dlt:+.3f}" if dlt is not None else " n/a"
        print(f"  err_after={err:7.3f}  delta_vs_before={dlt_str}  split={split:5s}  video={vid}")


if __name__ == "__main__":
    main()
