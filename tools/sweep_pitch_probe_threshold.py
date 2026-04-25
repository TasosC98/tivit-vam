#!/usr/bin/env python3
"""Sweep pitch/key thresholds and rank the resulting pitch-aware metrics."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep pitch thresholds and rank pitch-aware F1 metrics")
    ap.add_argument("--config", action="append", default=None, help="One or more config fragments to merge")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate")
    ap.add_argument("--calibration-path", default=None, help="Optional calibration JSON override to reuse")
    ap.add_argument("--split", default="valid", help="Dataset split to evaluate")
    ap.add_argument("--metric", default="patk_note_f1", help="Metric used to rank thresholds")
    ap.add_argument("--threshold", action="append", type=float, default=None, help="Explicit threshold(s) to test")
    ap.add_argument("--start", type=float, default=0.01, help="Start of threshold range when --threshold is omitted")
    ap.add_argument("--stop", type=float, default=0.50, help="End of threshold range when --threshold is omitted")
    ap.add_argument("--step", type=float, default=0.01, help="Step size for threshold range when --threshold is omitted")
    ap.add_argument("--top-k", type=int, default=10, help="How many rows to print after ranking")
    ap.add_argument("--work-dir", default=None, help="Optional output directory for per-threshold logs and summaries")
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="info")
    ap.add_argument("--max-batches", dest="max_batches", type=int)
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def build_thresholds(
    explicit: Iterable[float] | None,
    *,
    start: float,
    stop: float,
    step: float,
) -> list[float]:
    if explicit:
        values = [float(value) for value in explicit]
    else:
        if step <= 0.0:
            raise ValueError("--step must be > 0")
        lo = min(float(start), float(stop))
        hi = max(float(start), float(stop))
        values = []
        current = lo
        while current <= hi + (step * 0.5):
            values.append(current)
            current += step
    cleaned = sorted({round(min(max(value, 0.0), 1.0), 6) for value in values})
    if not cleaned:
        raise ValueError("No thresholds to evaluate")
    return cleaned


def metric_value(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _write_override(path: Path, *, log_dir: Path, threshold: float, calibration_path: str | None) -> None:
    payload: dict[str, object] = {
        "logging": {
            "log_dir": log_dir.as_posix(),
        },
        "training": {
            "metrics": {
                "key_probe": {
                    "threshold": float(threshold),
                },
                "patk": {
                    "threshold": float(threshold),
                },
            },
        },
    }
    if calibration_path:
        payload["calibration"] = {
            "output_path": str(Path(calibration_path).expanduser().as_posix()),
        }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    return {
        "threshold": record.get("threshold"),
        "metric": record.get("metric"),
        "metric_value": record.get("metric_value"),
        "ev_f1_mean": metrics.get("ev_f1_mean"),
        "ev_f1_mean_anypitch": metrics.get("ev_f1_mean_anypitch"),
        "patk_frame_f1": metrics.get("patk_frame_f1"),
        "patk_onset_f1": metrics.get("patk_onset_f1"),
        "patk_note_f1": metrics.get("patk_note_f1"),
        "pitch_active_precision": metrics.get("pitch_active_precision"),
        "pitch_active_recall": metrics.get("pitch_active_recall"),
        "pitch_active_f1": metrics.get("pitch_active_f1"),
        "pitch_inactive_specificity": metrics.get("pitch_inactive_specificity"),
        "pitch_exact_frame_match": metrics.get("pitch_exact_frame_match"),
        "pitch_frame_jaccard": metrics.get("pitch_frame_jaccard"),
        "pitch_pred_active_per_frame": metrics.get("pitch_pred_active_per_frame"),
        "pitch_gt_active_per_frame": metrics.get("pitch_gt_active_per_frame"),
        "pitch_polyphony_mae": metrics.get("pitch_polyphony_mae"),
    }


def _write_summary(records: list[Mapping[str, Any]], work_dir: Path) -> tuple[Path, Path]:
    json_path = work_dir / "summary.json"
    csv_path = work_dir / "summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    rows = [_flatten_record(record) for record in records]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def main() -> None:
    args = _parse_args()
    from tivit.pipelines.eval_single import eval_single

    base_configs = [Path(path) for path in (args.config or [Path("configs/default.yaml")])]
    thresholds = build_thresholds(args.threshold, start=args.start, stop=args.stop, step=args.step)

    managed_tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        managed_tmp = tempfile.TemporaryDirectory(prefix="pitch-threshold-sweep-")
        work_dir = Path(managed_tmp.name)

    records: list[Mapping[str, Any]] = []
    try:
        for threshold in thresholds:
            threshold_tag = f"{threshold:.6f}".rstrip("0").rstrip(".")
            threshold_dir = work_dir / f"thr_{threshold_tag.replace('.', 'p')}"
            threshold_dir.mkdir(parents=True, exist_ok=True)
            override_path = threshold_dir / "_auto_threshold_override.yaml"
            _write_override(
                override_path,
                log_dir=threshold_dir,
                threshold=threshold,
                calibration_path=args.calibration_path,
            )

            configs = [*base_configs, override_path]
            metrics = eval_single(
                configs=configs,
                verbose=args.verbose,
                split=args.split,
                checkpoint=args.checkpoint,
                max_batches=args.max_batches,
                max_clips=args.max_clips,
                frames=args.frames,
                seed=args.seed,
                deterministic=args.deterministic,
                smoke=bool(args.smoke),
            )

            record = {
                "threshold": float(threshold),
                "metric": args.metric,
                "metric_value": metric_value(metrics, args.metric),
                "metrics": dict(metrics),
            }
            records.append(record)
            print(
                f"[threshold] {threshold:.4f} "
                f"{args.metric}={metric_value(metrics, args.metric):.6f} "
                f"pitch_active_f1={metric_value(metrics, 'pitch_active_f1'):.6f} "
                f"patk_note_f1={metric_value(metrics, 'patk_note_f1'):.6f} "
                f"pred_keys/frame={metric_value(metrics, 'pitch_pred_active_per_frame'):.2f}"
            )

        records.sort(key=lambda record: metric_value(record, "metric_value"), reverse=True)

        if args.work_dir:
            json_path, csv_path = _write_summary(records, work_dir)
            print(f"\nWrote summary:\n  {json_path}\n  {csv_path}")

        print("\nTop thresholds:")
        for record in records[: max(int(args.top_k), 1)]:
            metrics = record.get("metrics", {})
            metrics = metrics if isinstance(metrics, Mapping) else {}
            print(
                f"  thr={float(record.get('threshold', 0.0)):.4f} "
                f"{args.metric}={float(record.get('metric_value', float('-inf'))):.6f} "
                f"pitch_active_f1={metric_value(metrics, 'pitch_active_f1'):.6f} "
                f"patk_note_f1={metric_value(metrics, 'patk_note_f1'):.6f} "
                f"pred_keys/frame={metric_value(metrics, 'pitch_pred_active_per_frame'):.2f} "
                f"polyphony_mae={metric_value(metrics, 'pitch_polyphony_mae'):.2f}"
            )
    finally:
        if managed_tmp is not None:
            managed_tmp.cleanup()


if __name__ == "__main__":
    main()
