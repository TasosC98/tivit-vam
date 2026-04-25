#!/usr/bin/env python3
"""Sweep checkpoints with calibrate + eval_single and rank by F1.

Purpose:
    - Reuse the existing calibration and eval_single pipelines across many checkpoints.
    - Persist one calibration JSON per checkpoint and rank results by a chosen metric.
    - Make it easy to search long training runs (for example 200 epochs) without
      manually wiring temporary config overrides.

Example:
    python tools/sweep_calibrated_eval.py \
        --base-config configs/default.yaml \
        --base-config configs/overrides/manual_training.yaml \
        --calib-config configs/overrides/manual_training_best_calibration.yaml \
        --eval-config configs/overrides/manual_training_best_eval.yaml \
        --checkpoint-dir /home/achatzigiannis/tivit-logs/manual_training/checkpoints \
        --work-dir /home/achatzigiannis/tivit-logs/manual_training/sweeps/best_known \
        --split valid \
        --metric ev_f1_mean
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EPOCH_RE = re.compile(r"epoch_(\d+)\.pt$")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calibrate + evaluate many checkpoints and rank them by F1")
    ap.add_argument("--base-config", action="append", default=[], help="Shared config(s) used by both stages")
    ap.add_argument("--calib-config", action="append", default=[], help="Extra config(s) used only for calibration")
    ap.add_argument("--eval-config", action="append", default=[], help="Extra config(s) used only for evaluation")
    ap.add_argument("--checkpoint-dir", required=True, help="Directory containing epoch_*.pt checkpoints")
    ap.add_argument("--work-dir", required=True, help="Directory where per-epoch outputs and summaries are written")
    ap.add_argument("--epochs", nargs="*", type=int, default=None, help="Optional explicit epoch numbers to sweep")
    ap.add_argument("--split", default="valid", help="Evaluation split")
    ap.add_argument("--metric", default="ev_f1_mean", help="Metric used for ranking")
    ap.add_argument("--top-k", type=int, default=10, help="How many rows to print after ranking")
    ap.add_argument("--max-batches", type=int)
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="info")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--reuse-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing per-epoch calibration.json if it already exists",
    )
    return ap.parse_args()


def _metric_value(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _discover_checkpoints(checkpoint_dir: Path, epochs: Iterable[int] | None) -> list[tuple[int, Path]]:
    requested = {int(epoch) for epoch in epochs} if epochs is not None else None
    found: list[tuple[int, Path]] = []
    for path in sorted(checkpoint_dir.glob("epoch_*.pt")):
        match = EPOCH_RE.search(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if requested is not None and epoch not in requested:
            continue
        found.append((epoch, path))
    return found


def _write_temp_override(epoch_dir: Path, calibration_path: Path) -> Path:
    epoch_dir.mkdir(parents=True, exist_ok=True)
    override_path = epoch_dir / "_auto_override.yaml"
    payload = {
        "logging": {
            "log_dir": epoch_dir.as_posix(),
            "postproc_debug": False,
        },
        "calibration": {
            "output_path": calibration_path.as_posix(),
        },
    }
    with override_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return override_path


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, Mapping) else {}


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    calibration_metrics = record.get("calibration_metrics", {})
    eval_metrics = record.get("eval_metrics", {})
    calibration_thresholds = record.get("calibration_thresholds", {})
    if not isinstance(calibration_metrics, Mapping):
        calibration_metrics = {}
    if not isinstance(eval_metrics, Mapping):
        eval_metrics = {}
    if not isinstance(calibration_thresholds, Mapping):
        calibration_thresholds = {}
    return {
        "epoch": record.get("epoch"),
        "checkpoint": record.get("checkpoint"),
        "calibration_path": record.get("calibration_path"),
        "calibration_onset_open": calibration_thresholds.get("onset"),
        "calibration_offset_open": calibration_thresholds.get("offset"),
        "calibration_ev_f1_mean": calibration_metrics.get("ev_f1_mean"),
        "eval_ev_f1_mean": eval_metrics.get("ev_f1_mean"),
        "eval_ev_f1_mean_anypitch": eval_metrics.get("ev_f1_mean_anypitch"),
        "eval_patk_onset_f1": eval_metrics.get("patk_onset_f1"),
        "eval_patk_note_f1": eval_metrics.get("patk_note_f1"),
        "eval_onset_event_f1": eval_metrics.get("onset_event_f1"),
        "eval_offset_event_f1": eval_metrics.get("offset_event_f1"),
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


def _run_one(args: argparse.Namespace, epoch: int, checkpoint: Path) -> Mapping[str, Any]:
    from tivit.pipelines.calibrate import calibrate
    from tivit.pipelines.eval_single import eval_single

    epoch_dir = Path(args.work_dir).expanduser() / f"epoch_{epoch:03d}"
    calibration_path = epoch_dir / "calibration.json"
    override_path = _write_temp_override(epoch_dir, calibration_path)

    base_configs = [Path(p) for p in args.base_config]
    calib_configs = base_configs + [Path(p) for p in args.calib_config] + [override_path]
    eval_configs = base_configs + [Path(p) for p in args.eval_config] + [override_path]

    if args.reuse_calibration and calibration_path.exists():
        calibration_payload = _read_json(calibration_path)
    else:
        calibration_payload = calibrate(
            configs=calib_configs,
            verbose=args.verbose,
            checkpoint=checkpoint,
            max_batches=args.max_batches,
            max_clips=args.max_clips,
            frames=args.frames,
            seed=args.seed,
            deterministic=args.deterministic,
            smoke=bool(args.smoke),
        )

    eval_metrics = eval_single(
        configs=eval_configs,
        verbose=args.verbose,
        split=args.split,
        checkpoint=checkpoint,
        max_batches=args.max_batches,
        max_clips=args.max_clips,
        frames=args.frames,
        seed=args.seed,
        deterministic=args.deterministic,
        smoke=bool(args.smoke),
    )

    calibration_metrics = calibration_payload.get("metrics", {}) if isinstance(calibration_payload, Mapping) else {}
    calibration_thresholds = calibration_payload.get("thresholds", {}) if isinstance(calibration_payload, Mapping) else {}
    return {
        "epoch": epoch,
        "checkpoint": checkpoint.as_posix(),
        "calibration_path": calibration_path.as_posix(),
        "calibration_metrics": calibration_metrics,
        "calibration_thresholds": calibration_thresholds,
        "eval_metrics": eval_metrics,
    }


def main() -> None:
    args = _parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    work_dir = Path(args.work_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _discover_checkpoints(checkpoint_dir, args.epochs)
    if not checkpoints:
        raise SystemExit(f"No matching checkpoints found in {checkpoint_dir}")

    records: list[Mapping[str, Any]] = []
    for epoch, checkpoint in checkpoints:
        print(f"[sweep] epoch={epoch} checkpoint={checkpoint}")
        record = _run_one(args, epoch, checkpoint)
        records.append(record)
        eval_metrics = record.get("eval_metrics", {})
        metric_val = _metric_value(eval_metrics if isinstance(eval_metrics, Mapping) else {}, args.metric)
        print(
            f"[result] epoch={epoch} {args.metric}={metric_val:.6f} "
            f"cal={record.get('calibration_path')}"
        )

    records.sort(
        key=lambda record: _metric_value(
            record.get("eval_metrics", {}) if isinstance(record.get("eval_metrics", {}), Mapping) else {},
            args.metric,
        ),
        reverse=True,
    )
    json_path, csv_path = _write_summary(records, work_dir)

    print("\nTop checkpoints:")
    for record in records[: max(int(args.top_k), 1)]:
        eval_metrics = record.get("eval_metrics", {})
        eval_metrics = eval_metrics if isinstance(eval_metrics, Mapping) else {}
        print(
            f"  epoch={record.get('epoch')} "
            f"{args.metric}={_metric_value(eval_metrics, args.metric):.6f} "
            f"patk_note_f1={_metric_value(eval_metrics, 'patk_note_f1'):.6f} "
            f"calibration={record.get('calibration_path')}"
        )

    print(f"\nWrote summary:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
