#!/usr/bin/env python3
"""Run evaluation and print a focused key-understanding report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate how well the model understands pressed keys and pitch")
    ap.add_argument("--config", action="append", default=None, help="One or more config fragments to merge")
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="quiet")
    ap.add_argument("--split", help="Dataset split to evaluate")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calibration-path", default=None, help="Optional calibration JSON override to reuse")
    ap.add_argument(
        "--pitch-threshold",
        type=float,
        default=None,
        help="Optional override for key_probe/patk pitch threshold",
    )
    ap.add_argument("--max-batches", dest="max_batches", type=int)
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _write_override(
    path: Path,
    *,
    log_dir: Path,
    calibration_path: str | None,
    pitch_threshold: float | None,
) -> None:
    payload: dict[str, object] = {
        "logging": {
            "log_dir": log_dir.as_posix(),
        }
    }
    if calibration_path:
        payload["calibration"] = {
            "output_path": str(Path(calibration_path).expanduser().as_posix()),
        }
    if pitch_threshold is not None:
        payload["training"] = {
            "metrics": {
                "key_probe": {
                    "threshold": float(pitch_threshold),
                },
                "patk": {
                    "threshold": float(pitch_threshold),
                },
            }
        }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def main() -> None:
    args = _parse_args()
    from tivit.pipelines.eval_single import eval_single

    configs = args.config or [Path("configs/default.yaml")]
    managed_tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.calibration_path or args.pitch_threshold is not None:
        managed_tmp = tempfile.TemporaryDirectory(prefix="key-understanding-report-")
        override_dir = Path(managed_tmp.name)
        override_path = override_dir / "_auto_report_override.yaml"
        _write_override(
            override_path,
            log_dir=override_dir,
            calibration_path=args.calibration_path,
            pitch_threshold=args.pitch_threshold,
        )
        configs = [*configs, override_path]

    try:
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
    finally:
        if managed_tmp is not None:
            managed_tmp.cleanup()

    print("\nKey Understanding Report")
    print(f"Pitch threshold: {metrics.get('pitch_probe_threshold', 'n/a')}")
    print(f"Pressed-key precision: {_pct(metrics.get('pitch_active_precision'))}")
    print(f"Pressed-key recall: {_pct(metrics.get('pitch_active_recall'))}")
    print(f"Pressed-key F1: {_pct(metrics.get('pitch_active_f1'))}")
    print(f"Not-pressed specificity: {_pct(metrics.get('pitch_inactive_specificity'))}")
    print(f"Overall key accuracy: {_pct(metrics.get('pitch_overall_accuracy'))}")
    print(f"Balanced accuracy: {_pct(metrics.get('pitch_balanced_accuracy'))}")
    print(f"Exact frame pitch-set match: {_pct(metrics.get('pitch_exact_frame_match'))}")
    print(f"Frame pitch-set Jaccard: {_pct(metrics.get('pitch_frame_jaccard'))}")
    print(f"Pred active keys/frame: {metrics.get('pitch_pred_active_per_frame', 'n/a')}")
    print(f"GT active keys/frame: {metrics.get('pitch_gt_active_per_frame', 'n/a')}")
    print(f"Polyphony MAE: {metrics.get('pitch_polyphony_mae', 'n/a')}")


if __name__ == "__main__":
    main()
