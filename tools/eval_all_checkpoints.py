#!/usr/bin/env python3
"""Evaluate every saved checkpoint and emit a CSV showing per-epoch F1 scores.

We currently calibrate + evaluate only the LAST epoch. But training validation
loss flattened by ~epoch 75 in the previous runs, and there's no guarantee
that epoch 200 is the best-generalizing one. This tool sweeps every
`epoch_*.pt` checkpoint in a directory, evaluates each on the chosen split,
and writes:

    <out-dir>/per_epoch_f1.csv     (one row per checkpoint, sorted by epoch)
    <out-dir>/per_epoch_summary.txt (best epoch by each metric)

Usage:
    PYTHONPATH=. python tools/eval_all_checkpoints.py \\
        --config configs/default.yaml \\
        --config configs/overrides/manual_training.yaml \\
        --config configs/overrides/manual_training_keyroi_train.yaml \\
        --config configs/overrides/06-05-2026/manual_training_keyroi_video_06_05_2026.yaml \\
        --config configs/overrides/06-05-2026/phase5_fix_pitch_head.yaml \\
        --config configs/overrides/06-05-2026/phase5_eval_only_decoder_fix.yaml \\
        --checkpoint-dir /home/achatzigiannis/tivit-logs/phase5_pitch_fix_06_05_2026/checkpoints \\
        --split valid \\
        --out-dir /home/achatzigiannis/tivit-logs/phase7_debug_13_05_2026/per_epoch_eval

After it finishes, the CSV columns are:
    epoch, pitch_active_f1, patk_note_f1, patk_onset_f1, onset_event_f1,
    onset_event_f1_anypitch, pitch_balanced_accuracy, pitch_loss, onset_loss
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Mapping

os.environ.setdefault("TIVIT_GEOMETRY_GATE_DISABLED", "0")  # use the gate normally for eval

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_TRACKED_METRICS = (
    "pitch_active_f1",
    "pitch_active_precision",
    "pitch_active_recall",
    "pitch_balanced_accuracy",
    "patk_note_f1",
    "patk_onset_f1",
    "patk_frame_f1",
    "onset_event_f1",
    "offset_event_f1",
    "onset_event_f1_anypitch",
    "offset_event_f1_anypitch",
    "pitch",
    "onset",
    "offset",
    "loss",
)


def _epoch_of(p: Path) -> int:
    m = re.search(r"epoch_(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from tivit.pipelines.evaluate import evaluate

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not ckpt_dir.is_dir():
        raise SystemExit(f"checkpoint dir not found: {ckpt_dir}")
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=_epoch_of)
    if not ckpts:
        raise SystemExit(f"no epoch_*.pt files in {ckpt_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_epoch_f1.csv"
    print(f"Found {len(ckpts)} checkpoints. Evaluating each on split={args.split} ...")

    rows: List[dict] = []
    for ck in ckpts:
        epoch = _epoch_of(ck)
        print(f"\n  >>> epoch {epoch} ({ck.name}) ...")
        try:
            metrics = evaluate(
                configs=args.config,
                verbose="quiet",
                split=args.split,
                checkpoint=str(ck),
            )
        except Exception as exc:
            print(f"  epoch {epoch}: FAILED ({type(exc).__name__}: {exc})")
            continue
        row = {"epoch": int(epoch)}
        for k in _TRACKED_METRICS:
            row[k] = float(metrics.get(k, 0.0)) if metrics.get(k) is not None else None
        rows.append(row)
        print(
            f"  epoch={epoch}  pitch_active_f1={row.get('pitch_active_f1')}  "
            f"patk_note_f1={row.get('patk_note_f1')}  "
            f"onset_event_f1={row.get('onset_event_f1')}  "
            f"onset_anypitch={row.get('onset_event_f1_anypitch')}"
        )

    if not rows:
        raise SystemExit("no checkpoints evaluated successfully")

    rows.sort(key=lambda r: r["epoch"])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fields = ["epoch"] + list(_TRACKED_METRICS)
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary: best epoch per metric
    summary_lines = [f"Evaluated {len(rows)} checkpoints on split={args.split}"]
    for key_metric in (
        "pitch_active_f1",
        "patk_note_f1",
        "patk_onset_f1",
        "onset_event_f1",
        "onset_event_f1_anypitch",
        "pitch_balanced_accuracy",
    ):
        valid = [(r["epoch"], r.get(key_metric)) for r in rows if r.get(key_metric) is not None]
        if not valid:
            continue
        best_epoch, best_val = max(valid, key=lambda x: x[1])
        summary_lines.append(f"  best {key_metric:<26} = {best_val:.4f}  at epoch={best_epoch}")

    summary = "\n".join(summary_lines)
    print()
    print(summary)
    (out_dir / "per_epoch_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print()
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
