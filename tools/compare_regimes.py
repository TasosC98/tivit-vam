#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


METRICS_RE = re.compile(r"metrics=(\{.*\})")
CAL_SAVED_RE = re.compile(r"calibration saved to (.+)")
CAL_USED_RE = re.compile(r"using calibration overrides from (.+)")
TEMP_DEBUG_RE = re.compile(
    r"TEMP_DEBUG logits: onset\(mean=([0-9.eE+-]+) max=([0-9.eE+-]+)\) "
    r"offset\(mean=([0-9.eE+-]+) max=([0-9.eE+-]+)\)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare legacy vs current calibration/eval regimes on the same checkpoints."
    )
    p.add_argument(
        "--base-config",
        default="logs/pianovam_hdf5_randstart/resolved_config.yaml",
        help="Base resolved config used for both regimes.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="logs/pianovam_hdf5_randstart/checkpoints",
        help="Checkpoint directory to compare on. Keep this fixed for a fair comparison.",
    )
    p.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        required=True,
        help="Epoch numbers to evaluate, e.g. --epochs 10 11 12 13 14 15 16 17 18 19 20",
    )
    p.add_argument("--split", default="valid")
    p.add_argument("--verbose", default="info")
    p.add_argument(
        "--work-dir",
        default="logs/compare_regimes",
        help="Where to store temp overrides, raw logs, and summaries.",
    )

    # Legacy regime: the one that gave ~0.1
    p.add_argument(
        "--legacy-calib-config",
        action="append",
        default=["configs/overrides/calib_wide.yaml"],
        help="Extra config(s) for legacy calibration. Repeatable.",
    )
    p.add_argument(
        "--legacy-eval-config",
        action="append",
        default=[],
        help="Extra config(s) for legacy eval. Repeatable.",
    )

    # Current regime: the one that currently gives 0.0
    p.add_argument(
        "--current-calib-config",
        action="append",
        default=[
            "configs/overrides/pianovam_hdf5_density_control_v1.yaml",
            "configs/overrides/pianovam_hdf5_density_control_v1_calib.yaml",
        ],
        help="Extra config(s) for current calibration. Repeatable.",
    )
    p.add_argument(
        "--current-eval-config",
        action="append",
        default=["configs/overrides/pianovam_hdf5_density_control_v1.yaml"],
        help="Extra config(s) for current eval. Repeatable.",
    )

    return p.parse_args()


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: Path) -> str:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.stdout


def extract_last_metrics(text: str) -> Dict[str, Any]:
    matches = METRICS_RE.findall(text)
    if not matches:
        return {}
    try:
        return ast.literal_eval(matches[-1])
    except Exception:
        return {}


def extract_cal_saved_path(text: str) -> Optional[str]:
    m = CAL_SAVED_RE.search(text)
    return m.group(1).strip() if m else None


def extract_cal_used_path(text: str) -> Optional[str]:
    m = CAL_USED_RE.search(text)
    return m.group(1).strip() if m else None


def extract_temp_debug(text: str) -> Dict[str, Optional[float]]:
    m = TEMP_DEBUG_RE.search(text)
    if not m:
        return {
            "onset_prob_mean": None,
            "onset_prob_max": None,
            "offset_prob_mean": None,
            "offset_prob_max": None,
        }
    return {
        "onset_prob_mean": float(m.group(1)),
        "onset_prob_max": float(m.group(2)),
        "offset_prob_mean": float(m.group(3)),
        "offset_prob_max": float(m.group(4)),
    }


def read_decoder_from_json(path_str: Optional[str]) -> Dict[str, Any]:
    if not path_str:
        return {}
    p = Path(path_str)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    decoder = payload.get("decoder", {})
    return decoder if isinstance(decoder, dict) else {}


def make_temp_override(
    regime_name: str,
    epoch: int,
    work_dir: Path,
) -> Path:
    regime_dir = work_dir / regime_name / f"epoch_{epoch:03d}"
    regime_dir.mkdir(parents=True, exist_ok=True)
    override_path = regime_dir / "_tmp_override.yaml"
    yaml_text = f"""logging:
  log_dir: {regime_dir.as_posix()}
  postproc_debug: false

calibration:
  output_path: calibration/epoch_{epoch:03d}.json
"""
    override_path.write_text(yaml_text, encoding="utf-8")
    return override_path


def build_cmd(
    module_name: str,
    base_config: str,
    extra_configs: List[str],
    temp_override: Path,
    checkpoint: Path,
    split: str,
    verbose: str,
) -> List[str]:
    cmd = [sys.executable, "-m", module_name, "--config", base_config]
    for cfg in extra_configs:
        cmd += ["--config", cfg]
    cmd += ["--config", str(temp_override), "--checkpoint", str(checkpoint)]
    if module_name.endswith("eval_single"):
        cmd += ["--split", split]
    cmd += ["--verbose", verbose]
    return cmd


def flatten_result(record: Dict[str, Any]) -> Dict[str, Any]:
    metrics = record.pop("metrics", {}) or {}
    flat = dict(record)

    important_keys = [
        "loss",
        "onset_event_f1",
        "offset_event_f1",
        "ev_f1_mean",
        "onset_event_f1_anypitch",
        "offset_event_f1_anypitch",
        "ev_f1_mean_anypitch",
        "onset_pred_events_per_clip",
        "offset_pred_events_per_clip",
        "onset_event_density_ratio",
        "offset_event_density_ratio",
        "patk_onset_f1",
        "patk_note_f1",
    ]
    for key in important_keys:
        flat[key] = metrics.get(key)

    return flat


def compare_one_regime(
    regime_name: str,
    epoch: int,
    base_config: str,
    calib_configs: List[str],
    eval_configs: List[str],
    checkpoint: Path,
    split: str,
    verbose: str,
    work_dir: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    temp_override = make_temp_override(regime_name, epoch, work_dir)

    calib_cmd = build_cmd(
        "tivit.pipelines.calibrate",
        base_config,
        calib_configs,
        temp_override,
        checkpoint,
        split,
        verbose,
    )
    calib_log = work_dir / regime_name / f"epoch_{epoch:03d}" / "calibrate.log"
    calib_text = run_cmd(calib_cmd, env=env, log_path=calib_log)
    cal_saved_path = extract_cal_saved_path(calib_text)

    eval_cmd = build_cmd(
        "tivit.pipelines.eval_single",
        base_config,
        eval_configs,
        temp_override,
        checkpoint,
        split,
        verbose,
    )
    eval_log = work_dir / regime_name / f"epoch_{epoch:03d}" / "eval.log"
    eval_text = run_cmd(eval_cmd, env=env, log_path=eval_log)

    metrics = extract_last_metrics(eval_text)
    cal_used_path = extract_cal_used_path(eval_text)
    temp_debug = extract_temp_debug(eval_text)
    decoder_saved = read_decoder_from_json(cal_saved_path)
    decoder_used = read_decoder_from_json(cal_used_path)

    return {
        "regime": regime_name,
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "temp_override": str(temp_override),
        "calibration_saved_path": cal_saved_path,
        "calibration_used_by_eval": cal_used_path,
        "decoder_saved_json": json.dumps(decoder_saved, ensure_ascii=False),
        "decoder_used_json": json.dumps(decoder_used, ensure_ascii=False),
        "calibrate_log": str(calib_log),
        "eval_log": str(eval_log),
        **temp_debug,
        "metrics": metrics,
    }


def write_outputs(records: List[Dict[str, Any]], work_dir: Path) -> None:
    json_path = work_dir / "summary.json"
    csv_path = work_dir / "summary.csv"

    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    flat_rows = [flatten_result(dict(r)) for r in records]
    fieldnames: List[str] = []
    for row in flat_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"\nWrote:\n  {json_path}\n  {csv_path}\n")


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    env["DEBUG_PRED"] = "1"

    records: List[Dict[str, Any]] = []

    for epoch in args.epochs:
        ckpt = Path(args.checkpoint_dir) / f"epoch_{epoch}.pt"
        if not ckpt.exists():
            print(f"[WARN] missing checkpoint: {ckpt}")
            continue

        print(f"\n=== epoch {epoch} :: legacy ===")
        legacy = compare_one_regime(
            regime_name="legacy",
            epoch=epoch,
            base_config=args.base_config,
            calib_configs=args.legacy_calib_config,
            eval_configs=args.legacy_eval_config,
            checkpoint=ckpt,
            split=args.split,
            verbose=args.verbose,
            work_dir=work_dir,
            env=env,
        )
        records.append(legacy)

        print(
            f"legacy ev_f1_mean={legacy['metrics'].get('ev_f1_mean')} "
            f"anypitch={legacy['metrics'].get('ev_f1_mean_anypitch')} "
            f"onset_pred={legacy['metrics'].get('onset_pred_events_per_clip')} "
            f"cal_used={legacy.get('calibration_used_by_eval')}"
        )

        print(f"\n=== epoch {epoch} :: current ===")
        current = compare_one_regime(
            regime_name="current",
            epoch=epoch,
            base_config=args.base_config,
            calib_configs=args.current_calib_config,
            eval_configs=args.current_eval_config,
            checkpoint=ckpt,
            split=args.split,
            verbose=args.verbose,
            work_dir=work_dir,
            env=env,
        )
        records.append(current)

        print(
            f"current ev_f1_mean={current['metrics'].get('ev_f1_mean')} "
            f"anypitch={current['metrics'].get('ev_f1_mean_anypitch')} "
            f"onset_pred={current['metrics'].get('onset_pred_events_per_clip')} "
            f"cal_used={current.get('calibration_used_by_eval')}"
        )

    write_outputs(records, work_dir)


if __name__ == "__main__":
    main()