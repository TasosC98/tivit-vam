"""Note-level evaluation metrics for piano transcription.

The existing `metrics/event_f1.py` operates on binary mask sequences (treats
each frame as an independent event). For real piano transcription we want
NOTE-LEVEL metrics: a predicted note matches a reference note when its onset
time falls within tolerance AND its pitch matches.

This module provides:

    note_f1(predicted_notes, reference_notes, *, onset_tolerance_s=0.05)
        -> NoteEvalResult

    where each note is a (onset_s, offset_s, pitch_midi) tuple. Returns
    precision, recall, F1 plus per-key-class breakdowns (white vs black,
    keyboard regions) and adjacent-pitch confusion counts.

These metrics surface the diagnostic signals that distinguish geometry
errors from sync errors from model errors:

    - Many pitch±1 errors  -> geometry shift of one key
    - Many time-shifted errors but pitch correct -> sync drift
    - Random errors        -> model is the bottleneck

Usage:
    from tivit.metrics.note_level import note_f1, NoteEvalResult, summarize

    result = note_f1(pred_notes, ref_notes, onset_tolerance_s=0.05)
    print(summarize(result))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np


_WHITE_PITCHES = {0, 2, 4, 5, 7, 9, 11}


def _is_white(pitch: int) -> bool:
    return (int(pitch) % 12) in _WHITE_PITCHES


def _pitch_region(pitch: int) -> str:
    """Coarse keyboard regions: low (A0..B2), mid (C3..B5), high (C6..C8)."""
    p = int(pitch)
    if p < 48:
        return "low"
    if p < 84:
        return "mid"
    return "high"


@dataclass
class NoteEvalResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    onset_tolerance_s: float = 0.05
    n_pred: int = 0
    n_ref: int = 0

    # Per-class TP/FP/FN
    by_color: Mapping[str, Tuple[int, int, int]] = field(default_factory=dict)  # "white"/"black"
    by_region: Mapping[str, Tuple[int, int, int]] = field(default_factory=dict)

    # Confusion (counted only for unmatched predictions vs unmatched refs)
    pitch_off_by_one: int = 0
    pitch_off_by_two: int = 0

    # Onset timing error on matched notes (signed: pred - ref, milliseconds)
    onset_errors_ms: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1e-9)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1e-9)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)

    def asdict(self) -> dict:
        d = {
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "onset_tolerance_s": float(self.onset_tolerance_s),
            "n_pred": int(self.n_pred),
            "n_ref": int(self.n_ref),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "pitch_off_by_one": int(self.pitch_off_by_one),
            "pitch_off_by_two": int(self.pitch_off_by_two),
            "onset_error_ms_median": float(np.median(self.onset_errors_ms)) if self.onset_errors_ms else None,
            "onset_error_ms_mean": float(np.mean(self.onset_errors_ms)) if self.onset_errors_ms else None,
            "onset_error_ms_std": float(np.std(self.onset_errors_ms)) if self.onset_errors_ms else None,
        }
        for color in ("white", "black"):
            tp, fp, fn = self.by_color.get(color, (0, 0, 0))
            f1 = (2 * tp / max(2 * tp + fp + fn, 1e-9))
            d[f"f1_{color}"] = float(f1)
            d[f"tp_{color}"] = int(tp)
            d[f"fp_{color}"] = int(fp)
            d[f"fn_{color}"] = int(fn)
        for region in ("low", "mid", "high"):
            tp, fp, fn = self.by_region.get(region, (0, 0, 0))
            f1 = (2 * tp / max(2 * tp + fp + fn, 1e-9))
            d[f"f1_{region}"] = float(f1)
        return d


def note_f1(
    predicted: Sequence[Tuple[float, float, int]],
    reference: Sequence[Tuple[float, float, int]],
    *,
    onset_tolerance_s: float = 0.05,
    require_offset: bool = False,
    offset_tolerance_s: float = 0.10,
    offset_min_overlap: float = 0.20,
) -> NoteEvalResult:
    """Match predicted notes to reference notes by greedy nearest-onset
    within `onset_tolerance_s` AND identical pitch.

    A note is a (onset_s, offset_s, pitch_midi) tuple. If `require_offset` is
    True, also require that the predicted offset is within `offset_tolerance_s`
    of the reference, OR that pred and ref intervals overlap by at least
    `offset_min_overlap` of the reference duration.

    Returns: NoteEvalResult with TP/FP/FN, per-class breakdowns, pitch
    confusion (off-by-1, off-by-2 in MIDI), and onset timing error stats.
    """

    pred = sorted(predicted, key=lambda n: float(n[0]))
    ref = sorted(reference, key=lambda n: float(n[0]))
    n_pred, n_ref = len(pred), len(ref)
    if n_pred == 0 and n_ref == 0:
        return NoteEvalResult(
            tp=0, fp=0, fn=0, onset_tolerance_s=onset_tolerance_s,
            n_pred=0, n_ref=0,
            by_color={"white": (0, 0, 0), "black": (0, 0, 0)},
            by_region={"low": (0, 0, 0), "mid": (0, 0, 0), "high": (0, 0, 0)},
        )

    # Group by pitch for fast lookup
    ref_by_pitch: dict[int, List[Tuple[int, float, float]]] = {}
    for j, (on, off, p) in enumerate(ref):
        ref_by_pitch.setdefault(int(p), []).append((j, float(on), float(off)))
    used_ref = [False] * n_ref
    matched_pred = [False] * n_pred

    tp = 0
    onset_errs: List[float] = []
    by_color = {"white": [0, 0, 0], "black": [0, 0, 0]}
    by_region = {"low": [0, 0, 0], "mid": [0, 0, 0], "high": [0, 0, 0]}

    def _bump(arr: dict, key: str, slot: int) -> None:
        triple = arr.get(key)
        if triple is None:
            arr[key] = [0, 0, 0]
            triple = arr[key]
        triple[slot] += 1

    # First pass: match predictions to refs at same pitch within onset tolerance
    for i, (p_on, p_off, p_pitch) in enumerate(pred):
        candidates = ref_by_pitch.get(int(p_pitch), [])
        best_j = -1
        best_dt = float("inf")
        for j, r_on, r_off in candidates:
            if used_ref[j]:
                continue
            dt = abs(float(p_on) - float(r_on))
            if dt > onset_tolerance_s:
                continue
            if require_offset:
                # Either predicted offset is within tolerance,
                # or pred/ref intervals overlap by min ratio of ref duration.
                ref_dur = max(float(r_off) - float(r_on), 1e-6)
                overlap = max(0.0, min(float(p_off), float(r_off)) - max(float(p_on), float(r_on)))
                if abs(float(p_off) - float(r_off)) > offset_tolerance_s and (overlap / ref_dur) < offset_min_overlap:
                    continue
            if dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j >= 0:
            used_ref[best_j] = True
            matched_pred[i] = True
            tp += 1
            r_on = ref_by_pitch[int(p_pitch)][[idx for idx, _, _ in ref_by_pitch[int(p_pitch)]].index(best_j)][1]
            onset_errs.append(1000.0 * (float(p_on) - float(r_on)))
            color = "white" if _is_white(int(p_pitch)) else "black"
            region = _pitch_region(int(p_pitch))
            _bump(by_color, color, 0)
            _bump(by_region, region, 0)

    # Unmatched predictions are FPs; classify them and look for ±1/±2 confusions
    pitch_off_by_one = 0
    pitch_off_by_two = 0
    for i, (p_on, p_off, p_pitch) in enumerate(pred):
        if matched_pred[i]:
            continue
        color = "white" if _is_white(int(p_pitch)) else "black"
        region = _pitch_region(int(p_pitch))
        _bump(by_color, color, 1)
        _bump(by_region, region, 1)
        # Look for an unmatched ref at pitch ± 1 or ± 2 within onset tolerance.
        for delta in (-2, -1, 1, 2):
            cand = ref_by_pitch.get(int(p_pitch) + delta, [])
            for j, r_on, r_off in cand:
                if used_ref[j]:
                    continue
                if abs(float(p_on) - float(r_on)) <= onset_tolerance_s:
                    if abs(delta) == 1:
                        pitch_off_by_one += 1
                    else:
                        pitch_off_by_two += 1
                    break
            else:
                continue
            break

    # Unmatched references are FNs
    for j, (r_on, r_off, r_pitch) in enumerate(ref):
        if used_ref[j]:
            continue
        color = "white" if _is_white(int(r_pitch)) else "black"
        region = _pitch_region(int(r_pitch))
        _bump(by_color, color, 2)
        _bump(by_region, region, 2)

    fp = sum(1 for m in matched_pred if not m)
    fn = sum(1 for u in used_ref if not u)
    return NoteEvalResult(
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        onset_tolerance_s=float(onset_tolerance_s),
        n_pred=int(n_pred),
        n_ref=int(n_ref),
        by_color={k: tuple(v) for k, v in by_color.items()},
        by_region={k: tuple(v) for k, v in by_region.items()},
        pitch_off_by_one=int(pitch_off_by_one),
        pitch_off_by_two=int(pitch_off_by_two),
        onset_errors_ms=onset_errs,
    )


def summarize(result: NoteEvalResult) -> str:
    """Render a NoteEvalResult to a human-readable multi-line string."""

    d = result.asdict()
    lines = []
    lines.append(
        f"Note F1 (onset tol = {d['onset_tolerance_s']*1000:.0f} ms): "
        f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
        f"(TP={d['tp']} FP={d['fp']} FN={d['fn']} | n_pred={d['n_pred']} n_ref={d['n_ref']})"
    )
    lines.append(
        f"  By color : white F1={d['f1_white']:.3f} (TP={d['tp_white']} FP={d['fp_white']} FN={d['fn_white']}) "
        f" |  black F1={d['f1_black']:.3f} (TP={d['tp_black']} FP={d['fp_black']} FN={d['fn_black']})"
    )
    lines.append(
        f"  By region: low F1={d['f1_low']:.3f}  mid F1={d['f1_mid']:.3f}  high F1={d['f1_high']:.3f}"
    )
    if d.get("onset_error_ms_median") is not None:
        lines.append(
            f"  Onset timing error on matched notes (ms): "
            f"median={d['onset_error_ms_median']:+.1f}  mean={d['onset_error_ms_mean']:+.1f}  "
            f"std={d['onset_error_ms_std']:.1f}"
        )
    fp = max(d["fp"], 1)
    lines.append(
        f"  Confusion: FP at pitch ±1 = {d['pitch_off_by_one']} "
        f"({100*d['pitch_off_by_one']/fp:.0f}% of FP); "
        f"FP at pitch ±2 = {d['pitch_off_by_two']} ({100*d['pitch_off_by_two']/fp:.0f}% of FP)"
    )
    # Diagnostic interpretation
    if d["fp"] + d["fn"] > 0:
        diag_threshold = 0.30
        ratio_pm1 = d["pitch_off_by_one"] / max(d["fp"], 1)
        if ratio_pm1 > diag_threshold:
            lines.append(
                f"  DIAGNOSIS: {ratio_pm1*100:.0f}% of FPs are at pitch ±1 — likely GEOMETRY drift"
            )
        elif d.get("onset_error_ms_std", 0) and abs(d.get("onset_error_ms_median") or 0) > 30.0:
            lines.append(
                f"  DIAGNOSIS: median onset error ±{d['onset_error_ms_median']:.0f} ms — likely SYNC drift"
            )
    return "\n".join(lines)


__all__ = ["NoteEvalResult", "note_f1", "summarize"]
