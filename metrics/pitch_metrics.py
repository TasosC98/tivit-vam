"""Pitch-level metrics."""

from __future__ import annotations

import numpy as np


def frame_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape or pred.size == 0:
        return 0.0
    matches = (pred == target).sum()
    return float(matches) / float(pred.size)

def key_state_counts(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Count pressed/unpressed key outcomes and frame-level set agreement.

    ``pred`` and ``target`` are expected to be boolean or binary arrays with
    shape ``(..., P)`` where the last dimension is the pitch/key axis.
    """
    pred_arr = np.asarray(pred).astype(bool, copy=False)
    target_arr = np.asarray(target).astype(bool, copy=False)
    if pred_arr.shape != target_arr.shape:
        raise ValueError("Prediction and target masks must share shape for key-state metrics")
    if pred_arr.size == 0:
        return {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
            "frames": 0.0,
            "frames_exact": 0.0,
            "jaccard_sum": 0.0,
            "polyphony_abs_err_sum": 0.0,
            "pred_active_sum": 0.0,
            "target_active_sum": 0.0,
        }

    tp = float(np.logical_and(pred_arr, target_arr).sum())
    fp = float(np.logical_and(pred_arr, np.logical_not(target_arr)).sum())
    fn = float(np.logical_and(np.logical_not(pred_arr), target_arr).sum())
    tn = float(np.logical_and(np.logical_not(pred_arr), np.logical_not(target_arr)).sum())

    flat_pred = pred_arr.reshape(-1, pred_arr.shape[-1])
    flat_target = target_arr.reshape(-1, target_arr.shape[-1])
    frames = float(flat_pred.shape[0])
    frames_exact = float(np.all(flat_pred == flat_target, axis=1).sum())

    intersection = np.logical_and(flat_pred, flat_target).sum(axis=1).astype(np.float64, copy=False)
    union = np.logical_or(flat_pred, flat_target).sum(axis=1).astype(np.float64, copy=False)
    jaccard = np.ones_like(union, dtype=np.float64)
    non_empty = union > 0.0
    jaccard[non_empty] = intersection[non_empty] / union[non_empty]

    pred_active = flat_pred.sum(axis=1).astype(np.float64, copy=False)
    target_active = flat_target.sum(axis=1).astype(np.float64, copy=False)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "frames": frames,
        "frames_exact": frames_exact,
        "jaccard_sum": float(jaccard.sum()),
        "polyphony_abs_err_sum": float(np.abs(pred_active - target_active).sum()),
        "pred_active_sum": float(pred_active.sum()),
        "target_active_sum": float(target_active.sum()),
    }


def summarize_key_state_counts(counts: dict[str, float], *, eps: float = 1e-8) -> dict[str, float]:
    """Convert accumulated key-state counts into user-facing metrics."""
    tp = float(counts.get("tp", 0.0) or 0.0)
    fp = float(counts.get("fp", 0.0) or 0.0)
    fn = float(counts.get("fn", 0.0) or 0.0)
    tn = float(counts.get("tn", 0.0) or 0.0)
    frames = float(counts.get("frames", 0.0) or 0.0)

    precision = tp / (tp + fp + eps) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn + eps) if (tp + fn) > 0.0 else 0.0
    specificity = tn / (tn + fp + eps) if (tn + fp) > 0.0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps) if (tp + tn + fp + fn) > 0.0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall + eps)

    frames_exact = float(counts.get("frames_exact", 0.0) or 0.0)
    jaccard_sum = float(counts.get("jaccard_sum", 0.0) or 0.0)
    polyphony_abs_err_sum = float(counts.get("polyphony_abs_err_sum", 0.0) or 0.0)
    pred_active_sum = float(counts.get("pred_active_sum", 0.0) or 0.0)
    target_active_sum = float(counts.get("target_active_sum", 0.0) or 0.0)

    frame_exact_match = frames_exact / (frames + eps) if frames > 0.0 else 0.0
    frame_jaccard = jaccard_sum / (frames + eps) if frames > 0.0 else 0.0
    polyphony_mae = polyphony_abs_err_sum / (frames + eps) if frames > 0.0 else 0.0
    pred_active_per_frame = pred_active_sum / (frames + eps) if frames > 0.0 else 0.0
    target_active_per_frame = target_active_sum / (frames + eps) if frames > 0.0 else 0.0

    return {
        "pitch_active_precision": float(precision),
        "pitch_active_recall": float(recall),
        "pitch_active_f1": float(f1),
        "pitch_inactive_specificity": float(specificity),
        "pitch_overall_accuracy": float(accuracy),
        "pitch_balanced_accuracy": float(balanced_accuracy),
        "pitch_exact_frame_match": float(frame_exact_match),
        "pitch_frame_jaccard": float(frame_jaccard),
        "pitch_polyphony_mae": float(polyphony_mae),
        "pitch_pred_active_per_frame": float(pred_active_per_frame),
        "pitch_gt_active_per_frame": float(target_active_per_frame),
    }


__all__ = ["frame_accuracy", "key_state_counts", "summarize_key_state_counts"]
