"""
PianoVAM dataset loader.

Purpose:
    - Resolve PianoVAM roots/manifests, parse labels/hand metadata, and crops.
    - Reuse shared decode/tiling/target logic from BasePianoDataset.
Key Functions/Classes:
    - PianoVAMDataset
CLI Arguments:
    - (none)
Usage:
    - ds = PianoVAMDataset(cfg, split="train", full_cfg=cfg)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .base import BasePianoDataset, DatasetEntry, safe_collate_fn
from ..targets.identifiers import canonical_video_id


LOGGER = logging.getLogger(__name__)


def _parse_point(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except (TypeError, ValueError):
                return None
    if isinstance(value, Sequence) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _crop_from_points(entry: Mapping[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    """Compute (min_y, max_y, min_x, max_x) from Point_LT/RT/RB/LB metadata when available."""
    labels = ["Point_LT", "Point_RT", "Point_RB", "Point_LB"]
    points: list[Tuple[float, float]] = []
    for key in labels:
        pt = _parse_point(entry.get(key))
        if pt is not None:
            points.append(pt)
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    try:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (int(round(min_y)), int(round(max_y)), int(round(min_x)), int(round(max_x)))
    except Exception:
        return None


def _load_metadata(root: Path) -> Dict[str, Dict[str, Any]]:
    """Load PianoVAM metadata_v2.json (if present) and index by record_time.

    IMPORTANT: metadata_v2.json keys are "0","1",... so we index by record_time.
    This allows lookup by video stem e.g. "2024-02-14_19-10-09".
    """
    meta_path = root / "metadata_v2.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    table: Dict[str, Dict[str, Any]] = {}
    for _k, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        rec = value.get("record_time")
        if not rec:
            continue
        canon = canonical_video_id(str(rec))
        table[canon] = dict(value)
    return table


class PianoVAMDataset(BasePianoDataset):
    """PianoVAM dataset using shared decoding/target logic."""

    def _resolve_root(self, root_dir: Optional[str]) -> Path:
        """Resolve dataset root with env fallbacks and ${VARS} expansion."""
        if root_dir:
            resolved = os.path.expandvars(os.path.expanduser(str(root_dir)))
            return Path(resolved)

        env = os.environ.get("PIANOVAM_ROOT") or os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
        if env:
            cand = Path(os.path.expandvars(os.path.expanduser(env)))
            # If env points directly to PianoVAM root:
            if (cand / "Video").exists():
                return cand
            # If env is a datasets parent:
            alt = cand / "PianoVAM_v1.0"
            if (alt / "Video").exists():
                return alt

        return Path("~/datasets/PianoVAM_v1.0").expanduser()

    def _resolve_manifest(self) -> Optional[Mapping[str, Any]]:
        """Load manifest JSON when provided.

        If no manifest is provided, return None and we will fallback to
        scanning root/Video/*.mp4 and use metadata_v2.json for split.
        """
        manifest_cfg = (self.dataset_cfg.get("manifest", {}) or {}) if isinstance(self.dataset_cfg, Mapping) else {}
        path = manifest_cfg.get(self.split)
        if not path:
            return None

        resolved = os.path.expandvars(os.path.expanduser(str(path)))
        p = Path(resolved)
        if not p.exists():
            return None

        try:
            with p.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None

        return data if isinstance(data, Mapping) else None

    def _list_entries(self, root: Path, split: str, manifest: Optional[Mapping[str, Any]]) -> List[DatasetEntry]:
        """List video/label entries for PianoVAM split."""
        entries: List[DatasetEntry] = []

        preproc = (self.dataset_cfg or {}).get("preprocessed_format")
        hdf5_root_cfg = (self.dataset_cfg or {}).get("hdf5_root")

        meta_table = _load_metadata(root)

        # If manifest exists and has split list, use it. Otherwise fallback to scanning Video/.
        meta = manifest or {}
        split_meta = meta.get(split, []) if isinstance(meta, Mapping) else []

        if isinstance(split_meta, list) and len(split_meta) > 0:
            # Manifest-based entries (optional; keep working if you later create a manifest)
            for item in split_meta:
                if not isinstance(item, Mapping):
                    continue
                vid = canonical_video_id(item.get("video_id", ""))
                if not vid:
                    continue

                video_rel = item.get("video")
                label_rel = item.get("label")
                video_path = root / video_rel if video_rel else None
                label_path = root / label_rel if label_rel else None

                # Prefer HDF5 if configured
                hdf5_path = None
                if video_path is not None and preproc == "hdf5":
                    try:
                        if hdf5_root_cfg and video_rel:
                            candidate = Path(os.path.expanduser(str(hdf5_root_cfg))) / Path(video_rel).with_suffix(".h5")
                        else:
                            candidate = video_path.with_suffix(".h5")
                        if candidate.exists():
                            hdf5_path = candidate
                    except Exception:
                        pass


                if video_path is None or not video_path.exists():
                    continue

                metadata = dict(item)
                meta_entry = meta_table.get(vid, {})
                crop = metadata.get("crop") or _crop_from_points(metadata) or _crop_from_points(meta_entry)
                if crop is not None:
                    metadata["crop"] = crop

                entries.append(
                    DatasetEntry(
                        video_path=video_path,                 # mp4
                        hdf5_path=hdf5_path,                   # h5 (optional)
                        label_path=label_path if (label_path and label_path.exists()) else None,
                        video_id=vid,
                        metadata=metadata,
                    )
                )


            return entries

        # Fallback: scan root/Video/*.mp4 and use metadata_v2.json for split filtering
        video_dir = root / "Video"
        if not video_dir.exists():
            LOGGER.warning("PianoVAM root has no Video/ folder: %s", root)
            return entries

        for video_path in sorted(video_dir.rglob("*.mp4")):
            vid = canonical_video_id(video_path.stem)

            meta_entry = meta_table.get(vid, {})
            # Split filtering: metadata_v2.json has split field
            if isinstance(meta_entry, Mapping):
                meta_split = meta_entry.get("split")
                if isinstance(meta_split, str) and meta_split and meta_split != split:
                    continue

            label_path = (root / "TSV" / f"{video_path.stem}.tsv")
            metadata = dict(meta_entry) if isinstance(meta_entry, Mapping) else {}

            crop = _crop_from_points(metadata)
            if crop is not None:
                metadata["crop"] = crop

            # Prefer HDF5 if configured: look in hdf5_root_cfg/<stem>.h5
            hdf5_path = None
            if preproc == "hdf5":
                try:
                    if hdf5_root_cfg:
                        candidate = Path(os.path.expanduser(str(hdf5_root_cfg))) / f"{video_path.stem}.h5"
                    else:
                        candidate = video_path.with_suffix(".h5")
                    if candidate.exists():
                        hdf5_path = candidate
                except Exception:
                    pass

            entries.append(
                DatasetEntry(
                    video_path=video_path,  # mp4
                    hdf5_path=hdf5_path,    # h5 (optional)
                    label_path=label_path if label_path.exists() else None,  # TSV (important!)
                    video_id=vid,
                    metadata=metadata,
                )
            )

        return entries

    def _read_labels(self, entry: DatasetEntry) -> Mapping[str, Any]:
        """Parse events and optional hand/clef metadata.

        Supports:
          - embedded labels in HDF5 (labels_ts or label_raw)
          - TSV labels in root/TSV/<stem>.tsv
          - MIDI labels if label_path points to .mid/.midi (optional)
        """
        # has_hdf5_video = entry.video_path is not None and entry.video_path.suffix.lower() in {".h5", ".hdf5"}
        # has_label_file = entry.label_path is not None and entry.label_path.exists()
        has_hdf5_video = entry.hdf5_path is not None and entry.hdf5_path.suffix.lower() in {".h5", ".hdf5"}
        has_label_file = entry.label_path is not None and entry.label_path.exists()


        if not has_hdf5_video and not has_label_file:
            if getattr(self, "require_labels", False):
                raise FileNotFoundError(f"Missing label for {entry.video_id}")
            return {}

        events: list[Any] = []
        hand_meta: dict[str, Any] = {}

        # 1) Try labels from HDF5 video file (if present)
        if has_hdf5_video:
            try:
                import h5py

                with h5py.File(entry.hdf5_path, "r") as hf:
                    if "labels_ts" in hf:
                        arr = hf["labels_ts"][:]
                        for row in arr:
                            try:
                                events.append((float(row[0]), float(row[1]), int(row[2])))
                            except Exception:
                                continue
                    elif "label_raw" in hf:
                        raw = hf["label_raw"][()]
                        if isinstance(raw, (bytes, bytearray)):
                            raw_text = raw.decode("utf-8", errors="ignore")
                        else:
                            raw_text = str(raw)
                        for line in raw_text.splitlines():
                            parts = line.strip().split()
                            if len(parts) < 3:
                                continue
                            try:
                                onset, offset, pitch = float(parts[0]), float(parts[1]), int(parts[2])
                            except Exception:
                                continue
                            events.append((onset, offset, pitch))
            except Exception:
                # silently fallback to external label file
                pass

        # 2) External label file (TSV/MIDI/etc.)
        if has_label_file:
            suffix = entry.label_path.suffix.lower()

            if suffix in {".mid", ".midi"}:
                try:
                    import pretty_midi  # type: ignore

                    pm = pretty_midi.PrettyMIDI(str(entry.label_path))
                    for inst in pm.instruments:
                        for note in inst.notes:
                            events.append((float(note.start), float(note.end), int(note.pitch)))
                except Exception as exc:
                    if getattr(self, "require_labels", False):
                        raise
                    LOGGER.warning("Failed to parse MIDI for %s (%s)", entry.video_id, exc)

            elif suffix == ".tsv":
                try:
                    with entry.label_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.startswith("#") or not line.strip():
                                continue
                            parts = line.strip().split("\t")
                            if len(parts) < 3:
                                parts = line.strip().split()
                            if len(parts) < 3:
                                continue
                            try:
                                onset = float(parts[0])
                                offset = float(parts[1])
                                # PianoVAM TSVs may be 5-column (pitch at index 3)
                                pitch_raw = parts[3] if len(parts) >= 5 else parts[2]
                                pitch = int(round(float(pitch_raw)))
                            except (TypeError, ValueError):
                                continue
                            events.append((onset, offset, pitch))
                except Exception as exc:
                    if getattr(self, "require_labels", False):
                        raise
                    LOGGER.warning("Failed to parse TSV for %s (%s)", entry.video_id, exc)

            else:
                # generic whitespace format: onset offset pitch [hand] [clef]
                try:
                    with entry.label_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            parts = line.strip().split()
                            if len(parts) < 3:
                                continue
                            onset, offset, pitch = float(parts[0]), float(parts[1]), int(parts[2])
                            hand_val = int(parts[3]) if len(parts) >= 4 else None
                            clef_val = int(parts[4]) if len(parts) >= 5 else None
                            events.append((onset, offset, pitch, hand_val, clef_val))
                            if hand_val is not None:
                                hand_meta.setdefault("hand_seq", []).append(hand_val)
                            if clef_val is not None:
                                hand_meta.setdefault("clef_seq", []).append(clef_val)
                except Exception as exc:
                    if getattr(self, "require_labels", False):
                        raise
                    LOGGER.warning("Failed to parse labels for %s (%s)", entry.video_id, exc)

        # metadata hints
        if isinstance(entry.metadata, Mapping):
            if "hand" in entry.metadata:
                hand_meta["hand_hint"] = entry.metadata.get("hand")
            if "clef" in entry.metadata:
                hand_meta["clef_hint"] = entry.metadata.get("clef")

        payload: dict[str, Any] = {"events": events, "metadata": dict(entry.metadata)}
        payload.update(hand_meta)
        if os.environ.get("DEBUG_LABELS") == "1":
            print(f"[DEBUG_LABELS] {entry.video_id}: events={len(events)} label_file={entry.label_path}")
        return payload


def make_dataloader(cfg: Mapping[str, Any], split: str, drop_last: bool = False, *, seed: Optional[int] = None):
    dcfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    batch_size = int(dcfg.get("batch_size", 1))
    shuffle = bool(dcfg.get("shuffle", True)) if split == "train" else False
    num_workers = int(dcfg.get("num_workers", 0))
    prefetch_factor = dcfg.get("prefetch_factor")

    ds = PianoVAMDataset(cfg, split, full_cfg=cfg)
    return torch.utils.data.DataLoader(  # type: ignore[attr-defined]
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        collate_fn=safe_collate_fn,
    )
