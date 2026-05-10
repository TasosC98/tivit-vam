"""Lookup helper for `key_geometry/{split}/{video_id}.json` files produced by
`tools/recalibrate_all.py`.

The dataset uses this to enforce the calibration gate without re-running the
calibrator at training time. The geometry JSON is the single source of truth
for per-video keyboard layout; the dataset, the audit, and the manual-fix UI
all read from the same place.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

LOGGER = logging.getLogger(__name__)


# 2026-05-08: visual inspection of overlays shows accepted_loose calibrations
# (median < 6 px, ~20% of white-key width) are visibly correct on the
# keyboard image — the polygons sit on the right keys. Excluding them from
# training threw away ~40 of 71 videos for a residual difference that is
# below the noise floor of the underlying detection. Including loose for
# train. If F1 turns out to be poor, we'll re-tighten this threshold and
# pivot to the manual-clicker tool, but only with that evidence in hand.
_ALLOWED_STATUSES_FOR_TRAIN = {"ok", "manual_fixed", "accepted_loose"}
_ALLOWED_STATUSES_FOR_EVAL = {"ok", "manual_fixed", "accepted_loose"}


@dataclass(frozen=True)
class GeometryEntry:
    video_id: str
    split: str
    payload: Mapping[str, Any]
    json_path: Path

    @property
    def calibration_status(self) -> str:
        return str(self.payload.get("calibration_status") or "missing")

    @property
    def residual_median_px(self) -> Optional[float]:
        v = self.payload.get("residual_median_px")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def residual_p95_px(self) -> Optional[float]:
        v = self.payload.get("residual_p95_px")
        return float(v) if isinstance(v, (int, float)) else None

    def is_usable_for(self, split: str) -> bool:
        s = self.calibration_status
        if str(split).lower() == "train":
            return s in _ALLOWED_STATUSES_FOR_TRAIN
        return s in _ALLOWED_STATUSES_FOR_EVAL


class GeometryIndex:
    """Lazy index over key_geometry/{split}/{video_id}.json files."""

    def __init__(self, root: Optional[Path]):
        self._root = Path(root).expanduser().resolve() if root else None
        self._cache: Dict[tuple[str, str], Optional[GeometryEntry]] = {}

    @property
    def root(self) -> Optional[Path]:
        return self._root

    def is_active(self) -> bool:
        return self._root is not None and self._root.exists()

    def get(self, split: str, video_id: str) -> Optional[GeometryEntry]:
        if self._root is None:
            return None
        key = (str(split), str(video_id))
        if key in self._cache:
            return self._cache[key]
        path = self._root / split / f"{video_id}.json"
        if not path.exists():
            self._cache[key] = None
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            LOGGER.warning("geometry: failed to read %s (%s)", path, exc)
            self._cache[key] = None
            return None
        entry = GeometryEntry(video_id=video_id, split=str(split), payload=payload, json_path=path)
        self._cache[key] = entry
        return entry

    def inject_sync_into_entries(
        self,
        entries: Iterable[Any],
        *,
        split: str,
    ) -> int:
        """For each entry whose geometry JSON has a `sync.video_time_offset_ms`,
        write `lag_ms` into the entry's metadata so the existing
        `resolve_sync` flow picks it up at sample-time. Returns the number of
        entries that received an injected lag.

        This is the bridge between `tools/sync_sweep_per_video.py` (which
        writes per-video time offsets into the geometry JSON) and the
        dataset's per-sample sync application. If we don't inject this, the
        dataset reads lag_ms only from metadata_v2.json (which doesn't have
        it for PianoVAM) and silently uses 0 ms — labels stay misaligned.
        """

        if not self.is_active():
            return 0
        injected = 0
        for entry in entries:
            video_id = getattr(entry, "video_id", None)
            if video_id is None:
                continue
            geom = self.get(split, str(video_id))
            if geom is None:
                continue
            sync = geom.payload.get("sync") if isinstance(geom.payload, Mapping) else None
            if not isinstance(sync, Mapping):
                continue
            offset_ms = sync.get("video_time_offset_ms")
            if offset_ms is None:
                continue
            try:
                offset_ms_int = int(round(float(offset_ms)))
            except (TypeError, ValueError):
                continue
            md = getattr(entry, "metadata", None)
            if isinstance(md, dict):
                md["lag_ms"] = offset_ms_int
                injected += 1
            elif isinstance(md, Mapping):
                # Replace with a mutable dict so resolve_sync can read it.
                new_md = dict(md)
                new_md["lag_ms"] = offset_ms_int
                try:
                    entry.metadata = new_md  # type: ignore[attr-defined]
                    injected += 1
                except Exception:
                    pass
        return injected

    def filter_entries(
        self,
        entries: Iterable[Any],
        *,
        split: str,
        excluded_log: Optional[Path] = None,
    ) -> "tuple[list[Any], list[dict[str, Any]]]":
        """Return (kept, excluded). Excluded rows include the reason."""

        kept: list[Any] = []
        excluded: list[dict[str, Any]] = []
        if not self.is_active():
            return list(entries), []
        for entry in entries:
            video_id = getattr(entry, "video_id", None)
            if video_id is None:
                kept.append(entry)
                continue
            geom = self.get(split, str(video_id))
            if geom is None:
                excluded.append(
                    {
                        "split": split,
                        "video_id": str(video_id),
                        "reason": "geometry_json_missing",
                        "video_path": str(getattr(entry, "video_path", "")),
                    }
                )
                continue
            if not geom.is_usable_for(split):
                excluded.append(
                    {
                        "split": split,
                        "video_id": str(video_id),
                        "reason": (
                            f"calibration_status={geom.calibration_status} "
                            f"median={geom.residual_median_px} p95={geom.residual_p95_px}"
                        ),
                        "video_path": str(getattr(entry, "video_path", "")),
                    }
                )
                continue
            kept.append(entry)
        if excluded_log is not None and excluded:
            try:
                excluded_log.parent.mkdir(parents=True, exist_ok=True)
                import csv as _csv
                fields = ["split", "video_id", "reason", "video_path"]
                write_header = not excluded_log.exists()
                with excluded_log.open("a", newline="", encoding="utf-8") as handle:
                    writer = _csv.DictWriter(handle, fieldnames=fields)
                    if write_header:
                        writer.writeheader()
                    writer.writerows(excluded)
            except Exception as exc:
                LOGGER.warning("geometry: failed to append excluded log %s (%s)", excluded_log, exc)
        return kept, excluded


__all__ = ["GeometryEntry", "GeometryIndex"]
