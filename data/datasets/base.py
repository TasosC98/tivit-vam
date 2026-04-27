"""
Shared piano dataset backbone.

Purpose:
    - Provide common decode, crop/registration, tiling, normalization, and target prep.
    - Centralize sampler metadata, AV lag handling, cache hookups, and optional hand supervision.
Key Functions/Classes:
    - DatasetEntry
    - BasePianoDataset
CLI Arguments:
    - (none)
Usage:
    - Subclass BasePianoDataset and override root/manifest/label hooks.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from ..decode.video_reader import VideoReaderConfig, load_clip
from ..roi.keyboard_roi import (
    RegistrationRefiner,
    RegistrationResult,
    resolve_registration_cache_path,
    _apply_crop_np,
)
from ..roi.geometry_index import GeometryIndex
from ..roi.tiling import tile_vertical_token_aligned
from ..targets.frame_targets import (
    FrameTargetResult,
    FrameTargetSpec,
    SoftTargetConfig,
    resolve_frame_target_spec,
    resolve_soft_target_config,
    prepare_frame_targets,
)
from ..targets.time_grid import sec_to_frame
from ..sync import resolve_sync, apply_sync, SyncInfo
from ..transforms.augment import apply_global_augment
from ..transforms.normalize import normalize
from ..sampler import build_onset_balanced_sampler
from ..targets.av_sync import AVLagCache
from ..cache.frame_target_cache import FrameTargetCache, NullFrameTargetCache
from ..targets.identifiers import canonical_video_id

LOGGER = logging.getLogger(__name__)
_HDF5_FALLBACK_LOGGED: set[str] = set()


@dataclass
class DatasetEntry:
    # mp4 path (used by registration / caches)
    video_path: Path
    # optional per-video HDF5 frames path (used by decoder when preprocessed_format=hdf5)
    hdf5_path: Optional[Path] = None
    # labels (TSV/MIDI etc)
    label_path: Optional[Path] = None
    video_id: str = ""
    metadata: Mapping[str, Any] = None



class BasePianoDataset(Dataset):
    """Base class implementing decode, registration crop, tiling, and targets."""

    def __init__(self, cfg: Mapping[str, Any], split: str, *, full_cfg: Optional[Mapping[str, Any]] = None):
        super().__init__()
        self.cfg = cfg
        self.full_cfg = full_cfg or {}
        self.split = split
        self.dataset_cfg = dict(cfg.get("dataset", {}))
        self.testing_cfg = dict(self.dataset_cfg.get("testing", {}) or {})
        self.frames = int(self.dataset_cfg.get("frames", 32))
        self.decode_fps = float(self.dataset_cfg.get("decode_fps", 30.0))
        hop_seconds = float(self.dataset_cfg.get("hop_seconds", 1.0 / self.decode_fps))
        self.stride = max(1, int(round(hop_seconds * self.decode_fps)))
        self.skip_seconds = float(self.dataset_cfg.get("skip_seconds", 0.0))
        self.start_frame_offset = max(0, int(round(self.skip_seconds * self.decode_fps)))
        resize_cfg = self.dataset_cfg.get("resize", [224, 224])
        self.resize_hw = (int(resize_cfg[0]), int(resize_cfg[1]))
        self.tiles = int(self.dataset_cfg.get("tiles", 3))
        self.channels = int(self.dataset_cfg.get("channels", 3))
        self.grayscale = bool(self.dataset_cfg.get("grayscale", False))
        self.normalize = bool(self.dataset_cfg.get("normalize", True))
        self.norm_mean = tuple(self.dataset_cfg.get("norm_mean", (0.5, 0.5, 0.5)))
        self.norm_std = tuple(self.dataset_cfg.get("norm_std", (0.5, 0.5, 0.5)))
        self.apply_crop = bool(self.dataset_cfg.get("apply_crop", True))
        self.require_labels = bool(self.dataset_cfg.get("require_labels", False))
        self.target_cfg = dict(self.dataset_cfg.get("frame_targets", {}) or {})
        self.tiling_cfg = dict(self.full_cfg.get("tiling", {}) or {})
        self.sampler_cfg = dict(self.dataset_cfg.get("sampler", {}) or {})
        self.return_debug_extras = bool(self.testing_cfg.get("enable_debug_extras", False))
        avlag_cfg = dict(self.dataset_cfg.get("avlag", {}) or {})
        self.avlag_enabled = bool(avlag_cfg.get("enable", True))

        seed_val = self.full_cfg.get("experiment", {}).get("seed") if isinstance(self.full_cfg, Mapping) else None
        self._rng = torch.Generator()
        self._py_rng = None
        if seed_val is not None:
            self._rng.manual_seed(int(seed_val))
            import random

            self._py_rng = random.Random(int(seed_val))

        canonical_cfg = self.dataset_cfg.get("canonical_hw", self.resize_hw)
        self.canonical_hw = (int(canonical_cfg[0]), int(canonical_cfg[1]))
        reg_cfg = dict(self.dataset_cfg.get("registration", {}) or {})
        self.registration_enabled = bool(reg_cfg.get("enable", True))
        self.registration_refiner = (
            RegistrationRefiner(
                self.canonical_hw,
                cache_path=resolve_registration_cache_path(reg_cfg.get("cache_path")),
                sample_frames=int(reg_cfg.get("sample_frames", self.frames)),
            )
            if self.registration_enabled
            else None
        )

        self._av_sync_cache = AVLagCache() if self.avlag_enabled else None
        if self._av_sync_cache is not None:
            self._av_sync_cache.preload()
        cache_labels = bool(self.target_cfg.get("cache_labels", True))
        if cache_labels:
            self._frame_target_cache = FrameTargetCache(self.target_cfg.get("cache_dir"))
        else:
            self._frame_target_cache = NullFrameTargetCache()
        self.frame_target_spec: Optional[FrameTargetSpec] = resolve_frame_target_spec(
            self.target_cfg,
            frames=self.frames,
            stride=self.stride,
            fps=self.decode_fps,
            canonical_hw=self.canonical_hw,
        )
        training_cfg = self.full_cfg.get("training", {}) if isinstance(self.full_cfg, Mapping) else {}
        soft_target_source: Optional[Mapping[str, Any]] = None
        if isinstance(training_cfg, Mapping):
            training_soft = training_cfg.get("soft_targets")
            if isinstance(training_soft, Mapping):
                soft_target_source = training_soft
        if soft_target_source is None:
            nested_soft = self.target_cfg.get("soft_targets")
            if isinstance(nested_soft, Mapping):
                soft_target_source = nested_soft
        if soft_target_source is None:
            soft_target_source = self.target_cfg
        self.soft_target_cfg: Optional[SoftTargetConfig] = resolve_soft_target_config(soft_target_source)

        root = self._resolve_root(self.dataset_cfg.get("root_dir"))
        manifest = self._resolve_manifest()
        entries = self._list_entries(root, split, manifest)

        # Apply per-video calibration gate when configured. The geometry root
        # points at `key_geometry/` produced by tools/recalibrate_all.py.
        # In TRAIN, exclude videos whose calibration_status is not in
        # {ok, manual_fixed}. In VAL/TEST, also accept accepted_loose.
        # Excluded videos are appended to excluded_videos.csv at the project
        # root for inspection.
        #
        # The gate is bypassed when:
        #   - TIVIT_GEOMETRY_GATE_DISABLED=1 in the environment (used by
        #     tools that populate the geometry root, like recalibrate_all.py).
        #   - The geometry root contains no JSONs for this split yet
        #     (avoids excluding everything before the calibrator has run).
        geom_root = self.dataset_cfg.get("key_geometry_root")
        gate_disabled = os.environ.get("TIVIT_GEOMETRY_GATE_DISABLED", "0") == "1"
        self.geometry_index = GeometryIndex(Path(geom_root)) if geom_root else GeometryIndex(None)
        split_geom_dir = (Path(geom_root) / str(split)) if geom_root else None
        split_has_jsons = bool(
            split_geom_dir is not None
            and split_geom_dir.is_dir()
            and any(split_geom_dir.glob("*.json"))
        )
        if not gate_disabled and self.geometry_index.is_active() and split_has_jsons:
            excluded_log = (
                Path(self.full_cfg.get("logging", {}).get("log_dir", ".")).expanduser()
                / "excluded_videos.csv"
                if isinstance(self.full_cfg, Mapping)
                else None
            )
            kept, excluded = self.geometry_index.filter_entries(
                entries, split=str(split), excluded_log=excluded_log
            )
            if excluded:
                LOGGER.warning(
                    "geometry_gate: split=%s excluded=%d kept=%d (logged to %s)",
                    split,
                    len(excluded),
                    len(kept),
                    excluded_log,
                )
            entries = kept
        elif gate_disabled and self.geometry_index.is_active():
            LOGGER.info(
                "geometry_gate: split=%s DISABLED via TIVIT_GEOMETRY_GATE_DISABLED",
                split,
            )
        elif geom_root and not split_has_jsons:
            LOGGER.warning(
                "geometry_gate: split=%s no JSONs under %s yet — gate not applied. "
                "Run tools/recalibrate_all.py first.",
                split,
                split_geom_dir,
            )

        max_clips = self.dataset_cfg.get("max_clips")
        if max_clips is not None and len(entries) > int(max_clips):
            entries = entries[: int(max_clips)]
        self.entries: List[DatasetEntry] = entries

        if str(self.split).lower() == "train":
            self._sampler = build_onset_balanced_sampler(self, self.sampler_cfg, base_seed=int(seed_val or 0))
        else:
            self._sampler = None

    def build_onset_sampler_metadata(self, nearmiss_radius: int = 0) -> Mapping[str, Any]:
        """Collect onset/background indices plus start frames for the sampler."""

        onset_indices: List[int] = []
        background_indices: List[int] = []
        start_frames: Dict[int, int] = {}
        for idx, entry in enumerate(self.entries):
            try:
                labels = self._read_labels(entry)
                events = labels.get("events", []) if isinstance(labels, Mapping) else []
                if events:
                    onset_indices.append(idx)
                    start_frames[idx] = int(round(float(events[0][0] * self.decode_fps)))
                else:
                    background_indices.append(idx)
            except Exception:
                background_indices.append(idx)

        nearmiss: List[int] = []
        if nearmiss_radius > 0 and start_frames:
            onset_set = set(onset_indices)
            for idx, start in start_frames.items():
                for on_idx in onset_indices:
                    if idx == on_idx:
                        continue
                    on_start = start_frames.get(on_idx, None)
                    if on_start is None:
                        continue
                    if abs(start - on_start) <= nearmiss_radius:
                        nearmiss.append(idx)
                        break

        return {
            "onset": onset_indices,
            "nearmiss": sorted(set(nearmiss)),
            "background": background_indices,
            "start_frames": start_frames,
        }

    # ---- hooks to override -------------------------------------------------

    def _resolve_root(self, root_dir: Optional[str]) -> Path:
        if root_dir:
            return Path(root_dir).expanduser()
        env = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
        return Path(env).expanduser() if env else Path("~/datasets").expanduser()

    def _resolve_manifest(self) -> Optional[Mapping[str, Any]]:
        """Return manifest mapping video_id→annotation info when applicable."""
        return None

    def _list_entries(self, root: Path, split: str, manifest: Optional[Mapping[str, Any]]):
        """Return dataset entries for the split."""
        raise NotImplementedError

    def _read_labels(self, entry: DatasetEntry) -> Mapping[str, Any]:
        """Parse labels for ``entry`` and return a mapping (events, metadata, etc.)."""
        raise NotImplementedError

    # ---- core helpers ------------------------------------------------------

    def _decode_clip(self, path: Path, *, start_frame: int, fallback_path: Optional[Path] = None) -> torch.Tensor:
        """Decode video to tensor using shared reader config."""
        cfg = VideoReaderConfig(
            frames=self.frames,
            stride=self.stride,
            resize_hw=None,
            channels=self.channels,
            start_frame=int(start_frame),
        )
        # Support pre-decoded HDF5 per-video files when available.
        try:
            if path.suffix.lower() in {".h5", ".hdf5"}:
                from ..decode.hdf5_reader import load_clip_from_hdf5

                return load_clip_from_hdf5(path, cfg)
        except Exception as exc:
            # Some older local HDF5 caches contain only a fixed clip per video.
            # Fall back to the original MP4 so the requested start_frame remains
            # aligned with the label window.
            if fallback_path is not None and fallback_path.exists():
                log_key = str(path)
                if log_key not in _HDF5_FALLBACK_LOGGED:
                    LOGGER.warning("HDF5 decode failed for %s (%s); falling back to %s", path, exc, fallback_path)
                    _HDF5_FALLBACK_LOGGED.add(log_key)
                return load_clip(fallback_path, cfg)
            raise
        return load_clip(path, cfg)

    def _choose_start_frame(self, events: Sequence[Sequence[float]]) -> int:
        """
        Choose a clip start frame.
        For training, sample around onsets to increase useful positives and
        to ensure we cover the full temporal extent across iterations.
        """
        # default: respect skip_seconds
        base = int(self.start_frame_offset)
        if str(self.split).lower() != "train":
            return base

        # prefer event-driven sampling
        onset_times = []
        for evt in events or []:
            if len(evt) >= 1:
                try:
                    onset_times.append(float(evt[0]))
                except Exception:
                    pass

        rng = self._py_rng
        import random as _random
        if rng is None:
            rng = _random

        if onset_times:
            t = rng.choice(onset_times)
            onset_frame = int(round(t * self.decode_fps))
            # start a bit before the onset with jitter
            pre = int(round(0.8 * self.decode_fps))   # ~0.8s before
            jitter = int(round(0.6 * self.decode_fps)) # +/- ~0.6s
            start = onset_frame - pre + rng.randint(-jitter, jitter)
            return max(base, int(start))

        # fallback when no events exist: small random walk around skip
        # (still better than fixed)
        return max(base, base + rng.randint(0, int(round(5.0 * self.decode_fps))))

    def _to_grayscale(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert frames to grayscale; replicate to 3 channels when requested."""

        if frames.ndim != 4 or frames.shape[1] == 0:
            return frames
        if frames.shape[1] == 1:
            return frames if self.channels == 1 else frames.repeat(1, 3, 1, 1)
        # frames: T,C,H,W
        r = frames[:, 0:1, :, :]
        g = frames[:, 1:2, :, :]
        b = frames[:, 2:3, :, :]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        if self.channels == 1:
            return gray
        return gray.repeat(1, 3, 1, 1)

    def _apply_calibrated_homography(
        self,
        frames: torch.Tensor,
        payload: Mapping[str, Any],
    ) -> torch.Tensor:
        """Warp frames using a homography produced by the per-video calibrator.

        `frames` is (T, C, H_src, W_src) AFTER the metadata crop has already
        been applied. The calibrator stored its homography in source-cropped
        coordinates so we can apply it directly. Output is (T, C, H_t, W_t)
        where (H_t, W_t) is `payload['target_hw']`.
        """

        H_list = payload.get("homography")
        target_hw = payload.get("target_hw")
        if not (isinstance(H_list, (list, tuple)) and len(H_list) == 9):
            return frames
        if not (isinstance(target_hw, (list, tuple)) and len(target_hw) >= 2):
            return frames
        h_t, w_t = int(target_hw[0]), int(target_hw[1])
        if h_t <= 0 or w_t <= 0:
            return frames
        H = np.asarray(H_list, dtype=np.float32).reshape(3, 3)
        try:
            H_inv = np.linalg.inv(H).astype(np.float32)
        except np.linalg.LinAlgError:
            return frames
        h_src, w_src = int(frames.shape[-2]), int(frames.shape[-1])
        # Build the inverse-warp grid: for each (y, x) in target, find (xs, ys) in source.
        ys = np.linspace(0.0, float(h_t - 1), h_t, dtype=np.float32)
        xs = np.linspace(0.0, float(w_t - 1), w_t, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        ones = np.ones_like(xv)
        coords = np.stack([xv, yv, ones], axis=-1).reshape(-1, 3).T
        mapped = H_inv @ coords
        denom = np.maximum(mapped[2], 1e-6)
        x_src = (mapped[0] / denom).reshape(h_t, w_t)
        y_src = (mapped[1] / denom).reshape(h_t, w_t)
        x_norm = ((x_src + 0.5) / max(float(w_src), 1.0)) * 2.0 - 1.0
        y_norm = ((y_src + 0.5) / max(float(h_src), 1.0)) * 2.0 - 1.0
        grid = torch.from_numpy(np.stack([np.clip(x_norm, -2.0, 2.0), np.clip(y_norm, -2.0, 2.0)], axis=-1).astype(np.float32))
        grid = grid.unsqueeze(0).expand(frames.shape[0], -1, -1, -1).to(frames.device)
        return F.grid_sample(frames, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _apply_registration(
        self,
        frames: torch.Tensor,
        entry: DatasetEntry,
        *,
        debug_meta: Optional[Dict[str, Any]] = None,
        dataset_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply crop based on registration metadata when available."""
        if not self.apply_crop and not self.registration_enabled:
            return frames
        meta_crop = entry.metadata.get("crop") if isinstance(entry.metadata, Mapping) else None
        crop_debug: Optional[Dict[str, Any]] = {} if debug_meta is not None else None
        reg_result: Optional[RegistrationResult] = None
        try:
            if meta_crop is not None:
                arr = frames.permute(0, 2, 3, 1).cpu().numpy()
                cropped = np.stack([_apply_crop_np(frame, meta_crop, crop_debug) for frame in arr], axis=0)
                frames = torch.from_numpy(cropped).permute(0, 3, 1, 2)
        except Exception:
            pass

        # Prefer the per-video calibrated homography (key_geometry/) when
        # present; fall back to the legacy RegistrationRefiner only if no
        # calibrated geometry exists for this video.
        calibrated_used = False
        geom_idx = getattr(self, "geometry_index", None)
        if geom_idx is not None and geom_idx.is_active():
            geom_entry = geom_idx.get(self.split, entry.video_id)
            if geom_entry is not None and geom_entry.calibration_status in {"ok", "manual_fixed", "accepted_loose"}:
                try:
                    frames = self._apply_calibrated_homography(frames, geom_entry.payload)
                    calibrated_used = True
                    if debug_meta is not None:
                        debug_meta["registration"] = {
                            "status": geom_entry.calibration_status,
                            "err_before": float(geom_entry.payload.get("residual_median_px") or 0.0),
                            "err_after": float(geom_entry.payload.get("residual_median_px") or 0.0),
                            "err_white": 0.0,
                            "err_black": 0.0,
                            "frames": int(frames.shape[0]) if frames.ndim >= 4 else 0,
                            "source_hw": list(geom_entry.payload.get("source_hw") or [0, 0]),
                            "target_hw": list(geom_entry.payload.get("target_hw") or [0, 0]),
                            "cache_geometry": {
                                "rectified_width": float(geom_entry.payload.get("target_hw", [0, 0])[1]),
                                "target_hw": list(geom_entry.payload.get("target_hw") or [0, 0]),
                                "key_polygons_rectified": geom_entry.payload.get("key_polygons_rectified"),
                                "key_bounds_px": [
                                    [
                                        min(float(p[0]) for p in poly),
                                        max(float(p[0]) for p in poly),
                                    ]
                                    for poly in (geom_entry.payload.get("key_polygons_rectified") or [])
                                ],
                                "tile_bounds_px": [
                                    [i * float(geom_entry.payload.get("target_hw", [0, 0])[1]) / 3.0,
                                     (i + 1) * float(geom_entry.payload.get("target_hw", [0, 0])[1]) / 3.0]
                                    for i in range(3)
                                ],
                                "calibration_source": "key_geometry",
                            },
                        }
                except Exception as exc:
                    LOGGER.warning(
                        "calibrated homography failed for %s (%s); falling back to legacy refiner",
                        entry.video_id,
                        exc,
                    )
                    calibrated_used = False
        try:
            # Run registration refiner to warp frames to canonical HW when enabled.
            if not calibrated_used and self.registration_enabled and self.registration_refiner is not None:
                debug_context = {"split": self.split, "dataset_index": dataset_index}
                transformed = self.registration_refiner.transform_clip(
                    frames,
                    video_id=entry.video_id,
                    video_path=entry.video_path,
                    crop_meta=meta_crop,
                    interp="bilinear",
                    debug_context=debug_context if self.return_debug_extras else None,
                    return_result=self.return_debug_extras,
                )
                if isinstance(transformed, tuple):
                    frames, reg_result = transformed
                else:
                    frames = transformed
        except Exception:
            pass
        if debug_meta is not None:
            debug_meta["crop"] = crop_debug
            if reg_result is not None:
                debug_meta["registration"] = {
                    "status": reg_result.status,
                    "err_before": float(reg_result.err_before),
                    "err_after": float(reg_result.err_after),
                    "err_white": float(reg_result.err_white_edges),
                    "err_black": float(reg_result.err_black_gaps),
                    "frames": int(reg_result.frames),
                    "source_hw": list(reg_result.source_hw),
                    "target_hw": list(reg_result.target_hw),
                    "cache_geometry": reg_result.geometry_meta,
                }
        return frames

    def _tile_frames(self, frames: torch.Tensor, *, return_meta: bool = False):
        """Tile frames into token-aligned slices."""
        patch_w = int(self.tiling_cfg.get("patch_w", self.dataset_cfg.get("patch_w", 16)))
        tokens_split = self.tiling_cfg.get("tokens_split", "auto")
        overlap_tokens = int(self.tiling_cfg.get("overlap_tokens", 0))
        h_pre, w_pre = int(frames.shape[-2]), int(frames.shape[-1])
        tiles, tokens_per_tile, widths, bounds, aligned_w, original_w = tile_vertical_token_aligned(
            frames,
            tiles=self.tiles,
            patch_w=patch_w,
            tokens_split=tokens_split,
            overlap_tokens=overlap_tokens,
        )
        pre_pad_widths = [int(r - l) for (l, r) in bounds]
        pad_right: List[int] = []
        pad_left: List[int] = []
        if tiles:
            max_w = max(tile.shape[-1] for tile in tiles)
            if any(tile.shape[-1] != max_w for tile in tiles):
                padded: List[torch.Tensor] = []
                for idx, (tile, pre_w) in enumerate(zip(tiles, pre_pad_widths)):
                    pad_w = max_w - tile.shape[-1]
                    if pad_w > 0:
                        if idx == 0:
                            # First tile pads on the left edge.
                            edge = tile[..., :1].expand(*tile.shape[:-1], pad_w)
                            tile = torch.cat([edge, tile], dim=-1)
                            pad_left.append(pad_w)
                            pad_right.append(0)
                        else:
                            edge = tile[..., -1:].expand(*tile.shape[:-1], pad_w)
                            tile = torch.cat([tile, edge], dim=-1)
                            pad_left.append(0)
                            pad_right.append(pad_w)
                    else:
                        pad_left.append(0)
                        pad_right.append(0)
                    padded.append(tile)
                tiles = padded
            else:
                pad_right = [0 for _ in tiles]
                pad_left = [0 for _ in tiles]
        max_w = max(tile.shape[-1] for tile in tiles) if tiles else 0
        stacked = torch.stack(tiles, dim=2)
        if not return_meta:
            return stacked
        tile_hw = [(int(tile.shape[-2]), int(tile.shape[-1])) for tile in tiles]
        pad_w_left = pad_left if pad_left else [0 for _ in pad_right]
        canonical_before_pad = (h_pre, int(aligned_w))
        canonical_after_pad = (h_pre, max_w)
        tile_xyxy = [(int(l), 0, int(r), int(h_pre)) for (l, r) in bounds]
        meta = {
            "tiles": int(self.tiles),
            "patch_w": patch_w,
            "tokens_split": tokens_split,
            "overlap_tokens": overlap_tokens,
            "tokens_per_tile": tokens_per_tile,
            "tile_bounds_px": bounds,
            "aligned_width": int(aligned_w),
            "original_width": int(original_w),
            "tile_hw": tile_hw,
            "tile_xyxy": tile_xyxy,
            "pad_w_left": pad_w_left,
            "pad_w_right": pad_right,
            "canonical_before_pad": canonical_before_pad,
            "canonical_after_pad": canonical_after_pad,
            "coord_system": "post_crop_aligned",
        }
        return stacked, meta

    def _prepare_targets(
        self,
        events: Sequence[Sequence[float]],
        clip_meta: Mapping[str, Any],
        *,
        hand_seq: Optional[Sequence[int]] = None,
        clef_seq: Optional[Sequence[int]] = None,
        debug_meta: Optional[Dict[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Build frame targets from aligned events and optional hand/clef sequences."""
        if events is None or len(events) == 0:
            if self.require_labels:
                raise FileNotFoundError("Missing labels")
            return {}
        labels_tensor = torch.tensor(events, dtype=torch.float32)
        if self.frame_target_spec is None:
            return {}
        lag_seconds = float(clip_meta.get("lag_seconds", 0.0))
        lag_result = None
        try:
            from ..targets.av_sync import AVLagResult  # type: ignore
            lag_ms = lag_seconds * 1000.0
            lag_frames = int(round(lag_ms / max(self.frame_target_spec.fps, 1e-6)))
            lag_result = AVLagResult(
                lag_frames=lag_frames,
                lag_ms=lag_ms,
                corr=1.0,
                from_cache=True,
                success=True,
                runtime_s=0.0,
                flags=set(["proxy"]),
            )
        except Exception:
            lag_result = None
        target_trace: Optional[Dict[str, Any]] = {} if debug_meta is not None else None
        result: FrameTargetResult = prepare_frame_targets(
            labels=labels_tensor,
            lag_result=lag_result,
            spec=self.frame_target_spec,
            cache=self._frame_target_cache,
            split=self.split,
            video_id=str(clip_meta.get("video_uid", "")),
            clip_start=float(clip_meta.get("clip_start", 0.0)),
            soft_targets=self.soft_target_cfg,
            trace=target_trace,
        )
        payload = result.payload or {}
        clip_start_sec = float(clip_meta.get("clip_start", 0.0))
        hop_seconds = self.stride / max(self.decode_fps, 1e-6)
        duration_sec = hop_seconds * max(self.frames - 1, 0)
        # Prefer explicit hand/clef hints
        if payload.get("hand_frame") is None and hand_seq:
            payload["hand_frame"] = torch.tensor(int(round(sum(hand_seq) / len(hand_seq))), dtype=torch.long)
        if payload.get("clef_frame") is None and clef_seq:
            payload["clef_frame"] = torch.tensor(int(round(sum(clef_seq) / len(clef_seq))), dtype=torch.long)
        # Add pitch-based hints when still missing.
        if payload.get("hand_frame") is None and events:
            mean_pitch = float(sum(evt[2] for evt in events if len(evt) >= 3) / max(len(events), 1))
            payload["hand_frame"] = torch.tensor(0 if mean_pitch < 60 else 1, dtype=torch.long)
        if payload.get("clef_frame") is None and events:
            mean_pitch = float(sum(evt[2] for evt in events if len(evt) >= 3) / max(len(events), 1))
            clef_val = 0 if mean_pitch < 60 else (1 if mean_pitch > 64 else 2)
            payload["clef_frame"] = torch.tensor(clef_val, dtype=torch.long)
        if debug_meta is not None:
            debug_meta["target_build"] = {
                "status": result.status,
                "cache_key": result.cache_key,
                "cache_meta": dict(result.cache_meta) if isinstance(result.cache_meta, Mapping) else result.cache_meta,
                "lag_ms": result.lag_ms,
                "lag_frames": result.lag_frames,
                "lag_source": result.lag_source,
                "events_count": len(events),
                "clip_start_sec": clip_start_sec,
                "hop_seconds": hop_seconds,
                "duration_sec": duration_sec,
                "spec": {
                    "frames": self.frame_target_spec.frames if self.frame_target_spec is not None else None,
                    "stride": self.frame_target_spec.stride if self.frame_target_spec is not None else None,
                    "fps": self.frame_target_spec.fps if self.frame_target_spec is not None else None,
                    "canonical_hw": list(self.frame_target_spec.canonical_hw) if self.frame_target_spec is not None else None,
                },
            }
            if target_trace is not None:
                try:
                    debug_meta["target_build"].update(target_trace)
                except Exception:
                    pass
        return {
            "pitch": payload.get("pitch_roll"),
            "onset": payload.get("onset_roll"),
            "offset": payload.get("offset_roll"),
            "hand": payload.get("hand_frame"),
            "clef": payload.get("clef_frame"),
        }

    def _prepare_hand_supervision(
        self,
        *,
        entry: DatasetEntry,
        clip_meta: Mapping[str, Any],
        raw: Mapping[str, Any],
        source_hw: Optional[Sequence[int]],
    ) -> Mapping[str, Any]:
        cfg = dict(self.dataset_cfg.get("hand_supervision", {}) or {})
        if not cfg or not bool(cfg.get("enable", False)):
            return {}
        if self.registration_refiner is None:
            return {}

        hand_path = raw.get("hand_skeleton_path")
        if not isinstance(hand_path, (str, Path)):
            return {}
        if source_hw is None or len(source_hw) < 2:
            return {}

        try:
            from ..hand_labels import (
                EventHandLabelConfig,
                build_event_hand_labels,
                compute_hand_reach,
                key_centers_from_geometry,
                load_pianovam_hand_landmarks,
                map_landmarks_to_canonical,
            )
        except Exception:
            return {}

        clip_start = float(clip_meta.get("clip_start", 0.0))
        time_tol = cfg.get("time_tolerance")
        min_conf = float(cfg.get("min_confidence", 0.0))
        aligned = load_pianovam_hand_landmarks(
            hand_path,
            clip_start_sec=clip_start,
            frames=self.frames,
            stride=self.stride,
            decode_fps=self.decode_fps,
            min_confidence=min_conf,
            time_tolerance=float(time_tol) if time_tol is not None else None,
        )

        reg_payload = self.registration_refiner.get_cache_entry_payload(entry.video_id)
        canonical = map_landmarks_to_canonical(
            aligned,
            registration=reg_payload,
            source_hw=source_hw,
            crop_meta=entry.metadata,
        )
        geometry_meta = self.registration_refiner.get_geometry_metadata(entry.video_id)
        key_centers = key_centers_from_geometry(geometry_meta)
        if key_centers is None:
            return {}

        outputs: Dict[str, Any] = {}

        events = raw.get("events", [])
        if events:
            onsets: List[float] = []
            pitches: List[int] = []
            for evt in events:
                if len(evt) < 3:
                    continue
                try:
                    onsets.append(float(evt[0]))
                    pitches.append(int(round(float(evt[2]))))
                except (TypeError, ValueError):
                    continue
            if onsets and pitches:
                onsets_t = torch.tensor(onsets, dtype=torch.float32)
                pitch_t = torch.tensor(pitches, dtype=torch.int64)
                note_min = int(self.target_cfg.get("note_min", 21))
                key_indices = pitch_t - int(note_min)
                valid = (key_indices >= 0) & (key_indices < key_centers.numel())
                if valid.any():
                    onsets_t = onsets_t[valid]
                    key_indices = key_indices[valid]
                    cfg_obj = EventHandLabelConfig(
                        time_tolerance=float(cfg.get("time_tolerance", 0.05)),
                        max_dx=float(cfg.get("max_dx", 0.12)),
                        min_points=int(cfg.get("min_points", 4)),
                        unknown_class=int(cfg.get("unknown_class", 2)),
                    )
                    evt_labels = build_event_hand_labels(
                        onsets_sec=onsets_t,
                        key_indices=key_indices,
                        key_centers_norm=key_centers,
                        frame_times=aligned.frame_times,
                        canonical=canonical,
                        config=cfg_obj,
                    )

                    T = aligned.frame_times.numel()
                    frame_lab = torch.zeros((T,), dtype=torch.long)
                    frame_mask = torch.zeros((T,), dtype=torch.bool)
                    for onset, lbl, valid_evt in zip(onsets_t.tolist(), evt_labels.labels.tolist(), evt_labels.mask.tolist()):
                        if not valid_evt:
                            continue
                        idx = int(torch.abs(aligned.frame_times - float(onset)).argmin().item())
                        if idx < 0 or idx >= T:
                            continue
                        if not frame_mask[idx]:
                            frame_lab[idx] = int(lbl)
                            frame_mask[idx] = True
                        elif frame_lab[idx] != int(lbl):
                            frame_lab[idx] = 0
                            frame_mask[idx] = False

                    outputs["hand"] = frame_lab
                    outputs["hand_frame_mask"] = frame_mask
                    outputs["hand_coverage"] = evt_labels.coverage
                    outputs["hand_events"] = evt_labels.labels
                    outputs["hand_event_mask"] = evt_labels.mask

        reach_cfg = cfg.get("reach", {}) or {}
        if bool(reach_cfg.get("enable", False)):
            reach = compute_hand_reach(
                canonical,
                key_centers_norm=key_centers,
                radius=float(reach_cfg.get("radius", 0.12)),
                dilate=float(reach_cfg.get("dilate_margin", 0.02)),
                min_points=int(reach_cfg.get("min_points", 4)),
            )
            outputs["hand_reach"] = reach.reach
            outputs["hand_reach_valid"] = reach.valid
            outputs["hand_reach_coverage"] = reach.coverage

        return outputs

    def _sampler_meta(self, events: Sequence[Sequence[float]]) -> Mapping[str, Any]:
        """Return minimal sampler metadata for onset-balanced samplers."""

        has_onset = bool(events)
        onset_count = len(events)
        pitches = [evt[2] for evt in events if len(evt) >= 3]
        mean_pitch = float(sum(pitches) / len(pitches)) if pitches else None
        min_pitch = min(pitches) if pitches else None
        max_pitch = max(pitches) if pitches else None
        onset_times = [float(evt[0]) for evt in events if len(evt) >= 2]
        start_frame = int(round(onset_times[0] * self.decode_fps)) if onset_times else None
        return {
            "has_onset": has_onset,
            "event_count": onset_count,
            "mean_pitch": mean_pitch,
            "min_pitch": min_pitch,
            "max_pitch": max_pitch,
            "onset_times": onset_times[:8],
            "start_frame": start_frame,
        }

    # ---- Dataset API -------------------------------------------------------

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        """Return one sample dict containing video, targets, sync, sampler meta, and metadata."""
        entry = self.entries[idx]
        if os.getenv("DEBUG_ENTRY", "0") == "1":
            # τύπωσε μόνο τα πρώτα λίγα για να μην γίνει spam
            if idx < int(os.getenv("DEBUG_ENTRY_N", "5")):
                print(
                    f"[DEBUG_ENTRY] idx={idx} "
                    f"video_path={getattr(entry, 'video_path', None)} "
                    f"hdf5_path={getattr(entry, 'hdf5_path', None)} "
                    f"label_path={getattr(entry, 'label_path', None)} "
                    f"preprocessed_format={getattr(self, 'preprocessed_format', None)}"
                )
        debug_extras: Optional[Dict[str, Any]] = {} if self.return_debug_extras else None
        decode_path = entry.video_path
        preproc = str(self.dataset_cfg.get("preprocessed_format") or "").lower()
        if preproc == "hdf5" and entry.hdf5_path is not None:
            decode_path = entry.hdf5_path
        if os.getenv("DEBUG_ENTRY", "0") == "1":
            if idx < int(os.getenv("DEBUG_ENTRY_N", "5")):
                print(f"[DEBUG_ENTRY] idx={idx} decode_path={decode_path}")

         # Read labels early so we can choose a good start frame.

        raw_labels = self._read_labels(entry)
        raw: MutableMapping[str, Any] = {"events": raw_labels.get("events", []) if isinstance(raw_labels, Mapping) else []}
        if isinstance(raw_labels, Mapping):
            for key in ("metadata", "hand_skeleton_path"):
                if key in raw_labels:
                    raw[key] = raw_labels[key]
            if "hand_seq" in raw_labels:
                raw["hand_seq"] = raw_labels["hand_seq"]
            if "clef_seq" in raw_labels:
                raw["clef_seq"] = raw_labels["clef_seq"]

        events_before_sync = list(raw.get("events", []))
        if self.avlag_enabled:
            sync_info = resolve_sync(entry.video_id, entry.metadata, self._av_sync_cache)
            apply_sync(raw, sync_info)
        else:
            sync_info = SyncInfo(lag_seconds=0.0, source="disabled")

        # Choose a per-sample start frame (train: random around onsets)
        start_frame = self._choose_start_frame(raw.get("events", []))
        fallback_path = entry.video_path if decode_path != entry.video_path else None
        frames = self._decode_clip(decode_path, start_frame=start_frame, fallback_path=fallback_path)
        source_hw = (int(frames.shape[-2]), int(frames.shape[-1])) if frames.ndim >= 4 else None
        if debug_extras is not None:
            debug_extras["decode"] = {
                "frames": self.frames,
                "stride": self.stride,
                "decode_fps": self.decode_fps,
                "start_frame": int(start_frame),
                "source_hw": [int(frames.shape[-2]), int(frames.shape[-1])] if frames.ndim >= 4 else None,
                "channels": self.channels,
            }
        if self.grayscale:
            frames = self._to_grayscale(frames)
            if debug_extras is not None:
                debug_extras["grayscale"] = {"enabled": True, "mode": "convert_after_decode" if self.channels != 1 else "decode_gray"}
        frames = self._apply_registration(frames, entry, debug_meta=debug_extras, dataset_index=idx)
        if not self.registration_enabled and self.canonical_hw:
            try:
                frames = F.interpolate(frames, size=self.canonical_hw, mode="bilinear", align_corners=False)
            except Exception:
                pass
        frames = apply_global_augment(frames, self.dataset_cfg.get("registration", {}).get("global_aug"), rng=self._py_rng)
        if self.normalize:
            frames = normalize(frames, mean=self.norm_mean, std=self.norm_std)
        tiles_out = self._tile_frames(frames, return_meta=self.return_debug_extras)
        if self.return_debug_extras:
            tiles, tiling_meta = tiles_out  # type: ignore[misc,assignment]
            if debug_extras is not None:
                debug_extras["tiling"] = tiling_meta
        else:
            tiles = tiles_out  # type: ignore[assignment]
        clip_meta = {
            # Keep mp4 path here for registration/cache consistency.
            "path": str(entry.video_path),
            # Extra paths for debugging/inspection.
            "video_path": str(entry.video_path),
            "hdf5_path": str(entry.hdf5_path) if entry.hdf5_path is not None else None,
            "decode_path": str(decode_path),
            "video_uid": entry.video_id,
            "clip_start": float(start_frame) / max(self.decode_fps, 1e-6),
        }
        
        if debug_extras is not None:
            debug_extras["sync"] = {
                "lag_seconds": sync_info.lag_seconds,
                "source": sync_info.source,
                "events_before_sync": events_before_sync,
                "events_after_sync": list(raw.get("events", [])),
            }
        targets = self._prepare_targets(
            raw.get("events", []),
            clip_meta,
            hand_seq=raw.get("hand_seq"),
            clef_seq=raw.get("clef_seq"),
            debug_meta=debug_extras,
        ) if raw.get("events") else {}
        hand_supervision = self._prepare_hand_supervision(
            entry=entry,
            clip_meta=clip_meta,
            raw=raw,
            source_hw=source_hw,
        )
        sample: MutableMapping[str, Any] = {
            "video": tiles,
            "path": clip_meta["path"],
            "video_path": clip_meta.get("video_path"),
            "hdf5_path": clip_meta.get("hdf5_path"),
            "decode_path": clip_meta.get("decode_path"),
            "video_uid": str(clip_meta["video_uid"]), # IMPORTANT: keep raw unique id to avoid collisions
            "sync": {"lag_seconds": sync_info.lag_seconds, "source": sync_info.source},
            "sampler_meta": self._sampler_meta(raw.get("events", [])),
            "metadata": raw.get("metadata", {}),
        }
        if "hand_skeleton_path" in raw:
            sample["hand_skeleton_path"] = raw["hand_skeleton_path"]
        sample.update(targets)
        if hand_supervision:
            sample.update(hand_supervision)
        if debug_extras is not None:
            debug_extras["post_tile_shape"] = list(tiles.shape) if torch.is_tensor(tiles) else None
            debug_extras["canonical_hw"] = list(self.canonical_hw)
            debug_extras["index"] = idx
            debug_extras["video_path"] = str(entry.video_path)
            debug_extras["video_uid"] = str(clip_meta["video_uid"])
            sample["_debug_extras"] = debug_extras
        return sample


class DatasetAdapter:
    """Adapter that builds a DataLoader from a BasePianoDataset subclass."""

    def __init__(self, cfg: Mapping[str, Any], split: str, *, seed: Optional[int] = None):
        self.cfg = cfg
        self.split = split
        self.seed = seed

    def dataloader(self, drop_last: bool = False):
        raise NotImplementedError


# Keep metadata-like fields as per-sample lists to avoid default_collate failures.
_NON_COLLATE_KEYS = {"metadata", "sampler_meta", "_debug_extras", "hand_skeleton_path"}
_CRITICAL_KEYS = {
    "video",
    "pitch",
    "onset",
    "offset",
    "hand",
    "clef",
    "hand_frame_mask",
    "hand_mask",
    "hand_reach",
    "hand_reach_valid",
}


def safe_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate batch dicts while leaving variable-length metadata unstacked.

    This prevents DataLoader from crashing on lists with inconsistent lengths,
    while still enforcing strict collation for tensors used in training.
    """
    if not batch:
        return {}
    keys: set[str] = set()
    for sample in batch:
        keys.update(sample.keys())
    collated: Dict[str, Any] = {}
    for key in keys:
        values = [sample.get(key) for sample in batch]
        if key in _NON_COLLATE_KEYS:
            # Preserve per-sample metadata/debug objects without stacking.
            collated[key] = values
            continue
        try:
            collated[key] = default_collate(values)
        except Exception:
            if key in _CRITICAL_KEYS:
                raise
            # Fall back to a list for auxiliary fields that may vary per sample.
            collated[key] = values
    return collated
