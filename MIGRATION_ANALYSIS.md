# TiViT-VAM Repository Path & Configuration Analysis

## Executive Summary

This document provides a comprehensive analysis of the tivit-vam ML repository for migration to a new server. The codebase uses **environment variables for dataset paths** and **automatic GPU detection**. Most paths are configurable through YAML, but some are hardcoded relative paths and file lookups.

---

## 1. CONFIGURATION HIERARCHY

### Primary Config Entry Point
- **Location**: [configs/default.yaml](configs/default.yaml)
- **Structure**: Uses YAML composition with `bases` (implicit stacking)

```yaml
bases:
  - experiment/base.yaml
  - dataset/pianovam.yaml
  - model/vivit_tiled.yaml
  - train/single_run.yaml
  - decoder/decoder.yaml
  - priors/priors.yaml
  - train/autopilot.yaml
```

### Config Resolution
- **Module**: [core/config.py](core/config.py)
- **Function**: `load_experiment_config(configs, default_base=DEFAULT_CONFIG_PATH)`
- **Process**:
  1. Loads config files from `configs/` directory (relative to repo root)
  2. Recursively merges `base`/`bases` entries
  3. Deep-merges overlays on top of base
  4. Validates against `ALLOWED_TOP_LEVEL_KEYS`
  5. Automatically expands `~` (home) and environment variables via `os.path.expandvars()`

### Configuration Subdirectories

| Directory | Purpose | Files |
|-----------|---------|-------|
| `configs/experiment/` | Metadata, logging, inference paths | base.yaml |
| `configs/dataset/` | Dataset names, paths, sampling params | pianovam.yaml, pianoyt.yaml, omaps.yaml |
| `configs/train/` | Hyperparameters, loss config, metrics | single_run.yaml, autopilot.yaml |
| `configs/model/` | Backbone type, transformer config | vivit_tiled.yaml, vit_small_tiled.yaml |
| `configs/decoder/` | Post-processing thresholds, decoders | decoder.yaml |
| `configs/calib/` | Calibration method and parameters | platt.yaml, temperature.yaml, threshold_sweep.yaml |
| `configs/priors/` | Constraint configurations | priors.yaml |

---

## 2. ENVIRONMENT VARIABLES (CRITICAL FOR MIGRATION)

The repository **relies on environment variables** for dataset path resolution. These must be set before running training/evaluation.

### Dataset Path Variables

| Variable | Priority | Used By | Fallback |
|----------|----------|---------|----------|
| `PIANOVAM_ROOT` | 1st | PianoVAMDataset | TIVIT_DATA_DIR |
| `PIANOVAM_HDF5_ROOT` | 1st (HDF5 mode) | PianoVAMDataset | None - required for HDF5 |
| `TIVIT_DATA_DIR` | 2nd | All datasets | DATASETS_HOME |
| `DATASETS_HOME` | 3rd | All datasets | Fallback paths |

### Other Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TIVIT_LOG_DIR` | Logging output directory | "logs" |
| `TIVIT_DATASET_CONFIG` | Config file path (test) | None |
| `TIVIT_DATASET` | Dataset key (test) | None |
| `TIVIT_CANARY_COUNT` | Number of canary samples | Default from config |
| `TIVIT_CANARY_AUDIT_DIR` | Audit output directory | tivit/tests/resources/audit/ |
| `TIVIT_REG_REFINED` | Registration cache override | None |
| `PYTHONHASHSEED` | Python hash randomization | Set by configure_determinism() |

---

## 3. HARDCODED PATHS IN PYTHON CODE

### Registration Cache
**File**: [data/roi/keyboard_roi.py](data/roi/keyboard_roi.py) line ~930

```python
candidate = Path("reg_refined.json")  # Fallback path in current directory
cache_candidate = Path(cache_path) if cache_path else Path("reg_refined.json")
```

**Also in repo root:**
- `reg_refined.json` (PianoYT)
- `reg_refined_pianovam.json` (PianoVAM)

These are JSON files containing keyboard geometry/registration data. Referenced in `configs/dataset/*.yaml` under `registration.cache_path`.

### Default Dataset Paths (Fallbacks)
**File**: [data/datasets/pianovam_impl.py](data/datasets/pianovam_impl.py) line ~118

```python
return Path("~/datasets/PianoVAM_v1.0").expanduser()  # Fallback if no env vars
```

Similar fallbacks in:
- `pianoyt_impl.py`: `~/datasets/PianoYT`
- `omaps_impl.py`: `~/datasets/OMAPS`

### Pos-Weight Path
**File**: [configs/train/single_run.yaml](configs/train/single_run.yaml) line 92

```yaml
pos_weight_path: tivit/logs/pos_weights.json
```

**Resolved in**: [losses/multitask_loss.py](losses/multitask_loss.py) line 100

This path is relative to the repo root. JSON file with precomputed position weights for loss balancing.

### Logging Directories
**File**: [utils/logging.py](utils/logging.py) line 69

```python
log_dir_path = Path(log_dir or os.environ.get("TIVIT_LOG_DIR", "logs")).expanduser()
```

**Config**: [configs/experiment/base.yaml](configs/experiment/base.yaml)

```yaml
logging:
  log_dir: logs
  checkpoint_dir: checkpoints
```

---

## 4. DATASET CONFIGURATION

### PianoVAM (Primary Dataset)
**Config**: [configs/dataset/pianovam.yaml](configs/dataset/pianovam.yaml) & [configs/pianovam_local.yaml](configs/pianovam_local.yaml)

```yaml
dataset:
  name: PianoVAM
  root_dir: data/PianoVAM_v1.0                    # Fallback path (relative to repo)
  root_dir: ${PIANOVAM_ROOT}                      # ENV+config overrides this
  hdf5_root: ${PIANOVAM_HDF5_ROOT}               # For preprocessed HDF5 files
  preprocessed_format: null                       # "hdf5" or null (mp4)
  frames: 96                                      # Clip length
  resize: [180, 1536]                            # Output frame size
  tiles: 3                                        # Number of tile columns
  batch_size: 2
  num_workers: 2
```

**Directory Structure Expected:**
```
$PIANOVAM_ROOT/
  Video/                  # MP4 files
  metadata_v2.json       # Video metadata indexed by record_time
  MIDI/                  # MIDI annotations
  HDF5/                  # Preprocessed files (if using hdf5_root)
```

### PianoYT Dataset
**Config**: [configs/dataset/pianoyt.yaml](configs/dataset/pianoyt.yaml)

```yaml
dataset:
  name: PianoYT
  root_dir: data/PianoYT
  frames: 96
  resize: [145, 1024]
  tiles: 3
```

### OMAPS Dataset
**Config**: [configs/dataset/omaps.yaml](configs/dataset/omaps.yaml)

```yaml
dataset:
  name: OMAPS
  root_dir: data/omaps
  frames: 96
  resize: [145, 1024]
  tiles: 3
```

---

## 5. GPU & DEVICE CONFIGURATION

### Automatic GPU Detection
**File**: [pipelines/_common.py](pipelines/_common.py) line 127

```python
def setup_runtime(cfg, seed=None, deterministic=None):
    # ... determinism setup ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return seed_val, det_flag, device
```

**Key Points:**
- **No explicit GPU selection** - Uses `torch.cuda.is_available()`
- Falls back to CPU if CUDA unavailable
- Device is passed to all model/data operations
- Used in: training loop, evaluation, calibration

### Mixed Precision (AMP)
**File**: [configs/train/single_run.yaml](configs/train/single_run.yaml)

```yaml
training:
  amp: false  # Enable Automatic Mixed Precision
```

**Config Check**: [pipelines/evaluate.py](pipelines/evaluate.py) line 244

```python
amp_enabled = bool(training_cfg.get("amp", False)) and torch.cuda.is_available()
```

### CUDA Determinism
**File**: [core/determinism.py](core/determinism.py) line 75

```python
def configure_determinism(seed, deterministic=True):
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
```

---

## 6. MAIN ENTRY POINTS & SCRIPTS

### Training
**File**: [pipelines/train_single.py](pipelines/train_single.py)

```bash
python -m tivit.pipelines.train_single \
  --config configs/default.yaml \
  --config configs/overrides/my_override.yaml \
  --train-split train \
  --val-split val \
  --frames 96 \
  --batch-size 2 \
  --seed 1332 \
  --verbose quiet
```

**Key Logic**:
1. Calls `prepare_run()` to load configs + setup logging
2. Calls `run_training()` from [train/loop.py](train/loop.py)
3. Saves resolved config to `logs/resolved_config.yaml`

### Evaluation
**File**: [pipelines/evaluate.py](pipelines/evaluate.py)

```bash
python -m tivit.pipelines.evaluate \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt \
  --split val
```

### Calibration
**File**: [pipelines/calibrate.py](pipelines/calibrate.py)

```bash
python -m tivit.pipelines.calibrate \
  --config configs/default.yaml \
  --config configs/calib/threshold_sweep.yaml \
  --checkpoint checkpoints/best.pt
```

### Utility Scripts
**Location**: [scripts/](scripts/)

| Script | Purpose |
|--------|---------|
| `smoke_print_batch_shapes.py` | Test dataloader shapes (requires `PIANOVAM_ROOT` + `PIANOVAM_HDF5_ROOT`) |
| `convert_videos_to_hdf5.py` | Convert MP4 → HDF5 preprocessing |
| `convert_pianovam_all.py` | Batch convert with `--video-dir` and `--out-dir` |

### Registration Refinement
**File**: [preproc/refine_registration_cache.py](preproc/refine_registration_cache.py)

```bash
python -m tivit.preproc.refine_registration_cache \
  configs/dataset/pianovam.yaml \
  --debug
```

Populates `reg_refined_pianovam.json` registration cache.

---

## 7. CHECKPOINT & LOGGING PATHS

### Checkpoint Loading
**File**: [pipelines/_common.py](pipelines/_common.py) line 36

```python
def find_checkpoint(cfg, checkpoint=None):
    if checkpoint:
        resolved = Path(checkpoint).expanduser()
        return resolved if resolved.exists() else None
    
    log_cfg = cfg.get("logging", {})
    ckpt_dir = Path(log_cfg.get("checkpoint_dir", "./checkpoints")).expanduser()
    
    # Look for best.pt, then latest epoch_*.pt
    best = ckpt_dir / "best.pt"
    if best.exists():
        return best
    candidates = list(ckpt_dir.glob("epoch_*.pt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
```

### Logging Directory Resolution
**File**: [pipelines/_common.py](pipelines/_common.py) line 24

```python
def prepare_run(configs, stage_name, default_log_file, verbose=None):
    cfg = dict(load_experiment_config(configs))
    log_cfg = cfg.get("logging", {})
    log_dir = Path(log_cfg.get("log_dir", "logs")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    ...
    write_run_artifacts(cfg, log_dir=log_dir, command=sys.argv, configs=configs)
```

**Artifacts Generated**:
- `resolved_config.yaml` - Merged config used for run
- `git_commit.txt` - Git SHA for reproducibility
- `command.txt` - Full command line + configs used
- `train.log` / `eval.log` / `calibration.log` - Stage logs

---

## 8. CURRENT DATASET STRUCTURE (FROM EXAMPLES)

### PianoVAM Expected Structure
Based on code in [data/datasets/pianovam_impl.py](data/datasets/pianovam_impl.py):

```
/path/to/PianoVAM_v1.0/
├── Video/                          # Raw MP4 videos
│   ├── 2024-02-14_19-10-09.mp4
│   └── ...
├── metadata_v2.json               # Video metadata (indexed by record_time)
├── MIDI/                          # MIDI annotation files
├── HDF5/                          # (Optional) Preprocessed video clips
│   ├── 2024-02-14_19-10-09.h5
│   └── ...
└── (optional manifest files for splits)
```

### Cache Files (Root Directory)
```
repo_root/
├── reg_refined_pianovam.json      # Registration cache for PianoVAM
├── reg_refined.json               # Registration cache for other datasets
├── logs/
│   ├── pos_weights.json           # Precomputed loss weights
│   ├── checkpoints/
│   │   ├── best.pt                # Best model checkpoint
│   │   ├── epoch_0.pt
│   │   └── epoch_N.pt
│   ├── resolved_config.yaml       # Merged config
│   ├── git_commit.txt
│   └── command.txt
└── tivit/
    ├── configs/
    ├── train/
    └── logs/                       # Links to main logs (hardcoded path)
```

---

## 9. LOSS WEIGHT CONFIGURATION

**File**: [configs/train/single_run.yaml](configs/train/single_run.yaml) line 92

```yaml
loss:
  pos_weight_path: tivit/logs/pos_weights.json  # Relative to repo root
  head_weights:
    pitch: 1.0
    onset: 1.0
    offset: 1.0
    hand: 0.0
    clef: 0.0
```

**Generated by**: [preproc/threshold_priors.py](preproc/threshold_priors.py) line ~687

This JSON file contains per-class positive weights to balance class imbalance. Must exist before training or pass `pos_weight_path: null`.

---

## 10. MIGRATION CHECKLIST

### Before Migration

- [ ] **Identify data location**: Where will PianoVAM, PianoYT, OMAPS datasets be on new server?
- [ ] **Plan paths**: Decide on standardized paths:
  - `/data/PianoVAM_v1.0/` or `/mnt/datasets/PianoVAM/`?
  - `/data/HDF5/` for preprocessed files?
- [ ] **GPU availability**: Check CUDA availability on new server
- [ ] **Registration files**: Copy `reg_refined_pianovam.json`, `reg_refined.json` to new repo

### During Migration

1. **Set environment variables** (before running any scripts):
   ```bash
   export PIANOVAM_ROOT=/path/to/PianoVAM_v1.0
   export PIANOVAM_HDF5_ROOT=/path/to/HDF5
   export TIVIT_DATA_DIR=/data/datasets  # (optional, used as fallback)
   export DATASETS_HOME=/data  # (optional, final fallback)
   export TIVIT_LOG_DIR=/logs  # (optional, for custom log location)
   ```

2. **Verify dataset structure**:
   ```bash
   # Test with smoke script
   export PIANOVAM_ROOT=/path/to/PianoVAM_v1.0
   export PIANOVAM_HDF5_ROOT=/path/to/HDF5
   python scripts/smoke_print_batch_shapes.py \
     --config configs/pianovam_local.yaml \
     --mode hdf5 \
     --split train \
     --batch-size 1 \
     --num-workers 0
   ```

3. **Test GPU detection**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   print(f"Current device: {torch.cuda.current_device()}")
   ```

4. **Copy/regenerate loss weights** if needed:
   - Option A: Copy `logs/pos_weights.json` from old server
   - Option B: Regenerate via `preproc/threshold_priors.py` on new dataset sample

5. **Regenerate registration cache** if needed:
   ```bash
   python -m tivit.preproc.refine_registration_cache \
     configs/dataset/pianovam.yaml
   ```

### Configuration Overrides

Create a new config file for new server paths:

```yaml
# configs/overrides/new_server.yaml
dataset:
  root_dir: /path/to/PianoVAM_v1.0
  hdf5_root: /path/to/HDF5

logging:
  log_dir: /logs/tivit
  checkpoint_dir: /logs/tivit/checkpoints

training:
  amp: true      # Enable AMP if GPU available
```

Run with:
```bash
python -m tivit.pipelines.train_single \
  --config configs/default.yaml \
  --config configs/overrides/new_server.yaml
```

---

## 11. DEPENDENCY & INFRASTRUCTURE

### Python Dependencies
**No requirements.txt detected in provided structure.** Check parent project or look for:
- `pyproject.toml`
- `setup.py`
- `environment.yml` (conda)

### Optional Dependencies
From code inspection:
- `torch`, `torch.utils.data`
- `numpy`, `scipy`
- `yaml`, `json`
- `optionally: cv2` (OpenCV) - for registration
- `optionally: decord` - for video decoding
- `optionally: timm` - for ViT backbones

### No Docker/Containerization Found
Repository does not contain Dockerfile or docker-compose.yaml.

---

## 12. KEY FINDINGS

### ✅ Environment-Variable Driven
- All dataset paths resolved via env vars first, then config, then hardcoded fallbacks
- Easy to redirect data to new location without code changes

### ✅ Automatic GPU Detection
- No manual device selection needed
- Falls back to CPU gracefully
- CUDA determinism is configurable

### ✅ Reproducible Config Hierarchy
- All configs merged and saved to `logs/resolved_config.yaml` per run
- Git commit SHA recorded for full reproducibility

### ⚠️ Hardcoded Relative Paths
- `reg_refined_pianovam.json` in repo root (must be copied or regenerated)
- `tivit/logs/pos_weights.json` referenced from config (relative to repo)
- Fallback dataset paths use `~/datasets/`, not portable

### ⚠️ Checkpoint Detection
- Looks in `checkpoints/` subdirectory by default
- Latest epoch auto-detected if `best.pt` missing
- Can override with explicit `--checkpoint` arg

### ⚠️ HDF5 Mode Requires Two Paths
- `PIANOVAM_ROOT` - source MP4 videos
- `PIANOVAM_HDF5_ROOT` - preprocessed HDF5 files
- Preprocessing step converts MP4 → HDF5 for faster loading

---

## 13. TESTING & VALIDATION

### Smoke Tests Provided
- [tests/test_dataset_pianovam.py](tests/test_dataset_pianovam.py) - PianoVAM basic load
- [tests/test_dataset_pianoyt.py](tests/test_dataset_pianoyt.py) - PianoYT basic load
- [tests/test_dataset_omaps.py](tests/test_dataset_omaps.py) - OMAPS basic load
- [tests/test_dataset_real_canary.py](tests/test_dataset_real_canary.py) - Full canary with metadata check

Example run:
```bash
export PIANOVAM_ROOT=/path/to/PianoVAM_v1.0
python tests/test_dataset_pianovam.py
```

---

## Summary Table: Migration Points

| Component | Current | New Server | Notes |
|-----------|---------|-----------|-------|
| PianoVAM root | `${PIANOVAM_ROOT}` | Set env var | Required |
| PianoVAM HDF5 | `${PIANOVAM_HDF5_ROOT}` | Set env var | Optional, for preprocessed |
| Logs dir | `logs/` (repo) | Override in config | Can use `${TIVIT_LOG_DIR}` |
| Checkpoints | `checkpoints/` (repo) | Override in config | Auto-detected from latest |
| GPU device | Auto-detected | No change | Falls back to CPU |
| Seed/determinism | Config/CLI | No change | Fully configurable |
| Registration cache | `reg_refined_pianovam.json` | Copy or regenerate | In repo root |
| Loss weights | `logs/pos_weights.json` | Copy or regenerate | Can set to null |

---

Generated: 2026-04-07
Repository: tivit-vam
Analysis Type: Configuration & Path Infrastructure
