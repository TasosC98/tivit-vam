# TiViT-VAM (CPU) — Video-based Piano Transcription on PianoVAM (HDF5)

This repository is a CPU-friendly fork/workspace derived from a TiViT/ViViT-style
video transformer pipeline, adapted to run experiments on the PianoVAM dataset
using preprocessed HDF5 video tensors.

It supports:
- Loading PianoVAM from raw videos **or** from HDF5 caches
- A training pipeline (`tivit.pipelines.train_single`)
- (Optional) evaluation pipelines (`tivit.pipelines.evaluate`, `tivit.pipelines.eval_single`)
- Registration/refinement debug artefacts (saved under `logs/reg_debug/`)

> **Important:** This server is CPU-only. Use CPU PyTorch wheels.

---

## 1. Repository layout (high-level)

Top-level folders you will interact with most:

- `configs/`  
  YAML experiment configs. Example used in this project:
  `configs/experiment/pianovam_hdf5_baseline.yaml`

- `data/`  
  Dataset + decoding code.
  - `data/datasets/pianovam*.py` : PianoVAM dataset logic
  - `data/decode/` : video/HDF5 decoding readers

- `scripts/`  
  Utility scripts (smoke tests, conversion).
  - `scripts/smoke_print_batch_shapes.py` : verify dataloader tensors
  - `scripts/convert_pianovam_all.py` : (recommended) batch conversion to HDF5
  - `scripts/convert_videos_to_hdf5.py` : convert a **single** video to HDF5

- `pipelines/`  
  CLI entrypoints runnable via `python -m tivit.pipelines.<name>`
  - `pipelines/train_single.py` : training entrypoint
  - `pipelines/evaluate.py` / `pipelines/eval_single.py` : evaluation entrypoints (if enabled)

- `tivit/`  
  The main python package namespace used by the pipelines:
  - `tivit/losses/`, `tivit/core/`, `tivit/stage/`, etc.

- `train/`  
  Training loop implementation (checkpoints/logging live here).

- `logs/`  
  Outputs from runs:
  - `logs/<experiment_name>/train.log`
  - `logs/<experiment_name>/checkpoints/epoch_*.pt`
  - `logs/<experiment_name>/resolved_config.yaml`
  - `logs/<experiment_name>/command.txt`

---

## 2. Installation (CPU)
## Required local modules (postproc + theory)

Some evaluation/decoding utilities import:
- `tivit.postproc.*`
- `theory.*` (Key-aware prior runtime)

In this repo layout:
- `postproc/` exists at repo root (top-level)
- `theory/` must exist at repo root (top-level) and provide:
  `KeyAwarePrior`, `key_prior.py`, `key_prior_runtime.py`

If `tivit/postproc` is missing inside the `tivit` package namespace, create a symlink:

```bash
cd ~/dev/tivit-vam
ln -s ../postproc tivit/postproc 2>/dev/null || true
test -e tivit/postproc && echo "OK: tivit/postproc exists"

### 2.1 Create venv

```bash
cd ~/dev/tivit-vam
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel


###Smoke
**Smoke (HDF5, valid split):**
```bash
PYTHONPATH=. python scripts/smoke_print_batch_shapes.py \
  --config configs/experiment/pianovam_hdf5_baseline.yaml \
  --mode hdf5 \
  --split valid \
  --batch-size 1 \
  --num-workers 0 \
  --hdf5-root "$PIANOVAM_HDF5_ROOT"

###Train
PYTHONPATH=. python -m tivit.pipelines.train_single \
  --config configs/experiment/pianovam_hdf5_baseline.yaml \
  --frames 16 \
  --max-clips 8 \
  --verbose info

###Eval
PYTHONPATH=. python -m tivit.pipelines.eval_single \
  --config configs/experiment/pianovam_hdf5_baseline.yaml \
  --checkpoint logs/pianovam_hdf5_baseline/checkpoints/epoch_2.pt \
  --split valid \
  --max-clips 2 \
  --verbose info

## Where results are stored

After training, outputs land in:
- `logs/<experiment_name>/train.log`
- `logs/<experiment_name>/checkpoints/epoch_*.pt`
- `logs/<experiment_name>/resolved_config.yaml`
- `logs/<experiment_name>/command.txt`

Example:
- `logs/pianovam_hdf5_baseline/train.log`
- `logs/pianovam_hdf5_baseline/checkpoints/epoch_2.pt`

Quick grep for epoch losses:
```bash
grep -n "metrics={'loss'" logs/pianovam_hdf5_baseline/train.log


python scripts/convert_pianovam_all.py \
  --root "$PIANOVAM_ROOT" \
  --out "$PIANOVAM_HDF5_ROOT"
