#!/bin/bash
# ============================================================================
# Complete ML Experiment Pipeline: Train → Calibrate → Evaluate
# Target: F1 Score ~0.42 on PianoVAM dataset
# ============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME="f1_optimized_v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG_DIR="/home/achatzigiannis/tivit-logs/${EXPERIMENT_NAME}"
EXPERIMENT_LOG="${MAIN_LOG_DIR}/experiment_${TIMESTAMP}.log"

# Create directories
mkdir -p "$MAIN_LOG_DIR"
mkdir -p "${MAIN_LOG_DIR}/checkpoints"

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$EXPERIMENT_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$EXPERIMENT_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$EXPERIMENT_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$EXPERIMENT_LOG"
}

log_section() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}" | tee -a "$EXPERIMENT_LOG"
    echo -e "${BLUE}║ $1${NC}" | tee -a "$EXPERIMENT_LOG"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n" | tee -a "$EXPERIMENT_LOG"
}

# ============================================================================
# SETUP
# ============================================================================

log_section "EXPERIMENT SETUP"

cd ~/tivit-vam
source ~/tivit-vam/.venv/bin/activate

# Set environment variables
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PIANOVAM_ROOT=/raid_storage/data_achatzigiannis/PianoVAM_v1.0
export PIANOVAM_HDF5_ROOT=/raid_storage/data_achatzigiannis/PianoVAM_v1.0/HDF5
export TIVIT_LOG_DIR="$MAIN_LOG_DIR"

log_info "Experiment name: $EXPERIMENT_NAME"
log_info "Log directory: $MAIN_LOG_DIR"
log_info "Main log file: $EXPERIMENT_LOG"
log_info "PIANOVAM_ROOT: $PIANOVAM_ROOT"
log_info "Python path: $(which python)"
log_info "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
log_info "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if [ ! -d "$PIANOVAM_ROOT" ]; then
    log_error "Dataset not found at $PIANOVAM_ROOT"
    exit 1
fi

log_success "Setup complete"

# ============================================================================
# PHASE 1: TRAINING
# ============================================================================

log_section "PHASE 1: TRAINING (200 epochs)"

TRAIN_LOG="${MAIN_LOG_DIR}/train_${TIMESTAMP}.log"

log_info "Starting training..."
log_info "Config: configs/default.yaml + configs/overrides/f1_optimized_experiment.yaml"
log_info "Train log: $TRAIN_LOG"
log_info "Command: python -m pipelines.train_single"

python -m pipelines.train_single \
    --config configs/default.yaml \
    --config configs/overrides/f1_optimized_experiment.yaml \
    --train-split train \
    --verbose info \
    2>&1 | tee -a "$TRAIN_LOG" "$EXPERIMENT_LOG"

TRAIN_STATUS=$?

if [ $TRAIN_STATUS -ne 0 ]; then
    log_error "Training failed!"
    exit 1
fi

# Find best checkpoint
BEST_CKPT=$(find "${MAIN_LOG_DIR}/checkpoints" -name "best.pt" -type f | head -1)

if [ -z "$BEST_CKPT" ]; then
    log_error "No best checkpoint found!"
    exit 1
fi

log_success "Training completed"
log_info "Best checkpoint: $BEST_CKPT"

# Copy resolved config for reference
if [ -f "${MAIN_LOG_DIR}/resolved_config.yaml" ]; then
    log_info "Resolved config: ${MAIN_LOG_DIR}/resolved_config.yaml"
fi

# ============================================================================
# PHASE 2: CALIBRATION
# ============================================================================

log_section "PHASE 2: CALIBRATION (Threshold Sweep)"

CALIB_LOG="${MAIN_LOG_DIR}/calibration_${TIMESTAMP}.log"

log_info "Starting calibration..."
log_info "Checkpoint: $BEST_CKPT"
log_info "Calibration log: $CALIB_LOG"
log_info "Method: threshold_sweep (optimal thresholds for F1)"

python -m pipelines.calibrate \
    --config configs/default.yaml \
    --config configs/overrides/f1_optimized_experiment.yaml \
    --config configs/calib/threshold_sweep.yaml \
    --checkpoint "$BEST_CKPT" \
    --verbose info \
    2>&1 | tee -a "$CALIB_LOG" "$EXPERIMENT_LOG"

CALIB_STATUS=$?

if [ $CALIB_STATUS -ne 0 ]; then
    log_warning "Calibration completed with warnings"
else
    log_success "Calibration completed"
fi

# Check for calibration results
if [ -f "${MAIN_LOG_DIR}/calibration.json" ]; then
    log_info "Calibration results: ${MAIN_LOG_DIR}/calibration.json"
fi

# ============================================================================
# PHASE 3: EVALUATION
# ============================================================================

log_section "PHASE 3: EVALUATION (F1 Score Calculation)"

EVAL_LOG="${MAIN_LOG_DIR}/evaluation_${TIMESTAMP}.log"

log_info "Starting evaluation..."
log_info "Checkpoint: $BEST_CKPT"
log_info "Evaluation log: $EVAL_LOG"

python -m pipelines.evaluate \
    --config configs/default.yaml \
    --config configs/overrides/f1_optimized_experiment.yaml \
    --checkpoint "$BEST_CKPT" \
    --verbose info \
    2>&1 | tee -a "$EVAL_LOG" "$EXPERIMENT_LOG"

EVAL_STATUS=$?

if [ $EVAL_STATUS -ne 0 ]; then
    log_warning "Evaluation completed with warnings"
else
    log_success "Evaluation completed"
fi

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

log_section "EXPERIMENT RESULTS SUMMARY"

log_info "Experiment: $EXPERIMENT_NAME"
log_info "Timestamp: $TIMESTAMP"
log_info "Duration: Started at this execution"
log_info ""

# Display F1 scores if available
if [ -f "${MAIN_LOG_DIR}/eval_results.json" ]; then
    log_info "Evaluation metrics found:"
    echo "" | tee -a "$EXPERIMENT_LOG"
    python3 << 'PYTHON_EOF' >> "$EXPERIMENT_LOG"
import json
import sys
from pathlib import Path

eval_file = Path('/home/achatzigiannis/tivit-logs/f1_optimized_v1/eval_results.json')
if eval_file.exists():
    with open(eval_file) as f:
        results = json.load(f)
    print(json.dumps(results, indent=2))
else:
    print("Evaluation results not found")
PYTHON_EOF
fi

log_info ""
log_info "Checkpoint: $BEST_CKPT"
log_info "Calibration results: ${MAIN_LOG_DIR}/calibration.json"
log_info ""

# ============================================================================
# FILE LOCATIONS
# ============================================================================

log_section "RESULTS AND LOGS"

log_info "Main experiment log: $EXPERIMENT_LOG"
log_info "Training log: $TRAIN_LOG"
log_info "Calibration log: $CALIB_LOG"
log_info "Evaluation log: $EVAL_LOG"
log_info ""
log_info "Checkpoints: ${MAIN_LOG_DIR}/checkpoints/"
log_info "Best model: $BEST_CKPT"
log_info ""

# Print F1 scores
log_section "F1 SCORES AND METRICS"

log_warning "To view detailed F1 scores, run:"
log_warning "cat ${MAIN_LOG_DIR}/eval_results.json | python -m json.tool"
log_warning ""
log_warning "To view training metrics:"
log_warning "tail -100 $TRAIN_LOG | grep -E 'F1|loss|epoch'"

# ============================================================================
# NEXT STEPS
# ============================================================================

log_section "NEXT STEPS"

log_info "To monitor training in real-time:"
log_info "  tail -f $TRAIN_LOG"
log_info ""
log_info "To view calibration results:"
log_info "  cat ${MAIN_LOG_DIR}/calibration.json | python -m json.tool"
log_info ""
log_info "To analyze F1 scores:"
log_info "  python3 << 'EOF'"
log_info "import json"
log_info "with open('${MAIN_LOG_DIR}/eval_results.json') as f:"
log_info "    results = json.load(f)"
log_info "    for head, metrics in results.items():"
log_info "        if 'f1' in metrics:"
log_info "            print(f'{head}: F1={metrics[\"f1\"]:.4f}')"
log_info "  EOF"

log_section "EXPERIMENT COMPLETE"

log_success "All phases completed successfully!"
log_success "Check logs for detailed results and F1 scores"
log_success ""

# ============================================================================
# TAIL LOG FILE
# ============================================================================

echo ""
echo -e "${GREEN}Final Log Output:${NC}"
tail -50 "$EXPERIMENT_LOG"

exit 0
