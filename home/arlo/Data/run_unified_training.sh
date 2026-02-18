#!/bin/bash
# Run unified self-improving classifier training overnight
#
# Usage:
#   ./run_unified_training.sh           # Full pipeline (train + refine + eval)
#   ./run_unified_training.sh train     # Just training
#   ./run_unified_training.sh refine    # Just refinement
#   ./run_unified_training.sh eval      # Just evaluation

set -e

SCRIPT_DIR="/home/arlo/Data"
OUTPUT_DIR="/home/arlo/Data/unified_classifier"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/run_${TIMESTAMP}.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate conda environment (ace_step has all dependencies)
eval "$(conda shell.bash hook)"
conda activate ace_step

# Get mode from argument or default to 'full'
MODE="${1:-full}"

echo "=============================================="
echo "Unified Self-Improving Classifier"
echo "=============================================="
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Started: $(date)"
echo "=============================================="

# Run training
cd "$SCRIPT_DIR"

# Check if running interactively or in background
if [ -t 0 ]; then
    # Interactive - show output
    python3 unified_self_improving_classifier.py \
        --mode "$MODE" \
        --epochs 30 \
        --rounds 5 \
        --batch-size 64 \
        --device cuda \
        2>&1 | tee "$LOG_FILE"
else
    # Background/nohup - just log
    python3 unified_self_improving_classifier.py \
        --mode "$MODE" \
        --epochs 30 \
        --rounds 5 \
        --batch-size 64 \
        --device cuda \
        >> "$LOG_FILE" 2>&1
fi

echo "=============================================="
echo "Completed: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=============================================="
