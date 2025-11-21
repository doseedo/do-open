#!/bin/bash
# Launch script for single-GPU training
# Agent 5: Distributed Training Infrastructure
# Date: November 21, 2025

set -e  # Exit on error

# Configuration
CONFIG_FILE="${1:-config/training_config_default.yaml}"
DATA_PATH="${2:-labeled_dataset_comprehensive.json}"
FEATURES_DIR="${3:-features/}"
OUTPUT_DIR="${4:-outputs/single_gpu_training}"
GPU_ID="${5:-0}"

echo "=================================================="
echo "Launching Single-GPU Training"
echo "=================================================="
echo "Config: $CONFIG_FILE"
echo "GPU ID: $GPU_ID"
echo "Data path: $DATA_PATH"
echo "Features dir: $FEATURES_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "=================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Launch training
python midi_generator/training/hierarchical_mtl/scripts/train_distributed.py \
    --config "$CONFIG_FILE" \
    --data-path "$DATA_PATH" \
    --features-dir "$FEATURES_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "=================================================="
echo "Training Complete!"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints"
echo "Logs saved to: $OUTPUT_DIR/logs"
echo "=================================================="
