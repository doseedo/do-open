#!/bin/bash
# Launch script for distributed training across multiple GPUs
# Agent 5: Distributed Training Infrastructure
# Date: November 21, 2025

set -e  # Exit on error

# Configuration
CONFIG_FILE="${1:-config/training_config_distributed.yaml}"
NUM_GPUS="${2:-4}"
DATA_PATH="${3:-labeled_dataset_comprehensive.json}"
FEATURES_DIR="${4:-features/}"
OUTPUT_DIR="${5:-outputs/distributed_training}"

echo "=================================================="
echo "Launching Distributed Training"
echo "=================================================="
echo "Config: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Data path: $DATA_PATH"
echo "Features dir: $FEATURES_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "=================================================="

# Check if GPUs are available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Are you on a GPU machine?"
    exit 1
fi

# Check number of available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ $NUM_GPUS -gt $AVAILABLE_GPUS ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    echo "Using $AVAILABLE_GPUS GPUs instead"
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables for optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=0

# Launch training with torchrun
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    midi_generator/training/hierarchical_mtl/scripts/train_distributed.py \
    --config "$CONFIG_FILE" \
    --data-path "$DATA_PATH" \
    --features-dir "$FEATURES_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "=================================================="
echo "Training Complete!"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints"
echo "Logs saved to: $OUTPUT_DIR/logs"
echo "=================================================="
