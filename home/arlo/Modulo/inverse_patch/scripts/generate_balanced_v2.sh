#!/bin/bash
# Generate balanced inverse AFx training data v2
# This script:
# 1. Generates audio with balanced magnitude tiers (subtle/mild/moderate/strong)
# 2. Converts to DCAE latents
# 3. Uploads to GCS while keeping originals

set -e  # Exit on error

# Configuration
OUTPUT_DIR="/mnt/models/inverse_afx_v2"
GCS_DIR="/home/arlo/gcs-bucket/inverse_afx_training_data_v2"
MANIFEST="/home/arlo/gcs-bucket/Manifests/combined_manifest.json"
NUM_WORKERS=16
BATCH_SIZE=5000

echo "=============================================="
echo "Inverse AFx v2 - Balanced Data Generation"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo "GCS: $GCS_DIR"
echo ""

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ace_step

# Set pythonpath
export PYTHONPATH=/home/arlo/do-repo/home/arlo/Modulo/nablafx:$PYTHONPATH

# Step 1: Generate balanced audio data
echo ""
echo "=============================================="
echo "Step 1: Generating balanced audio data"
echo "=============================================="

python /home/arlo/do-repo/home/arlo/Modulo/nablafx/inverse_afx/scripts/generate_inverse_afx_data.py \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers $NUM_WORKERS \
    --balanced \
    --sample_rate 48000 \
    --segment_length 144000 \
    --max_chain_length 4

# Step 2: Convert to latents
echo ""
echo "=============================================="
echo "Step 2: Converting to DCAE latents"
echo "=============================================="

# Create latent output dirs
mkdir -p "$OUTPUT_DIR/latent_pairs"
mkdir -p "$GCS_DIR/latent_pairs"

python /home/arlo/do-repo/home/arlo/Modulo/nablafx/inverse_afx/scripts/precompute_latents.py \
    --manifest "$OUTPUT_DIR/manifest.json" \
    --output-dir "$OUTPUT_DIR/latent_pairs" \
    --gcs-dir "$GCS_DIR/latent_pairs" \
    --batch-size $BATCH_SIZE

# Step 3: Copy audio and manifests to GCS
echo ""
echo "=============================================="
echo "Step 3: Uploading to GCS"
echo "=============================================="

# Copy dry/wet audio
echo "Copying dry audio..."
mkdir -p "$GCS_DIR/dry"
cp -r "$OUTPUT_DIR/dry/"* "$GCS_DIR/dry/" 2>/dev/null || true

echo "Copying wet audio..."
mkdir -p "$GCS_DIR/wet"
cp -r "$OUTPUT_DIR/wet/"* "$GCS_DIR/wet/" 2>/dev/null || true

# Copy manifests
echo "Copying manifests..."
cp "$OUTPUT_DIR/manifest.json" "$GCS_DIR/manifest_audio.json"
cp "$OUTPUT_DIR/generation_config.json" "$GCS_DIR/generation_config.json" 2>/dev/null || true

# Final latent manifest should already be at GCS from precompute_latents.py

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo "Audio data: $GCS_DIR/dry/, $GCS_DIR/wet/"
echo "Latent pairs: $GCS_DIR/latent_pairs/"
echo "Audio manifest: $GCS_DIR/manifest_audio.json"
echo "Latent manifest: $GCS_DIR/manifest_latent.json"
echo ""
echo "Original v1 data preserved at:"
echo "  /home/arlo/gcs-bucket/inverse_afx_training_data/"
