#!/bin/bash

# ACE-Step Lyrics Processing with Conda Environment
# This script ensures we're running in the ace_step conda environment

echo "🔧 Setting up ACE-Step environment..."

# Source conda
source /home/arlo/miniconda3/etc/profile.d/conda.sh

# Activate ace_step environment
conda activate ace_step

echo "📦 Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "🐍 Python: $(which python)"

# Run the lyrics processing script with all arguments passed through
echo "🎵 Starting ACE-Step lyrics processing..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/arlo/Data/lyrics.py "$@"