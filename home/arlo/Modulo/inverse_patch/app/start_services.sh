#!/bin/bash
# Start Sparse Synthesizer services

cd /home/arlo/do-repo/home/arlo/Modulo/inverse_patch/app

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ace_step

# Start FastAPI backend (port 8097)
echo "Starting FastAPI backend on port 8097..."
nohup python sparse_synth_api.py > /tmp/sparse_api.log 2>&1 &
echo "FastAPI PID: $!"

# Wait for API to be ready
sleep 5

# Start Gradio UI (port 8098)
echo "Starting Gradio UI on port 8098..."
nohup python gradio_ui.py > /tmp/gradio_ui.log 2>&1 &
echo "Gradio PID: $!"

echo ""
echo "Services starting..."
echo "  API:    http://localhost:8097"
echo "  Gradio: http://localhost:8098"
echo "  Public: https://doseedo.com/do"
echo ""
echo "Logs:"
echo "  tail -f /tmp/sparse_api.log"
echo "  tail -f /tmp/gradio_ui.log"
