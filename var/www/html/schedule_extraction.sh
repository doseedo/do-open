#!/bin/bash
# Schedule extraction script to run in 1 hour 30 minutes

# Calculate delay (90 minutes = 5400 seconds)
DELAY_SECONDS=5400

echo "========================================"
echo "Scheduled Extraction Script"
echo "========================================"
echo "Script: /home/arlo/Data/extract_missing_pr_and_dcae.py"
echo "Will start in: 1 hour 30 minutes (90 minutes)"
echo "Scheduled time: $(date -d "+90 minutes" '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "Waiting ${DELAY_SECONDS} seconds..."
echo ""

# Wait for the specified time
sleep ${DELAY_SECONDS}

# Run the extraction script
echo "========================================"
echo "Starting extraction at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate ace_step

cd /home/arlo/Data
python extract_missing_pr_and_dcae.py 2>&1 | tee extraction_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "Extraction completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
