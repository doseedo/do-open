#!/bin/bash
# Monitor GPU utilization during discovery

echo "GPU-Optimized Discovery Monitor"
echo "================================"
echo ""
echo "Process status:"
ps aux | grep -E "python.*start_discovery" | grep -v grep
echo ""
echo "GPU utilization:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
echo ""
echo "Latest log output:"
tail -50 /home/arlo/do-repo/midi_generator/discovery_full.log
