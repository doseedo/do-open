#!/usr/bin/env python3
"""
Real-time Training Monitoring Dashboard.

Monitors training progress with TensorBoard and provides CLI dashboard.

Agent 5: Distributed Training Infrastructure
Date: November 21, 2025

Usage:
    # Start monitoring
    python monitor_dashboard.py --log-dir logs/hierarchical_mtl

    # Start TensorBoard server
    python monitor_dashboard.py --log-dir logs/hierarchical_mtl --tensorboard

    # Watch checkpoint directory
    python monitor_dashboard.py --checkpoint-dir checkpoints/hierarchical_mtl
"""

import argparse
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys


class TrainingMonitor:
    """Monitor training progress from logs and checkpoints."""

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        self.log_dir = Path(log_dir) if log_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def get_latest_metrics(self) -> Dict:
        """Get latest training metrics from checkpoint metadata."""
        if not self.checkpoint_dir:
            return {}

        metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'

        if not metadata_file.exists():
            return {}

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return {}

    def get_checkpoint_history(self) -> List[Dict]:
        """Get history of all checkpoints."""
        if not self.checkpoint_dir:
            return []

        metadata = self.get_latest_metrics()
        return metadata.get('saved_checkpoints', [])

    def get_tensorboard_metrics(self) -> Dict:
        """Parse TensorBoard event files for metrics."""
        # This would parse TensorBoard event files
        # Simplified version for now
        return {}

    def display_dashboard(self):
        """Display real-time dashboard in terminal."""
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H")  # ANSI escape codes

                print("=" * 80)
                print("HIERARCHICAL MTL TRAINING MONITOR")
                print("=" * 80)
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)

                # Get latest metrics
                metadata = self.get_latest_metrics()

                if metadata:
                    print("\nLatest Metrics:")
                    print(f"  Epoch: {metadata.get('last_epoch', 'N/A')}")
                    print(f"  Best Score: {metadata.get('best_score', 'N/A'):.6f}")
                    print(f"  Monitor: {metadata.get('monitor', 'N/A')}")

                    latest_metrics = metadata.get('latest_metrics', {})
                    if latest_metrics:
                        print("\n  Current Metrics:")
                        for key, value in latest_metrics.items():
                            if isinstance(value, float):
                                print(f"    {key}: {value:.6f}")
                            else:
                                print(f"    {key}: {value}")

                # Checkpoint history
                checkpoints = self.get_checkpoint_history()
                if checkpoints:
                    print(f"\nSaved Checkpoints ({len(checkpoints)}):")
                    for ckpt in checkpoints[-5:]:  # Last 5
                        print(f"  Epoch {ckpt['epoch']}: score={ckpt['score']:.6f}")

                # GPU usage (if available)
                try:
                    gpu_info = self._get_gpu_info()
                    if gpu_info:
                        print("\nGPU Usage:")
                        print(gpu_info)
                except:
                    pass

                print("\n" + "=" * 80)
                print("Press Ctrl+C to exit")
                print("=" * 80)

                # Refresh every 5 seconds
                time.sleep(5)

        except KeyboardInterrupt:
            print("\nMonitoring stopped")

    def _get_gpu_info(self) -> str:
        """Get GPU usage information."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) == 5:
                        idx, name, util, mem_used, mem_total = parts
                        gpu_info.append(
                            f"  GPU {idx} ({name}): {util}% util, {mem_used}MB / {mem_total}MB"
                        )
                return '\n'.join(gpu_info)
        except:
            pass

        return ""

    def start_tensorboard(self, port: int = 6006):
        """Start TensorBoard server."""
        if not self.log_dir:
            print("Error: log_dir not specified")
            return

        print(f"Starting TensorBoard on port {port}...")
        print(f"Log directory: {self.log_dir}")
        print(f"\nOpen in browser: http://localhost:{port}")
        print("\nPress Ctrl+C to stop")

        try:
            subprocess.run(
                ['tensorboard', '--logdir', str(self.log_dir), '--port', str(port)],
                check=True
            )
        except KeyboardInterrupt:
            print("\nTensorBoard stopped")
        except subprocess.CalledProcessError as e:
            print(f"Error starting TensorBoard: {e}")
        except FileNotFoundError:
            print("Error: TensorBoard not installed. Install with: pip install tensorboard")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor training progress"
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Path to log directory'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Start TensorBoard server'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6006,
        help='TensorBoard port (default: 6006)'
    )
    parser.add_argument(
        '--refresh',
        type=int,
        default=5,
        help='Dashboard refresh interval in seconds (default: 5)'
    )

    return parser.parse_args()


def main():
    """Main monitoring function."""
    args = parse_args()

    monitor = TrainingMonitor(
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )

    if args.tensorboard:
        # Start TensorBoard
        monitor.start_tensorboard(port=args.port)
    else:
        # Display dashboard
        monitor.display_dashboard()


if __name__ == '__main__':
    main()
