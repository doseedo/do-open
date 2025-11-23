#!/usr/bin/env python3
"""
Simple CSV Logger - Works Without TensorBoard

Drop-in replacement for TensorBoard when you can't install it.
Logs metrics to CSV file that can be plotted later.

Usage:
    from simple_csv_logger import SimpleLogger

    logger = SimpleLogger('training_metrics.csv')

    for epoch in range(100):
        train_loss = train()
        val_loss = validate()

        logger.log(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=current_lr
        )

    logger.close()

    # Then plot with: python simple_csv_logger.py --plot training_metrics.csv
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class SimpleLogger:
    """CSV logger that works without TensorBoard"""

    def __init__(self, log_file: str, metadata: Optional[Dict] = None):
        """
        Initialize logger.

        Args:
            log_file: Path to CSV log file
            metadata: Optional metadata (hyperparameters, etc.)
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.metrics = []
        self.metadata = metadata or {}
        self.metadata['created_at'] = datetime.now().isoformat()

        # Save metadata
        metadata_file = self.log_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✅ SimpleLogger initialized")
        print(f"   Log file: {self.log_file}")
        print(f"   Metadata: {metadata_file}")

    def log(self, step: int, **kwargs):
        """
        Log metrics for a step/epoch.

        Args:
            step: Training step or epoch number
            **kwargs: Metric name-value pairs
        """
        entry = {'step': step, **kwargs}
        self.metrics.append(entry)

        # Write to CSV (overwrites each time for real-time viewing)
        with open(self.log_file, 'w', newline='') as f:
            if self.metrics:
                writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.metrics)

    def close(self):
        """Close logger and print summary"""
        print(f"\n✅ Metrics saved to {self.log_file}")
        print(f"   Total steps logged: {len(self.metrics)}")

        if self.metrics:
            last_entry = self.metrics[-1]
            print(f"\n   Final metrics:")
            for key, value in last_entry.items():
                if key != 'step':
                    print(f"     {key}: {value}")

        print(f"\n   To plot: python {__file__} --plot {self.log_file}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def plot_metrics(csv_file: Path):
    """Plot metrics from CSV file"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("❌ matplotlib and pandas required for plotting")
        print("   Install with: pip install matplotlib pandas")
        return

    # Load data
    df = pd.read_csv(csv_file)

    # Get metric columns (all except 'step')
    metric_cols = [col for col in df.columns if col != 'step']

    # Create subplots
    n_metrics = len(metric_cols)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each metric
    for idx, col in enumerate(metric_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx]

        ax.plot(df['step'], df[col], linewidth=2)
        ax.set_xlabel('Step/Epoch')
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        axes[row, col_idx].set_visible(False)

    plt.tight_layout()

    # Save plot
    plot_file = csv_file.with_suffix('.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to {plot_file}")

    # Show plot
    plt.show()


# ============================================================================
# Example Usage
# ============================================================================

def example():
    """Example usage of SimpleLogger"""
    import numpy as np

    print("="*70)
    print("SimpleLogger Example")
    print("="*70)

    # Create logger with metadata
    logger = SimpleLogger(
        'example_training.csv',
        metadata={
            'model': 'harmony_encoder',
            'hidden_dim': 1024,
            'learning_rate': 0.01,
            'batch_size': 32
        }
    )

    # Simulate training
    print("\nSimulating training...")
    for epoch in range(50):
        # Simulate decreasing losses
        train_loss = 1000 / (epoch + 1) + np.random.randn() * 10
        val_loss = 1200 / (epoch + 1) + np.random.randn() * 12
        lr = 0.01 * (0.95 ** epoch)

        # Log metrics
        logger.log(
            step=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=lr,
            sparsity=np.random.rand()
        )

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:8.2f}, val_loss={val_loss:8.2f}")

    # Close logger
    logger.close()

    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)

    # Try to plot
    try:
        plot_metrics(Path('example_training.csv'))
    except Exception as e:
        print(f"\n⚠️  Could not plot: {e}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--plot':
        if len(sys.argv) < 3:
            print("Usage: python simple_csv_logger.py --plot <csv_file>")
            sys.exit(1)
        plot_metrics(Path(sys.argv[2]))
    else:
        example()
