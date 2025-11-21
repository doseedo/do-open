#!/usr/bin/env python3
"""
Resume Training Script for Hierarchical MTL.

Automatically resumes training from the latest or best checkpoint.

Agent 5: Distributed Training Infrastructure
Date: November 21, 2025

Usage:
    # Resume from latest checkpoint
    python resume_training.py --checkpoint-dir checkpoints/hierarchical_mtl

    # Resume from specific checkpoint
    python resume_training.py --checkpoint checkpoints/hierarchical_mtl/checkpoint_epoch_50.pt

    # Resume best model
    python resume_training.py --checkpoint-dir checkpoints/hierarchical_mtl --use-best

    # Multi-GPU resume
    torchrun --nproc_per_node=4 resume_training.py --checkpoint-dir checkpoints/hierarchical_mtl
"""

import argparse
import json
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the latest checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Look for checkpoint metadata
    metadata_file = checkpoint_dir / 'checkpoint_metadata.json'

    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Get latest checkpoint from metadata
        if metadata.get('saved_checkpoints'):
            # Sort by epoch and get latest
            checkpoints = sorted(
                metadata['saved_checkpoints'],
                key=lambda x: x['epoch'],
                reverse=True
            )
            latest_path = Path(checkpoints[0]['path'])

            if latest_path.exists():
                return latest_path

    # Fallback: search for checkpoint files
    checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Sort by epoch number
    def extract_epoch(path):
        try:
            return int(path.stem.split('_')[-1])
        except:
            return -1

    checkpoint_files.sort(key=extract_epoch, reverse=True)
    return checkpoint_files[0]


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the best checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    best_checkpoint = checkpoint_dir / 'best_model.pt'

    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint not found in {checkpoint_dir}")

    return best_checkpoint


def load_config_from_checkpoint(checkpoint_path: Path) -> dict:
    """Load training configuration from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain configuration")

    return checkpoint['config']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Resume training from checkpoint"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--checkpoint',
        type=str,
        help='Path to specific checkpoint file'
    )
    group.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory containing checkpoints (will use latest)'
    )

    parser.add_argument(
        '--use-best',
        action='store_true',
        help='Use best checkpoint instead of latest (requires --checkpoint-dir)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Override with new config file (optional)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of additional epochs to train'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override learning rate'
    )

    return parser.parse_args()


def main():
    """Main resume function."""
    args = parse_args()

    print("=" * 80)
    print("RESUME TRAINING")
    print("=" * 80)

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        if args.use_best:
            checkpoint_path = find_best_checkpoint(checkpoint_dir)
            print(f"Using best checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            print(f"Using latest checkpoint: {checkpoint_path}")

    # Load checkpoint info
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best score: {checkpoint.get('best_score', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")

    # Load or use checkpoint config
    if args.config:
        print(f"\nUsing config from: {args.config}")
        config_path = args.config
    else:
        # Extract config from checkpoint
        print(f"\nUsing config from checkpoint")
        config_dict = checkpoint.get('config')

        if config_dict is None:
            raise ValueError("Checkpoint does not contain config and no --config provided")

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f, indent=2)
            config_path = f.name

    # Override config parameters if specified
    if args.epochs or args.learning_rate:
        from midi_generator.training.hierarchical_mtl.config.training_config import (
            HierarchicalMTLConfig
        )

        config = HierarchicalMTLConfig.load(Path(config_path))

        if args.epochs:
            print(f"Overriding epochs: {config.num_epochs} -> {args.epochs}")
            config.num_epochs = args.epochs

        if args.learning_rate:
            print(f"Overriding learning rate: {config.optimizer.learning_rate} -> {args.learning_rate}")
            config.optimizer.learning_rate = args.learning_rate

        # Save modified config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(Path(f.name))
            config_path = f.name

    print("=" * 80)
    print("Launching training with resume...")
    print("=" * 80)

    # Import and call train_distributed with resume
    from midi_generator.training.hierarchical_mtl.scripts.train_distributed import (
        main as train_main
    )

    # Monkey patch sys.argv for train_distributed
    sys.argv = [
        'train_distributed.py',
        '--config', config_path,
        '--resume', str(checkpoint_path)
    ]

    train_main()


if __name__ == '__main__':
    main()
