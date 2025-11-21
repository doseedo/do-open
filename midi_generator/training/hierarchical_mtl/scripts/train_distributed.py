#!/usr/bin/env python3
"""
Distributed Training Script for Hierarchical MTL.

Launch distributed training across multiple GPUs with PyTorch DDP.

Agent 5: Distributed Training Infrastructure
Date: November 21, 2025

Usage:
    # Single GPU
    python train_distributed.py --config config/training_config_default.yaml

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 train_distributed.py --config config/training_config_distributed.yaml

    # Multi-node (2 nodes, 4 GPUs each)
    # Node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=12355 train_distributed.py --config config.yaml
    # Node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=12355 train_distributed.py --config config.yaml
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from midi_generator.training.hierarchical_mtl.config.training_config import (
    HierarchicalMTLConfig
)
from midi_generator.training.hierarchical_mtl.loops.distributed_trainer import (
    DistributedHierarchicalMTLTrainer
)
from midi_generator.training.hierarchical_mtl.data.distributed_dataloader import (
    create_distributed_dataloaders
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed training for Hierarchical MTL"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Override labeled dataset path'
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default=None,
        help='Override features directory path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training (auto-set by torchrun)'
    )

    return parser.parse_args()


def setup_environment(config: HierarchicalMTLConfig, local_rank: int):
    """Setup training environment and reproducibility."""
    # Set random seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.benchmark
        torch.backends.cudnn.deterministic = config.deterministic

        # Set device
        torch.cuda.set_device(local_rank)

    # Environment variables for optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


def main():
    """Main training function."""
    args = parse_args()

    # Get distributed training info
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    is_main_process = (rank == 0)

    # Load configuration
    config = HierarchicalMTLConfig.load(Path(args.config))

    # Override config with command line args
    if args.data_path:
        config.data.labeled_dataset_path = Path(args.data_path)
    if args.features_dir:
        config.data.features_dir = Path(args.features_dir)
    if args.output_dir:
        config.checkpoint_dir = Path(args.output_dir) / 'checkpoints'
        config.log_dir = Path(args.output_dir) / 'logs'

    # Update distributed config
    config.distributed = (world_size > 1)
    config.world_size = world_size
    config.local_rank = local_rank

    # Setup environment
    setup_environment(config, local_rank)

    if is_main_process:
        print("=" * 80)
        print("HIERARCHICAL MTL DISTRIBUTED TRAINING")
        print("=" * 80)
        print(f"Configuration: {args.config}")
        print(f"World size: {world_size}")
        print(f"Local rank: {local_rank}")
        print(f"Effective batch size: {config.data.batch_size * world_size * getattr(config, 'accumulation_steps', 1)}")
        print(f"Mixed precision: {config.use_amp}")
        print("=" * 80)

    # Create model (Agent 4's architecture will be imported here)
    # For now, using a placeholder
    # TODO: Import actual ScaledHierarchicalMTL model from Agent 4
    from midi_generator.training.hierarchical_mtl.models.placeholder import create_model
    model = create_model(
        input_dim=600,  # Updated for Agent 2's enhanced features
        shared_dim=config.shared_encoder_dim,
        output_hierarchical=50,
        output_modular=120,
        output_rich=130
    )

    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create data loaders
    train_loader, val_loader, test_loader = create_distributed_dataloaders(
        labeled_dataset_path=config.data.labeled_dataset_path,
        features_dir=config.data.features_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        use_augmentation=config.data.use_augmentation,
        augmentation_prob=config.data.augmentation_prob,
        normalize=config.data.normalize_features,
        world_size=world_size,
        rank=rank
    )

    # Create trainer
    trainer = DistributedHierarchicalMTLTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        local_rank=local_rank,
        world_size=world_size
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    try:
        # Train
        results = trainer.train()

        if is_main_process:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)
            print(f"Best validation loss: {results['best_val_loss']:.6f}")
            print(f"Final epoch: {results['final_epoch']}")
            if results['test_metrics']:
                print(f"Test metrics: {results['test_metrics']}")
            print("=" * 80)

    except KeyboardInterrupt:
        if is_main_process:
            print("\nTraining interrupted by user")
            print("Saving checkpoint...")
            trainer.save_checkpoint(config.checkpoint_dir / 'interrupted.pt')

    except Exception as e:
        if is_main_process:
            print(f"\nTraining failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise

    finally:
        # Cleanup
        trainer.cleanup()


if __name__ == '__main__':
    main()
