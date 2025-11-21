"""
Distributed Data Loading Utilities for Hierarchical MTL.

Provides DistributedSampler and optimized data loaders for multi-GPU training.

Agent 5: Distributed Training Infrastructure
Date: November 21, 2025
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from typing import Optional, Tuple

from midi_generator.training.hierarchical_mtl.data.dataset import (
    HierarchicalMIDIDataset,
    DataAugmenter
)


def create_distributed_dataloaders(
    labeled_dataset_path: Path,
    features_dir: Optional[Path] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    use_augmentation: bool = True,
    augmentation_prob: float = 0.3,
    normalize: bool = True,
    world_size: int = 1,
    rank: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create distributed data loaders for multi-GPU training.

    Args:
        labeled_dataset_path: Path to labeled dataset JSON
        features_dir: Directory with pre-extracted features
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers per GPU
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch
        use_augmentation: Whether to use data augmentation (train only)
        augmentation_prob: Probability of applying augmentation
        normalize: Whether to normalize features
        world_size: Total number of processes
        rank: Current process rank

    Returns:
        (train_loader, val_loader, test_loader)
    """

    # Create augmenter
    augmenter = DataAugmenter(prob=augmentation_prob) if use_augmentation else None

    # Create train dataset and compute normalization stats
    train_dataset = HierarchicalMIDIDataset(
        labeled_dataset_path=labeled_dataset_path,
        features_dir=features_dir,
        split="train",
        transform=augmenter,
        normalize=normalize
    )

    # Get normalization stats from train
    norm_stats = train_dataset.normalization_stats if normalize else None

    # Create val and test datasets
    val_dataset = HierarchicalMIDIDataset(
        labeled_dataset_path=labeled_dataset_path,
        features_dir=features_dir,
        split="val",
        transform=None,  # No augmentation for validation
        normalize=normalize,
        normalization_stats=norm_stats
    )

    test_dataset = HierarchicalMIDIDataset(
        labeled_dataset_path=labeled_dataset_path,
        features_dir=features_dir,
        split="test",
        transform=None,  # No augmentation for test
        normalize=normalize,
        normalization_stats=norm_stats
    )

    # Create distributed samplers (if distributed training)
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )

        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )

        shuffle_train = False  # Sampler handles shuffling

        if rank == 0:
            print(f"Using DistributedSampler across {world_size} processes")

    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # Drop last incomplete batch for consistent training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )

    if rank == 0:
        print(f"Created distributed dataloaders:")
        print(f"  Train: {len(train_dataset)} samples (per process: {len(train_dataset) // world_size})")
        print(f"  Val:   {len(val_dataset)} samples (per process: {len(val_dataset) // world_size})")
        print(f"  Test:  {len(test_dataset)} samples (per process: {len(test_dataset) // world_size})")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Total effective batch: {batch_size * world_size}")

    return train_loader, val_loader, test_loader


class PrefetchDataLoader:
    """
    Data loader with GPU prefetching for improved throughput.

    Prefetches the next batch to GPU while the model is processing the current batch.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device
    ):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None

    def __iter__(self):
        """Iterate with prefetching."""
        if self.stream is None:
            # CPU or no CUDA - regular iteration
            for batch in self.dataloader:
                yield self._move_to_device(batch)
            return

        # GPU prefetching
        first = True
        for next_batch in self.dataloader:
            with torch.cuda.stream(self.stream):
                next_batch = self._move_to_device(next_batch)

            if not first:
                yield batch
            else:
                first = False

            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch

        yield batch

    def __len__(self):
        return len(self.dataloader)

    def _move_to_device(self, batch):
        """Move batch to device (non-blocking)."""
        if isinstance(batch, dict):
            return {
                k: self._move_to_device(v) if not isinstance(v, str) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, list):
            return [self._move_to_device(item) for item in batch]
        else:
            return batch


def optimize_dataloader_params(
    dataset_size: int,
    batch_size: int,
    num_gpus: int = 1
) -> dict:
    """
    Automatically optimize data loader parameters based on dataset size.

    Args:
        dataset_size: Number of samples in dataset
        batch_size: Batch size per GPU
        num_gpus: Number of GPUs

    Returns:
        Dictionary with optimized parameters
    """
    import multiprocessing

    # Determine optimal num_workers
    # Rule of thumb: 2-4 workers per GPU
    max_workers = multiprocessing.cpu_count()
    num_workers = min(4 * num_gpus, max_workers)

    # Prefetch factor
    # Larger for bigger datasets
    if dataset_size > 5000:
        prefetch_factor = 4
    elif dataset_size > 1000:
        prefetch_factor = 2
    else:
        prefetch_factor = 2

    # Pin memory (always True for GPU)
    pin_memory = True

    return {
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'pin_memory': pin_memory,
        'persistent_workers': True if num_workers > 0 else False
    }
