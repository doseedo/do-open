#!/usr/bin/env python3
"""
Dataset and DataLoader for Compositional Token Training
========================================================

Handles variable-length interval+COMPOSE sequences with:
- Padding to max length
- Random cropping for long sequences
- BOS/EOS tokens for sequence boundaries
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class CompositionalDataset(Dataset):
    """Dataset for compositional pattern sequences."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        add_bos_eos: bool = True,
        random_crop: bool = True,
    ):
        """
        Args:
            data_path: Path to compositional_tokens_v2 directory
            max_length: Maximum sequence length (crop or pad)
            add_bos_eos: Add BOS/EOS tokens
            random_crop: If True, randomly crop long sequences; else take first max_length
        """
        self.max_length = max_length
        self.add_bos_eos = add_bos_eos
        self.random_crop = random_crop

        data_dir = Path(data_path)

        # Load vocabulary
        with open(data_dir / 'vocab.json') as f:
            vocab_data = json.load(f)
        self.vocab = vocab_data['vocab']
        self.vocab_size = vocab_data['vocab_size']

        # Special tokens
        self.pad_id = self.vocab['PAD']
        self.bos_id = self.vocab['BOS']
        self.eos_id = self.vocab['EOS']

        # Load sequences
        data = torch.load(data_dir / 'rule_sequences.pt')
        self.sequences = data['sequences']

        # Filter out sequences that are too short
        min_len = 3 if not add_bos_eos else 1
        self.sequences = [s for s in self.sequences if len(s) >= min_len]

        print(f"Loaded {len(self.sequences)} sequences")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Max length: {max_length}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence."""
        seq = self.sequences[idx].copy() if isinstance(self.sequences[idx], list) else self.sequences[idx]

        # Convert to list if tensor
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()

        # Crop if too long
        effective_max = self.max_length - 2 if self.add_bos_eos else self.max_length
        if len(seq) > effective_max:
            if self.random_crop:
                start = random.randint(0, len(seq) - effective_max)
            else:
                start = 0
            seq = seq[start:start + effective_max]

        # Add BOS/EOS
        if self.add_bos_eos:
            seq = [self.bos_id] + seq + [self.eos_id]

        # Convert to tensor
        input_ids = torch.tensor(seq, dtype=torch.long)

        # For language modeling: input is seq[:-1], target is seq[1:]
        return {
            'input_ids': input_ids[:-1],
            'labels': input_ids[1:],
            'length': len(input_ids) - 1,
        }


def collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 = ignore in loss

    # Attention mask
    attention_mask = (input_ids_padded != pad_id).long()

    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded,
        'attention_mask': attention_mask,
        'lengths': lengths,
    }


def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    train_split: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    dataset = CompositionalDataset(
        data_path=data_path,
        max_length=max_length,
        add_bos_eos=True,
        random_crop=True,
    )

    # Split into train/val
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Create collate function with correct pad_id
    pad_id = dataset.pad_id
    collate = lambda batch: collate_fn(batch, pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    return train_loader, val_loader, dataset


if __name__ == '__main__':
    # Test the dataset
    data_path = '/home/arlo/do-repo/midi_generator/1_approaches/transform_based/compositional_tokens_v2'

    train_loader, val_loader, dataset = create_dataloader(
        data_path=data_path,
        batch_size=8,
        max_length=256,
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Sample batch
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  lengths: {batch['lengths']}")
