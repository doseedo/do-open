#!/usr/bin/env python3
"""
Dataset and DataLoader for Piece-Level Training
================================================

Handles piece sequences where patterns are atomic tokens.
The model learns: what patterns follow what patterns in real music.

Features:
- Random cropping for long sequences (avg ~2348 tokens)
- Efficient batching with padding
- Train/val split
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from pathlib import Path
from typing import List, Dict, Tuple
import random


class PieceLevelDataset(Dataset):
    """Dataset for piece-level pattern sequences."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 1024,
        random_crop: bool = True,
    ):
        """
        Args:
            data_path: Path to piece_level_tokens directory
            max_length: Maximum sequence length (crop if longer)
            random_crop: If True, randomly crop; else take from start
        """
        self.max_length = max_length
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
        data = torch.load(data_dir / 'piece_sequences.pt')
        self.sequences = data['sequences']
        self.piece_names = data['piece_names']

        # Filter out sequences that are too short (need at least 3 tokens for LM)
        valid_indices = [i for i, s in enumerate(self.sequences) if len(s) >= 4]
        self.sequences = [self.sequences[i] for i in valid_indices]
        self.piece_names = [self.piece_names[i] for i in valid_indices]

        print(f"Loaded {len(self.sequences)} sequences")
        print(f"  Vocab size: {self.vocab_size:,}")
        print(f"  Max length: {max_length}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence."""
        seq = self.sequences[idx]

        # Convert to list if needed
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        else:
            seq = list(seq)

        # Crop if too long (leave room for prediction)
        if len(seq) > self.max_length:
            if self.random_crop:
                start = random.randint(0, len(seq) - self.max_length)
            else:
                start = 0
            seq = seq[start:start + self.max_length]

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
    batch_size: int = 8,
    max_length: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    train_split: float = 0.9,
) -> Tuple[DataLoader, DataLoader, PieceLevelDataset]:
    """Create train and validation dataloaders."""

    dataset = PieceLevelDataset(
        data_path=data_path,
        max_length=max_length,
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
    data_path = '/home/arlo/do-repo/midi_generator/1_approaches/transform_based/piece_level_tokens'

    train_loader, val_loader, dataset = create_dataloader(
        data_path=data_path,
        batch_size=4,
        max_length=512,
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

    # Decode sample
    id_to_token = {v: k for k, v in dataset.vocab.items()}
    sample = batch['input_ids'][0][:20]
    decoded = [id_to_token.get(t.item(), '?') for t in sample]
    print(f"\n  First 20 tokens decoded: {decoded}")
