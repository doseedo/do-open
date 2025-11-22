"""
Variable-Length Collate Functions - Agent 8
============================================

Custom collate functions for batching variable-length MIDI sequences.

MIDI files have different lengths, requiring custom batching strategies:
1. Padding-based: Pad sequences to max length in batch
2. Packed sequences: More efficient, no wasted computation on padding

This module provides:
- variable_length_collate_fn: Padding-based collate
- packed_sequence_collate_fn: Packed sequence collate
- create_attention_mask: Mask for padded positions

Author: Agent 8 - Data Pipeline & Preprocessing
"""

import warnings
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")


# ============================================================================
# Collate Functions
# ============================================================================

def variable_length_collate_fn(
    batch: List[Dict[str, Any]],
    pad_value: float = 0.0,
    include_attention_mask: bool = True
) -> Dict[str, Any]:
    """
    Collate function for variable-length MIDI sequences with padding.

    Pads all sequences in batch to max length and creates attention masks
    to indicate valid (non-padded) positions.

    Expected batch item format:
        {
            'pianoroll': torch.Tensor,      # (time, 128, tracks)
            'dna': torch.Tensor,            # (300,)
            'features': torch.Tensor,       # (200,)
            'file_id': str,
            'length': int,                   # Original sequence length
            ...                              # Other metadata
        }

    Args:
        batch: List of dataset items (dicts)
        pad_value: Value to use for padding (default 0.0)
        include_attention_mask: Whether to create attention mask

    Returns:
        Batched dictionary:
        {
            'pianorolls': torch.Tensor,     # (batch, max_time, 128, tracks)
            'dnas': torch.Tensor,           # (batch, 300)
            'features': torch.Tensor,       # (batch, 200)
            'attention_mask': torch.Tensor, # (batch, max_time) - 1=valid, 0=padding
            'lengths': torch.Tensor,        # (batch,) - original lengths
            'file_ids': List[str],
            ...
        }
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")

    if len(batch) == 0:
        return {}

    # Extract components
    pianorolls = []
    dnas = []
    features = []
    lengths = []
    file_ids = []
    metadata = {}

    for item in batch:
        # Pianoroll: (time, 128, tracks)
        if 'pianoroll' in item:
            pianoroll = item['pianoroll']
            if not isinstance(pianoroll, torch.Tensor):
                pianoroll = torch.from_numpy(pianoroll).float()
            pianorolls.append(pianoroll)
            lengths.append(pianoroll.shape[0])

        # DNA: (300,)
        if 'dna' in item:
            dna = item['dna']
            if not isinstance(dna, torch.Tensor):
                dna = torch.from_numpy(dna).float()
            dnas.append(dna)

        # Features: (200,)
        if 'features' in item:
            feat = item['features']
            if not isinstance(feat, torch.Tensor):
                feat = torch.from_numpy(feat).float()
            features.append(feat)

        # File ID
        if 'file_id' in item:
            file_ids.append(item['file_id'])

        # Collect other metadata
        for key, value in item.items():
            if key not in ['pianoroll', 'dna', 'features', 'file_id', 'length']:
                if key not in metadata:
                    metadata[key] = []
                metadata[key].append(value)

    # Batch tensors
    batched = {}

    # Batch pianorolls with padding
    if pianorolls:
        # Pad to max length in batch
        max_time = max(lengths)
        batch_size = len(pianorolls)
        n_pitches = pianorolls[0].shape[1]
        n_tracks = pianorolls[0].shape[2]

        # Create padded tensor
        padded_pianorolls = torch.full(
            (batch_size, max_time, n_pitches, n_tracks),
            pad_value,
            dtype=torch.float32
        )

        for i, pianoroll in enumerate(pianorolls):
            length = pianoroll.shape[0]
            padded_pianorolls[i, :length] = pianoroll

        batched['pianorolls'] = padded_pianorolls
        batched['lengths'] = torch.tensor(lengths, dtype=torch.long)

        # Create attention mask
        if include_attention_mask:
            attention_mask = create_attention_mask(lengths, max_time)
            batched['attention_mask'] = attention_mask

    # Batch DNAs
    if dnas:
        batched['dnas'] = torch.stack(dnas)

    # Batch features
    if features:
        batched['features'] = torch.stack(features)

    # Add file IDs
    if file_ids:
        batched['file_ids'] = file_ids

    # Add other metadata
    for key, values in metadata.items():
        # Try to stack/batch if possible
        if all(isinstance(v, (int, float)) for v in values):
            batched[key] = torch.tensor(values)
        elif all(isinstance(v, torch.Tensor) for v in values):
            # Check if all same shape
            if all(v.shape == values[0].shape for v in values):
                batched[key] = torch.stack(values)
            else:
                batched[key] = values  # Keep as list
        else:
            batched[key] = values  # Keep as list

    return batched


def packed_sequence_collate_fn(
    batch: List[Dict[str, Any]],
    pad_value: float = 0.0
) -> Dict[str, Any]:
    """
    Collate function for variable-length sequences using PackedSequence.

    More efficient than padding for RNNs/LSTMs as it avoids wasted computation
    on padded positions.

    Expected batch item format: Same as variable_length_collate_fn

    Args:
        batch: List of dataset items (dicts)
        pad_value: Value to use for padding

    Returns:
        Batched dictionary with PackedSequence for pianorolls:
        {
            'pianorolls_packed': PackedSequence,  # Packed pianoroll sequences
            'dnas': torch.Tensor,                 # (batch, 300)
            'features': torch.Tensor,             # (batch, 200)
            'lengths': torch.Tensor,              # (batch,) - for unpacking
            'file_ids': List[str],
            ...
        }
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")

    if len(batch) == 0:
        return {}

    # First collate normally
    batched = variable_length_collate_fn(
        batch,
        pad_value=pad_value,
        include_attention_mask=False
    )

    # Pack pianorolls if present
    if 'pianorolls' in batched:
        pianorolls = batched['pianorolls']  # (batch, max_time, 128, tracks)
        lengths = batched['lengths']  # (batch,)

        # Reshape for packing: (batch, max_time, 128*tracks)
        batch_size, max_time, n_pitches, n_tracks = pianorolls.shape
        pianorolls_flat = pianorolls.reshape(batch_size, max_time, n_pitches * n_tracks)

        # Sort by length (descending) for pack_padded_sequence
        sorted_lengths, sorted_indices = lengths.sort(descending=True)
        sorted_pianorolls = pianorolls_flat[sorted_indices]

        # Pack sequences
        packed = pack_padded_sequence(
            sorted_pianorolls,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # Update batched dict
        batched['pianorolls_packed'] = packed
        batched['sorted_indices'] = sorted_indices
        batched['original_shape'] = (n_pitches, n_tracks)
        del batched['pianorolls']  # Remove unpacked version

    return batched


def create_attention_mask(
    lengths: List[int],
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Create attention mask for padded sequences.

    Args:
        lengths: List of sequence lengths
        max_length: Maximum length (if None, use max of lengths)

    Returns:
        Attention mask tensor: (batch, max_length)
        - 1 for valid positions
        - 0 for padded positions

    Example:
        lengths = [3, 5, 2]
        mask = create_attention_mask(lengths, max_length=5)
        # mask = [[1, 1, 1, 0, 0],
        #         [1, 1, 1, 1, 1],
        #         [1, 1, 0, 0, 0]]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")

    if max_length is None:
        max_length = max(lengths)

    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    return mask


# ============================================================================
# Utility Functions
# ============================================================================

def unpack_pianorolls(
    packed_pianorolls: 'torch.nn.utils.rnn.PackedSequence',
    original_shape: Tuple[int, int],
    sorted_indices: torch.Tensor
) -> torch.Tensor:
    """
    Unpack PackedSequence back to padded tensor in original order.

    Args:
        packed_pianorolls: PackedSequence from packed_sequence_collate_fn
        original_shape: (n_pitches, n_tracks) from collate output
        sorted_indices: Indices used to sort sequences

    Returns:
        Padded pianorolls: (batch, max_time, 128, tracks)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")

    from torch.nn.utils.rnn import pad_packed_sequence

    # Unpack sequences
    pianorolls_flat, lengths = pad_packed_sequence(
        packed_pianorolls,
        batch_first=True
    )

    # Restore original shape
    batch_size, max_time, _ = pianorolls_flat.shape
    n_pitches, n_tracks = original_shape
    pianorolls = pianorolls_flat.reshape(batch_size, max_time, n_pitches, n_tracks)

    # Restore original order (undo sorting)
    inverse_indices = torch.argsort(sorted_indices)
    pianorolls = pianorolls[inverse_indices]

    return pianorolls


def apply_mask_to_loss(
    loss_tensor: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply attention mask to loss tensor to ignore padded positions.

    Args:
        loss_tensor: (batch, time, ...) loss values
        attention_mask: (batch, time) mask (1=valid, 0=padding)

    Returns:
        Masked loss tensor with same shape
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")

    # Expand mask to match loss tensor dimensions
    while attention_mask.ndim < loss_tensor.ndim:
        attention_mask = attention_mask.unsqueeze(-1)

    # Apply mask (set padded positions to 0)
    masked_loss = loss_tensor * attention_mask.float()

    return masked_loss


def get_valid_loss_mean(
    loss_tensor: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute mean loss over valid (non-padded) positions only.

    Args:
        loss_tensor: (batch, time, ...) loss values
        attention_mask: (batch, time) mask (1=valid, 0=padding)

    Returns:
        Scalar mean loss over valid positions
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")

    # Apply mask
    masked_loss = apply_mask_to_loss(loss_tensor, attention_mask)

    # Compute mean over valid positions
    total_loss = masked_loss.sum()
    valid_positions = attention_mask.sum()

    if valid_positions > 0:
        return total_loss / valid_positions
    else:
        return torch.tensor(0.0, device=loss_tensor.device)


# ============================================================================
# Batching Strategy Selector
# ============================================================================

def get_collate_fn(
    strategy: str = 'padding',
    **kwargs
) -> callable:
    """
    Get appropriate collate function based on strategy.

    Args:
        strategy: 'padding' or 'packed'
        **kwargs: Additional arguments for collate function

    Returns:
        Collate function

    Example:
        from torch.utils.data import DataLoader

        collate_fn = get_collate_fn(
            strategy='padding',
            pad_value=0.0,
            include_attention_mask=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=collate_fn
        )
    """
    if strategy == 'padding':
        def collate(batch):
            return variable_length_collate_fn(batch, **kwargs)
        return collate

    elif strategy == 'packed':
        def collate(batch):
            return packed_sequence_collate_fn(batch, **kwargs)
        return collate

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'padding' or 'packed'")


# ============================================================================
# Debug Utilities
# ============================================================================

def print_batch_info(batch: Dict[str, Any], prefix: str = ""):
    """
    Print information about batched data.

    Useful for debugging collate functions.

    Args:
        batch: Batched dictionary
        prefix: Prefix for print statements
    """
    print(f"\n{prefix}Batch Information:")
    print(f"{prefix}{'='*60}")

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{prefix}  {key}: {value.shape} ({value.dtype})")
        elif isinstance(value, list):
            print(f"{prefix}  {key}: List[{len(value)}]")
            if len(value) > 0:
                print(f"{prefix}    First item: {type(value[0])}")
        else:
            print(f"{prefix}  {key}: {type(value)}")

    print(f"{prefix}{'='*60}\n")


def validate_batch(
    batch: Dict[str, Any],
    expected_batch_size: int,
    expected_dna_dim: int = 300,
    expected_feature_dim: int = 200
) -> bool:
    """
    Validate batched data has expected shapes.

    Args:
        batch: Batched dictionary
        expected_batch_size: Expected batch size
        expected_dna_dim: Expected DNA dimension
        expected_feature_dim: Expected feature dimension

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check DNAs
        if 'dnas' in batch:
            assert batch['dnas'].shape == (expected_batch_size, expected_dna_dim), \
                f"Invalid DNA shape: {batch['dnas'].shape}"

        # Check features
        if 'features' in batch:
            assert batch['features'].shape == (expected_batch_size, expected_feature_dim), \
                f"Invalid feature shape: {batch['features'].shape}"

        # Check pianorolls
        if 'pianorolls' in batch:
            assert batch['pianorolls'].shape[0] == expected_batch_size, \
                f"Invalid pianoroll batch size: {batch['pianorolls'].shape[0]}"

        # Check lengths
        if 'lengths' in batch:
            assert batch['lengths'].shape == (expected_batch_size,), \
                f"Invalid lengths shape: {batch['lengths'].shape}"

        # Check attention mask
        if 'attention_mask' in batch:
            assert batch['attention_mask'].shape[0] == expected_batch_size, \
                f"Invalid attention mask batch size: {batch['attention_mask'].shape[0]}"

        return True

    except AssertionError as e:
        print(f"⚠️ Batch validation failed: {e}")
        return False
