# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Collation function for DO1 training.

Handles variable-length latents by padding to batch maximum,
and creates appropriate attention masks.
"""

from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F


def pad_to_length(
    tensor: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad tensor along last dimension to target length.

    Args:
        tensor: Input tensor [..., T]
        target_length: Target length for padding
        pad_value: Value to pad with

    Returns:
        Padded tensor [..., target_length]
    """
    current_length = tensor.shape[-1]
    if current_length >= target_length:
        return tensor[..., :target_length]
    pad_size = target_length - current_length
    return F.pad(tensor, (0, pad_size), value=pad_value)


def do1_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DO1 dataset.

    Handles variable-length latents by:
    1. Finding max temporal length for each tensor type
    2. Padding all tensors to their respective max lengths
    3. Creating attention masks for padded positions

    Expected sample format:
    {
        'x_cond': Tensor [8, 16, T],
        'x_ref': Tensor [8, 16, T'],
        'z_target': Tensor [8, 16, T],
        'mask': Tensor [1, 16, T],
        'task': str,
        'loss_weight': float,
    }

    Returns:
    {
        'x_cond': Tensor [B, 8, 16, max_T],
        'x_ref': Tensor [B, 8, 16, max_T_ref],
        'z_target': Tensor [B, 8, 16, max_T],
        'mask': Tensor [B, 1, 16, max_T],
        'attention_mask': Tensor [B, max_T],
        'ref_mask': Tensor [B, max_T_ref],
        'tasks': List[str],
        'loss_weights': Tensor [B],
    }
    """
    # Find max lengths
    max_T_target = max(sample['z_target'].shape[-1] for sample in batch)
    max_T_ref = max(sample['x_ref'].shape[-1] for sample in batch)

    # Lists to collect batched tensors
    x_cond_list = []
    x_ref_list = []
    z_target_list = []
    mask_list = []
    attention_mask_list = []
    ref_mask_list = []
    tasks = []
    loss_weights = []

    for sample in batch:
        T_target = sample['z_target'].shape[-1]
        T_ref = sample['x_ref'].shape[-1]

        # Pad main tensors (same temporal length as target)
        x_cond = pad_to_length(sample['x_cond'], max_T_target, pad_value=0.0)
        z_target = pad_to_length(sample['z_target'], max_T_target, pad_value=0.0)
        task_mask = pad_to_length(sample['mask'], max_T_target, pad_value=0.0)

        # Pad reference tensor
        x_ref = pad_to_length(sample['x_ref'], max_T_ref, pad_value=0.0)

        # Create attention masks (1 = valid, 0 = padding)
        attention_mask = torch.zeros(max_T_target)
        attention_mask[:T_target] = 1.0

        ref_mask = torch.zeros(max_T_ref)
        ref_mask[:T_ref] = 1.0

        # Append to lists
        x_cond_list.append(x_cond)
        x_ref_list.append(x_ref)
        z_target_list.append(z_target)
        mask_list.append(task_mask)
        attention_mask_list.append(attention_mask)
        ref_mask_list.append(ref_mask)
        tasks.append(sample.get('task', 'unknown'))
        loss_weights.append(sample.get('loss_weight', 1.0))

    return {
        'x_cond': torch.stack(x_cond_list),
        'x_ref': torch.stack(x_ref_list),
        'z_target': torch.stack(z_target_list),
        'mask': torch.stack(mask_list),
        'attention_mask': torch.stack(attention_mask_list),
        'ref_mask': torch.stack(ref_mask_list),
        'tasks': tasks,
        'loss_weights': torch.tensor(loss_weights),
    }


def do1_collate_fn_with_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that preserves additional metadata.

    Same as do1_collate_fn but also passes through any extra
    keys in the sample dicts.
    """
    # Standard collation
    result = do1_collate_fn(batch)

    # Collect any additional metadata keys
    known_keys = {'x_cond', 'x_ref', 'z_target', 'mask', 'task', 'loss_weight'}

    for key in batch[0].keys():
        if key not in known_keys:
            # Collect metadata (don't try to stack, just list)
            result[key] = [sample[key] for sample in batch]

    return result


def create_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Create padding mask from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths [B]
        max_length: Maximum sequence length

    Returns:
        Padding mask [B, max_length] where 1=valid, 0=padding
    """
    batch_size = lengths.shape[0]
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    return mask.float()


def merge_attention_masks(
    query_mask: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Merge query and key masks for attention.

    Creates a 2D attention mask where position (i, j) is valid
    only if both query position i and key position j are valid.

    Args:
        query_mask: [B, S_q] mask for query positions
        key_mask: [B, S_k] mask for key positions

    Returns:
        Attention mask [B, S_q, S_k]
    """
    # [B, S_q, 1] * [B, 1, S_k] -> [B, S_q, S_k]
    return query_mask.unsqueeze(-1) * key_mask.unsqueeze(1)
