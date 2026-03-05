# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Corruption pipeline for the reconstruction task.

6 corruption types are applied to z_target to create x_cond:
1. light_noise (20%): Add small Gaussian noise
2. channel_dropout (20%): Zero out 30-50% of channels
3. temporal_masking (15%): Zero out random temporal spans
4. mean_swap (15%): Replace temporal mean with another stem's mean
5. full_substitution (15%): Replace with a completely different recording
6. blended (15%): Linear blend with another recording
"""

import random
from typing import Tuple, Callable, Optional

import torch


# Corruption type distribution
CORRUPTION_DISTRIBUTION = {
    'light_noise': 0.20,
    'channel_dropout': 0.20,
    'temporal_masking': 0.15,
    'mean_swap': 0.15,
    'full_substitution': 0.15,
    'blended': 0.15,
}


def sample_corruption_type() -> str:
    """Sample a corruption type according to the distribution."""
    types = list(CORRUPTION_DISTRIBUTION.keys())
    weights = list(CORRUPTION_DISTRIBUTION.values())
    return random.choices(types, weights=weights, k=1)[0]


def apply_light_noise(
    z_target: torch.Tensor,
    sigma_range: Tuple[float, float] = (0.05, 0.2),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add small Gaussian noise to the target.

    Args:
        z_target: Target latent [8, 16, T]
        sigma_range: Range of noise standard deviation

    Returns:
        Tuple of (corrupted, mask)
        - corrupted: z_target + noise
        - mask: ones (all positions are "valid" but noisy)
    """
    sigma = random.uniform(*sigma_range)
    noise = sigma * torch.randn_like(z_target)
    x_cond = z_target + noise

    # Mask is all ones (positions exist but are noisy)
    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    return x_cond, mask


def apply_channel_dropout(
    z_target: torch.Tensor,
    drop_fraction_range: Tuple[float, float] = (0.3, 0.5),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero out 30-50% of the 8 latent channels randomly.

    Args:
        z_target: Target latent [8, 16, T]
        drop_fraction_range: Range of fraction of channels to drop

    Returns:
        Tuple of (corrupted, mask)
    """
    num_channels = z_target.shape[0]
    drop_fraction = random.uniform(*drop_fraction_range)
    num_drop = max(1, int(num_channels * drop_fraction))

    # Select channels to drop
    drop_channels = random.sample(range(num_channels), num_drop)

    x_cond = z_target.clone()
    x_cond[drop_channels] = 0.0

    # Mask is all ones (all temporal positions exist, some channels zeroed)
    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    return x_cond, mask


def apply_temporal_masking(
    z_target: torch.Tensor,
    num_spans_range: Tuple[int, int] = (2, 5),
    span_length_range: Tuple[int, int] = (5, 20),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero out random temporal spans (5-20 frames each, 2-5 spans).

    Args:
        z_target: Target latent [8, 16, T]
        num_spans_range: Range of number of spans to mask
        span_length_range: Range of span lengths in frames

    Returns:
        Tuple of (corrupted, mask)
        - mask: 0 where masked, 1 where valid
    """
    T = z_target.shape[2]
    x_cond = z_target.clone()
    mask = torch.ones(1, z_target.shape[1], T)

    num_spans = random.randint(*num_spans_range)

    for _ in range(num_spans):
        span_len = random.randint(*span_length_range)
        span_len = min(span_len, T // 4)  # Don't mask more than 25% per span
        start = random.randint(0, max(0, T - span_len))

        x_cond[:, :, start:start + span_len] = 0.0
        mask[:, :, start:start + span_len] = 0.0

    return x_cond, mask


def apply_mean_swap(
    z_target: torch.Tensor,
    z_random: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace temporal mean with a random stem's mean.

    This preserves temporal structure but changes the overall
    timbral characteristics.

    Args:
        z_target: Target latent [8, 16, T]
        z_random: Random latent to get mean from [8, 16, T']

    Returns:
        Tuple of (corrupted, mask)
    """
    # Compute means
    mean_target = z_target.mean(dim=-1, keepdim=True)  # [8, 16, 1]
    mean_random = z_random.mean(dim=-1, keepdim=True)

    # Swap means
    x_cond = z_target - mean_target + mean_random

    # Mask is all ones
    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    return x_cond, mask


def apply_full_substitution(
    z_target: torch.Tensor,
    z_substitute: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace x_cond with a completely different recording.

    Args:
        z_target: Target latent [8, 16, T]
        z_substitute: Substitute latent [8, 16, T']

    Returns:
        Tuple of (corrupted, mask)
    """
    # Match length
    x_cond = match_length(z_substitute, z_target.shape[2])

    # Mask is all ones (need to fully regenerate but content exists)
    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    return x_cond, mask


def apply_blended(
    z_target: torch.Tensor,
    z_random: torch.Tensor,
    alpha_range: Tuple[float, float] = (0.3, 0.7),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear blend between target and random recording.

    Args:
        z_target: Target latent [8, 16, T]
        z_random: Random latent [8, 16, T']
        alpha_range: Range of blend factor (how much of target to keep)

    Returns:
        Tuple of (corrupted, mask)
    """
    alpha = random.uniform(*alpha_range)

    # Match length
    z_random = match_length(z_random, z_target.shape[2])

    x_cond = alpha * z_target + (1 - alpha) * z_random

    # Mask is all ones
    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    return x_cond, mask


def match_length(z: torch.Tensor, target_T: int) -> torch.Tensor:
    """
    Truncate or zero-pad tensor along time dimension to target length.

    Args:
        z: Input tensor [..., T]
        target_T: Target temporal length

    Returns:
        Tensor [..., target_T]
    """
    T = z.shape[-1]
    if T >= target_T:
        return z[..., :target_T]
    else:
        pad_size = target_T - T
        return torch.nn.functional.pad(z, (0, pad_size), value=0.0)


def apply_corruption(
    z_target: torch.Tensor,
    z_random_fn: Optional[Callable[[], torch.Tensor]] = None,
    corruption_type: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random corruption to create x_cond from z_target.

    Args:
        z_target: Target latent [8, 16, T]
        z_random_fn: Function to get a random latent (for mean_swap, substitution, blended)
        corruption_type: Specific corruption type, or None for random sampling

    Returns:
        Tuple of (x_cond, mask)
        - x_cond: Corrupted version of z_target [8, 16, T]
        - mask: Task mask [1, 16, T] where 1=valid, 0=needs generation
    """
    if corruption_type is None:
        corruption_type = sample_corruption_type()

    if corruption_type == 'light_noise':
        return apply_light_noise(z_target)

    elif corruption_type == 'channel_dropout':
        return apply_channel_dropout(z_target)

    elif corruption_type == 'temporal_masking':
        return apply_temporal_masking(z_target)

    elif corruption_type == 'mean_swap':
        if z_random_fn is None:
            # Fallback to light noise
            return apply_light_noise(z_target)
        z_random = z_random_fn()
        return apply_mean_swap(z_target, z_random)

    elif corruption_type == 'full_substitution':
        if z_random_fn is None:
            return apply_light_noise(z_target)
        z_substitute = z_random_fn()
        return apply_full_substitution(z_target, z_substitute)

    elif corruption_type == 'blended':
        if z_random_fn is None:
            return apply_light_noise(z_target)
        z_random = z_random_fn()
        return apply_blended(z_target, z_random)

    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
