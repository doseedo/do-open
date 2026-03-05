# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Task-specific sampling functions for DO1 training.

Each task has its own sampling function that constructs
the (x_cond, x_ref, z_target, mask) tuple appropriately.

Task Distribution:
- reconstruction (35%): Corrupt z_target, x_ref from same session
- separation (20%): Mix stems, x_ref is instrument reference
- cross_instrument (15%): Transfer to different instrument
- fx (10%): FX removal/application
- generation (10%): Generate from scratch with reference
- inpainting (5%): Fill temporal gaps
- synth_diversity (5%): Transfer between VST patches
"""

import random
from typing import Dict, Any, Callable, Optional, List, Tuple

import torch

from .corruption import apply_corruption, match_length
from .latent_synth import LatentSynthesizer, create_easy_mix, create_hard_mix, create_dense_mix


# Task distribution weights
TASK_DISTRIBUTION = {
    'reconstruction': 0.35,
    'separation': 0.20,
    'cross_instrument': 0.15,
    'fx': 0.10,
    'generation': 0.10,
    'inpainting': 0.05,
    'synth_diversity': 0.05,
}

# Separation difficulty distribution
SEPARATION_DIFFICULTY = {
    'easy': 0.60,
    'hard': 0.25,
    'dense': 0.15,
}


def sample_task() -> str:
    """Sample a task according to the distribution."""
    tasks = list(TASK_DISTRIBUTION.keys())
    weights = list(TASK_DISTRIBUTION.values())
    return random.choices(tasks, weights=weights, k=1)[0]


def get_reconstruction_sample(
    z_target: torch.Tensor,
    session_stems: Dict[str, torch.Tensor],
    random_stem_fn: Callable[[], torch.Tensor],
    cfg_dropout_rate: float = 0.3,
) -> Dict[str, Any]:
    """
    Generate a reconstruction task sample.

    Reconstruction task:
    - x_cond = corrupt(z_target) with random corruption
    - x_ref = another stem from same session (or random if single stem)
    - mask = depends on corruption type (ones or partial)

    Args:
        z_target: Target latent [8, 16, T]
        session_stems: Dict of other stems in the same session
        random_stem_fn: Function to get a random stem from dataset
        cfg_dropout_rate: Probability of dropping x_ref for CFG

    Returns:
        Sample dict with x_cond, x_ref, z_target, mask, task, loss_weight
    """
    # Apply random corruption
    x_cond, mask = apply_corruption(z_target, z_random_fn=random_stem_fn)

    # Get reference from same session
    if session_stems:
        ref_key = random.choice(list(session_stems.keys()))
        x_ref = session_stems[ref_key]
    else:
        x_ref = random_stem_fn()

    # CFG dropout
    if random.random() < cfg_dropout_rate:
        x_ref = torch.zeros_like(x_ref)

    return {
        'x_cond': x_cond,
        'x_ref': x_ref,
        'z_target': z_target,
        'mask': mask,
        'task': 'reconstruction',
        'loss_weight': 1.0,
    }


def get_separation_sample(
    z_target: torch.Tensor,
    session_stems: Dict[str, torch.Tensor],
    instrument_ref_fn: Callable[[str], torch.Tensor],
    target_instrument: Optional[str] = None,
    instrument_labels: Optional[Dict[str, str]] = None,
    cfg_dropout_rate: float = 0.3,
) -> Dict[str, Any]:
    """
    Generate a separation task sample.

    Separation task:
    - x_cond = z_mix (sum of multiple stems)
    - x_ref = instrument reference (same instrument from elsewhere)
    - mask = zeros (all positions need prediction)

    Args:
        z_target: Target stem to extract [8, 16, T]
        session_stems: Dict of all stems in the session
        instrument_ref_fn: Function to get a reference for the target instrument
        target_instrument: Instrument label for the target
        instrument_labels: Optional instrument labels for stems
        cfg_dropout_rate: Probability of dropping x_ref for CFG

    Returns:
        Sample dict
    """
    synthesizer = LatentSynthesizer()

    # Sample difficulty
    difficulties = list(SEPARATION_DIFFICULTY.keys())
    weights = list(SEPARATION_DIFFICULTY.values())
    difficulty = random.choices(difficulties, weights=weights, k=1)[0]

    # Get all stems including target
    all_stems = dict(session_stems)
    all_stems['_target'] = z_target

    # Create mix based on difficulty
    if difficulty == 'easy':
        z_mix, _ = create_easy_mix(all_stems, '_target', num_stems=random.randint(2, 4))
    elif difficulty == 'hard':
        z_mix, _ = create_hard_mix(all_stems, '_target', num_stems=random.randint(2, 4))
    else:  # dense
        z_mix, _ = create_dense_mix(all_stems, '_target', min_stems=5, max_stems=10)

    # Match mix length to target
    z_mix = match_length(z_mix, z_target.shape[-1])

    # Get instrument reference
    try:
        x_ref = instrument_ref_fn(target_instrument)
    except (NotImplementedError, ValueError):
        # Fallback: use target itself as reference
        x_ref = z_target.clone()

    # Match reference length
    x_ref = match_length(x_ref, z_target.shape[-1])

    # CFG dropout
    if random.random() < cfg_dropout_rate:
        x_ref = torch.zeros_like(x_ref)

    # Mask is all zeros for separation (predict everything)
    mask = torch.zeros(1, z_target.shape[1], z_target.shape[2])

    return {
        'x_cond': z_mix,
        'x_ref': x_ref,
        'z_target': z_target,
        'mask': mask,
        'task': 'separation',
        'loss_weight': 1.0,
    }


def get_cross_instrument_sample(
    z_target: torch.Tensor,
    session_stems: Dict[str, torch.Tensor],
    latent_synth: Optional[LatentSynthesizer] = None,
    midi_data: Optional[Any] = None,
    cfg_dropout_rate: float = 0.3,
) -> Dict[str, Any]:
    """
    Generate a cross-instrument task sample.

    Cross-instrument task:
    - x_cond = synthesized version or different instrument playing same part
    - x_ref = reference of target instrument
    - mask = ones (transfer timbre while preserving content)

    Args:
        z_target: Target latent [8, 16, T]
        session_stems: Dict of other stems in the session
        latent_synth: Optional latent synthesizer for MIDI rendering
        midi_data: Optional MIDI data for the target
        cfg_dropout_rate: Probability of dropping x_ref for CFG

    Returns:
        Sample dict
    """
    # Option 1: Use latent synth if available and MIDI exists
    # Option 2: Use another stem from same session as x_cond

    if latent_synth is not None and midi_data is not None:
        try:
            # Render with random synth params
            vcf = latent_synth.sample_random_vcf_params()
            vca = latent_synth.sample_random_vca_params()
            x_cond = latent_synth.render_midi(midi_data, vcf, vca, z_target.shape[-1])
        except NotImplementedError:
            # Fallback to natural cross-instrument
            x_cond = _get_natural_cross_instrument(z_target, session_stems)
    else:
        x_cond = _get_natural_cross_instrument(z_target, session_stems)

    # Match length
    x_cond = match_length(x_cond, z_target.shape[-1])

    # Reference: another occurrence of target instrument (or self)
    if session_stems:
        ref_key = random.choice(list(session_stems.keys()))
        x_ref = session_stems[ref_key]
    else:
        x_ref = z_target.clone()

    # CFG dropout
    if random.random() < cfg_dropout_rate:
        x_ref = torch.zeros_like(x_ref)

    # Mask is all ones for transfer
    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    return {
        'x_cond': x_cond,
        'x_ref': x_ref,
        'z_target': z_target,
        'mask': mask,
        'task': 'cross_instrument',
        'loss_weight': 1.0,
    }


def _get_natural_cross_instrument(
    z_target: torch.Tensor,
    session_stems: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Get a natural cross-instrument source from session stems."""
    if session_stems:
        stem_key = random.choice(list(session_stems.keys()))
        return session_stems[stem_key]
    else:
        # Fallback: add noise to target
        return z_target + 0.1 * torch.randn_like(z_target)


def get_generation_sample(
    z_target: torch.Tensor,
    instrument_ref_fn: Callable[[str], torch.Tensor],
    target_instrument: Optional[str] = None,
    cfg_dropout_rate: float = 0.3,
) -> Dict[str, Any]:
    """
    Generate a generation task sample.

    Generation task:
    - x_cond = zeros (no content provided)
    - x_ref = instrument reference (style/timbre to generate)
    - mask = zeros (generate everything)

    Args:
        z_target: Target latent [8, 16, T]
        instrument_ref_fn: Function to get a reference for the target instrument
        target_instrument: Instrument label for the target
        cfg_dropout_rate: Probability of dropping x_ref for CFG

    Returns:
        Sample dict
    """
    # x_cond is all zeros
    x_cond = torch.zeros_like(z_target)

    # Get instrument reference
    try:
        x_ref = instrument_ref_fn(target_instrument)
        x_ref = match_length(x_ref, z_target.shape[-1])
    except (NotImplementedError, ValueError):
        # Fallback: use portion of target as reference
        T = z_target.shape[-1]
        ref_len = max(T // 4, 50)
        start = random.randint(0, max(0, T - ref_len))
        x_ref = z_target[..., start:start + ref_len]
        x_ref = match_length(x_ref, T)

    # CFG dropout
    if random.random() < cfg_dropout_rate:
        x_ref = torch.zeros_like(x_ref)

    # Mask is all zeros for generation
    mask = torch.zeros(1, z_target.shape[1], z_target.shape[2])

    return {
        'x_cond': x_cond,
        'x_ref': x_ref,
        'z_target': z_target,
        'mask': mask,
        'task': 'generation',
        'loss_weight': 1.0,
    }


def get_inpainting_sample(
    z_target: torch.Tensor,
    session_stems: Dict[str, torch.Tensor],
    num_gaps_range: Tuple[int, int] = (1, 3),
    gap_fraction_range: Tuple[float, float] = (0.1, 0.3),
    cfg_dropout_rate: float = 0.3,
) -> Dict[str, Any]:
    """
    Generate an inpainting task sample.

    Inpainting task:
    - x_cond = z_target with temporal gaps zeroed out
    - x_ref = same-session reference
    - mask = partial (0 in gaps, 1 elsewhere)

    Args:
        z_target: Target latent [8, 16, T]
        session_stems: Dict of other stems in the session
        num_gaps_range: Range of number of gaps to create
        gap_fraction_range: Range of gap size as fraction of total length
        cfg_dropout_rate: Probability of dropping x_ref for CFG

    Returns:
        Sample dict
    """
    T = z_target.shape[2]
    x_cond = z_target.clone()
    mask = torch.ones(1, z_target.shape[1], T)

    # Create gaps
    num_gaps = random.randint(*num_gaps_range)
    for _ in range(num_gaps):
        gap_frac = random.uniform(*gap_fraction_range)
        gap_len = int(T * gap_frac)
        gap_len = max(5, min(gap_len, T // 2))  # At least 5 frames, at most half
        start = random.randint(0, max(0, T - gap_len))

        x_cond[:, :, start:start + gap_len] = 0.0
        mask[:, :, start:start + gap_len] = 0.0

    # Reference from same session
    if session_stems:
        ref_key = random.choice(list(session_stems.keys()))
        x_ref = session_stems[ref_key]
    else:
        x_ref = z_target.clone()

    # CFG dropout
    if random.random() < cfg_dropout_rate:
        x_ref = torch.zeros_like(x_ref)

    return {
        'x_cond': x_cond,
        'x_ref': x_ref,
        'z_target': z_target,
        'mask': mask,
        'task': 'inpainting',
        'loss_weight': 1.0,
    }


def get_synth_diversity_sample(
    vst_dataset: Dict[str, Dict[str, torch.Tensor]],
    cfg_dropout_rate: float = 0.3,
) -> Dict[str, Any]:
    """
    Generate a synth diversity task sample.

    Synth diversity task:
    - x_cond = same MIDI, different VST patch (wrong timbre)
    - x_ref = target patch playing different MIDI (right timbre)
    - mask = ones (transfer timbre)

    VST dataset structure: {midi_id: {patch_id: z_latent}}

    Args:
        vst_dataset: Dict of MIDI -> patch -> latent
        cfg_dropout_rate: Probability of dropping x_ref for CFG

    Returns:
        Sample dict
    """
    # Pick a random MIDI
    midi_ids = list(vst_dataset.keys())
    midi_id = random.choice(midi_ids)
    patches = list(vst_dataset[midi_id].keys())

    if len(patches) < 2:
        # Need at least 2 patches for cross-patch transfer
        # Find another MIDI with the target patch
        patch_a = patches[0]
        z_target = vst_dataset[midi_id][patch_a]

        # Find another MIDI for reference
        other_midis = [m for m in midi_ids if m != midi_id]
        if other_midis:
            midi_ref = random.choice(other_midis)
            patches_ref = list(vst_dataset[midi_ref].keys())
            patch_ref = random.choice(patches_ref)
            x_ref = vst_dataset[midi_ref][patch_ref]
        else:
            x_ref = z_target.clone()

        # x_cond: same MIDI with different patch (not available, fallback to noise)
        x_cond = z_target + 0.15 * torch.randn_like(z_target)
    else:
        # Two different patches for same MIDI
        patch_a, patch_b = random.sample(patches, 2)
        z_target = vst_dataset[midi_id][patch_a]
        x_cond = vst_dataset[midi_id][patch_b]  # Same MIDI, different patch

        # Reference: target patch on different MIDI
        other_midis = [m for m in midi_ids if m != midi_id and patch_a in vst_dataset[m]]
        if other_midis:
            midi_ref = random.choice(other_midis)
            x_ref = vst_dataset[midi_ref][patch_a]
        else:
            x_ref = z_target.clone()

    # Match lengths
    T = z_target.shape[-1]
    x_cond = match_length(x_cond, T)
    x_ref = match_length(x_ref, T)

    # CFG dropout
    if random.random() < cfg_dropout_rate:
        x_ref = torch.zeros_like(x_ref)

    # Mask is all ones for timbre transfer
    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    return {
        'x_cond': x_cond,
        'x_ref': x_ref,
        'z_target': z_target,
        'mask': mask,
        'task': 'synth_diversity',
        'loss_weight': 1.0,
    }
