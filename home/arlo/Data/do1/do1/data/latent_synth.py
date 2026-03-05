# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Latent Synthesizer for DO1.

This module handles:
1. Mixing stems in latent space (for separation task)
2. Rendering MIDI through latent synth (for cross-instrument task)

TODO: Implement full latent synthesis from MIDI
TODO: Implement VCF/VCA parameter rendering
"""

import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F


# DCAE latent space normalization constants
DCAE_SCALE_FACTOR = 0.1786
DCAE_SHIFT_FACTOR = -1.9091


class LatentSynthesizer:
    """
    Synthesizer for latent-space operations.

    Handles mixing stems in latent space and (eventually) rendering
    MIDI through a learned latent synthesizer.

    The DCAE latent space is approximately linear for mixing operations,
    allowing us to construct mixes on-the-fly during training.

    Args:
        scale_factor: DCAE scale factor for unnormalization
        shift_factor: DCAE shift factor for unnormalization
        mix_noise_std: Standard deviation of noise to add during mixing
            (helps model learn to handle mixing artifacts)
    """

    def __init__(
        self,
        scale_factor: float = DCAE_SCALE_FACTOR,
        shift_factor: float = DCAE_SHIFT_FACTOR,
        mix_noise_std: float = 0.02,
    ):
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        self.mix_noise_std = mix_noise_std

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Convert normalized latent to raw latent space."""
        return z / self.scale_factor + self.shift_factor

    def normalize(self, z_raw: torch.Tensor) -> torch.Tensor:
        """Convert raw latent to normalized latent space."""
        return (z_raw - self.shift_factor) * self.scale_factor

    def mix_stems(
        self,
        stems: Dict[str, torch.Tensor],
        gains: Optional[Dict[str, float]] = None,
        exclude: Optional[List[str]] = None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Mix multiple stems in latent space.

        The mixing is done by unnormalizing, summing with gains, and renormalizing.
        A small amount of noise is added to help the model learn to handle
        mixing artifacts.

        Args:
            stems: Dict of stem_name -> latent [8, 16, T]
            gains: Optional per-stem gains (default: all 1.0)
            exclude: Stems to exclude from mix
            add_noise: Whether to add noise for robustness

        Returns:
            z_mix: Mixed latent [8, 16, max_T]
        """
        if gains is None:
            gains = {k: 1.0 for k in stems}

        # Filter excluded stems
        if exclude:
            stems = {k: v for k, v in stems.items() if k not in exclude}
            gains = {k: v for k, v in gains.items() if k not in exclude}

        if len(stems) == 0:
            raise ValueError("No stems to mix after exclusion")

        # Align time dimension
        max_T = max(s.shape[-1] for s in stems.values())
        aligned = {}
        for name, z in stems.items():
            if z.shape[-1] < max_T:
                z = F.pad(z, (0, max_T - z.shape[-1]))
            aligned[name] = z

        # Mix in raw latent space
        z_mix_raw = torch.zeros_like(next(iter(aligned.values())))
        for name, z in aligned.items():
            z_raw = self.unnormalize(z)
            z_mix_raw += gains[name] * z_raw

        # Normalize back
        z_mix = self.normalize(z_mix_raw)

        # Add noise for robustness
        if add_noise:
            z_mix = z_mix + torch.randn_like(z_mix) * self.mix_noise_std

        return z_mix

    def render_midi(
        self,
        midi_data: Any,
        vcf_params: Optional[Dict[str, float]] = None,
        vca_params: Optional[Dict[str, float]] = None,
        target_T: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Render MIDI through latent synthesizer.

        TODO: Implement this functionality.

        Args:
            midi_data: MIDI data to render
            vcf_params: VCF (filter) parameters
            vca_params: VCA (amplitude) parameters
            target_T: Target temporal length

        Returns:
            Rendered latent [8, 16, T]
        """
        raise NotImplementedError(
            "Latent MIDI synthesis not yet implemented. "
            "Use mix_stems() for now or provide pre-rendered latents."
        )

    def sample_random_vcf_params(self) -> Dict[str, float]:
        """
        Sample random VCF (filter) parameters for synthesis.

        TODO: Define actual parameter ranges based on the synthesizer design.

        Returns:
            Dict of VCF parameters
        """
        return {
            'cutoff': random.uniform(0.1, 0.9),
            'resonance': random.uniform(0.0, 0.8),
            'envelope_amount': random.uniform(0.0, 1.0),
            'attack': random.uniform(0.001, 0.1),
            'decay': random.uniform(0.01, 0.5),
            'sustain': random.uniform(0.0, 1.0),
            'release': random.uniform(0.05, 1.0),
        }

    def sample_random_vca_params(self) -> Dict[str, float]:
        """
        Sample random VCA (amplitude) parameters for synthesis.

        TODO: Define actual parameter ranges based on the synthesizer design.

        Returns:
            Dict of VCA parameters
        """
        return {
            'attack': random.uniform(0.001, 0.1),
            'decay': random.uniform(0.01, 0.5),
            'sustain': random.uniform(0.3, 1.0),
            'release': random.uniform(0.05, 1.0),
            'velocity_sensitivity': random.uniform(0.0, 1.0),
        }


def create_easy_mix(
    stems: Dict[str, torch.Tensor],
    target_stem: str,
    num_stems: int = 3,
    instrument_labels: Optional[Dict[str, str]] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Create an easy separation task mix with distinct instruments.

    Args:
        stems: Dict of stem_name -> latent
        target_stem: The stem we want to extract
        num_stems: Number of stems in the mix (including target)
        instrument_labels: Optional instrument labels for selecting distinct instruments

    Returns:
        Tuple of (mix, list of stem names in mix)
    """
    synthesizer = LatentSynthesizer()

    # Select stems for mix
    available = [k for k in stems.keys() if k != target_stem]

    # TODO: Use instrument_labels to select distinct instruments
    # For now, just randomly select
    num_additional = min(num_stems - 1, len(available))
    mix_stems = random.sample(available, num_additional)
    mix_stems.append(target_stem)

    # Create mix
    mix_dict = {k: stems[k] for k in mix_stems}
    z_mix = synthesizer.mix_stems(mix_dict)

    return z_mix, mix_stems


def create_hard_mix(
    stems: Dict[str, torch.Tensor],
    target_stem: str,
    num_stems: int = 3,
    instrument_labels: Optional[Dict[str, str]] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Create a hard separation task mix with similar instruments.

    TODO: Use instrument_labels to select same-family instruments.

    Args:
        stems: Dict of stem_name -> latent
        target_stem: The stem we want to extract
        num_stems: Number of stems in the mix
        instrument_labels: Instrument labels for selecting similar instruments

    Returns:
        Tuple of (mix, list of stem names in mix)
    """
    # For now, same as easy mix
    # TODO: Implement proper instrument family grouping
    return create_easy_mix(stems, target_stem, num_stems, instrument_labels)


def create_dense_mix(
    stems: Dict[str, torch.Tensor],
    target_stem: str,
    min_stems: int = 5,
    max_stems: int = 10,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Create a dense separation task mix with many stems.

    Args:
        stems: Dict of stem_name -> latent
        target_stem: The stem we want to extract
        min_stems: Minimum number of stems
        max_stems: Maximum number of stems

    Returns:
        Tuple of (mix, list of stem names in mix)
    """
    synthesizer = LatentSynthesizer()

    available = [k for k in stems.keys() if k != target_stem]
    num_additional = random.randint(
        min(min_stems - 1, len(available)),
        min(max_stems - 1, len(available))
    )

    mix_stems = random.sample(available, num_additional)
    mix_stems.append(target_stem)

    mix_dict = {k: stems[k] for k in mix_stems}
    z_mix = synthesizer.mix_stems(mix_dict)

    return z_mix, mix_stems
